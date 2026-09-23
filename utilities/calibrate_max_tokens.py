"""
Calibrates aat/english/token_budget.py's max_tokens estimate against real
LM output, instead of the untuned fallback constants token_budget.py ships
with.

Why: AgentActionTarget's output (a `reasoning` field plus a JSON-serialized
`nodes` list) grows with how long and how syntactically complex a passage
is, not by a fixed amount -- so the right `max_tokens` budget for a call is
a function of the input token count, not a constant. This script measures
that function directly: it runs every GOLD_EXAMPLES passage
(tests/fixtures/gold_examples.py) through the real configured LM with a
generous max_tokens ceiling so nothing truncates, records how many
completion tokens each one actually used, and fits `completion_tokens ~
intercept + slope * num_input_tokens` by ordinary least squares. The fitted
(intercept, slope) is written to aat/english/token_budget_calibration.json,
where token_budget.estimate_max_tokens() picks it up automatically.

Usage (from the repo root, so REPO_ROOT below and aat_main.py's own .env
lookup both resolve correctly -- same convention utilities/optimize_gepa.py
already uses):

    python3 utilities/calibrate_max_tokens.py

Needs the same .env aat_main.py uses (see USAGE.md's "Running an analysis
from the command line"):

    API_BASE=https://your-litellm-proxy/...
    MODEL=litellm_proxy/your-model
    API_KEY=your-key-here

This is a live-LM script with real API cost -- one call per GOLD_EXAMPLES
entry. Re-run it whenever the configured MODEL, the AgentActionTarget
prompt, or AATNode's own shape changes substantially, since any of those
shifts how many output tokens a given passage actually needs. GOLD_EXAMPLES
itself is a small corpus of short, illustrative single-construction
sentences (see that module's own docstring), not long real-world passages
-- the fit is a genuine measurement over that range, but treat max_tokens
estimates for much longer passages (e.g. a multi-citation-unit sentence
group from aat.english.analyze_units_by_sentence()) as an extrapolation,
and lean on token_budget.py's safety_margin/ceiling/retry machinery rather
than trusting the raw line far past the calibrated range.

--calibration-ceiling controls the max_tokens used *during calibration
itself* (not the fitted result) -- generous by default so calibration runs
aren't the ones getting truncated; --limit runs a quick smoke test over
just the first N examples instead of the whole corpus.
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse aat_main.py's own .env-loading + LM-config helper rather than
# duplicating it -- same convention utilities/optimize_gepa.py already
# uses (see that script's own comment on the point).
sys.path.insert(0, str(REPO_ROOT))
from aat_main import _configure_lm  # noqa: E402

# tests/ isn't an installed package -- add it to sys.path the same way
# pytest does, so "from fixtures.gold_examples import GOLD_EXAMPLES"
# resolves the same way it does under pytest, without duplicating the
# fixtures module here. Same convention utilities/optimize_gepa.py uses.
sys.path.insert(0, str(REPO_ROOT / "tests"))
from fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

from aat.english.dspy_signatures import analyze  # noqa: E402

CALIBRATION_FILE = REPO_ROOT / "aat" / "english" / "token_budget_calibration.json"


def _fit_line(xs, ys):
    """Ordinary least squares for y = a + b*x, plain Python (no numpy
    dependency needed for a fit this simple). Returns (a, b). Raises
    ValueError if there are fewer than 2 distinct x values -- a line isn't
    identifiable from a single point."""
    n = len(xs)
    if len({x for x in xs}) < 2:
        raise ValueError(
            "Need at least 2 examples with different token counts to fit a "
            "line; every calibrated example had the same num_tokens."
        )

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    var = sum((x - mean_x) ** 2 for x in xs)
    b = cov / var
    a = mean_y - b * mean_x
    return a, b


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate token_budget.py's max_tokens estimate against the real configured LM."
    )
    parser.add_argument(
        "--calibration-ceiling",
        type=int,
        default=8000,
        help="max_tokens used for calibration calls themselves (default: 8000) -- "
             "should comfortably exceed anything GOLD_EXAMPLES needs; raise it if "
             "examples are still getting skipped as truncated even at the default.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only calibrate against the first N GOLD_EXAMPLES (for a quick smoke run).",
    )
    args = parser.parse_args()

    lm = _configure_lm()

    examples = GOLD_EXAMPLES[: args.limit] if args.limit else GOLD_EXAMPLES

    rows = []  # (slug, num_tokens, completion_tokens)
    skipped = []
    for example in examples:
        tokens = example.tokens()
        try:
            analyze(
                passage=example.passage,
                tokens=tokens,
                config={"max_tokens": args.calibration_ceiling},
            )
        except Exception as exc:  # noqa: BLE001 -- report and keep calibrating
            skipped.append((example.slug, f"raised {exc.__class__.__name__}: {exc}"))
            continue

        usage = lm.history[-1].get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if completion_tokens is None:
            skipped.append((example.slug, "no completion_tokens in usage -- provider didn't report it"))
            continue

        choices = getattr(lm.history[-1].get("response"), "choices", [])
        if any(getattr(c, "finish_reason", None) == "length" for c in choices):
            skipped.append((example.slug, f"still truncated even at max_tokens={args.calibration_ceiling}"))
            continue

        rows.append((example.slug, len(tokens), completion_tokens))

    print(f"Calibrated against {len(rows)}/{len(examples)} examples.")
    if skipped:
        print(f"\nSkipped {len(skipped)}:")
        for slug, reason in skipped:
            print(f"  - {slug}: {reason}")

    if len(rows) < 2:
        raise RuntimeError(
            f"Only {len(rows)} usable example(s) -- need at least 2 to fit a line. "
            "Check the skipped list above."
        )

    print("\nslug                                       num_tokens  completion_tokens")
    for slug, num_tokens, completion_tokens in rows:
        print(f"{slug:<42}  {num_tokens:>10}  {completion_tokens:>17}")

    xs = [r[1] for r in rows]
    ys = [r[2] for r in rows]
    intercept, slope = _fit_line(xs, ys)

    residuals = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    max_abs_residual = max(abs(r) for r in residuals)

    print(f"\nFitted: completion_tokens ~= {intercept:.1f} + {slope:.2f} * num_tokens")
    print(f"Largest residual over the calibration set: {max_abs_residual:.1f} tokens")
    print(
        "(token_budget.estimate_max_tokens() applies its own safety_margin on top "
        "of this fit -- the margin is what actually covers residual variance like "
        "this, not the fit itself.)"
    )

    payload = {
        "intercept": intercept,
        "slope": slope,
        "sample_size": len(rows),
        "model": lm.model,
        "calibrated_at": datetime.datetime.now().isoformat(),
    }

    CALIBRATION_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {CALIBRATION_FILE}")


if __name__ == "__main__":
    main()
