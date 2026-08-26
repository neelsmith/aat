"""
Optimizes aat.english.dspy_signatures.AgentActionTarget's prompt with
dspy.GEPA, using tests/fixtures/gold_examples.py's GOLD_EXAMPLES as the
trainset and aat.english.gepa_metric.aat_metric as the scoring/feedback
function. See OPTIMIZING.md for usage.

This is a LIVE-LM script: unlike the pytest suite (entirely DummyLM-backed,
see TESTING.md), every trial here actually calls the configured task LM,
plus a reflection LM GEPA uses to read the metric's feedback and propose
better instructions. Budget is controlled by --auto (dspy's light/medium/
heavy presets -- default "light", the cheapest) or --max-metric-calls for
an exact call count.

There is currently no separate valset: all of GOLD_EXAMPLES is used as the
trainset, and (per dspy.GEPA's own behavior when valset=None) also as the
Pareto-tracking set. That's a reasonable choice while the gold set is
still small; revisit it (holding out a few examples as a valset) once
there are enough gold examples to spare.

Usage:
    python optimize_gepa.py                       # --auto light (default)
    python optimize_gepa.py --auto medium
    python optimize_gepa.py --max-metric-calls 40
    python optimize_gepa.py --skip-baseline        # skip the pre-GEPA scoring pass

Needs the same .env as aat_main.py (API_BASE/MODEL/API_KEY). Optionally
set REFLECTION_MODEL (and REFLECTION_API_BASE/REFLECTION_API_KEY, if they
differ) to use a different model for GEPA's own reflective step -- GEPA's
own docs recommend a strong reasoning model specifically for reflection.
"""

import argparse
import sys
from pathlib import Path

import dspy

# Reuse aat_main.py's own .env-loading + LM-config helpers rather than
# duplicating them.
sys.path.insert(0, str(Path(__file__).parent))
from aat_main import _configure_lm, _env  # noqa: E402

# tests/ isn't an installed package -- add it to sys.path the same way
# pytest does (see pytest.ini's own comment about this) so
# "from fixtures.gold_examples import GOLD_EXAMPLES" resolves the same way
# it does under pytest, without duplicating the fixtures module here.
sys.path.insert(0, str(Path(__file__).parent / "tests"))
from fixtures.gold_examples import GOLD_EXAMPLES  # noqa: E402

from aat.core import AATNode
from aat.english.dspy_signatures import analyze
from aat.english.gepa_metric import aat_metric


def build_trainset():
    """Turn every GoldExample in GOLD_EXAMPLES into a dspy.Example GEPA can
    train against: `passage`/`tokens` as inputs (matching
    AgentActionTarget's own InputFields), `nodes` as the gold output
    aat_metric compares predictions to."""
    trainset = []
    for example in GOLD_EXAMPLES:
        nodes = [AATNode(**n) for n in example.canned_nodes]
        trainset.append(
            dspy.Example(
                passage=example.passage,
                tokens=example.tokens(),
                nodes=nodes,
            ).with_inputs("passage", "tokens")
        )
    return trainset


def _configure_reflection_lm(task_lm):
    """Build the LM GEPA uses to read aat_metric's feedback and propose
    better instructions. Defaults to `task_lm` itself unless
    REFLECTION_MODEL is set in .env, in which case a separate dspy.LM is
    built for it (same API_BASE/API_KEY unless REFLECTION_API_BASE/
    REFLECTION_API_KEY override those too)."""
    reflection_model = _env("REFLECTION_MODEL", "REFLECTION_MODEL", None)
    if not reflection_model:
        return task_lm

    api_base = _env("REFLECTION_API_BASE", "REFLECTION_API_BASE", None) or _env(
        "API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm"
    )
    api_key = _env("REFLECTION_API_KEY", "REFLECTION_API_KEY", None) or _env("API_KEY", "API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key for the reflection LM. Set REFLECTION_API_KEY or API_KEY in .env."
        )
    return dspy.LM(model=reflection_model, api_base=api_base, api_key=api_key)


def _evaluate(program, trainset, label):
    """Run `program` over every trainset example, score each with
    aat_metric, and print a min/mean/max summary."""
    scores = []
    for example in trainset:
        pred = program(passage=example.passage, tokens=example.tokens)
        scores.append(aat_metric(example, pred).score)
    mean = sum(scores) / len(scores)
    print(f"{label}: mean={mean:.3f}  min={min(scores):.3f}  max={max(scores):.3f}  (n={len(scores)})")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Optimize AgentActionTarget's prompt with GEPA.")
    budget = parser.add_mutually_exclusive_group()
    budget.add_argument(
        "--auto",
        choices=["light", "medium", "heavy"],
        default="light",
        help="dspy's auto budget preset (default: %(default)s -- fewest LM calls).",
    )
    budget.add_argument(
        "--max-metric-calls",
        type=int,
        default=None,
        help="Exact LM-call budget instead of an --auto preset.",
    )
    parser.add_argument(
        "--out",
        default="optimized_agent_action_target.json",
        help="Where to save the optimized program (default: %(default)s).",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip the pre-GEPA scoring pass (saves one live LM call per gold example).",
    )
    args = parser.parse_args()

    task_lm = _configure_lm()
    reflection_lm = _configure_reflection_lm(task_lm)

    trainset = build_trainset()
    print(f"Built a trainset of {len(trainset)} gold examples from tests/fixtures/gold_examples.py.")
    print(
        "No separate valset -- GEPA will use the trainset for both reflective updates "
        "and Pareto-score tracking. See this script's module docstring for why."
    )

    if not args.skip_baseline:
        print()
        _evaluate(analyze, trainset, "Baseline (before GEPA)")

    optimizer_kwargs = dict(
        metric=aat_metric,
        reflection_lm=reflection_lm,
        track_stats=True,
        log_dir=str(Path(__file__).parent / "gepa_logs"),
    )
    if args.max_metric_calls is not None:
        optimizer_kwargs["max_metric_calls"] = args.max_metric_calls
    else:
        optimizer_kwargs["auto"] = args.auto

    gepa = dspy.GEPA(**optimizer_kwargs)

    print("\nRunning GEPA -- this makes many real LM calls through the configured proxy.")
    optimized = gepa.compile(student=analyze, trainset=trainset)

    print()
    _evaluate(optimized, trainset, "Optimized (after GEPA)")

    optimized.save(args.out)
    print(f"\nSaved the optimized program to {args.out}.")
    print(
        "To use it, right after `from aat.english.dspy_signatures import analyze`, call:\n"
        f"    analyze.load({args.out!r})\n"
        "before running analyze_passage()/analyze_passages() -- see OPTIMIZING.md."
    )


if __name__ == "__main__":
    main()
