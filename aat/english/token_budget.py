"""
Estimating and enforcing a `max_tokens` output budget for AgentActionTarget
calls, and retrying with a larger one when a call actually gets truncated.

Background: `aat.english.dspy_signatures.analyze` (a `dspy.ChainOfThought`)
produces a free-text `reasoning` field plus a JSON-serialized `nodes` list.
That output's size scales with how long and how syntactically complex the
passage is -- and, since `aat_corpus_graph.py`'s sentence-clustering feature
(aat.english.sentences) can combine several citation units into one
passage, "how long" isn't bounded by a single citation unit's own length
either -- so any single hard-coded `max_tokens` is eventually wrong for
either a short passage (wastes budget) or a long/complex one (truncates
mid-output). This is exactly what a real incident hit: a multi-unit
sentence group's `analyze()` call came back truncated, and the resulting
`AdapterParseError` reported `max_tokens=None` -- which should never be
possible for a real budget. That `None` had two separate causes, both
fixed as part of adding this module:

1. Every `dspy.LM(...)` construction in this repo (aat_main.py,
   marimo/aat_graph.py, marimo/aat_corpus_graph.py) left `max_tokens`
   unset, so it fell through to `dspy.LM.__init__`'s own default of
   `None` -- meaning every call before this module existed ran with
   *no* explicit output budget at all, at the mercy of whatever the
   provider/proxy defaults to. Each of those call sites now passes
   `max_tokens=DEFAULT_CEILING` (see that constant below) as an
   explicit numeric baseline -- not because ordinary calls are expected
   to need that much, but because `dspy.LM._check_truncation()`'s own
   truncation warning always reports *this* baseline, never whatever a
   per-call `config={"max_tokens": ...}` override (see below) actually
   used, so leaving it at `None` made every such warning misleadingly
   claim the call had no budget at all -- exactly the symptom that
   surfaced.
2. Even with that baseline in place, a single passage can still need
   more than a fixed ceiling allows. That's what this module actually
   solves.

This module takes the same hybrid approach `arsgrammatica.token_budget`
(a sibling project, also built on dspy) uses for its own analogous
truncation problem:

1. `estimate_max_tokens()` picks a per-call budget from a simple linear
   model (`completion_tokens ~= intercept + slope * num_input_tokens`),
   calibrated empirically by `utilities/calibrate_max_tokens.py` against
   real LM output over GOLD_EXAMPLES (see that script's own docstring),
   with a safety margin on top. Until that script has been run against
   the model you're actually using, a conservative, deliberately-generous
   fallback fit is used instead (see `_FALLBACK_INTERCEPT`/
   `_FALLBACK_SLOPE` below) -- meant to overestimate rather than
   truncate, not to be a good fit.
2. `analyze_with_retry()` wraps `aat.english.dspy_signatures.analyze` and,
   if a call still comes back truncated despite that estimate, retries
   with a larger budget rather than silently returning an incomplete
   result or leaving the caller to guess a bigger number by hand.

One deliberate divergence from arsgrammatica's own version of this
module, worth being explicit about: arsgrammatica's SentenceAnalysis
signature produces one `TokenAnalysis` entry per INPUT token (including
implied/elided ones), so "does the output cover every input token id?"
is a reliable, LM-independent truncation signal there regardless of
`finish_reason`. AgentActionTarget's `nodes` output has no such
contract -- a short passage with no agent/action/target content at all
can correctly produce zero nodes, and even a fully analyzed passage
normally covers only some of its tokens (see AgentActionTarget's own
docstring) -- so "how many nodes came back" says nothing reliable about
whether the response was cut off. This module therefore has no
`_missing_token_ids()`-equivalent primary signal; truncation detection
here leans on the two signals that ARE schema-agnostic:
`_finish_reason_was_length()` (the LM/provider's own claim) and, for a
response cut off badly enough to not parse as JSON at all,
`_looks_truncated()` (a bracket/brace-balance check on the raw response
text) -- see each function's own docstring for why relying on
`finish_reason` alone isn't safe enough. A parse failure that neither
signal flags as truncation is retried once at the SAME budget instead
(with the LM's own response cache bypassed), on the assumption it's a
one-off malformed-output glitch a bigger budget wouldn't have fixed --
same reasoning arsgrammatica's own version already documents.

Re-run `utilities/calibrate_max_tokens.py` whenever the configured
model, the AgentActionTarget prompt, or AATNode's own shape changes
substantially -- all three shift how many output tokens a given passage
actually needs.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import List, Optional

# dspy is imported here, not at unconditional module scope, for the same
# reason arsgrammatica.token_budget does this (see that module's own
# comment on the point): everything else in this module (DEFAULT_CEILING,
# get_calibration(), estimate_max_tokens(), _looks_truncated()) is usable
# without a live dspy at all, so only the two names that actually need one
# (_finish_reason_was_length(), analyze_with_retry()) should force the
# import -- and importing aat.english already pulls this module in via its
# own __init__.py, so gating it narrowly here matters even though aat, unlike
# arsgrammatica, doesn't currently offer a dspy-free WASM/browser export.
# `analyze` is kept as a real module-level name either way (not deferred
# into the one function that uses it) specifically so
# tests/test_english_token_budget.py's own
# monkeypatch.setattr("aat.english.token_budget.analyze", ...) calls keep
# working unchanged whenever dspy *is* installed -- which is every test run
# today, since aat's own `dev` extra depends on `english`.
try:
    import dspy
    from dspy.utils.exceptions import AdapterParseError

    from .dspy_signatures import analyze
except ImportError as _dspy_exc:
    # `except ... as name` implicitly deletes `name` once this block ends,
    # so it's reassigned to a plain variable first -- analyze() below,
    # called later, can still reference it.
    _dspy_import_error = _dspy_exc

    # `dspy.settings.lm` in _finish_reason_was_length() below then raises
    # AttributeError on `None`, which that function already catches and
    # treats as "no" -- no dspy installed means no dspy.LM history to
    # check, so failing safe here is correct, not just convenient.
    dspy = None

    class AdapterParseError(Exception):  # pragma: no cover -- only stands in so `except AdapterParseError:` below stays valid; analyze()'s stub (next line) never actually raises it.
        pass

    def analyze(*_args, **_kwargs):  # pragma: no cover
        raise ImportError(
            "analyze_with_retry() needs the optional 'english' extra (dspy, "
            "and a configured LM) actually installed to analyze new text -- "
            "install it with: pip install 'aat[english]'."
        ) from _dspy_import_error

from aat.core import CitableToken

# ---------------------------------------------------------------------------
# Calibration data
# ---------------------------------------------------------------------------

# utilities/calibrate_max_tokens.py writes its fitted (intercept, slope)
# here. Kept next to this module (not under tests/) since it's runtime
# configuration, not test fixture data -- any script or notebook using
# aat.english benefits from it, not just the test suite.
CALIBRATION_FILE = Path(__file__).with_name("token_budget_calibration.json")

# Untuned stand-ins, used only until utilities/calibrate_max_tokens.py has
# actually been run once against the real configured model. Deliberately
# generous -- an overestimate here just spends a bit more of the model's
# output budget than necessary; an underestimate is what causes the
# truncation this module exists to avoid. Unlike arsgrammatica's own
# fallback fit (90 completion tokens per input token, calibrated against a
# per-token TokenAnalysis output with several long field names per entry),
# AATNode's output is a much smaller, five-field object (context/id/value/
# role/related_node) and `nodes` covers only some of a passage's tokens,
# not all of them -- so this fit is deliberately shallower per input
# token. The intercept still has to cover a full ChainOfThought
# `reasoning` field plus JSON structural overhead even for a very short
# passage, which is the harder part to guess well; 700 is a conservative
# guess at that floor, not a measurement.
_FALLBACK_INTERCEPT = 700.0
_FALLBACK_SLOPE = 40.0

DEFAULT_SAFETY_MARGIN = 1.4
# See _FALLBACK_INTERCEPT's own comment for why this is set well above
# what a strict per-node accounting would suggest -- it's the same "even
# the shortest passage still needs roughly this much for reasoning + JSON
# structure" floor that intercept represents, so a calibrated fit with a
# much smaller intercept (plausible: GOLD_EXAMPLES' passages are short and
# often produce very few nodes) doesn't leave a short real passage
# underbudgeted. analyze_with_retry()'s own retry-with-growth still
# applies on top of this if even this floor turns out to be too little for
# some particular passage.
DEFAULT_FLOOR = 700
# Stand-in for "this model's real max output tokens" -- there's no single
# correct value across providers/models; override this with whatever your
# configured MODEL actually allows (check its provider's documentation)
# rather than relying on this default for anything but a rough starting
# point. Set generously (not just at _FALLBACK_INTERCEPT/_FALLBACK_SLOPE's
# own scale) since aat_corpus_graph.py's sentence-clustering feature can
# combine several citation units into one unusually long passage -- see
# this module's own top docstring for the real incident that motivated
# this whole module. If your model's own true ceiling is lower than this,
# a request for more than it allows should surface as an explicit error
# from the provider (naming the real limit) rather than a silent
# truncation -- lower this to match if that happens.
DEFAULT_CEILING = 20000


def _load_calibration() -> dict:
    """Read utilities/calibrate_max_tokens.py's saved fit, if any.

    Returns a dict with at least "intercept", "slope", and "source" keys.
    "source" is "calibrated" when CALIBRATION_FILE was read successfully,
    or "fallback" when it's missing, unreadable, or malformed -- callers
    that want to know which one is active (or tests that want to force the
    fallback) can check that field rather than re-deriving it.
    """
    try:
        with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            "intercept": float(data["intercept"]),
            "slope": float(data["slope"]),
            "source": "calibrated",
            "sample_size": data.get("sample_size"),
            "model": data.get("model"),
            "calibrated_at": data.get("calibrated_at"),
        }
    except (OSError, ValueError, KeyError, TypeError):
        return {
            "intercept": _FALLBACK_INTERCEPT,
            "slope": _FALLBACK_SLOPE,
            "source": "fallback",
            "sample_size": None,
            "model": None,
            "calibrated_at": None,
        }


def get_calibration() -> dict:
    """Public introspection: what (intercept, slope) is estimate_max_tokens()
    currently using, and did it come from utilities/calibrate_max_tokens.py's
    fit or from this module's untuned fallback? See _load_calibration()'s
    docstring for the shape returned."""
    return _load_calibration()


def estimate_max_tokens(
    num_tokens: int,
    *,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
) -> int:
    """Estimate a `max_tokens` budget for an AgentActionTarget call over a
    passage with `num_tokens` input tokens.

    `raw = intercept + slope * num_tokens` comes from the calibrated (or
    fallback) linear fit (see _load_calibration()); `safety_margin`
    multiplies that to leave room for the reasoning field's length being
    only roughly, not exactly, a function of passage length. The result is
    clamped to `[floor, ceiling]` -- `floor` guards against a degenerate
    tiny estimate for a very short passage, `ceiling` is a hard cap you
    should set to your actual model's real max-output-tokens limit (see
    DEFAULT_CEILING's docstring note).

    Raises ValueError if `num_tokens` is negative.
    """
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be >= 0, got {num_tokens}")

    calibration = _load_calibration()
    raw = calibration["intercept"] + calibration["slope"] * num_tokens
    budget = math.ceil(raw * safety_margin)
    return max(floor, min(ceiling, budget))


# ---------------------------------------------------------------------------
# Retry-on-truncation wrapper
# ---------------------------------------------------------------------------


def _finish_reason_was_length() -> bool:
    """Best-effort check of whether the most recent call made against the
    currently configured LM was cut off for hitting max_tokens, via the
    same `finish_reason == "length"` signal dspy.LM._check_truncation()
    itself warns on.

    This is a *secondary* corroborating signal for a response that failed
    to parse at all (see _looks_truncated()'s docstring for why it isn't
    trusted alone there), and the ONLY post-parse truncation signal this
    module has (see this module's own top docstring for why the
    per-token-coverage check arsgrammatica's version of this function
    also has access to doesn't translate to AgentActionTarget's schema).
    Fails safe: any missing attribute, empty history, or non-dspy.LM
    configured LM (e.g. DummyLM in tests, which doesn't populate
    `.history` the same way) just returns False rather than raising.
    """
    try:
        lm = dspy.settings.lm
        entry = lm.history[-1]
        response = entry["response"]
        # Dict-style access on `response` itself, matching dspy.LM's own
        # _check_truncation() exactly (`results["choices"]`); attribute-style
        # access on each choice, same as that method's `c.finish_reason`.
        return any(getattr(c, "finish_reason", None) == "length" for c in response["choices"])
    except (AttributeError, IndexError, KeyError, TypeError):
        return False


def _looks_truncated(lm_response: str) -> bool:
    """A cheap, provider/adapter-agnostic heuristic for "does this raw LM
    response text look like it was cut off mid-emission, rather than a
    complete-but-malformed response?" -- used alongside
    `_finish_reason_was_length()` when an `AdapterParseError` is raised
    (i.e. no `result` exists at all).

    Counts `{`/`}` and `[`/`]`: a genuinely truncated response almost
    always stops mid-token, leaving at least one of these pairs
    unbalanced. A well-terminated response that's merely malformed in
    SHAPE (e.g. one `nodes` entry missing a required field) is still
    syntactically complete JSON/field-marker text, so its brackets/braces
    balance even though a schema validator rejects it.

    This exists because `_finish_reason_was_length()` isn't always
    reliable enough to be the ONLY truncation signal: it depends on the
    configured provider (or an intermediary proxy, e.g. a self-hosted
    litellm proxy -- see this module's own top docstring for the real
    incident) faithfully forwarding litellm's own finish_reason="length"
    convention, which not every provider/proxy does. When it
    under-reports, gating the retry-with-a-larger-budget branch on it
    ALONE misclassifies a real truncation as a one-off malformation and
    retries at the SAME (still too small) budget -- reaching an identical
    failure on the very next attempt too, and then raising once
    max_retries is exhausted, instead of ever actually growing the
    budget. This heuristic is deliberately conservative in the OTHER
    direction: a false positive just means an unnecessary (but harmless)
    budget increase, not a wrong answer -- unlike a false negative here,
    which reproduces exactly the failure this function exists to catch.
    """
    return lm_response.count("{") != lm_response.count("}") or lm_response.count("[") != lm_response.count("]")


def analyze_with_retry(
    passage: str,
    tokens: List[CitableToken],
    *,
    max_retries: int = 3,
    growth_factor: float = 2.0,
    safety_margin: float = DEFAULT_SAFETY_MARGIN,
    floor: int = DEFAULT_FLOOR,
    ceiling: int = DEFAULT_CEILING,
    initial_max_tokens: Optional[int] = None,
):
    """Call `aat.english.dspy_signatures.analyze`, detecting truncation and
    retrying with a larger `max_tokens` budget instead of either crashing
    or silently returning an incomplete result.

    The starting budget is `initial_max_tokens` if given, else
    `estimate_max_tokens(len(tokens), safety_margin=safety_margin,
    floor=floor, ceiling=ceiling)`.

    `max_retries` defaults to 3 (not 1) with `growth_factor=2.0`, so a
    genuinely under-budgeted call has real headroom to grow (~8x by the
    third retry) rather than giving up after a single doubling --
    mirroring arsgrammatica.token_budget.analyze_with_retry()'s own
    `max_retries` default and the reasoning documented there: a
    passage's `reasoning` field length is stochastic enough call to call
    that even a good estimate can occasionally need more than one
    doubling. Each retry is one more live LM call, so this does raise the
    worst-case cost/latency of a single passage -- still bounded by
    `ceiling`, and far better than surfacing a truncated result or a
    raised exception to the caller.

    After each attempt: if the call raised `AdapterParseError` (the JSON
    was cut off badly enough to not parse at all, or was malformed some
    other way), EITHER `_finish_reason_was_length()` OR `_looks_truncated()`
    on the raw response text is treated as a truncation signal -- either
    one alone is enough (see `_looks_truncated()`'s own docstring for why
    `finish_reason` alone isn't reliable enough across every
    provider/proxy). If truncation is detected and a retry is still
    available (fewer than `max_retries` attempts so far, and the budget
    hasn't already hit `ceiling`), the budget is multiplied by
    `growth_factor` (capped at `ceiling`) and the call is retried.
    `max_tokens` is part of DSPy's own LM cache key, so a retry with a
    different budget always reaches the LM again rather than replaying a
    cached truncated response.

    An `AdapterParseError` that neither signal flags as truncation means
    the response was well-terminated but still malformed somewhere -- a
    bigger budget wouldn't have fixed that, but the malformation itself
    is very often a one-off sampling glitch rather than a systematic
    prompt/schema problem, so it's retried once too (still counted
    against `max_retries`, at the SAME budget) with dspy's own response
    cache explicitly bypassed for that one attempt
    (`config={"cache": False, ...}`) -- without that, an identical
    request would just replay the identical broken response, retrying
    nothing. If that retry also fails to parse, or `max_retries` is
    already exhausted, the exception propagates.

    When the call DOES return a parsed result, this module has no
    reliable per-response signal that it's incomplete (see this module's
    own top docstring for why AgentActionTarget's `nodes` output can't be
    checked for "coverage" the way arsgrammatica's per-token tokengraph
    can) other than `_finish_reason_was_length()` itself -- a provider
    can, in principle, report `finish_reason="length"` even though the
    JSON it emitted up to that point happens to be complete and parses
    cleanly. That case is treated as truncated too (retried with a larger
    budget, same as the parse-failure path), and if retries are exhausted
    while it's still true, the result is returned anyway with a
    `UserWarning` -- matching this codebase's existing convention of
    surfacing analysis problems as warnings (see pipeline.py's own
    validate()-warning-printing) rather than treating an imperfect LM
    result as fatal.
    """
    budget = initial_max_tokens if initial_max_tokens is not None else estimate_max_tokens(
        len(tokens), safety_margin=safety_margin, floor=floor, ceiling=ceiling
    )

    attempt = 0
    bypass_cache = False
    while True:
        old_budget = budget
        call_config = {"max_tokens": budget}
        if bypass_cache:
            call_config["cache"] = False
        bypass_cache = False  # only meant for the one attempt it was set for
        try:
            result = analyze(passage=passage, tokens=tokens, config=call_config)
        except AdapterParseError as exc:
            if attempt >= max_retries:
                raise
            if budget < ceiling and (_finish_reason_was_length() or _looks_truncated(exc.lm_response)):
                attempt += 1
                budget = min(ceiling, math.ceil(budget * growth_factor))
                warnings.warn(
                    f"AgentActionTarget call truncated at max_tokens={old_budget} before "
                    f"it could be parsed at all; retrying with max_tokens={budget} "
                    f"(attempt {attempt}/{max_retries}).",
                    stacklevel=2,
                )
                continue
            # Not a (detectable) truncation -- the response finished
            # normally but was malformed somewhere. Retry once more at the
            # SAME budget, with dspy's cache explicitly bypassed for that
            # one attempt, so the retry is a genuinely fresh LM call
            # rather than a replay of the same broken response.
            attempt += 1
            bypass_cache = True
            warnings.warn(
                f"AgentActionTarget call at max_tokens={old_budget} returned output that "
                f"failed to parse, but doesn't look like a truncation (finish_reason "
                f"wasn't 'length'): {exc} Retrying once at the same budget with the LM "
                f"cache bypassed, in case this was a one-off malformed-output glitch "
                f"(attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        truncated = _finish_reason_was_length()
        if truncated and attempt < max_retries and budget < ceiling:
            attempt += 1
            budget = min(ceiling, math.ceil(budget * growth_factor))
            warnings.warn(
                f"AgentActionTarget call at max_tokens={old_budget} parsed, but its "
                f"finish_reason reported truncation anyway; retrying with a larger "
                f"max_tokens={budget} (attempt {attempt}/{max_retries}).",
                stacklevel=2,
            )
            continue

        if truncated:
            warnings.warn(
                f"AgentActionTarget call still reports truncation after {attempt} "
                f"retry(ies) (max_tokens={old_budget}) -- returning it anyway; the "
                f"result may be missing nodes for content past the cutoff.",
                stacklevel=2,
            )

        return result
