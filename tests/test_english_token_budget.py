"""
Tests for aat/english/token_budget.py: estimate_max_tokens()'s formula and
clamping, and analyze_with_retry()'s detect-truncation-and-retry loop.

Modeled closely on arsgrammatica's own tests/test_token_budget.py (see that
project's tests for the sibling implementation this one is adapted from),
with one structural difference throughout: AgentActionTarget's `nodes`
output isn't one-entry-per-input-token the way arsgrammatica's `tokengraph`
is (see token_budget.py's own module docstring for why), so there's no
"missing ids" primary truncation signal here -- every test that exercises
the "parsed successfully, but still looks truncated" path does so by
monkeypatching `_finish_reason_was_length()` directly instead.
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from aat.english import analyze_with_retry, estimate_max_tokens, get_calibration
from aat.english.dspy_signatures import AgentActionTarget
from aat.english.token_budget import DEFAULT_CEILING, DEFAULT_FLOOR
from tests.fixtures.gold_examples import GOLD_EXAMPLES


def _example(slug):
    return next(e for e in GOLD_EXAMPLES if e.slug == slug)


# ---------------------------------------------------------------------------
# estimate_max_tokens()
# ---------------------------------------------------------------------------


def test_estimate_max_tokens_grows_with_input_length():
    small = estimate_max_tokens(5)
    large = estimate_max_tokens(50)
    assert large > small


def test_estimate_max_tokens_rejects_negative_input():
    with pytest.raises(ValueError):
        estimate_max_tokens(-1)


def test_estimate_max_tokens_respects_floor():
    # A tiny passage still gets at least `floor`, even though the raw
    # fitted value for num_tokens=0 could fall under it for a small
    # enough calibration.
    assert estimate_max_tokens(0, floor=10_000, ceiling=20_000) == 10_000


def test_estimate_max_tokens_respects_ceiling():
    # An enormous passage is clamped, not allowed to grow unbounded.
    assert estimate_max_tokens(1_000_000, floor=0, ceiling=4_096) == 4_096


def test_estimate_max_tokens_safety_margin_scales_the_budget():
    baseline = estimate_max_tokens(20, safety_margin=1.0, floor=0, ceiling=1_000_000)
    margined = estimate_max_tokens(20, safety_margin=2.0, floor=0, ceiling=1_000_000)
    assert margined == pytest.approx(2 * baseline, rel=0.05)


def test_estimate_max_tokens_uses_fallback_constants_when_uncalibrated(tmp_path, monkeypatch):
    # Point CALIBRATION_FILE at a path that doesn't exist, forcing the
    # fallback constants token_budget.py ships with.
    monkeypatch.setattr("aat.english.token_budget.CALIBRATION_FILE", tmp_path / "missing.json")
    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_get_calibration_reads_a_real_calibration_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text(
        '{"intercept": 100.0, "slope": 10.0, "sample_size": 43, "model": "test-model", '
        '"calibrated_at": "2026-01-01T00:00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("aat.english.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "calibrated"
    assert calibration["intercept"] == 100.0
    assert calibration["slope"] == 10.0

    # And estimate_max_tokens() actually uses it: with intercept=100,
    # slope=10, safety_margin=1.0, num_tokens=10 -> raw 200, well inside
    # generous floor/ceiling.
    assert estimate_max_tokens(10, safety_margin=1.0, floor=0, ceiling=10_000) == 200


def test_get_calibration_falls_back_on_malformed_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr("aat.english.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_default_floor_and_ceiling_are_sane():
    # Not much to assert here beyond "floor is well under ceiling" -- this
    # just guards against a typo swapping the two.
    assert 0 < DEFAULT_FLOOR < DEFAULT_CEILING


# ---------------------------------------------------------------------------
# analyze_with_retry()
# ---------------------------------------------------------------------------


def test_analyze_with_retry_matches_analyze_on_a_normal_gold_example():
    """No truncation at all: behaves exactly like calling analyze()
    directly, and consumes exactly one DummyLM answer."""
    example = _example("dog-ate-homework")
    dspy.configure(lm=DummyLM([example.canned_answer]))

    result = analyze_with_retry(example.passage, example.tokens())

    assert [n.id for n in result.nodes] == [e["id"] for e in example.canned_answer["nodes"]]


def test_analyze_with_retry_honors_initial_max_tokens_over_the_estimate():
    """A caller-supplied initial_max_tokens is used as-is for the first
    attempt rather than being recomputed from estimate_max_tokens()."""
    example = _example("dog-ate-homework")
    dspy.configure(lm=DummyLM([example.canned_answer]))

    # A tiny explicit budget, well under what estimate_max_tokens() would
    # pick -- if this were ignored in favor of the estimate, this
    # assertion wouldn't tell us anything, so the point is just that the
    # call succeeds and the config kwarg is accepted at all.
    result = analyze_with_retry(example.passage, example.tokens(), initial_max_tokens=50)
    assert result is not None


def test_analyze_with_retry_grows_budget_and_succeeds_when_adapter_parse_error_looks_truncated(monkeypatch):
    """A response cut off mid-JSON raises AdapterParseError; even when
    finish_reason doesn't say "length" (simulating a proxy/provider that
    doesn't forward it faithfully -- see the real incident in
    token_budget.py's own module docstring), _looks_truncated()'s
    bracket/brace-balance check on the raw response text is enough on its
    own to grow the budget and retry."""
    example = _example("he-said-that-dog-ate")
    tokens = example.tokens()

    good_result = dspy.Prediction(
        reasoning="fine",
        nodes=[dspy.Prediction(**e) for e in example.canned_answer["nodes"]],
    )

    truncated_raw_response = (
        '{"reasoning": "...", "nodes": [{"context": "gold.4", "id": "t2", '
        '"value": "said", "role": "acti'
    )

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        if len(calls) == 1:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=AgentActionTarget,
                lm_response=truncated_raw_response,
                message="Failed to parse field nodes: unterminated string",
            )
        return good_result

    monkeypatch.setattr("aat.english.token_budget.analyze", fake_analyze)
    # Simulates a misreporting provider/proxy: even though this call was
    # genuinely truncated, finish_reason doesn't say so.
    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="truncated"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert result is good_result
    assert len(calls) == 2
    # The budget actually grew on the retry, rather than being replayed
    # unchanged.
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_analyze_with_retry_gives_up_after_max_retries_on_persistent_truncation(monkeypatch):
    """If every attempt keeps coming back truncated (AdapterParseError,
    brackets never balance), the exception propagates once max_retries is
    exhausted -- there's no partial result to fall back to when parsing
    never once succeeded."""
    example = _example("dog-ate-homework")
    tokens = example.tokens()

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=AgentActionTarget,
            lm_response='{"reasoning": "cut off", "nodes": [{"id": "t',
            message="persistently truncated",
        )

    monkeypatch.setattr("aat.english.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="truncated"):
        with pytest.raises(AdapterParseError, match="persistently truncated"):
            analyze_with_retry(example.passage, tokens, max_retries=1)

    # One normal attempt, one retry with a grown budget, then give up --
    # no third attempt beyond max_retries=1.
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_analyze_with_retry_retries_once_on_malformed_non_truncated_output_then_succeeds(monkeypatch):
    """A parse failure whose finish_reason ISN'T "length" and whose raw
    response text has balanced brackets/braces (well-terminated, just
    schema-invalid -- e.g. a `nodes` entry missing a required field) is a
    different failure mode from truncation -- retried once anyway, at the
    SAME budget, with the LM cache explicitly bypassed for that one
    retry."""
    example = _example("dog-ate-homework")
    tokens = example.tokens()

    good_result = dspy.Prediction(
        reasoning="fine",
        nodes=[dspy.Prediction(**e) for e in example.canned_answer["nodes"]],
    )

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        if len(calls) == 1:
            raise AdapterParseError(
                adapter_name="ChatAdapter",
                signature=AgentActionTarget,
                lm_response='{"reasoning": "ok", "nodes": [{"id": "t3"}]}',
                message="Failed to parse field nodes with value [...]. Error message: "
                "1 validation error for list[AATNode]\n0\n  Field required "
                "[type=missing, input_value={'id': 't3'}, input_type=dict]",
            )
        return good_result

    monkeypatch.setattr("aat.english.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="doesn't look like a truncation"):
        result = analyze_with_retry(example.passage, tokens, max_retries=1)

    assert result is good_result
    assert len(calls) == 2
    # Same budget both times (no growth -- a bigger budget wouldn't have
    # fixed a malformed entry), but the cache is bypassed only for the
    # retry, not the first (normal) attempt.
    assert calls[0]["max_tokens"] == calls[1]["max_tokens"]
    assert "cache" not in calls[0]
    assert calls[1]["cache"] is False


def test_analyze_with_retry_gives_up_after_max_retries_on_persistent_malformed_output(monkeypatch):
    """If the retry ALSO comes back malformed (not truncated), the
    exception propagates once max_retries is exhausted -- same as any
    other unrecoverable failure, not silently swallowed."""
    example = _example("dog-ate-homework")
    tokens = example.tokens()

    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=AgentActionTarget,
            lm_response='{"reasoning": "ok", "nodes": [{"id": "t3"}]}',
            message="persistently malformed",
        )

    monkeypatch.setattr("aat.english.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="doesn't look like a truncation"):
        with pytest.raises(AdapterParseError, match="persistently malformed"):
            analyze_with_retry(example.passage, tokens, max_retries=1)

    # One normal attempt, one cache-bypassed retry, then give up -- no
    # third attempt beyond max_retries=1.
    assert len(calls) == 2
    assert calls[1]["cache"] is False


def test_analyze_with_retry_retries_when_finish_reason_says_truncated_despite_parsing(monkeypatch):
    """A response that DID parse successfully can still, in principle, be
    reported as truncated by the provider's own finish_reason -- the only
    post-parse truncation signal this module has (see token_budget.py's
    own module docstring for why there's no per-node coverage check the
    way arsgrammatica's per-token one works). This is exercised by
    flipping a fake _finish_reason_was_length() from True (first call) to
    False (second) -- DummyLM's own list mode returns one answer per call
    in order, so reaching the second, distinguishable answer is only
    possible if a retry actually happened."""
    example = _example("he-said-that-dog-ate")
    other_example = _example("dog-ate-homework")
    dspy.configure(lm=DummyLM([example.canned_answer, other_example.canned_answer]))

    calls = {"n": 0}

    def fake_finish_reason():
        calls["n"] += 1
        return calls["n"] == 1

    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", fake_finish_reason)

    with pytest.warns(UserWarning, match="finish_reason reported truncation"):
        result = analyze_with_retry(example.passage, example.tokens(), max_retries=1)

    # The SECOND (different) answer -- only reachable via a retry.
    assert [n.id for n in result.nodes] == [e["id"] for e in other_example.canned_answer["nodes"]]


def test_analyze_with_retry_does_not_retry_past_the_ceiling(monkeypatch):
    """If the budget is already pinned at `ceiling`, a result that still
    looks truncated per finish_reason is returned anyway (with a warning)
    rather than retried again."""
    example = _example("dog-ate-homework")
    dspy.configure(lm=DummyLM([example.canned_answer]))
    monkeypatch.setattr("aat.english.token_budget._finish_reason_was_length", lambda: True)

    with pytest.warns(UserWarning, match="still reports truncation"):
        result = analyze_with_retry(
            example.passage,
            example.tokens(),
            max_retries=1,
            initial_max_tokens=500,
            ceiling=500,  # already at the ceiling -- no room to grow
        )

    assert [n.id for n in result.nodes] == [e["id"] for e in example.canned_answer["nodes"]]
