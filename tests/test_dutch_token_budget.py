"""
Tests for aat/dutch/token_budget.py: estimate_max_tokens()'s formula and
clamping, and analyze_with_retry()'s detect-truncation-and-retry loop.
Mirrors tests/test_english_token_budget.py's coverage -- this module's
logic is entirely language-agnostic (see token_budget.py's own module
docstring), so the same test shapes apply; only the monkeypatch targets
(aat.dutch.token_budget.* instead of aat.english.token_budget.*) and the
example passage differ. Built against a directly-constructed passage/
token list here rather than a GOLD_EXAMPLES-style fixture, since there
isn't a curated Dutch gold-examples corpus yet (see aat/dutch/__init__.py's
own note on what's not yet ported from aat.english).
"""

import dspy
import pytest
from dspy.utils.dummies import DummyLM
from dspy.utils.exceptions import AdapterParseError

from aat.core import CitedPassage
from aat.dutch import analyze_with_retry, estimate_max_tokens, get_calibration, tokenize
from aat.dutch.dspy_signatures import AgentActionTarget
from aat.dutch.token_budget import DEFAULT_CEILING, DEFAULT_FLOOR

_PASSAGE_TEXT = "Hij heeft Cicero nooit gelezen."
_TOKENS = tokenize(CitedPassage(context="ex.1", text=_PASSAGE_TEXT))
_CANNED_ANSWER = {
    "reasoning": "gelezen is the principal verb; Hij is agent; Cicero is target.",
    "nodes": [
        {"context": "ex.1", "id": "t5", "value": "heeft gelezen", "role": "action", "related_node": None},
        {"context": "ex.1", "id": "t1", "value": "Hij", "role": "agent", "related_node": "t5"},
        {"context": "ex.1", "id": "t3", "value": "Cicero", "role": "target", "related_node": "t5"},
    ],
}


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
    assert estimate_max_tokens(0, floor=10_000, ceiling=20_000) == 10_000


def test_estimate_max_tokens_respects_ceiling():
    assert estimate_max_tokens(1_000_000, floor=0, ceiling=4_096) == 4_096


def test_estimate_max_tokens_safety_margin_scales_the_budget():
    baseline = estimate_max_tokens(20, safety_margin=1.0, floor=0, ceiling=1_000_000)
    margined = estimate_max_tokens(20, safety_margin=2.0, floor=0, ceiling=1_000_000)
    assert margined == pytest.approx(2 * baseline, rel=0.05)


def test_estimate_max_tokens_uses_fallback_constants_when_uncalibrated(tmp_path, monkeypatch):
    # No aat/dutch/token_budget_calibration.json exists yet at all (see
    # this module's own comment on the point) -- this just makes that
    # explicit and future-proof against one eventually being added.
    monkeypatch.setattr("aat.dutch.token_budget.CALIBRATION_FILE", tmp_path / "missing.json")
    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_get_calibration_reads_a_real_calibration_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text(
        '{"intercept": 100.0, "slope": 10.0, "sample_size": 12, "model": "test-model", '
        '"calibrated_at": "2026-01-01T00:00:00"}',
        encoding="utf-8",
    )
    monkeypatch.setattr("aat.dutch.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "calibrated"
    assert calibration["intercept"] == 100.0
    assert calibration["slope"] == 10.0
    assert estimate_max_tokens(10, safety_margin=1.0, floor=0, ceiling=10_000) == 200


def test_get_calibration_falls_back_on_malformed_file(tmp_path, monkeypatch):
    calibration_file = tmp_path / "token_budget_calibration.json"
    calibration_file.write_text("not valid json", encoding="utf-8")
    monkeypatch.setattr("aat.dutch.token_budget.CALIBRATION_FILE", calibration_file)

    calibration = get_calibration()
    assert calibration["source"] == "fallback"


def test_default_floor_and_ceiling_are_sane():
    assert 0 < DEFAULT_FLOOR < DEFAULT_CEILING


# ---------------------------------------------------------------------------
# analyze_with_retry()
# ---------------------------------------------------------------------------


def test_analyze_with_retry_matches_analyze_on_a_normal_call():
    """No truncation at all: behaves exactly like calling analyze()
    directly, and consumes exactly one DummyLM answer."""
    dspy.configure(lm=DummyLM([_CANNED_ANSWER]))

    result = analyze_with_retry(_PASSAGE_TEXT, _TOKENS)

    assert [n.id for n in result.nodes] == [e["id"] for e in _CANNED_ANSWER["nodes"]]


def test_analyze_with_retry_honors_initial_max_tokens_over_the_estimate():
    dspy.configure(lm=DummyLM([_CANNED_ANSWER]))
    result = analyze_with_retry(_PASSAGE_TEXT, _TOKENS, initial_max_tokens=50)
    assert result is not None


def test_analyze_with_retry_grows_budget_and_succeeds_when_adapter_parse_error_looks_truncated(monkeypatch):
    """A response cut off mid-JSON raises AdapterParseError; even when
    finish_reason doesn't say "length" (simulating a proxy/provider that
    doesn't forward it faithfully), _looks_truncated()'s bracket/brace-
    balance check on the raw response text is enough on its own to grow
    the budget and retry."""
    good_result = dspy.Prediction(
        reasoning="fine",
        nodes=[dspy.Prediction(**e) for e in _CANNED_ANSWER["nodes"]],
    )

    truncated_raw_response = (
        '{"reasoning": "...", "nodes": [{"context": "ex.1", "id": "t5", '
        '"value": "heeft gele'
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

    monkeypatch.setattr("aat.dutch.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("aat.dutch.token_budget._finish_reason_was_length", lambda: False)

    with pytest.warns(UserWarning, match="truncated"):
        result = analyze_with_retry(_PASSAGE_TEXT, _TOKENS, max_retries=1)

    assert result is good_result
    assert len(calls) == 2
    assert calls[1]["max_tokens"] > calls[0]["max_tokens"]


def test_analyze_with_retry_gives_up_after_max_retries_on_persistent_truncation(monkeypatch):
    calls = []

    def fake_analyze(*, passage, tokens, config):
        calls.append(dict(config))
        raise AdapterParseError(
            adapter_name="ChatAdapter",
            signature=AgentActionTarget,
            lm_response='{"reasoning": "cut off", "nodes": [{"id": "t',
            message="Failed to parse field nodes: unterminated string",
        )

    monkeypatch.setattr("aat.dutch.token_budget.analyze", fake_analyze)
    monkeypatch.setattr("aat.dutch.token_budget._finish_reason_was_length", lambda: True)

    with pytest.warns(UserWarning, match="truncated"):
        with pytest.raises(AdapterParseError):
            analyze_with_retry(_PASSAGE_TEXT, _TOKENS, max_retries=2)

    assert len(calls) == 3  # initial attempt + 2 retries
