"""
Offline tests for aat.lm_cost.summarize_lm_cost/format_lm_cost -- pure
functions over plain dict/list data, no dspy or network access needed
(history entries are hand-built dicts, the same shape dspy.LM's own
`.history` uses).
"""

from aat.lm_cost import LMCostSummary, format_lm_cost, summarize_lm_cost


def _entry(cost):
    return {"cost": cost, "model": "test-model"}


def test_empty_history_never_raises():
    summary = summarize_lm_cost([])
    assert summary == LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=0)
    assert summary.total_calls == 0
    assert format_lm_cost(summary) == "no LM calls yet"


def test_all_cache_hits_have_no_total_but_do_count():
    summary = summarize_lm_cost([_entry(None), _entry(None)])
    assert summary.total_cost is None
    assert summary.priced_calls == 0
    assert summary.uncosted_calls == 2
    assert summary.total_calls == 2
    text = format_lm_cost(summary)
    assert "0.00" in text
    assert "2 calls" in text
    assert "cache" in text


def test_sums_across_every_priced_call_not_just_the_last():
    summary = summarize_lm_cost([_entry(0.01), _entry(0.02), _entry(0.03)])
    assert summary.priced_calls == 3
    assert summary.uncosted_calls == 0
    assert round(summary.total_cost, 2) == 0.06
    text = format_lm_cost(summary)
    assert "$0.0600" in text
    assert "3 calls" in text
    assert "cache" not in text


def test_mix_of_priced_and_cached_notes_the_cached_ones_separately():
    summary = summarize_lm_cost([_entry(0.05), _entry(None), _entry(0.05)])
    assert summary.priced_calls == 2
    assert summary.uncosted_calls == 1
    assert round(summary.total_cost, 2) == 0.10
    text = format_lm_cost(summary)
    assert "$0.1000" in text
    assert "2 calls" in text
    assert "1 more call" in text
    assert "cache" in text


def test_singular_call_wording():
    summary = summarize_lm_cost([_entry(0.01)])
    text = format_lm_cost(summary)
    assert "1 call" in text
    assert "1 calls" not in text


def test_reads_cost_from_object_attribute_not_just_dict():
    class Entry:
        cost = 0.42

    summary = summarize_lm_cost([Entry()])
    assert summary.priced_calls == 1
    assert round(summary.total_cost, 2) == 0.42


def test_entry_missing_cost_key_entirely_reads_as_uncosted():
    summary = summarize_lm_cost([{"model": "test-model"}])
    assert summary.priced_calls == 0
    assert summary.uncosted_calls == 1
