"""
Offline tests for aat.dutch.sentences -- pure functions, no dspy or
network access needed. Mirrors tests/test_english_sentences.py's
coverage (the underlying logic is identical -- CTS URN plumbing and
sentence-boundary clustering, not grammar), with Dutch example text
substituted in.
"""

import pytest

from aat.core import CitedPassage
from aat.dutch.sentences import (
    cluster_sentences,
    ends_sentence,
    passage_component,
    tokenize_corpus_by_sentence,
    tokenize_units,
    urn_prefix,
)

_URN = "urn:cts:latijn:cicero.proarchia.nl:{}"


def _unit(passage, text):
    return CitedPassage(context=_URN.format(passage), text=text)


def test_passage_component_and_urn_prefix():
    urn = "urn:cts:latijn:cicero.proarchia.nl:2.1"
    assert passage_component(urn) == "2.1"
    assert urn_prefix(urn) == "urn:cts:latijn:cicero.proarchia.nl"


def test_passage_component_rejects_non_cts_urn():
    with pytest.raises(ValueError, match="doesn't look like a CTS URN"):
        passage_component("not-a-urn")


def test_ends_sentence_defaults():
    assert ends_sentence("Hij heeft Cicero nooit gelezen.")
    assert ends_sentence("Is dat zo?")
    assert ends_sentence("Wat een verrassing!")
    assert not ends_sentence("die tot de menselijke beschaving bijdragen,")
    assert not ends_sentence("")
    assert not ends_sentence("   ")


def test_ends_sentence_trailing_whitespace_ignored():
    assert ends_sentence("Hij heeft Cicero nooit gelezen.   \n")


def test_ends_sentence_custom_terminators():
    assert ends_sentence("...gaat verder:", terminators=":;")
    assert not ends_sentence("...gaat verder:", terminators=".?!")


def test_cluster_sentences_one_unit_per_sentence_when_each_ends_cleanly():
    units = [_unit("2.1", "Hij heeft Cicero nooit gelezen."), _unit("2.2", "Bezitten zij een band?")]
    groups = cluster_sentences(units)
    assert groups == [[units[0]], [units[1]]]


def test_cluster_sentences_spans_units_until_sentence_ends():
    # A sentence split across two citation units, the same shape as the
    # real English Genesis 1:14-15 case aat.dutch.sentences's own
    # docstring describes (ported from aat.english.sentences).
    u1 = _unit("2.1", "Alle kunsten en wetenschappen die tot de menselijke beschaving bijdragen,")
    u2 = _unit("2.2", "bezitten een gemeenschappelijke band.")
    u3 = _unit("2.3", "Dat is de kern van dit betoog.")
    groups = cluster_sentences([u1, u2, u3])
    assert groups == [[u1, u2], [u3]]


def test_cluster_sentences_trailing_unterminated_group_still_included():
    u1 = _unit("2.1", "Hij heeft Cicero nooit gelezen.")
    u2 = _unit("2.2", "En de tekst werd vertaald door Jones en")
    groups = cluster_sentences([u1, u2])
    assert groups == [[u1], [u2]]
    assert not ends_sentence(groups[-1][-1].text)


def test_cluster_sentences_empty_input():
    assert cluster_sentences([]) == []


def test_tokenize_units_single_unit_context_matches_passage_component():
    units = [_unit("2.1", "Hij heeft Cicero nooit gelezen.")]
    context, text, tokens = tokenize_units(units)
    assert context == "urn:cts:latijn:cicero.proarchia.nl:2.1"
    assert text == "Hij heeft Cicero nooit gelezen."
    assert [t.id for t in tokens] == ["2.1.t1", "2.1.t2", "2.1.t3", "2.1.t4", "2.1.t5", "2.1.t6"]
    assert all(t.context == context for t in tokens)


def test_tokenize_units_multi_unit_context_is_a_cts_range():
    units = [_unit("2.1", "De kunsten"), _unit("2.2", "bezitten een band.")]
    context, text, tokens = tokenize_units(units)
    assert context == "urn:cts:latijn:cicero.proarchia.nl:2.1-2.2"
    assert text == "De kunsten bezitten een band."
    ids = [t.id for t in tokens]
    # Each unit's own local numbering restarts at t1 -- prefixed by that
    # unit's own passage component, not renumbered flat across the pair.
    assert ids == ["2.1.t1", "2.1.t2", "2.2.t1", "2.2.t2", "2.2.t3", "2.2.t4"]


def test_tokenize_units_rejects_mismatched_urn_prefix():
    units = [
        CitedPassage(context="urn:cts:latijn:cicero.proarchia.nl:1.1", text="a"),
        CitedPassage(context="urn:cts:latijn:cicero.catilina.nl:1.1", text="b"),
    ]
    with pytest.raises(ValueError, match="don't share one urn_prefix"):
        tokenize_units(units)


def test_tokenize_units_rejects_empty_list():
    with pytest.raises(ValueError):
        tokenize_units([])


def test_tokenize_corpus_by_sentence_end_to_end():
    u1 = _unit("2.1", "Alle kunsten en wetenschappen die tot de menselijke beschaving bijdragen,")
    u2 = _unit("2.2", "bezitten een gemeenschappelijke band.")
    u3 = _unit("2.3", "Dat is de kern van dit betoog.")
    results = tokenize_corpus_by_sentence([u1, u2, u3])

    assert len(results) == 2
    context1, text1, tokens1 = results[0]
    assert context1 == "urn:cts:latijn:cicero.proarchia.nl:2.1-2.2"
    assert tokens1[0].id == "2.1.t1"
    assert tokens1[-1].id.startswith("2.2.")

    context2, text2, tokens2 = results[1]
    assert context2 == "urn:cts:latijn:cicero.proarchia.nl:2.3"
    assert all(t.id.startswith("2.3.") for t in tokens2)


def test_tokenize_corpus_by_sentence_is_deterministic():
    units = [_unit("2.1", "...bijdragen,"), _unit("2.2", "...band.")]
    first = tokenize_corpus_by_sentence(units)
    second = tokenize_corpus_by_sentence(units)
    assert first == second
