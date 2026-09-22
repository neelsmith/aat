"""
Offline tests for aat.english.sentences -- pure functions, no dspy or
network access needed. See that module's own docstring for why it
exists and TESTING.md for how this fits with the rest of the suite.
"""

import pytest

from aat.core import CitedPassage
from aat.english.sentences import (
    cluster_sentences,
    ends_sentence,
    passage_component,
    tokenize_corpus_by_sentence,
    tokenize_units,
    urn_prefix,
)

_URN = "urn:cts:compnov:bible.genesis.rvvpl:{}"


def _unit(passage, text):
    return CitedPassage(context=_URN.format(passage), text=text)


def test_passage_component_and_urn_prefix():
    urn = "urn:cts:compnov:bible.genesis.rvvpl:1.14"
    assert passage_component(urn) == "1.14"
    assert urn_prefix(urn) == "urn:cts:compnov:bible.genesis.rvvpl"


def test_passage_component_rejects_non_cts_urn():
    with pytest.raises(ValueError, match="doesn't look like a CTS URN"):
        passage_component("not-a-urn")


def test_ends_sentence_defaults():
    assert ends_sentence("And it was so.")
    assert ends_sentence("Is that so?")
    assert ends_sentence("It is so!")
    assert not ends_sentence("for signs, and for seasons, and for days and years:")
    assert not ends_sentence("were made sure")
    assert not ends_sentence("")
    assert not ends_sentence("   ")


def test_ends_sentence_trailing_whitespace_ignored():
    assert ends_sentence("And it was so.   \n")


def test_ends_sentence_custom_terminators():
    assert ends_sentence("...continues:", terminators=":;")
    assert not ends_sentence("...continues:", terminators=".?!")


def test_cluster_sentences_one_unit_per_sentence_when_each_ends_cleanly():
    units = [_unit("1.1", "In the beginning."), _unit("1.2", "And the earth was waste.")]
    groups = cluster_sentences(units)
    assert groups == [[units[0]], [units[1]]]


def test_cluster_sentences_spans_units_until_sentence_ends():
    # Genesis 1:14-15's own real shape: 14 ends ':', 15 finishes the
    # sentence -- see the module docstring's worked example.
    u14 = _unit("1.14", "...for signs, and for seasons, and for days and years:")
    u15 = _unit("1.15", "and let them be for lights...: and it was so.")
    u16 = _unit("1.16", "And God made the two great lights.")
    groups = cluster_sentences([u14, u15, u16])
    assert groups == [[u14, u15], [u16]]


def test_cluster_sentences_trailing_unterminated_group_still_included():
    u1 = _unit("1.1", "In the beginning.")
    u2 = _unit("1.2", "And the earth was waste and void; and darkness")
    groups = cluster_sentences([u1, u2])
    assert groups == [[u1], [u2]]
    assert not ends_sentence(groups[-1][-1].text)


def test_cluster_sentences_empty_input():
    assert cluster_sentences([]) == []


def test_tokenize_units_single_unit_context_matches_passage_component():
    units = [_unit("1.1", "In the beginning.")]
    context, text, tokens = tokenize_units(units)
    assert context == "urn:cts:compnov:bible.genesis.rvvpl:1.1"
    assert text == "In the beginning."
    assert [t.id for t in tokens] == ["1.1.t1", "1.1.t2", "1.1.t3", "1.1.t4"]
    assert [t.value for t in tokens] == ["In", "the", "beginning", "."]
    assert all(t.context == context for t in tokens)


def test_tokenize_units_multi_unit_context_is_a_cts_range():
    units = [_unit("1.14", "The dog"), _unit("1.15", "ate the homework.")]
    context, text, tokens = tokenize_units(units)
    assert context == "urn:cts:compnov:bible.genesis.rvvpl:1.14-1.15"
    assert text == "The dog ate the homework."
    ids = [t.id for t in tokens]
    # Each unit's own local numbering restarts at t1 -- prefixed by that
    # unit's own passage component, not renumbered flat across the pair.
    assert ids == ["1.14.t1", "1.14.t2", "1.15.t1", "1.15.t2", "1.15.t3", "1.15.t4"]


def test_tokenize_units_rejects_mismatched_urn_prefix():
    units = [
        CitedPassage(context="urn:cts:compnov:bible.genesis.rvvpl:1.1", text="a"),
        CitedPassage(context="urn:cts:compnov:bible.exodus.rvvpl:1.1", text="b"),
    ]
    with pytest.raises(ValueError, match="don't share one urn_prefix"):
        tokenize_units(units)


def test_tokenize_units_rejects_empty_list():
    with pytest.raises(ValueError):
        tokenize_units([])


def test_tokenize_corpus_by_sentence_end_to_end():
    u14 = _unit("1.14", "...for signs, and for seasons, and for days and years:")
    u15 = _unit("1.15", "and let them be for lights...: and it was so.")
    u16 = _unit("1.16", "And God made the two great lights.")
    results = tokenize_corpus_by_sentence([u14, u15, u16])

    assert len(results) == 2
    context1, text1, tokens1 = results[0]
    assert context1 == "urn:cts:compnov:bible.genesis.rvvpl:1.14-1.15"
    assert tokens1[0].id == "1.14.t1"
    assert tokens1[-1].id.startswith("1.15.")

    context2, text2, tokens2 = results[1]
    assert context2 == "urn:cts:compnov:bible.genesis.rvvpl:1.16"
    assert all(t.id.startswith("1.16.") for t in tokens2)


def test_tokenize_corpus_by_sentence_is_deterministic():
    # Same input, called twice with no LM involved at all -- must
    # produce byte-identical output, since this is exactly the function
    # a future reader would re-run to pair tokens back up with an
    # already-serialized graph with no LM access (see the module
    # docstring).
    units = [_unit("1.14", "...years:"), _unit("1.15", "...so.")]
    first = tokenize_corpus_by_sentence(units)
    second = tokenize_corpus_by_sentence(units)
    assert first == second
