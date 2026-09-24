"""Offline tests for aat.dutch.tokenize -- no dspy or network needed."""

from aat.core import CitedPassage
from aat.dutch import tokenize


def test_tokenize_matches_dutch_md_context_and_tokens_example():
    # notes/dutch.md's own "Context and tokens" worked example.
    passage = CitedPassage(
        context="proarchia_nl",
        text="Alle kunsten en wetenschappen bezitten een gemeenschappelijke band.",
    )
    tokens = tokenize(passage)
    assert tokens[0].context == "proarchia_nl"
    assert tokens[0].id == "t1"
    assert tokens[0].value == "Alle"
    assert [t.value for t in tokens] == [
        "Alle", "kunsten", "en", "wetenschappen", "bezitten", "een",
        "gemeenschappelijke", "band", ".",
    ]
    assert all(t.context == "proarchia_nl" for t in tokens)


def test_tokenize_keeps_diaeresis_word_as_one_token():
    # "ideeën" has an internal ë (diaeresis) -- must not be split into
    # separate "ide"/"ë"/"en" tokens the way an ASCII-only word regex
    # (aat.english.tokenize's own) would.
    passage = CitedPassage(context="c1", text="De ideeën zijn goed.")
    tokens = tokenize(passage)
    assert [t.value for t in tokens] == ["De", "ideeën", "zijn", "goed", "."]


def test_tokenize_keeps_acute_accent_word_as_one_token():
    passage = CitedPassage(context="c1", text="Dat is financiële steun.")
    tokens = tokenize(passage)
    assert "financiële" in [t.value for t in tokens]


def test_tokenize_keeps_apostrophe_plural_as_one_token():
    # Dutch plurals-after-a-vowel take an apostrophe ("auto's", "foto's"),
    # the same internal-apostrophe-joining behavior as English "dog's".
    passage = CitedPassage(context="c1", text="De auto's zijn snel.")
    tokens = tokenize(passage)
    assert [t.value for t in tokens] == ["De", "auto's", "zijn", "snel", "."]


def test_tokenize_compound_verb_sentence():
    # "Hij heeft Cicero nooit gelezen." -- notes/dutch.md's own worked
    # example for a compound action with an interrupting adverb.
    passage = CitedPassage(context="ex.1", text="Hij heeft Cicero nooit gelezen.")
    tokens = tokenize(passage)
    assert [t.value for t in tokens] == ["Hij", "heeft", "Cicero", "nooit", "gelezen", "."]
    assert [t.id for t in tokens] == ["t1", "t2", "t3", "t4", "t5", "t6"]


def test_tokenize_empty_text_gives_no_tokens():
    passage = CitedPassage(context="c1", text="")
    assert tokenize(passage) == []
