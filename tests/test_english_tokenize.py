"""Offline tests for aat.english.tokenize -- no dspy or network needed."""

from aat.core import CitedPassage
from aat.english import tokenize


def test_tokenize_dog_ate_homework_matches_aat_model_md_worked_example():
    passage = CitedPassage(context="c1", text="The dog ate my homework.")
    tokens = tokenize(passage)
    assert [t.id for t in tokens] == ["t1", "t2", "t3", "t4", "t5", "t6"]
    assert [t.value for t in tokens] == ["The", "dog", "ate", "my", "homework", "."]
    assert all(t.context == "c1" for t in tokens)


def test_tokenize_keeps_contraction_as_one_token():
    passage = CitedPassage(context="c1", text="He didn't eat it.")
    tokens = tokenize(passage)
    assert [t.value for t in tokens] == ["He", "didn't", "eat", "it", "."]


def test_tokenize_keeps_possessive_as_one_token():
    passage = CitedPassage(context="c1", text="The dog's bone.")
    tokens = tokenize(passage)
    assert [t.value for t in tokens] == ["The", "dog's", "bone", "."]


def test_tokenize_empty_text_gives_no_tokens():
    passage = CitedPassage(context="c1", text="")
    assert tokenize(passage) == []
