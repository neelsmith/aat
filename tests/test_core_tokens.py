"""Offline tests for aat.core.tokens -- no dspy or network needed."""

from aat.core import CitableToken, CitedPassage


def test_cited_passage_fields():
    p = CitedPassage(context="c1", text="hello world")
    assert p.context == "c1"
    assert p.text == "hello world"


def test_citable_token_fields():
    t = CitableToken(context="c1", id="t1", value="hello")
    assert (t.context, t.id, t.value) == ("c1", "t1", "hello")
