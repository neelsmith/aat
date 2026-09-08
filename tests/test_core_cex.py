"""Offline tests for aat.core.cex -- no dspy or network needed."""

import pytest

from aat.core import CitedPassage
from aat.core.cex import parse_cex_ctsdata, read_cex_passages

_SAMPLE = """\
#!cexversion
3.0

#!citelibrary
name#Iliad sample
urn#urn:cite2:aat:examples.v1:
license#CC-BY

// a comment line -- ignored, like a blank line
#!ctsdata
urn:cts:greekLit:tlg0012.tlg001:1.1#Μῆνιν ἄειδε θεὰ
urn:cts:greekLit:tlg0012.tlg001:1.2#Πηληϊάδεω Ἀχιλῆος

#!ctscatalog
urn#citationScheme#groupName#workTitle#versionLabel#exemplarLabel#online#lang
urn:cts:greekLit:tlg0012.tlg001:#book/line#Homer#Iliad#sample##true#grc
"""


def test_reads_only_the_ctsdata_block():
    passages = parse_cex_ctsdata(_SAMPLE)
    assert passages == [
        CitedPassage(context="urn:cts:greekLit:tlg0012.tlg001:1.1", text="Μῆνιν ἄειδε θεὰ"),
        CitedPassage(context="urn:cts:greekLit:tlg0012.tlg001:1.2", text="Πηληϊάδεω Ἀχιλῆος"),
    ]


def test_comment_and_blank_lines_are_ignored():
    text = "#!ctsdata\n\n// a comment\nurn:x:1#hello\n"
    passages = parse_cex_ctsdata(text)
    assert passages == [CitedPassage(context="urn:x:1", text="hello")]


def test_custom_delimiter():
    text = "#!ctsdata\nurn:x:1|hello there\n"
    passages = parse_cex_ctsdata(text, delimiter="|")
    assert passages == [CitedPassage(context="urn:x:1", text="hello there")]


def test_text_containing_the_delimiter_is_kept_intact_via_first_split():
    # The default '#' delimiter can plausibly recur inside real prose --
    # splitting on the *first* occurrence only keeps the rest of the line
    # as the passage's own text, rather than raising or truncating it.
    text = "#!ctsdata\nurn:x:1#see note #3 below\n"
    passages = parse_cex_ctsdata(text)
    assert passages == [CitedPassage(context="urn:x:1", text="see note #3 below")]


def test_multiple_ctsdata_blocks_are_concatenated_in_file_order():
    text = "#!ctsdata\nurn:x:1#one\n\n#!ctscatalog\nignored#row\n\n#!ctsdata\nurn:x:2#two\n"
    passages = parse_cex_ctsdata(text)
    assert [p.context for p in passages] == ["urn:x:1", "urn:x:2"]


def test_row_with_no_delimiter_raises_value_error_naming_the_line():
    text = "#!ctsdata\nurn:x:1 no delimiter here\n"
    with pytest.raises(ValueError) as excinfo:
        parse_cex_ctsdata(text)
    message = str(excinfo.value)
    assert "line 2" in message
    assert "no delimiter here" in message


def test_file_with_no_ctsdata_block_raises_value_error():
    text = "#!citelibrary\nname#x\nurn#y\nlicense#z\n"
    with pytest.raises(ValueError) as excinfo:
        parse_cex_ctsdata(text)
    assert "ctsdata" in str(excinfo.value)


def test_empty_text_raises_value_error():
    with pytest.raises(ValueError):
        parse_cex_ctsdata("")


def test_ctsdata_block_with_zero_rows_is_not_an_error():
    text = "#!ctsdata\n#!ctscatalog\nignored#row\n"
    assert parse_cex_ctsdata(text) == []


def test_read_cex_passages_reads_from_a_real_file(tmp_path):
    path = tmp_path / "corpus.cex"
    path.write_text(_SAMPLE, encoding="utf-8")
    passages = read_cex_passages(str(path))
    assert len(passages) == 2
    assert passages[0].context == "urn:cts:greekLit:tlg0012.tlg001:1.1"
