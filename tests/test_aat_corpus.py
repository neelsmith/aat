"""Offline tests for aat_corpus.py -- exercises its own _read_corpus()
helper directly, and the analyze -> serialize composition its __main__
block performs, using DummyLM (no real LM call, no .env, no network
access). The __main__ block itself (argparse + _configure_lm()) isn't
exercised here, since _configure_lm() needs real .env/API credentials --
test_aat_main.py's own tests take the same approach for aat_main.py's
equivalent script-level LM configuration. See TESTING.md."""

import io

import dspy
import pytest
from dspy.utils.dummies import DummyLM

from aat.core import CitedPassage, read_analysis, serialize_analysis
from aat.english import analyze_passages

import aat_corpus

_CEX_SAMPLE = """\
#!ctsdata
ex.1#The dog ate my homework.
ex.2#The cat sat on the mat.
"""

_ANSWER_1 = {
    "reasoning": "ate is the action; dog is the agent; homework is the target.",
    "nodes": [
        {"context": "ex.1", "id": "t3", "value": "ate", "role": "action", "related_node": None},
        {"context": "ex.1", "id": "t2", "value": "dog", "role": "agent", "related_node": "t3"},
        {"context": "ex.1", "id": "t5", "value": "homework", "role": "target", "related_node": "t3"},
    ],
}
_ANSWER_2 = {
    "reasoning": "sat is the action; cat is the agent; mat is the target.",
    "nodes": [
        {"context": "ex.2", "id": "t3", "value": "sat", "role": "action", "related_node": None},
        {"context": "ex.2", "id": "t2", "value": "cat", "role": "agent", "related_node": "t3"},
        {"context": "ex.2", "id": "t6", "value": "mat", "role": "target", "related_node": "t3"},
    ],
}


def test_read_corpus_from_a_real_file(tmp_path):
    path = tmp_path / "corpus.cex"
    path.write_text(_CEX_SAMPLE, encoding="utf-8")
    passages = aat_corpus._read_corpus(str(path), delimiter="#")
    assert passages == [
        CitedPassage(context="ex.1", text="The dog ate my homework."),
        CitedPassage(context="ex.2", text="The cat sat on the mat."),
    ]


def test_read_corpus_from_stdin(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(_CEX_SAMPLE))
    passages = aat_corpus._read_corpus("-", delimiter="#")
    assert len(passages) == 2
    assert passages[0].context == "ex.1"


def test_read_corpus_honors_custom_delimiter(tmp_path):
    path = tmp_path / "corpus.cex"
    path.write_text("#!ctsdata\nex.1|Hi there.\n", encoding="utf-8")
    passages = aat_corpus._read_corpus(str(path), delimiter="|")
    assert passages == [CitedPassage(context="ex.1", text="Hi there.")]


def test_read_corpus_missing_file_raises():
    with pytest.raises(OSError):
        aat_corpus._read_corpus("/does/not/exist.cex", delimiter="#")


def test_read_corpus_malformed_cex_raises_value_error(tmp_path):
    path = tmp_path / "not_cex.txt"
    path.write_text("this isn't a CEX file at all\n", encoding="utf-8")
    with pytest.raises(ValueError):
        aat_corpus._read_corpus(str(path), delimiter="#")


def test_full_composition_analyzes_every_passage_and_serializes_once(tmp_path):
    # Exercises exactly what the __main__ block does after _configure_lm()
    # (skipped here in favor of DummyLM): _read_corpus() ->
    # analyze_passages() -> serialize_analysis() -> one combined string.
    path = tmp_path / "corpus.cex"
    path.write_text(_CEX_SAMPLE, encoding="utf-8")
    passages = aat_corpus._read_corpus(str(path), delimiter="#")

    dspy.configure(lm=DummyLM([_ANSWER_1, _ANSWER_2]))
    _tokens, graph = analyze_passages(passages)

    result = serialize_analysis(passages, graph)
    assert result.count("#!passages") == 1
    assert result.count("#!aatnodes") == 1
    assert "ex.1|The dog ate my homework." in result
    assert "ex.2|The cat sat on the mat." in result
    assert "ex.1|t3|ate|action|" in result
    assert "ex.2|t3|sat|action|" in result


def test_full_composition_round_trips_through_read_analysis(tmp_path):
    path = tmp_path / "corpus.cex"
    path.write_text(_CEX_SAMPLE, encoding="utf-8")
    passages = aat_corpus._read_corpus(str(path), delimiter="#")

    dspy.configure(lm=DummyLM([_ANSWER_1, _ANSWER_2]))
    _tokens, graph = analyze_passages(passages)
    result = serialize_analysis(passages, graph)

    out_path = tmp_path / "analysis.txt"
    out_path.write_text(result, encoding="utf-8")

    reloaded_passages, reloaded_graph = read_analysis(str(out_path))
    assert reloaded_passages == passages
    assert reloaded_graph == graph
