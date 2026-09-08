"""
DummyLM-backed tests for aat.english.pipeline -- exercises
analyze_passage()/analyze_passages() without any network access. See
TESTING.md.
"""

import dspy
from dspy.utils.dummies import DummyLM

from aat.core import CitedPassage
from aat.english import analyze_passage, analyze_passages

_ANSWER = {
    "reasoning": "ate is the action; dog is the agent; homework is the target.",
    "nodes": [
        {"context": "ex.1", "id": "t3", "value": "ate", "role": "action", "related_node": None},
        {"context": "ex.1", "id": "t2", "value": "dog", "role": "agent", "related_node": "t3"},
        {"context": "ex.1", "id": "t5", "value": "homework", "role": "target", "related_node": "t3"},
    ],
}


def test_analyze_passage_single_string():
    dspy.configure(lm=DummyLM([_ANSWER]))

    tokens, graph = analyze_passage("The dog ate my homework.", context="ex.1")

    assert [t.id for t in tokens] == ["t1", "t2", "t3", "t4", "t5", "t6"]
    assert graph.actions()[0].id == "t3"
    assert graph.agents()[0].value == "dog"
    assert graph.targets()[0].value == "homework"


def test_analyze_passage_defaults_to_empty_context():
    dspy.configure(lm=DummyLM([{**_ANSWER, "nodes": []}]))
    tokens, graph = analyze_passage("Hi.")
    assert tokens[0].context == ""


def test_analyze_passages_concatenates_multiple_passages():
    dspy.configure(lm=DummyLM([_ANSWER, _ANSWER]))

    passages = [
        CitedPassage(context="ex.1", text="The dog ate my homework."),
        CitedPassage(context="ex.1", text="The dog ate my homework."),
    ]
    tokens, graph = analyze_passages(passages)

    assert len(tokens) == 12  # 6 tokens per passage, two passages
    assert len(graph.actions()) == 2


def test_validation_warnings_go_to_stderr_not_stdout(capsys):
    # A broken related_node (points at a nonexistent action id) triggers
    # a validate() problem -- confirms analyze_passages() reports it on
    # stderr, not stdout, so a caller writing a machine-parseable result
    # to stdout (aat_main.py's serialized analysis, see test_aat_main.py)
    # never gets it corrupted by a stray warning print.
    broken_answer = {
        "reasoning": "ate is the action; dog is the agent, but related_node is broken.",
        "nodes": [
            {"context": "ex.1", "id": "t3", "value": "ate", "role": "action", "related_node": None},
            {"context": "ex.1", "id": "t2", "value": "dog", "role": "agent", "related_node": "ghost"},
        ],
    }
    dspy.configure(lm=DummyLM([broken_answer]))

    analyze_passage("The dog ate my homework.", context="ex.1")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validation warnings" in captured.err
    assert "ghost" in captured.err
