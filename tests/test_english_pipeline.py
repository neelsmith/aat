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

def test_analyze_units_by_sentence_spans_citation_units():
    # Two citation units that only form a complete sentence together --
    # same shape as the real Genesis 1:14-15 case aat.english.sentences's
    # own docstring describes. "The dog" (1.1) + "ate the homework."
    # (1.2) -> tokenized separately, prefixed, combined into one
    # DummyLM-backed analysis.
    from aat.core import CitedPassage
    from aat.english import analyze_units_by_sentence

    answer = {
        "reasoning": "ate is the action; dog is the agent; homework is the target.",
        "nodes": [
            {"context": "urn:cts:test:work:1.1-1.2", "id": "1.2.t1", "value": "ate", "role": "action", "related_node": None},
            {"context": "urn:cts:test:work:1.1-1.2", "id": "1.1.t2", "value": "dog", "role": "agent", "related_node": "1.2.t1"},
            {"context": "urn:cts:test:work:1.1-1.2", "id": "1.2.t3", "value": "homework", "role": "target", "related_node": "1.2.t1"},
        ],
    }
    dspy.configure(lm=DummyLM([answer]))

    units = [
        CitedPassage(context="urn:cts:test:work:1.1", text="The dog"),
        CitedPassage(context="urn:cts:test:work:1.2", text="ate the homework."),
    ]
    tokens, graph = analyze_units_by_sentence(units)

    assert [t.id for t in tokens] == ["1.1.t1", "1.1.t2", "1.2.t1", "1.2.t2", "1.2.t3", "1.2.t4"]
    assert all(t.context == "urn:cts:test:work:1.1-1.2" for t in tokens)
    assert graph.actions()[0].id == "1.2.t1"
    assert graph.agents()[0].id == "1.1.t2"
    assert graph.targets()[0].id == "1.2.t3"


def test_analyze_units_by_sentence_one_sentence_per_group():
    # Three units forming two sentences (1.1 alone; 1.2+1.3 together) --
    # confirms one LM call per sentence GROUP, not per citation unit.
    from aat.core import CitedPassage
    from aat.english import analyze_units_by_sentence

    answer1 = {
        "reasoning": "single independent action, no agent/target expressed.",
        "nodes": [
            {"context": "urn:cts:test:work:1.1", "id": "1.1.t2", "value": "waited", "role": "action", "related_node": None},
        ],
    }
    answer2 = {
        "reasoning": "ate is the action; dog is the agent.",
        "nodes": [
            {"context": "urn:cts:test:work:1.2-1.3", "id": "1.2.t1", "value": "ate", "role": "action", "related_node": None},
            {"context": "urn:cts:test:work:1.2-1.3", "id": "1.2.t2", "value": "dog", "role": "agent", "related_node": "1.2.t1"},
        ],
    }
    dspy.configure(lm=DummyLM([answer1, answer2]))

    units = [
        CitedPassage(context="urn:cts:test:work:1.1", text="He waited."),
        CitedPassage(context="urn:cts:test:work:1.2", text="The dog"),
        CitedPassage(context="urn:cts:test:work:1.3", text="ate."),
    ]
    tokens, graph = analyze_units_by_sentence(units)

    assert len(graph.nodes) == 3
    contexts = {t.context for t in tokens}
    assert contexts == {"urn:cts:test:work:1.1", "urn:cts:test:work:1.2-1.3"}


def test_analyze_units_by_sentence_validation_warnings_go_to_stderr(capsys):
    from aat.core import CitedPassage
    from aat.english import analyze_units_by_sentence

    broken_answer = {
        "reasoning": "action found, but related_node points nowhere real.",
        "nodes": [
            {"context": "urn:cts:test:work:1.1", "id": "1.1.t2", "value": "ate", "role": "action", "related_node": None},
            {"context": "urn:cts:test:work:1.1", "id": "1.1.t1", "value": "dog", "role": "agent", "related_node": "ghost"},
        ],
    }
    dspy.configure(lm=DummyLM([broken_answer]))

    analyze_units_by_sentence([CitedPassage(context="urn:cts:test:work:1.1", text="The dog ate.")])

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validation warnings" in captured.err
    assert "urn:cts:test:work:1.1" in captured.err
    assert "ghost" in captured.err
