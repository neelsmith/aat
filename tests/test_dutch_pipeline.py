"""
DummyLM-backed tests for aat.dutch.pipeline -- exercises
analyze_passage()/analyze_passages()/analyze_units_by_sentence() without
any network access. Mirrors tests/test_english_pipeline.py's coverage,
using notes/dutch.md's own worked examples as the passages under test.
See TESTING.md.
"""

import dspy
from dspy.utils.dummies import DummyLM

from aat.core import CitedPassage
from aat.dutch import analyze_passage, analyze_passages, tokenize

# "Hij heeft Cicero nooit gelezen." -- notes/dutch.md's compound-verb,
# active-voice worked example: action anchored on the principal verb
# "gelezen" (id t5), value "heeft gelezen" (the adverb "nooit" excluded);
# agent "Hij" (t1); target "Cicero" (t3).
_ANSWER = {
    "reasoning": "gelezen is the principal verb of the compound 'heeft gelezen'; Hij is agent; Cicero is target.",
    "nodes": [
        {"context": "ex.1", "id": "t5", "value": "heeft gelezen", "role": "action", "related_node": None},
        {"context": "ex.1", "id": "t1", "value": "Hij", "role": "agent", "related_node": "t5"},
        {"context": "ex.1", "id": "t3", "value": "Cicero", "role": "target", "related_node": "t5"},
    ],
}


def test_analyze_passage_compound_verb_worked_example():
    dspy.configure(lm=DummyLM([_ANSWER]))

    tokens, graph = analyze_passage("Hij heeft Cicero nooit gelezen.", context="ex.1")

    assert [t.id for t in tokens] == ["t1", "t2", "t3", "t4", "t5", "t6"]
    assert graph.actions()[0].id == "t5"
    assert graph.actions()[0].value == "heeft gelezen"
    assert graph.agents()[0].value == "Hij"
    assert graph.targets()[0].value == "Cicero"


def test_analyze_passage_defaults_to_empty_context():
    dspy.configure(lm=DummyLM([{**_ANSWER, "nodes": []}]))
    tokens, graph = analyze_passage("Hoi.")
    assert tokens[0].context == ""


def test_analyze_passages_concatenates_multiple_passages():
    dspy.configure(lm=DummyLM([_ANSWER, _ANSWER]))

    passages = [
        CitedPassage(context="ex.1", text="Hij heeft Cicero nooit gelezen."),
        CitedPassage(context="ex.1", text="Hij heeft Cicero nooit gelezen."),
    ]
    tokens, graph = analyze_passages(passages)

    assert len(tokens) == 12  # 6 tokens per passage, two passages
    assert len(graph.actions()) == 2


def test_independent_and_dependent_action_worked_example():
    # "Alle kunsten en wetenschappen die tot de menselijke beschaving
    # bijdragen, bezitten een gemeenschappelijke band." -- notes/dutch.md's
    # worked example for one independent action ("bezitten") and one
    # dependent action ("bijdragen") related to it.
    sentence = (
        "Alle kunsten en wetenschappen die tot de menselijke beschaving "
        "bijdragen, bezitten een gemeenschappelijke band."
    )
    tokens = tokenize(CitedPassage(context="ex.2", text=sentence))
    bijdragen_id = next(t.id for t in tokens if t.value == "bijdragen")
    bezitten_id = next(t.id for t in tokens if t.value == "bezitten")

    answer = {
        "reasoning": "bezitten is independent; bijdragen is dependent on bezitten.",
        "nodes": [
            {"context": "ex.2", "id": bezitten_id, "value": "bezitten", "role": "action", "related_node": None},
            {"context": "ex.2", "id": bijdragen_id, "value": "bijdragen", "role": "action", "related_node": bezitten_id},
        ],
    }
    dspy.configure(lm=DummyLM([answer]))

    _, graph = analyze_passage(sentence, context="ex.2")

    actions = {a.id: a for a in graph.actions()}
    assert actions[bezitten_id].related_node is None
    assert actions[bijdragen_id].related_node == bezitten_id


def test_validation_warnings_go_to_stderr_not_stdout(capsys):
    broken_answer = {
        "reasoning": "gelezen is the action; Hij is the agent, but related_node is broken.",
        "nodes": [
            {"context": "ex.1", "id": "t5", "value": "heeft gelezen", "role": "action", "related_node": None},
            {"context": "ex.1", "id": "t1", "value": "Hij", "role": "agent", "related_node": "ghost"},
        ],
    }
    dspy.configure(lm=DummyLM([broken_answer]))

    analyze_passage("Hij heeft Cicero nooit gelezen.", context="ex.1")

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Validation warnings" in captured.err
    assert "ghost" in captured.err


def test_analyze_units_by_sentence_spans_citation_units():
    from aat.dutch import analyze_units_by_sentence

    # Two citation units that only form a complete sentence together,
    # the same shape as aat.dutch.sentences's own module docstring
    # (ported from the real English Genesis 1:14-15 incident).
    answer = {
        "reasoning": "bezitten is the action; kunsten is the agent.",
        "nodes": [
            {"context": "urn:cts:test:werk:1.1-1.2", "id": "1.2.t1", "value": "bezitten", "role": "action", "related_node": None},
            {"context": "urn:cts:test:werk:1.1-1.2", "id": "1.1.t2", "value": "kunsten", "role": "agent", "related_node": "1.2.t1"},
        ],
    }
    dspy.configure(lm=DummyLM([answer]))

    units = [
        CitedPassage(context="urn:cts:test:werk:1.1", text="De kunsten"),
        CitedPassage(context="urn:cts:test:werk:1.2", text="bezitten een band."),
    ]
    tokens, graph = analyze_units_by_sentence(units)

    assert all(t.context == "urn:cts:test:werk:1.1-1.2" for t in tokens)
    assert graph.actions()[0].id == "1.2.t1"
    assert graph.agents()[0].id == "1.1.t2"
