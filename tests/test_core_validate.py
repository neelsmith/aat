"""Offline tests for aat.core.validate -- no dspy or network needed."""

from aat.core import AATGraph, AATNode, CitableToken, validate

_TOKENS = [
    CitableToken(context="c1", id="t2", value="dog"),
    CitableToken(context="c1", id="t3", value="ate"),
    CitableToken(context="c1", id="t5", value="homework"),
]


def test_validate_clean_graph_has_no_problems():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )
    assert validate(_TOKENS, graph) == []


def test_validate_flags_unknown_token_id():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t99", value="ghost", role="action", related_node=None)]
    )
    problems = validate(_TOKENS, graph)
    assert any("t99" in p for p in problems)


def test_validate_flags_agent_with_no_related_node():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node=None),
        ]
    )
    problems = validate(_TOKENS, graph)
    assert any("agent" in p and "related_node" in p for p in problems)


def test_validate_flags_related_node_not_pointing_at_an_action():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t2"),
        ]
    )
    problems = validate(_TOKENS, graph)
    assert any("t2" in p for p in problems)


def test_validate_flags_empty_value():
    graph = AATGraph(nodes=[AATNode(context="c1", id="t3", value="", role="action", related_node=None)])
    problems = validate(_TOKENS, graph)
    assert any("empty value" in p for p in problems)


def test_validate_flags_dependent_action_pointing_at_non_action():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node="t2"),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
        ]
    )
    problems = validate(_TOKENS, graph)
    assert any("t2" in p and "action node" in p for p in problems)
