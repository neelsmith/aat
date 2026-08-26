"""Offline tests for aat.core.graph -- no dspy or network needed."""

from aat.core import AATGraph, AATNode


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def test_by_id_finds_node():
    graph = _dog_ate_homework_graph()
    node = graph.by_id("c1", "t3")
    assert node is not None
    assert node.role == "action"


def test_by_id_returns_none_for_missing_id_or_wrong_context():
    graph = _dog_ate_homework_graph()
    assert graph.by_id("c1", "nope") is None
    assert graph.by_id("other-context", "t3") is None


def test_actions_agents_targets_accessors():
    graph = _dog_ate_homework_graph()
    assert [n.id for n in graph.actions()] == ["t3"]
    assert [n.id for n in graph.agents()] == ["t2"]
    assert [n.id for n in graph.targets()] == ["t5"]


def test_agents_for_and_targets_for():
    graph = _dog_ate_homework_graph()
    action = graph.actions()[0]
    assert [n.id for n in graph.agents_for(action)] == ["t2"]
    assert [n.id for n in graph.targets_for(action)] == ["t5"]


def test_governing_action_for_dependent_and_independent():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
        ]
    )
    dependent = graph.by_id("c1", "t6")
    independent = graph.by_id("c1", "t2")
    assert graph.governing_action(dependent).id == "t2"
    assert graph.governing_action(independent) is None


def test_governing_action_none_when_related_node_unresolved():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t6", value="ate", role="action", related_node="ghost")]
    )
    assert graph.governing_action(graph.actions()[0]) is None
