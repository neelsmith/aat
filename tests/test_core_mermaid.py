"""Offline tests for aat.core.mermaid -- no dspy or network needed."""

import pytest

from aat.core import AATGraph, AATNode
from aat.core.mermaid import graph_to_mermaid


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def test_every_node_becomes_a_shaped_node_line():
    diagram, warnings = graph_to_mermaid(_dog_ate_homework_graph())
    assert warnings == []
    assert '    t3["ate"]' in diagram
    assert '    t2("dog")' in diagram
    assert '    t5(["homework"])' in diagram


def test_agent_and_target_edges_point_at_the_action():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph())
    assert "    t2 -->|agent| t3" in diagram
    assert "    t5 -->|target| t3" in diagram


def test_independent_action_has_no_outgoing_edge():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph())
    assert "    t3 -->" not in diagram


def test_dependent_action_edge_is_labelled_dependent():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
        ]
    )
    diagram, warnings = graph_to_mermaid(graph)
    assert warnings == []
    assert "    t6 -->|dependent| t2" in diagram


def test_default_orientation_is_bt():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph())
    assert diagram.startswith("graph BT")


@pytest.mark.parametrize("orientation", ["TB", "TD", "BT", "RL", "LR"])
def test_every_valid_orientation_is_accepted_and_used_verbatim(orientation):
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph(), orientation=orientation)
    assert diagram.startswith(f"graph {orientation}")


def test_orientation_is_matched_case_insensitively_and_uppercased():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph(), orientation="lr")
    assert diagram.startswith("graph LR")


def test_orientation_surrounding_whitespace_is_stripped():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph(), orientation=" TB ")
    assert diagram.startswith("graph TB")


def test_invalid_orientation_raises_value_error_naming_valid_options():
    with pytest.raises(ValueError) as excinfo:
        graph_to_mermaid(_dog_ate_homework_graph(), orientation="sideways")
    message = str(excinfo.value)
    assert "sideways" in message
    for valid in ("TB", "TD", "BT", "RL", "LR"):
        assert valid in message


def test_broken_related_node_is_skipped_and_warned_not_crashed():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t5", value="homework", role="target", related_node="ghost")]
    )
    diagram, warnings = graph_to_mermaid(graph)
    assert "-->" not in diagram
    assert any("ghost" in w for w in warnings)


def test_color_by_action_gives_agent_and_target_the_same_class_as_their_action():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph(), color_by_action=True)
    assert "classDef c0" in diagram
    assert "class t3,t2,t5 c0;" in diagram


def test_dependent_action_gets_its_own_class_not_its_governors():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t1", value="He", role="agent", related_node="t2"),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
            AATNode(context="c1", id="t5", value="dog", role="agent", related_node="t6"),
        ]
    )
    diagram, _warnings = graph_to_mermaid(graph)
    assert "class t2,t1 c0;" in diagram
    assert "class t6,t5 c1;" in diagram


def test_color_by_action_false_gives_plain_diagram():
    diagram, _warnings = graph_to_mermaid(_dog_ate_homework_graph(), color_by_action=False)
    assert "classDef" not in diagram
    assert "class " not in diagram


def test_more_actions_than_palette_slots_warns_but_still_renders():
    nodes = [
        AATNode(context="c1", id=f"t{i}", value=f"action{i}", role="action", related_node=None)
        for i in range(10)
    ]
    diagram, warnings = graph_to_mermaid(AATGraph(nodes=nodes))
    assert any("only 8 palette" in w for w in warnings)
    # 10 actions cycle through 8 palette colors -- only 8 distinct
    # classDefs are emitted (c0..c7, not one per action), and the
    # two actions that land on a repeated color (t8 on t0's color,
    # t9 on t1's) share that class rather than getting a redundant
    # classDef of their own.
    assert "classDef c0" in diagram
    assert "classDef c7" in diagram
    assert "classDef c8" not in diagram
    assert "class t0,t8 c0;" in diagram
    assert "class t1,t9 c1;" in diagram


def test_empty_graph_renders_header_only():
    diagram, warnings = graph_to_mermaid(AATGraph(nodes=[]))
    assert diagram == "graph BT"
    assert warnings == []
