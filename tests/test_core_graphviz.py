"""Offline tests for aat.core.graphviz -- no dspy or network needed.
Mirrors tests/test_core_mermaid.py's structure and coverage, asserting
Graphviz DOT syntax instead of Mermaid syntax for the same fixtures."""

import pytest

from aat.core import AATGraph, AATNode
from aat.core.graphviz import graph_to_dot


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def test_every_node_becomes_a_shaped_node_line():
    dot, warnings = graph_to_dot(_dog_ate_homework_graph(), color_by_action=False)
    assert warnings == []
    assert '    t3 [shape=box, label="ate"];' in dot
    assert '    t2 [shape=box, style="rounded", label="dog"];' in dot
    assert '    t5 [shape=ellipse, label="homework"];' in dot


def test_agent_and_target_edges_point_at_the_action():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph())
    assert '    t2 -> t3 [label="agent"];' in dot
    assert '    t5 -> t3 [label="target"];' in dot


def test_independent_action_has_no_outgoing_edge():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph())
    assert "    t3 ->" not in dot


def test_dependent_action_edge_is_labelled_dependent():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
        ]
    )
    dot, warnings = graph_to_dot(graph)
    assert warnings == []
    assert '    t6 -> t2 [label="dependent"];' in dot


def test_default_orientation_is_bt():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph())
    assert "digraph aat {" in dot
    assert "    rankdir=BT;" in dot


@pytest.mark.parametrize("orientation", ["BT", "LR", "RL"])
def test_orientations_with_no_graphviz_specific_mapping_are_used_verbatim(orientation):
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), orientation=orientation)
    assert f"    rankdir={orientation};" in dot


def test_td_orientation_is_mapped_to_graphviz_own_tb_spelling():
    # Graphviz's rankdir has no "TD" synonym of its own -- unlike Mermaid,
    # which uses "TD" verbatim, graph_to_dot() maps it to "TB" (same
    # direction, Graphviz's own name for it).
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), orientation="TD")
    assert "    rankdir=TB;" in dot


def test_tb_orientation_is_also_used_verbatim():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), orientation="TB")
    assert "    rankdir=TB;" in dot


def test_orientation_is_matched_case_insensitively_and_uppercased():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), orientation="lr")
    assert "    rankdir=LR;" in dot


def test_orientation_surrounding_whitespace_is_stripped():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), orientation=" TB ")
    assert "    rankdir=TB;" in dot


def test_invalid_orientation_raises_value_error_naming_valid_options():
    with pytest.raises(ValueError) as excinfo:
        graph_to_dot(_dog_ate_homework_graph(), orientation="sideways")
    message = str(excinfo.value)
    assert "sideways" in message
    for valid in ("TB", "TD", "BT", "RL", "LR"):
        assert valid in message


def test_broken_related_node_is_skipped_and_warned_not_crashed():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t5", value="homework", role="target", related_node="ghost")]
    )
    dot, warnings = graph_to_dot(graph)
    assert "->" not in dot
    assert any("ghost" in w for w in warnings)


def test_color_by_action_gives_agent_and_target_the_same_fill_as_their_action():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), color_by_action=True)
    # Same (fill, stroke, text) triple as assign_action_colors()'s first
    # palette entry -- see test_core_coloring.py.
    assert dot.count('fillcolor="#E3F2FD"') == 3
    assert 'color="#1565C0"' in dot
    assert 'fontcolor="#0D47A1"' in dot
    assert 'style="filled"' in dot  # action and target: no role-extra style, just filled
    assert 'style="rounded,filled"' in dot  # agent: rounded + filled


def test_dependent_action_gets_its_own_color_not_its_governors():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t1", value="He", role="agent", related_node="t2"),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
            AATNode(context="c1", id="t5", value="dog", role="agent", related_node="t6"),
        ]
    )
    dot, _warnings = graph_to_dot(graph)
    lines = {line.strip() for line in dot.splitlines()}
    t2_line = next(line for line in lines if line.startswith("t2 ["))
    t1_line = next(line for line in lines if line.startswith("t1 ["))
    t6_line = next(line for line in lines if line.startswith("t6 ["))
    t5_line = next(line for line in lines if line.startswith("t5 ["))
    assert 'fillcolor="#E3F2FD"' in t2_line and 'fillcolor="#E3F2FD"' in t1_line
    assert 'fillcolor="#FFF3E0"' in t6_line and 'fillcolor="#FFF3E0"' in t5_line


def test_color_by_action_false_gives_plain_digraph():
    dot, _warnings = graph_to_dot(_dog_ate_homework_graph(), color_by_action=False)
    assert "fillcolor" not in dot
    assert "fontcolor" not in dot
    assert "filled" not in dot


def test_more_actions_than_palette_slots_warns_but_still_renders():
    nodes = [
        AATNode(context="c1", id=f"t{i}", value=f"action{i}", role="action", related_node=None)
        for i in range(10)
    ]
    dot, warnings = graph_to_dot(AATGraph(nodes=nodes))
    assert any("only 8 palette" in w for w in warnings)
    lines = {line.strip() for line in dot.splitlines()}
    t0_line = next(line for line in lines if line.startswith("t0 ["))
    t8_line = next(line for line in lines if line.startswith("t8 ["))
    t1_line = next(line for line in lines if line.startswith("t1 ["))
    t9_line = next(line for line in lines if line.startswith("t9 ["))
    fill = lambda line: line.split('fillcolor="')[1].split('"')[0]  # noqa: E731
    assert fill(t0_line) == fill(t8_line)
    assert fill(t1_line) == fill(t9_line)
    assert fill(t0_line) != fill(t1_line)


def test_empty_graph_renders_header_only():
    dot, warnings = graph_to_dot(AATGraph(nodes=[]))
    assert dot == "digraph aat {\n    rankdir=BT;\n}"
    assert warnings == []


def test_label_with_quote_and_backslash_is_escaped():
    graph = AATGraph(
        nodes=[AATNode(context="c1", id="t1", value='say "hi"\\bye', role="action", related_node=None)]
    )
    dot, _warnings = graph_to_dot(graph, color_by_action=False)
    assert '    t1 [shape=box, label="say \\"hi\\"\\\\bye"];' in dot
