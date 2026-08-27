"""Offline tests for aat.core.coloring.assign_action_colors()."""

from aat.core import AATGraph, AATNode
from aat.core.coloring import assign_action_colors


def _dog_ate_homework_graph():
    return AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
            AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
        ]
    )


def test_action_agent_and_target_in_one_cluster_share_a_color():
    colors, warnings = assign_action_colors(_dog_ate_homework_graph())
    assert warnings == []
    assert colors[("c1", "t3")] == colors[("c1", "t2")] == colors[("c1", "t5")]


def test_dependent_action_gets_its_own_color_not_its_governors():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t2", value="said", role="action", related_node=None),
            AATNode(context="c1", id="t1", value="He", role="agent", related_node="t2"),
            AATNode(context="c1", id="t6", value="ate", role="action", related_node="t2"),
            AATNode(context="c1", id="t5", value="dog", role="agent", related_node="t6"),
        ]
    )
    colors, _warnings = assign_action_colors(graph)
    assert colors[("c1", "t2")] == colors[("c1", "t1")]
    assert colors[("c1", "t6")] == colors[("c1", "t5")]
    assert colors[("c1", "t2")] != colors[("c1", "t6")]


def test_colors_repeat_and_warn_past_palette_size():
    nodes = [
        AATNode(context="c1", id=f"t{i}", value=f"action{i}", role="action", related_node=None)
        for i in range(10)
    ]
    colors, warnings = assign_action_colors(AATGraph(nodes=nodes))
    assert any("only 8 palette" in w for w in warnings)
    assert colors[("c1", "t0")] == colors[("c1", "t8")]
    assert colors[("c1", "t1")] == colors[("c1", "t9")]
    assert colors[("c1", "t0")] != colors[("c1", "t1")]


def test_empty_graph_gives_no_colors_and_no_warnings():
    colors, warnings = assign_action_colors(AATGraph(nodes=[]))
    assert colors == {}
    assert warnings == []


def test_node_with_broken_related_node_is_left_uncolored():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
            AATNode(context="c1", id="t9", value="ghost", role="agent", related_node="t404"),
        ]
    )
    colors, _warnings = assign_action_colors(graph)
    assert ("c1", "t3") in colors
    assert ("c1", "t9") not in colors


def test_two_contexts_use_distinct_compound_keys_and_palette_slots():
    graph = AATGraph(
        nodes=[
            AATNode(context="c1", id="t1", value="ran", role="action", related_node=None),
            AATNode(context="c2", id="t1", value="jumped", role="action", related_node=None),
        ]
    )
    colors, _warnings = assign_action_colors(graph)
    # Same bare id in two different contexts -- tracked under distinct
    # (context, id) compound keys, per graph.py's own (context, id)
    # convention. Palette assignment walks the full node list (across
    # every context, in list order), not restarted per context, so the
    # second context's only action still gets its own, later, slot.
    assert set(colors.keys()) == {("c1", "t1"), ("c2", "t1")}
    assert colors[("c1", "t1")] != colors[("c2", "t1")]
