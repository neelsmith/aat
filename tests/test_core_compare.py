"""Tests for aat.core.compare: aat_identical, aat_similar, aat_compare."""

import pytest

from aat.core import AATGraph, AATNode, aat_identical, aat_similar, aat_compare, AATComparison


def _dog_ate_homework_graph():
    """"The dog ate the homework." -- one independent action, one agent,
    one target. Same shape as the example in test_core_graph.py."""
    return AATGraph(nodes=[
        AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
        AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
        AATNode(context="c1", id="t5", value="homework", role="target", related_node="t3"),
    ])


def _cat_ate_fish_graph():
    """"The cat ate the fish." -- same shape as above, entirely different
    context/ids/values."""
    return AATGraph(nodes=[
        AATNode(context="c2", id="x9", value="ate", role="action", related_node=None),
        AATNode(context="c2", id="x8", value="cat", role="agent", related_node="x9"),
        AATNode(context="c2", id="x11", value="fish", role="target", related_node="x9"),
    ])


def _dog_ate_graph_no_target():
    """"The dog ate." -- one action, one agent, no target."""
    return AATGraph(nodes=[
        AATNode(context="c1", id="t3", value="ate", role="action", related_node=None),
        AATNode(context="c1", id="t2", value="dog", role="agent", related_node="t3"),
    ])


def _dependent_action_graph():
    """"The dog ate the homework because it was hungry." -- an independent
    action with a dependent action subordinate to it, each with their own
    agent."""
    return AATGraph(nodes=[
        AATNode(context="c3", id="a1", value="ate", role="action", related_node=None),
        AATNode(context="c3", id="a2", value="dog", role="agent", related_node="a1"),
        AATNode(context="c3", id="a3", value="homework", role="target", related_node="a1"),
        AATNode(context="c3", id="a4", value="was hungry", role="action", related_node="a1"),
        AATNode(context="c3", id="a5", value="it", role="agent", related_node="a4"),
    ])


def _empty_graph():
    return AATGraph(nodes=[])


# -- aat_identical --------------------------------------------------------

def test_identical_same_shape_different_passage():
    assert aat_identical(_dog_ate_homework_graph(), _cat_ate_fish_graph()) is True


def test_identical_node_order_does_not_matter():
    g = _dog_ate_homework_graph()
    reordered = AATGraph(nodes=list(reversed(g.nodes)))
    assert aat_identical(g, reordered) is True


def test_identical_false_when_target_missing():
    assert aat_identical(_dog_ate_homework_graph(), _dog_ate_graph_no_target()) is False


def test_identical_false_for_dependent_action_shape():
    assert aat_identical(_dog_ate_homework_graph(), _dependent_action_graph()) is False


def test_identical_true_for_same_dependent_shape():
    other = AATGraph(nodes=[
        AATNode(context="zz", id="p1", value="ran", role="action", related_node=None),
        AATNode(context="zz", id="p2", value="she", role="agent", related_node="p1"),
        AATNode(context="zz", id="p3", value="race", role="target", related_node="p1"),
        AATNode(context="zz", id="p4", value="was tired", role="action", related_node="p1"),
        AATNode(context="zz", id="p5", value="she", role="agent", related_node="p4"),
    ])
    assert aat_identical(_dependent_action_graph(), other) is True


def test_identical_both_empty():
    assert aat_identical(_empty_graph(), _empty_graph()) is True


def test_identical_unreachable_mutual_reference_is_not_a_crash():
    # Two action nodes that point at each other are never roots (neither
    # has related_node=None), so they're simply unreachable from any root
    # -- not a cycle the traversal ever walks into. This isn't malformed
    # in a way compare.py needs to reject; aat.core.validate() is where
    # referential-integrity problems like a dangling related_node belong.
    mutual = AATGraph(nodes=[
        AATNode(context="c9", id="n1", value="a", role="action", related_node="n2"),
        AATNode(context="c9", id="n2", value="b", role="action", related_node="n1"),
    ])
    assert aat_identical(mutual, _empty_graph()) is True


def test_identical_cycle_raises():
    # A genuine infinite loop in this data model requires a duplicate
    # (context, id) pair to reappear on its own descendant chain -- since
    # each node has a single, fixed related_node, two *distinct* ids can
    # never form a cycle reachable from a root (see the mutual-reference
    # test above). Here "r1" appears twice: once as the real root, and
    # again (a different AATNode object, same id) as a "grandchild" of
    # itself via r2 -- so walking down from the real root loops forever
    # without the cycle guard.
    cyclic = AATGraph(nodes=[
        AATNode(context="c9", id="r1", value="a", role="action", related_node=None),
        AATNode(context="c9", id="r2", value="b", role="action", related_node="r1"),
        AATNode(context="c9", id="r1", value="a-dup", role="action", related_node="r2"),
    ])
    with pytest.raises(ValueError):
        aat_identical(cyclic, cyclic)


# -- aat_similar ------------------------------------------------------------

def test_similar_true_when_agents_targets_differ_but_actions_match():
    # Same single independent action, but one graph has an extra target.
    g1 = _dog_ate_homework_graph()
    g2 = _dog_ate_graph_no_target()
    assert aat_similar(g1, g2) is True


def test_similar_false_when_action_shape_differs():
    assert aat_similar(_dog_ate_homework_graph(), _dependent_action_graph()) is False


def test_similar_implies_not_necessarily_identical():
    g1 = _dog_ate_homework_graph()
    g2 = _dog_ate_graph_no_target()
    assert aat_similar(g1, g2) is True
    assert aat_identical(g1, g2) is False


# -- aat_compare --------------------------------------------------------

def test_compare_ratios():
    result = aat_compare(_dependent_action_graph(), _dog_ate_homework_graph())
    assert isinstance(result, AATComparison)
    assert result.action_node_ratio == pytest.approx(2 / 1)
    assert result.depth_ratio == pytest.approx(2 / 1)
    assert result.size_ratio == pytest.approx(5 / 3)


def test_compare_identical_graphs_all_ratios_one():
    g1 = _dog_ate_homework_graph()
    g2 = _cat_ate_fish_graph()
    result = aat_compare(g1, g2)
    assert result.action_node_ratio == pytest.approx(1.0)
    assert result.depth_ratio == pytest.approx(1.0)
    assert result.size_ratio == pytest.approx(1.0)


def test_compare_raises_on_zero_denominator():
    with pytest.raises(ValueError):
        aat_compare(_dog_ate_homework_graph(), _empty_graph())
