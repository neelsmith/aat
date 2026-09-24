"""
Structural comparison of AAT graphs (quarto/bg/aatgraphs.qmd): "AAT-
identical", "AAT-similar", and the three differencing metrics (action-node
count, depth, size) for graphs that are neither.

Comparison here is purely structural: two nodes correspond if they play
the same *role* at the same position in the graph's shape, regardless of
the underlying token id, context, or value -- so `aat_identical` can
compare graphs built from entirely different passages (or even different
languages -- see aat.english vs aat.dutch), not just re-analyses of the
same text. This is a different job from aat.english.gepa_metric's
aat_metric, which does an exact (context, id, role, value, related_node)
comparison for scoring a prediction against gold on one specific passage.

This module deliberately mirrors AATGraph's own "node order is not
semantically significant" stance (see graph.py): children at any level of
a graph's shape are compared as an unordered multiset, not a sequence, so
two graphs with the same nodes in a different order still compare equal.
"""

from typing import List

from pydantic import BaseModel, Field

from .graph import AATGraph, AATNode


class AATComparison(BaseModel):
    """Result of `aat_compare`: three g1/g2 ratios, meant for graphs that
    are neither AAT-identical nor AAT-similar (aatgraphs.qmd's
    "Differences in AAT graphs" section names these three as the useful
    ones)."""

    action_node_ratio: float = Field(
        description="(number of action nodes in g1) / (number of action nodes in g2)."
    )
    depth_ratio: float = Field(description="(depth of g1) / (depth of g2).")
    size_ratio: float = Field(description="(size of g1) / (size of g2).")


class _CycleError(ValueError):
    """Raised internally when related_node chains among action nodes form
    a cycle -- a malformed graph validate() doesn't itself catch (it only
    checks referential integrity per-node, not multi-hop chains). Caught
    and re-raised with graph-comparison-specific context by the public
    functions below, rather than left to surface as unbounded recursion."""


def _dependents(graph: AATGraph, action: AATNode) -> List[AATNode]:
    """Every action node whose related_node points at `action`'s id,
    within the same context -- the action-tree counterpart of
    AATGraph.agents_for/targets_for. Only this module needs it, so it
    isn't added to AATGraph's own public API."""
    return [
        n
        for n in graph.nodes
        if n.role == "action"
        and n.context == action.context
        and n.related_node == action.id
    ]


def _roots(graph: AATGraph) -> List[AATNode]:
    """Every independent action node (related_node is None) -- the roots
    of the graph's action forest."""
    return [n for n in graph.actions() if n.related_node is None]


def _canonical(graph: AATGraph, node: AATNode, _seen: frozenset = frozenset()) -> str:
    """A canonical string for the subtree rooted at `node`, ignoring node
    value/id/context entirely -- only `role`, and (for action nodes) the
    unordered multiset of children's own canonical strings, contribute.
    Two subtrees have the same canonical string iff they have the same
    AAT structure and edge values (aatgraphs.qmd's "AAT-identical").
    `_seen` tracks the (context, id) pairs on the current action chain, to
    raise _CycleError instead of recursing forever on a malformed graph
    with a related_node cycle."""
    if node.role != "action":
        return node.role  # agent/target nodes are always leaves

    key = (node.context, node.id)
    if key in _seen:
        raise _CycleError(f"related_node cycle detected at action node {key!r}")
    seen = _seen | {key}

    children = [
        _canonical(graph, child, seen)
        for child in graph.agents_for(node) + graph.targets_for(node) + _dependents(graph, node)
    ]
    children.sort()
    return "action(" + ",".join(children) + ")"


def _graph_canonical(graph: AATGraph) -> List[str]:
    """The graph's overall canonical form: the sorted multiset of its
    root action subtrees' canonical strings. Sorted (not just listed) so
    two graphs with the same roots in a different order compare equal."""
    return sorted(_canonical(graph, root) for root in _roots(graph))


def aat_identical(g1: AATGraph, g2: AATGraph) -> bool:
    """True if `g1` and `g2` are AAT-identical (aatgraphs.qmd): same
    structure and same edge values (role labels at each position),
    comparing node role and tree shape only -- node value, id, and
    context are never considered, so this can compare graphs built from
    entirely different passages. Raises ValueError if either graph's
    action nodes form a related_node cycle."""
    try:
        return _graph_canonical(g1) == _graph_canonical(g2)
    except _CycleError as exc:
        raise ValueError(str(exc)) from exc


def _actions_only(graph: AATGraph) -> AATGraph:
    """`graph` with every agent/target node removed, keeping only action
    nodes (and therefore only the action-dependency edges between them)."""
    return AATGraph(nodes=[n for n in graph.nodes if n.role == "action"])


def aat_similar(g1: AATGraph, g2: AATGraph) -> bool:
    """True if `g1` and `g2` are AAT-similar (aatgraphs.qmd): same number
    of action nodes in the same relations -- equivalently, per the doc,
    AAT-identical once agent and target nodes are removed from both."""
    return aat_identical(_actions_only(g1), _actions_only(g2))


def _depth(graph: AATGraph) -> int:
    """The graph's depth: the longest chain of action nodes connected by
    related_node (an independent action -> a dependent action -> that
    action's own dependent, and so on), counting the root action itself
    as depth 1. Agent/target attachments don't contribute to depth. 0 for
    a graph with no action nodes. Raises ValueError on a related_node
    cycle."""

    def chain_depth(action: AATNode, seen: frozenset) -> int:
        key = (action.context, action.id)
        if key in seen:
            raise _CycleError(f"related_node cycle detected at action node {key!r}")
        seen = seen | {key}
        dependents = _dependents(graph, action)
        if not dependents:
            return 1
        return 1 + max(chain_depth(child, seen) for child in dependents)

    try:
        return max((chain_depth(root, frozenset()) for root in _roots(graph)), default=0)
    except _CycleError as exc:
        raise ValueError(str(exc)) from exc


def _size(graph: AATGraph) -> int:
    """The graph's size: its total node count (agents + actions + targets
    together) -- deliberately not just the action-node count, so it also
    reflects how elaborated a graph's agent/target annotations are (the
    action_node_ratio field covers the action-only comparison)."""
    return len(graph.nodes)


def aat_compare(g1: AATGraph, g2: AATGraph) -> AATComparison:
    """Compare `g1` to `g2` on the three metrics aatgraphs.qmd names as
    useful for graphs that are neither AAT-identical nor AAT-similar:
    number of action nodes, depth, and size, each as a g1/g2 ratio.
    Raises ValueError if any of g2's three corresponding values is 0 (the
    ratio would be undefined), or on a related_node cycle in either
    graph."""
    g1_actions, g2_actions = len(g1.actions()), len(g2.actions())
    g1_depth, g2_depth = _depth(g1), _depth(g2)
    g1_size, g2_size = _size(g1), _size(g2)

    if g2_actions == 0:
        raise ValueError("cannot compute action_node_ratio: g2 has no action nodes")
    if g2_depth == 0:
        raise ValueError("cannot compute depth_ratio: g2 has depth 0 (no action nodes)")
    if g2_size == 0:
        raise ValueError("cannot compute size_ratio: g2 has no nodes")

    return AATComparison(
        action_node_ratio=g1_actions / g2_actions,
        depth_ratio=g1_depth / g2_depth,
        size_ratio=g1_size / g2_size,
    )
