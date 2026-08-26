"""
Confirms GOLD_EXAMPLES collectively exercises every role and both
related_node states for an action node (independent vs dependent) -- a
small-scale analog of arsgrammatica's test_coverage.py, scaled to aat's
much smaller vocabulary (three roles, plus "does an action have a
related_node or not", plus "is an action simple or compound").

The no-agent/no-target checks below work per-*action*, not per-example:
an example can contain more than one action node (e.g. "He said that the
dog ate his homework." has both "said", which has no target, and "ate",
which has one) -- aggregating role membership across a whole example
would let one action's target mask another action's missing one. Building
a real AATGraph and using agents_for()/targets_for() per action avoids
that.
"""

from aat.core import AATGraph, AATNode
from fixtures.gold_examples import GOLD_EXAMPLES


def _graph(example):
    return AATGraph(nodes=[AATNode(**n) for n in example.canned_nodes])


def test_every_role_is_exercised():
    seen_roles = {n["role"] for ex in GOLD_EXAMPLES for n in ex.canned_nodes}
    assert seen_roles == {"agent", "action", "target"}


def test_both_independent_and_dependent_actions_are_exercised():
    action_related = {
        n["related_node"] is None
        for ex in GOLD_EXAMPLES
        for n in ex.canned_nodes
        if n["role"] == "action"
    }
    assert action_related == {True, False}, "need both an independent and a dependent action example"


def test_both_simple_and_compound_actions_are_exercised():
    action_values = [n["value"] for ex in GOLD_EXAMPLES for n in ex.canned_nodes if n["role"] == "action"]
    assert any(" " in v for v in action_values), "need a compound-action example"
    assert any(" " not in v for v in action_values), "need a simple single-token action example"


def test_both_active_and_passive_voice_are_exercised():
    tags = {t for ex in GOLD_EXAMPLES for t in ex.tags}
    assert "active-voice" in tags
    assert "passive-voice" in tags


def test_an_action_with_no_agent_is_exercised():
    for ex in GOLD_EXAMPLES:
        graph = _graph(ex)
        for action in graph.actions():
            if not graph.agents_for(action):
                return
    raise AssertionError("need an action with no agent (e.g. an agentless passive)")


def test_an_action_with_no_target_is_exercised():
    for ex in GOLD_EXAMPLES:
        graph = _graph(ex)
        for action in graph.actions():
            if not graph.targets_for(action):
                return
    raise AssertionError("need an action with no target (e.g. a verb taking a clausal complement)")
