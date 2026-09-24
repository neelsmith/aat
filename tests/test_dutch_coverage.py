"""
Confirms GOLD_EXAMPLES_DUTCH collectively exercises every role and both
related_node states for an action node (independent vs dependent) -- the
Dutch counterpart to test_coverage.py (see that file's own docstring for
the full rationale; identical checks, against the Dutch fixture).
"""

from aat.core import AATGraph, AATNode
from fixtures.gold_examples_dutch import GOLD_EXAMPLES_DUTCH


def _graph(example):
    return AATGraph(nodes=[AATNode(**n) for n in example.canned_nodes])


def test_every_role_is_exercised():
    seen_roles = {n["role"] for ex in GOLD_EXAMPLES_DUTCH for n in ex.canned_nodes}
    assert seen_roles == {"agent", "action", "target"}


def test_both_independent_and_dependent_actions_are_exercised():
    action_related = {
        n["related_node"] is None
        for ex in GOLD_EXAMPLES_DUTCH
        for n in ex.canned_nodes
        if n["role"] == "action"
    }
    assert action_related == {True, False}, "need both an independent and a dependent action example"


def test_both_simple_and_compound_actions_are_exercised():
    action_values = [n["value"] for ex in GOLD_EXAMPLES_DUTCH for n in ex.canned_nodes if n["role"] == "action"]
    assert any(" " in v for v in action_values), "need a compound-action example"
    assert any(" " not in v for v in action_values), "need a simple single-token action example"


def test_both_active_and_passive_voice_are_exercised():
    tags = {t for ex in GOLD_EXAMPLES_DUTCH for t in ex.tags}
    assert "active-voice" in tags
    assert "passive-voice" in tags


def test_an_action_with_no_agent_is_exercised():
    for ex in GOLD_EXAMPLES_DUTCH:
        graph = _graph(ex)
        for action in graph.actions():
            if not graph.agents_for(action):
                return
    raise AssertionError("need an action with no agent")


def test_an_action_with_no_target_is_exercised():
    for ex in GOLD_EXAMPLES_DUTCH:
        graph = _graph(ex)
        for action in graph.actions():
            if not graph.targets_for(action):
                return
    raise AssertionError("need an action with no target")


def test_multiple_subordination_types_are_exercised():
    # Dutch-specific, beyond test_coverage.py's own checks: notes/dutch.md
    # documents three distinct ways a clause can be subordinated (relative
    # pronoun, purpose construction, subordinating conjunction) -- confirm
    # the fixture actually exercises all three, not just "dependent" in
    # general.
    tags = {t for ex in GOLD_EXAMPLES_DUTCH for t in ex.tags}
    assert "relative-clause-subordination" in tags
    assert "purpose-subordination" in tags
    assert "conjunction-subordination" in tags


def test_a_sentence_with_two_dependent_clauses_on_one_independent_action_is_exercised():
    # notes/dutch.md's own worked example for this: a sentence can have
    # more than one dependent action, and every one of them points at the
    # SAME single independent action, not at each other. Confirm at least
    # one fixture actually has two actions sharing one related_node.
    for ex in GOLD_EXAMPLES_DUTCH:
        related = [n["related_node"] for n in ex.canned_nodes if n["role"] == "action" and n["related_node"]]
        if len(related) >= 2 and len(set(related)) == 1:
            return
    raise AssertionError("need an example with two dependent actions sharing one governing action")


def test_a_linking_verb_agent_and_target_are_exercised():
    # notes/dutch.md's Agents/Targets sections both explicitly name
    # linking verbs ("subjects of ... linking verbs" as agents,
    # "predicates of linking verbs" as targets) -- confirm the fixture
    # actually has an example tagged as exercising one.
    tags = {t for ex in GOLD_EXAMPLES_DUTCH for t in ex.tags}
    assert "linking-verb" in tags
