"""
Runs every GOLD_EXAMPLES entry through analyze() via DummyLM and checks
that the resulting graph passes validate() cleanly and matches the gold
nodes exactly. See DEVELOPMENT.md for how GOLD_EXAMPLES grows over time.
"""

from aat.core import AATGraph, AATNode, validate
from conftest import run_gold_example
from fixtures.gold_examples import GOLD_EXAMPLES


def test_every_gold_example_validates_and_matches_exactly():
    for example in GOLD_EXAMPLES:
        tokens, result = run_gold_example(example)
        graph = AATGraph(nodes=list(result.nodes))

        problems = validate(tokens, graph)
        assert problems == [], f"{example.slug}: {problems}"

        expected = [AATNode(**n) for n in example.canned_nodes]
        assert graph.nodes == expected, f"{example.slug}: node mismatch"
