"""
Runs every GOLD_EXAMPLES_DUTCH entry through aat.dutch.analyze() via
DummyLM and checks that the resulting graph passes validate() cleanly and
matches the gold nodes exactly. Mirrors test_gold_examples.py; see that
file and notes/DEVELOPMENT.md for how the fixture grows over time (the
English document, but the same principle applies here).
"""

from aat.core import AATGraph, AATNode, validate
from aat.dutch import analyze as dutch_analyze
from conftest import run_gold_example
from fixtures.gold_examples_dutch import GOLD_EXAMPLES_DUTCH


def test_every_gold_example_validates_and_matches_exactly():
    for example in GOLD_EXAMPLES_DUTCH:
        tokens, result = run_gold_example(example, analyze=dutch_analyze)
        graph = AATGraph(nodes=list(result.nodes))

        problems = validate(tokens, graph)
        assert problems == [], f"{example.slug}: {problems}"

        expected = [AATNode(**n) for n in example.canned_nodes]
        assert graph.nodes == expected, f"{example.slug}: node mismatch"
