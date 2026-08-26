"""
Offline tests for aat.english.gepa_metric.aat_metric -- no dspy network
access needed; dspy.Example/dspy.Prediction are just plain containers
here.
"""

import dspy

from aat.core import AATNode
from aat.english.gepa_metric import aat_metric

_GOLD = [
    {"context": "c1", "id": "t3", "value": "ate", "role": "action", "related_node": None},
    {"context": "c1", "id": "t2", "value": "dog", "role": "agent", "related_node": "t3"},
    {"context": "c1", "id": "t5", "value": "homework", "role": "target", "related_node": "t3"},
]


def _example(nodes):
    return dspy.Example(nodes=[AATNode(**n) for n in nodes]).with_inputs()


def _prediction(nodes):
    return dspy.Prediction(nodes=[AATNode(**n) for n in nodes])


def test_perfect_match_scores_one():
    result = aat_metric(_example(_GOLD), _prediction(_GOLD))
    assert result.score == 1.0
    assert "Perfect match" in result.feedback


def test_missing_node_is_penalized_and_named():
    pred = _GOLD[:2]  # drop the target
    result = aat_metric(_example(_GOLD), _prediction(pred))
    assert result.score < 1.0
    assert "missing target node" in result.feedback
    assert "t5" in result.feedback


def test_wrong_value_is_penalized_and_named():
    pred = [dict(n) for n in _GOLD]
    pred[1]["value"] = "cat"
    result = aat_metric(_example(_GOLD), _prediction(pred))
    assert result.score < 1.0
    assert "expected value='dog', got 'cat'" in result.feedback


def test_wrong_related_node_is_penalized_and_named():
    pred = [dict(n) for n in _GOLD]
    pred[2]["related_node"] = "t99"
    result = aat_metric(_example(_GOLD), _prediction(pred))
    assert result.score < 1.0
    assert "expected related_node='t3', got 't99'" in result.feedback


def test_extra_node_is_penalized_and_named():
    pred = _GOLD + [
        {"context": "c1", "id": "t9", "value": "ghost", "role": "agent", "related_node": "t3"}
    ]
    result = aat_metric(_example(_GOLD), _prediction(pred))
    assert result.score < 1.0
    assert "unexpected extra agent node" in result.feedback


def test_empty_gold_and_prediction_scores_one():
    result = aat_metric(_example([]), _prediction([]))
    assert result.score == 1.0
