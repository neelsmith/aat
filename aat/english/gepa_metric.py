"""
GEPA metric for optimizing dspy_signatures.py's AgentActionTarget
signature.

GEPA (see dspy.GEPA) is a "reflective" prompt optimizer: it uses an LM to
read the metric's *feedback* text -- not just its numeric score -- and
propose better instructions. That makes the feedback string here at least
as important as the score itself: it should name the actual node, role,
and expected-vs-got values, in the same vocabulary aat-model.md and
AgentActionTarget's docstring use.

Scoring is a plain per-node comparison: for each gold node, does the
predicted graph have a node at the same (context, id, role) with a
matching `value` and `related_node`? This module has no dependency on
tests/fixtures/gold_examples.py or dspy's GEPA machinery itself --
optimize_gepa.py wires this metric, GOLD_EXAMPLES, and dspy.GEPA
together. Keeping the metric here, dependency-free, makes it importable
and unit-testable (see tests/test_gepa_metric.py) without ever touching
the network or the GOLD_EXAMPLES fixtures module.
"""

from typing import Optional

import dspy


def _key(node) -> tuple:
    return (node.context, node.id, node.role)


def aat_metric(
    gold: "dspy.Example",
    pred: "dspy.Prediction",
    trace: Optional[object] = None,
    pred_name: Optional[str] = None,
    pred_trace: Optional[object] = None,
    program_trace: Optional[object] = None,
) -> "dspy.Prediction":
    """Score a prediction's `nodes` against `gold.nodes` (both lists of
    AATNode). Returns dspy.Prediction(score=..., feedback=...) -- GEPA's
    expected "ScoreWithFeedback" shape. `pred_name`/`pred_trace` are
    accepted for protocol compatibility (GEPA can call a metric either at
    the program level or per-predictor level) but not used to change
    scoring -- there's only one predictor in `analyze`.

    `score` is in [0, 1]: the fraction of (gold-node-count +
    extra-predicted-node-count) correctly accounted for, where a gold
    node counts as correct only if the prediction has a node at the same
    (context, id, role) with a matching `value` and `related_node`.
    `feedback` names every missing, extra, or mismatched node.
    """
    problems = []

    gold_nodes = {_key(n): n for n in gold.nodes}
    pred_nodes_list = list(getattr(pred, "nodes", None) or [])
    pred_nodes = {}
    for n in pred_nodes_list:
        k = _key(n)
        if k in pred_nodes:
            problems.append(
                f"nodes has more than one entry for {k} (only the first is scored)"
            )
            continue
        pred_nodes[k] = n

    correct = 0
    total = len(gold_nodes)

    for key, g in gold_nodes.items():
        context, id_, role = key
        if key not in pred_nodes:
            problems.append(f"missing {role} node ({context!r}, {id_!r}) -- value {g.value!r}")
            continue
        p = pred_nodes[key]
        ok = True
        if p.value != g.value:
            problems.append(
                f"{role} node ({context!r}, {id_!r}): expected value={g.value!r}, got {p.value!r}"
            )
            ok = False
        if p.related_node != g.related_node:
            problems.append(
                f"{role} node ({context!r}, {id_!r}): expected related_node={g.related_node!r}, "
                f"got {p.related_node!r}"
            )
            ok = False
        if ok:
            correct += 1

    extra_keys = set(pred_nodes) - set(gold_nodes)
    for context, id_, role in sorted(extra_keys):
        problems.append(
            f"unexpected extra {role} node ({context!r}, {id_!r}) not in the gold answer"
        )
    total += len(extra_keys)

    score = correct / total if total else 1.0

    if not problems:
        feedback = (
            "Perfect match with the gold analysis: every node's role, value, "
            "and related_node are correct."
        )
    else:
        feedback = (
            f"Score {score:.2f} ({correct}/{total} nodes correct). Problems found:\n- "
            + "\n- ".join(problems)
        )

    return dspy.Prediction(score=score, feedback=feedback)
