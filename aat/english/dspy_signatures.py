"""
DSPy program that applies the Agent-Action-Target model (aat-model.md) to
a passage of English, given its pre-tokenized token list.

This module covers only the extraction stage:
  1. AgentActionTarget -- a dspy.Signature that takes the passage's text
     plus its pre-tokenized CitableToken list and produces `nodes`, an
     AATGraph's worth of AATNode, using the ids handed to it.
  2. analyze -- the dspy.ChainOfThought instance built from that
     signature; the same module-level instance every pipeline call and
     every GEPA run uses (see pipeline.py, optimize_gepa.py).
  3. validate() -- a thin wrapper around aat.core.validate.validate,
     checking that analyze()'s output is referentially sound given the
     input tokens.

Run this file directly for a quick smoke test against the configured LM:
    python -m aat.english.dspy_signatures
"""

from typing import List

import dspy

from aat.core import AATGraph, AATNode, CitableToken
from aat.core.validate import validate as _validate_graph


class AgentActionTarget(dspy.Signature):
    """Analyze an English passage according to the Agent-Action-Target
    (AAT) model documented in aat-model.md.

    Extract every *action* (verbal expression): a single token for a
    simple verb (e.g. "ate"), or, for a compound verbal expression (e.g.
    "was eating", "was being eaten"), a node whose `id` is the *principal
    verb*'s token id (the last, most specific component -- e.g. "eating"
    in "was eating", "eaten" in "was being eaten") and whose `value` is
    every component auxiliary/verb token's text joined by spaces in
    surface order (e.g. "was eating"). An adverb that happens to
    interrupt the auxiliary chain (e.g. "not", "yet" in "was not yet
    being eaten") is NOT a component token -- it is excluded from
    `value`, which is still just "was being eaten". An *independent*
    action (not embedded in another clause) has related_node=None; a
    *dependent* (subordinate) action has related_node set to the id of
    the action node for the clause that governs it.

    For every action, also extract:
      - its *agent*: the subject of an active-voice, intransitive, or
        linking verb, or the by-phrase agent of a passive-voice verb.
        related_node = that action's id.
      - its *target*: the direct object of a transitive active-voice
        verb, the predicate of a linking verb, or the subject of a
        transitive passive-voice verb. related_node = that action's id.
      Either may be absent if the passage doesn't express one (e.g. an
      intransitive verb has no target; an agentless passive has no
      agent; a verb taking a clausal complement rather than a noun-phrase
      direct object -- e.g. "said" in "He said that...") has no target).

    Every node's `id` must be the id of an existing token from the input
    `tokens` list (the principal verb's id, for a compound action) --
    never a new id not present in `tokens`. Every node's `context` must
    match the context shared by every token in `tokens`.

    Worked examples from aat-model.md:
      - "The dog ate my homework." -> action node id=<id of "ate">,
        value="ate", related_node=None; agent node id=<id of "dog">,
        value="dog", related_node=<the action's id>; target node id=<id
        of "homework">, value="homework", related_node=<the action's id>.
      - "The homework was eaten by the dog." (passive) -> action node
        anchored on "eaten" (value "was eaten"); agent node on "dog"
        (the by-phrase agent); target node on "homework" (the passive
        subject).
      - "The homework was not yet being eaten." (compound, passive,
        no agent expressed) -> one action node, id=<id of "eaten">,
        value="was being eaten" (the adverbs "not"/"yet" are excluded),
        related_node=None; target node on "homework"; no agent node.
      - "He said that the dog ate his homework." (independent + dependent
        clause) -> action node on "said" (related_node=None, no target
        since it takes a clausal complement) with its own agent ("He");
        action node on "ate" with related_node=<"said"'s id>, with its
        own agent ("dog") and target ("homework").
    """

    passage: str = dspy.InputField(description="The English passage's raw text.")
    tokens: List[CitableToken] = dspy.InputField(
        description="This passage's pre-tokenized CitableToken list, in reading order."
    )
    nodes: List[AATNode] = dspy.OutputField(
        description="Every agent/action/target node this passage yields, per the AAT model."
    )


analyze = dspy.ChainOfThought(AgentActionTarget)


def validate(tokens: List[CitableToken], prediction: "dspy.Prediction") -> List[str]:
    """Validate an `analyze()` prediction's `nodes` against its input
    `tokens` -- a thin wrapper around aat.core.validate.validate, so
    callers working only with aat.english don't need their own separate
    import from aat.core just for this."""
    return _validate_graph(tokens, AATGraph(nodes=list(prediction.nodes)))


if __name__ == "__main__":
    from aat.core import CitedPassage

    from .tokenize import tokenize

    passage = CitedPassage(context="smoketest", text="The dog ate my homework.")
    tokens = tokenize(passage)
    result = analyze(passage=passage.text, tokens=tokens)
    print(result.nodes)
    for problem in validate(tokens, result):
        print("problem:", problem)
