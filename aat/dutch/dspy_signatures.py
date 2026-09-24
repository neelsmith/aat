"""
DSPy program that applies the Agent-Action-Target model (aat-model.md,
notes/dutch.md) to a passage of Dutch, given its pre-tokenized token
list. Parallel to aat.english.dspy_signatures -- same model, same
AATNode/AATGraph shape, different language and a different worked-example
set (Dutch's own, from notes/dutch.md, rather than aat-model.md's
English ones).

This module covers only the extraction stage:
  1. AgentActionTarget -- a dspy.Signature that takes the passage's text
     plus its pre-tokenized CitableToken list and produces `nodes`, an
     AATGraph's worth of AATNode, using the ids handed to it.
  2. analyze -- the dspy.ChainOfThought instance built from that
     signature; the same module-level instance every pipeline call uses
     (see pipeline.py).
  3. validate() -- a thin wrapper around aat.core.validate.validate,
     checking that analyze()'s output is referentially sound given the
     input tokens.

Run this file directly for a quick smoke test against the configured LM:
    python -m aat.dutch.dspy_signatures
"""

from typing import List

import dspy

from aat.core import AATGraph, AATNode, CitableToken
from aat.core.validate import validate as _validate_graph


class AgentActionTarget(dspy.Signature):
    """Analyze a Dutch passage according to the Agent-Action-Target (AAT)
    model documented in aat-model.md and notes/dutch.md.

    Extract every *action* (verbal expression): a single token for a
    simple verb (e.g. "bezitten"), or, for a compound verbal expression
    (e.g. "heeft gelezen", "werd vertaald"), a node whose `id` is the
    *principal verb*'s token id (the most specific component -- e.g.
    "gelezen" in "heeft gelezen", "vertaald" in "werd vertaald", as
    distinct from the *auxiliary*, e.g. "heeft"/"werd") and whose `value`
    is every component auxiliary/verb token's text joined by spaces in
    surface order (e.g. "heeft gelezen"). An adverb that happens to
    interrupt the auxiliary chain (e.g. "nooit" in "heeft nooit
    gelezen") is NOT a component token -- it is excluded from `value`,
    which is still just "heeft gelezen". An *independent* action (not
    embedded in another clause) has related_node=None; a *dependent* (or
    *subordinate*) action has related_node set to the id of the action
    node for the clause that governs it. A clause can be subordinated
    several ways, all treated the same for this purpose: a relative
    pronoun (e.g. "die"), a purpose construction (e.g. "om ... te ..."),
    or a subordinating conjunction (e.g. "want"). When a sentence has
    more than one dependent clause, each one's related_node points at
    the single independent (main) action that governs the sentence, not
    at each other.

    For every action, also extract:
      - its *agent*: the subject of an active-voice, intransitive, or
        linking verb, or the "door"-phrase agent of a passive-voice verb
        (e.g. "Jones" in "... werd vertaald door Jones"). related_node =
        that action's id.
      - its *target*: the direct object of a transitive active-voice
        verb, the predicate of a linking verb, or the subject of a
        transitive passive-voice verb. related_node = that action's id.
      Either may be absent if the passage doesn't express one (e.g. an
      intransitive verb has no target; an agentless passive has no
      agent).

    Every node's `id` must be the id of an existing token from the input
    `tokens` list (the principal verb's id, for a compound action) --
    never a new id not present in `tokens`. Every node's `context` must
    match the context shared by every token in `tokens`.

    Worked examples from notes/dutch.md:
      - "Alle kunsten en wetenschappen bezitten een gemeenschappelijke
        band." -> a single, non-compound, independent action: node
        id=<id of "bezitten">, value="bezitten", related_node=None.
      - "Hij heeft Cicero nooit gelezen." (compound, active voice) ->
        action node id=<id of "gelezen">, value="heeft gelezen" (the
        adverb "nooit" is excluded), related_node=None; agent node on
        "Hij" (the subject), related_node=<the action's id>; target
        node on "Cicero" (the direct object), related_node=<the
        action's id>.
      - "De Engelse tekst werd vertaald door Jones." (compound, passive
        voice) -> action node id=<id of "vertaald">, value="werd
        vertaald"; agent node on "Jones" (the "door"-phrase agent),
        related_node=<the action's id>; target node on "tekst" (the
        passive-voice subject), related_node=<the action's id>.
      - "Alle kunsten en wetenschappen die tot de menselijke beschaving
        bijdragen, bezitten een gemeenschappelijke band." (one
        independent clause, one dependent clause, subordinated by the
        relative pronoun "die") -> action node on "bezitten"
        (independent, related_node=None; target node on "band", the
        direct object); action node on "bijdragen" (dependent,
        related_node=<"bezitten"'s own id>; agent node on "die", the
        relative pronoun itself, since it stands in for the subject of
        "bijdragen").
      - "En de gehele wereld kwam naar Egypte om bij Jozef koren te
        kopen, want de honger was sterk op de gehele aarde." (one
        independent clause, TWO dependent clauses, subordinated two
        different ways) -> action node on "kwam" (independent,
        related_node=None; agent node on "wereld"); action node on
        "kopen" (dependent, related_node=<"kwam"'s own id>, subordinated
        by the purpose construction "om ... te ..."; target node on
        "koren", the direct object -- no agent, since this clause's own
        subject is never overtly expressed as a separate token); action
        node on "was" (dependent, related_node=<"kwam"'s own id> also --
        both dependent clauses attach to the same single independent
        action -- subordinated by the conjunction "want"; this is a
        linking verb, so agent node on "honger" (its subject) and target
        node on "sterk" (its predicate complement), not a direct
        object).
    """

    passage: str = dspy.InputField(description="The Dutch passage's raw text.")
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
    callers working only with aat.dutch don't need their own separate
    import from aat.core just for this."""
    return _validate_graph(tokens, AATGraph(nodes=list(prediction.nodes)))


if __name__ == "__main__":
    from aat.core import CitedPassage

    from .tokenize import tokenize

    passage = CitedPassage(context="smoketest", text="Hij heeft Cicero nooit gelezen.")
    tokens = tokenize(passage)
    result = analyze(passage=passage.text, tokens=tokens)
    print(result.nodes)
    for problem in validate(tokens, result):
        print("problem:", problem)
