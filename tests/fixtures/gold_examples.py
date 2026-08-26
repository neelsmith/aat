"""
Gold-annotated example sentences for aat's test suite, drawn from
aat-model.md's own worked examples.

Each GoldExample pairs an English passage with a hand-written,
aat-model.md-correct `canned_answer` -- the same dict shape
dspy.utils.dummies.DummyLM expects, and the same shape a dspy.Example's
outputs will eventually take when these feed optimize_gepa.py's trainset.
`tags` names the construction(s) the example is meant to exercise.

Unlike a language whose tokenization is itself ambiguous, aat.english's
tokenize() is fully deterministic (see tokenize.py) -- so a GoldExample
doesn't need to hand-author its own token list at all; `tokens()` just
calls the real tokenizer on `passage`/`context`, guaranteeing the tokens a
test sees always agree with what a real pipeline call would produce.

Add new examples here, not in the test files -- test_gold_examples.py and
test_coverage.py both read GOLD_EXAMPLES rather than defining their own
fixtures.
"""

from dataclasses import dataclass
from typing import Any, List

from aat.core import CitableToken, CitedPassage
from aat.english import tokenize


@dataclass
class GoldExample:
    slug: str
    passage: str
    context: str
    tags: List[str]
    canned_answer: dict[str, Any]

    def tokens(self) -> List[CitableToken]:
        """The CitableToken list aat.english.tokenize() derives from this
        example's own passage/context -- the same tokens a real pipeline
        call would produce."""
        return tokenize(CitedPassage(context=self.context, text=self.passage))

    @property
    def canned_nodes(self) -> List[dict]:
        """Just the `nodes` list out of canned_answer -- e.g. for building
        a GEPA trainset's gold outputs (optimize_gepa.py's
        build_trainset())."""
        return self.canned_answer["nodes"]


# ---------------------------------------------------------------------------
# "The dog ate my homework."
#   t1 The  t2 dog  t3 ate  t4 my  t5 homework  t6 .
#
# aat-model.md's own worked example for a simple active-voice transitive
# verb: "ate" is the action (independent, no related_node), "dog" is its
# agent, "homework" is its target.
# ---------------------------------------------------------------------------

_DOG_ATE_HOMEWORK = GoldExample(
    slug="dog-ate-homework",
    passage="The dog ate my homework.",
    context="gold.1",
    tags=["active-voice", "simple-action", "independent-action"],
    canned_answer={
        "reasoning": (
            "'ate' is a simple, independent active-voice verb: 'dog' is its "
            "subject/agent, 'homework' is its direct object/target."
        ),
        "nodes": [
            {"context": "gold.1", "id": "t3", "value": "ate", "role": "action", "related_node": None},
            {"context": "gold.1", "id": "t2", "value": "dog", "role": "agent", "related_node": "t3"},
            {"context": "gold.1", "id": "t5", "value": "homework", "role": "target", "related_node": "t3"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "The homework was eaten by the dog."
#   t1 The  t2 homework  t3 was  t4 eaten  t5 by  t6 the  t7 dog  t8 .
#
# aat-model.md's own worked example for passive voice: the action node is
# anchored on "eaten" (the principal verb of the compound "was eaten"),
# with the by-phrase "dog" as agent and the passive subject "homework" as
# target.
# ---------------------------------------------------------------------------

_HOMEWORK_WAS_EATEN = GoldExample(
    slug="homework-was-eaten-by-dog",
    passage="The homework was eaten by the dog.",
    context="gold.2",
    tags=["passive-voice", "compound-action", "independent-action"],
    canned_answer={
        "reasoning": (
            "Passive voice: 'was eaten' is a compound action anchored on its "
            "principal verb 'eaten'; 'dog' (the by-phrase) is the agent; "
            "'homework' (the passive subject) is the target."
        ),
        "nodes": [
            {"context": "gold.2", "id": "t4", "value": "was eaten", "role": "action", "related_node": None},
            {"context": "gold.2", "id": "t7", "value": "dog", "role": "agent", "related_node": "t4"},
            {"context": "gold.2", "id": "t2", "value": "homework", "role": "target", "related_node": "t4"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "The homework was not yet being eaten."
#   t1 The  t2 homework  t3 was  t4 not  t5 yet  t6 being  t7 eaten  t8 .
#
# aat-model.md's own worked example for a longer compound verb, with two
# adverbs ("not", "yet") interrupting the auxiliary chain. Judgment call
# (not made explicit in aat-model.md, but implied by its "concatenates the
# text values of all the component tokens" wording -- "component tokens"
# meaning the auxiliary/verb chain itself, not anything that merely sits
# between them): the interrupting adverbs are NOT component tokens of the
# verbal expression -- they're ordinary adverbs modifying it -- so the
# action's value is "was being eaten" (was + being + eaten), not "was not
# yet being eaten". No agent is expressed (no by-phrase); the passive
# subject "homework" is still the target.
# ---------------------------------------------------------------------------

_HOMEWORK_NOT_YET_EATEN = GoldExample(
    slug="homework-not-yet-eaten",
    passage="The homework was not yet being eaten.",
    context="gold.3",
    tags=["passive-voice", "compound-action", "no-agent", "interrupted-auxiliary-chain"],
    canned_answer={
        "reasoning": (
            "Compound passive 'was being eaten', anchored on principal verb "
            "'eaten'; 'not' and 'yet' are adverbs interrupting the auxiliary "
            "chain, not part of the action's own value. No by-phrase, so no "
            "agent; 'homework' (the passive subject) is the target."
        ),
        "nodes": [
            {"context": "gold.3", "id": "t7", "value": "was being eaten", "role": "action", "related_node": None},
            {"context": "gold.3", "id": "t2", "value": "homework", "role": "target", "related_node": "t7"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "He said that the dog ate his homework."
#   t1 He  t2 said  t3 that  t4 the  t5 dog  t6 ate  t7 his  t8 homework  t9 .
#
# aat-model.md's own worked example for a dependent (subordinate) action:
# "said" is independent (related_node=None); "ate" is dependent, with
# related_node pointing at "said"'s own node id (t2). "said" takes a
# clausal complement, not a direct-object noun phrase, so it has no target
# under this scheme; "ate" has its own agent ("dog") and target
# ("homework").
# ---------------------------------------------------------------------------

_HE_SAID_THAT = GoldExample(
    slug="he-said-that-dog-ate",
    passage="He said that the dog ate his homework.",
    context="gold.4",
    tags=["dependent-action", "independent-action", "active-voice", "no-target"],
    canned_answer={
        "reasoning": (
            "'said' is the independent main verb, with 'He' as its agent and "
            "no target (it takes a clausal complement, not a direct object). "
            "'ate' is a dependent action inside that complement clause, "
            "related to 'said' (t2); 'dog' is ate's agent, 'homework' is "
            "ate's target."
        ),
        "nodes": [
            {"context": "gold.4", "id": "t2", "value": "said", "role": "action", "related_node": None},
            {"context": "gold.4", "id": "t1", "value": "He", "role": "agent", "related_node": "t2"},
            {"context": "gold.4", "id": "t6", "value": "ate", "role": "action", "related_node": "t2"},
            {"context": "gold.4", "id": "t5", "value": "dog", "role": "agent", "related_node": "t6"},
            {"context": "gold.4", "id": "t8", "value": "homework", "role": "target", "related_node": "t6"},
        ],
    },
)

GOLD_EXAMPLES: List[GoldExample] = [
    _DOG_ATE_HOMEWORK,
    _HOMEWORK_WAS_EATEN,
    _HOMEWORK_NOT_YET_EATEN,
    _HE_SAID_THAT,
]
