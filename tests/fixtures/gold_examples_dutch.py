"""
Gold-annotated example sentences for aat.dutch's test suite, drawn from
notes/dutch.md's own worked examples -- the Dutch counterpart to
tests/fixtures/gold_examples.py (that file's own docstring explains the
GoldExample shape and how it's used; this one only notes what's different
for Dutch).

Two of the four examples below (_KUNSTEN_WETENSCHAPPEN_BEZITTEN and
_WERELD_KWAM_NAAR_EGYPTE) include a role dutch.md doesn't literally work
through in prose -- see each example's own comment for exactly which
token and why it's a safe, low-risk inference (a single, unambiguous head
noun) rather than a guess. One role is deliberately left OUT for the same
reason in the other direction: _KUNSTEN_WETENSCHAPPEN_BEZITTEN has no
agent for "bezitten" at all, because its real subject is a coordinated
noun phrase ("kunsten en wetenschappen", arts AND sciences) with no
single clear head token, and dutch.md gives no convention for picking one
half of a coordination over the other. Flagged in that example's own
comment, not silently resolved either way.

Add new examples here, not in the test files -- test_dutch_gold_examples.py
and test_dutch_coverage.py both read GOLD_EXAMPLES_DUTCH rather than
defining their own fixtures.
"""

from dataclasses import dataclass
from typing import Any, List

from aat.core import CitableToken, CitedPassage
from aat.dutch import tokenize


@dataclass
class GoldExample:
    slug: str
    passage: str
    context: str
    tags: List[str]
    canned_answer: dict[str, Any]

    def tokens(self) -> List[CitableToken]:
        """The CitableToken list aat.dutch.tokenize() derives from this
        example's own passage/context -- the same tokens a real pipeline
        call would produce."""
        return tokenize(CitedPassage(context=self.context, text=self.passage))

    @property
    def canned_nodes(self) -> List[dict]:
        """Just the `nodes` list out of canned_answer -- mirrors
        gold_examples.py's own property, for the same reason (a future
        Dutch GEPA trainset)."""
        return self.canned_answer["nodes"]


# ---------------------------------------------------------------------------
# "Hij heeft Cicero nooit gelezen." ("He has never read Cicero.")
#   t1 Hij  t2 heeft  t3 Cicero  t4 nooit  t5 gelezen  t6 .
#
# notes/dutch.md's own compound-verb, active-voice worked example: the
# action is anchored on the principal verb "gelezen" (t5), value "heeft
# gelezen" -- the adverb "nooit" interrupts the auxiliary chain but is NOT
# a component token, same judgment call aat.english's own
# _HOMEWORK_NOT_YET_EATEN makes for English. "Hij" is the agent, "Cicero"
# the target.
# ---------------------------------------------------------------------------

_HIJ_HEEFT_CICERO_GELEZEN = GoldExample(
    slug="hij-heeft-cicero-gelezen",
    passage="Hij heeft Cicero nooit gelezen.",
    context="gold.nl.1",
    tags=["active-voice", "compound-action", "independent-action"],
    canned_answer={
        "reasoning": (
            "'heeft gelezen' is a compound action anchored on its principal "
            "verb 'gelezen'; the adverb 'nooit' interrupts the auxiliary "
            "chain but isn't a component token. 'Hij' is the subject/agent, "
            "'Cicero' the direct object/target."
        ),
        "nodes": [
            {"context": "gold.nl.1", "id": "t5", "value": "heeft gelezen", "role": "action", "related_node": None},
            {"context": "gold.nl.1", "id": "t1", "value": "Hij", "role": "agent", "related_node": "t5"},
            {"context": "gold.nl.1", "id": "t3", "value": "Cicero", "role": "target", "related_node": "t5"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "Alle kunsten en wetenschappen die tot de menselijke beschaving
#  bijdragen, bezitten een gemeenschappelijke band."
# ("All arts and sciences that contribute to human civilization possess a
#  common bond.")
#   t1 Alle  t2 kunsten  t3 en  t4 wetenschappen  t5 die  t6 tot  t7 de
#   t8 menselijke  t9 beschaving  t10 bijdragen  t11 ,  t12 bezitten
#   t13 een  t14 gemeenschappelijke  t15 band  t16 .
#
# notes/dutch.md's own worked example for independent vs. dependent
# actions connected by a relative pronoun: "bezitten" (t12) is
# independent (related_node=None); "bijdragen" (t10) is dependent,
# related_node=<"bezitten"'s id>, subordinated by the relative pronoun
# "die" (t5).
#
# Two role judgment calls beyond what dutch.md spells out role-by-role
# for this specific sentence (it only works through the action/
# related_node structure here):
#   - target of "bezitten" = "band" (t15), the clear single head noun of
#     its direct object "een gemeenschappelijke band" ("a common bond") --
#     no coordination or ambiguity, same kind of determiner-stripping
#     already used throughout (English's own "the dog"/"the homework").
#   - agent of "bijdragen" = "die" (t5) itself: the relative pronoun IS
#     the (single-token) subject of "bijdragen", and dutch.md explicitly
#     names it as the connector between the two clauses, so this is
#     directly supported, not inferred.
#   - deliberately NO agent for "bezitten": its real subject is the
#     coordinated NP "kunsten en wetenschappen" (t2 "kunsten" + t4
#     "wetenschappen"), with no single clear head token to anchor on --
#     dutch.md gives no rule for a coordinated subject, and picking one
#     noun over the other would be a real guess, not a safe inference.
#     Left out rather than resolved arbitrarily; see notes/DEVELOPMENT.md's
#     own "flag the judgment call, don't just quietly pick one" precedent
#     (written for aat.english, same principle here).
#   - deliberately NO target for "bijdragen": "tot de menselijke
#     beschaving" ("to human civilization") is a prepositional phrase,
#     not a direct object, so under this scheme it isn't a target.
# ---------------------------------------------------------------------------

_KUNSTEN_WETENSCHAPPEN_BEZITTEN = GoldExample(
    slug="kunsten-wetenschappen-bezitten",
    passage=(
        "Alle kunsten en wetenschappen die tot de menselijke beschaving "
        "bijdragen, bezitten een gemeenschappelijke band."
    ),
    context="gold.nl.2",
    tags=[
        "active-voice",
        "simple-action",
        "independent-action",
        "dependent-action",
        "relative-clause-subordination",
        "no-agent",
        "no-target",
    ],
    canned_answer={
        "reasoning": (
            "'bezitten' is the independent main verb (no single-token "
            "subject to anchor an agent on, since its own subject is the "
            "coordinated NP 'kunsten en wetenschappen'); 'band' is its "
            "target. 'bijdragen' is dependent on 'bezitten', subordinated "
            "by the relative pronoun 'die', which is itself 'bijdragen's "
            "agent; 'tot de menselijke beschaving' is a prepositional "
            "phrase, not a direct object, so 'bijdragen' has no target."
        ),
        "nodes": [
            {"context": "gold.nl.2", "id": "t12", "value": "bezitten", "role": "action", "related_node": None},
            {"context": "gold.nl.2", "id": "t15", "value": "band", "role": "target", "related_node": "t12"},
            {"context": "gold.nl.2", "id": "t10", "value": "bijdragen", "role": "action", "related_node": "t12"},
            {"context": "gold.nl.2", "id": "t5", "value": "die", "role": "agent", "related_node": "t10"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "De Engelse tekst werd vertaald door Jones."
# ("The English text was translated by Jones.")
#   t1 De  t2 Engelse  t3 tekst  t4 werd  t5 vertaald  t6 door  t7 Jones
#   t8 .
#
# notes/dutch.md's own worked example for passive voice, fully spelled
# out role-by-role: the action is anchored on "vertaald" (t5, the
# principal verb of the compound "werd vertaald"); "Jones" (t7, the
# "door"-phrase) is the agent; "tekst" (t3, the passive-voice subject) is
# the target.
# ---------------------------------------------------------------------------

_TEKST_WERD_VERTAALD = GoldExample(
    slug="tekst-werd-vertaald-door-jones",
    passage="De Engelse tekst werd vertaald door Jones.",
    context="gold.nl.3",
    tags=["passive-voice", "compound-action", "independent-action"],
    canned_answer={
        "reasoning": (
            "Passive voice: 'werd vertaald' is a compound action anchored "
            "on its principal verb 'vertaald'; 'Jones' (the door-phrase) is "
            "the agent; 'tekst' (the passive subject) is the target."
        ),
        "nodes": [
            {"context": "gold.nl.3", "id": "t5", "value": "werd vertaald", "role": "action", "related_node": None},
            {"context": "gold.nl.3", "id": "t7", "value": "Jones", "role": "agent", "related_node": "t5"},
            {"context": "gold.nl.3", "id": "t3", "value": "tekst", "role": "target", "related_node": "t5"},
        ],
    },
)

# ---------------------------------------------------------------------------
# "En de gehele wereld kwam naar Egypte om bij Jozef koren te kopen, want
#  de honger was sterk op de gehele aarde."
# ("And all the world came to Egypt to buy grain from Joseph, because the
#  famine was severe over all the earth." -- Genesis 41:57, the same
#  episode as scratch/eng-rv-vpl-genesis.cex's own English text.)
#   t1 En  t2 de  t3 gehele  t4 wereld  t5 kwam  t6 naar  t7 Egypte
#   t8 om  t9 bij  t10 Jozef  t11 koren  t12 te  t13 kopen  t14 ,
#   t15 want  t16 de  t17 honger  t18 was  t19 sterk  t20 op  t21 de
#   t22 gehele  t23 aarde  t24 .
#
# notes/dutch.md's own worked example for a sentence with ONE independent
# clause and TWO dependent clauses, subordinated two different ways:
# "kwam" (t5) is independent; "kopen" (t13) is dependent, subordinated by
# the purpose construction "om ... te ..."; "was" (t18) is dependent,
# subordinated by the conjunction "want". Both dependent actions'
# related_node point at "kwam" (t5) -- the sentence's only independent
# action -- not at each other.
#
# Role assignments beyond the action/related_node structure dutch.md
# spells out by name for this example, each grounded directly in a
# general rule notes/dutch.md DOES state (not invented for this sentence
# specifically):
#   - agent of "kwam" = "wereld" (t4), the single clear head noun of its
#     subject "de gehele wereld" ("the whole world") -- determiner/
#     adjective stripped, same convention as every other example here. No
#     target: "kwam" ("came") is an intransitive motion verb; "naar
#     Egypte" is a locative prepositional phrase, not a direct object.
#   - "kopen" ("to buy"): target = "koren" (t11, "grain"), its direct
#     object. No agent: this purpose clause's own subject is never
#     overtly expressed as a separate token (understood to be the same as
#     "kwam"'s), so there is no token to anchor an agent node on.
#   - "was" ("was"): dutch.md's own Agents/Targets sections explicitly
#     name linking verbs -- "subjects of ... linking verbs" as agents,
#     "predicates of linking verbs" as targets -- so, applying that
#     general rule here: agent = "honger" (t17, "famine", the subject of
#     the linking verb) and target = "sterk" (t19, "severe", its
#     predicate complement).
# ---------------------------------------------------------------------------

_WERELD_KWAM_NAAR_EGYPTE = GoldExample(
    slug="wereld-kwam-naar-egypte",
    passage=(
        "En de gehele wereld kwam naar Egypte om bij Jozef koren te kopen, "
        "want de honger was sterk op de gehele aarde."
    ),
    context="gold.nl.4",
    tags=[
        "active-voice",
        "simple-action",
        "independent-action",
        "dependent-action",
        "purpose-subordination",
        "conjunction-subordination",
        "linking-verb",
        "no-agent",
        "no-target",
    ],
    canned_answer={
        "reasoning": (
            "'kwam' is the independent main verb; 'wereld' is its subject/"
            "agent, no target (intransitive motion, 'naar Egypte' is "
            "locative, not a direct object). 'kopen' is dependent on "
            "'kwam', subordinated by 'om ... te ...'; 'koren' is its "
            "target, no agent (subject not overtly expressed). 'was' is "
            "also dependent on 'kwam', subordinated by the conjunction "
            "'want'; it's a linking verb, so 'honger' is its agent and "
            "'sterk' (the predicate complement) is its target."
        ),
        "nodes": [
            {"context": "gold.nl.4", "id": "t5", "value": "kwam", "role": "action", "related_node": None},
            {"context": "gold.nl.4", "id": "t4", "value": "wereld", "role": "agent", "related_node": "t5"},
            {"context": "gold.nl.4", "id": "t13", "value": "kopen", "role": "action", "related_node": "t5"},
            {"context": "gold.nl.4", "id": "t11", "value": "koren", "role": "target", "related_node": "t13"},
            {"context": "gold.nl.4", "id": "t18", "value": "was", "role": "action", "related_node": "t5"},
            {"context": "gold.nl.4", "id": "t17", "value": "honger", "role": "agent", "related_node": "t18"},
            {"context": "gold.nl.4", "id": "t19", "value": "sterk", "role": "target", "related_node": "t18"},
        ],
    },
)

GOLD_EXAMPLES_DUTCH: List[GoldExample] = [
    _HIJ_HEEFT_CICERO_GELEZEN,
    _KUNSTEN_WETENSCHAPPEN_BEZITTEN,
    _TEKST_WERD_VERTAALD,
    _WERELD_KWAM_NAAR_EGYPTE,
]
