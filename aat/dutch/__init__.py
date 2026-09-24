"""
aat.dutch: applies the Agent-Action-Target model (aat.core, aat-model.md,
notes/dutch.md) to Dutch text, using a DSPy program to do the actual
semantic extraction. Parallel to aat.english -- same model, same
AATGraph/AATNode shape, same tokenize -> analyze -> validate pipeline
structure, a different language's tokenizer and dspy.Signature.

Needs the 'dutch' extra (dspy) -- `pip install aatgraph[dutch]`, or (from
GitHub) `pip install "aatgraph[dutch] @ git+https://github.com/neelsmith/aat.git"`.
Nothing in aat.core imports this module or requires dspy; the dependency
runs one way only, so a downstream project can depend on aat.core alone.

Not yet mirrored from aat.english, deliberately left for later: a
GEPA optimization metric (aat.english.gepa_metric) and a
utilities/calibrate_max_tokens.py counterpart, both of which need a
curated Dutch gold-examples fixture (aat.english's own
tests/fixtures/gold_examples.py) that doesn't exist yet; a CLI script
analogous to aat_main.py; and marimo notebook equivalents of
marimo/aat_graph.py etc. See notes/CLAUDE_WORKFLOW.md's session log for
this module's own addition.
"""

from ..core.html import tokens_to_html
from .dspy_signatures import AgentActionTarget, analyze, validate
from .pipeline import analyze_passage, analyze_passages, analyze_units_by_sentence
from .sentences import (
    DEFAULT_SENTENCE_TERMINATORS,
    cluster_sentences,
    ends_sentence,
    passage_component,
    tokenize_corpus_by_sentence,
    tokenize_units,
    urn_prefix,
)
from .token_budget import (
    DEFAULT_CEILING,
    DEFAULT_FLOOR,
    DEFAULT_SAFETY_MARGIN,
    analyze_with_retry,
    estimate_max_tokens,
    get_calibration,
)
from .tokenize import tokenize

__all__ = [
    "tokenize",
    "AgentActionTarget",
    "analyze",
    "validate",
    "analyze_passage",
    "analyze_passages",
    "analyze_units_by_sentence",
    "tokens_to_html",
    "DEFAULT_SENTENCE_TERMINATORS",
    "cluster_sentences",
    "ends_sentence",
    "passage_component",
    "urn_prefix",
    "tokenize_units",
    "tokenize_corpus_by_sentence",
    "analyze_with_retry",
    "estimate_max_tokens",
    "get_calibration",
    "DEFAULT_CEILING",
    "DEFAULT_FLOOR",
    "DEFAULT_SAFETY_MARGIN",
]
