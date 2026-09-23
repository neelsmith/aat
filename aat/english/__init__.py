"""
aat.english: applies the Agent-Action-Target model (aat.core, aat-model.md)
to English text, using a DSPy program to do the actual semantic
extraction.

Needs the 'english' extra (dspy) -- `pip install aat[english]`, or (from
GitHub) `pip install "aat[english] @ git+https://github.com/neelsmith/aat.git"`.
Nothing in aat.core imports this module or requires dspy; the dependency
runs one way only, so a downstream project can depend on aat.core alone.
"""

from .dspy_signatures import AgentActionTarget, analyze, validate
from .gepa_metric import aat_metric
from ..core.html import tokens_to_html
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
    "aat_metric",
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
