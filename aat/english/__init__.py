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
from .pipeline import analyze_passage, analyze_passages
from .tokenize import tokenize

__all__ = [
    "tokenize",
    "AgentActionTarget",
    "analyze",
    "validate",
    "analyze_passage",
    "analyze_passages",
    "aat_metric",
]
