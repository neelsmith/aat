"""
aat: a reductive Agent-Action-Target (AAT) model of natural-language
syntax (see aat-model.md), plus a DSPy-based pipeline applying it to
English (aat.english).

This top-level module re-exports only aat.core -- the language-agnostic
model itself -- so `import aat` never requires dspy to be installed.
English-specific application code lives in aat.english and is imported
separately:

    from aat import CitableToken, AATNode, AATGraph, graph_to_mermaid   # always available
    from aat.english import analyze_passage                              # needs the 'english' extra

See README.md for the reasoning behind this split, and USAGE.md for
worked examples of both.
"""

from .core import (
    CitableToken,
    CitedPassage,
    AATGraph,
    AATNode,
    Role,
    ROLES,
    validate,
    serialize_nodes,
    write_nodes,
    read_nodes,
    read_graph,
    graph_to_mermaid,
    save_mermaid,
    assign_action_colors,
)

__version__ = "0.1.0"

__all__ = [
    "CitableToken",
    "CitedPassage",
    "AATGraph",
    "AATNode",
    "Role",
    "ROLES",
    "validate",
    "serialize_nodes",
    "write_nodes",
    "read_nodes",
    "read_graph",
    "graph_to_mermaid",
    "save_mermaid",
    "assign_action_colors",
    "__version__",
]
