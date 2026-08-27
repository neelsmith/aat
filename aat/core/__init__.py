"""
aat.core: the Agent-Action-Target model itself (aat-model.md), with no
dependency on any particular language or on dspy.

Everything in this subpackage represents, validates, serializes, or
renders an AAT graph; nothing here knows how to *produce* one from real
text -- that's aat.english's job (or, eventually, another language-
specific sibling subpackage). Keeping this boundary means aat.core can be
imported and reused on its own by any downstream project that just wants
the data model, the file format, and the Mermaid renderer, without
pulling in dspy at all.
"""

from .tokens import CitableToken, CitedPassage
from .graph import AATGraph, AATNode, Role, ROLES
from .validate import validate
from .serialization import (
    serialize_nodes,
    write_nodes,
    read_nodes,
    read_graph,
    serialize_passages,
    write_passages,
    read_passages,
    serialize_analysis,
    write_analysis,
    read_analysis,
)
from .mermaid import graph_to_mermaid, save_mermaid
from .coloring import assign_action_colors

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
    "serialize_passages",
    "write_passages",
    "read_passages",
    "serialize_analysis",
    "write_analysis",
    "read_analysis",
    "graph_to_mermaid",
    "save_mermaid",
    "assign_action_colors",
]
