"""
Shared orientation validation for graph_to_mermaid() and graph_to_dot():
both accept the same four-way diagram-direction vocabulary and render it
as their own tool's own top-of-file direction attribute (Mermaid's
`graph BT`, Graphviz's `rankdir=BT;`) -- see each renderer's own
docstring for exactly how. Living in its own module keeps that one
validation rule from being duplicated (and from drifting) across both.
"""

# Mermaid's own flowchart direction codes (https://mermaid.js.org/syntax/
# flowchart.html#direction): TB and TD are synonyms (top-down); BT, RL, LR
# are the other three directions. Graphviz's `rankdir` graph attribute
# recognizes the same four *directions* but has no "TD" synonym of its
# own -- graph_to_dot() maps "TD" to "TB" itself when it emits `rankdir`,
# after normalize_orientation() here has already validated it.
VALID_ORIENTATIONS = frozenset({"TB", "TD", "BT", "RL", "LR"})


def normalize_orientation(orientation: str) -> str:
    """Validate `orientation` against VALID_ORIENTATIONS, case-
    insensitively and with surrounding whitespace stripped, and return
    it stripped and uppercased (e.g. "lr" -> "LR"; "TD" is returned as
    "TD", unchanged -- a caller whose own target syntax has no "TD" of
    its own still has to map it further, see graph_to_dot()).

    Raises ValueError naming the valid options if `orientation` doesn't
    match one of them, rather than letting a typo become silently
    invalid Mermaid or Graphviz syntax downstream.
    """
    normalized = orientation.strip().upper()
    if normalized not in VALID_ORIENTATIONS:
        raise ValueError(
            f"invalid orientation {orientation!r} -- must be one of "
            f"{sorted(VALID_ORIENTATIONS)} (Mermaid's flowchart direction "
            "codes, also used here for Graphviz's rankdir -- see "
            "https://mermaid.js.org/syntax/flowchart.html#direction)"
        )
    return normalized
