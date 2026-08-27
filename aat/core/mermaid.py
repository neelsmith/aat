"""
Render an AATGraph as a Mermaid flowchart.

- Every node becomes a Mermaid node, labelled with its own `value` and
  shaped by its `role`: an action is a plain rectangle (`[...]`), an agent
  is rounded (`(...)`), and a target is a stadium shape (`([...])`) -- so
  role is visible from shape alone, before reading any label.
- Every node with a `related_node` becomes a labelled edge FROM that node
  TO the node it relates to (mirroring AATNode.related_node's own
  direction -- see graph.py): an agent or target points at its action,
  labelled with its own role ("agent"/"target"); a dependent action points
  at its governing action, labelled "dependent". An independent action
  (related_node=None) gets no outgoing edge of its own.
- By default (`color_by_action=True`), every node is also colored by which
  action it clusters with: an action node and every agent/target node
  whose related_node points at it share one color, assigned from a small
  palette in the order each action first appears in `graph.nodes` -- so
  the separate clauses in a multi-action passage (e.g. an independent verb
  and a dependent clause it governs) are visually distinguishable at a
  glance. A dependent action gets its OWN color, not its governor's -- it's
  still its own cluster's anchor; the "dependent" edge just also points at
  another cluster's anchor.

Multiple contexts in one `graph` are all drawn into a single diagram, with
no special separation between them -- if you want one diagram per context,
filter `graph.nodes` first (e.g. `AATGraph(nodes=[n for n in graph.nodes
if n.context == wanted])`).

`orientation` (default "BT", bottom-to-top) is validated against Mermaid's
own set of flowchart direction codes -- see `_VALID_ORIENTATIONS` and
`graph_to_mermaid()`'s own docstring.

The action-cluster color assignment itself lives in aat.core.coloring
(assign_action_colors()), not here, so it can be reused wherever else a
node needs to be colored the same way -- see that module's own docstring.
"""

from typing import Dict, List, Tuple

from .coloring import ColorTriple, assign_action_colors
from .graph import AATGraph, AATNode

# Mermaid's own flowchart direction codes (https://mermaid.js.org/syntax/
# flowchart.html#direction): TB and TD are synonyms (top-down); BT, RL, LR
# are the other three directions. graph_to_mermaid() checks `orientation`
# against this set (case-insensitively) rather than passing it through
# unchecked, so a typo becomes a clear ValueError here instead of silently
# invalid Mermaid syntax in the output.
_VALID_ORIENTATIONS = {"TB", "TD", "BT", "RL", "LR"}


def _normalize_orientation(orientation: str) -> str:
    normalized = orientation.strip().upper()
    if normalized not in _VALID_ORIENTATIONS:
        raise ValueError(
            f"invalid orientation {orientation!r} -- must be one of "
            f"{sorted(_VALID_ORIENTATIONS)} (Mermaid's flowchart direction "
            "codes: https://mermaid.js.org/syntax/flowchart.html#direction)"
        )
    return normalized


# Characters that need escaping inside a Mermaid quoted label.
_LABEL_ESCAPES = {
    '"': "&quot;",
    "<": "&lt;",
    ">": "&gt;",
}


def _escape_label(text: str) -> str:
    for char, replacement in _LABEL_ESCAPES.items():
        text = text.replace(char, replacement)
    return text


# Mermaid node-shape delimiters, keyed by AATNode.role. A role this module
# doesn't recognize (shouldn't happen -- Role is a Literal of exactly
# these three -- but AATNode itself doesn't enforce that at the type level
# for a hand-built or deserialized node) falls back to a plain rectangle.
_ROLE_BRACKETS = {
    "action": ("[", "]"),
    "agent": ("(", ")"),
    "target": ("([", "])"),
}


def _node_key(node: AATNode) -> Tuple[str, str]:
    return (node.context, node.id)


def graph_to_mermaid(
    graph: AATGraph,
    orientation: str = "BT",
    color_by_action: bool = True,
) -> Tuple[str, List[str]]:
    """Build a Mermaid `graph` diagram from an AATGraph.

    `orientation` is Mermaid's own flowchart direction code -- `BT`
    (bottom-to-top, the default here), `TB`/`TD` (top-down -- synonyms),
    `LR`, or `RL` -- used verbatim (uppercased) in the diagram's opening
    line (`graph BT`, etc.). Matched case-insensitively against
    `_VALID_ORIENTATIONS`; anything else raises `ValueError` naming the
    valid options, rather than silently producing invalid Mermaid syntax.
    See https://mermaid.js.org/syntax/flowchart.html#direction.

    `color_by_action` (default True) -- see this module's own docstring.
    Pass False for a plain, uncolored diagram.

    Returns (diagram_text, warnings). A node whose `related_node` doesn't
    resolve to any node actually present in `graph` (same context) is
    still drawn, but its edge is skipped and reported as a warning --
    normally a sign `graph` failed aat.core.validate.validate() upstream,
    worth checking there first. If `color_by_action` is True and the graph
    has more distinct actions than the palette has colors (currently 8),
    one warning notes that colors repeat.
    """
    orientation = _normalize_orientation(orientation)

    by_key: Dict[Tuple[str, str], AATNode] = {_node_key(n): n for n in graph.nodes}

    lines = [f"graph {orientation}"]
    for node in graph.nodes:
        open_b, close_b = _ROLE_BRACKETS.get(node.role, ("[", "]"))
        lines.append(f'    {node.id}{open_b}"{_escape_label(node.value)}"{close_b}')

    warnings: List[str] = []
    for node in graph.nodes:
        if node.related_node is None:
            continue
        target_key = (node.context, node.related_node)
        if target_key not in by_key:
            warnings.append(
                f"skipped edge {node.id} -[{node.role}]-> {node.related_node}: "
                "target is not a node in this graph"
            )
            continue
        edge_label = "dependent" if node.role == "action" else node.role
        lines.append(f"    {node.id} -->|{edge_label}| {node.related_node}")

    if color_by_action:
        color_of_node, color_warnings = assign_action_colors(graph)
        warnings.extend(color_warnings)

        if color_of_node:
            lines.append("")

            # Group by color VALUE, not by action -- two actions that
            # cycle to the same palette color (more actions than the
            # palette has slots) share a single classDef/class pair here
            # rather than emitting two redundant classes with identical
            # colors.
            colors_in_order: List[ColorTriple] = []
            seen_colors = set()
            for node in graph.nodes:
                color = color_of_node.get(_node_key(node))
                if color is not None and color not in seen_colors:
                    seen_colors.add(color)
                    colors_in_order.append(color)

            class_name_of_color: Dict[ColorTriple, str] = {
                color: f"c{i}" for i, color in enumerate(colors_in_order)
            }
            for color in colors_in_order:
                fill, stroke, text = color
                lines.append(
                    f"    classDef {class_name_of_color[color]} "
                    f"fill:{fill},stroke:{stroke},color:{text};"
                )

            nodes_by_class: Dict[str, List[str]] = {}
            for node in graph.nodes:
                color = color_of_node.get(_node_key(node))
                if color is None:
                    continue
                class_name = class_name_of_color[color]
                nodes_by_class.setdefault(class_name, []).append(node.id)

            for class_name, node_ids in nodes_by_class.items():
                lines.append(f"    class {','.join(node_ids)} {class_name};")

    return "\n".join(lines), warnings


def save_mermaid(
    graph: AATGraph,
    path: str,
    orientation: str = "BT",
    color_by_action: bool = True,
) -> List[str]:
    """Write the diagram to `path` (e.g. 'analysis.mmd') and return any
    warnings from graph_to_mermaid(). `orientation` is validated the same
    way -- see graph_to_mermaid()'s own docstring."""
    diagram, warnings = graph_to_mermaid(graph, orientation=orientation, color_by_action=color_by_action)
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram + "\n")
    return warnings
