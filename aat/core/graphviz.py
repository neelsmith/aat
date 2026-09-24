"""
Render an AATGraph as a Graphviz DOT digraph -- the same graph
aat.core.mermaid.graph_to_mermaid() renders as a Mermaid flowchart,
translated into Graphviz's own syntax instead. The underlying node/edge/
coloring model is identical (see that module's own docstring); only the
target syntax differs:

- Every node becomes a DOT node, labelled with its own `value` and
  shaped by its `role`. Graphviz has no shape literally called
  "stadium", so target's shape is chosen as the closest visual analogue
  to Mermaid's `([...])` (fully rounded, no square corners at all):
    - action: `shape=box` (plain rectangle, matches Mermaid's `[...]`)
    - agent: `shape=box, style=rounded` (rounded-corner rectangle,
      matches Mermaid's `(...)`)
    - target: `shape=ellipse` (fully rounded oval, matches Mermaid's
      `([...])`)
- Every node with a `related_node` becomes a labelled edge FROM that node
  TO the node it relates to, exactly as graph_to_mermaid() describes.
- By default (`color_by_action=True`), every node is colored by the same
  action-cluster assignment graph_to_mermaid() uses
  (aat.core.coloring.assign_action_colors()) -- but applied as inline
  `style=filled, fillcolor=..., color=..., fontcolor=...` attributes
  directly on each node's own line. DOT has no equivalent to Mermaid's
  separate classDef/class mechanism, so there's no separate "class"
  grouping step here; each node just carries its own color attributes.
- By default (`rooted=True`), every *independent* action (an action node
  whose own `related_node` is `None` -- see aat-model.md) gets an extra
  edge to a single synthetic `root` node, shared across the whole
  digraph (not one per context), so a multi-sentence/multi-context graph
  reads as one connected tree rather than several disconnected ones, one
  per independent action. `root` is drawn in Graphviz's own default node
  form -- no `shape=`, no color/fill attributes at all -- deliberately,
  so it reads as scaffolding rather than a token the source text
  actually contains, standing apart from every other node here (all of
  which always carry an explicit `shape=` and, when `color_by_action` is
  on, fill/border/font colors too). `graph_to_mermaid()` has no
  equivalent of this -- `rooted` is DOT-only, not part of the shared
  node/edge/coloring model the rest of this docstring describes.

Multiple contexts in one `graph` are all drawn into a single digraph,
with no special separation between them -- same caveat as
graph_to_mermaid(); filter `graph.nodes` first for one digraph per
context.

Node/edge ids are written as bare DOT identifiers when that's valid
(e.g. "t3"), or double-quoted (see _dot_id()) when it isn't -- notably a
sentence-spanning composite id such as "1.14.t3"
(aat.english.sentences.tokenize_units(), aat.corpus) starts with a digit
and contains '.', which Graphviz's own lexer otherwise misreads as a
malformed number literal.

`orientation` (default "BT", bottom-to-top) is validated the same way
graph_to_mermaid() validates it (aat.core.orientation, shared between
both renderers) -- but Graphviz's `rankdir` graph attribute has no "TD"
synonym of its own, so an orientation of "TD" (top-down) is mapped to
Graphviz's "TB" (top-to-bottom -- the same direction, just Graphviz's own
spelling of it) when it's written out; see graph_to_dot()'s own
docstring.
"""

import re
from typing import Dict, List, Tuple

from .coloring import ColorTriple, assign_action_colors
from .graph import AATGraph, AATNode
from .orientation import normalize_orientation

# Graphviz node shape/style, keyed by AATNode.role -- see this module's
# own docstring for why target maps to shape=ellipse rather than a
# "stadium" Graphviz has no name for. A role this module doesn't
# recognize (shouldn't happen -- Role is a Literal of exactly these
# three -- but AATNode itself doesn't enforce that at the type level for
# a hand-built or deserialized node) falls back to a plain box, same
# fallback graph_to_mermaid() uses for an unrecognized role.
_ROLE_SHAPE = {
    "action": "box",
    "agent": "box",
    "target": "ellipse",
}

# Extra `style=` keywords layered on top of a node's own shape, keyed the
# same way -- only agent needs one of its own (rounded corners); a
# color_by_action fill (see graph_to_dot()) adds "filled" to whatever's
# here rather than overwriting it, so a colored agent node still ends up
# rounded (style="rounded,filled"), not a plain filled rectangle.
_ROLE_EXTRA_STYLES: Dict[str, List[str]] = {
    "agent": ["rounded"],
}


def _node_key(node: AATNode) -> Tuple[str, str]:
    return (node.context, node.id)


def _escape_label(text: str) -> str:
    """Escape `text` for use inside a DOT double-quoted string literal --
    backslash first (so escaping the quote below doesn't get re-escaped
    itself), then the quote character."""
    return text.replace("\\", "\\\\").replace('"', '\\"')


_BARE_DOT_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _dot_id(node_id: str) -> str:
    """Render `node_id` (an AATNode/CitableToken id -- e.g. "t3", or a
    sentence-spanning composite id like "1.14.t3", see
    aat.english.sentences's own module docstring) as a DOT identifier.

    A plain id like "t3" is already a valid bare DOT ID (letter/
    underscore, then letters/digits/underscores) and is returned as-is,
    matching this module's own pre-existing output exactly for the
    common single-citation-unit case. A composite id such as "1.14.t3"
    is NOT a valid bare DOT ID -- it starts with a digit and contains
    '.', which Graphviz's own lexer tries to parse as part of a number
    literal and then rejects ("badly delimited number") -- so it's
    quoted instead (same escaping _escape_label() uses for a label),
    which DOT accepts for any identifier at all."""
    if _BARE_DOT_ID.match(node_id):
        return node_id
    return f'"{_escape_label(node_id)}"'


def graph_to_dot(
    graph: AATGraph,
    orientation: str = "BT",
    color_by_action: bool = True,
    rooted: bool = True,
) -> Tuple[str, List[str]]:
    """Build a Graphviz DOT digraph from an AATGraph -- see this module's
    own docstring for the node-shape/edge/coloring mapping, which mirrors
    aat.core.mermaid.graph_to_mermaid() exactly except for the target
    syntax.

    `orientation` is validated exactly like graph_to_mermaid()'s (see
    aat.core.orientation) -- `BT` (bottom-to-top, the default here),
    `TB`/`TD` (top-down -- synonyms), `LR`, or `RL`. Written out as
    Graphviz's own `rankdir` graph attribute; since `rankdir` has no
    "TD" of its own, "TD" is mapped to "TB" here (same direction, just
    Graphviz's own name for it) -- every other value is used verbatim.
    Anything outside that vocabulary raises `ValueError` naming the
    valid options, rather than silently producing invalid DOT syntax.

    `color_by_action` (default True) -- see this module's own docstring.
    Pass False for a plain, uncolored digraph.

    `rooted` (default True) -- see this module's own docstring. When
    True and the graph has at least one independent action (role
    "action", related_node None), a single `root` node (Graphviz's own
    default plain-oval shape, no color) is added, with an edge from
    every independent action to it. Pass False to skip this -- an
    independent action then simply has no outgoing edge, same as this
    function's behavior before `rooted` existed.

    Returns (dot_text, warnings) -- same warning cases as
    graph_to_mermaid(): a node whose `related_node` doesn't resolve to
    another node actually present in `graph` (same context) is still
    drawn, but its edge is skipped and reported as a warning; and, if
    `color_by_action` is True and the graph has more distinct actions
    than the palette has colors (currently 8), one warning notes that
    colors repeat.
    """
    orientation = normalize_orientation(orientation)
    rankdir = "TB" if orientation == "TD" else orientation

    by_key: Dict[Tuple[str, str], AATNode] = {_node_key(n): n for n in graph.nodes}

    color_of_node: Dict[Tuple[str, str], ColorTriple] = {}
    warnings: List[str] = []
    if color_by_action:
        color_of_node, color_warnings = assign_action_colors(graph)
        warnings.extend(color_warnings)

    lines = ["digraph aat {", f"    rankdir={rankdir};"]

    has_independent_action = any(
        node.role == "action" and node.related_node is None for node in graph.nodes
    )
    if rooted and has_independent_action:
        lines.append('    root [label="root"];')

    for node in graph.nodes:
        attrs = [f"shape={_ROLE_SHAPE.get(node.role, 'box')}"]
        styles = list(_ROLE_EXTRA_STYLES.get(node.role, []))

        color = color_of_node.get(_node_key(node))
        if color is not None:
            fill, stroke, text = color
            styles.append("filled")
            attrs.append(f'fillcolor="{fill}"')
            attrs.append(f'color="{stroke}"')
            attrs.append(f'fontcolor="{text}"')

        if styles:
            attrs.append(f'style="{",".join(styles)}"')
        attrs.append(f'label="{_escape_label(node.value)}"')

        lines.append(f'    {_dot_id(node.id)} [{", ".join(attrs)}];')

    for node in graph.nodes:
        if node.related_node is None:
            if rooted and node.role == "action":
                lines.append(f'    {_dot_id(node.id)} -> root [label="root"];')
            continue
        target_key = (node.context, node.related_node)
        if target_key not in by_key:
            warnings.append(
                f"skipped edge {node.id} -[{node.role}]-> {node.related_node}: "
                "target is not a node in this graph"
            )
            continue
        edge_label = "dependent" if node.role == "action" else node.role
        lines.append(f'    {_dot_id(node.id)} -> {_dot_id(node.related_node)} [label="{edge_label}"];')

    lines.append("}")
    return "\n".join(lines), warnings


def save_dot(
    graph: AATGraph,
    path: str,
    orientation: str = "BT",
    color_by_action: bool = True,
    rooted: bool = True,
) -> List[str]:
    """Write the digraph to `path` (e.g. 'analysis.dot') and return any
    warnings from graph_to_dot(). `orientation`, `color_by_action`, and
    `rooted` are all passed straight through -- see graph_to_dot()'s own
    docstring for each."""
    dot_text, warnings = graph_to_dot(
        graph, orientation=orientation, color_by_action=color_by_action, rooted=rooted
    )
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot_text + "\n")
    return warnings
