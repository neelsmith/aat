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
"""

from typing import Dict, List, Tuple

from .graph import AATGraph, AATNode

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


# (fill, stroke, text) hex triples, chosen for readable contrast between
# fill and text in both light- and dark-themed Mermaid renderers. Cycles
# (with a warning -- see graph_to_mermaid()'s own docstring) if a graph has
# more distinct actions than this palette has slots.
_ACTION_COLOR_PALETTE = [
    ("#E3F2FD", "#1565C0", "#0D47A1"),  # blue
    ("#FFF3E0", "#EF6C00", "#E65100"),  # orange
    ("#E8F5E9", "#2E7D32", "#1B5E20"),  # green
    ("#FCE4EC", "#AD1457", "#880E4F"),  # pink
    ("#EDE7F6", "#5E35B1", "#4527A0"),  # purple
    ("#FFFDE7", "#F9A825", "#F57F17"),  # yellow
    ("#E0F7FA", "#00838F", "#006064"),  # cyan
    ("#FBE9E7", "#D84315", "#BF360C"),  # deep orange
]

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
    orientation: str = "LR",
    color_by_action: bool = True,
) -> Tuple[str, List[str]]:
    """Build a Mermaid `graph` diagram from an AATGraph.

    `orientation` is Mermaid's own flowchart orientation code -- `LR`
    (left-to-right, the default here), `TB`, `BT`, or `RL` -- used
    verbatim in the diagram's opening line (`graph LR`, etc.). See
    https://mermaid.js.org/syntax/flowchart.html for what each value looks
    like; this function doesn't validate it, so a typo just becomes
    invalid Mermaid syntax in the output rather than an error here.

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
        actions = [n for n in graph.nodes if n.role == "action"]
        action_keys_in_order = [_node_key(a) for a in actions]

        if len(actions) > len(_ACTION_COLOR_PALETTE):
            warnings.append(
                f"{len(actions)} actions but only {len(_ACTION_COLOR_PALETTE)} palette "
                "colors -- colors repeat and some distinct actions will share a color"
            )

        # Every node's cluster is the action it belongs to: its own key
        # for an action node (dependent or not -- see module docstring),
        # or the action its related_node points at for an agent/target. A
        # node with no related_node that isn't an action (shouldn't
        # happen in a validate()-clean graph) is left uncolored.
        cluster_of_node: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for node in graph.nodes:
            key = _node_key(node)
            if node.role == "action":
                cluster_of_node[key] = key
            elif node.related_node is not None:
                related_key = (node.context, node.related_node)
                if related_key in by_key:
                    cluster_of_node[key] = related_key

        if action_keys_in_order:
            lines.append("")
            class_name_of_action = {key: f"a{i}" for i, key in enumerate(action_keys_in_order)}
            for i, key in enumerate(action_keys_in_order):
                fill, stroke, text = _ACTION_COLOR_PALETTE[i % len(_ACTION_COLOR_PALETTE)]
                lines.append(f"    classDef a{i} fill:{fill},stroke:{stroke},color:{text};")

            nodes_by_class: Dict[str, List[str]] = {}
            for node in graph.nodes:
                cluster_key = cluster_of_node.get(_node_key(node))
                class_name = class_name_of_action.get(cluster_key)
                if class_name is None:
                    continue
                nodes_by_class.setdefault(class_name, []).append(node.id)

            for class_name, node_ids in nodes_by_class.items():
                lines.append(f"    class {','.join(node_ids)} {class_name};")

    return "\n".join(lines), warnings


def save_mermaid(
    graph: AATGraph,
    path: str,
    orientation: str = "LR",
    color_by_action: bool = True,
) -> List[str]:
    """Write the diagram to `path` (e.g. 'analysis.mmd') and return any
    warnings from graph_to_mermaid()."""
    diagram, warnings = graph_to_mermaid(graph, orientation=orientation, color_by_action=color_by_action)
    with open(path, "w", encoding="utf-8") as f:
        f.write(diagram + "\n")
    return warnings
