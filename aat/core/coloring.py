"""
Shared color-by-action-cluster assignment.

Every action node, plus every agent or target node whose `related_node`
points at it, shares one color -- assigned from a small palette in the
order each action first appears in a graph's own node list. This lives in
its own module, separate from mermaid.py, so it can be reused wherever a
token or node needs to be colored the same way it is in a Mermaid diagram
without depending on Mermaid-specific rendering code -- e.g.
aat.english.html's tokens_to_html(), which highlights passage text with
these same colors. This mirrors why arsgrammatica's verbal_units.py is
shared between its own mermaid.py and rendering.py.

A dependent action gets its OWN color, not its governor's -- it's still
its own cluster's anchor, even though something else (a Mermaid edge, an
HTML border) may also point at or otherwise reference another cluster's
anchor.
"""

from typing import Dict, List, Tuple

from .graph import AATGraph, AATNode

NodeKey = Tuple[str, str]
ColorTriple = Tuple[str, str, str]

# (fill, stroke, text) hex triples, chosen for readable contrast between
# fill and text in both light- and dark-themed renderers. Cycles (with a
# warning -- see assign_action_colors()'s own docstring) if a graph has
# more distinct actions than this palette has slots.
_ACTION_COLOR_PALETTE: List[ColorTriple] = [
    ("#E3F2FD", "#1565C0", "#0D47A1"),  # blue
    ("#FFF3E0", "#EF6C00", "#E65100"),  # orange
    ("#E8F5E9", "#2E7D32", "#1B5E20"),  # green
    ("#FCE4EC", "#AD1457", "#880E4F"),  # pink
    ("#EDE7F6", "#5E35B1", "#4527A0"),  # purple
    ("#FFFDE7", "#F9A825", "#F57F17"),  # yellow
    ("#E0F7FA", "#00838F", "#006064"),  # cyan
    ("#FBE9E7", "#D84315", "#BF360C"),  # deep orange
]


def _node_key(node: AATNode) -> NodeKey:
    return (node.context, node.id)


def assign_action_colors(graph: AATGraph) -> Tuple[Dict[NodeKey, ColorTriple], List[str]]:
    """Return (color_of_node, warnings).

    `color_of_node` maps every node's (context, id) to a (fill, stroke,
    text) hex triple, for every node that clusters with some action: an
    action node itself (dependent or not -- see this module's own
    docstring), or an agent/target node whose `related_node` resolves to
    an action node in the same context. A node with no resolvable cluster
    (e.g. an agent/target with a broken related_node -- shouldn't happen
    in a graph that's passed aat.core.validate.validate()) is simply
    absent from the mapping.

    Colors are assigned to actions in the order they first appear in
    `graph.nodes`, cycling through the palette (currently 8 colors) if
    there are more distinct actions than that -- in which case two
    distinct actions (and their own agent/target nodes) end up sharing the
    identical color. `warnings` has one entry naming this when it happens;
    otherwise it's empty.
    """
    actions = [n for n in graph.nodes if n.role == "action"]

    warnings: List[str] = []
    if len(actions) > len(_ACTION_COLOR_PALETTE):
        warnings.append(
            f"{len(actions)} actions but only {len(_ACTION_COLOR_PALETTE)} palette "
            "colors -- colors repeat and some distinct actions will share a color"
        )

    action_index: Dict[NodeKey, int] = {_node_key(a): i for i, a in enumerate(actions)}

    by_key: Dict[NodeKey, AATNode] = {_node_key(n): n for n in graph.nodes}
    cluster_of_node: Dict[NodeKey, NodeKey] = {}
    for node in graph.nodes:
        key = _node_key(node)
        if node.role == "action":
            cluster_of_node[key] = key
        elif node.related_node is not None:
            related_key = (node.context, node.related_node)
            if related_key in by_key:
                cluster_of_node[key] = related_key

    color_of_node: Dict[NodeKey, ColorTriple] = {}
    for node in graph.nodes:
        key = _node_key(node)
        cluster_key = cluster_of_node.get(key)
        if cluster_key is None or cluster_key not in action_index:
            continue
        color_of_node[key] = _ACTION_COLOR_PALETTE[action_index[cluster_key] % len(_ACTION_COLOR_PALETTE)]

    return color_of_node, warnings
