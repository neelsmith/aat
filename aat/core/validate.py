"""
Referential validation for an AATGraph: does every node actually resolve
to something real, given the token list it claims to be built from? This
checks referential integrity only -- never correctness of the underlying
linguistic analysis; see aat.english.pipeline / DEVELOPMENT.md for how the
English pipeline builds on top of this.
"""

from typing import List

from .graph import AATGraph
from .tokens import CitableToken


def validate(tokens: List[CitableToken], graph: AATGraph) -> List[str]:
    """Return a list of human-readable problem descriptions found in
    `graph`, given the `tokens` it was built from. An empty list means no
    referential problems were found (not that the analysis is correct).

    Checks:
      - every node's (context, id) matches a real token in `tokens`;
      - every node's `value` is non-empty;
      - an action node's `related_node`, if set, resolves to another
        action node's id in the same context;
      - an agent or target node's `related_node` is set (not None) and
        resolves to an action node's id in the same context.
    """
    problems: List[str] = []

    token_ids = {(t.context, t.id) for t in tokens}
    action_ids = {(n.context, n.id) for n in graph.nodes if n.role == "action"}

    for node in graph.nodes:
        if (node.context, node.id) not in token_ids:
            problems.append(
                f"node ({node.context!r}, {node.id!r}, role={node.role!r}) "
                "does not match any token in the input token list"
            )
        if not node.value:
            problems.append(
                f"node ({node.context!r}, {node.id!r}) has an empty value"
            )

        if node.role == "action":
            if node.related_node is not None and (node.context, node.related_node) not in action_ids:
                problems.append(
                    f"action node ({node.context!r}, {node.id!r}) has related_node "
                    f"{node.related_node!r}, which is not the id of any action node "
                    "in this context"
                )
        else:  # agent or target
            if node.related_node is None:
                problems.append(
                    f"{node.role} node ({node.context!r}, {node.id!r}) has no "
                    "related_node -- every agent/target node must relate to an "
                    "action node"
                )
            elif (node.context, node.related_node) not in action_ids:
                problems.append(
                    f"{node.role} node ({node.context!r}, {node.id!r}) has related_node "
                    f"{node.related_node!r}, which is not the id of any action node "
                    "in this context"
                )

    return problems
