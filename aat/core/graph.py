"""
The AAT graph itself: the output side of the AAT model (aat-model.md, "The
AAT graph for English" -- despite that section header, the graph *shape*
it defines is not English-specific; only the rules for how to populate it
from English text are English-specific. See aat.english for those rules.)

An AATGraph is a selection of nodes drawn from a passage's citable tokens.
Every node keeps the context/id/value of the token(s) it derives from (for
a compound action, see AATNode's own docstring on id/value), and adds two
fields the raw tokens don't have: `role` (agent/action/target) and
`related_node`.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

Role = Literal["agent", "action", "target"]

ROLES: tuple = ("agent", "action", "target")


class AATNode(BaseModel):
    """One node in an AAT graph.

    `context` and `id` identify the token this node is built from -- for a
    single-token action, an agent, or a target, `id` is just that token's
    own id. For a *compound* action (aat-model.md's "Actions" section --
    e.g. "was eating"), `id` is instead the id of the *principal verb*
    token within the compound (e.g. the token for "eating"), and `value`
    is the space-joined text of every component token in surface order
    (e.g. "was eating"), not just the principal verb's own text.

    `related_node` is:
      - `None`, for an *independent* action node (aat-model.md's
        "Actions" section);
      - another action node's `id`, for a *dependent* action node (the id
        of the governing action it depends on);
      - another action node's `id`, for an agent or target node (the id
        of the action it's the agent/target of) -- agent and target nodes
        always have a related_node; only an action node's related_node
        can be `None`.
    """

    context: str = Field(description="Context reference this node belongs to.")
    id: str = Field(
        description=(
            "Token id this node is built from -- the principal verb's id "
            "for a compound action, otherwise the single underlying "
            "token's id."
        )
    )
    value: str = Field(
        description=(
            "This node's string value -- a single token's own text, or, "
            "for a compound action, every component token's text joined "
            "by spaces in surface order."
        )
    )
    role: Role = Field(description="'agent', 'action', or 'target'.")
    related_node: Optional[str] = Field(
        default=None,
        description=(
            "For an action node: the id of the governing action node, if "
            "this action is dependent/subordinate; None if independent. "
            "For an agent or target node: the id of the action node it "
            "relates to (always set)."
        ),
    )


class AATGraph(BaseModel):
    """A full AAT graph for one or more citable passages: an ordered list
    of AATNode. Node order is not semantically significant -- the
    convenience accessors below don't depend on it -- but callers that
    built the graph from a specific passage will typically find document
    order convenient for display."""

    nodes: List[AATNode] = Field(default_factory=list)

    def by_id(self, context: str, id: str) -> Optional[AATNode]:
        """The node with this (context, id), or None. `id` alone isn't a
        reliable key across contexts (CitableToken/AATNode ids are only
        unique *within* one context -- see CitableToken's docstring), so
        both are required here."""
        for node in self.nodes:
            if node.context == context and node.id == id:
                return node
        return None

    def actions(self) -> List[AATNode]:
        """Every action node, in list order."""
        return [n for n in self.nodes if n.role == "action"]

    def agents(self) -> List[AATNode]:
        """Every agent node, in list order."""
        return [n for n in self.nodes if n.role == "agent"]

    def targets(self) -> List[AATNode]:
        """Every target node, in list order."""
        return [n for n in self.nodes if n.role == "target"]

    def agents_for(self, action: AATNode) -> List[AATNode]:
        """Every agent node whose related_node points at `action`'s id,
        within the same context."""
        return [
            n for n in self.nodes
            if n.role == "agent" and n.context == action.context and n.related_node == action.id
        ]

    def targets_for(self, action: AATNode) -> List[AATNode]:
        """Every target node whose related_node points at `action`'s id,
        within the same context."""
        return [
            n for n in self.nodes
            if n.role == "target" and n.context == action.context and n.related_node == action.id
        ]

    def governing_action(self, action: AATNode) -> Optional[AATNode]:
        """For a dependent action node, the action node it's subordinate
        to; None if `action` is independent (related_node is None) or if
        related_node doesn't resolve to any node in this graph."""
        if action.related_node is None:
            return None
        return self.by_id(action.context, action.related_node)
