"""
Plain-text, pipe-delimited persistence for AATGraph, independent of
language or of how the graph was produced -- an analyzed passage can be
saved, diffed, hand-edited, or reloaded without re-running any pipeline.

File shape: one or more '#!aatnodes' blocks, each followed by the header
line 'context|id|value|role|related_node', then one row per node:

    #!aatnodes
    context|id|value|role|related_node
    homework1|t1|dog|agent|t2
    homework1|t2|ate|action|
    homework1|t3|homework|target|t2

`related_node` is left blank (not the literal string 'None') for a node
with no related_node -- currently only ever an independent action node,
per AATNode's own docstring; agent/target nodes always have one.

Multiple '#!aatnodes' blocks in one file are concatenated, in file order,
into the one list read_nodes() returns -- so simply concatenating several
write_nodes()/serialize_nodes() outputs together and reading the result
back gives one combined graph.
"""

from typing import List

from .graph import AATGraph, AATNode

AATNODES_LABEL = "#!aatnodes"
_HEADER = "context|id|value|role|related_node"


def serialize_nodes(nodes: List[AATNode]) -> str:
    """Render `nodes` as one '#!aatnodes' block, pipe-delimited, and
    return it as a string. Does not check for '|' inside any field's own
    value -- there is no escaping mechanism, so avoid the delimiter
    character in `value` if you plan to round-trip through this format."""
    lines = [AATNODES_LABEL, _HEADER]
    for node in nodes:
        related = node.related_node or ""
        lines.append(f"{node.context}|{node.id}|{node.value}|{node.role}|{related}")
    return "\n".join(lines) + "\n"


def write_nodes(nodes: List[AATNode], path: str) -> None:
    """Write serialize_nodes(nodes) to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_nodes(nodes))


def read_nodes(path: str) -> List[AATNode]:
    """Read every '#!aatnodes' block in `path` and return their rows,
    concatenated in file order, as a list of AATNode.

    Raises ValueError, naming the offending line, for: a data line before
    any '#!aatnodes' block label; a label line with no header line before
    the next block or before the file ends; a header line that doesn't
    match `_HEADER` exactly; or a row that isn't exactly 5 columns. Raises
    ValueError (not returning an empty list) if the file has no
    '#!aatnodes' block at all, so a caller can't mistake "wrong file" for
    "file with zero nodes".
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    nodes: List[AATNode] = []
    seen_block = False
    awaiting_header = False

    for line_no, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            continue

        if line == AATNODES_LABEL:
            if awaiting_header:
                raise ValueError(
                    f"line {line_no}: a {AATNODES_LABEL!r} block has a label "
                    "line but no header line before the next block starts"
                )
            seen_block = True
            awaiting_header = True
            continue

        if not seen_block:
            raise ValueError(
                f"line {line_no}: data line {line!r} appears before any "
                f"{AATNODES_LABEL!r} block label"
            )

        if awaiting_header:
            if line != _HEADER:
                raise ValueError(
                    f"line {line_no}: expected header {_HEADER!r} for a "
                    f"{AATNODES_LABEL!r} block, got {line!r}"
                )
            awaiting_header = False
            continue

        parts = line.split("|")
        if len(parts) != 5:
            raise ValueError(
                f"line {line_no}: {AATNODES_LABEL!r} row has {len(parts)} "
                f"column(s), expected 5: {line!r}"
            )
        context, id_, value, role, related = parts
        nodes.append(
            AATNode(
                context=context,
                id=id_,
                value=value,
                role=role,
                related_node=related or None,
            )
        )

    if not seen_block:
        raise ValueError(f"file has no {AATNODES_LABEL!r} block")
    if awaiting_header:
        raise ValueError(
            f"a {AATNODES_LABEL!r} block has a label line but no header "
            "line (and no data) -- the file ends too early"
        )

    return nodes


def read_graph(path: str) -> AATGraph:
    """Convenience wrapper: read_nodes(path) wrapped as an AATGraph."""
    return AATGraph(nodes=read_nodes(path))
