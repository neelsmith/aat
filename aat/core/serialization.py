"""
Plain-text, pipe-delimited persistence for AATGraph and CitedPassage,
independent of language or of how the graph was produced -- an analyzed
passage can be saved, diffed, hand-edited, or reloaded without re-running
any pipeline. There are two independent block types, either of which may
appear on its own or together in one file:

    #!aatnodes
    context|id|value|role|related_node
    homework1|t1|dog|agent|t2
    homework1|t2|ate|action|
    homework1|t3|homework|target|t2

    #!passages
    context|text
    homework1|The dog ate the homework.

`#!aatnodes` (read_nodes()/write_nodes()/read_graph()) is the AAT graph
itself. `#!passages` (read_passages()/write_passages()) is the passage(s)
the graph was built from -- context and raw text, nothing else -- so a
caller with no LM access can still recover a passage's tokens
deterministically (aat.english.tokenize() needs no LM, only the raw
text) and pair them back up with the already-analyzed graph, without
re-running the (LM-dependent) analysis step. serialize_analysis()/
write_analysis() (a thin wrapper around it) and read_analysis() build/
write/read both blocks together for exactly this use
case -- see their own docstrings.

`related_node` is left blank (not the literal string 'None') for a node
with no related_node -- currently only ever an independent action node,
per AATNode's own docstring; agent/target nodes always have one.

Multiple blocks of the SAME type in one file are concatenated, in file
order, into the one list their reader returns -- so simply concatenating
several write_nodes()/serialize_nodes() (or write_passages()/
serialize_passages()) outputs together and reading the result back gives
one combined list. A file may freely mix block types (as write_analysis()
does): each typed reader (read_nodes(), read_passages()) reads only the
blocks matching its own label and skips over any other block's rows
entirely, so the two block types never interfere with each other's
parsing or validation.
"""

from typing import List, Optional, Tuple

from .graph import AATGraph, AATNode
from .tokens import CitedPassage

AATNODES_LABEL = "#!aatnodes"
_AATNODES_HEADER = "context|id|value|role|related_node"

PASSAGES_LABEL = "#!passages"
_PASSAGES_HEADER = "context|text"


def _read_blocks(path: str, label: str, header: str) -> Tuple[List[List[str]], bool]:
    """Shared block-parsing core behind read_nodes() and read_passages():
    scan every '#!'-labelled block in `path`, and return (rows, seen) --
    `rows` is the split (by '|') data rows from every block whose own
    label is exactly `label`, concatenated in file order; `seen` is
    whether a block with that label appeared at all (distinguishes "no
    matching block in this file" from "a matching block with zero data
    rows", which `rows == []` alone can't).

    A block whose label is something OTHER than `label` (e.g. reading
    for `AATNODES_LABEL` in a file that also has a `PASSAGES_LABEL`
    block) is skipped entirely -- its header line is consumed but never
    checked against `header`, and its data rows are ignored without
    being column-count-validated -- so one file can hold both block
    types side by side, each read independently by its own typed reader
    without the other's rows or column shape tripping up validation
    here.

    Raises ValueError, naming the offending line, for: a data line
    before any '#!'-labelled block at all (of any label); ANY block
    (matching `label` or not) whose label line has no header line before
    the next block starts or before the file ends -- a block missing its
    header is malformed regardless of which reader happens to be
    parsing, so this is not deferred to the matching-label-only checks
    below; a `label`-block whose header line doesn't match `header`
    exactly; or a `label`-block row that doesn't split into the same
    column count as `header`.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    rows: List[List[str]] = []
    seen = False
    current_label: Optional[str] = None
    awaiting_header = False
    expected_cols = header.count("|") + 1

    for line_no, line in enumerate(raw_lines, start=1):
        if line.strip() == "":
            continue

        if line.startswith("#!"):
            if awaiting_header:
                raise ValueError(
                    f"line {line_no}: a {current_label!r} block has a label "
                    "line but no header line before the next block starts"
                )
            current_label = line
            awaiting_header = True
            if current_label == label:
                seen = True
            continue

        if current_label is None:
            raise ValueError(
                f"line {line_no}: data line {line!r} appears before any "
                "'#!'-labelled block"
            )

        if awaiting_header:
            if current_label == label and line != header:
                raise ValueError(
                    f"line {line_no}: expected header {header!r} for a "
                    f"{label!r} block, got {line!r}"
                )
            awaiting_header = False
            continue

        if current_label != label:
            continue  # a row belonging to some other block type -- skip

        parts = line.split("|")
        if len(parts) != expected_cols:
            raise ValueError(
                f"line {line_no}: {label!r} row has {len(parts)} column(s), "
                f"expected {expected_cols}: {line!r}"
            )
        rows.append(parts)

    if awaiting_header:
        raise ValueError(
            f"a {current_label!r} block has a label line but no header "
            "line (and no data) -- the file ends too early"
        )

    return rows, seen


def serialize_nodes(nodes: List[AATNode]) -> str:
    """Render `nodes` as one '#!aatnodes' block, pipe-delimited, and
    return it as a string. Does not check for '|' inside any field's own
    value -- there is no escaping mechanism, so avoid the delimiter
    character in `value` if you plan to round-trip through this format."""
    lines = [AATNODES_LABEL, _AATNODES_HEADER]
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
    concatenated in file order, as a list of AATNode -- other block
    types in the same file (e.g. a '#!passages' block) are ignored, see
    _read_blocks().

    Raises ValueError (via _read_blocks()) for a malformed file -- see
    that function's own docstring for the exact cases. Also raises
    ValueError (not returning an empty list) if the file has no
    '#!aatnodes' block at all, so a caller can't mistake "wrong file" for
    "file with zero nodes".
    """
    rows, seen = _read_blocks(path, AATNODES_LABEL, _AATNODES_HEADER)
    if not seen:
        raise ValueError(f"file has no {AATNODES_LABEL!r} block")

    nodes: List[AATNode] = []
    for context, id_, value, role, related in rows:
        nodes.append(
            AATNode(
                context=context,
                id=id_,
                value=value,
                role=role,
                related_node=related or None,
            )
        )
    return nodes


def read_graph(path: str) -> AATGraph:
    """Convenience wrapper: read_nodes(path) wrapped as an AATGraph."""
    return AATGraph(nodes=read_nodes(path))


def serialize_passages(passages: List[CitedPassage]) -> str:
    """Render `passages` as one '#!passages' block, pipe-delimited, and
    return it as a string. Same no-escaping caveat as serialize_nodes():
    avoid '|' in `text` if you plan to round-trip through this format."""
    lines = [PASSAGES_LABEL, _PASSAGES_HEADER]
    for passage in passages:
        lines.append(f"{passage.context}|{passage.text}")
    return "\n".join(lines) + "\n"


def write_passages(passages: List[CitedPassage], path: str) -> None:
    """Write serialize_passages(passages) to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_passages(passages))


def read_passages(path: str) -> List[CitedPassage]:
    """Read every '#!passages' block in `path` and return their rows,
    concatenated in file order, as a list of CitedPassage -- other block
    types in the same file (e.g. a '#!aatnodes' block) are ignored, see
    _read_blocks().

    Raises ValueError (via _read_blocks()) for a malformed file. Also
    raises ValueError if the file has no '#!passages' block at all, same
    reasoning as read_nodes()."""
    rows, seen = _read_blocks(path, PASSAGES_LABEL, _PASSAGES_HEADER)
    if not seen:
        raise ValueError(f"file has no {PASSAGES_LABEL!r} block")
    return [CitedPassage(context=context, text=text) for context, text in rows]


def serialize_analysis(passages: List[CitedPassage], graph: AATGraph) -> str:
    """Render a complete, re-displayable analysis as one string: a
    '#!passages' block (so aat.english.tokenize() can deterministically
    rebuild tokens later with no LM access needed at all) followed by a
    '#!aatnodes' block (the AAT graph an earlier, LM-dependent analysis
    step produced). write_analysis() is a thin wrapper that writes this
    same string to a file; call this directly instead when the text
    itself is what's wanted -- e.g. to hand to a UI that writes the file
    somewhere else (a chosen directory, a download, ...), or to embed
    the analysis in something larger without touching disk here at all.
    read_analysis() is the matching reader for a file written either
    way -- see that function's own docstring for the intended round
    trip."""
    return serialize_passages(passages) + "\n" + serialize_nodes(graph.nodes)


def write_analysis(passages: List[CitedPassage], graph: AATGraph, path: str) -> None:
    """Write serialize_analysis(passages, graph) to `path`. See that
    function's own docstring for exactly what gets written, and
    read_analysis() for the matching reader."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_analysis(passages, graph))


def read_analysis(path: str) -> Tuple[List[CitedPassage], AATGraph]:
    """Read a file written by write_analysis(): both its '#!passages'
    block (as a list of CitedPassage) and its '#!aatnodes' block (as an
    AATGraph). Raises ValueError if either block is missing -- see
    read_passages()/read_nodes().

    A caller that only has `path` and needs tokens to pair back up with
    the returned graph (e.g. to highlight them with
    aat.english.html.tokens_to_html()) re-tokenizes each returned
    passage with aat.english.tokenize() -- deterministic and LM-free, so
    this whole round trip needs no LM access at any point."""
    passages = read_passages(path)
    graph = read_graph(path)
    return passages, graph
