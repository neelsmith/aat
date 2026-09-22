"""
Plain-text, pipe-delimited persistence for AATGraph and CitableToken,
independent of language or of how the graph was produced -- an analyzed
passage can be saved, diffed, hand-edited, or reloaded without re-running
any pipeline. There are two independent block types, either of which may
appear on its own or together in one file:

    #!aatnodes
    context|id|value|role|related_node
    homework1|t1|dog|agent|t2
    homework1|t2|ate|action|
    homework1|t3|homework|target|t2

    #!tokens
    context|id|value
    homework1|t1|The
    homework1|t2|dog
    homework1|t3|ate
    homework1|t4|my
    homework1|t5|homework
    homework1|t6|.

`#!aatnodes` (read_nodes()/write_nodes()/read_graph()) is the AAT graph
itself. `#!tokens` (read_tokens()/write_tokens()) is the complete,
already-tokenized input every node's `id` refers back to -- EVERY token
of the passage(s) analyzed, not just the ones that became a node (a
passage with six tokens and three nodes still has six rows here), in
reading order (row order matters for this block, unlike `#!aatnodes`,
where a node's position is meaningless and only its `id` and
`related_node` are read). serialize_analysis()/write_analysis() (a thin
wrapper around it) and read_analysis() build/write/read both blocks
together for exactly this use case -- see their own docstrings.

An earlier version of this format saved a `#!passages` block instead (a
passage's raw context/text, not its tokens), on the theory that
aat.english.tokenize() could deterministically re-derive the token list
later with no LM access needed. That turned out to have two real
problems worth recording here: a reload's correctness silently depended
on the tokenizer staying byte-for-byte identical to whatever version
produced the original analysis, forever, with nothing to catch a future
drift; and it had no way at all to represent a token list
aat.english.tokenize_corpus_by_sentence() produced (composite ids like
"1.14.t3" spanning several citation units, per aat.english.sentences),
so a file `aat_corpus_graph.py` saved couldn't be reloaded except by
separately re-running that same clustering logic. Storing the actual
resolved token list sidesteps both: reading a `#!tokens` block back is
just data, never re-derived by re-running any particular version of any
particular function, and it represents whatever token list the original
analysis actually used, composite ids included, with no special-casing
for how they were produced.

`related_node` (in an `#!aatnodes` block) is left blank (not the literal
string 'None') for a node with no related_node -- currently only ever an
independent action node, per AATNode's own docstring; agent/target nodes
always have one.

Multiple blocks of the SAME type in one file are concatenated, in file
order, into the one list their reader returns -- so simply concatenating
several write_nodes()/serialize_nodes() (or write_tokens()/
serialize_tokens()) outputs together and reading the result back gives
one combined list. A file may freely mix block types (as write_analysis()
does): each typed reader (read_nodes(), read_tokens()) reads only the
blocks matching its own label and skips over any other block's rows
entirely, so the two block types never interfere with each other's
parsing or validation.
"""

from typing import List, Optional, Tuple

from .graph import AATGraph, AATNode
from .tokens import CitableToken

AATNODES_LABEL = "#!aatnodes"
_AATNODES_HEADER = "context|id|value|role|related_node"

TOKENS_LABEL = "#!tokens"
_TOKENS_HEADER = "context|id|value"


def _read_blocks(path: str, label: str, header: str) -> Tuple[List[List[str]], bool]:
    """Shared block-parsing core behind read_nodes() and read_tokens():
    scan every '#!'-labelled block in `path`, and return (rows, seen) --
    `rows` is the split (by '|') data rows from every block whose own
    label is exactly `label`, concatenated in file order; `seen` is
    whether a block with that label appeared at all (distinguishes "no
    matching block in this file" from "a matching block with zero data
    rows", which `rows == []` alone can't).

    A block whose label is something OTHER than `label` (e.g. reading
    for `AATNODES_LABEL` in a file that also has a `TOKENS_LABEL`
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
    types in the same file (e.g. a '#!tokens' block) are ignored, see
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


def serialize_tokens(tokens: List[CitableToken]) -> str:
    """Render `tokens` as one '#!tokens' block, pipe-delimited, and
    return it as a string, in `tokens`' own order -- unlike
    serialize_nodes(), row order here is meaningful (it's the only thing
    that records reading order; a token's `id` alone doesn't, especially
    for a composite sentence-spanning id like "1.14.t3" -- see this
    module's own top docstring). Same no-escaping caveat as
    serialize_nodes(): avoid '|' in `value` if you plan to round-trip
    through this format."""
    lines = [TOKENS_LABEL, _TOKENS_HEADER]
    for token in tokens:
        lines.append(f"{token.context}|{token.id}|{token.value}")
    return "\n".join(lines) + "\n"


def write_tokens(tokens: List[CitableToken], path: str) -> None:
    """Write serialize_tokens(tokens) to `path`."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_tokens(tokens))


def read_tokens(path: str) -> List[CitableToken]:
    """Read every '#!tokens' block in `path` and return their rows,
    concatenated in file order (which is also reading order -- see
    serialize_tokens()'s own docstring), as a list of CitableToken --
    other block types in the same file (e.g. a '#!aatnodes' block) are
    ignored, see _read_blocks().

    Raises ValueError (via _read_blocks()) for a malformed file. Also
    raises ValueError if the file has no '#!tokens' block at all, same
    reasoning as read_nodes()."""
    rows, seen = _read_blocks(path, TOKENS_LABEL, _TOKENS_HEADER)
    if not seen:
        raise ValueError(f"file has no {TOKENS_LABEL!r} block")
    return [CitableToken(context=context, id=id_, value=value) for context, id_, value in rows]


def serialize_analysis(tokens: List[CitableToken], graph: AATGraph) -> str:
    """Render a complete, re-displayable analysis as one string: a
    '#!tokens' block (the complete, already-tokenized input every node
    in `graph` refers back to -- no re-tokenization needed to reload it,
    see this module's own top docstring) followed by a '#!aatnodes'
    block (the AAT graph an earlier, LM-dependent analysis step
    produced). write_analysis() is a thin wrapper that writes this same
    string to a file; call this directly instead when the text itself is
    what's wanted -- e.g. to hand to a UI that writes the file somewhere
    else (a chosen directory, a download, ...), or to embed the analysis
    in something larger without touching disk here at all. read_analysis()
    is the matching reader for a file written either way -- see that
    function's own docstring for the intended round trip."""
    return serialize_tokens(tokens) + "\n" + serialize_nodes(graph.nodes)


def write_analysis(tokens: List[CitableToken], graph: AATGraph, path: str) -> None:
    """Write serialize_analysis(tokens, graph) to `path`. See that
    function's own docstring for exactly what gets written, and
    read_analysis() for the matching reader."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(serialize_analysis(tokens, graph))


def read_analysis(path: str) -> Tuple[List[CitableToken], AATGraph]:
    """Read a file written by write_analysis(): both its '#!tokens'
    block (as a list of CitableToken, in reading order) and its
    '#!aatnodes' block (as an AATGraph). Raises ValueError if either
    block is missing -- see read_tokens()/read_nodes().

    Unlike the earlier '#!passages'-based format this replaced, the
    returned `tokens` are already exactly what the original analysis
    used -- e.g. straight into aat.english.tokens_to_html(tokens,
    graph=graph) -- no re-tokenization step, and so no dependency on
    aat.english.tokenize() (or aat.english.tokenize_corpus_by_sentence()
    for a sentence-spanning file) at all. This whole round trip still
    needs no LM access at any point; it just no longer needs to re-run
    any deterministic code either.
    """
    tokens = read_tokens(path)
    graph = read_graph(path)
    return tokens, graph
