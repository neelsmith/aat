"""
A runnable script that reads a serialized AAT analysis from stdin -- the
same plain-text format aat.core.write_nodes()/serialize_nodes() produce
(a '#!aatnodes' block; a '#!passages' block, if also present -- e.g.
piped straight from aat_main.py's own stdout -- is simply ignored, see
aat.core.read_graph()) -- and writes it back out as a Graphviz DOT
digraph on stdout, via aat.core.graph_to_dot().

Typical use -- pipe aat_main.py's own output straight into this script:

    python3 aat_main.py --passage "The dog ate my homework." | python3 aat_to_dot.py > analysis.dot

Or reuse a file saved earlier (by aat_main.py, write_nodes(), or
write_analysis()):

    python3 aat_to_dot.py < analysis.txt > analysis.dot

No LM, no `.env`, no network access needed at all -- this only re-renders
an already-analyzed graph, the same way aat.core.graph_to_dot()/
save_dot() do.
"""

import argparse
import os
import sys
import tempfile

from aat.core import AATGraph, graph_to_dot, read_graph


def _read_graph_from_stdin() -> AATGraph:
    """Read all of stdin and parse it as an AATGraph via
    aat.core.read_graph() -- which itself needs a real file path to open,
    so this writes stdin's text to a throwaway temp file first and
    removes it again afterward; nothing here leaves a file behind for
    the caller to notice or clean up themselves.

    Raises ValueError (propagated from aat.core.read_graph(), via
    aat.core.serialization's _read_blocks()) if stdin isn't in this
    format at all, or has no '#!aatnodes' block -- e.g. empty input, or a
    file that's just a bare passage with no analysis in it yet.
    """
    text = sys.stdin.read()
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        return read_graph(path)
    finally:
        os.remove(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read a serialized AAT analysis from stdin (aat.core's plain-text "
            "'#!aatnodes' format -- a '#!passages' block, if present, is "
            "ignored) and write it out as a Graphviz DOT digraph on stdout."
        )
    )
    parser.add_argument(
        "--orientation",
        default="BT",
        help='Diagram orientation: "BT" (default, bottom-to-top), "TB"/"TD" '
        '(top-down), "LR", or "RL".',
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable the default action-cluster coloring (write a plain digraph).",
    )
    args = parser.parse_args()

    try:
        graph = _read_graph_from_stdin()
        dot_text, warnings = graph_to_dot(
            graph, orientation=args.orientation, color_by_action=not args.no_color
        )
    except ValueError as exc:
        sys.exit(f"aat_to_dot.py: {exc}")

    # Warnings on stderr, never stdout -- stdout is exactly the DOT text,
    # so it can be piped straight into `dot` or redirected to a file
    # (same stdout-purity discipline as aat_main.py's own serialized
    # output -- see notes/CLAUDE_WORKFLOW.md's "Verifying before
    # delivering").
    for w in warnings:
        print(f"Warning: {w}", file=sys.stderr)

    sys.stdout.write(dot_text + "\n")
