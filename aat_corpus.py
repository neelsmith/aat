"""
A runnable script that reads an entire corpus of citable passages from a
CEX (CITE Exchange) file's '#!ctsdata' block (see aat.core.cex for the
format itself) and analyzes every passage through the aat.english
pipeline, writing ONE combined serialized analysis to stdout -- the same
'#!passages'/'#!aatnodes' plain-text format aat.core.serialize_analysis()/
aat_main.py's own stdout use, just covering every passage in the corpus
at once instead of a single `--passage` string.

Usage:
    python3 aat_corpus.py corpus.cex > analysis.txt
    python3 aat_corpus.py --delimiter "|" corpus.cex > analysis.txt
    cat corpus.cex | python3 aat_corpus.py - > analysis.txt

Needs the same `.env` as aat_main.py (API_BASE/MODEL/API_KEY) -- see
.env.example and USAGE.md; `_configure_lm()` is reused directly from
aat_main.py rather than duplicated here (both scripts live at the repo
root, so no sys.path adjustment is needed to import it, unlike
utilities/optimize_gepa.py's own cross-directory import of the same
helper).

Every passage in the corpus gets its own separate LM call (via
aat.english.analyze_passages()) -- a large corpus means real API cost and
real wall-clock time; there's no batching, parallelism, or caching here.
Any referential problem aat.core.validate() catches along the way is
reported on stderr (never stdout, same discipline as aat_main.py and
aat_to_dot.py), so it never corrupts the serialized output.
"""

import argparse
import sys
from typing import List

from aat.core import CitedPassage, parse_cex_ctsdata, read_cex_passages, serialize_analysis
from aat.english import analyze_passages
from aat_main import _configure_lm


def _read_corpus(path: str, delimiter: str) -> List[CitedPassage]:
    """Read the corpus at `path` as a CEX file's own '#!ctsdata' block
    (see aat.core.cex.read_cex_passages()) -- or, if `path` is exactly
    "-", read the same format from stdin instead, so a corpus can be
    piped in rather than saved to a file first."""
    if path == "-":
        return parse_cex_ctsdata(sys.stdin.read(), delimiter=delimiter)
    return read_cex_passages(path, delimiter=delimiter)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze every passage in a CEX corpus file's '#!ctsdata' block "
            "and write one combined serialized analysis to stdout."
        )
    )
    parser.add_argument(
        "cex_file",
        help='Path to a CEX file (or "-" to read the same format from stdin).',
    )
    parser.add_argument(
        "--delimiter",
        default="#",
        help=(
            "Field delimiter used in the file's own '#!ctsdata' block "
            '(default: "#", the common CEX convention -- CEX itself never '
            "declares its delimiter inside the file, so override this if the "
            'corpus uses something else, e.g. "|").'
        ),
    )
    args = parser.parse_args()

    try:
        passages = _read_corpus(args.cex_file, args.delimiter)
    except (OSError, ValueError) as exc:
        sys.exit(f"aat_corpus.py: {exc}")

    _configure_lm()
    _tokens, graph = analyze_passages(passages)
    sys.stdout.write(serialize_analysis(passages, graph))
