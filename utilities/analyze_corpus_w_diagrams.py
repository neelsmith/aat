"""
A runnable script wrapping aat.corpus.analyze_corpus_w_diagrams(): does
everything utilities/analyze_corpus.py does (see that script's own
docstring), plus renders each successfully analyzed sentence cluster's
graph as a Graphviz PNG in `{output_dir}/pngs/` -- via the real `dot`
command-line tool, not a Python Graphviz binding, so `dot` must be
installed and on PATH.

Usage (from the repo root, same convention as utilities/analyze_corpus.py
and utilities/optimize_gepa.py):

    python3 utilities/analyze_corpus_w_diagrams.py corpus.cex out/
    python3 utilities/analyze_corpus_w_diagrams.py --lang nl --orientation LR mijn-corpus.cex out/   # "|" delimiter by default
    python3 utilities/analyze_corpus_w_diagrams.py --delimiter "#" corpus-using-hash.cex out/
    cat corpus.cex | python3 utilities/analyze_corpus_w_diagrams.py - out/

Needs the same .env aat_main.py uses (API_BASE/MODEL/API_KEY), plus
Graphviz's own `dot` CLI on PATH (checked up front, before any LM call --
see aat.corpus.analyze_corpus_w_diagrams()'s own docstring). A cluster
whose own `dot` invocation fails is logged as an ERROR in the run's
warnings file and skipped for that PNG only; its serialized text
analysis is still written normally.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse aat_main.py's own .env-loading + LM-config helper rather than
# duplicating it -- same convention utilities/analyze_corpus.py,
# utilities/optimize_gepa.py, and utilities/calibrate_max_tokens.py
# already use.
sys.path.insert(0, str(REPO_ROOT))
from aat_main import _configure_lm  # noqa: E402

from aat.corpus import analyze_corpus_w_diagrams  # noqa: E402


def _resolve_cex_path(cex_file: str) -> str:
    """Same "-" (stdin) convention as utilities/analyze_corpus.py's own
    helper of the same name -- see that docstring."""
    if cex_file != "-":
        return cex_file
    fd, path = tempfile.mkstemp(suffix=".cex", prefix="aat-corpus-stdin-", text=True)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(sys.stdin.read())
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a CEX corpus sentence by sentence, serializing each "
            "sentence cluster's own analysis AND a Graphviz PNG diagram "
            "to an output directory (aat.corpus.analyze_corpus_w_diagrams())."
        )
    )
    parser.add_argument(
        "cex_file",
        help='Path to a CEX file (or "-" to read the same format from stdin).',
    )
    parser.add_argument(
        "output_dir",
        help=(
            "Directory to write each cluster's analysis into (created if it "
            "doesn't already exist); PNGs go in {output_dir}/pngs/."
        ),
    )
    parser.add_argument(
        "--lang",
        choices=["en", "nl"],
        default="en",
        help='Language pipeline to run: "en" (English, default) or "nl" (Dutch).',
    )
    parser.add_argument(
        "--delimiter",
        default="|",
        help=(
            "Field delimiter used in the file's own '#!ctsdata' block "
            '(default: "|" -- this project\'s own convention, NOT CEX\'s '
            'own common "#" convention; CEX itself never declares its '
            "delimiter inside the file, so pass e.g. --delimiter '#' for a "
            "file using that convention instead)."
        ),
    )
    parser.add_argument(
        "--warnings-filename",
        default="warnings.txt",
        help=(
            "Filename (within output_dir) for the run's cost summary and any "
            'errors/warnings (default: "warnings.txt").'
        ),
    )
    parser.add_argument(
        "--orientation",
        default="BT",
        help='Diagram orientation: "BT" (default, bottom-to-top), "TB"/"TD" '
        '(top-down), "LR", or "RL" -- same as aat_to_dot.py\'s own flag.',
    )
    args = parser.parse_args()

    cex_path = _resolve_cex_path(args.cex_file)

    _configure_lm()

    try:
        summary = analyze_corpus_w_diagrams(
            cex_path,
            args.output_dir,
            lang=args.lang,
            delimiter=args.delimiter,
            warnings_filename=args.warnings_filename,
            orientation=args.orientation,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        # RuntimeError alongside aat_corpus.py's own OSError/ValueError --
        # analyze_corpus_w_diagrams() raises it specifically when `dot`
        # isn't on PATH, checked before any LM call is made.
        sys.exit(f"analyze_corpus_w_diagrams.py: {exc}")

    print(
        f"Analyzed {summary.clusters_succeeded}/{summary.clusters_total} clusters "
        f"({summary.clusters_failed} failed) -- output written to {args.output_dir}/"
    )
    print(f"Rendered {len(summary.png_paths)} PNG diagram(s) to {args.output_dir}/pngs/")
    print(f"See {summary.warnings_path} for the cost summary and any errors/warnings.")
