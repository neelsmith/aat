"""
A runnable script wrapping aat.corpus.analyze_corpus(): analyzes a full
CEX corpus sentence by sentence (see "Analyzing a corpus by sentence,
across citation-unit boundaries" and "Analyzing a full corpus, one file
per sentence cluster" in USAGE.md), serializing each sentence cluster's
own analysis to its own file in an output directory, rather than
aat_corpus.py's one-combined-file-to-stdout approach -- meant for a
corpus large enough that holding one combined result in memory, or
losing all of it if a run is interrupted partway through, isn't
practical.

Usage (from the repo root, so REPO_ROOT below and aat_main.py's own .env
lookup both resolve correctly -- same convention utilities/
optimize_gepa.py and utilities/calibrate_max_tokens.py already use):

    python3 utilities/analyze_corpus.py corpus.cex out/
    python3 utilities/analyze_corpus.py --lang nl mijn-corpus.cex out/   # "|" delimiter by default
    python3 utilities/analyze_corpus.py --delimiter "#" corpus-using-hash.cex out/
    cat corpus.cex | python3 utilities/analyze_corpus.py - out/

Needs the same .env aat_main.py uses (API_BASE/MODEL/API_KEY) -- see
USAGE.md's "Running an analysis from the command line". `_configure_lm()`
is reused directly from aat_main.py, same as aat_corpus.py's own script
and utilities/optimize_gepa.py.

Every sentence cluster gets its own separate LM call -- a large corpus
means real API cost and real wall-clock time; there's no batching,
parallelism, or caching here. A per-cluster progress line goes to
stderr as the run proceeds (never stdout, so this script's own stdout
stays just its final one-line summary); any error or validate()
warning goes to both stderr and the run's own warnings file. See
aat.corpus.analyze_corpus()'s own docstring for the full behavior,
including that ONE failed cluster does not abort the whole run.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Reuse aat_main.py's own .env-loading + LM-config helper rather than
# duplicating it -- same convention utilities/optimize_gepa.py and
# utilities/calibrate_max_tokens.py already use.
sys.path.insert(0, str(REPO_ROOT))
from aat_main import _configure_lm  # noqa: E402

from aat.corpus import analyze_corpus  # noqa: E402


def _resolve_cex_path(cex_file: str) -> str:
    """Return a real filesystem path for `cex_file` -- itself, unless
    it's exactly "-", in which case stdin is read in full and written to
    a throwaway temp file first (aat.corpus.analyze_corpus() needs a real
    path -- it opens the CEX file itself, via
    aat.core.read_cex_passages() -- so there's nothing to hand it
    directly for stdin). Same "-" convention aat_corpus.py's own
    _read_corpus() already uses; the temp file is left on disk rather
    than cleaned up immediately (unlike aat_to_dot.py's own stdin
    helper), since analyze_corpus() reads it lazily and this script has
    no single point after which it's safe to delete."""
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
            "sentence cluster's own analysis to its own file in an output "
            "directory (aat.corpus.analyze_corpus())."
        )
    )
    parser.add_argument(
        "cex_file",
        help='Path to a CEX file (or "-" to read the same format from stdin).',
    )
    parser.add_argument(
        "output_dir",
        help="Directory to write each cluster's analysis into (created if it doesn't already exist).",
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
    args = parser.parse_args()

    cex_path = _resolve_cex_path(args.cex_file)

    _configure_lm()

    try:
        summary = analyze_corpus(
            cex_path,
            args.output_dir,
            lang=args.lang,
            delimiter=args.delimiter,
            warnings_filename=args.warnings_filename,
        )
    except (OSError, ValueError) as exc:
        sys.exit(f"analyze_corpus.py: {exc}")

    print(
        f"Analyzed {summary.clusters_succeeded}/{summary.clusters_total} clusters "
        f"({summary.clusters_failed} failed) -- output written to {args.output_dir}/"
    )
    print(f"See {summary.warnings_path} for the cost summary and any errors/warnings.")
