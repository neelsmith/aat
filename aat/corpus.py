"""
aat.corpus: analyze a full citable corpus -- normally loaded from a CEX
source file -- sentence by sentence, serializing each sentence's own
analysis to its own file in an output directory. See notes/corpus.qmd
for the algorithm/implementation spec this module implements.

Why this exists as its own top-level module, rather than living inside
aat.english or aat.dutch: it needs dspy transitively (an LM has to be
configured and called), so it can't live in aat.core, but it also
dispatches between *both* language packages, so it doesn't belong inside
either single language package either. `aat/__init__.py` re-exports only
aat.core (see aat.lm_cost's own docstring for the same reasoning), so
import this module directly:

    from aat.corpus import analyze_corpus, analyze_corpus_w_diagrams

    summary = analyze_corpus("mycorpus.cex", "out/")

Algorithm (notes/corpus.qmd): a citable corpus groups text by citation
unit, but citation units don't always align with sentence boundaries (a
poem citable by line where a sentence spans several lines is the classic
case -- see aat.english.sentences's own module docstring for a real
example). So a corpus is first clustered, algorithmically and without
any LM call, into the smallest possible runs of citation units such that
each run's last unit actually ends a sentence
(aat.english.sentences.tokenize_corpus_by_sentence() /
aat.dutch.sentences.tokenize_corpus_by_sentence() -- identical logic in
both language packages). Each resulting cluster is then submitted for
analysis one at a time, in citation order.

This deliberately does NOT reuse aat.english.analyze_units_by_sentence()
(or aat.dutch's equivalent) directly: that function combines every
cluster's nodes into one big AATGraph and returns it as a single result,
which is exactly right for a notebook that wants to display or export
"the whole corpus" as one graph, but wrong here -- notes/corpus.qmd asks
for each cluster's analysis to be serialized *individually*, with
progress, cost, and error/warning tracking as the run goes rather than
only once at the end. So this module re-implements that same per-cluster
loop itself, serializing (aat.core.write_analysis()) after each cluster
rather than accumulating everything into one combined graph first.

Beyond notes/corpus.qmd's own two literal parameters (a CEX source path,
an output directory), both functions here also take a `lang` keyword
("en" or "nl", default "en") selecting which language package's own
tokenize/analyze/validate pipeline runs the corpus -- an intentional
extension beyond the literal spec (which predates aat.dutch even being
mentioned), continuing this session's established bilingual-support
pattern rather than leaving analyze_corpus() English-only while every
other entry point in the project now supports both languages. `lang`
uses short codes ("en"/"nl") rather than the "English"/"Dutch" names
LANGUAGE_MODULES (below) and the marimo notebooks' own language pickers
use internally -- that's this function's own public parameter shape, by
request; _LANG_TO_MODULE_KEY maps one to the other.

Neither function configures the LM itself -- same convention as
aat.english.analyze_passage()/analyze_passages(): the caller is expected
to have already called dspy.configure(lm=...) (or dspy.context(lm=...))
before calling in, exactly as every marimo notebook and CLI script in
this project already does its own LM setup separately from analysis.

Progress and any errors/warnings are printed one line at a time to
stderr, not stdout, as the run proceeds (notes/corpus.qmd: "display a
progress message visible to the user (not on standard out)") -- this
module doesn't write anything at all to stdout. The very first such
line, printed once clustering finishes and before the first cluster is
analyzed, reports the run's own scope: how many citable passages
read_cex_passages() read, and how many sentence clusters
tokenize_corpus_by_sentence() grouped them into (see this module's own
top docstring for why the second number is often smaller than the
first). The same lines (that scope line, plus errors and warnings, not
the bare per-cluster progress notices) are also collected and written,
together with a final LM cost summary
(aat.lm_cost.summarize_lm_cost()/format_lm_cost(), reading
dspy.settings.lm.history directly -- same source the marimo notebooks'
own cost display reads from), to a single text file in the output
directory (`warnings_filename`, default "warnings.txt").

A cluster whose analysis raises (e.g. aat.english.token_budget.
analyze_with_retry() giving up and propagating a persistent
AdapterParseError -- see that module's own docstring) is caught,
logged as an ERROR line, and skipped -- the run continues with the next
cluster rather than aborting the whole corpus over one bad sentence.
A cluster that raises no exception but fails validate() (a referential
problem, not a crash) is still serialized -- same "warn, don't fail"
convention aat.english.pipeline.analyze_passages()/
analyze_units_by_sentence() already use -- with each problem logged as
a WARNING line.
"""

import shutil
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import dspy

import aat.dutch
import aat.english
from aat.core import AATGraph, graph_to_dot, read_cex_passages, write_analysis
from aat.lm_cost import format_lm_cost, summarize_lm_cost

# Same dispatch table as marimo/aat_graph.py and marimo/aat_corpus_graph.py
# -- kept in sync with those by hand, same as those two notebooks are kept
# in sync with each other, since none of the three can import from either
# of the others (the notebooks aren't importable modules).
LANGUAGE_MODULES = {"English": aat.english, "Dutch": aat.dutch}

# This module's own public `lang` parameter uses short codes ("en"/"nl")
# rather than LANGUAGE_MODULES's own "English"/"Dutch" keys -- see this
# module's top docstring for why. Keep in sync with LANGUAGE_MODULES.
_LANG_TO_MODULE_KEY = {"en": "English", "nl": "Dutch"}

DEFAULT_WARNINGS_FILENAME = "warnings.txt"


def _safe_filename(context: str) -> str:
    """Turn a (possibly sentence-composite) context string -- e.g.
    "urn:cts:test:work:1.2-1.3" -- into a filesystem-safe filename stem.
    Same sanitization rule the marimo notebooks (aat_graph.py,
    aat_corpus_graph.py) already use for their own downloadable analysis
    filenames, reused here so a corpus run's per-cluster filenames follow
    an identical, already-established convention rather than a new one."""
    cleaned = "".join(c if c.isalnum() or c in "_-" else "_" for c in context).strip("_")
    return cleaned or "analysis"


def _resolve_lang(lang: str):
    """Validate `lang` ("en" or "nl") and return its LANGUAGE_MODULES
    entry. Raises ValueError, naming the accepted codes, for anything
    else -- a plain KeyError from a bare dict lookup wouldn't say what
    values ARE accepted."""
    try:
        module_key = _LANG_TO_MODULE_KEY[lang]
    except KeyError:
        raise ValueError(
            f"Unknown lang {lang!r} -- expected one of "
            f"{sorted(_LANG_TO_MODULE_KEY)}"
        ) from None
    return LANGUAGE_MODULES[module_key]


@dataclass
class CorpusRunSummary:
    """What analyze_corpus()/analyze_corpus_w_diagrams() returns: a
    record of one run, useful for tests and for a caller (e.g. a future
    CLI wrapper) that wants more than just "go look in the output
    directory". Every path here is a plain str, not a Path, matching
    this module's own public function signatures.

    `clusters_total` is how many sentence-level clusters
    tokenize_corpus_by_sentence() produced; `clusters_succeeded` +
    `clusters_failed` always sum to it. `output_paths` lists every
    per-cluster analysis file actually written, in cluster order (one
    fewer entry than `clusters_total` for each failed cluster).
    `png_paths` is only ever non-empty for analyze_corpus_w_diagrams()."""

    clusters_total: int
    clusters_succeeded: int
    clusters_failed: int
    output_paths: List[str] = field(default_factory=list)
    png_paths: List[str] = field(default_factory=list)
    warnings_path: str = ""
    cost_summary: str = ""


def _run(
    cex_path: str,
    output_dir: str,
    *,
    lang: str,
    delimiter: str,
    warnings_filename: str,
    make_diagrams: bool,
    orientation: str,
) -> CorpusRunSummary:
    module = _resolve_lang(lang)

    if make_diagrams and shutil.which("dot") is None:
        # Checked up front, before spending any LM budget on a run that
        # would just fail at the first diagram anyway.
        raise RuntimeError(
            "analyze_corpus_w_diagrams() needs the Graphviz 'dot' "
            "command-line tool, which isn't on PATH -- install Graphviz "
            "(e.g. 'apt install graphviz' or 'brew install graphviz') "
            "before calling this."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png_dir = out_dir / "pngs"
    if make_diagrams:
        png_dir.mkdir(parents=True, exist_ok=True)

    units = read_cex_passages(cex_path, delimiter=delimiter)
    # LM-free clustering -- see this module's own top docstring.
    groups = module.tokenize_corpus_by_sentence(units)

    log_lines: List[str] = []
    output_paths: List[str] = []
    png_paths: List[str] = []
    succeeded = 0
    failed = 0
    used_names = set()

    total = len(groups)

    # Reported once, after clustering but before the first cluster is
    # analyzed -- so the user knows the shape of the run (how many
    # citable passages were read, how many sentence clusters they were
    # grouped into -- always <= the passage count, and < it whenever a
    # sentence actually spans citation units) before any LM cost is
    # spent. Stderr only, never stdout, same discipline as the
    # per-cluster progress lines below; also recorded as the first line
    # of warnings.txt, ahead of the run-summary line written once the
    # loop finishes, so a report of the run states its own scope even
    # before a single cluster succeeds or fails.
    cluster_plan_message = (
        f"Read {len(units)} citable passage(s), clustered into {total} sentence cluster(s)."
    )
    print(cluster_plan_message, file=sys.stderr)

    for i, (combined_context, combined_text, tokens) in enumerate(groups, start=1):
        print(f"[{i}/{total}] analyzing {combined_context!r} ...", file=sys.stderr)

        try:
            # Recorded rather than left to Python's default stderr
            # printing, so a truncation-retry UserWarning (see
            # token_budget.analyze_with_retry()'s own docstring) both
            # shows up here as a progress line AND lands in
            # warnings.txt -- matching this project's own convention
            # (aat_to_dot.py) of printing warnings explicitly rather
            # than relying on the `warnings` module's default handler.
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = module.analyze_with_retry(passage=combined_text, tokens=tokens)
            for w in caught:
                message = f"WARNING ({combined_context}): {w.message}"
                print(message, file=sys.stderr)
                log_lines.append(message)
        except Exception as exc:
            # One bad cluster (e.g. analyze_with_retry() giving up after
            # persistent truncation) shouldn't abort the whole corpus --
            # log it and move on to the next cluster.
            message = f"ERROR ({combined_context}): {exc}"
            print(message, file=sys.stderr)
            log_lines.append(message)
            failed += 1
            continue

        problems = module.validate(tokens, result)
        for p in problems:
            message = f"WARNING ({combined_context}): {p}"
            print(message, file=sys.stderr)
            log_lines.append(message)

        graph = AATGraph(nodes=result.nodes)

        stem = _safe_filename(combined_context)
        name = stem
        suffix = 2
        while name in used_names:
            name = f"{stem}_{suffix}"
            suffix += 1
        used_names.add(name)

        out_path = out_dir / f"{name}.txt"
        write_analysis(tokens, graph, str(out_path))
        output_paths.append(str(out_path))
        succeeded += 1

        if make_diagrams:
            dot_text, dot_warnings = graph_to_dot(graph, orientation=orientation)
            for w in dot_warnings:
                message = f"WARNING ({combined_context}): {w}"
                print(message, file=sys.stderr)
                log_lines.append(message)

            proc = subprocess.run(
                ["dot", "-Tpng"],
                input=dot_text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            if proc.returncode != 0:
                message = (
                    f"ERROR ({combined_context}): 'dot' failed to render a PNG: "
                    f"{proc.stderr.decode('utf-8', errors='replace').strip()}"
                )
                print(message, file=sys.stderr)
                log_lines.append(message)
            else:
                png_path = png_dir / f"{name}.png"
                png_path.write_bytes(proc.stdout)
                png_paths.append(str(png_path))

    lm = dspy.settings.lm
    cost_summary = format_lm_cost(summarize_lm_cost(lm.history)) if lm is not None else "no LM configured"

    warnings_path = out_dir / warnings_filename
    with open(warnings_path, "w", encoding="utf-8") as f:
        f.write(cluster_plan_message + "\n")
        f.write(f"Clusters analyzed: {succeeded}/{total} ({failed} failed)\n")
        f.write(f"LM cost: {cost_summary}\n")
        if log_lines:
            f.write("\n")
            for line in log_lines:
                f.write(line + "\n")
        else:
            f.write("\nNo errors or warnings.\n")

    return CorpusRunSummary(
        clusters_total=total,
        clusters_succeeded=succeeded,
        clusters_failed=failed,
        output_paths=output_paths,
        png_paths=png_paths,
        warnings_path=str(warnings_path),
        cost_summary=cost_summary,
    )


def analyze_corpus(
    cex_path: str,
    output_dir: str,
    *,
    lang: str = "en",
    delimiter: str = "|",
    warnings_filename: str = DEFAULT_WARNINGS_FILENAME,
) -> CorpusRunSummary:
    """Analyze every sentence-level cluster in the CEX corpus at
    `cex_path`, writing each cluster's own analysis to its own file in
    `output_dir` (created if it doesn't already exist).

    `lang` selects which language package's own tokenize/analyze/
    validate pipeline to run: "en" (English, the default) or "nl"
    (Dutch). Raises ValueError for anything else. `delimiter` is passed
    straight through to aat.core.read_cex_passages() -- defaults to "|"
    here (this module's own convention, by request -- NOT
    read_cex_passages()'s own "#" default), so pass `delimiter="#"`
    explicitly for a file using that convention instead. Assumes an LM
    is already configured
    (dspy.configure(lm=...)) -- this function doesn't configure one
    itself, same convention as aat.english.analyze_passage()/
    analyze_passages().

    Clustering (grouping citation units into sentences) is done
    algorithmically, with no LM call, via that language's own
    tokenize_corpus_by_sentence() -- see this module's own top docstring
    for why that matters. Each resulting cluster is then analyzed and
    immediately serialized (aat.core.write_analysis()) to
    `{output_dir}/{sanitized context}.txt` before moving on to the next
    cluster, so a run that's interrupted partway through still leaves
    every cluster analyzed so far on disk.

    After clustering but before analyzing the first cluster, a one-line
    report of the run's own scope -- how many citable passages were
    read, how many sentence clusters they were grouped into -- is
    printed to stderr (never stdout), followed by a per-cluster progress
    line as the run proceeds, plus any error or validation warning. The
    same scope line, the same errors/warnings, and a final LM cost
    summary (aat.lm_cost.format_lm_cost()), are written to
    `{output_dir}/{warnings_filename}` (default "warnings.txt").

    Returns a CorpusRunSummary. A malformed or missing `cex_path` raises
    the same errors aat.core.read_cex_passages() itself would.
    """
    return _run(
        cex_path,
        output_dir,
        lang=lang,
        delimiter=delimiter,
        warnings_filename=warnings_filename,
        make_diagrams=False,
        orientation="BT",
    )


def analyze_corpus_w_diagrams(
    cex_path: str,
    output_dir: str,
    *,
    lang: str = "en",
    delimiter: str = "|",
    warnings_filename: str = DEFAULT_WARNINGS_FILENAME,
    orientation: str = "BT",
) -> CorpusRunSummary:
    """Identical to analyze_corpus(), with one addition: for every
    successfully analyzed cluster, also renders a Graphviz DOT digraph
    (aat.core.graph_to_dot()) and writes it as a PNG to
    `{output_dir}/pngs/{sanitized context}.png`, using the same
    sanitized filename stem as that cluster's own serialized analysis
    file. `{output_dir}/pngs/` is created if it doesn't already exist.
    `orientation` is passed straight through to graph_to_dot() (default
    "BT" -- see that function's own docstring for the other accepted
    values).

    Rendering shells out to the Graphviz `dot` command-line tool (piping
    the DOT text to `dot -Tpng` via stdin) -- NOT a Python Graphviz
    binding -- so `dot` must be installed and on PATH. This is checked
    up front, before any LM call is made, raising RuntimeError
    immediately if `dot` isn't found, rather than failing partway
    through a corpus after real LM cost has already been spent. A
    cluster whose own `dot` invocation fails (e.g. malformed DOT text)
    is logged as an ERROR in `warnings_filename` and skipped for that
    cluster's PNG only -- its serialized text analysis is still written
    normally.

    See analyze_corpus()'s own docstring for every other parameter
    (including `lang`) and behavior (both functions share the exact same
    clustering, serialization, progress-reporting, and error-handling
    logic).
    """
    return _run(
        cex_path,
        output_dir,
        lang=lang,
        delimiter=delimiter,
        warnings_filename=warnings_filename,
        make_diagrams=True,
        orientation=orientation,
    )
