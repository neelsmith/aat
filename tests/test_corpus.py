"""
Offline tests for aat.corpus (analyze_corpus() / analyze_corpus_w_diagrams())
-- notes/corpus.qmd's own spec. Every test uses DummyLM (no real LM call,
no .env, no network access), same convention as every other DummyLM-backed
test in this project (see tests/conftest.py). The diagrams tests are the
one exception that reaches outside DummyLM: they invoke the REAL `dot`
command-line tool (matching tests/test_aat_to_dot.py's own established
"exercise the real subprocess, don't mock it" convention for Graphviz-
related code), so they need Graphviz actually installed -- skipped
automatically (not failed) if `dot` isn't on PATH, the same graceful-
skip pattern conftest.py's own `real_lm` fixture uses for a missing API
key.
"""

import shutil

import dspy
import pytest
from dspy.utils.dummies import DummyLM

import aat.dutch
import aat.english
from aat.core import read_analysis

from aat.corpus import CorpusRunSummary, analyze_corpus, analyze_corpus_w_diagrams

_DOT_MISSING = shutil.which("dot") is None

# -- Fixture CEX corpora ------------------------------------------------
# Real CTS URNs (5 ':'-delimited fields) -- aat.english.sentences.
# passage_component()/urn_prefix() both require at least that many
# fields, same as every other sentence-boundary test in this project
# (see tests/test_english_pipeline.py's own urn:cts:test:work:... URNs).

# '|' delimiter for every ctsdata row below -- aat.corpus's own default
# (unlike aat.core.cex.read_cex_passages()'s own "#" default, which this
# module deliberately does NOT change -- see aat/corpus.py's own
# docstring). '#!ctsdata' itself is a block label, not a row, so it's
# unaffected either way.

_TWO_SENTENCE_CEX = """\
#!ctsdata
urn:cts:test:corpus.sample:1.1|The dog ate.
urn:cts:test:corpus.sample:1.2|The cat sat.
"""

_SPANNING_CEX = """\
#!ctsdata
urn:cts:test:corpus.sample:1.1|The dog
urn:cts:test:corpus.sample:1.2|ate the homework.
"""

_THREE_UNIT_CEX = """\
#!ctsdata
urn:cts:test:corpus.sample:1.1|He waited.
urn:cts:test:corpus.sample:1.2|The dog
urn:cts:test:corpus.sample:1.3|ate.
"""

_ANSWER_DOG_ATE = {
    "reasoning": "ate is the action; dog is the agent.",
    "nodes": [
        {"context": "urn:cts:test:corpus.sample:1.1", "id": "1.1.t3", "value": "ate", "role": "action", "related_node": None},
        {"context": "urn:cts:test:corpus.sample:1.1", "id": "1.1.t2", "value": "dog", "role": "agent", "related_node": "1.1.t3"},
    ],
}
_ANSWER_CAT_SAT = {
    "reasoning": "sat is the action; cat is the agent.",
    "nodes": [
        {"context": "urn:cts:test:corpus.sample:1.2", "id": "1.2.t3", "value": "sat", "role": "action", "related_node": None},
        {"context": "urn:cts:test:corpus.sample:1.2", "id": "1.2.t2", "value": "cat", "role": "agent", "related_node": "1.2.t3"},
    ],
}
_ANSWER_DOG_ATE_HOMEWORK_SPANNING = {
    "reasoning": "ate is the action; dog is the agent; homework is the target.",
    "nodes": [
        {"context": "urn:cts:test:corpus.sample:1.1-1.2", "id": "1.2.t1", "value": "ate", "role": "action", "related_node": None},
        {"context": "urn:cts:test:corpus.sample:1.1-1.2", "id": "1.1.t2", "value": "dog", "role": "agent", "related_node": "1.2.t1"},
        {"context": "urn:cts:test:corpus.sample:1.1-1.2", "id": "1.2.t3", "value": "homework", "role": "target", "related_node": "1.2.t1"},
    ],
}


def test_analyze_corpus_serializes_one_file_per_sentence_cluster(tmp_path):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_TWO_SENTENCE_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE, _ANSWER_CAT_SAT]))
    summary = analyze_corpus(str(cex_path), str(out_dir))

    assert isinstance(summary, CorpusRunSummary)
    assert summary.clusters_total == 2
    assert summary.clusters_succeeded == 2
    assert summary.clusters_failed == 0
    assert len(summary.output_paths) == 2

    tokens1, graph1 = read_analysis(summary.output_paths[0])
    assert graph1.nodes[0].value == "ate"
    tokens2, graph2 = read_analysis(summary.output_paths[1])
    assert graph2.nodes[0].value == "sat"


def test_analyze_corpus_clusters_a_sentence_spanning_two_citation_units(tmp_path):
    # Same shape as test_english_pipeline.py's own
    # test_analyze_units_by_sentence_spans_citation_units -- "The dog" +
    # "ate the homework." only form a complete sentence together, so
    # this must produce exactly ONE output file (one LM call), not two.
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_SPANNING_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE_HOMEWORK_SPANNING]))
    summary = analyze_corpus(str(cex_path), str(out_dir))

    assert summary.clusters_total == 1
    assert summary.clusters_succeeded == 1
    assert len(summary.output_paths) == 1

    tokens, graph = read_analysis(summary.output_paths[0])
    assert [t.id for t in tokens] == ["1.1.t1", "1.1.t2", "1.2.t1", "1.2.t2", "1.2.t3", "1.2.t4"]
    assert all(t.context == "urn:cts:test:corpus.sample:1.1-1.2" for t in tokens)


def test_analyze_corpus_writes_a_clean_warnings_file(tmp_path):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_TWO_SENTENCE_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE, _ANSWER_CAT_SAT]))
    summary = analyze_corpus(str(cex_path), str(out_dir))

    warnings_text = (out_dir / "warnings.txt").read_text(encoding="utf-8")
    assert summary.warnings_path == str(out_dir / "warnings.txt")
    # The passage/cluster scope line is the FIRST line in the file --
    # written before the run-summary line, matching the order it's
    # reported to the user (after clustering, before the first cluster
    # is analyzed).
    assert warnings_text.splitlines()[0] == "Read 2 citable passage(s), clustered into 2 sentence cluster(s)."
    assert "Clusters analyzed: 2/2 (0 failed)" in warnings_text
    assert "LM cost:" in warnings_text
    assert "No errors or warnings." in warnings_text


def test_analyze_corpus_reports_passage_and_cluster_counts_before_the_first_cluster(tmp_path, capsys):
    # _SPANNING_CEX has 2 citation units that cluster into exactly 1
    # sentence -- distinct passage/cluster counts, so a test that used
    # equal counts (e.g. _TWO_SENTENCE_CEX's 2-and-2) couldn't catch the
    # two numbers being swapped or both reading the same count.
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_SPANNING_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE_HOMEWORK_SPANNING]))
    analyze_corpus(str(cex_path), str(out_dir))

    captured = capsys.readouterr()
    assert captured.out == ""  # never stdout
    stderr_lines = captured.err.splitlines()
    scope_line = "Read 2 citable passage(s), clustered into 1 sentence cluster(s)."
    assert scope_line in stderr_lines
    # Reported strictly BEFORE the first per-cluster progress line --
    # "after clustering but before the first analysis is reported",
    # exactly as asked.
    first_progress_index = next(i for i, line in enumerate(stderr_lines) if line.startswith("[1/"))
    assert stderr_lines.index(scope_line) < first_progress_index

    warnings_text = (out_dir / "warnings.txt").read_text(encoding="utf-8")
    assert warnings_text.splitlines()[0] == scope_line


def test_analyze_corpus_honors_custom_delimiter_and_warnings_filename(tmp_path):
    cex_path = tmp_path / "corpus.cex"
    # '#' delimiter here -- the non-default case, exercising that
    # `delimiter` actually overrides aat.corpus's own "|" default rather
    # than merely accepting it.
    cex_path.write_text(
        "#!ctsdata\n"
        "urn:cts:test:corpus.sample:1.1#The dog ate.\n"
        "urn:cts:test:corpus.sample:1.2#The cat sat.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE, _ANSWER_CAT_SAT]))
    summary = analyze_corpus(
        str(cex_path), str(out_dir), delimiter="#", warnings_filename="run-log.txt"
    )

    assert summary.warnings_path == str(out_dir / "run-log.txt")
    assert (out_dir / "run-log.txt").exists()
    assert not (out_dir / "warnings.txt").exists()


def test_analyze_corpus_logs_validation_problems_as_warnings(tmp_path, capsys):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(
        "#!ctsdata\nurn:cts:test:corpus.sample:1.1|The dog ate.\n", encoding="utf-8"
    )
    out_dir = tmp_path / "out"

    broken_answer = {
        "reasoning": "action found, but related_node points nowhere real.",
        "nodes": [
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "1.1.t3", "value": "ate", "role": "action", "related_node": None},
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "1.1.t2", "value": "dog", "role": "agent", "related_node": "ghost"},
        ],
    }
    dspy.configure(lm=DummyLM([broken_answer]))
    summary = analyze_corpus(str(cex_path), str(out_dir))

    # Still serialized -- a referential problem warns, it doesn't fail
    # the cluster (same convention as aat.english.pipeline's own
    # validate()-and-warn behavior).
    assert summary.clusters_succeeded == 1
    assert summary.clusters_failed == 0

    warnings_text = (out_dir / "warnings.txt").read_text(encoding="utf-8")
    assert "WARNING" in warnings_text
    assert "ghost" in warnings_text

    captured = capsys.readouterr()
    assert "WARNING" in captured.err
    assert captured.out == ""  # progress/warnings never go to stdout


def test_analyze_corpus_one_failed_cluster_does_not_abort_the_run(tmp_path, monkeypatch):
    # Three sentences -> three clusters (see
    # test_english_pipeline.py's own
    # test_analyze_units_by_sentence_one_sentence_per_group, same
    # corpus shape). Simulate the MIDDLE cluster's own analyze call
    # raising, and confirm the run still analyzes the other two and
    # reports the failure rather than propagating it.
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_THREE_UNIT_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    answer1 = {
        "reasoning": "single independent action, no agent/target expressed.",
        "nodes": [
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "1.1.t2", "value": "waited", "role": "action", "related_node": None},
        ],
    }
    answer3 = {
        "reasoning": "single independent action, no agent/target expressed.",
        "nodes": [
            {"context": "urn:cts:test:corpus.sample:1.3", "id": "1.3.t1", "value": "chased", "role": "action", "related_node": None},
        ],
    }
    dspy.configure(lm=DummyLM([answer1, answer3]))

    real_analyze_with_retry = aat.english.analyze_with_retry

    def flaky_analyze_with_retry(passage, tokens, **kwargs):
        if "dog" in passage:
            raise RuntimeError("simulated LM failure")
        return real_analyze_with_retry(passage=passage, tokens=tokens, **kwargs)

    monkeypatch.setattr(aat.english, "analyze_with_retry", flaky_analyze_with_retry)

    summary = analyze_corpus(str(cex_path), str(out_dir))

    assert summary.clusters_total == 2  # "He waited." (1.1) + "The dog\nate." (1.2-1.3)
    assert summary.clusters_succeeded == 1
    assert summary.clusters_failed == 1
    assert len(summary.output_paths) == 1

    warnings_text = (out_dir / "warnings.txt").read_text(encoding="utf-8")
    assert "ERROR" in warnings_text
    assert "simulated LM failure" in warnings_text
    assert "Clusters analyzed: 1/2 (1 failed)" in warnings_text

    # The one cluster that DID succeed is still readable.
    tokens, graph = read_analysis(summary.output_paths[0])
    assert graph.nodes[0].value == "waited"


def test_analyze_corpus_rejects_an_unknown_lang(tmp_path):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_TWO_SENTENCE_CEX, encoding="utf-8")

    with pytest.raises(ValueError, match="fr"):
        analyze_corpus(str(cex_path), str(tmp_path / "out"), lang="fr")


def test_analyze_corpus_lang_nl_dispatches_to_the_dutch_package(tmp_path, monkeypatch):
    # Doesn't depend on any real Dutch tokenization/analysis detail --
    # spies on both language packages' own tokenize_corpus_by_sentence()
    # to confirm lang="nl" reaches aat.dutch's pipeline and NEVER
    # aat.english's, whatever the actual corpus content is.
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(
        "#!ctsdata\nurn:cts:test:corpus.sample:1.1|Hij las een boek.\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "out"

    calls = {"dutch": 0, "english": 0}
    real_dutch_cluster = aat.dutch.tokenize_corpus_by_sentence
    real_dutch_analyze = aat.dutch.analyze_with_retry
    real_dutch_validate = aat.dutch.validate

    def spy_cluster(units):
        calls["dutch"] += 1
        return real_dutch_cluster(units)

    def spy_analyze(passage, tokens, **kwargs):
        return real_dutch_analyze(passage=passage, tokens=tokens, **kwargs)

    def english_should_not_run(*args, **kwargs):
        calls["english"] += 1
        raise AssertionError("aat.english must not be used for lang='nl'")

    monkeypatch.setattr(aat.dutch, "tokenize_corpus_by_sentence", spy_cluster)
    monkeypatch.setattr(aat.english, "tokenize_corpus_by_sentence", english_should_not_run)
    monkeypatch.setattr(aat.english, "analyze_with_retry", english_should_not_run)

    dutch_answer = {
        "reasoning": "las is the action; Hij is the agent; boek is the target.",
        "nodes": [
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "t2", "value": "las", "role": "action", "related_node": None},
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "t1", "value": "Hij", "role": "agent", "related_node": "t2"},
            {"context": "urn:cts:test:corpus.sample:1.1", "id": "t4", "value": "boek", "role": "target", "related_node": "t2"},
        ],
    }
    dspy.configure(lm=DummyLM([dutch_answer]))

    summary = analyze_corpus(str(cex_path), str(out_dir), lang="nl")

    assert calls["dutch"] == 1
    assert calls["english"] == 0
    assert summary.clusters_succeeded == 1
    tokens, graph = read_analysis(summary.output_paths[0])
    assert graph.nodes[0].value == "las"


# -- analyze_corpus_w_diagrams() -----------------------------------------


def test_analyze_corpus_w_diagrams_missing_dot_raises_before_any_lm_call(tmp_path, monkeypatch):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_TWO_SENTENCE_CEX, encoding="utf-8")

    monkeypatch.setattr(shutil, "which", lambda name: None)
    lm = DummyLM([_ANSWER_DOG_ATE, _ANSWER_CAT_SAT])
    dspy.configure(lm=lm)

    with pytest.raises(RuntimeError, match="Graphviz"):
        analyze_corpus_w_diagrams(str(cex_path), str(tmp_path / "out"))

    assert lm.history == []  # no LM budget spent before the dot check


@pytest.mark.skipif(_DOT_MISSING, reason="needs the real Graphviz 'dot' CLI on PATH")
def test_analyze_corpus_w_diagrams_renders_a_real_png_per_cluster(tmp_path):
    cex_path = tmp_path / "corpus.cex"
    cex_path.write_text(_TWO_SENTENCE_CEX, encoding="utf-8")
    out_dir = tmp_path / "out"

    dspy.configure(lm=DummyLM([_ANSWER_DOG_ATE, _ANSWER_CAT_SAT]))
    summary = analyze_corpus_w_diagrams(str(cex_path), str(out_dir))

    assert len(summary.png_paths) == 2
    for p in summary.png_paths:
        from pathlib import Path

        data = Path(p).read_bytes()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"  # real PNG magic bytes
        assert Path(p).parent == out_dir / "pngs"

    # Text analyses are still written normally alongside the PNGs.
    assert len(summary.output_paths) == 2
