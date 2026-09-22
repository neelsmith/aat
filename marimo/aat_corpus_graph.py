import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Build an Agent-Action-Target graph from a CEX corpus
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    >**Prerequisites**: access to a LM configured in `.env` in the root of this repository (see [`.env.example`](https://github.com/neelsmith/aat/blob/main/.env.example) and
    [`USAGE.md`](https://github.com/neelsmith/aat/blob/main/USAGE.md)).

    *Browse to a corpus file in two-column CEX format (a CTS URN and that
    citation unit's own text per line, e.g. one row per verse), set its
    delimiter and how many citation units to analyze, then click "Analyze
    corpus". A sentence's grammar can cross a citation-unit boundary
    (Genesis 1:14-15 is a real example: verse 14 ends mid-clause with ':',
    verse 15 finishes the sentence), so citation units are first grouped
    into the smallest runs that end a sentence, and each group is
    analyzed as ONE passage -- not one LM call per citation unit. Every
    token's id stays unique by combining its own citation unit's passage
    reference with its position within that unit (e.g. "1.14.t3") -- see
    `aat/english/sentences.py` for the full reasoning. Once you have a
    graph, the diagram orientation control updates it live -- no need to
    re-analyze. You can also save the analysis (every citation unit read,
    plus the combined graph): pick a directory and click "Save analysis
    to file" -- the filename is derived from the corpus file's own name
    -- and reopen it later in `aat_reader.py`.*
    """)
    return


@app.cell(hide_code=True)
def _(corpus_file_browser):
    corpus_file_browser
    return


@app.cell(hide_code=True)
def _(delimiter_input, limit_input, mo):
    mo.hstack([delimiter_input, limit_input], justify="start")
    return


@app.cell(hide_code=True)
def _(analyze_button):
    analyze_button
    return


@app.cell(hide_code=True)
def _(load_error, mo):
    mo.callout(mo.md(load_error), kind="danger") if load_error else None
    return


@app.cell(hide_code=True)
def _(corpus_info, mo):
    mo.md(corpus_info) if corpus_info else None
    return


@app.cell(hide_code=True)
def _(htmlstack):
    htmlstack
    return


@app.cell(hide_code=True)
def _(showdiagram):
    showdiagram
    return


@app.cell(hide_code=True)
def _(orientation_input):
    orientation_input
    return


@app.cell(hide_code=True)
def _(costdisplay):
    costdisplay
    return


@app.cell(hide_code=True)
def _(mo, save_button, save_dir_browser):
    mo.vstack([save_dir_browser, save_button])
    return


@app.cell(hide_code=True)
def _(save_status):
    save_status
    return


@app.cell(hide_code=True)
def _(diagram_warnings, mo):
    mo.callout(mo.md("\n".join(f"- {w}" for w in diagram_warnings)), kind="warn") if diagram_warnings else None
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("<hr/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Implementation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Configuration
    """)
    return


@app.cell
def _():
    import os
    import sys
    from pathlib import Path

    import dspy
    from dotenv import load_dotenv

    return Path, dspy, load_dotenv, os, sys


@app.cell
def _(Path, sys):
    # So `import aat` resolves to this repo's own package, no matter what
    # directory `marimo edit`/`marimo run` was launched from -- same as
    # aat_graph.py.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    return


@app.cell
def _(Path, load_dotenv):
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    return


@app.cell
def _(os):
    def getenv(name: str, fallback_name: str, default: str | None = None) -> str | None:
        value = os.getenv(name)
        if value:
            return value
        value = os.getenv(fallback_name)
        if value:
            return value
        return default

    return (getenv,)


@app.cell
def _(DEFAULT_CEILING, dspy, getenv, os):
    def configure_lm():
        # Reuse an already-configured LM across reactive re-runs -- cheap
        # insurance if this cell itself is ever re-run by hand.
        if dspy.settings.lm is not None:
            return dspy.settings.lm

        api_base = getenv("API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm")
        model = getenv("MODEL", "MODEL", "litellm_proxy/anthropic/Claude Opus 5")

        # See aat_main.py's _configure_lm() for why this checks os.environ
        # directly: an empty API_KEY= is fine for a local model (e.g.
        # Ollama) that doesn't need one; a missing API_KEY line entirely is
        # likely an oversight.
        if "API_KEY" not in os.environ:
            raise RuntimeError(
                "Missing API key. Set API_KEY in your .env file (see "
                "../.env.example) -- an empty value is fine for a local "
                "model that doesn't need one."
            )
        api_key = os.environ["API_KEY"]

        # An explicit numeric baseline, not None (dspy.LM's own default) --
        # see aat_main.py's _configure_lm() for why: aat.english.token_budget.
        # analyze_with_retry() overrides max_tokens per call anyway, but
        # leaving this baseline at None made dspy's own truncation warning
        # misleadingly report max_tokens=None even when a real, larger
        # per-call budget had actually been used.
        lm_kwargs = dict(model=model, api_base=api_base, max_tokens=DEFAULT_CEILING)
        if api_key:
            lm_kwargs["api_key"] = api_key

        lm = dspy.LM(**lm_kwargs)
        dspy.configure(lm=lm)
        return lm

    return (configure_lm,)


@app.cell
def _(configure_lm):
    lm = configure_lm()
    return (lm,)


@app.cell
def _():
    from aat.core import graph_to_mermaid, read_cex_passages, write_analysis
    from aat.english import DEFAULT_CEILING, analyze_units_by_sentence, cluster_sentences, ends_sentence, tokens_to_html
    from aat.lm_cost import format_lm_cost, summarize_lm_cost

    return (
        DEFAULT_CEILING,
        analyze_units_by_sentence,
        cluster_sentences,
        ends_sentence,
        format_lm_cost,
        graph_to_mermaid,
        read_cex_passages,
        summarize_lm_cost,
        tokens_to_html,
        write_analysis,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(Path, mo):
    # selection_mode="file" -- browse to and pick exactly one corpus
    # file, same as aat_reader.py's own file_browser. Navigating between
    # directories doesn't itself change .value (only actually selecting
    # a file does), so nothing downstream re-runs just from browsing
    # around looking for the right file.
    corpus_file_browser = mo.ui.file_browser(
        initial_path=Path(__file__).parent.parent,
        selection_mode="file",
        multiple=False,
        label="*Choose a CEX corpus file*:",
    )
    return (corpus_file_browser,)


@app.cell
def _(mo):
    # CEX never declares its own delimiter inside the file (see
    # aat.core.cex's own docstring) -- '#' is the common convention and
    # this notebook's own default, but e.g.
    # scratch/eng-rv-vpl-genesis.cex uses '|' instead.
    delimiter_input = mo.ui.text(value="#", label="*Delimiter*:")
    return (delimiter_input,)


@app.cell
def _(mo):
    # A real safety valve, not a UI nicety: every sentence GROUP is its
    # own billed LM call (see the "See cost" checkbox below), and a
    # corpus like a whole book of Genesis is well over a thousand
    # citation units. Limits how many of the file's own citation units
    # (not sentence groups -- grouping happens after this cut) get read
    # at all; raise it once you know roughly what a run costs.
    limit_input = mo.ui.number(
        start=1,
        stop=100_000,
        step=1,
        value=20,
        label="*Analyze at most this many citation units*:",
    )
    return (limit_input,)


@app.cell
def _(mo):
    # Deliberately a separate run_button, not a form wrapping the file
    # browser -- the LM calls this triggers are real, billed work, so
    # nothing here should re-run just because the delimiter or limit
    # field changed; only an explicit click does. Same reasoning as
    # aat_graph.py's own save_button.
    analyze_button = mo.ui.run_button(label="Analyze corpus")
    return (analyze_button,)


@app.cell
def _(mo):
    # Deliberately NOT gated behind analyze_button -- orientation is a
    # rendering choice, not something that should require re-analyzing
    # (and re-billing) just to try a different layout. Changing it live
    # re-runs graph_to_mermaid() only.
    orientation_input = mo.ui.radio(
        options=["BT", "TB", "LR", "RL"],
        value="BT",
        inline=True,
        label="*Diagram orientation*:",
    )
    return (orientation_input,)


@app.cell
def _(mo):
    seecost = mo.ui.checkbox(label="*See cost*")
    seecost
    return (seecost,)


@app.cell
def _(Path, mo):
    # selection_mode="directory" + multiple=False -- pick exactly one
    # directory to save into; the filename itself is derived from the
    # corpus file's own name (see filename_base below), not typed here.
    save_dir_browser = mo.ui.file_browser(
        initial_path=Path(__file__).parent.parent,
        selection_mode="directory",
        multiple=False,
        label="*Save analysis in*:",
    )
    return (save_dir_browser,)


@app.cell
def _(mo):
    save_button = mo.ui.run_button(label="Save analysis to file")
    return (save_button,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Read the corpus and cluster into sentences
    """)
    return


@app.cell
def _(analyze_button, corpus_file_browser, delimiter_input, limit_input, read_cex_passages):
    # Only (re-)reads the file -- and only spends any LM budget -- when
    # analyze_button is actually clicked (mo.ui.run_button's value is
    # True for exactly the run triggered by that click, then resets to
    # False). Browsing to a different file, or changing the
    # delimiter/limit, does nothing on its own.
    units, load_error = [], None
    if analyze_button.value:
        if not corpus_file_browser.value:
            load_error = "Choose a corpus file first."
        else:
            path = corpus_file_browser.path(0)
            try:
                all_units = read_cex_passages(str(path), delimiter=delimiter_input.value or "#")
            except (OSError, ValueError) as exc:
                load_error = f"Couldn't read `{path}`: {exc}"
            else:
                units = all_units[: int(limit_input.value or 0)]
                if not units:
                    load_error = "Delimiter or limit left zero citation units to analyze."
    return load_error, units


@app.cell
def _(analyze_units_by_sentence, cluster_sentences, ends_sentence, units):
    # cluster_sentences() here is purely for the "N citation units, M
    # sentences" info line below -- analyze_units_by_sentence() below
    # re-derives the same grouping internally (it's a cheap, pure
    # function; recomputing it once more costs nothing, and keeps this
    # display-only info decoupled from analyze_units_by_sentence()'s own
    # return shape).
    tokens, graph = [], None
    sentence_count = 0
    incomplete_final_sentence = False
    if units:
        groups = cluster_sentences(units)
        sentence_count = len(groups)
        incomplete_final_sentence = bool(groups) and not ends_sentence(groups[-1][-1].text)
        tokens, graph = analyze_units_by_sentence(units)
    return graph, incomplete_final_sentence, sentence_count, tokens


@app.cell
def _(incomplete_final_sentence, sentence_count, units):
    corpus_info = None
    if units:
        corpus_info = f"*{len(units)} citation unit(s) read, grouped into {sentence_count} sentence(s).*"
        if incomplete_final_sentence:
            corpus_info += (
                " *(The last group doesn't end in sentence-final punctuation -- "
                "likely cut off by the citation-unit limit above, or the "
                "corpus's own last unit genuinely ends mid-sentence.)*"
            )
    return (corpus_info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Display
    """)
    return


@app.cell
def _(diagram, mo):
    showdiagram = None
    if diagram:
        showdiagram = mo.vstack([mo.md("**Graph**"), mo.mermaid(diagram)])
    return (showdiagram,)


@app.cell
def _(graph, mo, tokens, tokens_to_html):
    htmltext = None
    htmlhilite = None
    if graph:
        htmlhilite = mo.md(tokens_to_html(tokens, graph=graph))
        htmltext = mo.md("*" + tokens_to_html(tokens) + "*")
    return htmlhilite, htmltext


@app.cell
def _(htmlhilite, htmltext, mo):
    leftcol = mo.vstack([mo.md("**Text**"), htmltext])
    rightcol = mo.vstack([mo.md("**Analysis**"), htmlhilite])
    htmlstack = mo.hstack([leftcol, rightcol])
    return (htmlstack,)


@app.cell
def _(graph, graph_to_mermaid, orientation_input):
    diagram, diagram_warnings = None, []
    if graph is not None:
        diagram, diagram_warnings = graph_to_mermaid(graph, orientation=orientation_input.value)
    return diagram, diagram_warnings


@app.cell
def _(graph, lm, summarize_lm_cost):
    # `_ = graph` doesn't do anything with `graph` -- it exists purely so
    # marimo sees this cell as depending on it and re-runs the cell on
    # every new analysis. summarize_lm_cost() (aat.lm_cost) sums cost
    # across every call in lm.history -- every sentence group's own LM
    # call, not just the last -- and never raises on an empty history or
    # on a call served from dspy's own cache (cost=None).
    _ = graph
    cost_summary = summarize_lm_cost(lm.history)
    return (cost_summary,)


@app.cell
def _(cost_summary, format_lm_cost, mo, seecost):
    costdisplay = None
    if seecost.value:
        costdisplay = mo.md(f"**LM cost so far**: {format_lm_cost(cost_summary)}")
    return (costdisplay,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Serialization
    """)
    return


@app.cell
def _(corpus_file_browser):
    # Derived from the corpus file's own name, not any single citation
    # unit's context -- there's no one "the" context for a whole corpus.
    filename_base = "analysis"
    if corpus_file_browser.value:
        filename_base = corpus_file_browser.path(0).stem
    return (filename_base,)


@app.cell
def _(
    Path,
    filename_base,
    graph,
    mo,
    save_button,
    save_dir_browser,
    units,
    write_analysis,
):
    # write_analysis() gets `units` -- the ORIGINAL, per-citation-unit
    # passages exactly as read_cex_passages() returned them, not the
    # merged per-sentence text -- so the saved file's own '#!passages'
    # block round-trips through aat.english.sentences.
    # tokenize_corpus_by_sentence() later with no LM access needed, the
    # same reasoning aat.core.serialization's own docstring gives for
    # the single-passage case. (aat_reader.py itself doesn't yet know to
    # re-cluster on reload -- see this notebook's own session-log entry.)
    save_status = None
    if save_button.value:
        if graph is None or not units:
            save_status = mo.callout(
                mo.md("No analysis to save yet -- analyze a corpus first."), kind="warn"
            )
        else:
            save_dir = (
                save_dir_browser.path(0)
                if save_dir_browser.value
                else Path(__file__).parent.parent
            )
            save_path = Path(save_dir) / f"{filename_base}_analysis.txt"
            write_analysis(units, graph, str(save_path))
            save_status = mo.callout(
                mo.md(f"Saved analysis to `{save_path}`."), kind="success"
            )
    return (save_status,)


if __name__ == "__main__":
    app.run()
