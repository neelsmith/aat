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
    # Build an Agent-Action-Target graph
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    >**Prerequisites**: access to a LM configured in`.env` in the root of this repository (see [`.env.example`](https://github.com/neelsmith/aat/blob/main/.env.example) and
    [`USAGE.md`](https://github.com/neelsmith/aat/blob/main/USAGE.md)).


    *Choose a language, enter a context reference, and a passage of text
    in that language. Once you have a graph, the diagram orientation
    control updates it live -- no need to resubmit the form. Switching
    languages also requires resubmitting the form, since it changes which
    extraction model actually runs. You can also save the analysis
    (passage + graph): pick a directory and click "Save analysis to file"
    -- the filename is derived automatically from the context ID -- and
    reopen it later in `aat_reader.py`, which replicates this same
    display but needs no LM access at all.*
    """)
    return


@app.cell(hide_code=True)
def _(passage_form):
    passage_form
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
    # directory `marimo edit`/`marimo run` was launched from.
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
def _(LM_MAX_TOKENS_CEILING, dspy, getenv, os):
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
        # see aat_main.py's _configure_lm() for why: whichever language's
        # own token_budget.analyze_with_retry() ends up running overrides
        # max_tokens per call anyway, but leaving this baseline at None
        # made dspy's own truncation warning misleadingly report
        # max_tokens=None even when a real, larger per-call budget had
        # actually been used. LM_MAX_TOKENS_CEILING is the max across every
        # language module's own DEFAULT_CEILING (see above), so this stays
        # correct no matter which language ends up selected.
        lm_kwargs = dict(model=model, api_base=api_base, max_tokens=LM_MAX_TOKENS_CEILING)
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
    import aat.dutch
    import aat.english
    from aat.core import graph_to_mermaid, tokens_to_html, write_analysis
    from aat.lm_cost import format_lm_cost, summarize_lm_cost

    # Every language module this notebook can drive analyze_passage()
    # through -- add a new language here (and nowhere else in this cell)
    # once it has its own aat.<language> package mirroring aat.english's
    # shape, and it picks up the radio button option automatically.
    LANGUAGE_MODULES = {"English": aat.english, "Dutch": aat.dutch}

    return (
        LANGUAGE_MODULES,
        format_lm_cost,
        graph_to_mermaid,
        summarize_lm_cost,
        tokens_to_html,
        write_analysis,
    )


@app.cell
def _(LANGUAGE_MODULES):
    # A single numeric baseline for dspy.LM's own max_tokens, used only to
    # construct the LM once (see configure_lm below) -- every actual
    # analyze() call overrides max_tokens per call via analyze_with_retry(),
    # regardless of which language's token_budget module is doing the
    # retrying. Taking the max across every language module here just means
    # this baseline never undersells whichever language ends up selected;
    # it has no effect on per-call behavior.
    LM_MAX_TOKENS_CEILING = max(module.DEFAULT_CEILING for module in LANGUAGE_MODULES.values())
    return (LM_MAX_TOKENS_CEILING,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(mo):
    context_input = mo.ui.text(
        placeholder="urn:cts:aat:examples.gettysburg.hay:1",
        label="*Context ID*:",
    )
    return (context_input,)


@app.cell
def _(mo):
    passage_input = mo.ui.text_area(
        value="Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
        full_width=True,
        label="*Passage*:",
    )
    return (passage_input,)


@app.cell
def _(LANGUAGE_MODULES, mo):
    # Which language module (aat.english vs aat.dutch) analyze_passage()
    # below actually calls -- part of passage_form's own batch (see below),
    # not a live control like orientation_input, since changing it changes
    # which LM signature runs and so needs a real re-submission, not just a
    # re-render.
    language_input = mo.ui.radio(
        options=list(LANGUAGE_MODULES.keys()),
        value="English",
        inline=True,
        label="*Language*:",
    )
    return (language_input,)


@app.cell
def _(mo):
    # Deliberately NOT part of passage_form's batch() below -- orientation
    # is a rendering choice, not something that should require resubmitting
    # the form (and re-running analyze_passage() against the LM) just to
    # try a different layout. Changing it live re-runs graph_to_mermaid()
    # only.
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
    # selection_mode="directory" + multiple=False -- the user picks
    # exactly one directory to save into; the filename itself is derived
    # from the passage's own context (see filename_base below), not
    # typed here. Starts browsing from the repo root, and .value stays
    # empty until the user actually picks something -- see save_status
    # below for the fallback that applies until then.
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


@app.cell
def _(context_input, language_input, mo, passage_input):
    # All three inputs as one form -- marimo only updates passage_form.value
    # (and so only re-triggers the analysis cell below) when the form is
    # submitted, never on every keystroke in a text field or every click of
    # the language radio. That's deliberate for language_input too, not
    # just the text fields: switching languages changes which LM signature
    # analyze_passage() below actually calls, so it should require the same
    # explicit re-submission a new passage would, not take effect live.
    passage_form = (
        mo.md(
            """
            {language_input}

            {context_input}

            {passage_input}
            """
        )
        .batch(
            context_input=context_input,
            language_input=language_input,
            passage_input=passage_input,
        )
        .form(submit_button_label="Build AAT graph")
    )
    return (passage_form,)


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
        htmltext = mo.md("*" + tokens_to_html(tokens) + "*" )
    return htmlhilite, htmltext


@app.cell
def _(htmlhilite, htmltext, mo):
    leftcol = mo.vstack([mo.md("**Text**"), htmltext])
    rightcol = mo.vstack([mo.md("**Analysis**"), htmlhilite])
    htmlstack = mo.hstack([leftcol, rightcol])
    return (htmlstack,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Serialization
    """)
    return


@app.cell
def _(tokens):
    # A safe filename base derived from the passage's own context
    # reference (e.g. a CTS/CITE URN like "urn:cite2:aat:examples.v1:ex1",
    # full of ':' and '.') -- taken from the first token's own context
    # (every token in `tokens` shares one context here, since this
    # notebook analyzes a single passage at a time). Every run of
    # characters that isn't a letter, digit, '_', or '-' collapses to a
    # single '_', with leading/trailing '_' stripped. Falls back to
    # "analysis" if that leaves nothing (e.g. no context was given, or
    # no analysis has run yet).
    filename_base = "analysis"
    if tokens:
        slug = "".join(c if (c.isalnum() or c in "_-") else "_" for c in tokens[0].context)
        slug = slug.strip("_")
        filename_base = slug or "analysis"
    return (filename_base,)


@app.cell
def _(
    Path,
    filename_base,
    graph,
    mo,
    save_button,
    save_dir_browser,
    tokens,
    write_analysis,
):
    # Only runs (writes a file) when save_button is actually clicked --
    # mo.ui.run_button's value is True for exactly the run triggered by
    # that click, then resets to False, so this cell is a no-op on every
    # other reactive re-run (e.g. re-submitting the form, changing the
    # orientation control, or just browsing to a different directory
    # without clicking Save). write_analysis() gets `tokens` directly --
    # the exact CitableToken list analyze_passage() produced, not a raw
    # passage to be re-tokenized later -- so the saved file's own
    # '#!tokens' block already IS the complete input, no re-derivation
    # needed on reload (see aat.core.serialization's own module
    # docstring).
    save_status = None
    if save_button.value:
        if graph is None or not tokens:
            save_status = mo.callout(
                mo.md("No analysis to save yet -- build a graph first."), kind="warn"
            )
        else:
            # save_dir_browser.value is empty until the user actually
            # picks a directory -- default to this notebook's own parent
            # directory (the repo root) rather than erroring.
            save_dir = (
                save_dir_browser.path(0)
                if save_dir_browser.value
                else Path(__file__).parent.parent
            )
            save_path = Path(save_dir) / f"{filename_base}.txt"
            write_analysis(tokens, graph, str(save_path))
            save_status = mo.callout(
                mo.md(f"Saved analysis to `{save_path}`."), kind="success"
            )
    return (save_status,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Tokenize text and build graph
    """)
    return


@app.cell
def _(LANGUAGE_MODULES, passage_form):
    # Tokenize and analyze -- only once the form has been submitted at
    # least once (passage_form.value is None until then), and only again on
    # each subsequent submission, not on every keystroke or radio click in
    # the form's own inputs. The language picked in the form (default
    # "English", since language_input's own value defaults to that even
    # before a first submission) selects which language module's
    # analyze_passage() actually runs -- each has its own dspy signature,
    # tokenizer, and token-budget calibration.
    tokens, graph = [], None
    if passage_form.value and passage_form.value.get("passage_input"):
        context = passage_form.value.get("context_input") or ""
        text = passage_form.value["passage_input"]
        language = passage_form.value.get("language_input") or "English"
        analyze_passage = LANGUAGE_MODULES[language].analyze_passage
        tokens, graph = analyze_passage(text, context=context)
    return graph, tokens


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
    # across every call in lm.history, not just the last one, and never
    # raises on an empty history (true before the form's first
    # submission) or on a call served from dspy's own cache (cost=None)
    # -- see that module's own docstring.
    _ = graph
    cost_summary = summarize_lm_cost(lm.history)
    return (cost_summary,)


@app.cell
def _(cost_summary, format_lm_cost, mo, seecost):
    costdisplay = None
    if seecost.value:
        costdisplay = mo.md(f"**LM cost so far**: {format_lm_cost(cost_summary)}")
    return (costdisplay,)


if __name__ == "__main__":
    app.run()
