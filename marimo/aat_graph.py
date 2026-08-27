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


    *Enter a context reference and a passage of English text. Once you have
    a graph, the diagram orientation control updates it live -- no need to
    resubmit the form. You can also save the analysis (passage + graph):
    pick a directory and click "Save analysis to file" -- the filename is
    derived automatically from the context ID -- and reopen it later in
    `aat_reader.py`, which replicates this same display but needs
    no LM access at all.*
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
def _(dspy, getenv, os):
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

        lm_kwargs = dict(model=model, api_base=api_base)
        if api_key:
            lm_kwargs["api_key"] = api_key

        lm = dspy.LM(**lm_kwargs)
        dspy.configure(lm=lm)
        return lm

    return (configure_lm,)


@app.cell
def _(configure_lm):
    lm = configure_lm()
    return


@app.cell
def _():
    from aat.core import CitedPassage, graph_to_mermaid, write_analysis
    from aat.english import analyze_passage, tokens_to_html

    return (
        CitedPassage,
        analyze_passage,
        graph_to_mermaid,
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
def _(mo):
    context_input = mo.ui.text(
        placeholder="urn:cite2:aat:examples.v1:ex1",
        label="*Context ID*:",
    )
    return (context_input,)


@app.cell
def _(mo):
    passage_input = mo.ui.text_area(
        value="The dog ate my homework.",
        full_width=True,
        label="*Passage*:",
    )
    return (passage_input,)


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
def _(context_input, mo, passage_input):
    # Both inputs as one form -- marimo only updates passage_form.value (and
    # so only re-triggers the analysis cell below) when the form is
    # submitted, never on every keystroke in either field.
    passage_form = (
        mo.md(
            """
            {context_input}

            {passage_input}
            """
        )
        .batch(context_input=context_input, passage_input=passage_input)
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
def _(CitedPassage, passage_form):
    # The passage last submitted, kept as a CitedPassage so it can be
    # written back out (write_analysis()) alongside its graph -- built
    # straight from the form's own value, so it's available for saving
    # even independent of how the analysis cell below is implemented.
    passage_for_save = None
    if passage_form.value and passage_form.value.get("passage_input"):
        passage_for_save = CitedPassage(
            context=passage_form.value.get("context_input") or "",
            text=passage_form.value["passage_input"],
        )
    return (passage_for_save,)


@app.cell
def _(passage_for_save):
    # A safe filename base derived from the passage's own context
    # reference (e.g. a CTS/CITE URN like "urn:cite2:aat:examples.v1:ex1",
    # full of ':' and '.') -- every run of characters that isn't a
    # letter, digit, '_', or '-' collapses to a single '_', with
    # leading/trailing '_' stripped. Falls back to "analysis" if that
    # leaves nothing (e.g. no context was given).
    filename_base = "analysis"
    if passage_for_save is not None:
        slug = "".join(
            c if (c.isalnum() or c in "_-") else "_" for c in passage_for_save.context
        )
        slug = slug.strip("_")
        filename_base = slug or "analysis"
    return (filename_base,)


@app.cell
def _(
    Path,
    filename_base,
    graph,
    mo,
    passage_for_save,
    save_button,
    save_dir_browser,
    write_analysis,
):
    # Only runs (writes a file) when save_button is actually clicked --
    # mo.ui.run_button's value is True for exactly the run triggered by
    # that click, then resets to False, so this cell is a no-op on every
    # other reactive re-run (e.g. re-submitting the form, changing the
    # orientation control, or just browsing to a different directory
    # without clicking Save).
    save_status = None
    if save_button.value:
        if graph is None or passage_for_save is None:
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
            write_analysis([passage_for_save], graph, str(save_path))
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
def _(analyze_passage, passage_form):
    # Tokenize and analyze -- only once the form has been submitted at
    # least once (passage_form.value is None until then), and only again on
    # each subsequent submission, not on every keystroke in the form's own
    # inputs.
    tokens, graph = [], None
    if passage_form.value and passage_form.value.get("passage_input"):
        context = passage_form.value.get("context_input") or ""
        text = passage_form.value["passage_input"]
        tokens, graph = analyze_passage(text, context=context)
    return graph, tokens


@app.cell
def _(graph, graph_to_mermaid, orientation_input):
    diagram, diagram_warnings = None, []
    if graph is not None:
        diagram, diagram_warnings = graph_to_mermaid(graph, orientation=orientation_input.value)
    return diagram, diagram_warnings


if __name__ == "__main__":
    app.run()
