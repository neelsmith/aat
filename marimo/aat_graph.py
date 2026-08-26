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

    Enter a context reference and a passage of English text, then submit
    the form. The passage is tokenized (`aat.english.tokenize`), analyzed
    into agent/action/target nodes (`aat.english.analyze_passage`), and
    rendered as a Mermaid diagram (`aat.core.graph_to_mermaid`).

    Needs a working `.env` in the repo root (see `../.env.example` and
    `USAGE.md`) -- the LM is configured as soon as this notebook loads,
    before you submit anything.
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
    from aat.core import graph_to_mermaid
    from aat.english import analyze_passage

    return analyze_passage, graph_to_mermaid


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Enter a passage
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


@app.cell(hide_code=True)
def _(passage_form):
    passage_form
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## AAT graph
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
    return (graph,)


@app.cell
def _(graph, graph_to_mermaid):
    diagram, diagram_warnings = None, []
    if graph is not None:
        diagram, diagram_warnings = graph_to_mermaid(graph)
    return diagram, diagram_warnings


@app.cell(hide_code=True)
def _(diagram, mo):
    mo.mermaid(diagram) if diagram else mo.md("*Submit the form above to build a diagram.*")
    return


@app.cell(hide_code=True)
def _(diagram_warnings, mo):
    mo.callout(mo.md("\n".join(f"- {w}" for w in diagram_warnings)), kind="warn") if diagram_warnings else None
    return


if __name__ == "__main__":
    app.run()
