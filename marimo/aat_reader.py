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
    # Display a saved Agent-Action-Target analysis
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    *Browse to and select a file previously saved with `aat_graph.py`'s
    "Save analysis to file" button (or written directly with
    `aat.core.write_analysis()`). The passage is re-tokenized
    (`aat.english.tokenize` -- deterministic, no LM call) and paired back
    up with the graph the file already has, so this notebook needs no `.env`,
    no configured LM, and makes no network access at all -- everything it
    shows comes straight from the file. Once loaded, the diagram
    orientation control updates it live.*
    """)
    return


@app.cell(hide_code=True)
def _(file_browser):
    file_browser
    return


@app.cell(hide_code=True)
def _(load_error, mo):
    mo.callout(mo.md(load_error), kind="danger") if load_error else None
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
    import sys
    from pathlib import Path

    return Path, sys


@app.cell
def _(Path, sys):
    # So `import aat` resolves to this repo's own package, no matter what
    # directory `marimo edit`/`marimo run` was launched from -- same as
    # aat_graph.py. No dotenv, no dspy.LM, no API key: this notebook never
    # calls an LM, so none of that configuration is needed here at all.
    sys.path.insert(0, str(Path(__file__).parent.parent))
    return


@app.cell
def _():
    from aat.core import graph_to_mermaid, read_analysis
    from aat.english import tokenize, tokens_to_html

    return graph_to_mermaid, read_analysis, tokenize, tokens_to_html


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## UI
    """)
    return


@app.cell
def _(Path, mo):
    # selection_mode="file" -- the user browses to and picks exactly one
    # analysis file, rather than typing its path. Unlike a text field,
    # navigating between directories doesn't itself change .value (only
    # actually selecting a file does), so no submit-button/form wrapper
    # is needed to avoid re-reading the file on every click -- the load
    # cell below just reacts directly to a real selection.
    file_browser = mo.ui.file_browser(
        initial_path=Path(__file__).parent.parent,
        selection_mode="file",
        multiple=False,
        label="*Choose a saved analysis file*:",
    )
    return (file_browser,)


@app.cell
def _(mo):
    # Deliberately separate from file_browser above -- orientation is a
    # rendering choice, not something that should require re-selecting
    # the file just to try a different layout. Changing it live re-runs
    # graph_to_mermaid() only, same as aat_graph.py's own orientation_input.
    orientation_input = mo.ui.radio(
        options=["BT", "TB", "LR", "RL"],
        value="BT",
        inline=True,
        label="*Diagram orientation*:",
    )
    return (orientation_input,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Read the file and re-tokenize its passage(s)
    """)
    return


@app.cell
def _(file_browser, read_analysis, tokenize):
    # Only (re-)reads the file once a selection has actually been made
    # (file_browser.value is empty until then), and only again when the
    # selection changes -- not on every directory the user browses
    # through on the way there.
    #
    # A file can hold more than one '#!passages' row (see
    # aat.core.serialization's own docstring); every passage found is
    # re-tokenized and the token lists concatenated in file order, so a
    # single-passage file (the common case -- what aat_graph.py's own
    # "Save analysis to file" button writes) round-trips exactly like the
    # original analyze_passage() call did. read_analysis() itself never
    # touches an LM -- it just parses the file -- and neither does
    # tokenize(), so nothing in this cell can make a network call.
    tokens, graph, load_error = [], None, None
    if file_browser.value:
        path = file_browser.path(0)
        try:
            passages, graph = read_analysis(str(path))
        except (OSError, ValueError) as exc:
            graph = None
            load_error = f"Couldn't load `{path}`: {exc}"
        else:
            tokens = []
            for passage in passages:
                tokens.extend(tokenize(passage))
    return graph, load_error, tokens


@app.cell
def _(graph, graph_to_mermaid, orientation_input):
    diagram, diagram_warnings = None, []
    if graph is not None:
        diagram, diagram_warnings = graph_to_mermaid(graph, orientation=orientation_input.value)
    return diagram, diagram_warnings


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


if __name__ == "__main__":
    app.run()
