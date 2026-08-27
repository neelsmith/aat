# aat -- Usage Guide

A dspy program that analyzes an English passage into an Agent-Action-Target graph. The analytic scheme itself is documented in `aat-model.md`; the English-specific extraction rules are documented in `AgentActionTarget`'s own docstring (`aat/english/dspy_signatures.py`).


## Running an analysis from the command line

You can run an analysis from the command line with the wrapper script `aat_main.py`. It needs an `.env` file in this folder with your LM credentials -- copy `.env.example` to `.env` and fill in real values:

```
API_BASE=https://localmodel/api
MODEL=litellm/modelname
API_KEY=your-key-here
```

Then:

```bash
python3 aat_main.py --passage "The dog ate my homework."
```

`--context` is an optional argument giving a context reference for the passage (e.g. a CTS URN), recorded on every resulting token via `CitableToken.context`; it defaults to an empty string if omitted:

```bash
python3 aat_main.py --passage "The homework was eaten by the dog." --context "urn:cite2:aat:examples.v1:ex1"
```

`aat_main.py` reads `API_BASE`/`MODEL`/`API_KEY` from `.env`, configures the LM, and prints the resulting tokens and AAT nodes.

For a local, unauthenticated model (e.g. Ollama), leave `API_KEY` present but empty:

```
API_BASE=http://localhost:11434
MODEL=ollama_chat/llama3
API_KEY=
```

`aat_main.py` only raises "Missing API key" when `API_KEY` isn't in `.env` at all; an empty value is treated as "this model doesn't need one" and is left out of the LM call entirely, rather than sent through as an empty credential.


## Using `aat` in a script

To call the pipeline from your own script or a REPL instead of the CLI, configure a `dspy.LM` yourself and use `aat.english` directly:

```python
import dspy
from aat.english import analyze_passage

dspy.configure(lm=dspy.LM(model="litellm_proxy/anthropic/Claude Opus 5",
                           api_base="https://api_url/litellm",
                           api_key="your-key-here"))

tokens, graph = analyze_passage("The dog ate my homework.")

for action in graph.actions():
    agents = graph.agents_for(action)
    targets = graph.targets_for(action)
    print(f"{action.value!r} (independent={action.related_node is None})")
    for a in agents:
        print(f"  agent:  {a.value!r}")
    for t in targets:
        print(f"  target: {t.value!r}")
```

Explanation:

- `analyze_passage()` returns `(tokens, graph)`: `tokens` is the passage's `CitableToken` list (from `aat.english.tokenize`), `graph` is an `AATGraph` -- a flat, unordered list of `AATNode` under the hood, but with `actions()`/`agents()`/`targets()`/`agents_for()`/`targets_for()`/`governing_action()` convenience accessors (see `aat/core/graph.py`).
- `analyze_passage()` also prints a warning if the LM's output fails `validate()` -- e.g. it refers to a token id that doesn't exist in the input tokens, or an agent/target node with no `related_node`. That's a sign the output needs a re-run or a prompt tweak, not necessarily that your code is broken; see `aat.core.validate.validate`'s own docstring for exactly what it checks (referential integrity only, never linguistic correctness).


## Analyzing multiple citable passages

`analyze_passage()` is a convenience wrapper around `analyze_passages()`, which takes a list of `CitedPassage` (context + text) and returns one combined `(tokens, graph)`:

```python
from aat.core import CitedPassage
from aat.english import analyze_passages

passages = [
    CitedPassage(context="urn:cite2:aat:examples.v1:ex1", text="The dog ate my homework."),
    CitedPassage(context="urn:cite2:aat:examples.v1:ex2", text="The homework was eaten by the dog."),
]
tokens, graph = analyze_passages(passages)
```

Each passage's own tokens are numbered from `t1` within its own context (`CitableToken.id` is only unique *within* one context, not globally -- see its docstring), so `graph.by_id(context, id)` always needs both.


## Saving and loading a graph

`write_nodes()`/`read_nodes()` (in `aat/core/serialization.py`) save and reload an `AATGraph`'s nodes as one deterministic, pipe-delimited plain-text file, so you can persist an analysis, diff it, hand-edit it, or reload it later without re-running the LM:

```python
from aat.core import write_nodes, read_graph

write_nodes(graph.nodes, "analysis.txt")
reloaded = read_graph("analysis.txt")
```

The file has one `#!aatnodes` block per call to `write_nodes()`/`serialize_nodes()`, each with the fixed header `context|id|value|role|related_node` -- see `serialization.py`'s module docstring for the exact format. Multiple blocks in one file are concatenated, in file order, into the list `read_nodes()` returns, so simply concatenating several `write_nodes()` outputs together and reading the result back gives you one combined graph.


## Rendering a graph as Mermaid

`graph_to_mermaid()` (in `aat/core/mermaid.py`) renders an `AATGraph` as a [Mermaid](https://mermaid.js.org) flowchart: an action is a rectangle, an agent is rounded, a target is a stadium shape, and every node with a `related_node` becomes a labelled edge pointing at it. By default every node is also colored by which action it clusters with, so the separate clauses in a multi-action passage are visually distinguishable.

```python
from aat.core import graph_to_mermaid

diagram, warnings = graph_to_mermaid(graph)
print(diagram)
for w in warnings:
    print(f"Warning: {w}")
```

`orientation` controls the diagram's layout direction -- Mermaid's own flowchart direction codes: `"BT"` (bottom-to-top, the default), `"TB"`/`"TD"` (top-down, synonyms), `"LR"`, or `"RL"`. Matched case-insensitively (`"lr"` works the same as `"LR"`); anything else raises `ValueError` naming the valid options, rather than silently producing invalid Mermaid syntax:

```python
diagram, warnings = graph_to_mermaid(graph, orientation="LR")
```

Pass `color_by_action=False` for a plain, uncolored diagram. `save_mermaid(graph, path, ...)` takes the same `orientation`/`color_by_action` arguments and writes the diagram straight to a file (e.g. `analysis.mmd`).

`warnings` lists any node whose `related_node` doesn't resolve to another node actually present in `graph` -- normally a sign the graph failed `validate()` upstream (see "Analyzing multiple citable passages" above), worth checking there first -- plus, if the graph has more distinct actions than the color palette has slots (currently 8), one warning that colors repeat.


## Rendering tokens as highlighted HTML

`tokens_to_html()` (in `aat/english/html.py`) renders a passage's tokens as one continuous HTML string, reconstructing normal reading spacing (punctuation attaches to the preceding word; opening brackets and the first of a paired quote attach to what follows) rather than putting a space before every token. Pass the same `AATGraph` you'd hand to `graph_to_mermaid()` and every token that's also an AAT graph node is highlighted using the *same* color that node gets in the Mermaid diagram (`aat.core.coloring.assign_action_colors()` -- one shared assignment behind both renderers), with a border style keyed on the node's role: a box around an `action` token, a rounded box around an `agent` token, and an underline under a `target` token.

```python
from aat.english import tokenize, tokens_to_html
from aat.core import CitedPassage

tokens, graph = analyze_passage("The dog ate the homework.", context="urn:cts:...")
html = tokens_to_html(tokens, graph=graph)
```

Omit `graph` (or pass `graph=None`) for plain, unhighlighted text -- still with the same spacing reconstruction.

Every token's text is HTML-escaped before being emitted (`&`, `<`, `>`, and quote characters), so passage text containing any of those characters round-trips safely rather than being mistaken for markup.

Note: for a *compound* action (e.g. "was eating"), only the principal-verb token is highlighted, since that's the only token id the `AATNode` itself records -- see the module's own docstring.


## Interactive notebook

`marimo/aat_graph.py` is a [marimo](https://marimo.io) notebook: enter a context ID and a passage in one form, submit it, and it tokenizes the passage, runs it through `analyze_passage()`, and renders the resulting `AATGraph` as a Mermaid diagram via `aat.core.graph_to_mermaid()`. A separate orientation control (default `BT`) updates the diagram live, without resubmitting the form or making another LM call. Needs the 'dev' extra (`pip install -e ".[dev]"`) and a working `.env` (see above) -- the LM is configured as soon as the notebook loads.

```bash
marimo edit marimo/aat_graph.py
```

opens it in an editable, reactive browser session; `marimo run marimo/aat_graph.py` runs the same notebook as a read-only app (code cells hidden, just the form and the diagram).


## Using an optimized prompt

If you've run `optimize_gepa.py` (see OPTIMIZING.md) and saved an optimized program, load it into `analyze` before calling `analyze_passage()`/`analyze_passages()`:

```python
from aat.english.dspy_signatures import analyze
analyze.load("optimized_agent_action_target.json")
```

`analyze` is the same module-level `dspy.ChainOfThought` instance the whole pipeline uses, so loading into it in place is enough -- nothing else needs to change.
