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


## Interactive notebook

`marimo/aat_graph.py` is a [marimo](https://marimo.io) notebook: enter a context ID and a passage in one form, submit it, and it tokenizes the passage, runs it through `analyze_passage()`, and renders the resulting `AATGraph` as a Mermaid diagram via `aat.core.graph_to_mermaid()`. Needs the 'dev' extra (`pip install -e ".[dev]"`) and a working `.env` (see above) -- the LM is configured as soon as the notebook loads.

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
