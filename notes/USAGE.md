# Usage Guide



## Running an analysis from the command line

*Prerequisite*: an `.env` file in this folder with your LM credentials. Copy `.env.example` to `.env` and fill in real values: 

```
API_BASE=https://localmodel/api
MODEL=litellm/modelname
API_KEY=your-key-here
```

Run an analysis from the command line:

```bash
python3 aat_main.py --passage "The dog ate my homework."
```

Include a citable reference for the passage (such as a CTS URN) with the `--context` argument:

```bash
python3 aat_main.py --passage "The homework was eaten by the dog." --context "urn:cite2:aat:examples.v1:ex1"
```

`aat_main.py` writes the analysis to stdout as a plain-text serialized analysis -- the same `#!tokens`/`#!aatnodes` format `aat.core.serialize_analysis()`/`write_analysis()` produce (see "Saving and loading a graph" below), and nothing else -- so it can be redirected straight to a file and reloaded later, with no separate save step:

```bash
python3 aat_main.py --passage "The homework was eaten by the dog." --context "urn:cite2:aat:examples.v1:ex1" > analysis.txt
```

```python
from aat.core import read_analysis

passages, graph = read_analysis("analysis.txt")
```

(A referential problem `validate()` catches along the way is reported on *stderr*, not stdout, so it never corrupts the redirected file -- see "Analyzing multiple citable passages" below.)

Pipe that output straight into `aat_to_dot.py` to render it as a Graphviz digraph without a second LM call -- see "Rendering a graph as Graphviz dot" below.


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

## Analyzing a whole corpus from a CEX file

`aat_corpus.py` is the corpus-level version of `aat_main.py`: it reads every passage from a [CEX (CITE Exchange)](https://cite-architecture.github.io/citedx/CEX-spec-3.0.1/) file's `#!ctsdata` block -- an external plain-text interchange format, not this project's own `#!tokens`/`#!aatnodes` serialization -- analyzes all of them, and writes ONE combined serialized analysis to stdout, in the same `#!tokens`/`#!aatnodes` format `aat_main.py` uses for a single passage:

```bash
python3 aat_corpus.py corpus.cex > analysis.txt
```

A `#!ctsdata` row is two fields -- a CTS URN, then that node's own text -- separated by a delimiter the file's own author chose; CEX never declares its delimiter inside the file itself. `"#"` is the common convention and this script's own default:

```
#!ctsdata
urn:cite2:aat:examples.v1:ex1#The dog ate my homework.
urn:cite2:aat:examples.v1:ex2#The homework was eaten by the dog.
```

Pass `--delimiter` if a particular corpus uses something else (e.g. `--delimiter "|"`). Every other CEX block type (`#!citelibrary`, `#!ctscatalog`, ...) is ignored, so a full CEX file -- not just a bare `#!ctsdata` block -- works as input; `aat.core.cex.read_cex_passages()`/`parse_cex_ctsdata()` are the underlying functions, if you want to read a CEX corpus into a list of `CitedPassage` yourself without also running the LM pipeline. Use `-` instead of a filename to read the same format from stdin:

```bash
cat corpus.cex | python3 aat_corpus.py - > analysis.txt
```

Every passage gets its own separate LM call (via `analyze_passages()`), so a large corpus means real API cost and real wall-clock time -- there's no batching or parallelism. As with `aat_main.py`, any referential problem `validate()` catches is reported on stderr, never stdout, so it never corrupts the redirected file; and the output pipes straight into `aat_to_dot.py` (see "Rendering a graph as Graphviz dot" below) exactly like `aat_main.py`'s does.

`aat_corpus.py` analyzes every citation unit independently, which is wrong whenever a sentence's own grammar crosses a citation-unit boundary -- a real example from `scratch/eng-rv-vpl-genesis.cex`: Genesis 1:14 ends mid-clause with `:`, and the sentence only completes in 1:15. See the next section for the alternative that handles this.

## Analyzing a corpus by sentence, across citation-unit boundaries

`aat.english.sentences` groups an ordered list of citation units into the smallest runs that each end a sentence (`cluster_sentences()`, using `.`/`?`/`!` as sentence-final punctuation by default -- pass a different `terminators` string if a corpus needs a different rule), then tokenizes each group as one combined passage (`tokenize_units()`) rather than tokenizing each citation unit on its own. Every token's id stays unique within the combined group by combining its own citation unit's CTS passage component with its position within that unit -- e.g. `"1.14.t3"` -- and the combined group's own context is a CTS range reference (`urn:cts:compnov:bible.genesis.rvvpl:1.14-1.15` for a two-unit group, or just `...:1.14` for a one-unit group). `aat.english.pipeline.analyze_units_by_sentence(units)` is the LM-dependent counterpart that runs each group through `analyze_with_retry()`/`validate()` (see "Managing the LM's output token budget" below for what `analyze_with_retry()` adds over calling `analyze()` directly) and returns `(tokens, graph)` in the same shape `analyze_passages()` does:

```python
from aat.core import read_cex_passages, write_analysis
from aat.english import analyze_units_by_sentence

units = read_cex_passages("scratch/eng-rv-vpl-genesis.cex", delimiter="|")
tokens, graph = analyze_units_by_sentence(units)

# Save the tokens analyze_units_by_sentence() actually produced --
# composite sentence-spanning ids (e.g. "1.14.t3") and all -- not the
# original per-citation-unit `units`. See "Saving and loading a graph"
# above for why: the saved file's own '#!tokens' block already IS the
# resolved token list, so reloading it needs no re-run of this module
# at all.
write_analysis(tokens, graph, "analysis.txt")
```

Every function in `aat.english.sentences` is pure and LM-free, deterministic given the same ordered citation units -- useful if you want to reproduce a particular grouping or composite id scheme yourself, but *not* something a caller reloading a saved file needs to rely on: `write_analysis()` saves the actual resolved `tokens` (see "Saving and loading a graph" above), so `marimo/aat_reader.py` (or any other `read_analysis()` caller) gets the exact composite ids `analyze_units_by_sentence()` assigned straight back from the file, with no re-clustering and no dependency on this module at all.


## Managing the LM's output token budget

Both `analyze_passages()` (and its `analyze_passage()` wrapper) and `analyze_units_by_sentence()` call `aat.english.token_budget.analyze_with_retry()` rather than `aat.english.analyze()` directly. A passage's `reasoning` field plus its `nodes` list grows with how long and syntactically complex the passage is -- and, for a sentence group spanning several citation units, "how long" isn't bounded by any single citation unit's own length either -- so a fixed `max_tokens` is eventually wrong: too small and a real passage gets truncated mid-response (a `dspy.utils.exceptions.AdapterParseError`, or in rarer cases a response that parses but whose `finish_reason` says `"length"` anyway); too large and every call wastes part of its budget.

`analyze_with_retry()` estimates a starting budget from the passage's own token count (`estimate_max_tokens()`, a simple linear fit -- untuned by default, deliberately generous so it overestimates rather than truncates; see `utilities/calibrate_max_tokens.py` below to replace that fit with a real measurement) and, if a call still comes back truncated, retries with a larger budget instead of surfacing the raw error or silently returning an incomplete result:

```python
from aat.english import analyze_with_retry

result = analyze_with_retry(passage="...", tokens=tokens)
```

Every marimo notebook and `aat_main.py` also passes an explicit `max_tokens` baseline (`aat.english.DEFAULT_CEILING`) when constructing its `dspy.LM` -- not because an ordinary call needs that much, but because `dspy.LM`'s own truncation warning always reports *that* baseline, never whatever a per-call override `analyze_with_retry()` used, so leaving it at dspy's own default of `None` made every such warning misleadingly claim the call had no budget at all.

To replace the untuned fallback fit with one measured against your actual configured model:

```bash
python3 utilities/calibrate_max_tokens.py
```

This runs every `GOLD_EXAMPLES` passage (`tests/fixtures/gold_examples.py`) through the real LM with a generous ceiling, fits `completion_tokens ~ intercept + slope * num_input_tokens` against the results, and writes it to `aat/english/token_budget_calibration.json`, where `estimate_max_tokens()` picks it up automatically. It's a live-LM script with real API cost (one call per gold example) -- re-run it whenever the configured model, the `AgentActionTarget` prompt, or `AATNode`'s own shape changes enough to shift how many output tokens a passage needs.


## Saving and loading a graph

`write_nodes()`/`read_nodes()` (in `aat/core/serialization.py`) save and reload an `AATGraph`'s nodes as one deterministic, pipe-delimited plain-text file, so you can persist an analysis, diff it, hand-edit it, or reload it later without re-running the LM:

```python
from aat.core import write_nodes, read_graph

write_nodes(graph.nodes, "analysis.txt")
reloaded = read_graph("analysis.txt")
```

The file has one `#!aatnodes` block per call to `write_nodes()`/`serialize_nodes()`, each with the fixed header `context|id|value|role|related_node` -- see `serialization.py`'s module docstring for the exact format. Multiple blocks in one file are concatenated, in file order, into the list `read_nodes()` returns, so simply concatenating several `write_nodes()` outputs together and reading the result back gives you one combined graph.

`serialize_analysis()`/`write_analysis()` (a thin wrapper that writes `serialize_analysis()`'s string to a file) and `read_analysis()` save and reload a *complete, re-displayable* analysis -- the graph AND the complete, already-tokenized input every node's `id` refers back to -- so a later reader gets the exact same tokens straight back, with no re-tokenization step and so no dependency on `aat.english.tokenize()` (or any other tokenizer) at all:

```python
from aat.core import write_analysis, read_analysis

write_analysis(tokens, graph, "analysis.txt")
reloaded_tokens, reloaded_graph = read_analysis("analysis.txt")
```

`tokens` here is whatever `analyze_passage()`/`analyze_passages()`/`analyze_units_by_sentence()` returned alongside `graph` -- pass it straight through, don't reconstruct it. Call `serialize_analysis()` directly (no `path` argument) when you want the text itself rather than a file -- this is what powers `aat_graph.py`'s and `aat_corpus_graph.py`'s own "Save analysis to file" buttons, which write the string wherever the user's own directory picker points, not to a fixed path. `aat_reader.py` is the matching file-loading notebook -- see "Interactive notebook" below. The file has a `#!tokens` block (header `context|id|value`, one row per token, in reading order) alongside the `#!aatnodes` block; each is read independently by its own function (`read_tokens()`/`read_nodes()`), so the two block types can coexist in one file without interfering with each other. Unlike an earlier version of this format (a `#!passages` block of raw passage text, requiring a fresh `tokenize()` call to recover tokens on reload), a `#!tokens` block already *is* the resolved token list -- every token an `#!aatnodes` node's `id` can point at, not just the ones that became nodes, in file order -- so an id like `t3` (or a sentence-spanning composite id like `1.14.t3`, from `analyze_units_by_sentence()`) is resolvable back to its surface text and its position in the passage directly from the file, without running any code at all. `aat_main.py` (see "Running an analysis from the command line" above) is a third way to get this same text: it writes `serialize_analysis()`'s output straight to stdout instead of a file, so redirecting it (`> analysis.txt`) is equivalent to calling `write_analysis()` yourself.


## Comparing AAT graphs

`aat_identical()`, `aat_similar()`, and `aat_compare()` (in `aat/core/compare.py`) compare two `AATGraph`s structurally -- by role and shape only, never by node value, id, or context, so you can compare graphs built from entirely different passages (or different languages -- an English passage against a Dutch translation, say), not just re-analyses of the same text. See `quarto/bg/aatgraphs.qmd` for the underlying definitions.

```python
from aat.core import aat_identical, aat_similar, aat_compare

aat_identical(g1, g2)   # same structure AND same edge (role) values
aat_similar(g1, g2)     # same action-node structure once agents/targets are removed
aat_compare(g1, g2)     # g1/g2 ratios, for graphs that are neither
```

`aat_identical()` is True when `g1` and `g2` have the same shape -- the same action-dependency tree, with the same agent/target roles attached at each position -- regardless of what the underlying tokens actually say. `aat_similar()` is a looser check: True when the two graphs' *action* nodes alone (agents and targets stripped out first) are AAT-identical, i.e. they agree on how many verbal units there are and how they're subordinated to each other, even if their agent/target structure differs.

For graphs that are neither identical nor similar, `aat_compare()` returns an `AATComparison` with three g1/g2 ratios:

```python
result = aat_compare(g1, g2)
result.action_node_ratio  # (# action nodes in g1) / (# action nodes in g2)
result.depth_ratio        # (depth of g1) / (depth of g2)
result.size_ratio         # (size of g1) / (size of g2)
```

"Depth" is the longest chain of subordinate actions (an independent action alone is depth 1); "size" is the graph's total node count (agents + actions + targets together). `aat_compare()` raises `ValueError` if any of `g2`'s three values is 0, since the ratio would be undefined.


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


## Rendering a graph as Graphviz dot

`graph_to_dot()` (in `aat/core/graphviz.py`) renders the same `AATGraph` as a [Graphviz](https://graphviz.org) DOT digraph -- the same node/edge/coloring model as `graph_to_mermaid()`, just in Graphviz's own syntax. An action is `shape=box`, an agent is `shape=box, style=rounded`, and a target is `shape=ellipse` (Graphviz has no shape literally called "stadium", so the fully-rounded ellipse is the closest analogue to Mermaid's stadium shape). Every node with a `related_node` becomes a labelled edge pointing at it, exactly as in the Mermaid diagram. By default every node is colored by the same action-cluster assignment `graph_to_mermaid()` uses (`aat.core.coloring.assign_action_colors()`), applied as inline `fillcolor`/`color`/`fontcolor`/`style` attributes on each node's own line rather than Mermaid's separate `classDef`/`class` mechanism -- DOT has no equivalent grouping construct.

```python
from aat.core import graph_to_dot

dot, warnings = graph_to_dot(graph)
print(dot)
for w in warnings:
    print(f"Warning: {w}")
```

`orientation` takes the same four codes as `graph_to_mermaid()` (`"BT"`, `"TB"`/`"TD"`, `"LR"`, `"RL"` -- validated the same way, case-insensitively, by the shared `aat.core.orientation` module) and is written out as Graphviz's own `rankdir` graph attribute. Graphviz's `rankdir` has no `"TD"` synonym of its own, so `"TD"` is mapped to `"TB"` (the same direction, just Graphviz's own name for it) -- every other value is used verbatim:

```python
dot, warnings = graph_to_dot(graph, orientation="LR")
```

Pass `color_by_action=False` for a plain, uncolored digraph. `save_dot(graph, path, ...)` takes the same `orientation`/`color_by_action` arguments and writes the digraph straight to a file (e.g. `analysis.dot`), which the `dot` command-line tool (or any other Graphviz frontend) can render directly: `dot -Tsvg analysis.dot -o analysis.svg`.

`warnings` has the same two cases as `graph_to_mermaid()`'s: a node whose `related_node` doesn't resolve to another node actually present in `graph`, and, if the graph has more distinct actions than the color palette has slots, one warning that colors repeat.

`aat_to_dot.py` is the command-line version of this: it reads a serialized analysis from stdin (the same `#!aatnodes` plain-text format -- a `#!tokens` block alongside it, if present, is ignored) and writes the DOT digraph to stdout, so you can pipe `aat_main.py`'s own output straight into it:

```bash
python3 aat_main.py --passage "The dog ate my homework." | python3 aat_to_dot.py > analysis.dot
```

or render a file saved earlier:

```bash
python3 aat_to_dot.py --orientation LR --no-color < analysis.txt > analysis.dot
```

`--orientation` and `--no-color` mirror `graph_to_dot()`'s own `orientation`/`color_by_action` arguments; warnings go to stderr, never stdout, so stdout stays exactly the DOT text -- pipe it straight into Graphviz's own `dot` CLI: `python3 aat_to_dot.py < analysis.txt | dot -Tsvg -o analysis.svg`.

## Rendering tokens as highlighted HTML

`tokens_to_html()` (in `aat/core/html.py`) renders a passage's tokens as one continuous HTML string, reconstructing normal reading spacing (punctuation attaches to the preceding word; opening brackets and the first of a paired quote attach to what follows) rather than putting a space before every token. Pass the same `AATGraph` you'd hand to `graph_to_mermaid()` and every token that's also an AAT graph node is highlighted using the *same* color that node gets in the Mermaid diagram (`aat.core.coloring.assign_action_colors()` -- one shared assignment behind both renderers), with a border style keyed on the node's role: a box around an `action` token, a rounded box around an `agent` token, and an underline under a `target` token.

It lives in `aat.core`, not `aat.english`, even though it only renders English-looking punctuation conventions -- nothing in it needs dspy, so `from aat.core import tokens_to_html` (or `from aat import tokens_to_html`) works with just the base `aat` install, no `english` extra required. It's still re-exported from `aat.english` too, so existing code importing it from there keeps working unchanged.

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

`marimo/aat_graph.py` is a [marimo](https://marimo.io) notebook: pick a language (English or Dutch), enter a context ID and a passage of text in that language, submit the form, and it tokenizes the passage, runs it through the selected language's own `analyze_passage()` (`aat.english.analyze_passage()` or `aat.dutch.analyze_passage()` -- each with its own tokenizer, dspy signature, and token-budget calibration), and renders the resulting `AATGraph` both as a Mermaid diagram (`aat.core.graph_to_mermaid()`) and as highlighted passage text (`aat.core.tokens_to_html()`, language-agnostic), side by side. The language choice is part of the form, not a live control -- like the passage text itself, changing it needs an explicit re-submission before it takes effect, since it changes which LM call actually runs. A separate orientation control (default `BT`) updates the diagram live, without resubmitting the form or making another LM call. A "See cost" checkbox shows the cumulative dollar cost of every LM call made so far this session (`aat.lm_cost.summarize_lm_cost()`/`format_lm_cost()`, reading `lm.history`) -- every marimo notebook that connects to an LM has this same checkbox, in the same place. A directory picker and a "Save analysis to file" button write the current passage and graph via `aat.core.write_analysis()`; the filename is derived automatically from the context ID (non-alphanumeric characters collapsed to `_`, falling back to `analysis.txt`), so you can reopen the result later without re-running the LM -- `aat_reader.py` doesn't need to know which language produced a saved file, since the saved `#!tokens`/`#!aatnodes` format doesn't record the source language at all. Needs dspy available -- installed via the 'english' extra, the 'dutch' extra, or 'dev' (which also pulls it in directly) -- and a working `.env` (see above) -- the LM is configured as soon as the notebook loads.

```bash
marimo edit marimo/aat_graph.py
```

opens it in an editable, reactive browser session; `marimo run marimo/aat_graph.py` runs the same notebook as a read-only app (code cells hidden, just the form and the diagram).

`marimo/aat_corpus_graph.py` is the sentence-spanning, whole-corpus counterpart: pick a language (English or Dutch), browse to a CEX corpus file (two columns -- a CTS URN and that citation unit's own text -- any delimiter), set how many citation units to read (a real safety valve, not a nicety -- every sentence group is its own billed LM call) and click "Analyze corpus". Unlike `aat_graph.py`'s form, this notebook has no `.form()` -- the language, corpus file, delimiter, and unit-limit controls are all live, but the actual read and analysis only happen inside the "Analyze corpus" button's own gated cell, which also captures the language selected at that moment; touching any of those controls afterward clears the current display (no new LM call) until the button is clicked again, so what's on screen always matches the inputs that produced it. It groups citation units into sentences and analyzes each group as one passage through the selected language's own `analyze_units_by_sentence()` (`aat.english.analyze_units_by_sentence()` or `aat.dutch.analyze_units_by_sentence()` -- see "Analyzing a corpus by sentence" above), each call going through the same automatic truncation-retry as every other notebook (see "Managing the LM's output token budget" above -- worth knowing here specifically, since a sentence spanning several citation units can need a noticeably larger budget than any single unit alone would), then shows the same Mermaid-diagram-plus-highlighted-text display, the same "See cost" checkbox, and a "Save analysis to file" button (filename derived from the corpus file's own name, `write_analysis()` given the actual tokens `analyze_units_by_sentence()` produced -- composite sentence-spanning ids and all -- so the file stays reloadable with no LM access and no re-clustering -- see above, and reloads the same way regardless of which language produced it).

`marimo/aat_reader.py` is a companion notebook with the identical Mermaid-diagram-plus-highlighted-text display, but instead of a passage form and an LM call, it has a file picker: browse to and select a file any of this project's "Save analysis to file" buttons wrote (`aat_graph.py`, `aat_corpus_graph.py`), or one written directly with `aat.core.write_analysis()` (`aat_main.py`'s and `aat_corpus.py`'s own stdout included), and it reads the saved token list straight from the file's own `#!tokens` block and pairs it back up with the saved graph -- no re-tokenization step, so it works identically regardless of which script or notebook produced the file, sentence-spanning composite ids included. It needs no `.env`, no configured LM, and makes no network access at all -- everything it shows comes straight from the file:

```bash
marimo edit marimo/aat_reader.py
```


## Using an optimized prompt

If you've run `utilities/optimize_gepa.py` (see OPTIMIZING.md) and saved an optimized program, load it into `analyze` before calling `analyze_passage()`/`analyze_passages()`:

```python
from aat.english.dspy_signatures import analyze
analyze.load("optimized_agent_action_target.json")
```

`analyze` is the same module-level `dspy.ChainOfThought` instance the whole pipeline uses, so loading into it in place is enough -- nothing else needs to change.
