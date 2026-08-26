# aat

`aat` is a python package implementing a reductive model of natural-language syntax called Agent-Action-Target (AAT), documented in [`aat-model.md`](aat-model.md), plus a [dspy](https://dspy.ai)-based pipeline that applies the model to English text.

The package is deliberately split into two halves that don't depend on each other in the direction that matters:

- **`aat.core`** -- the AAT model itself: `CitableToken`/`CitedPassage` (input), `AATNode`/`AATGraph` (output), `validate()`, and a plain-text serialization format. No dependency on dspy, or on any particular language. A downstream project that wants to apply the AAT model to a *different* language can depend on `aat.core` alone and write its own extraction code the way `aat.english` does here.
- **`aat.english`** -- a dspy program (`AgentActionTarget`) that actually extracts agent/action/target nodes from English text, plus a tokenizer, a pipeline (`analyze_passage`/`analyze_passages`), and a GEPA scoring metric. Depends on `aat.core`, and on `dspy`.

Released under the [GNU General Public License v3 or later](LICENSE).


## Installing

To use `aat` from another project, install it straight from this repository (no PyPI account or release process needed):

```sh
pip install git+https://github.com/neelsmith/aat.git
```

That installs `aat.core` only -- no dspy dependency at all. To also install `aat.english` (the dspy-based English pipeline):

```sh
pip install "aat[english] @ git+https://github.com/neelsmith/aat.git"
```

Pin to a specific branch, tag, or commit by appending `@<ref>` to the URL, e.g. `...aat.git@v0.1.0` once a version is tagged.

Working on `aat` itself (this repo checked out locally): `pip install -e ".[dev]"` from the repo root installs it in editable mode with every dev dependency (dspy, pytest, python-dotenv, pdoc), so source edits take effect immediately without reinstalling.


## Using `aat`

- [USAGE.md](USAGE.md) -- running the pipeline, from the command line or from your own code
- [TESTING.md](TESTING.md) -- running the offline test suite
- [OPTIMIZING.md](OPTIMIZING.md) -- tuning `AgentActionTarget`'s prompt with GEPA
- [DEVELOPMENT.md](DEVELOPMENT.md) -- how the above fit together into one development loop
- `marimo/aat_graph.py` -- an interactive marimo notebook: enter a passage, submit, see its AAT graph as a Mermaid diagram
- API documentation -- published automatically to GitHub Pages on every push to `main` (see `.github/workflows/docs.yml`); once enabled for this repo, it's at `https://neelsmith.github.io/aat/`

See the [project issue tracker](https://github.com/neelsmith/aat/issues) for known gaps and work in progress.


## The model

`aat-model.md` documents the AAT model itself: how a citable passage of text tokenizes, and how a selection of those tokens becomes a graph of `agent`/`action`/`target` nodes. `aat.core` implements that graph shape and its referential-integrity checks; it says nothing about *how* to decide which tokens get which role in a given language -- that's `aat.english`'s job (see `AgentActionTarget`'s docstring in `aat/english/dspy_signatures.py` for the English-specific rules it follows).
