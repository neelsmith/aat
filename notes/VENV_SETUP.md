# Setting up a virtualenv after a clean clone

Each machine needs its own `.venv` -- it's `.gitignore`d (along with `.env`) and never travels with the repo, so a fresh clone (or a second clone on another machine, per `CLAUDE_WORKFLOW.md`'s "Local checkouts across machines") always starts without one. These steps get a clean checkout to a state where `pytest` passes fully offline.

## 1. Create and activate the venv

Requires Python 3.10+ (the project is tested against 3.10, 3.11, and 3.12 in CI -- see `.github/workflows/tests.yml`).

```bash
cd aat
python3 -m venv .venv
source .venv/bin/activate      # zsh/bash; `.venv/bin/activate.fish` for fish
```

## 2. Install the project

Editable install, with the `dev` extra (pytest, dspy, python-dotenv, pdoc, marimo -- everything needed to run tests, build docs, and use the marimo notebooks):

```bash
pip install -e ".[dev]"
```

If you only need to *use* `aat.english` (the dspy-based pipeline) without the dev tooling, `pip install -e ".[english]"` is enough. `aat.core` alone (no dspy at all) needs no extra: `pip install -e .`.

## 3. Set up `.env`

`aat_main.py`, `aat_corpus.py`, and `optimize_gepa.py` all load model config from `.env` (see USAGE.md):

```bash
cp .env.example .env
```

Then edit `.env` with real values for `API_BASE` / `MODEL` / `API_KEY`. `.env` is also `.gitignore`d -- like the venv, it's local to each machine and never committed.

## 4. Verify

```bash
pytest
```

Should pass fully offline against `DummyLM` -- no `.env` needed for this step (see TESTING.md). `pytest -m live` additionally exercises the real, configured LM, and needs a working `.env`.

## Notes

- A venv isn't portable between machines, or even between clones on the same machine -- rebuild it (steps 1-2) in every checkout, including any second clone set up on another Cowork-linked machine.
- `uv` works too if you prefer it over plain `venv`/`pip` (`uv venv && uv pip install -e ".[dev]"`) -- `uv.lock` is already `.gitignore`d for exactly this case.
