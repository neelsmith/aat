# Testing without network access

```bash
pytest
```

Runs the whole suite against `tests/`, using dspy's `DummyLM` in place of a real LM call -- useful for confirming the models/signatures/pipeline still fit together after you change something, without spending API calls. Tests that call the actual configured LM are marked `live` and skipped by default (they're the only way to check the LM itself gets a scenario right, not just that the code can represent a correct answer); run them explicitly with:

```bash
pytest -m live
```

(`live` tests need a working `.env`, same as `aat_main.py`; they skip gracefully if `API_KEY` isn't set.)

Some standard `pytest` shorthands:

- `pytest` to run all.
- `pytest -v` for per-test names instead of dots.
- `pytest tests/test_gold_examples.py` to run just one file.
- `pytest -k tokenize` to run only tests matching a substring.
- `pytest --collect-only` if you just want to see what it discovered without running anything.

## What's covered where

- `tests/test_core_*.py` -- `aat.core` (tokens, graph accessors, `validate()`, serialization). Pure pydantic/Python, no dspy involved at all.
- `tests/test_english_tokenize.py` -- the deterministic English tokenizer.
- `tests/test_english_pipeline.py` -- `analyze_passage()`/`analyze_passages()`, DummyLM-backed.
- `tests/test_gold_examples.py` -- every entry in `tests/fixtures/gold_examples.py`'s `GOLD_EXAMPLES`, run through `analyze()` with DummyLM returning that example's own hand-written `canned_answer`, checked against `validate()` and against the gold nodes exactly.
- `tests/test_coverage.py` -- confirms `GOLD_EXAMPLES` collectively exercises every role, both related_node states for an action, both simple and compound actions, both voices, and both "no agent"/"no target" cases -- so a change that silently breaks coverage of one of these is caught here rather than only showing up later as a gap in `optimize_gepa.py`'s trainset.
- `tests/test_gepa_metric.py` -- `aat.english.gepa_metric.aat_metric` in isolation (missing/extra/mismatched nodes, perfect match), independent of `GOLD_EXAMPLES` or any LM.
