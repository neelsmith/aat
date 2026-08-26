# Developing `aat`

`USAGE.md` describes how to call the pipeline, `TESTING.md` how to run the offline test suite, `OPTIMIZING.md` how to tune `AgentActionTarget`'s prompt with GEPA. This document describes how they fit together into a repeatable development loop, centered on testing the analyzer against real English text rather than just the hand-picked `GOLD_EXAMPLES` corpus -- and on keeping `aat.core` and `aat.english` cleanly separated as the package grows.


## The core/english boundary

`aat.core` (`aat/core/`) is the AAT model itself: `CitableToken`, `CitedPassage`, `AATNode`, `AATGraph`, `validate()`, and the plain-text serialization format. It has no dependency on dspy and no knowledge of any particular language. `aat.english` (`aat/english/`) is one application of that model: a deterministic tokenizer plus a dspy program that decides which tokens get which role, specific to English syntax (active/passive voice, compound verbs, subject/object relations, and so on).

When adding a feature, ask which side it belongs on:

- Does it describe the *shape* of an AAT graph, or check its referential integrity, without needing to know what language produced it? -> `aat.core`.
- Does it decide *how* a role gets assigned to a token, or how a passage gets tokenized, and does that decision depend on English grammar specifically? -> `aat.english`.
- If you were later asked to apply the AAT model to a different language (a sibling `aat.latin`, say), would this code need to change? If yes, it belongs in `aat.english`, not `aat.core` -- `aat.core` should need zero changes to support a new language.

`aat/english/__init__.py` importing from `aat.core` is fine and expected; nothing in `aat/core/` should ever import from `aat.english` (or `dspy`) -- that's the whole point of the split, and what lets a downstream project depend on `aat` alone (getting `aat.core`) without pulling in dspy at all.


## Why real-world testing, not just the gold-example suite

`tests/fixtures/gold_examples.py`'s `GOLD_EXAMPLES` is a deliberately small, curated corpus, straight from `aat-model.md`'s own worked examples: `test_coverage.py` enforces that every role, both related_node states for an action, both voices, and both "no agent"/"no target" cases have at least one example exercising them, and the whole `pytest` suite runs against `DummyLM`, not a real model. That combination is exactly right for what it's for -- proving the code (`aat/core/graph.py`'s pydantic models, `validate()`, `serialization.py`) correctly *represents* a correct answer -- but it can't tell you whether a live model actually *produces* one, and it can't surface a construction nobody has thought to write a fixture for yet (imperatives, coordinated verbs, existential "there is", reported speech without "that", and so on -- aat-model.md doesn't yet say how any of these should be analyzed).

Running the analyzer against real English passages -- actual prose, not fixtures written to order -- is where you find the things the gold-example suite structurally can't show you:

- a construction `aat-model.md` doesn't document at all yet;
- a construction the scheme already documents, but the current prompt still gets wrong;
- a genuinely ambiguous case that exposes a modeling choice worth deciding and flagging explicitly (the interrupted-auxiliary-chain judgment call in `gold_examples.py`'s `_HOMEWORK_NOT_YET_EATEN` fixture is a real example of this already resolved one way -- flag the judgment call in a comment near the fixture, don't just quietly pick one and move on);
- an ordinary, everyday construction the model already handles correctly -- not new information about the scheme, but real evidence worth locking in as a regression guardrail so a future prompt or model change can't silently break it without anyone noticing.


## The core loop: analyze, check automatically, review by hand, triage, act

1. **Analyze a real passage.** `analyze_passage(text)` for a single string, or `analyze_passages(passages)` for a list of context-labeled `CitedPassage` (see USAGE.md) -- either returns `(tokens, graph)`.
2. **Run the automated check first, before reading anything by hand.** `analyze_passage()`/`analyze_passages()` already call `validate()` for you and print any referential problems they find (a token id that doesn't exist, an agent/target node missing its related_node). That's cheap and mechanical -- let it rule out the "obviously broken" case before you spend a human read on anything.
3. **Read the surviving result against `aat-model.md` by hand.** This is the one step nothing in the codebase can do for you: `validate()` only checks referential integrity, never correctness.
4. **Triage what you found into one of three outcomes, and act accordingly** (see the next section).


## The three outcomes, and what to do with each

### Outcome A: a failure

Something is referentially broken (`validate()` caught it) or substantively wrong (you caught it by hand). Triage further, in this order:

1. **Is `aat-model.md` actually silent or ambiguous about this construction?** If so, this is a scheme gap, not a model mistake -- extend `aat-model.md` first, describe the new rule in `AgentActionTarget`'s docstring (`aat/english/dspy_signatures.py`), then hand-write a corrected `GoldExample` in `tests/fixtures/gold_examples.py` exercising it.
2. **Is the scheme already clear, but the prompt/model got it wrong anyway?** Hand-write a corrected `GoldExample` for the passage, so the failure becomes a concrete, checkable trainset entry rather than an anecdote. If you're unsure between two defensible readings, say so explicitly in a comment above the fixture (matching the existing convention for judgment calls, e.g. the interrupted-auxiliary-chain case) rather than silently committing to one.
3. Re-run `pytest` (fast, `DummyLM`-backed -- see TESTING.md) to confirm the new/corrected fixture actually validates and that `test_coverage.py` is satisfied, *before* spending any real API budget re-testing it against the live model.

### Outcome B: a success against a rare or tricky construction

The model got something genuinely uncommon or structurally hard right. Worth *reinforcing*: hand-write (or adapt from the real result) a `GoldExample` and add it to `GOLD_EXAMPLES` -- a correct demonstration of a rare case is precisely the kind of thing `optimize_gepa.py`'s trainset benefits from having more of.

### Outcome C: a success against a common, ordinary construction

The model got something right that it was already expected to get right. Real evidence, but low training value -- `optimize_gepa.py` has no held-out split today (see OPTIMIZING.md), so anything added to `GOLD_EXAMPLES` is immediately part of what GEPA both trains against and scores itself against, and an easy case the model already nails teaches the optimizer nothing new. Still worth harvesting as a regression check once the corpus is large enough to afford holding some examples out -- a known gap to revisit as `GOLD_EXAMPLES` grows past its current 4 entries.


## Suggested cadence

- After any batch of real-world testing, run `pytest` (TESTING.md) first -- it's fast and `DummyLM`-backed, and will immediately tell you if a new or corrected fixture doesn't actually validate or if `test_coverage.py` regressed.
- Run `optimize_gepa.py` (OPTIMIZING.md) periodically to refresh the shipped, production prompt against whatever `GOLD_EXAMPLES` has grown into since the last run -- this is a live-LM script with real API cost, so batch it rather than running it after every single new fixture.
- Docs (`docs/build_docs.py`) rebuild automatically on every push to `main` via `.github/workflows/docs.yml` -- no manual step needed there; run it locally (`python docs/build_docs.py`) only if you want to preview `docs/site/` before pushing.
