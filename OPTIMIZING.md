# Optimizing with GEPA

`optimize_gepa.py` uses [dspy.GEPA](https://dspy.ai) -- a reflective prompt optimizer -- to improve `AgentActionTarget`'s instructions against the gold examples in `tests/fixtures/gold_examples.py`. Unlike `pytest` (entirely `DummyLM`-backed), this is a **live-LM script**: every trial makes a real call to the configured task model, plus a reflection model GEPA uses to read scoring feedback and propose better instructions. Expect it to use real API usage against the configured proxy.

```bash
python optimize_gepa.py                    # --auto light (cheapest; default)
python optimize_gepa.py --auto medium       # more thorough, more expensive
python optimize_gepa.py --auto heavy        # most thorough, most expensive
python optimize_gepa.py --max-metric-calls 40   # exact call budget instead of a preset
python optimize_gepa.py --skip-baseline     # skip the pre-GEPA scoring pass (saves N calls)
```

Needs the same `.env` as `aat_main.py` (`API_BASE`/`MODEL`/`API_KEY`). Optionally set `REFLECTION_MODEL` (and `REFLECTION_API_BASE`/`REFLECTION_API_KEY`, if they differ) to use a different model specifically for GEPA's reflective step -- GEPA's own docs recommend a strong reasoning model for this. Without `REFLECTION_MODEL` set, the task model doubles as the reflection model, a reasonable default for a first run.

**Scope and data**: this optimizes only `AgentActionTarget` (the `analyze` module in `aat/english/dspy_signatures.py`), not the tokenizer (which is deterministic code, not an LM call -- see `aat/english/tokenize.py`). It trains on all gold examples in `GOLD_EXAMPLES` (4 as of this writing) with no separate held-out valset -- per `dspy.GEPA`'s own behavior when no valset is given, it uses the trainset for both reflective updates and Pareto-score tracking. That's a reasonable starting point while the gold set is still small, but expect the optimized prompt to fit these exact sentences well without a guarantee it generalizes to new ones -- worth revisiting (holding out a few examples as a valset) once there are more gold examples to spare.

**Scoring**: `aat/english/gepa_metric.py`'s `aat_metric` compares a prediction's `nodes` against the gold answer's `nodes` -- for each gold node, is there a predicted node at the same `(context, id, role)` with a matching `value` and `related_node`? -- and returns a score in `[0, 1]` plus specific, human-readable feedback naming every missing, extra, or mismatched node, for GEPA's reflection model to read. See `tests/test_gepa_metric.py` for fully offline tests of the metric itself.

**Using the result**: `optimize_gepa.py` saves the optimized program's instructions to `optimized_agent_action_target.json` (configurable via `--out`).

To use it:

```python
from aat.english.dspy_signatures import analyze
analyze.load("optimized_agent_action_target.json")
```

right after import and before calling `analyze_passage()`/`analyze_passages()` -- `analyze` is the same module-level `ChainOfThought` instance the whole pipeline uses, so loading into it in place is enough; nothing else needs to change. `gepa_logs/` (GEPA's own run logs) is gitignored; `optimized_agent_action_target.json` is not -- commit it once you're satisfied with a run, or gitignore it yourself if you'd rather treat it as a local, disposable artifact.
