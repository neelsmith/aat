"""
Summarizes a `dspy.LM` instance's own `.history` -- the list `dspy.LM`
appends one entry to per call it makes. Each entry is a plain dict
carrying, among other keys, `cost` -- that call's price in US dollars --
except `cost` is `None` specifically when that call was served from
cache (dspy's own per-call response cache, keyed on model + messages +
config) rather than actually billed. `lm.history` itself is always a
list -- possibly empty, e.g. before any call has been made -- never
`None`.

Deliberately has no `dspy` import of its own (every entry is read as a
plain dict, or defensively via a `cost` attribute if a future dspy
version ever represents one as an object instead) -- same reasoning as
the rest of `aat.core`: a caller that only wants this small helper
shouldn't need dspy installed to get it. `aat/__init__.py` re-exports
only `aat.core`, so import this module directly:

    from aat.lm_cost import summarize_lm_cost, format_lm_cost

    cost_summary = summarize_lm_cost(lm.history)
    mo.md(f"**LM cost so far**: {format_lm_cost(cost_summary)}")

`summarize_lm_cost()` never raises on an empty or all-cache-hit history
-- an empty or freshly-configured LM's `.history` is the normal starting
state for any notebook before its first analysis, not an edge case, so
a naive `lm.history[-1].get('cost')` breaks on exactly the common case;
see each function's own docstring for exactly what's returned instead.
"""

from typing import Any, List, NamedTuple, Optional


class LMCostSummary(NamedTuple):
    """The result of `summarize_lm_cost()`.

    `total_cost` is the dollar sum of every history entry that actually
    recorded a cost, or `None` if there were no such entries at all --
    either because `history` was empty, or because every call in it was
    a cache hit. Callers should treat `None` as "unknown", not as "$0"
    -- summing an empty/all-`None` set of costs is not the same claim as
    "this cost nothing".

    `priced_calls` and `uncosted_calls` count entries that did and
    didn't record a `cost`, respectively; `total_calls` is their sum,
    i.e. `len(history)`."""

    total_cost: Optional[float]
    priced_calls: int
    uncosted_calls: int

    @property
    def total_calls(self) -> int:
        return self.priced_calls + self.uncosted_calls


def _entry_cost(entry: Any) -> Optional[float]:
    """Read one history entry's own `cost`, however it's shaped: a plain
    dict (every entry `dspy.LM` itself produces, as of this writing) via
    `.get('cost')`, or -- defensively, in case a future dspy version
    ever represents an entry as an object instead -- via a `cost`
    attribute. Missing either way reads as `None`, same as an explicit
    cache-hit `None`."""
    if isinstance(entry, dict):
        return entry.get("cost")
    return getattr(entry, "cost", None)


def summarize_lm_cost(history: List[Any]) -> LMCostSummary:
    """Sum the `cost` recorded on every entry of `history` (a `dspy.LM`
    instance's own `.history` list, or any list shaped like it) that
    actually has one, and count how many did versus didn't.

    Never raises: an empty `history` returns
    `LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=0)`,
    and a `history` where every entry's `cost` is `None` (every call
    served from cache) returns `LMCostSummary(total_cost=None,
    priced_calls=0, uncosted_calls=len(history))` -- `total_cost` is
    only ever a number when at least one entry actually recorded one.
    """
    priced_calls = 0
    uncosted_calls = 0
    total_cost = 0.0

    for entry in history:
        cost = _entry_cost(entry)
        if cost is None:
            uncosted_calls += 1
        else:
            priced_calls += 1
            total_cost += cost

    if priced_calls == 0:
        return LMCostSummary(total_cost=None, priced_calls=0, uncosted_calls=uncosted_calls)
    return LMCostSummary(total_cost=total_cost, priced_calls=priced_calls, uncosted_calls=uncosted_calls)


def format_lm_cost(summary: LMCostSummary) -> str:
    """Render an `LMCostSummary` as one short, human-readable line for
    display (e.g. a notebook's own "See cost" checkbox) -- covering
    every case `summarize_lm_cost()` can return without the caller
    needing to branch on `None` itself:

    - no calls at all -> "no LM calls yet"
    - calls, but every one served from cache -> says so explicitly,
      rather than printing a bare, unexplained "None"
    - a mix of priced and cached calls -> the priced total, plus a note
      that some calls aren't included in it
    - every call priced -> the total alone
    """
    if summary.total_calls == 0:
        return "no LM calls yet"

    if summary.total_cost is None:
        call_word = "call" if summary.uncosted_calls == 1 else "calls"
        return f"$0.00 billed -- {summary.uncosted_calls} {call_word}, all served from cache (no cost recorded)"

    priced_word = "call" if summary.priced_calls == 1 else "calls"
    base = f"${summary.total_cost:.4f} across {summary.priced_calls} {priced_word}"
    if summary.uncosted_calls:
        cached_word = "call" if summary.uncosted_calls == 1 else "calls"
        return f"{base} (+ {summary.uncosted_calls} more {cached_word} served from cache, not included)"
    return base
