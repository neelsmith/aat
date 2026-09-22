"""
Orchestrates aat.english's tokenize -> analyze -> validate pipeline: the
convenience layer most callers should use instead of calling
tokenize.py/dspy_signatures.py directly.
"""

import sys
from typing import List, Tuple

from aat.core import AATGraph, AATNode, CitableToken, CitedPassage

from .dspy_signatures import validate
from .sentences import DEFAULT_SENTENCE_TERMINATORS, tokenize_corpus_by_sentence
from .token_budget import analyze_with_retry
from .tokenize import tokenize


def analyze_passages(passages: List[CitedPassage]) -> Tuple[List[CitableToken], AATGraph]:
    """Tokenize and analyze every passage in `passages`, independently
    (each passage's tokens are numbered from 't1' within its own context
    -- see CitableToken's docstring), and return (tokens, graph): `tokens`
    is every passage's tokens concatenated in order, `graph` is one
    AATGraph combining every passage's nodes.

    Prints a warning to stderr (not stdout -- callers such as aat_main.py
    write a plain-text serialized analysis to stdout, and a stray print()
    there would corrupt it) for any passage whose analysis fails
    validate() (a referential problem -- see aat.core.validate.validate);
    it does not raise, since a referential problem is a sign the LM's
    output needs a re-run or a prompt tweak, not necessarily that the
    caller's own code is broken.
    """
    all_tokens: List[CitableToken] = []
    all_nodes: List[AATNode] = []

    for passage in passages:
        tokens = tokenize(passage)
        # analyze_with_retry() (token_budget.py), not analyze() directly --
        # estimates a max_tokens budget from this passage's own token
        # count and retries with a larger one if the call still comes
        # back truncated, instead of surfacing a raw AdapterParseError
        # or a silently incomplete result. See token_budget.py's own
        # module docstring for the full design.
        result = analyze_with_retry(passage=passage.text, tokens=tokens)

        problems = validate(tokens, result)
        if problems:
            print(f"Validation warnings (context {passage.context!r}):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)

        all_tokens.extend(tokens)
        all_nodes.extend(result.nodes)

    return all_tokens, AATGraph(nodes=all_nodes)


def analyze_passage(text: str, context: str = "") -> Tuple[List[CitableToken], AATGraph]:
    """Convenience wrapper for the common case of a single string rather
    than a list of CitedPassage. Wraps `text` as one CitedPassage (using
    `context` if given, else an empty string) and runs it through
    analyze_passages()."""
    return analyze_passages([CitedPassage(context=context, text=text)])


def analyze_units_by_sentence(
    units: List[CitedPassage], terminators: str = DEFAULT_SENTENCE_TERMINATORS
) -> Tuple[List[CitableToken], AATGraph]:
    """Like analyze_passages(), but for a corpus of citation units (e.g.
    every row from a CEX file's `#!ctsdata` block, in citation order)
    where a sentence's grammar may span more than one unit -- see
    aat.english.sentences's own module docstring for why that matters
    and worked examples.

    Groups `units` into sentences (aat.english.sentences.
    cluster_sentences()) and analyzes each group as ONE combined passage
    (aat.english.sentences.tokenize_units() builds its combined context/
    text/tokens; every unit's own token ids get prefixed with that
    unit's passage component -- e.g. "1.14.t1" -- so every token across
    the whole sentence stays unique even though several citation units
    contributed to it), rather than analyzing each citation unit
    independently the way analyze_passages()/aat_corpus.py both do.

    Returns (tokens, graph) in the same shape analyze_passages() does --
    `tokens` is every sentence group's tokens concatenated in order,
    `graph` is one AATGraph combining every group's nodes -- so a caller
    (e.g. a marimo notebook) can hand either straight to
    aat.core.tokens_to_html()/aat.core.graph_to_mermaid() exactly as
    it would analyze_passages()'s own return value.

    To later serialize this analysis (e.g. aat.core.write_analysis()),
    pass this function's own returned `tokens` straight through -- they
    already carry every composite sentence-spanning id (e.g.
    "1.14.t3") this function assigned, in reading order, so a reload
    needs no LM and no re-run of aat.english.sentences.
    tokenize_corpus_by_sentence() (or any other tokenization step) at
    all; see aat.core.serialization's own module docstring for why.

    Same validate()-and-warn-on-stderr behavior as analyze_passages(),
    once per sentence group rather than once per citation unit -- a
    referential problem in a sentence spanning three citation units is
    reported once, tagged with that sentence's own combined context, not
    three times.
    """
    all_tokens: List[CitableToken] = []
    all_nodes: List[AATNode] = []

    for combined_context, combined_text, tokens in tokenize_corpus_by_sentence(units, terminators):
        # Same analyze_with_retry() as analyze_passages() -- doubly
        # important here, since a sentence group spanning several
        # citation units can be considerably longer than any single
        # citation unit's own text, so a budget sized for one unit alone
        # is exactly the kind of case that used to truncate.
        result = analyze_with_retry(passage=combined_text, tokens=tokens)

        problems = validate(tokens, result)
        if problems:
            print(f"Validation warnings (context {combined_context!r}):", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)

        all_tokens.extend(tokens)
        all_nodes.extend(result.nodes)

    return all_tokens, AATGraph(nodes=all_nodes)
