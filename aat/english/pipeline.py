"""
Orchestrates aat.english's tokenize -> analyze -> validate pipeline: the
convenience layer most callers should use instead of calling
tokenize.py/dspy_signatures.py directly.
"""

from typing import List, Tuple

from aat.core import AATGraph, AATNode, CitableToken, CitedPassage

from .dspy_signatures import analyze, validate
from .tokenize import tokenize


def analyze_passages(passages: List[CitedPassage]) -> Tuple[List[CitableToken], AATGraph]:
    """Tokenize and analyze every passage in `passages`, independently
    (each passage's tokens are numbered from 't1' within its own context
    -- see CitableToken's docstring), and return (tokens, graph): `tokens`
    is every passage's tokens concatenated in order, `graph` is one
    AATGraph combining every passage's nodes.

    Prints a warning for any passage whose analysis fails validate() (a
    referential problem -- see aat.core.validate.validate); it does not
    raise, since a referential problem is a sign the LM's output needs a
    re-run or a prompt tweak, not necessarily that the caller's own code
    is broken.
    """
    all_tokens: List[CitableToken] = []
    all_nodes: List[AATNode] = []

    for passage in passages:
        tokens = tokenize(passage)
        result = analyze(passage=passage.text, tokens=tokens)

        problems = validate(tokens, result)
        if problems:
            print(f"Validation warnings (context {passage.context!r}):")
            for p in problems:
                print(f"  - {p}")

        all_tokens.extend(tokens)
        all_nodes.extend(result.nodes)

    return all_tokens, AATGraph(nodes=all_nodes)


def analyze_passage(text: str, context: str = "") -> Tuple[List[CitableToken], AATGraph]:
    """Convenience wrapper for the common case of a single string rather
    than a list of CitedPassage. Wraps `text` as one CitedPassage (using
    `context` if given, else an empty string) and runs it through
    analyze_passages()."""
    return analyze_passages([CitedPassage(context=context, text=text)])
