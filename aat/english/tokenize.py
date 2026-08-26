"""
A deterministic tokenizer for English text: turns a CitedPassage into a
list of CitableToken, per aat-model.md's "Context and tokens" section
(worked example: "The dog ate my homework." tokenizes to "The", "dog",
"ate", "my", "homework", "." -- six tokens, "t1" as the first token's id).

This is deliberately mechanical, not semantic -- splitting words from
punctuation is unambiguous enough in English to not need an LLM call at
all (contrast a language like Latin, where enclitics can make tokenization
itself genuinely ambiguous and worth handing to an LLM). The actual
Agent-Action-Target role assignment -- the hard, semantic part -- is
dspy_signatures.py's job, not this module's.
"""

import re
from typing import List

from aat.core import CitableToken, CitedPassage

# A "word" is a run of letters/digits, optionally continuing through an
# internal apostrophe (a contraction like "didn't" or a possessive like
# "dog's") -- kept as one token rather than splitting at the apostrophe.
# Anything else that isn't whitespace (punctuation, symbols) is its own
# single-character token.
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)*|[^\sA-Za-z0-9]")


def tokenize(passage: CitedPassage) -> List[CitableToken]:
    """Tokenize `passage.text` into a list of CitableToken, with ids 't1',
    't2', ... in reading order (1-indexed, per aat-model.md's own worked
    example), all sharing `passage.context`.

    Splits on whitespace, keeps an apostrophe-joined contraction or
    possessive (e.g. "didn't", "dog's") as one token, and treats every
    other punctuation character as its own token. Doesn't attempt
    sentence segmentation -- a multi-sentence passage tokenizes as one
    flat token list; split `passage` into several CitedPassage yourself
    first if you need per-sentence token lists.
    """
    tokens: List[CitableToken] = []
    for i, match in enumerate(_TOKEN_RE.finditer(passage.text), start=1):
        tokens.append(CitableToken(context=passage.context, id=f"t{i}", value=match.group(0)))
    return tokens
