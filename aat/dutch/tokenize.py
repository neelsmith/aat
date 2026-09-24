"""
A deterministic tokenizer for Dutch text: turns a CitedPassage into a
list of CitableToken, the same way aat.english.tokenize() does for
English -- see notes/dutch.md's "Context and tokens" section for the
worked example this mirrors ("Alle kunsten en wetenschappen bezitten een
gemeenschappelijke band." tokenizes to "Alle", "kunsten", "en",
"wetenschappen", "bezitten", "een", "gemeenschappelijke", "band", "."
-- nine tokens, "t1" as the first token's id).

This is deliberately mechanical, not semantic, the same way aat.english's
tokenizer is -- splitting words from punctuation doesn't need an LLM call.
The actual Agent-Action-Target role assignment -- the hard, semantic part
-- is dspy_signatures.py's job, not this module's.

One real difference from aat.english.tokenize(): Dutch words routinely
contain letters outside plain ASCII (diaereses like "ë"/"ï" marking a
vowel that starts a new syllable rather than merging with the one before
it -- "ideeën", "financiële" -- plus borrowed/loan letters like "é" in
"café"), so the word-character class here is Python's Unicode-aware `\w`
rather than English's ASCII-only `[A-Za-z0-9]`. Using `\w` also means a
diacritic is kept as part of its word rather than becoming its own
stray single-character token.
"""

import re
from typing import List

from aat.core import CitableToken, CitedPassage

# A "word" is a run of Unicode word characters (letters -- including
# diacritics like "ë"/"ï"/"é" -- digits, and underscore, via `\w`),
# optionally continuing through an internal apostrophe (e.g. the plural
# "auto's", "foto's", or a genitive like "Jans'") -- kept as one token
# rather than splitting at the apostrophe, the same way aat.english
# handles "didn't"/"dog's". Anything else that isn't whitespace
# (punctuation, symbols) is its own single-character token.
#
# Known, accepted gap: a *leading* apostrophe standing in for an elided
# syllable (e.g. "'s morgens", "'s avonds", "'t") isn't joined to the
# word that follows it, since that's a different pattern from the
# internal/trailing apostrophes above -- it tokenizes as "'" then
# "s"/"t" then the next word, separately. Not addressed here for the
# same reason aat.english's own tokenizer doesn't try to be a complete
# morphological analyzer: worth revisiting if it turns out to matter for
# real passages, but not guessed at without one.
_TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\s\w]")


def tokenize(passage: CitedPassage) -> List[CitableToken]:
    """Tokenize `passage.text` into a list of CitableToken, with ids 't1',
    't2', ... in reading order (1-indexed, per aat-model.md's/
    notes/dutch.md's own worked examples), all sharing `passage.context`.

    Splits on whitespace, keeps an apostrophe-joined plural or genitive
    (e.g. "auto's") as one token, and treats every other punctuation
    character as its own token. Doesn't attempt sentence segmentation --
    a multi-sentence passage tokenizes as one flat token list; split
    `passage` into several CitedPassage yourself first if you need
    per-sentence token lists (or see aat.dutch.sentences for citation-
    unit-spanning sentence clustering).
    """
    tokens: List[CitableToken] = []
    for i, match in enumerate(_TOKEN_RE.finditer(passage.text), start=1):
        tokens.append(CitableToken(context=passage.context, id=f"t{i}", value=match.group(0)))
    return tokens
