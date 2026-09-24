"""
Group citable text passages (aat.core.CitedPassage, one per citation
unit -- typically one row from a CEX corpus's `#!ctsdata` block) into
the sentences they actually form, and tokenize each sentence with ids
that stay unique even though it may span more than one citation unit.
Parallel to aat.english.sentences -- this module's actual logic is
entirely language-agnostic (CTS URN plumbing and sentence-boundary
clustering, not grammar), so the two differ only in which language's
`tokenize()` each one calls; see that module's own docstring for the
real English (RSV Genesis 1:14-15) incident that motivated this design,
which applies here unchanged since the underlying problem -- a citation
unit boundary that doesn't line up with a sentence boundary -- isn't
specific to English.

Why this is needed at all: aat.dutch.tokenize() numbers a passage's
tokens 't1', 't2', ... *within that one passage's own context* (see
CitableToken's own docstring) -- correct for a single citation unit
analyzed on its own, but not when a sentence's grammar genuinely crosses
a citation-unit boundary, the same way it isn't for English.

The fix used here: take a CTS URN's own *passage* component (the fifth,
final ':'-delimited field -- e.g. "1.14" from
"urn:cts:compnov:bible.genesis.rvvpl:1.14") and use it to prefix that
unit's own locally-numbered token ids ("1.14.t1", "1.14.t2", ...) when
multiple units are combined into one sentence-level passage for
analysis. Every token across the whole combined sentence is still
guaranteed unique (two different citation units never share a passage
component), while each token's id still says exactly which citation unit
it came from.

Every function here is pure and LM-free -- deterministic given the same
ordered list of CitedPassage, the same way aat.dutch.tokenize() itself
needs no LM access. That determinism is what makes the ids this module
assigns (e.g. "1.14.t3") trustworthy in the first place: the same
ordered `units` always produce the same combined contexts/token ids, no
matter how many times or in what order analysis happens to run. A
caller that later wants to redisplay an already-saved analysis doesn't
need to re-run anything in this module at all, though -- write_analysis()
(aat.core.serialization) saves the actual resolved tokens this module
produced, composite ids included, so read_analysis() gets them straight
back from the file's own `#!tokens` block.
"""

from typing import List, Tuple

from aat.core import CitableToken, CitedPassage

from .tokenize import tokenize

# '.', '?', '!' -- the ordinary Dutch sentence-final marks too (same set
# as English's). Unlike aat.english.sentences's own version of this
# constant, this default hasn't been checked against a real Dutch
# citation-unit corpus the way English's was against RSV Genesis (see
# that module's own comment on the point, including why ':'/';' are
# deliberately excluded there) -- there wasn't one to check against yet.
# Revisit this once a real Dutch corpus exists to verify it against;
# until then, pass a different `terminators` string for a corpus that
# needs a different rule (e.g. one that also closes a sentence on a
# trailing closing quote after '.').
DEFAULT_SENTENCE_TERMINATORS = ".?!"


def passage_component(urn: str) -> str:
    """The fifth, final ':'-delimited field of a CTS URN `urn` -- e.g.
    "1.14" from "urn:cts:compnov:bible.genesis.rvvpl:1.14". Raises
    ValueError, naming `urn`, if it doesn't split into at least 5
    ':'-delimited fields (i.e. isn't a CTS URN at all)."""
    parts = urn.split(":")
    if len(parts) < 5:
        raise ValueError(
            f"{urn!r} doesn't look like a CTS URN (expected at least 5 "
            f"':'-delimited fields, got {len(parts)})"
        )
    return parts[4]


def urn_prefix(urn: str) -> str:
    """Everything in a CTS URN `urn` *before* its passage component --
    "urn:cts:compnov:bible.genesis.rvvpl" from
    "urn:cts:compnov:bible.genesis.rvvpl:1.14" -- i.e. the first four
    ':'-delimited fields, rejoined with ':'. Same ValueError as
    passage_component() for a non-CTS-URN `urn`."""
    parts = urn.split(":")
    if len(parts) < 5:
        raise ValueError(
            f"{urn!r} doesn't look like a CTS URN (expected at least 5 "
            f"':'-delimited fields, got {len(parts)})"
        )
    return ":".join(parts[:4])


def ends_sentence(text: str, terminators: str = DEFAULT_SENTENCE_TERMINATORS) -> bool:
    """True if `text`, stripped of trailing whitespace, ends in one of
    `terminators`'s characters. False for empty/all-whitespace `text`
    (there's nothing to call a sentence ending here, and indexing an
    empty stripped string would raise) -- a citation unit with no text
    at all just continues on to the next one, same as any other
    non-terminal unit."""
    stripped = text.rstrip()
    if not stripped:
        return False
    return stripped[-1] in terminators


def cluster_sentences(
    units: List[CitedPassage], terminators: str = DEFAULT_SENTENCE_TERMINATORS
) -> List[List[CitedPassage]]:
    """Group `units` (citation units, in their own citation order) into
    the smallest possible consecutive runs that each end with a
    sentence-ending unit (ends_sentence() on that run's own last unit),
    i.e. a one-unit "sentence" whenever a unit's own text already ends
    the sentence, or a multi-unit run when it takes more than one
    citation unit to reach one.

    The final group is included even if it never reaches a
    sentence-ending unit (e.g. the corpus's own last unit ends mid-clause,
    or `units` is empty) -- there is nothing else to do with trailing
    text than treat it as its own group; callers that care can check
    `not ends_sentence(group[-1].text, terminators)` on the last group
    returned to warn about it.

    Returns `[]` for empty `units`.
    """
    groups: List[List[CitedPassage]] = []
    current: List[CitedPassage] = []

    for unit in units:
        current.append(unit)
        if ends_sentence(unit.text, terminators):
            groups.append(current)
            current = []

    if current:
        groups.append(current)

    return groups


def tokenize_units(units: List[CitedPassage]) -> Tuple[str, str, List[CitableToken]]:
    """Combine one sentence's worth of citation units (`units`, in
    citation order -- typically one element of cluster_sentences()'s own
    return value) into a single analyzable passage: a combined context
    reference, the combined text, and a token list unique within that
    combined context.

    The combined context is a CTS range reference spanning every unit's
    own passage component: `{urn_prefix}:{first_passage}` for a single
    unit, `{urn_prefix}:{first_passage}-{last_passage}` for more than one
    (standard CTS range syntax). Every unit in `units` must share the
    same urn_prefix() -- raises ValueError, naming both, if two units
    disagree (a sign `units` mixes citation units from different works,
    which cluster_sentences() should never itself produce from one
    corpus, but this is cheap to check and a much clearer failure than a
    silently wrong combined context).

    The combined text is every unit's own text, in order, joined by a
    single space.

    Tokens: each unit is tokenized *on its own* (aat.dutch.tokenize(),
    unmodified -- so splitting behavior exactly matches a unit analyzed
    independently, e.g. via aat_corpus.py), giving each unit its own
    locally-numbered 't1', 't2', ... ids; every one of those ids is then
    rewritten to "{that unit's own passage component}.{local id}" (e.g.
    "1.14.t1") and given `context` = the combined context above, so the
    final list is unique within it regardless of how many units
    contributed. Tokenizing each unit separately like this (rather than
    tokenizing the combined text as one blob) also means the boundary
    between two citation units is never accidentally merged into one
    token, whatever whitespace does or doesn't separate them in the
    original text.

    Raises ValueError for empty `units`.
    """
    if not units:
        raise ValueError("tokenize_units() needs at least one CitedPassage")

    prefix = urn_prefix(units[0].context)
    passages = [passage_component(u.context) for u in units]
    for unit, unit_passage in zip(units[1:], passages[1:]):
        if urn_prefix(unit.context) != prefix:
            raise ValueError(
                f"units don't share one urn_prefix: {units[0].context!r} vs "
                f"{unit.context!r}"
            )

    if len(units) == 1:
        combined_context = f"{prefix}:{passages[0]}"
    else:
        combined_context = f"{prefix}:{passages[0]}-{passages[-1]}"

    combined_text = " ".join(u.text.strip() for u in units)

    tokens: List[CitableToken] = []
    for unit, unit_passage in zip(units, passages):
        local_tokens = tokenize(CitedPassage(context=unit.context, text=unit.text))
        for tok in local_tokens:
            tokens.append(
                CitableToken(
                    context=combined_context,
                    id=f"{unit_passage}.{tok.id}",
                    value=tok.value,
                )
            )

    return combined_context, combined_text, tokens


def tokenize_corpus_by_sentence(
    units: List[CitedPassage], terminators: str = DEFAULT_SENTENCE_TERMINATORS
) -> List[Tuple[str, str, List[CitableToken]]]:
    """cluster_sentences() `units` and tokenize_units() every resulting
    group, in order -- the full, LM-free half of analyzing a corpus by
    sentence rather than by citation unit. See aat.dutch.pipeline's
    analyze_units_by_sentence() for the LM-dependent counterpart that
    runs each group returned here through analyze()/validate() too.
    """
    return [tokenize_units(group) for group in cluster_sentences(units, terminators)]
