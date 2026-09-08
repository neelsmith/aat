"""
Read a corpus of citable text passages from a CEX (CITE Exchange) file --
an external plain-text interchange format from the CITE architecture
(https://cite-architecture.github.io/citedx/CEX-spec-3.0.1/), not this
project's own aat.core.serialization format (a different, purpose-built
convention -- see that module's own docstring). A CEX file can carry many
kinds of data in labelled '#!'-blocks ('#!citelibrary', '#!ctscatalog',
'#!imagedata', ...); this module reads only the one block aat cares
about -- '#!ctsdata' -- and ignores every other block type entirely, the
same way aat.core.serialization's own readers ignore block types other
than their own.

A '#!ctsdata' block is two columns per line -- a CTS URN, then that
node's own text -- separated by a delimiter the file's own author chose:
per the CEX spec, the delimiter is never declared inside the file itself,
just used consistently once picked. '#' is the most common convention in
practice (and this module's own default) -- pass `delimiter` for a file
that uses something else. Blank lines are ignored, and a line starting
with '//' is a CEX comment and is also ignored -- both per the CEX spec.

Splitting is done on the delimiter's *first* occurrence per line, not a
strict two-column check (contrast aat.core.serialization's own readers,
which require an exact column count) -- a CTS URN never contains '#'
(or any other reasonable delimiter choice), but the citable text half of
the line is real prose this module doesn't control, and (especially with
the default '#' delimiter) can plausibly contain the delimiter character
again somewhere in the text itself. Splitting on the first occurrence
only keeps that text intact rather than raising over it.
"""

from typing import List

from .tokens import CitedPassage

CTSDATA_LABEL = "#!ctsdata"


def parse_cex_ctsdata(text: str, delimiter: str = "#") -> List[CitedPassage]:
    """Parse every '#!ctsdata' block in `text` (a CEX file's own full
    contents, already read into a string) and return its rows,
    concatenated in file order, as a list of CitedPassage -- `context` is
    each row's own CTS URN, `text` is its citable text. Every other CEX
    block type is ignored entirely, whether it appears before, between,
    or after the '#!ctsdata' block(s); multiple '#!ctsdata' blocks in one
    file are concatenated, in file order, same as
    aat.core.serialization's own multi-block handling.

    Raises ValueError, naming the offending line, for a '#!ctsdata' row
    that has no `delimiter` in it at all -- most often a sign `delimiter`
    is wrong for this particular file. Raises ValueError (not returning
    an empty list) if the file has no '#!ctsdata' block at all, so a
    caller can't mistake "wrong file" for "file with zero passages" --
    same reasoning as aat.core.serialization.read_nodes()/read_passages().
    """
    passages: List[CitedPassage] = []
    in_ctsdata = False
    seen = False

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip("\r\n")
        if line == "" or line.startswith("//"):
            continue

        if line.startswith("#!"):
            in_ctsdata = line == CTSDATA_LABEL
            if in_ctsdata:
                seen = True
            continue

        if not in_ctsdata:
            continue

        urn, sep, passage_text = line.partition(delimiter)
        if not sep:
            raise ValueError(
                f"line {line_no}: no {delimiter!r} delimiter found in a "
                f"{CTSDATA_LABEL!r} row -- wrong --delimiter for this file? {line!r}"
            )
        passages.append(CitedPassage(context=urn, text=passage_text))

    if not seen:
        raise ValueError(f"file has no {CTSDATA_LABEL!r} block")

    return passages


def read_cex_passages(path: str, delimiter: str = "#") -> List[CitedPassage]:
    """Read `path` as a CEX file and return parse_cex_ctsdata() of its
    contents -- see that function's own docstring for the exact format
    and error cases."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return parse_cex_ctsdata(text, delimiter=delimiter)
