"""
Render a list of CitableToken as one continuous HTML string, with AAT
graph nodes highlighted -- the HTML counterpart to aat.core.mermaid's
Mermaid diagram, sharing the same node coloring so a passage's rendered
text and its Mermaid diagram agree visually (see aat.core.coloring's own
docstring, and arsgrammatica's rendering.py/verbal_units.py, which this
module's spacing and highlighting logic mirrors).

Two things this module does, independently of each other:

- **Whitespace reconstruction**: `" ".join(t.value for t in tokens)` puts
  a space before every token including punctuation, so "The dog ." would
  round-trip wrong. Every token is classified as one of:

  - **right-joining**: no space between it and the token that FOLLOWS
    it. Opening brackets (`(`, `[`, `{`), and the first quote of a quote
    pair (see below), are right-joining. A right-joining token is itself
    always preceded by a space (unless it's the very first token).
  - **left-joining**: attaches directly to whatever precedes it, no
    space, ever. The default for punctuation -- periods, commas,
    semicolons, hyphens, closing brackets -- and also the second quote
    of a quote pair.
  - **normal** (anything starting with a letter or digit -- ordinary
    words and numbers): gets a space before it, UNLESS the immediately
    preceding token was right-joining, in which case it attaches
    directly with no space.

  **Quote pairing** assumes non-nested pairs of the same literal
  character: the Nth occurrence of `"` (or, independently, of `'`) is
  right-joining if N is odd (an opening quote) and left-joining if N is
  even (a closing quote) -- handles the realistic case of alternating
  same-glyph quotes marking separate quoted spans, but not curly/
  directional quotes ("“"/"”", "‘"/"’"), each of which is
  unambiguous on its own and doesn't need this counting trick.

  aat.english.tokenize's regex never splits off an enclitic (unlike
  arsgrammatica's Latin tokenizer), so there's no separate enclitic case
  here -- every token is normal-word or single-character punctuation.

- **Highlighting**: a token whose (context, id) matches a node in
  `graph` is wrapped in a `<span>` colored the same way
  aat.core.mermaid.graph_to_mermaid() colors that node (via
  aat.core.coloring.assign_action_colors() -- same palette, same
  first-appearance ordering, so a passage rendered here and its own
  Mermaid diagram agree), with a border style keyed on the node's role:
  a box around an `action` token, a rounded box around an `agent`
  token, and an underline on a `target` token.

  Note that for a *compound* action (aat-model.md's "Actions" section,
  e.g. "was eating"), the AATNode's `id` is only the principal verb
  token's id ("eating"), not every component token's -- so only that one
  token gets highlighted here, not "was" too. Nothing in the current AAT
  graph shape records which other tokens make up a compound action, so
  this module can't recover them.
"""

import html
from typing import Dict, List, Optional, Tuple

from ..core.coloring import assign_action_colors
from ..core.graph import AATGraph
from ..core.tokens import CitableToken

_OPENING_BRACKETS = {"(", "[", "{"}
_CLOSING_BRACKETS = {")", "]", "}"}
_QUOTE_CHARS = {'"', "'"}

_LEFT = "left"
_RIGHT = "right"
_NORMAL = "normal"


def _classify(token: CitableToken, quote_counts: Dict[str, int]) -> str:
    """Return this token's join behavior: one of _LEFT, _RIGHT, or
    _NORMAL. `quote_counts` is mutated in place to track how many times
    each quote character has been seen so far, across the whole call to
    tokens_to_html() -- it must be threaded through in token order."""
    text = token.value

    if text and text[0].isalnum():
        return _NORMAL

    if text in _OPENING_BRACKETS:
        return _RIGHT
    if text in _CLOSING_BRACKETS:
        return _LEFT
    if text in _QUOTE_CHARS:
        quote_counts[text] = quote_counts.get(text, 0) + 1
        return _RIGHT if quote_counts[text] % 2 == 1 else _LEFT

    # Periods, commas, semicolons, hyphens, and any other punctuation not
    # called out above all default to left-joining.
    return _LEFT


# Border styles keyed by AATNode.role, applied on top of a highlighted
# token's background/text color. `{stroke}` is filled in with that
# node's own cluster stroke color (assign_action_colors()'s second
# element of each color triple), so a token's border matches its
# highlight rather than using one fixed color for every role.
_ROLE_STYLE = {
    "action": "border: 2px solid {stroke}; border-radius: 2px; padding: 0 0.15em;",
    "agent": "border: 2px solid {stroke}; border-radius: 0.8em; padding: 0 0.4em;",
    "target": "border-bottom: 3px solid {stroke}; padding-bottom: 0.05em;",
}


def tokens_to_html(tokens: List[CitableToken], graph: Optional[AATGraph] = None) -> str:
    """Render `tokens` as one continuous HTML string: ordinary words get
    a space before them, punctuation attaches per this module's own
    docstring (left-joining by default; opening brackets and the first
    of a quote pair right-join instead).

    If `graph` is given, every token whose (context, id) matches a node
    in `graph` is highlighted: background/text colored the same way
    that node is colored in `aat.core.mermaid.graph_to_mermaid()`'s
    diagram (via `aat.core.coloring.assign_action_colors()`), with a
    role-specific border -- a box around an `action` token, a rounded
    box around an `agent` token, an underline under a `target` token.
    Omit `graph` (or pass `None`) for plain, unhighlighted text.

    Every token's text is HTML-escaped before being emitted, span or
    not -- real passage text can contain a literal `<`, `>`, `&`, or
    quote character, which would otherwise be indistinguishable from
    markup to anything that re-parses this output.
    """
    color_of_node: Dict[Tuple[str, str], Tuple[str, str, str]] = {}
    role_of_node: Dict[Tuple[str, str], str] = {}
    if graph is not None:
        color_of_node, _warnings = assign_action_colors(graph)
        role_of_node = {(n.context, n.id): n.role for n in graph.nodes}

    quote_counts: Dict[str, int] = {}
    pieces: List[str] = []
    previous_class: Optional[str] = None

    for token in tokens:
        cls = _classify(token, quote_counts)
        rendered = html.escape(token.value)

        key = (token.context, token.id)
        color = color_of_node.get(key)
        if color is not None:
            fill, stroke, text_color = color
            role = role_of_node.get(key, "")
            role_style = _ROLE_STYLE.get(role, "").format(stroke=stroke)
            rendered = (
                f'<span style="background-color: {fill}; color: {text_color}; {role_style}">'
                f"{rendered}</span>"
            )

        if not pieces:
            # Nothing precedes the first token -- never prepend a space.
            pieces.append(rendered)
        elif cls == _LEFT:
            pieces.append(rendered)
        elif cls == _RIGHT:
            pieces.append(" " + rendered)
        else:  # _NORMAL
            if previous_class == _RIGHT:
                pieces.append(rendered)
            else:
                pieces.append(" " + rendered)

        previous_class = cls

    return "".join(pieces)
