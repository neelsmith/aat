"""
Citable text and tokens: the input side of the AAT model (aat-model.md,
"Context and tokens"), independent of any particular language or of how a
token list is derived (deterministic tokenizer, LLM-driven segmentation,
hand authoring, ...).

A CitedPassage pairs a passage's own text with a reference identifying its
context (aat-model.md's example: a CTS URN, though any string a caller
finds meaningful works equally well -- nothing here assumes CTS
specifically). Tokenizing a CitedPassage produces a list of CitableToken,
each one recording the context it came from, an id unique *within* that
context (not globally -- see CitableToken's own docstring), and the
token's own string value.
"""

from pydantic import BaseModel, Field


class CitedPassage(BaseModel):
    """One citable passage of text: a reference identifying its context
    (aat-model.md's example is a CTS URN, but any stable identifier a
    caller chooses works -- nothing here assumes CTS specifically) paired
    with the passage's own raw text."""

    context: str = Field(description="Reference identifying this passage's context, e.g. a CTS URN.")
    text: str = Field(description="This passage's raw text, exactly as written.")


class CitableToken(BaseModel):
    """One token belonging to a citable passage.

    `id` is unique *within* `context`, not globally -- aat-model.md's own
    worked example uses a per-context sequence like 't1', 't2', ...,
    starting over for each new context. A caller working across multiple
    contexts should treat (context, id) as the real compound key, not id
    alone -- see AATGraph.by_id().
    """

    context: str = Field(description="Context reference this token belongs to, matching its source CitedPassage.")
    id: str = Field(description="Token id, unique within `context` (e.g. 't1', 't2', ...).")
    value: str = Field(description="The token's own string value (its surface text).")
