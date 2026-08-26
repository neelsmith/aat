# `aat`

This repository hosts a Python package implementing a reductive model of natural-language syntax called "agent-action-target", or AAT, documented here. 


## Context and tokens

The AAT model applies to citable passages of text. A citable passage of text has a reference identifying its context, such as a CTS URN, and text content.

The `aat` module first tokenizes the text content of a passage to produce a list of citable tokens. Each token keeps the context reference, has a unique ID within that context, and a string value for the token. Example: if a passage with ID "homework1" has the text "The dog ate my homework.", the first token in the list could be have "homework1" for its context, "t1" for its contextually unique token ID, and "The" for its homework.


## The AAT graph for English

From a list of citable tokens, `aat` then constructs a graph comprised of selected tokens in the graph. Nodes in the AAT graph keep the context, ID and string value of the token, and add a `role` property and `related_node` property.


### Actions

*Actions* are verbal expressions. In English these may be single tokens (like "ate") or compound forms with multiple tokens (like "was eating"). In considering compound forms, we distinguish between 
Tokens for verbal expressions are extracted from the tokens list, and assigned the role `action`. Verbal expressions may be independent or dependent. For independent verbal expressions, the `related_node` property is `None`.

Example: "The dog ate my homework.", the citable token for the verb of the independent clause "ate" is extracted, assigned the value `action` for `role`, and `None` for `related_node`.

### Agents

*Agents* are either *subjects* of active voice verbs, intransitive verbs, or linking verbs; or the the *agent* of a passive voice verb. *Agent* nodes have the role `agent` and for `related_node` have the ID of the *action* node they relate to.
Examples: in "The dog ate my homework.", the token "dog" is the subject of the active verb "ate". We add a node for "dog" to the graph with `agent` for `role`,  and the id of the token for "ate" as its `related_node` value. In "The homework was eaten by the dog.", the token "dog" is the *agent* of the passive voice expression "was eaten".


