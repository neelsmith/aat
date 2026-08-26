# The Agent-Action-Target model

This repository hosts a Python package implementing a reductive model of natural-language syntax called "agent-action-target", or AAT, documented here. 


## Context and tokens

The AAT model applies to citable passages of text. A citable passage of text has a reference identifying its context, such as a CTS URN, and text content.

The `aat` module first tokenizes the text content of a passage to produce a list of citable tokens. Each token keeps track of the context reference, has a unique ID within that context, and a string value for the token. Example: if a passage with context ID `homework1` has the text "The dog ate my homework.", the first token in the list would  have `homework1` for its context, a value like "t1" for its contextually unique token ID, and "The" for its text content.


## The AAT graph for English

From a list of citable tokens, `aat` then constructs a graph comprising a selection of tokens from the list. Nodes in the AAT graph keep the context, ID and string value of the token, and add a `role` property and `related_node` property.


### Actions

*Actions* are verbal expressions. In English these may be single tokens (like "ate") or compound forms with multiple tokens (like "was eating"). In considering compound forms, we distinguish between the *principal verb* and auxiliaries. For a compound like "was eating", the *principal verb* is "eating"; "was" is an *auxiliary*.

Tokens for verbal expressions are extracted from the tokens list, and assigned the role `action`.  If the verbal expression is a single token, it directly uses the text value of the token.  Example: in "The dog ate my homework.", the citable token for the verb of the independent clause "ate" is extracted, assigned the value `action` for `role`.

If the verbal expression is a compound form, it uses the ID of the *principal verb* of its ID, and concatenates the text values of all the component tokens joined by spaces. For example, the sentence, "The homework was not yet being eaten." has a compound verb "was being eaten". The AAT graph will have a node with the value `action`, and the ID of the token "eaten" for its token ID, but the string "was being eaten" for its text value.

Verbal expressions may be independent or dependent. For independent verbal expressions, the `related_node` property is `None`. Example: the sentence "The dog ate my homework." has a single independent verbal expression "ate" so will have  `None` for `related_node`. For *dependent* (or *subordinate*) expressions, the verb will have the ID of the node  in the graph for the governing verbal expression. Example: the sentence "He said that the dog ate his homework." has two verbal expressions, the *independent* expression "said" and the *dependent* expression "ate".  Both will be added to the ATT graph with value `action` for `role`. The node for "said" will have `None` for `related_node`, but the ndoe for "ate" will have the ID of the token "said" for `related_node`.

### Agents

*Agents* are either *subjects* of active voice verbs, intransitive verbs, or linking verbs; or the the *agent* of a passive voice verb. *Agent* nodes have the role `agent` and for `related_node` have the ID of the *action* node they relate to.
Examples: in "The dog ate my homework.", the token "dog" is the subject of the active verb "ate". We add a node for "dog" to the graph with `agent` for `role`,  and the id of the token for "ate" as its `related_node` value. In "The homework was eaten by the dog.", the token "dog" is the *agent* of the passive voice expression "was eaten". We add a node for "dog" to the graph with `agent` for `role`, and the ID of "eaten" (the *principal* verb in the compound expression) for `related_node`.


## Target

*Targets* are either *direct objects* of transitive active voice verbs, or *predicates* of linking verbs; or the *subject* of transitive passive verbs. *Target* nodes have the role `target` and for `related_node` have the ID of the *action* node they relate to.
Examples: in "The dog ate my homework.", the token "homework" is the direct object of the active verb "ate". We add a node for "homework" to the graph with `target` for `role`,  and the id of the token for "ate" as its `related_node` value. In "The homework was eaten by the dog.", the token "homework" is the *subject* of the passive voice expression "was eaten". We add a node for "homework" to the graph with `target` for `role`, and the ID of "eaten" (the *principal* verb in the compound expression) for `related_node`.
