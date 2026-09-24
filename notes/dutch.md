Pro Arch 2: Alle kunsten en wetenschappen die tot de menselijke beschaving bijdragen, bezitten een gemeenschappelijke band

## Context and tokens

The dutch module works precisely analogously to the English module: it tokenizes a citable passage to produce a list of citable tokens. Each token keeps track of the context reference, has a unique ID within that context, and a string value for the token. 

Example: if a passage with context ID `proarchia_nl` has the text "Alle kunsten en wetenschappen bezitten een gemeenschappelijke band.", the first token in the list would  have `proarchia_nl` for its context, a value like "t1" for its contextually unique token ID, and "Alle" for its text content.


## The AAT graph for Dutch

From a list of citable tokens, `aat` then constructs a graph comprising a selection of tokens from the list. Nodes in the AAT graph keep the context, ID and string value of the token, and add a `role` property and `related_node` property, just as in English.


### Actions

*Actions* are verbal expressions. In Dutch these may be single tokens (like "bezitten") or compound forms with multiple tokens (like "wordt gevoerd"). In considering compound forms, we distinguish between the *principal verb* and auxiliaries. For a compound like "wordt gevoerd", the *principal verb* is "gevoerd"; "wordt" is an *auxiliary*.

Tokens for verbal expressions are extracted from the tokens list, and assigned the role `action`.  If the verbal expression is a single token, it directly uses the text value of the token.  Example: in "Alle kunsten en wetenschappen bezitten een gemeenschappelijke band.", the citable token for the verb of the independent clause "bezitten" is extracted, assigned the value `action` for `role`.


If the verbal expression is a compound form, it uses the ID of the *principal verb* of its ID, and concatenates the text values of all the component tokens joined by spaces. For example, the sentence, ""Hij heeft Cicero nooit gelezen." has a compound verb "heeft gelezen". The AAT graph will have a node with the value `action`, and the ID of the token "gelezen" for its token ID, but the string "heeft gelezen" for its text value.

Verbal expressions may be independent or dependent. For independent verbal expressions, the `related_node` property is `None`. Example: the sentence "Hij heeft Cicero nooit gelezen." has a single independent verbal expression "heeft gelezen" so will have  `None` for `related_node`. For *dependent* (or *subordinate*) expressions, the verb will have the ID of the node  in the graph for the governing verbal expression. Example: the sentence "Alle kunsten en wetenschappen die tot de menselijke beschaving bijdragen, bezitten een gemeenschappelijke band." has two verbal expressions, the *independent* expression "bezitten" and the *dependent* expression "bijdragen".  Both will be added to the ATT graph with value `action` for `role`. The node for "bezitten" will have `None` for `related_node`, but the node for "bijdragen" will have the ID of the token "bezitten" for `related_node`.

### Agents

*Agents* are either *subjects* of active voice verbs, intransitive verbs, or linking verbs; or the the *agent* of a passive voice verb. *Agent* nodes have the role `agent` and for `related_node` have the ID of the *action* node they relate to.


Examples: in "Hij heeft Cicero nooit gelezen.", the token "Hij" is the subject of the active verb "heeft gelezen". We add a node for "Hij" to the graph with `agent` for `role`,  and the id of the token for "gelezen" as its `related_node` value. 

In "De Engelse tekst werd vertaald door Jones.", the token "Jones" is the *agent* of the passive voice expression "werd vertaald". We add a node for "Jones" to the graph with `agent` for `role`, and the ID of "vertaald" (the *principal* verb in the compound expression) for `related_node`.


## Target

*Targets* are either *direct objects* of transitive active voice verbs, or *predicates* of linking verbs; or the *subject* of transitive passive verbs. *Target* nodes have the role `target` and for `related_node` have the ID of the *action* node they relate to.

Examples: in "Hij heeft Cicero nooit gelezen.", the token "Cicero" is the direct object of the active verb "gelezen". We add a node for "Cicero" to the graph with `target` for `role`,  and the id of the token for "gelezen" as its `related_node` value. In "De Engelse tekst werd vertaald door Jones.", the token "tekst" is the *subject* of the passive voice expression "werd vertaald". We add a node for "tekst" to the graph with `target` for `role`, and the ID of "vertaald" (the *principal* verb in the compound expression) for `related_node`.

