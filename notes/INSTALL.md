# Installing `aat` from PyPI

Once published (see `notes/PUBLISHING.md`), `aat` installs from PyPI under
the distribution name **`aatgraph`** -- the import path is unaffected.

**Just the data model, serialization, and renderers** (`CitableToken`,
`AATGraph`, `read_analysis()`/`write_analysis()`, `graph_to_mermaid()`,
`graph_to_dot()`, `tokens_to_html()`) -- no dspy, no LM access required:

```bash
pip install aatgraph
```

```python
from aat import read_analysis, graph_to_mermaid, tokens_to_html

tokens, graph = read_analysis("analysis.txt")
print(graph_to_mermaid(graph))
```

**To also run the English AAT pipeline itself** (tokenizing and analyzing
passages with an LM via dspy):

```bash
pip install "aatgraph[english]"
```

```python
from aat.english import analyze_passage

tokens, graph = analyze_passage("The dog ate the homework.", context="urn:cts:...")
```

See `notes/USAGE.md` for the full API and worked examples.
