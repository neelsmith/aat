# aat

> *See [release history](https://github.com/neelsmith/aat/blob/main/releases.md)*.


`aat` is a python package leveraging LMs with `dspy` to apply a reductive model of natural-language syntax called Agent-Action-Target (AAT) to citable text in English. The AAT model is documented in [`aat-model.md`](https://github.com/neelsmith/aat/blob/main/aat-model.md).

The package is deliberately split into two independent modules:

- **`aat.core`** implements the AAT model with classes for input (`CitableToken`/`CitedPassage`) and output (`AATNode`/`AATGraph`). It includes functions to validate and serialize analyses to a plain-text format.
- **`aat.english`** uses an LM  configured with `dspy` to tokenize English text and compose an `AATGraph`.

Released under the [GNU General Public License v3 or later](LICENSE).


## Installing

To use the `aat` model from another project, install the core module directly from this repository:

```sh
pip install git+https://github.com/neelsmith/aat.git
```

To include `aat.english` (the dspy-based English pipeline):

```sh
pip install "aat[english] @ git+https://github.com/neelsmith/aat.git"
```

To pin to a specific branch, tag, or commit  appending `@<ref>` to the URL, e.g.

```sh
pip install "aat[english] @ git+https://github.com/neelsmith/aat.git@v0.1.0`
```





See the [project issue tracker](https://github.com/neelsmith/aat/issues) for known gaps and work in progress or to submit an issue.


