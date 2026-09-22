# Publishing `aat` to PyPI (as `aatgraph`)

This records what's already set up and exactly what's left for Neel to do
himself -- registering a trusted publisher and cutting a release both
require logging into pypi.org/github.com directly, which a Claude session
can't do on your behalf.

## Why the PyPI name differs from the import name

PyPI's project name "aat" is already taken by an unrelated, inactive
package (`timkpaine/aat`, an algorithmic-trading library, last released
2019). Distribution name and import name are independent in Python
packaging, so this project publishes to PyPI as **`aatgraph`**
(`pip install aatgraph`) while every import anywhere -- in this codebase
and in any downstream project -- stays exactly `import aat`,
`from aat.core import ...`, `from aat.english import ...`. Nothing about
the package's own code or API changed for this; only `pyproject.toml`'s
`[project] name` field did.

## What's already done

- `pyproject.toml`: `name = "aatgraph"`; license declaration modernized to
  the current SPDX form (`license = "GPL-3.0-or-later"` +
  `license-files = ["LICENSE"]`, replacing the deprecated
  `license = {file = "LICENSE"}` + classifier pair -- `python -m build`
  was emitting deprecation warnings for the old form); a few more
  classifiers (`Operating System :: OS Independent`, explicit
  `Programming Language :: Python :: 3.10/3.11/3.12` matching
  `tests.yml`'s own test matrix, `Typing :: Typed`).
- `aat/py.typed` (empty marker file) + a `[tool.setuptools.package-data]`
  entry so it ships in the wheel -- a PEP 561 marker telling type
  checkers (mypy, pyright) to use `aat`'s own type hints rather than
  treating it as untyped. `aat` is fully type-hinted throughout, so this
  was free to add correctly.
- Verified locally (disposable venv, not committed): `python -m build`
  produces a clean `aatgraph-0.3.0-py3-none-any.whl` (pure Python, no
  warnings) and `aatgraph-0.3.0.tar.gz`; `twine check` passes on both;
  installing the built wheel into a fresh venv and running
  `import aat; from aat import tokens_to_html, graph_to_mermaid,
  read_analysis` all worked with **zero** dspy anywhere in `sys.modules`,
  and `import aat.english` fails cleanly with a plain `ModuleNotFoundError:
  No module named 'dspy'` (expected -- install `aatgraph[english]` for
  that).
- `.github/workflows/publish.yml`: builds the sdist+wheel, runs
  `twine check`, then publishes via PyPI's OIDC "Trusted Publishing" --
  no API token stored as a GitHub secret anywhere. Two triggers:
  - Publishing a **GitHub Release** publishes straight to the real PyPI
    index.
  - A manual **Actions -> Publish to PyPI -> Run workflow** dispatch lets
    you choose `testpypi` (a dry run against TestPyPI, no release needed)
    or `pypi` (e.g. to retry a failed publish without cutting a new
    release).

## What's left -- your steps, in order

1. **(Recommended first) Register a pending publisher on TestPyPI**, so
   you can dry-run the whole pipeline before it touches the real index.
   Log into <https://test.pypi.org> (a separate account/registration from
   pypi.org if you don't already have one), go to
   *Account settings -> Publishing*, and add a pending publisher:
   - PyPI project name: `aatgraph`
   - Owner: `neelsmith`
   - Repository name: `aat`
   - Workflow name: `publish.yml`
   - Environment name: `testpypi`

   Then, in the GitHub repo, go to *Actions -> Publish to PyPI -> Run
   workflow*, choose `testpypi`, and run it. If it succeeds, check
   <https://test.pypi.org/project/aatgraph/> -- that's the whole pipeline
   proven end to end without any risk to the real package name.

2. **Register a pending publisher on the real PyPI**, the same way, at
   <https://pypi.org/manage/account/publishing/>:
   - PyPI project name: `aatgraph`
   - Owner: `neelsmith`
   - Repository name: `aat`
   - Workflow name: `publish.yml`
   - Environment name: `pypi`

   You don't need to (and can't) create the `aatgraph` project on PyPI
   first -- registering it as a pending publisher is what lets the first
   successful run of this workflow create it.

3. **(Optional, recommended) Add a protection rule to the `pypi`
   environment** in the GitHub repo (*Settings -> Environments -> pypi*,
   created automatically the first time the workflow references it, or
   creatable manually now) -- e.g. "required reviewers: neelsmith" -- so
   a real publish to PyPI needs a manual approval click even if the
   workflow itself fires automatically. `testpypi` doesn't need this;
   mistakes there are free to undo.

4. **Cut the actual 0.3.0 release** once you're satisfied:
   - Bump `releases.md` (still says "Current release: *0.2.1*" with
     0.3.0 pending) -- this file is yours; not edited by this session.
   - `git tag v0.3.0 && git push origin v0.3.0`
   - On GitHub, *Releases -> Draft a new release*, pick the `v0.3.0` tag,
     write release notes (`releases.md`'s own 0.3.0 entry is a good
     starting point), and publish it. Publishing the release is what
     triggers `publish.yml` and sends the build to the real PyPI.

## One thing worth knowing, not fixed here

`README.md` becomes the PyPI project page's description, and it has two
GitHub-relative image links (`./quarto/imgs/4score.png` and
`./quarto/imgs/4score-hilites.png`). Those resolve fine on GitHub itself,
but PyPI has no way to know they're relative to this repo, so they'll
render as broken images on the `aatgraph` PyPI page. `README.md` is left
to you per your own standing instruction, so this wasn't touched -- the
straightforward fix, whenever you want it, is pointing those two links at
their raw GitHub URLs (`https://raw.githubusercontent.com/neelsmith/aat/main/quarto/imgs/...`)
instead of the relative paths.
