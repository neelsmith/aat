# Working with Claude on `aat`

This project gets ongoing help from Claude (Anthropic's assistant) across many separate sessions. Sessions don't share memory with each other except through whatever's written down -- conversation summaries carried forward by the tool are lossy and don't survive indefinitely, but this repo does. This file is where that continuity actually lives: how a session can reach this repo, what it delivers and how, and the standing constraints that apply regardless of which mode a given session is in. Read it at the start of any session picking this project back up, and add to it (see "Session log" below) at the end of any session that did something worth a future session -- or Neel -- knowing about.


## Two operating modes, and what changes because of them

A Claude session's access to this repo varies from one session to the next, and isn't something either side chooses in advance -- it's a property of how the session happens to be connected when it starts, and it can change mid-session too.

**Device-linked.** The session has a live bridge to Neel's own machine and can read and write files there directly, in place (via a sandboxed shell on the device, not by re-typing file contents from memory). This is the fast path: edits land exactly where they belong, there's nothing to reconcile, and Neel sees changes in his own working tree as they happen. When this bridge exists, use it -- don't manufacture the cloud-only workflow below out of habit when direct access is available. Note that the device's own shell is a separate, isolated Linux sandbox, not Neel's real macOS environment directly -- a `.venv` built with his native Python won't run there, so verifying by running the test suite from that shell means installing deps fresh into that sandbox's own Python (network permitting), not reusing his existing virtualenv.

**Cloud-only.** No bridge to Neel's machine exists (or one existed earlier in a session and dropped). In this mode Claude has only its own disposable cloud sandbox, with network access to clone the *public* `neelsmith/aat` GitHub repo but no push credentials and no way to reach Neel's filesystem at all. The only honest way to deliver work from this mode is:

1. Clone `https://github.com/neelsmith/aat.git` fresh into the sandbox.
2. Make the change there, verify it thoroughly (see "Verifying before delivering" below) -- this mode has no shortcut back to "just try it in the real checkout," so verification has to be conclusive on its own.
3. Produce a patch: `git add -N <new files>` to include untracked new files in the diff without staging or committing anything, then `git diff -- <paths>` to capture the change as a plain unified diff. Split large, mechanically-regenerated output (e.g. rebuilt `docs/`) into its own patch, separate from the hand-written change, so Neel can review each on its own terms.
4. Sanity-check the patch by applying it to a *second*, independent fresh clone and re-running the test suite there -- confirms the patch is self-contained and applies cleanly outside the sandbox it was authored in, not just in the checkout that produced it.
5. Deliver the patch file(s) as a download, with an explanation of what's in each and how to apply them (`git apply <file>` from the repo root), and say plainly that this session had no direct access to his machine.

A patch delivered into the conversation only reaches Neel's repo once *he* applies it -- it does not land in his working tree on its own, and there's no way to check that it did from cloud-only mode. If a later message in the same session gains device access after having delivered a patch this way, check first whether Neel already applied it (`git status`/`git diff` against the patch's own file list) before writing anything, rather than assuming the patch is still unapplied.

If device access becomes available *before* delivering anything -- even if it wasn't available earlier in the same session -- prefer writing directly and skip the patch dance entirely; a mid-session change back to device-linked mode is a reason to switch modes, not a reason to keep the cloud-only workflow going out of momentum.


## Standing constraints for AI-assisted changes

These hold in *either* mode, and don't expire when a session's conversation history gets compacted or a new session starts:

- Never touch anything under `quarto/` -- Neel's hand-written project website.
- Never touch `README.md` -- Neel hand-edits this himself.
- Never touch `releases.md` -- same, Neel's own.
- Never run `git commit` (or push) on Neel's behalf, under any circumstances. Every change -- whether it's a direct edit in device-linked mode or a patch in cloud-only mode -- is left for Neel to review and commit himself. This is exactly *why* cloud-only mode delivers patches rather than trying to find some other way to get a commit onto `main`.


## Verifying before delivering

Whichever mode produced a change, don't report it done until:

- The full `pytest` suite passes (see TESTING.md) -- not just the tests for the file touched, and run in an environment that actually has the project's dev dependencies installed (see the device-sandbox caveat above).
- Where the change produces output meant for some other real tool to consume -- a Mermaid diagram, a Graphviz digraph, an HTML page -- that output has actually been run through the real downstream tool at least once (e.g. Graphviz's own `dot` CLI, not just eyeballing the string), not merely unit-tested against assertions written by the same session that wrote the code.
- `docs/` regenerates cleanly via `utilities/build_docs.py` when the change touches anything with a public docstring, and (in cloud-only mode) that regeneration is delivered as its own patch, separate from the hand-written change, since it's large and mechanical.


## Session log

A short entry per session that changed something non-trivial or delivered work in cloud-only mode -- date, what happened, and where the result went. Not a substitute for git history; this is for the "why did I get a patch file" and "what was going on across sessions" questions git history doesn't answer on its own.

- **2026-09-07 -- Graphviz DOT rendering.** Started in cloud-only mode (no device bridge in the session's tool set at the time): cloned `neelsmith/aat` fresh, added `aat/core/graphviz.py` (`graph_to_dot()`/`save_dot()`, mirroring `graph_to_mermaid()`/`save_mermaid()`) and `aat/core/orientation.py` (orientation validation extracted out of `mermaid.py` so both renderers share it, same pattern as the earlier `coloring.py` extraction), plus tests and a new USAGE.md section. Verified the actual DOT output with the real `dot` CLI (`dot -Tsvg`), not just string assertions. Delivered as patches at first, since no device access was available -- but it turned out an *earlier* session (before this one's conversation history was compacted) had already written the core module changes (`graphviz.py`, `orientation.py`, the `mermaid.py`/`__init__.py` edits) directly to Neel's own checkout via a device bridge that was live at the time, so those patches duplicated work already sitting there uncommitted. Once device access came back mid-session, the remaining gap -- tests, the USAGE.md section, and this file -- was written directly to Neel's checkout instead, and the full suite was run there (`pip install -e ".[dev]"` into the device sandbox's own Python, not Neel's native `.venv` -- see the device-sandbox caveat above) rather than delivered as another patch. Lesson folded into the workflow above: check `git status` for a device-linked repo before assuming a prior patch is unapplied, and prefer direct writes the moment device access exists.
