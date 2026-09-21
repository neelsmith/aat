# aat — read this first

Full working conventions for this repo (write restrictions, operating modes, verification steps, and a session log) live in **[`notes/CLAUDE_WORKFLOW.md`](notes/CLAUDE_WORKFLOW.md)**. Read it before doing anything in this repo.

Quick summary (see that file for the full version and the reasoning behind it):

- `README.md`, `releases.md`, and `quarto/` are Neel's — read-only for Claude.
- `notes/` is where Claude's own notes and documentation go.
- `docs/` is generated output (via `quarto render` and `utilities/build_docs.py`) for GitHub Pages — not hand-edited, and `build_docs.py` wipes the directory before regenerating, so flag that before running it.
- Never `git commit` or push on Neel's behalf.
