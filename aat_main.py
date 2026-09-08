"""
A runnable script to run the aat.english pipeline from the command line.

Needs an `.env` file in this folder with your LM credentials -- see
.env.example and USAGE.md.
"""

import argparse
import os
import sys
from pathlib import Path

import dspy
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).with_name(".env"))


def _env(name: str, fallback_name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    value = os.getenv(fallback_name)
    if value:
        return value
    return default


def _configure_lm():
    api_base = _env("API_BASE", "API_BASE", "https://suarezai.holycross.edu/litellm")
    model = _env("MODEL", "MODEL", "litellm_proxy/anthropic/Claude Opus 5")

    # Distinguish "API_KEY isn't in .env at all" (a likely oversight -- keep
    # raising) from "API_KEY= is there but deliberately empty" (fine for a
    # local, unauthenticated model like Ollama). _env()'s own truthiness
    # check can't tell these apart (both look "falsy"), so this checks
    # os.environ directly instead.
    if "API_KEY" not in os.environ:
        raise RuntimeError(
            "Missing API key. Set API_KEY in your .env file -- an empty "
            "value (API_KEY=) is fine for a local model that doesn't need "
            "one, e.g. Ollama; this only checks that the line exists at all."
        )
    api_key = os.environ["API_KEY"]

    # Only pass api_key through when it's actually non-empty. dspy.LM/litellm
    # don't need one at all for a local Ollama daemon -- passing api_key=""
    # explicitly is unnecessary and, depending on the provider, can behave
    # differently than omitting it outright.
    lm_kwargs = dict(model=model, api_base=api_base)
    if api_key:
        lm_kwargs["api_key"] = api_key

    lm = dspy.LM(**lm_kwargs)
    dspy.configure(lm=lm)
    return lm


from aat.core import AATGraph, CitedPassage, serialize_analysis  # noqa: E402
from aat.english import analyze_passage  # noqa: E402


def _write_serialized_analysis(passage: CitedPassage, graph: AATGraph) -> None:
    """Write the analysis to stdout in the same plain-text format
    aat.core.serialize_analysis()/write_analysis() use: a `#!passages`
    block (this passage's own context/text) followed by a `#!aatnodes`
    block (the resulting AATGraph). Prints nothing else to stdout, so the
    output can be redirected straight to a file --

        python3 aat_main.py --passage "..." > analysis.txt

    -- and reloaded later with aat.core.read_analysis() (or the
    aat_reader.py notebook), with no separate save step needed.
    """
    sys.stdout.write(serialize_analysis([passage], graph))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run an Agent-Action-Target analysis of an English passage."
    )
    parser.add_argument(
        "--passage",
        default="Four score and seven years ago our fathers brought forth, upon this continent, a new nation, conceived in Liberty, and dedicated to the proposition that all men are created equal.",
        help="English passage to analyze (defaults to the built-in sample).",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Optional context reference for the passage (e.g. a CTS URN). Defaults to empty.",
    )
    args = parser.parse_args()

    _configure_lm()
    _tokens, graph = analyze_passage(args.passage, context=args.context)
    _write_serialized_analysis(CitedPassage(context=args.context, text=args.passage), graph)
