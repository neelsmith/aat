"""
Shared test helpers.

Kept deliberately small: every DummyLM-backed test reconfigures
dspy.settings.lm before it runs, so there's no cross-test state to tear
down here (yet). If that changes, this is the place for a pytest fixture
that resets dspy.settings between tests.
"""

import os
from pathlib import Path

import dspy
import pytest
from dotenv import load_dotenv
from dspy.utils.dummies import DummyLM

from aat.english import analyze


def run_gold_example(example):
    """Run a GoldExample's passage through analyze(), with DummyLM
    standing in for the real LM and returning that example's
    canned_answer (see fixtures/gold_examples.py). Returns (tokens,
    result) -- tokens from the example's own tokenize()-derived
    GoldExample.tokens()."""
    dspy.configure(lm=DummyLM([example.canned_answer]))
    tokens = example.tokens()
    result = analyze(passage=example.passage, tokens=tokens)
    return tokens, result


@pytest.fixture
def real_lm():
    """Configure dspy against the actual model from .env, for tests
    marked `live` -- e.g. `@pytest.mark.live` plus `def test_x(real_lm):
    ...`. No DummyLM-backed test should request this fixture; it's the
    one place a test actually calls out to the configured LM instead of a
    canned answer. Skips (not errors) if no API key is configured, so
    `pytest -m live` degrades gracefully rather than failing on missing
    credentials.
    """
    load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")
    api_key = os.getenv("API_KEY")
    if not api_key:
        pytest.skip("no API_KEY in .env -- skipping live test")

    api_base = os.getenv("API_BASE", "https://suarezai.holycross.edu/litellm")
    model = os.getenv("MODEL", "litellm_proxy/anthropic/Claude Opus 5")
    dspy.configure(lm=dspy.LM(model=model, api_base=api_base, api_key=api_key))
