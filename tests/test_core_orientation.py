"""Offline tests for aat.core.orientation.normalize_orientation() -- the
validation shared by graph_to_mermaid() and graph_to_dot(). Each of those
renderers has its own orientation tests too (default value, verbatim vs.
mapped output), but this file covers the shared validation rule itself
directly, mirroring how test_core_coloring.py tests assign_action_colors()
directly rather than only through graph_to_mermaid()."""

import pytest

from aat.core.orientation import VALID_ORIENTATIONS, normalize_orientation


def test_valid_orientations_is_the_expected_five_codes():
    assert VALID_ORIENTATIONS == {"TB", "TD", "BT", "RL", "LR"}


@pytest.mark.parametrize("orientation", ["TB", "TD", "BT", "RL", "LR"])
def test_every_valid_orientation_round_trips_unchanged(orientation):
    assert normalize_orientation(orientation) == orientation


def test_normalization_is_case_insensitive():
    assert normalize_orientation("lr") == "LR"
    assert normalize_orientation("Bt") == "BT"


def test_normalization_strips_surrounding_whitespace():
    assert normalize_orientation("  TB  ") == "TB"


def test_invalid_orientation_raises_value_error_naming_valid_options():
    with pytest.raises(ValueError) as excinfo:
        normalize_orientation("sideways")
    message = str(excinfo.value)
    assert "sideways" in message
    for valid in ("TB", "TD", "BT", "RL", "LR"):
        assert valid in message
