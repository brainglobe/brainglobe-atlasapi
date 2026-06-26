"""Unit tests for _validate_atlas_name in wrapup.py."""

import pytest

from brainglobe_atlasapi.atlas_generation.wrapup import _validate_atlas_name


def test_validate_atlas_name_valid_lowercase():
    """Lowercase atlas name passes validation without raising."""
    try:
        _validate_atlas_name("allen_mouse_25um")
    except ValueError:
        pytest.fail("Valid lowercase name raised ValueError unexpectedly")


def test_validate_atlas_name_uppercase_raises():
    """Uppercase characters in atlas name raise ValueError."""
    with pytest.raises(ValueError, match="must be lowercase"):
        _validate_atlas_name("Allen_Mouse_25um")


def test_validate_atlas_name_mixed_case_raises():
    """Mixed case atlas name raises ValueError."""
    with pytest.raises(ValueError, match="must be lowercase"):
        _validate_atlas_name("allen_Mouse_25um")


def test_validate_atlas_name_all_caps_raises():
    """All caps atlas name raises ValueError."""
    with pytest.raises(ValueError, match="must be lowercase"):
        _validate_atlas_name("ALLEN_MOUSE_25UM")
