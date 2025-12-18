from unittest.mock import Mock

import pytest

from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    check_unique_acronyms,
)


def test_check_unique_acronyms_pass():
    """Test that check_unique_acronyms passes for an atlas with unique acronyms."""
    # Mock the atlas object with unique acronyms
    mock_atlas = Mock(
        structures={
            1: {"acronym": "A"},
            2: {"acronym": "B"},
            3: {"acronym": "C"},
        }
    )
    # The function should execute without raising an error
    check_unique_acronyms(mock_atlas)


def test_check_unique_acronyms_fail():
    """Test that check_unique_acronyms raises AssertionError for duplicate acronyms."""
    # Mock the atlas object with duplicate acronyms (ID 1 and 3 both have "A")
    mock_atlas = Mock(
        structures={
            1: {"acronym": "A"},
            2: {"acronym": "B"},
            3: {"acronym": "A"},
        }
    )
    # Expect the function to raise an AssertionError
    with pytest.raises(AssertionError) as excinfo:
        check_unique_acronyms(mock_atlas)

    # Check the error message content
    assert "Duplicate acronyms found" in str(excinfo.value)
    assert "['A']" in str(excinfo.value)
