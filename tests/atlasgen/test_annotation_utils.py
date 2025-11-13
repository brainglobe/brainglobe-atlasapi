"""Tests for annotation utility functions."""

from pathlib import Path

import pytest

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    ITK_CLEAR_LABEL,
    ITK_SNAP_HEADER,
    read_itk_labels,
    split_label_text,
    write_itk_labels,
)


def test_split_label_text_name_square_acronym_square():
    """Test splitting label text with acronym in square brackets."""
    input_name = "BrainGlobeAtlas-Name (BGA-N)"
    expected_name = "BrainGlobeAtlas-Name"
    expected_acronym = "BGA-N"
    name, acronym = split_label_text(input_name)
    assert name == expected_name
    assert acronym == expected_acronym


def test_split_label_text_no_acronym_length_specified():
    """Test splitting label text without acronym, default length 1."""
    input_name = "BrainGlobeAtlas-Name"
    expected_name = "BrainGlobeAtlas-Name"
    expected_acronym = "B"
    name, acronym = split_label_text(input_name)
    assert name == expected_name
    assert acronym == expected_acronym


def test_split_label_text_acronym_length_specified():
    """Test splitting label text without acronym, specified length 3."""
    input_name = "BrainGlobeAtlas-Name"
    expected_name = "BrainGlobeAtlas-Name"
    expected_acronym = "Bra"
    acronym_length = 3
    name, acronym = split_label_text(input_name, acronym_length)
    assert name == expected_name
    assert acronym == expected_acronym


def test_split_label_text_acronym_length_too_long():
    """Test error when acronym length exceeds name length."""
    input_name = "BrainGlobeAtlas-Name"
    acronym_length = len(input_name) + 1
    with pytest.raises(ValueError) as exc_info:
        split_label_text(input_name, acronym_length)
    assert "Acronym length cannot be longer than the name itself." in str(
        exc_info.value
    )


@pytest.fixture
def itk_snap_labels():
    """Define ITK-SNAP labels that match the dummy file content.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary represents an
        ITK-SNAP label with its id, name, RGB triplet, and acronym.
    """
    return [
        {
            "id": 123,
            "name": "BrainGlobe",
            "rgb_triplet": [0, 255, 0],
            "acronym": "BG",
        },
        {
            "id": 456,
            "name": "Label without acronym",
            "rgb_triplet": [255, 145, 0],
            "acronym": "L",
        },
    ]


def test_read_itk_labels(itk_snap_labels):
    """Test reading ITK labels from a file.

    Parameters
    ----------
    itk_snap_labels : list of dict
        A list of dictionaries representing expected ITK-SNAP labels.
    """
    itk_labels_file = (
        Path(__file__).parent / "dummy_data" / "dummy_itk_snap_labels.txt"
    )
    expected_labels = itk_snap_labels
    assert read_itk_labels(itk_labels_file) == expected_labels


def test_write_itk_labels(tmp_path, itk_snap_labels):
    """Test writing ITK labels to a file.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest fixture.
    itk_snap_labels : list of dict
        A list of dictionaries representing ITK-SNAP labels to write.
    """
    output_file = tmp_path / "output_itk_labels.txt"
    write_itk_labels(output_file, itk_snap_labels)

    file_content = output_file.read_text()
    assert ITK_SNAP_HEADER and ITK_CLEAR_LABEL in file_content
    assert read_itk_labels(output_file) == itk_snap_labels
