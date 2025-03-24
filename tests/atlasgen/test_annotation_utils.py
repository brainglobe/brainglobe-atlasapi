from pathlib import Path

import pytest

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    ITK_CLEAR_LABEL,
    ITK_SNAP_HEADER,
    read_itk_labels,
    split_label_text,
    write_itk_labels,
)


@pytest.mark.parametrize(
    "input_name, expected_name, expected_acronym",
    [
        pytest.param(
            "BrainGlobeAtlas Name (BGA-N)",
            "BrainGlobeAtlas Name",
            "BGA-N",
            id="name (acronym)",
        ),
        pytest.param(
            "BrainGlobeAtlas Name", "BrainGlobeAtlas Name", "B", id="name"
        ),
    ],
)
def test_split_label_text(input_name, expected_name, expected_acronym):
    """Test splitting label text into name and acronym.

    If there's no acronym, the name's first letter is used as acronym.
    """
    name, acronym = split_label_text(input_name)
    assert name == expected_name
    assert acronym == expected_acronym


@pytest.fixture
def itk_snap_labels():
    """Labels match those in dummy_itk_snap_labels.txt file."""
    return [
        {
            "id": 123,
            "name": "BrainGlobe",
            "rgb_triplet": (0, 255, 0),
            "acronym": "BG",
        },
        {
            "id": 456,
            "name": "Label without acronym",
            "rgb_triplet": (255, 145, 0),
            "acronym": "L",
        },
    ]


def test_read_itk_labels(itk_snap_labels):
    """Test reading ITK labels from a file."""
    itk_labels_file = (
        Path(__file__).parent / "dummy_data" / "dummy_itk_snap_labels.txt"
    )
    expected_labels = itk_snap_labels
    assert read_itk_labels(itk_labels_file) == expected_labels


def test_write_itk_labels(tmp_path, itk_snap_labels):
    """Test writing ITK labels to a file."""
    output_file = tmp_path / "output_itk_labels.txt"
    write_itk_labels(output_file, itk_snap_labels)

    file_content = output_file.read_text()
    assert ITK_SNAP_HEADER and ITK_CLEAR_LABEL in file_content
    assert read_itk_labels(output_file) == itk_snap_labels
