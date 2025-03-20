from pathlib import Path

import pytest

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    read_itk_labels,
    split_label_text,
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


# TODO: Finish this test. This is only a skeleton
# TODO: Potentially remove or simplify "labels051224.txt"
def test_read_itk_labels():
    """Test reading ITK labels from a file."""
    itk_labels_file = Path(__file__).parent / "labels051224.txt"
    read_itk_labels(itk_labels_file)
