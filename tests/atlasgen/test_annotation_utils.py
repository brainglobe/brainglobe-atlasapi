import pytest

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
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
