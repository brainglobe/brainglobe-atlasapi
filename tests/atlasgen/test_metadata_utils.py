import numpy as np
import pytest

from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    generate_metadata_dict,
)


@pytest.fixture
def metadata_input_template():
    return {
        "name": "author_species",
        "citation": "Lazcano, I. et al. 2021, https://doi.org/10.1038/s41598-021-89357-3",
        "atlas_link": "https://zenodo.org/records/4595016",
        "species": "Ambystoma mexicanum",
        "symmetric": False,
        "resolution": (40, 40, 40),
        "orientation": "lpi",
        "version": "1.1",
        "shape": (172, 256, 154),
        "transformation_mat": np.array(
            [
                [0.000e00, -1.000e00, 0.000e00, 1.024e04],
                [0.000e00, 0.000e00, -1.000e00, 6.160e03],
                [-1.000e00, 0.000e00, 0.000e00, 6.880e03],
                [0.000e00, 0.000e00, 0.000e00, 1.000e00],
            ]
        ),
        "additional_references": [],
        "atlas_packager": "people who packaged the atlas",
    }


def test_generate_metadata_dict(metadata_input_template):
    """Test generate_metadata_dict using metadata_input_template."""
    output = generate_metadata_dict(**metadata_input_template)
    for key in metadata_input_template:
        if key != "transformation_mat":
            assert (
                output[key] == metadata_input_template[key]
            ), f"Field '{key}' has changed unexpectedly."
    assert output["trasform_to_bg"] == tuple(
        [tuple(m) for m in metadata_input_template["transformation_mat"]]
    )


@pytest.mark.parametrize(
    ["metadata", "error"],
    [
        pytest.param(
            {"name": "author1_author2_institute_species"},
            None,
            id="name=author1_author2_institute_species",
        ),
        pytest.param(
            {"name": "authorspecies"},
            AssertionError,
            id="name=authorspecies (error)",
        ),
        pytest.param(
            {"citation": "Axolotl et al., 2025"},
            AssertionError,
            id="citation without doi (error)",
        ),
        pytest.param(
            {"citation": "unpublished"},
            None,
            id="citation=unpublished",
        ),
        pytest.param(
            {"atlas_link": "invalid"},
            None,  # TODO: Change to InvalidURL after atlas_link url testing
            # is addressed in generate_metadata_dict
            id="atlas_link=invalid",
        ),
        pytest.param(
            {"symmetric": "True"},
            AssertionError,
            id="symmetric=string (error)",
        ),
        pytest.param(
            {"symmetric": False},
            None,
            id="symmetric=bool (False)",
        ),
        pytest.param(
            {"resolution": (40, 40)},
            AssertionError,
            id="2D resolution (error)",
        ),
        pytest.param(
            {"shape": (172, 256)},
            AssertionError,
            id="len(shape)>3 (error)",
        ),
        pytest.param(
            {"additional_references": "not a list"},
            AssertionError,
            id="additional_references is not a list but a string (error)",
        ),
    ],
)
def test_generate_metadata_dict_errors(
    metadata, error, metadata_input_template
):
    """Test generate_metadata_dict error raising with modified metadata."""
    metadata_input_template.update(metadata)
    if error is not None:
        with pytest.raises(error):
            generate_metadata_dict(**metadata_input_template)
    else:
        generate_metadata_dict(**metadata_input_template)
