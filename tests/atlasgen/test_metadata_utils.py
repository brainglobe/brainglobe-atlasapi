import json
from datetime import datetime

import numpy as np
import pytest
from requests.exceptions import InvalidURL

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    create_metadata_files,
    create_readme,
    create_structures_csv,
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
            InvalidURL,
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


@pytest.fixture
def structures():
    structure101 = {
        "id": 101,
        "acronym": "o",
        "name": "olfactory system",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 999, 101],
    }
    structure1 = {
        "id": 1,
        "acronym": "on",
        "name": "olfactory nerve",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 101, 1],
    }
    structure5 = {
        "id": 5,
        "acronym": "aon",
        "name": "anterior olfactory nucleus",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 101, 5],
    }

    root = {
        "name": "root",
        "acronym": "root",
        "id": 999,
        "rgb_triplet": [255, 255, 255],
        "structure_id_path": [999],
    }

    return [structure101, structure1, structure5, root]


def get_root_id(structures: list[dict]):
    """Helper function to get root id from a list of structures."""
    for s in structures:
        if s["name"] == "root":
            return s["id"]


def test_create_metadata_files(
    structures,
    metadata_input_template,
    tmp_path,
):
    """Test create_metadata_files."""

    metadata = generate_metadata_dict(**metadata_input_template)

    # json is expected to be present in the dest_dir
    with open(tmp_path / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures, f)

    create_metadata_files(
        dest_dir=tmp_path,
        metadata_dict=metadata,
        structures=structures,
        root_id=get_root_id(structures),
        additional_metadata={},
    )

    assert (tmp_path / "structures.csv").exists(), "structures.csv missing"
    assert (tmp_path / "README.txt").exists(), "readme.txt missing"


def test_create_structures_csv(structures, tmp_path):
    """Test create_structures_csv."""

    root = get_root_id(structures)

    # json is expected to be present in the dest_dir
    with open(tmp_path / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures, f)

    create_structures_csv(uncompr_atlas_path=tmp_path, root=root)
    with open(tmp_path / "structures.csv", "r") as structures_file:
        generated_structures_csv = structures_file.read()

    expected_colnames = "id,acronym,name,structure_id_path,parent_structure_id"
    expected_row1 = f"aon,anterior olfactory nucleus,/{root}/101/5/,101.0"
    assert expected_colnames in generated_structures_csv
    assert expected_row1 in generated_structures_csv


def test_create_readme(
    structures,
    metadata_input_template,
    tmp_path,
):
    """Test create_readme."""

    metadata = generate_metadata_dict(**metadata_input_template)

    create_readme(
        uncompr_atlas_path=tmp_path,
        metadata_dict=metadata,
        structures=structures,
    )

    with open(tmp_path / "README.txt", "r", encoding="utf-8") as readme:
        generated_readme = readme.read()

    expected_date = datetime.today().strftime("%d/%m/%Y")
    expected_structure_tree = (
        "TREE --\nroot (999)\n└── o (101)\n    ├── aon (5)\n    └── on (1)\n"
    )

    assert expected_date in generated_readme
    assert expected_structure_tree in generated_readme
    assert metadata["citation"] in generated_readme
