"""Tests for metadata utility functions."""

import json
from datetime import datetime

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
    """Provide a template dictionary for metadata input.

    Returns
    -------
    dict
        A dictionary containing template metadata.
    """
    return {
        "name": "author_species",
        "citation": "Lazcano, I. et al. 2021, https://doi.org/10.1038/s41598-021-89357-3",
        "atlas_link": "https://zenodo.org/records/4595016",
        "species": "Ambystoma mexicanum",
        "symmetric": False,
        "resolution": [40, 40, 40],  # Keep as list/tuple input type
        # orientation is be "asr", wrapup.py will pass "asr" after reorienting
        "orientation": "asr",
        "version": "1.1",
        "shape": [172, 256, 154],  # Keep as list/tuple input type
        "additional_references": [],
        "atlas_packager": "people who packaged the atlas",
    }


def test_generate_metadata_dict(metadata_input_template):
    """Test `generate_metadata_dict` using `metadata_input_template`.

    Parameters
    ----------
    metadata_input_template : dict
        A template dictionary for metadata input.
    """
    input_data = metadata_input_template.copy()
    output = generate_metadata_dict(**input_data)

    assert isinstance(output, dict)

    for key in input_data:  # Iterate through keys expected based on input
        # Assert key presence
        assert key in output, f"Expected key '{key}' missing in output"

        # Assert value correctness, handling type conversions
        if key == "resolution":
            # Check if the output tuple matches the input list/tuple elements
            assert output[key] == tuple(
                input_data[key]
            ), f"'{key}' value mismatch or type mismatch (expected tuple)"
        elif key == "shape":
            # Check if the output tuple matches the input list/tuple elements
            assert output[key] == tuple(
                input_data[key]
            ), f"'{key}' value mismatch or type mismatch (expected tuple)"
        elif key == "orientation":
            # Ensure the output orientation is the standard 'asr'
            assert (
                output[key] == "asr"
            ), f"'{key}' value mismatch (expected 'asr')"
        else:
            # Direct comparison for other keys (name, citation, species, etc.)
            assert output[key] == input_data[key], f"'{key}' value mismatch"

    expected_keys = set(input_data.keys())
    output_keys = set(output.keys())
    assert (
        output_keys == expected_keys
    ), f"Output keys {output_keys} do not match expected keys {expected_keys}"


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
    """Test `generate_metadata_dict` error raising with modified metadata.

    Parameters
    ----------
    metadata : dict
        Dictionary of metadata fields to update in the template for testing.
    error : type or None
        Expected exception type to be raised, or None if no error is expected.
    metadata_input_template : dict
        A template dictionary for metadata input.
    """
    metadata_input_template.update(metadata)
    if error is not None:
        with pytest.raises(error):
            generate_metadata_dict(**metadata_input_template)
    else:
        generate_metadata_dict(**metadata_input_template)


@pytest.fixture
def structures():
    """Provide a list of dummy structure dictionaries for testing.

    Returns
    -------
    list of dict
        A list of dictionaries, where each dictionary represents a structure.
    """
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
    """Get the root ID from a list of structures.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries representing structures.

    Returns
    -------
    int
        The ID of the root structure.
    """
    for s in structures:
        if s["name"] == "root":
            return s["id"]


def test_create_metadata_files(
    structures,
    metadata_input_template,
    tmp_path,
):
    """Test `create_metadata_files`.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries representing structures.
    metadata_input_template : dict
        A template dictionary for metadata input.
    tmp_path : Path
        Temporary directory path provided by pytest fixture.
    """
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
    """Test `create_structures_csv`.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries representing structures.
    tmp_path : Path
        Temporary directory path provided by pytest fixture.
    """
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
    """Test `create_readme`.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries representing structures.
    metadata_input_template : dict
        A template dictionary for metadata input.
    tmp_path : Path
        Temporary directory path provided by pytest fixture.
    """
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
