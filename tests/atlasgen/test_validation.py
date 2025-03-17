import os
import re

import numpy as np
import pytest
import tifffile

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    _assert_close,
    catch_missing_mesh_files,
    catch_missing_structures,
    get_all_validation_functions,
    validate_additional_references,
    validate_annotation_symmetry,
    validate_atlas_files,
    validate_atlas_name,
    validate_image_dimensions,
    validate_mesh_matches_image_extents,
    validate_metadata,
    validate_reference_image_pixels,
)
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.core import AdditionalRefDict


@pytest.fixture
def atlas():
    """A fixture providing a small atlas for testing.
    Tests assume this atlas is valid"""
    return BrainGlobeAtlas("kim_dev_mouse_e11-5_mri-adc_31.5um")


@pytest.fixture
def atlas_with_bad_reference_file():
    """A fixture providing an invalid version of Allen Mouse atlas for testing.
    The atlas will have a misnamed template file that won't be found by the API
    This fixture also does the clean-up after the test has run
    """
    good_name = get_brainglobe_dir() / "allen_mouse_100um_v1.2/reference.tiff"
    bad_name = (
        get_brainglobe_dir() / "allen_mouse_100um_v1.2/reference_bad.tiff"
    )
    os.rename(good_name, bad_name)
    yield BrainGlobeAtlas("allen_mouse_100um")
    os.rename(bad_name, good_name)


@pytest.fixture
def atlas_with_bad_reference_tiff_content():
    """A fixture providing an invalid version of Allen Mouse atlas for testing.
    The atlas will have a misnamed template file that won't be found by the API
    This fixture cleans up the invalid atlas after the test has run.
    """
    BrainGlobeAtlas("allen_mouse_100um")  # ensure atlas is locally downloaded
    actual_name = (
        get_brainglobe_dir() / "allen_mouse_100um_v1.2/reference.tiff"
    )
    backup_name = (
        get_brainglobe_dir() / "allen_mouse_100um_v1.2/reference_backup.tiff"
    )
    os.rename(actual_name, backup_name)
    too_small_reference = np.ones((3, 3, 3), dtype=np.uint16)
    tifffile.imwrite(actual_name, too_small_reference)
    yield BrainGlobeAtlas("allen_mouse_100um")
    os.remove(actual_name)
    os.rename(backup_name, actual_name)


@pytest.fixture
def atlas_with_missing_structure():
    atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_structures = atlas.structures.copy()
    modified_structures.pop(688)

    modified_atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_atlas.structures = modified_structures
    return modified_atlas


@pytest.fixture
def atlas_with_valid_additional_reference():
    """A fixture providing a testing-only version of the Allen Mouse atlas.
    The instance of the atlas returned has an additional reference
    consisting of an array of 1, of the correct size.
    This fixture cleans up the invalid atlas after the test has run.
    """
    allen_100 = BrainGlobeAtlas(
        "allen_mouse_100um"
    )  # ensure atlas is locally downloaded
    additional_reference_name = (
        get_brainglobe_dir()
        / "allen_mouse_100um_v1.2/mock_additional_reference.tiff"
    )
    additional_reference = np.ones(allen_100.reference.shape, dtype=np.uint16)
    allen_100.additional_references = AdditionalRefDict(
        ["mock_additional_reference"],
        data_path=get_brainglobe_dir() / "allen_mouse_100um_v1.2",
    )
    tifffile.imwrite(additional_reference_name, additional_reference)
    yield allen_100
    os.remove(additional_reference_name)


@pytest.fixture
def atlas_with_reference_matching_additional_reference():
    """A fixture providing an invalid version of Allen Mouse atlas for testing.
    It provides the atlas, with an additional reference containing
    the same data as the main reference image.
    This fixture cleans up the invalid atlas after the test has run.
    """
    allen_100 = BrainGlobeAtlas(
        "allen_mouse_100um"
    )  # ensure atlas is locally downloaded
    additional_reference_name = (
        get_brainglobe_dir()
        / "allen_mouse_100um_v1.2/mock_additional_reference.tiff"
    )
    additional_reference = allen_100.reference
    allen_100.additional_references = AdditionalRefDict(
        ["mock_additional_reference"],
        data_path=get_brainglobe_dir() / "allen_mouse_100um_v1.2",
    )
    tifffile.imwrite(additional_reference_name, additional_reference)
    yield allen_100
    os.remove(additional_reference_name)


def test_valid_atlas_passes_all_validations(atlas):
    """
    Check all our validation functions return True
    for a valid atlas
    """
    validation_functions = get_all_validation_functions()
    for validation_function in validation_functions:
        assert validation_function(
            atlas
        ), f"Function {validation_function.__name__} fails on valid atlas."


def test_validate_mesh_matches_image_extents_negative(mocker, atlas):
    flipped_annotation_image = np.transpose(atlas.annotation)
    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.annotation",
        new_callable=mocker.PropertyMock,
        return_value=flipped_annotation_image,
    )
    with pytest.raises(
        AssertionError, match="differ by more than 10 times pixel size"
    ):
        validate_mesh_matches_image_extents(atlas)


def test_invalid_atlas_path(atlas_with_bad_reference_file):
    with pytest.raises(AssertionError, match="Expected file not found"):
        validate_atlas_files(atlas_with_bad_reference_file)


def test_assert_close():
    assert _assert_close(99.5, 8, 10)


def test_assert_close_negative():
    with pytest.raises(
        AssertionError, match="differ by more than 10 times pixel size"
    ):
        _assert_close(99.5, 30, 2)


def test_catch_missing_mesh_files():
    """
    Tests if catch_missing_mesh_files function raises an error,
    when there is at least one structure in the atlas that doesn't have
    a corresponding obj file.

    Expected behaviour:
    True for "allen_mouse_100um" (structure 545 doesn't have an obj file):
    fails the validation function,
    raises an error --> no output from this test function
    """
    atlas_with_missing_mesh_file = BrainGlobeAtlas("allen_mouse_100um")
    with pytest.raises(
        AssertionError,
        match=r"Structures with IDs \[.*?\] are in the atlas, "
        "but don't have a corresponding mesh file.",
    ):
        catch_missing_mesh_files(atlas_with_missing_mesh_file)


def test_catch_missing_structures(atlas_with_missing_structure):
    """
    Tests if catch_missing_structures function raises an error,
    when there is at least one orphan obj file (doesn't have a
    corresponding structure in the atlas)

    Expected behaviour:
    Currently no atlas fails the validation function this way so the
    [] is always empty --> this test function should always raise an error
    """

    with pytest.raises(
        AssertionError,
        match=r"Structures with IDs \[.*?\] have a mesh file, "
        "but are not accessible through the atlas.",
    ):
        catch_missing_structures(atlas_with_missing_structure)


def test_atlas_image_dimensions_match_negative(
    atlas_with_bad_reference_tiff_content,
):
    """Checks that an atlas with different annotation and reference
    dimensions is flagged by the validation."""
    with pytest.raises(
        AssertionError,
        match=r"Annotation and reference image have different dimensions.*",
    ):
        validate_image_dimensions(atlas_with_bad_reference_tiff_content)


def test_atlas_additional_reference_same(
    atlas_with_reference_matching_additional_reference,
):
    """Checks that an atlas with a rduplicate additional reference
    fails the validation for this case."""
    with pytest.raises(
        AssertionError,
        match=r"Additional reference is not different to main reference.",
    ):
        validate_additional_references(
            atlas_with_reference_matching_additional_reference
        )


def test_badly_scaled_reference_fails(mocker, atlas):
    """Checks that an atlas with only ones as reference image fails"""
    invalid_reference = np.ones_like(atlas.reference)
    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.reference",
        new_callable=mocker.PropertyMock,
        return_value=invalid_reference,
    )
    with pytest.raises(
        AssertionError,
        match=r"Reference image is likely wrongly rescaled to.",
    ):
        validate_reference_image_pixels(atlas)


def test_asymmetrical_annotation_fails(mocker, atlas):
    """Checks that an atlas with LR annotation mismatch fails"""
    asymmetrical_lr_labels = atlas.annotation.copy()
    midsagittal_index = asymmetrical_lr_labels.shape[2] // 2
    asymmetrical_lr_labels[:, :, midsagittal_index:] += 1

    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.annotation",
        new_callable=mocker.PropertyMock,
        return_value=asymmetrical_lr_labels,
    )
    with pytest.raises(
        AssertionError,
        match=r"Annotation labels are asymmetric.",
    ):
        validate_annotation_symmetry(atlas)


@pytest.mark.parametrize(
    "atlas_name, should_pass, error_message",
    [
        pytest.param(
            "CONTAINS_CAPITALS_1um",
            False,
            "cannot contain capitals.",
            id="upper case",
        ),
        pytest.param(
            "contains_inv@lid_character_1um",
            False,
            "contains invalid characters.",
            id="invalid charachter (@)",
        ),
        pytest.param(
            "10um_atlas_name_does_not_end_with_resolution",
            False,
            "should end with a valid resolution (e.g., 5um, 1.5mm).",
            id="doesn't end with resolution",
        ),
        pytest.param(
            "atlas_name_with_invalid_resolution_100m",
            False,
            "should end with a valid resolution (e.g., 5um, 1.5mm).",
            id="invalid resolution (100m)",
        ),
        pytest.param("valid_name_100um", True, None, id="valid_name_100um"),
        pytest.param("valid-name_1mm", True, None, id="valid-name_1mm"),
    ],
)
def test_validate_atlas_name(atlas_name, should_pass, error_message):
    """Checks various atlas name validation cases"""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    if should_pass:
        assert validate_atlas_name(atlas)
    else:
        error_message = f"Atlas name {atlas_name} " + error_message
        with pytest.raises(AssertionError, match=re.escape(error_message)):
            validate_atlas_name(atlas)


@pytest.mark.parametrize(
    ["metadata", "expected_output", "error_message"],
    [
        pytest.param(
            {
                "key": "citation",
                "value": "BrainGlobe et. al., 2025., https://doi.org",
            },
            True,
            None,
            id="citation (multiple commas)",
        ),
        pytest.param(
            {
                "key": "resolution",
                "value": [1, 1],
            },
            True,
            None,
            id="2D resolution [1, 1]",
        ),
        pytest.param(
            {
                "key": "resolution",
                "value": (1, 1),
            },
            "AssertionError",
            "resolution should be of type list, but got tuple.",
            id="wrong resolution type (tuple)",
        ),
    ],
)
def test_validate_metadata(atlas, metadata, expected_output, error_message):
    """Checks whether atlas metadata is validated correctly."""
    atlas.metadata[metadata["key"]] = metadata["value"]
    if expected_output == "AssertionError":
        with pytest.raises(AssertionError, match=error_message):
            validate_metadata(atlas)
    else:
        assert validate_metadata(atlas) == expected_output
