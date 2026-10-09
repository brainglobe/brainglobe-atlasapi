"""Test atlas validation functions."""

import copy
import re

import numpy as np
import pytest

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
    validate_atlas_name_listed,
    validate_image_dimensions,
    validate_mesh_matches_image_extents,
    validate_metadata,
    validate_template_image_pixels,
    validate_unique_acronyms,
)
from brainglobe_atlasapi.core import AdditionalRefDict


@pytest.fixture
def atlas():
    """Provide a small atlas for testing.
    Tests assume this atlas is valid.

    Returns
    -------
    BrainGlobeAtlas
        A BrainGlobeAtlas instance for testing.
    """
    return BrainGlobeAtlas("kim_dev_mouse_e11-5_mri-adc_31.5um")


@pytest.fixture
def atlas_with_missing_template(atlas):
    """Provide an atlas whose manifest requests a template absent on disk.

    The template location is changed in the top-level `template` entry,
    `annotation_set.template` and `coordinate_space.template`, so the
    metadata stays internally consistent while pointing at a directory that
    does not exist. Only the in-memory metadata is changed; nothing on disk
    is modified.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A valid BrainGlobeAtlas instance.

    Returns
    -------
    BrainGlobeAtlas
        The atlas with its template location pointing at a missing directory.
    """
    atlas.metadata = copy.deepcopy(atlas.metadata)
    template_name = atlas.metadata["template"]["name"]
    missing_location = atlas.metadata["template"]["location"].replace(
        template_name, f"{template_name}-missing"
    )
    for template in (
        atlas.metadata["template"],
        atlas.metadata["annotation_set"]["template"],
        atlas.metadata["coordinate_space"]["template"],
    ):
        template["location"] = missing_location
    return atlas


@pytest.fixture
def atlas_with_missing_structure():
    """Provide an atlas with a manually removed structure for testing.

    Returns
    -------
    BrainGlobeAtlas
        A modified BrainGlobeAtlas instance with a missing structure.
    """
    atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_structures = atlas.structures.copy()
    modified_structures.pop(688)

    modified_atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_atlas.structures = modified_structures
    return modified_atlas


@pytest.fixture
def atlas_with_reference_matching_additional_reference(atlas):
    """Provide an atlas whose additional reference duplicates its template.

    The additional reference entry requests the main template's location,
    so it loads the same data as the main template. Only the in-memory
    atlas is changed; nothing on disk is modified.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A valid BrainGlobeAtlas instance.

    Returns
    -------
    BrainGlobeAtlas
        An invalid BrainGlobeAtlas instance with a duplicate additional
        reference.
    """
    main_template = atlas.metadata["annotation_set"]["template"]
    atlas.additional_references = AdditionalRefDict(
        references_list=[
            {**main_template, "name": "duplicate-of-main-template"}
        ],
        data_path=atlas.root_dir,
        resolution=atlas.resolution,
    )
    return atlas


@pytest.mark.xfail(
    reason="This test is currently failing as the validation functions "
    "have not been updated to work with the new atlas structure."
)
def test_valid_atlas_passes_all_validations(atlas):
    """Check all validation functions return True for a valid atlas.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A valid BrainGlobeAtlas instance.
    """
    validation_functions = get_all_validation_functions()
    for validation_function in validation_functions:
        assert validation_function(
            atlas
        ), f"Function {validation_function.__name__} fails on valid atlas."


def test_validate_mesh_matches_image_extents_negative(mocker, atlas):
    """Verify `validate_mesh_matches_image_extents` fails for
    mismatched extents.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    """
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


def test_invalid_atlas_path(atlas_with_missing_template):
    """Verify `validate_atlas_files` raises an error for a missing
    template file.

    Parameters
    ----------
    atlas_with_missing_template : BrainGlobeAtlas
        An atlas instance whose template location does not exist on disk.
    """
    with pytest.raises(AssertionError, match="Expected file not found"):
        validate_atlas_files(atlas_with_missing_template)


def test_validate_atlas_name_not_listed():
    """Verify `validate_atlas_name_listed` fails for names not in
    atlas_name.py.
    """

    class DummyAtlas:
        atlas_name = "unlisted_atlas_1um"

    with pytest.raises(AssertionError, match="not listed in atlas_name.py"):
        validate_atlas_name_listed(DummyAtlas())


def test_validate_atlas_name_listed_passes():
    """Verify `validate_atlas_name_listed` passes for listed atlas names."""

    class DummyAtlas:
        atlas_name = "example_mouse_100um"

    assert validate_atlas_name_listed(DummyAtlas())


def test_assert_close():
    """Verify `_assert_close` returns True for close values."""
    assert _assert_close(99.5, 8, 10)


def test_assert_close_negative():
    """Verify `_assert_close` raises an error for values not close enough."""
    with pytest.raises(
        AssertionError, match="differ by more than 10 times pixel size"
    ):
        _assert_close(99.5, 30, 2)


def test_catch_missing_mesh_files():
    """Test `catch_missing_mesh_files` raises an error for missing mesh files.

    Expected behaviour:
    For "allen_mouse_100um", structure 545 does not have an obj file,
    so this validation function should fail and raise an error.
    """
    atlas_with_missing_mesh_file = BrainGlobeAtlas("allen_mouse_100um")
    with pytest.raises(
        AssertionError,
        match=r"Structures with IDs \[.*?\] are in the atlas, "
        "but don't have a corresponding mesh file.",
    ):
        catch_missing_mesh_files(atlas_with_missing_mesh_file)


@pytest.mark.xfail(
    reason="This test is currently failing as the validation functions "
    "have not been updated to work with the new atlas structure."
)
def test_catch_missing_structures(atlas_with_missing_structure):
    """Test `catch_missing_structures` raises an error for orphan mesh files.

    Verify that an error is raised when there is at least one mesh file that
    does not have a corresponding structure in the atlas.

    Parameters
    ----------
    atlas_with_missing_structure : BrainGlobeAtlas
        An atlas instance with a missing structure.
    """
    with pytest.raises(
        AssertionError,
        match=r"Structures with IDs \[.*?\] have a mesh file, "
        "but are not accessible through the atlas.",
    ):
        catch_missing_structures(atlas_with_missing_structure)


def test_atlas_image_dimensions_match_negative(mocker, atlas):
    """Check that an atlas with different annotation and template
    dimensions is flagged by the validation.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    """
    too_small_template = np.ones((3, 3, 3), dtype=np.uint16)
    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.template",
        new_callable=mocker.PropertyMock,
        return_value=too_small_template,
    )
    with pytest.raises(
        AssertionError,
        match=r"Annotation and template image have different dimensions.*",
    ):
        validate_image_dimensions(atlas)


def test_atlas_additional_reference_same(
    atlas_with_reference_matching_additional_reference,
):
    """Check that an atlas with a duplicate additional reference
    fails the validation.

    Parameters
    ----------
    atlas_with_reference_matching_additional_reference : BrainGlobeAtlas
        An atlas instance where the additional reference matches the
        main reference.
    """
    with pytest.raises(
        AssertionError,
        match=r"Additional reference is not different to main reference.",
    ):
        validate_additional_references(
            atlas_with_reference_matching_additional_reference
        )


def test_badly_scaled_reference_fails(mocker, atlas):
    """Check that an atlas with only ones as reference image fails.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    """
    invalid_reference = np.ones_like(atlas.template)
    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.template",
        new_callable=mocker.PropertyMock,
        return_value=invalid_reference,
    )
    with pytest.raises(
        AssertionError,
        match=r"Template image is likely wrongly rescaled to.",
    ):
        validate_template_image_pixels(atlas)


def test_asymmetrical_annotation_fails(mocker, atlas):
    """Check that an atlas with LR annotation mismatch fails validation.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    """
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


@pytest.mark.parametrize("width", ["odd", "even"])
def test_odd_and_even_pass_atlas_symmetry(mocker, atlas, width):
    """Ensure odd- and even-width atlases pass symmetry validation."""
    # create a mirror image atlas of ascending and descending values
    hemisphere = np.zeros((50, 50, 50))
    hemisphere[:, :, :] = np.arange(50)
    annotation = np.concatenate(
        (hemisphere, hemisphere[:, :, ::-1]),
        axis=2,
    )

    if width == "odd":
        annotation = np.insert(annotation, 50, 0, axis=2)

    mocker.patch(
        "brainglobe_atlasapi.BrainGlobeAtlas.annotation",
        new_callable=mocker.PropertyMock,
        return_value=annotation,
    )

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
    """Check various atlas name validation cases.

    Parameters
    ----------
    atlas_name : str
        The name of the atlas to test.
    should_pass : bool
        True if the validation should pass, False otherwise.
    error_message : str, optional
        The expected error message if validation fails, by default None.
    """
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
    """Check whether atlas metadata is validated correctly.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    metadata : dict
        A dictionary containing the metadata key and value to test.
    expected_output : bool or str
        True if the validation should pass, "AssertionError" if it should
        raise an error.
    error_message : str, optional
        The expected error message if validation fails, by default None.
    """
    atlas.metadata[metadata["key"]] = metadata["value"]
    if expected_output == "AssertionError":
        with pytest.raises(AssertionError, match=error_message):
            validate_metadata(atlas)
    else:
        assert validate_metadata(atlas) == expected_output


def test_validate_unique_acronyms_fail(mocker, atlas):
    """Check that an atlas with duplicate acronyms fails validation.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    atlas : BrainGlobeAtlas
        A BrainGlobeAtlas instance.
    """
    # Create structures with duplicate acronyms
    structures_with_duplicates = {
        1: {"acronym": "root", "name": "Root"},
        2: {"acronym": "brain", "name": "Brain"},
        3: {"acronym": "brain", "name": "Brain Duplicate"},  # Duplicate!
        4: {"acronym": "cortex", "name": "Cortex"},
    }
    mocker.patch.object(atlas, "structures", structures_with_duplicates)

    with pytest.raises(AssertionError) as exc_info:
        validate_unique_acronyms(atlas)

    # Verify error contains the duplicate acronym and its name
    error_message = str(exc_info.value)
    assert "brain" in error_message
    assert "Brain Duplicate" in error_message
