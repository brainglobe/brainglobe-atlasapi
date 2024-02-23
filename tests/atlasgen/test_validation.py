import os

import numpy as np
import pytest

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    _assert_close,
    catch_missing_mesh_files,
    catch_missing_structures,
    validate_atlas_files,
    validate_mesh_matches_image_extents,
)
from brainglobe_atlasapi.config import get_brainglobe_dir


@pytest.fixture
def atlas():
    """A fixture providing a low-res Allen Mouse atlas for testing.
    Tests assume this atlas is valid"""
    return BrainGlobeAtlas("allen_mouse_100um")


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
def atlas_with_missing_structure():
    atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_structures = atlas.structures.copy()
    modified_structures.pop(688)

    modified_atlas = BrainGlobeAtlas("osten_mouse_100um")
    modified_atlas.structures = modified_structures
    return modified_atlas


def test_validate_mesh_matches_image_extents(atlas):
    assert validate_mesh_matches_image_extents(atlas)


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


def test_valid_atlas_files(atlas):
    assert validate_atlas_files(atlas)


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


def test_catch_missing_mesh_files(atlas):
    """
    Tests if catch_missing_mesh_files function raises an error,
    when there is at least one structure in the atlas that doesn't have
    a corresponding obj file.

    Expected behaviour:
    True for "allen_mouse_10um" (structure 545 doesn't have an obj file):
    fails the validation function,
    raises an error --> no output from this test function
    """

    with pytest.raises(
        AssertionError,
        match=r"Structures with IDs \[.*?\] are in the atlas, "
        "but don't have a corresponding mesh file.",
    ):
        catch_missing_mesh_files(atlas)


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
