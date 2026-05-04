"""
Tests for the atlas_packaging_data module.

Verifies the correctness of resolution standardisation, stack loading,
hemisphere auto-generation, and stack reorientation helpers.
"""

import brainglobe_space as bgs
import numpy as np
import pytest
import tifffile

from brainglobe_atlasapi import descriptors

# Imported for use in Tasks 2-5 tests (not yet used in Task 1)
from brainglobe_atlasapi.atlas_generation.atlas_packaging_data import (
    AnnotationInfo,  # noqa: F401
    AtlasPackagingData,  # noqa: F401
    ComponentInfo,  # noqa: F401
    CoordinateSpaceInfo,  # noqa: F401
    TemplateInfo,  # noqa: F401
    TerminologyInfo,  # noqa: F401
    _auto_generate_hemispheres,
    _load_stack,
    _reorient_stacks,
    _standardize_resolution,
    check_requested_component,  # noqa: F401
)

# --- _standardize_resolution ---


@pytest.mark.parametrize(
    "resolution, expected",
    [
        pytest.param(
            (10, 20, 30),
            [(10, 20, 30)],
            id="single tuple",
        ),
        pytest.param(
            [(10, 20, 30), (20, 40, 60)],
            [(10, 20, 30), (20, 40, 60)],
            id="list of tuples",
        ),
    ],
)
def test_standardize_resolution(resolution, expected):
    """Test `_standardize_resolution` with various inputs.

    Parameters
    ----------
    resolution : tuple or list of tuples
        The resolution input to standardize.
    expected : list of tuples
        The expected output after standardization.
    """
    assert _standardize_resolution(resolution) == expected


def test_standardize_resolution_invalid():
    """Test `_standardize_resolution` raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Resolution must be either"):
        _standardize_resolution("invalid")


# --- _load_stack ---


def test_load_stack_ndarray():
    """Test `_load_stack` wraps a numpy array in a list."""
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    result = _load_stack(arr)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] is arr


def test_load_stack_list_passthrough():
    """Test `_load_stack` passes through a list unchanged."""
    arr1 = np.zeros((4, 4, 4), dtype=np.uint16)
    arr2 = np.ones((4, 4, 4), dtype=np.uint16)
    result = _load_stack([arr1, arr2])
    assert result == [arr1, arr2]


def test_load_stack_path_returns_list(tmp_path):
    """Test `_load_stack` returns a single-element list when given a Path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(tiff_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_stack_path_reads_data(tmp_path):
    """Test `_load_stack` reads correct data from a TIFF file at a Path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(tiff_path)
    assert np.array_equal(result[0], arr)


def test_load_stack_str_returns_list(tmp_path):
    """Test `_load_stack` returns a list when given a string path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(str(tiff_path))
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_stack_str_reads_data(tmp_path):
    """Test `_load_stack` reads correct data from a string TIFF file path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(str(tiff_path))
    assert np.array_equal(result[0], arr)


# --- _auto_generate_hemispheres ---


def test_auto_generate_hemispheres_shape():
    """Test `_auto_generate_hemispheres` with matching shapes."""
    shapes = [(4, 4, 4)]
    annotation = [np.zeros((4, 4, 4), dtype=np.uint32)]
    result = _auto_generate_hemispheres(shapes, annotation)
    assert len(result) == 1
    assert result[0].shape == (4, 4, 4)


def test_auto_generate_hemispheres_values():
    """Test `_auto_generate_hemispheres` splits the volume at the midpoint.

    The left half (dim 2, indices < midpoint) should be 2.
    The right half (dim 2, indices >= midpoint) should be 1.
    """
    shapes = [(4, 4, 4)]
    annotation = [np.zeros((4, 4, 4), dtype=np.uint32)]
    result = _auto_generate_hemispheres(shapes, annotation)
    assert np.all(result[0][:, :, :2] == 2)
    assert np.all(result[0][:, :, 2:] == 1)


def test_auto_generate_hemispheres_multiple_scales():
    """Test `_auto_generate_hemispheres` handles multiple annotation scales."""
    shapes = [(4, 4, 4), (2, 2, 2)]
    annotations = [
        np.zeros((4, 4, 4), dtype=np.uint32),
        np.zeros((2, 2, 2), dtype=np.uint32),
    ]
    result = _auto_generate_hemispheres(shapes, annotations)
    assert len(result) == 2
    assert result[0].shape == (4, 4, 4)
    assert result[1].shape == (2, 2, 2)


# --- _reorient_stacks ---


def test_reorient_stacks_identity():
    """Test `_reorient_stacks` leaves stacks unchanged when already in asr."""
    arr = np.arange(64, dtype=np.uint16).reshape((4, 4, 4))
    space = bgs.AnatomicalSpace(descriptors.ATLAS_ORIENTATION, shape=arr.shape)
    result = _reorient_stacks([arr], space)
    assert len(result) == 1
    assert np.array_equal(result[0], arr)


def test_reorient_stacks_reorders_axes():
    """Test `_reorient_stacks` correctly reorders axes for non-asr input."""
    arr = np.arange(64, dtype=np.uint16).reshape((4, 4, 4))
    # "sar" orientation means the mapping to "asr" should permute axes
    space = bgs.AnatomicalSpace("sar", shape=arr.shape)
    result = _reorient_stacks([arr], space)
    assert len(result) == 1
    expected = space.map_stack_to(
        descriptors.ATLAS_ORIENTATION, arr, copy=True
    )
    assert np.array_equal(result[0], expected)
