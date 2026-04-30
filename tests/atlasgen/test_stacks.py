"""Unit tests for stack writing functions."""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile
import zarr

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.stacks import (
    BG_OME_ZARR_AXES,
    _save_as_ome_zarr,
    save_annotation,
    save_hemispheres,
    save_reference,
    save_template,
    write_multiscale_ome_zarr,
    write_stack,
)


@pytest.fixture
def image_uint16():
    """Fixture providing a dummy uint16 image stack."""
    image_uint16 = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)
    yield image_uint16


@pytest.fixture
def image_float32():
    """Fixture providing a dummy float32 image stack."""
    image_float32 = np.random.rand(10, 10, 10).astype(np.float32)
    yield image_float32


@pytest.fixture
def image_uint32():
    """Fixture providing a dummy uint32 image stack."""
    image_uint32 = np.random.randint(0, 100, (10, 10, 10), dtype=np.uint32)
    yield image_uint32


@pytest.fixture
def image_uint8():
    """Fixture providing a dummy uint8 image stack."""
    image_uint8 = np.random.randint(0, 255, (10, 10, 10), dtype=np.uint8)
    yield image_uint8


@pytest.fixture
def transformations():
    """Fixture providing dummy transformations for OME-Zarr metadata."""
    return [[{"type": "scale", "scale": [0.025, 0.025, 0.025]}]]


def test_write_stack_uint16(image_uint16, tmp_path):
    """Test writing image stacks to TIFF files."""
    # Test writing uint16 image stack
    output_file_uint = os.path.join(tmp_path, "test_uint16.tiff")
    write_stack(image_uint16, output_file_uint)
    assert Path(output_file_uint).exists()

    # Check if the written files are same to the original data
    read_image_unit = tifffile.imread(output_file_uint)
    assert np.array_equal(read_image_unit, image_uint16)


def test_write_stack_float32(image_float32, tmp_path):
    """Test writing image stacks to TIFF files."""
    # Test writing float32 image stack
    output_file_float = os.path.join(tmp_path, "test_float32.tiff")
    write_stack(image_float32, output_file_float)
    assert Path(output_file_float).exists()

    # Check if the written files are same to the original data
    read_image_float = tifffile.imread(output_file_float)
    assert np.array_equal(read_image_float, image_float32)


# --- write_multiscale_ome_zarr ---


def test_write_multiscale_ome_zarr_creates_zarr(
    image_uint16, transformations, tmp_path
):
    """Test writing a multiscale image pyramid to an OME-Zarr file."""
    output_path = tmp_path / "test.ome.zarr"
    write_multiscale_ome_zarr(
        images=[image_uint16],
        output_path=output_path,
        transformations=transformations,
    )
    assert output_path.exists()
    root = zarr.open_group(str(output_path), mode="r")
    arr_keys = list(root.keys())
    assert len(arr_keys) == 1
    assert arr_keys[0].endswith("0")


def test_write_multiscale_ome_zarr_default_axes(
    image_uint16, transformations, tmp_path
):
    """Test default axes are correctly written to OME-Zarr metadata."""
    output_path = tmp_path / "test.ome.zarr"
    write_multiscale_ome_zarr(
        images=[image_uint16],
        output_path=output_path,
        transformations=transformations,
    )

    out_zarr = zarr.open(str(output_path), mode="r")
    ome_metadata = out_zarr.attrs["ome"]
    axes = ome_metadata["multiscales"][0]["axes"]
    assert axes == BG_OME_ZARR_AXES


def test_write_multiscale_ome_zarr_custom_axes(
    image_uint16, transformations, tmp_path
):
    """Test custom axes are correctly written to OME-Zarr metadata."""
    custom_axes = [
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]
    output_path = tmp_path / "test.ome.zarr"
    write_multiscale_ome_zarr(
        images=[image_uint16],
        output_path=output_path,
        transformations=transformations,
        axes=custom_axes,
    )
    out_zarr = zarr.open(str(output_path), mode="r")
    ome_metadata = out_zarr.attrs["ome"]
    axes = ome_metadata["multiscales"][0]["axes"]
    assert axes == custom_axes


# --- save_reference ---


def test_save_reference_creates_file(image_uint16, tmp_path):
    """Test save_reference creates the expected TIFF file."""
    save_reference(image_uint16, tmp_path)
    assert (tmp_path / descriptors.REFERENCE_FILENAME).exists()


def test_save_reference_correct_dtype(image_uint16, tmp_path):
    """Test save_reference saves the image with the correct data type."""
    save_reference(image_uint16, tmp_path)
    result = tifffile.imread(tmp_path / descriptors.REFERENCE_FILENAME)
    assert result.dtype == descriptors.REFERENCE_DTYPE


def test_save_reference_converts_dtype(image_float32, tmp_path):
    """Test save_reference converts the image to the correct data type."""
    save_reference(image_float32, tmp_path)
    result = tifffile.imread(tmp_path / descriptors.REFERENCE_FILENAME)
    assert result.dtype == descriptors.REFERENCE_DTYPE


# --- _save_as_ome_zarr ---


def test_save_as_ome_zarr_converts_dtype(
    image_float32, transformations, tmp_path
):
    """Test _save_as_ome_zarr converts to the specified data type."""
    output_path = tmp_path / "test.ome.zarr"
    _save_as_ome_zarr([image_float32], np.uint16, output_path, transformations)
    root = zarr.open_group(str(output_path), mode="r")
    for arr_key in root:
        assert root[arr_key].dtype == np.uint16


def test_save_as_ome_zarr_raises_on_multiple_resolutions(
    image_uint16, tmp_path
):
    """Test _save_as_ome_zarr raises an error with multiple resolutions."""
    bad_transformations = [
        [{"type": "scale", "scale": [0.025, 0.025, 0.025]}],
        [{"type": "scale", "scale": [0.05, 0.05, 0.05]}],
    ]
    with pytest.raises(AssertionError):
        _save_as_ome_zarr(
            [image_uint16],
            np.uint16,
            tmp_path / "test.ome.zarr",
            bad_transformations,
        )


# --- save_template ---


def test_save_template_creates_zarr(image_uint16, transformations, tmp_path):
    """Test save_template creates the expected OME-Zarr file."""
    save_template([image_uint16], tmp_path, transformations)
    assert (tmp_path / descriptors.V2_TEMPLATE_NAME).exists()


def test_save_template_uses_reference_dtype(
    image_uint8, transformations, tmp_path
):
    """Test save_template saves the image with the correct data type."""
    save_template([image_uint8], tmp_path, transformations)
    root = zarr.open_group(
        str(tmp_path / descriptors.V2_TEMPLATE_NAME), mode="r"
    )
    for arr_key in root:
        assert root[arr_key].dtype == descriptors.REFERENCE_DTYPE


# --- save_annotation ---


def test_save_annotation_creates_zarr(image_uint32, transformations, tmp_path):
    """Test save_annotation creates the expected OME-Zarr file."""
    save_annotation([image_uint32], tmp_path, transformations)
    assert (tmp_path / descriptors.V2_ANNOTATION_NAME).exists()


def test_save_annotation_uses_annotation_dtype(
    image_uint8, transformations, tmp_path
):
    """Test save_annotation saves the image with the correct data type."""
    save_annotation([image_uint8], tmp_path, transformations)
    root = zarr.open_group(
        str(tmp_path / descriptors.V2_ANNOTATION_NAME), mode="r"
    )
    for arr_key in root:
        assert root[arr_key].dtype == descriptors.ANNOTATION_DTYPE


# --- save_hemispheres ---


def test_save_hemispheres_creates_zarr(image_uint8, transformations, tmp_path):
    """Test save_hemispheres creates the expected OME-Zarr file."""
    save_hemispheres([image_uint8], tmp_path, transformations)
    assert (tmp_path / descriptors.V2_HEMISPHERES_NAME).exists()


def test_save_hemispheres_uses_hemispheres_dtype(
    image_uint16, transformations, tmp_path
):
    """Test save_hemispheres saves the image with the correct data type."""
    save_hemispheres([image_uint16], tmp_path, transformations)
    root = zarr.open_group(
        str(tmp_path / descriptors.V2_HEMISPHERES_NAME), mode="r"
    )
    for arr_key in root:
        assert root[arr_key].dtype == descriptors.HEMISPHERES_DTYPE
