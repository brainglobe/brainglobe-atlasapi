"""Unit tests for stack writing functions."""

import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from brainglobe_atlasapi.atlas_generation.stacks import write_stack


@pytest.fixture
def image_uint16():
    """Fixture providing a dummy image stack."""
    image_uint16 = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)
    yield image_uint16


@pytest.fixture
def image_float32():
    """Fixture providing a dummy image stack."""
    image_float32 = np.random.rand(10, 10, 10).astype(np.float32)
    yield image_float32


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
