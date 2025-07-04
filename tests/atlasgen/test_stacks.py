import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from brainglobe_atlasapi.atlas_generation.stacks import write_stack


@pytest.fixture
def image_unit():
    """Fixture providing a dummy image stack."""
    image_unit = np.random.randint(0, 65535, (10, 10, 10), dtype=np.uint16)
    yield image_unit


@pytest.fixture
def image_float():
    """Fixture providing a dummy image stack."""
    image_float = np.random.rand(10, 10, 10).astype(np.float32)
    yield image_float


@pytest.fixture
def tmp_path():
    """Fixture providing a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


def test_write_stack(image_unit, image_float, tmp_path):
    """Test writing image stacks to TIFF files."""
    # Test writing uint16 image stack
    output_file_unit = os.path.join(tmp_path, "test_uint16.tiff")
    write_stack(image_unit, output_file_unit)
    assert Path(output_file_unit).exists()

    # Test writing float32 image stack
    output_file_float = os.path.join(tmp_path, "test_float32.tiff")
    write_stack(image_float, output_file_float)
    assert Path(output_file_float).exists()

    # Check if the written files are same to the original data
    read_image_unit = tifffile.imread(output_file_unit)
    read_image_float = tifffile.imread(output_file_float)
    assert np.array_equal(read_image_unit, image_unit)
    assert np.array_equal(read_image_float, image_float)
