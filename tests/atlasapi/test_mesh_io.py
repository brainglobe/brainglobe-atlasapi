"""Hermetic round-trip tests for the Draco mesh I/O functions."""

import DracoPy
import meshio
import numpy as np
import pytest

from brainglobe_atlasapi.mesh_io import read_mesh, write_mesh, write_mesh_info


@pytest.fixture
def stored_mesh():
    """Build a mesh in the stored (nm, XYZ) convention `write_mesh` expects.

    Returns
    -------
    meshio.Mesh
        Mesh whose points are in nanometres and XYZ axis order, mirroring
        what `wrapup._save_meshes` produces before calling `write_mesh`.
    """
    rng = np.random.default_rng(0)
    points_nm_xyz = rng.uniform(0.0, 5000.0, size=(20, 3)).astype(np.float32)
    faces = np.array(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11], [0, 4, 8], [1, 5, 9]],
        dtype=np.uint32,
    )
    return meshio.Mesh(points=points_nm_xyz, cells=[("triangle", faces)])


def test_write_mesh_creates_expected_files(tmp_path, stored_mesh):
    """`write_mesh` writes the fragment and its index for the segment id."""
    write_mesh(stored_mesh, tmp_path, segment_id=42)

    assert (tmp_path / "42").exists()
    assert (tmp_path / "42.index").exists()


def test_write_read_roundtrip(tmp_path, stored_mesh):
    """`read_mesh` recovers `write_mesh` geometry within quantization error.

    `write_mesh` stores points as-is (nm, XYZ); `read_mesh` converts back to
    um and ZYX. The recovered mesh must therefore equal the stored points
    scaled by 1/1000 with axes swapped, and the faces column-swapped, both
    within Draco's 16-bit quantization tolerance.
    """
    write_mesh(stored_mesh, tmp_path, segment_id=7)

    result = read_mesh(tmp_path / "7")

    expected_points = stored_mesh.points[:, [2, 1, 0]] / 1000.0
    # qrange ~5000 nm over 16 bits -> half-quantum ~4e-5 um;
    # atol=1e-3 um leaves ~25x margin.
    np.testing.assert_allclose(result.points, expected_points, atol=1e-3)

    expected_faces = stored_mesh.cells[0].data[:, [2, 1, 0]]
    np.testing.assert_array_equal(result.cells[0].data, expected_faces)


def test_read_mesh_invalid_file_raises(tmp_path):
    """`read_mesh` raises `DracoPy.FileTypeException` on a non-Draco file."""
    bad_file = tmp_path / "998"
    bad_file.write_bytes(b"not a draco encoded mesh")

    with pytest.raises(DracoPy.FileTypeException):
        read_mesh(bad_file)


def test_write_mesh_info(tmp_path):
    """`write_mesh_info` writes a valid multilod-draco `info` metadata file."""
    info = write_mesh_info(tmp_path)

    assert (tmp_path / "info").exists()
    assert info["@type"] == "neuroglancer_multilod_draco"
    assert info["vertex_quantization_bits"] == 16
