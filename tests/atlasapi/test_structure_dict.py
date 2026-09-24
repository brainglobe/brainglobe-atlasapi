"""Test the StructuresDict class for handling atlas structures."""

import DracoPy
import meshio as mio
import numpy as np
import pytest

from brainglobe_atlasapi import descriptors, structure_class
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_atlasapi.utils import load_structures_from_csv


def _draco_bytes():
    """Return valid Draco-encoded bytes for a minimal triangle mesh."""
    points = np.array(
        [[0, 0, 0], [1000, 0, 0], [0, 2000, 0], [0, 0, 3000]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.uint32)
    return DracoPy.encode(points, faces)


structures_list = [
    {
        "acronym": "root",
        "id": 997,
        "name": "root",
        "structure_id_path": [997],
        "rgb_triplet": [255, 255, 255],
        "mesh_filename": None,
    },
    {
        "acronym": "grey",
        "id": 8,
        "name": "Basic cell groups and regions",
        "structure_id_path": [997, 8],
        "rgb_triplet": [191, 218, 227],
        "mesh_filename": None,
    },
    {
        "acronym": "CH",
        "id": 567,
        "name": "Cerebrum",
        "structure_id_path": [997, 8, 567],
        "rgb_triplet": [176, 240, 255],
        "mesh_filename": None,
    },
]


def test_structure_indexing(atlas_path):
    """Test various indexing methods for StructuresDict.

    Verify that structures can be accessed by integer ID, float ID,
    and string ID, and that mesh loading errors are handled.
    """
    structures_dict = StructuresDict(structures_list)
    print(structures_dict)
    assert structures_dict[997] == structures_dict["root"]
    assert structures_dict[997.0] == structures_dict["root"]
    assert structures_dict["997"] == structures_dict["root"]


def test_mesh_loading(atlas_path):
    """Load meshes from a StructuresDict and verify type.

    Parameters
    ----------
    atlas_path : Path
        Path to the test atlas directory.
    """
    structures_list_real = load_structures_from_csv(
        atlas_path
        / "terminologies"
        / "example_mouse-terminology"
        / "3_0"
        / descriptors.V3_TERMINOLOGY_NAME
    )

    mesh_root_path = (
        atlas_path
        / "annotation-sets"
        / "example_mouse-annotation"
        / "3_0"
        / descriptors.V3_MESHES_DIRECTORY
    )

    # Add entry for file paths:
    for struct in structures_list_real:
        struct["mesh_filename"] = mesh_root_path / f"{struct['id']}"

    struct_dict = StructuresDict(structures_list_real)
    assert isinstance(struct_dict["997"]["mesh"], mio.Mesh)


def test_read_mesh_invalid_file_raises(tmp_path):
    """`read_mesh` raises `DracoPy.FileTypeException` on a non-Draco file."""
    bad_file = tmp_path / "997"
    bad_file.write_bytes(b"not a draco encoded mesh")

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]["mesh_filename"] = bad_file

    with pytest.raises(RuntimeError):
        _ = struct_dict["root"]["mesh"]


def _fake_s3_factory(exists, get_impl):
    """Build a fake `s3fs.S3FileSystem` class for monkeypatching."""

    class FakeS3FileSystem:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self, path):
            return exists

        def get(self, remote, local, callback=None):
            return get_impl(remote, local)

    return FakeS3FileSystem


def test_mesh_downloaded_when_missing_locally(tmp_path, monkeypatch):
    """A missing local mesh is downloaded from S3 and then read.

    Routes through `__getitem__` so the download-on-missing branch, the
    successful `s3fs.get`, and the subsequent mesh read are all exercised.
    """
    mesh_file = tmp_path / "997"

    def fake_get(remote, local):
        # Simulate the download by writing valid Draco bytes locally.
        local.write_bytes(_draco_bytes())

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(exists=True, get_impl=fake_get),
    )

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]["mesh_filename"] = mesh_file

    assert isinstance(struct_dict["root"]["mesh"], mio.Mesh)
    assert mesh_file.exists()


def test_download_mesh_missing_remotely_raises(tmp_path, monkeypatch):
    """`_download_mesh` raises `FileNotFoundError` if the remote is absent."""
    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(exists=False, get_impl=lambda remote, local: None),
    )

    struct_dict = StructuresDict(structures_list)
    struct = struct_dict["root"]

    with pytest.raises(FileNotFoundError):
        struct._download_mesh(tmp_path / "997")


def test_download_mesh_removes_corrupt_file_on_error(tmp_path, monkeypatch):
    """A failed download removes the partially written file and re-raises."""
    mesh_file = tmp_path / "997"
    mesh_file.write_bytes(b"partial download")

    def failing_get(remote, local):
        raise ConnectionError("network dropped mid-download")

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(exists=True, get_impl=failing_get),
    )

    struct_dict = StructuresDict(structures_list)
    struct = struct_dict["root"]

    with pytest.raises(ConnectionError):
        struct._download_mesh(mesh_file)

    assert not mesh_file.exists()
