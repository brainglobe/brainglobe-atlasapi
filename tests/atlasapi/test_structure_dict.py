"""Test the StructuresDict class for handling atlas structures."""

import io
import json

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


def _fake_s3_factory(exists, get_impl, cat_impl=None):
    """Build a fake `s3fs.S3FileSystem` class for monkeypatching.

    Parameters
    ----------
    exists : bool or callable
        A constant answer for every path, or a `path -> bool` predicate.
    get_impl : callable
        `(remote, local) -> None`, standing in for `fs.get`.
    cat_impl : callable, optional
        `path -> bytes`, standing in for `fs.cat`. Absent by default, so a
        test that does not opt in fails loudly if the code reaches `cat`.
    """

    class FakeS3FileSystem:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self, path):
            return exists(path) if callable(exists) else exists

        def get(self, remote, local, callback=None):
            return get_impl(remote, local)

        def cat(self, path):
            if cat_impl is None:
                raise AssertionError(f"unexpected cat({path!r})")
            return cat_impl(path)

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


def _legacy_bytes(points, faces):
    """Serialise a mesh to the neuroglancer_legacy_mesh byte layout.

    `<I` vertex count, then N x 3 little-endian float32 vertices, then
    M x 3 little-endian uint32 triangle indices -- exactly what the
    atlas-assets bucket stores at `<id>:0:0`.
    """
    buffer = io.BytesIO()
    mio.write(
        buffer,
        mio.Mesh(points=points, cells=[("triangle", faces)]),
        file_format="neuroglancer",
    )
    return buffer.getvalue()


class _FakeCat:
    """Minimal stand-in for the one `s3fs` method the readers use."""

    def __init__(self, contents):
        self.contents = contents

    def cat(self, path):
        if path not in self.contents:
            raise FileNotFoundError(path)
        return self.contents[path]


# Nanometre-scale so the 16-bit quantization tolerance is realistic.
LEGACY_POINTS = np.array(
    [[0, 0, 0], [1000, 0, 0], [0, 2000, 0], [0, 0, 3000]],
    dtype=np.float32,
)
LEGACY_FACES = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.uint32)


def test_read_legacy_mesh_single_fragment():
    """A one-fragment manifest yields that fragment's points and faces."""
    fs = _FakeCat(
        {
            "root/mesh/997:0": json.dumps({"fragments": ["997:0:0"]}).encode(),
            "root/mesh/997:0:0": _legacy_bytes(LEGACY_POINTS, LEGACY_FACES),
        }
    )

    points, faces = structure_class._read_legacy_mesh(fs, "root/mesh/997:0")

    np.testing.assert_array_equal(points, LEGACY_POINTS)
    np.testing.assert_array_equal(faces, LEGACY_FACES)


def test_read_legacy_mesh_concatenates_fragments():
    """A multi-fragment manifest concatenates points and offsets faces."""
    second_points = LEGACY_POINTS + np.float32(10000)
    fs = _FakeCat(
        {
            "root/mesh/997:0": json.dumps(
                {"fragments": ["997:0:0", "997:0:1"]}
            ).encode(),
            "root/mesh/997:0:0": _legacy_bytes(LEGACY_POINTS, LEGACY_FACES),
            "root/mesh/997:0:1": _legacy_bytes(second_points, LEGACY_FACES),
        }
    )

    points, faces = structure_class._read_legacy_mesh(fs, "root/mesh/997:0")

    assert points.shape == (8, 3)
    np.testing.assert_array_equal(points[:4], LEGACY_POINTS)
    np.testing.assert_array_equal(points[4:], second_points)
    # Second fragment's indices are offset by the first fragment's 4 points.
    np.testing.assert_array_equal(faces[:2], LEGACY_FACES)
    np.testing.assert_array_equal(faces[2:], LEGACY_FACES + 4)


def test_read_legacy_mesh_missing_fragment_raises():
    """A manifest naming an absent fragment raises FileNotFoundError."""
    fs = _FakeCat(
        {
            "root/mesh/997:0": json.dumps({"fragments": ["997:0:0"]}).encode(),
        }
    )

    with pytest.raises(FileNotFoundError):
        structure_class._read_legacy_mesh(fs, "root/mesh/997:0")


def test_read_legacy_mesh_empty_manifest_raises():
    """A manifest listing no fragments raises FileNotFoundError.

    `{"fragments": []}` is the legacy convention for a segment with no
    mesh. It must surface as a type `Structure.__getitem__` already
    catches, not as the bare ValueError numpy would raise.
    """
    fs = _FakeCat({"root/mesh/997:0": json.dumps({"fragments": []}).encode()})

    with pytest.raises(FileNotFoundError):
        structure_class._read_legacy_mesh(fs, "root/mesh/997:0")


def test_encode_draco_round_trips_within_quantization_error():
    """Encoded bytes decode back to the input within range / 65536."""
    encoded = structure_class._encode_draco(LEGACY_POINTS, LEGACY_FACES)
    decoded = DracoPy.decode(encoded)

    tolerance = 3000.0 / 65536  # bounding-cube range / 2**16
    assert np.abs(decoded.points - LEGACY_POINTS).max() <= tolerance
    np.testing.assert_array_equal(decoded.faces, LEGACY_FACES)


def _legacy_fakes(points, faces, fragments=("997:0:0",)):
    """Build (exists, cat) fakes for a bucket with only the legacy layout.

    `_download_mesh` derives the remote path from the last six segments of
    the local path, which varies with `tmp_path`, so the fakes key off path
    shape rather than a hard-coded absolute path: `exists` matches any path
    ending in `:0`, and `cat` distinguishes the manifest from a fragment by
    colon count (`997:0` has one, `997:0:0` has two or more).
    """
    manifest = json.dumps({"fragments": list(fragments)}).encode()
    blobs = {}
    for i, fragment in enumerate(fragments):
        # Offset each fragment's points to make them distinct
        fragment_points = points + np.float32(i * 10000)
        blobs[fragment] = _legacy_bytes(fragment_points, faces)

    def exists(path):
        return path.endswith(":0")

    def cat(path):
        last_part = path.rsplit("/", 1)[1]
        # Manifest has 1 colon (997:0), fragments have 2+ (997:0:0)
        if last_part.count(":") == 1:
            return manifest
        return blobs[last_part]

    return exists, cat


def test_legacy_mesh_converted_on_download(tmp_path, monkeypatch):
    """An atlas-assets structure is converted and read back correctly.

    Round trip through `__getitem__`: the legacy fragment is fetched,
    Draco-encoded to `<id>`, then `_read_mesh` applies nm -> um and
    XYZ -> ZYX.
    """
    mesh_file = tmp_path / "997"
    exists, cat = _legacy_fakes(LEGACY_POINTS, LEGACY_FACES)

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: pytest.fail("used fs.get"),
            cat_impl=cat,
        ),
    )

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]["mesh_filename"] = mesh_file
    mesh = struct_dict["root"]["mesh"]

    assert mesh_file.exists()
    expected = (LEGACY_POINTS / 1000.0)[:, [2, 1, 0]]
    tolerance = (3000.0 / 65536) / 1000.0  # quantization error, in um
    assert np.abs(mesh.points - expected).max() <= tolerance
    np.testing.assert_array_equal(
        mesh.cells[0].data, LEGACY_FACES[:, [2, 1, 0]]
    )


def test_legacy_conversion_writes_only_the_id_file(tmp_path, monkeypatch):
    """Conversion writes `<id>` alone -- no `<id>.index`, no `info`."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    mesh_file = mesh_dir / "997"
    exists, cat = _legacy_fakes(LEGACY_POINTS, LEGACY_FACES)

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: None,
            cat_impl=cat,
        ),
    )

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]._download_mesh(mesh_file)

    assert [p.name for p in mesh_dir.iterdir()] == ["997"]


def test_legacy_conversion_concatenates_fragments(tmp_path, monkeypatch):
    """A two-fragment manifest produces one mesh with all the geometry."""
    mesh_file = tmp_path / "997"
    exists, cat = _legacy_fakes(
        LEGACY_POINTS,
        LEGACY_FACES,
        fragments=("997:0:0", "997:0:1"),
    )

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: None,
            cat_impl=cat,
        ),
    )

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]._download_mesh(mesh_file)
    decoded = DracoPy.decode(mesh_file.read_bytes())

    assert len(decoded.points) == 2 * len(LEGACY_POINTS)
    assert decoded.faces.max() == 2 * len(LEGACY_POINTS) - 1


def test_brainglobe_mesh_skips_the_legacy_probe(tmp_path, monkeypatch):
    """A present `<id>` downloads directly, never probing `<id>:0`."""
    mesh_file = tmp_path / "997"
    probed = []

    def exists(path):
        probed.append(path)
        return not path.endswith(":0")

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: local.write_bytes(_draco_bytes()),
        ),
    )

    struct_dict = StructuresDict(structures_list)
    struct_dict["root"]._download_mesh(mesh_file)

    assert not any(path.endswith(":0") for path in probed)
    assert mesh_file.exists()


def test_interrupted_conversion_removes_the_partial_file(
    tmp_path, monkeypatch
):
    """A `cat` failure mid-conversion leaves no `<id>` behind.

    Extends the corrupt-file guarantee that
    `test_download_mesh_removes_corrupt_file_on_error` asserts for the
    BrainGlobe branch to the conversion branch.
    """
    mesh_file = tmp_path / "997"
    mesh_file.write_bytes(b"stale partial file")

    def failing_cat(path):
        raise ConnectionError("network dropped mid-conversion")

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=lambda path: path.endswith(":0"),
            get_impl=lambda remote, local: None,
            cat_impl=failing_cat,
        ),
    )

    struct_dict = StructuresDict(structures_list)

    with pytest.raises(ConnectionError):
        struct_dict["root"]._download_mesh(mesh_file)

    assert not mesh_file.exists()
