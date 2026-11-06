"""Test the StructuresDict class for handling atlas structures."""

import json
from types import SimpleNamespace

import DracoPy
import meshio as mio
import numpy as np
import pytest
import s3fs

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


ATLAS_ASSETS_ROOT = descriptors.ATLAS_ASSETS_REMOTE_ROOT

# A real fragment from the atlas-assets bucket: structure 332 of
# allen-adult-mouse-annotation/2017, the smallest mesh there that is also
# labelled in the annotation volume, so its coordinates can be checked
# against the atlas rather than against this module's own arithmetic.
# Filled by `load_allen_mesh`.
ALLEN = SimpleNamespace()
ALLEN_ID = "332"

# Where structure 332 sits, in nanometres and the bucket's XYZ order.
# Cross-checked against the 100 um annotation volume, whose label 332
# spans (5.2, 6.2, 6.2) mm to (6.2, 6.3, 6.3) mm in ZYX.
ALLEN_BBOX_NM = (
    np.array([5200000.0, 6150000.0, 6120000.0]),
    np.array([6200000.0, 6300000.0, 6300000.0]),
)
ALLEN_VERTEX_COUNT = 3088
ALLEN_FACE_COUNT = 6168
QUANTIZATION = float(2**16 - 1)


@pytest.fixture(scope="module", autouse=True)
def load_allen_mesh(atlas_assets_atlas):
    """Load the raw fragment, fetching it into the atlas cache if missing.

    The atlas only caches a mesh once converted, and these tests exercise
    that conversion, so the bucket's bytes are stored beside it. Autouse and
    module-scoped so it runs before any test replaces `s3fs.S3FileSystem`.

    Parameters
    ----------
    atlas_assets_atlas : BrainGlobeAtlas
        Shared atlas-assets atlas fixture.
    """
    annotation = atlas_assets_atlas.metadata["annotation_set"]["location"][1:]
    ALLEN.remote_dir = (
        f"{ATLAS_ASSETS_ROOT}/{annotation}/{descriptors.V3_MESHES_DIRECTORY}"
    )
    ALLEN.local_dir = atlas_assets_atlas.root_dir / "test-raw-meshes"
    ALLEN.file_names = (ALLEN_ID, f"{ALLEN_ID}.index", "info")

    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    ALLEN.local_dir.mkdir(exist_ok=True)
    for name in ALLEN.file_names:
        if not (ALLEN.local_dir / name).exists():
            fs.get(f"{ALLEN.remote_dir}/{name}", ALLEN.local_dir / name)

    ALLEN.fragment = (ALLEN.local_dir / ALLEN_ID).read_bytes()
    ALLEN.index = (ALLEN.local_dir / f"{ALLEN_ID}.index").read_bytes()
    ALLEN.info = (ALLEN.local_dir / "info").read_bytes()
    # The chunk the fragment is quantized into, as its index declares it.
    ALLEN.chunk_shape = np.frombuffer(
        ALLEN.index, np.float32, count=3, offset=0
    )


def test_cached_raw_mesh_matches_remote():
    """The cached fragment is what the bucket holds today.

    Keeps the conversion tests below on the bucket's current mesh layout.
    """
    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    stale = [
        name
        for name in ALLEN.file_names
        if (ALLEN.local_dir / name).read_bytes()
        != fs.cat_file(f"{ALLEN.remote_dir}/{name}")
    ]

    assert not stale, (
        f"{ALLEN.remote_dir} changed in place. Delete {ALLEN.local_dir} "
        f"(in CI, the atlas cache) to test against it. Differs: {stale}"
    )


def _index_bytes(
    vertex_offsets=(0.0, 0.0, 0.0),
    fragment_position=(0, 0, 0),
    fragment_positions=None,
    fragment_sizes=None,
    num_lods=1,
    fragments_per_lod=(1,),
):
    """Rebuild the real index header with fields overridden.

    Little-endian: chunk shape and grid origin as 3 x float32, the level
    count as uint32, then per level a float32 scale, three float32 vertex
    offsets and a uint32 fragment count, then the fragment positions and
    sizes. The chunk shape and grid origin are kept from the real
    fragment so the geometry stays the bucket's.

    Parameters
    ----------
    fragment_positions : sequence of (x, y, z), optional
        Per-fragment grid positions for lod 0, one row per fragment.
        Overrides `fragment_position` (tiled to every fragment) when
        given.
    fragment_sizes : sequence of int, optional
        Byte size of each lod-0 fragment within the mesh payload, in the
        same order as `fragment_positions`. Defaults to the whole
        payload in a single fragment.
    """
    parts = [
        ALLEN.index[:24],  # chunk shape and grid origin, as shipped
        np.uint32(num_lods).astype("<u4").tobytes(),
        np.ones(num_lods, dtype="<f4").tobytes(),
        np.tile(np.asarray(vertex_offsets, dtype="<f4"), num_lods).tobytes(),
        np.asarray(fragments_per_lod, dtype="<u4").tobytes(),
    ]
    for lod, count in enumerate(fragments_per_lod):
        if lod == 0 and fragment_positions is not None:
            positions = np.asarray(fragment_positions, dtype="<u4")
        else:
            positions = np.tile(
                np.asarray(fragment_position, dtype="<u4"), (count, 1)
            )
        # Stored as all-x, then all-y, then all-z (see _read_multilod_draco).
        parts.append(positions.tobytes(order="F"))

        if lod == 0 and fragment_sizes is not None:
            sizes = np.asarray(fragment_sizes, dtype="<u4")
        else:
            sizes = np.zeros(count, dtype="<u4")
            if count == 1:
                sizes[0] = len(ALLEN.fragment)
        parts.append(sizes.tobytes())
    return b"".join(parts)


def _info_bytes(transform=None, bits=16):
    """Serialise a `mesh/info` document."""
    if transform is None:
        transform = json.loads(ALLEN.info)["transform"]
    return json.dumps(
        {
            "@type": "neuroglancer_multilod_draco",
            "vertex_quantization_bits": bits,
            "transform": transform,
        }
    ).encode()


class _FakeCat:
    """Minimal stand-in for the one `s3fs` method the readers use."""

    def __init__(self, contents):
        self.contents = contents

    def cat(self, path):
        if path not in self.contents:
            raise FileNotFoundError(path)
        return self.contents[path]


def _mesh_contents(index=None, info=None, payload=None):
    """Bucket contents for one fragment: `<id>`, `<id>.index` and `info`."""
    return {
        f"root/mesh/{ALLEN_ID}": (
            ALLEN.fragment if payload is None else payload
        ),
        f"root/mesh/{ALLEN_ID}.index": ALLEN.index if index is None else index,
        "root/mesh/info": ALLEN.info if info is None else info,
    }


ALLEN_PATH = f"root/mesh/{ALLEN_ID}"


def test_read_multilod_draco_recovers_allen_coordinates():
    """A real fragment lands where the annotation volume says it should.

    The Draco payload alone decodes to 16-bit grid integers, far below these
    coordinates, so this is what separates a correct read from one that skips
    the index.
    """
    fs = _FakeCat(_mesh_contents())

    points, faces = structure_class._read_multilod_draco(fs, ALLEN_PATH)

    assert points.shape == (ALLEN_VERTEX_COUNT, 3)
    assert faces.shape == (ALLEN_FACE_COUNT, 3)
    lower, upper = ALLEN_BBOX_NM
    # One 10 um voxel of slack on the surface extraction.
    np.testing.assert_allclose(points.min(axis=0), lower, atol=10000)
    np.testing.assert_allclose(points.max(axis=0), upper, atol=10000)

    quantized = np.asarray(DracoPy.decode(ALLEN.fragment).points)
    assert quantized.max() <= QUANTIZATION < lower.min(), (
        "fragment is no longer chunk-quantized; the test would pass "
        "even without the index"
    )


def test_read_multilod_draco_honours_offsets_and_fragment_position():
    """Vertex offsets and a non-zero fragment position both shift the mesh."""
    offsets = (1.0, 2.0, 3.0)
    position = (1, 2, 3)
    base, _ = structure_class._read_multilod_draco(
        _FakeCat(_mesh_contents(index=_index_bytes())), ALLEN_PATH
    )
    shifted, _ = structure_class._read_multilod_draco(
        _FakeCat(
            _mesh_contents(
                index=_index_bytes(
                    vertex_offsets=offsets, fragment_position=position
                )
            )
        ),
        ALLEN_PATH,
    )

    affine = np.asarray(json.loads(ALLEN.info)["transform"]).reshape(3, 4)
    expected_shift = (
        np.asarray(offsets) + ALLEN.chunk_shape * np.asarray(position)
    ) @ affine[:, :3].T

    np.testing.assert_allclose(
        shifted - base,
        np.broadcast_to(expected_shift, shifted.shape),
        rtol=1e-5,
    )


def test_read_multilod_draco_applies_the_info_transform():
    """`info`'s 3 x 4 affine is applied, translation column included."""
    scaled = list(np.asarray(json.loads(ALLEN.info)["transform"]) * 2.0)
    scaled[3], scaled[7], scaled[11] = 5.0, 6.0, 7.0  # translation column
    base, _ = structure_class._read_multilod_draco(
        _FakeCat(_mesh_contents()), ALLEN_PATH
    )
    points, _ = structure_class._read_multilod_draco(
        _FakeCat(_mesh_contents(info=_info_bytes(transform=scaled))),
        ALLEN_PATH,
    )

    np.testing.assert_allclose(
        points, base * 2.0 + np.array([5.0, 6.0, 7.0]), rtol=1e-5
    )


def test_read_multilod_draco_rejects_multiple_levels():
    """More than one level of detail is refused rather than guessed at."""
    fs = _FakeCat(
        _mesh_contents(
            index=_index_bytes(num_lods=2, fragments_per_lod=(1, 1))
        )
    )

    with pytest.raises(NotImplementedError):
        structure_class._read_multilod_draco(fs, ALLEN_PATH)


def test_read_multilod_draco_all_empty_fragments_raises():
    """A mesh whose every fragment is empty says so, not a numpy error."""
    fs = _FakeCat(
        _mesh_contents(index=_index_bytes(fragment_sizes=[0]), payload=b"")
    )

    with pytest.raises(ValueError, match="no mesh data"):
        structure_class._read_multilod_draco(fs, ALLEN_PATH)


def test_read_multilod_draco_assembles_multiple_fragments():
    """Several lod-0 fragments are each placed, then concatenated.

    Regression test for the large-structure case (e.g. root/whole-brain)
    where the mesh is chunked into more than one fragment.
    """
    payload_a = _draco_bytes()
    payload_b = _draco_bytes()
    position_a = (0, 0, 0)
    position_b = (1, 2, 3)

    fs = _FakeCat(
        _mesh_contents(
            index=_index_bytes(
                fragments_per_lod=(2,),
                fragment_positions=(position_a, position_b),
                fragment_sizes=(len(payload_a), len(payload_b)),
            ),
            payload=payload_a + payload_b,
        )
    )

    points, faces = structure_class._read_multilod_draco(fs, ALLEN_PATH)

    fragment_a = DracoPy.decode(payload_a)
    fragment_b = DracoPy.decode(payload_b)
    n_a = len(fragment_a.points)

    assert points.shape[0] == n_a + len(fragment_b.points)
    assert faces.shape[0] == len(fragment_a.faces) + len(fragment_b.faces)
    # Fragment b's faces are shifted past fragment a's vertices.
    np.testing.assert_array_equal(
        faces[len(fragment_a.faces) :] - n_a,
        np.asarray(fragment_b.faces, dtype=np.uint32),
    )

    info = json.loads(ALLEN.info)
    quantization = float(2 ** info["vertex_quantization_bits"] - 1)
    grid_origin = np.frombuffer(ALLEN.index, np.float32, count=3, offset=12)
    transform = np.asarray(info["transform"]).reshape(3, 4)

    def _expected(quantized, position):
        stored = grid_origin + ALLEN.chunk_shape * (
            np.asarray(position) + np.asarray(quantized) / quantization
        )
        return stored @ transform[:, :3].T + transform[:, 3]

    np.testing.assert_allclose(
        points[:n_a], _expected(fragment_a.points, position_a), rtol=1e-5
    )
    np.testing.assert_allclose(
        points[n_a:], _expected(fragment_b.points, position_b), rtol=1e-5
    )


def test_read_multilod_draco_missing_index_raises():
    """A fragment with no index raises rather than decoding grid units."""
    contents = _mesh_contents()
    del contents[f"root/mesh/{ALLEN_ID}.index"]

    with pytest.raises(FileNotFoundError):
        structure_class._read_multilod_draco(_FakeCat(contents), ALLEN_PATH)


def test_encode_draco_round_trips_within_quantization_error():
    """Encoded bytes decode back to the input within range / 65536."""
    points, faces = structure_class._read_multilod_draco(
        _FakeCat(_mesh_contents()), ALLEN_PATH
    )
    decoded = DracoPy.decode(structure_class._encode_draco(points, faces))

    span = float(np.ptp(points, axis=0).max())
    assert np.abs(decoded.points - points).max() <= span / 65536
    np.testing.assert_array_equal(decoded.faces, faces)


def _mesh_fakes(contents=None):
    """Build (exists, cat) fakes for a bucket in the atlas-assets layout.

    `_download_mesh` derives the remote path from the last six segments of
    the local path, which varies with `tmp_path`, so `cat` keys off the
    trailing path segment rather than a hard-coded absolute path.
    """
    contents = _mesh_contents() if contents is None else contents
    blobs = {key.rsplit("/", 1)[1]: value for key, value in contents.items()}

    def exists(path):
        return True

    def cat(path):
        name = path.rsplit("/", 1)[1]
        if name not in blobs:
            raise FileNotFoundError(path)
        return blobs[name]

    return exists, cat


def test_atlas_assets_mesh_converted_on_download(tmp_path, monkeypatch):
    """A real Allen fragment is dequantized and read back correctly.

    Round trip through `__getitem__`: the fragment is fetched and
    dequantized, re-encoded to `<id>`, then `_read_mesh` applies
    nm -> um and XYZ -> ZYX.
    """
    mesh_file = tmp_path / ALLEN_ID
    exists, cat = _mesh_fakes()

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: pytest.fail("used fs.get"),
            cat_impl=cat,
        ),
    )

    struct_dict = StructuresDict(
        structures_list,
        remote_root=ATLAS_ASSETS_ROOT,
        dequantize_meshes=True,
    )
    struct_dict["root"]["mesh_filename"] = mesh_file
    mesh = struct_dict["root"]["mesh"]

    assert mesh_file.exists()
    lower, upper = (b[::-1] / 1000.0 for b in ALLEN_BBOX_NM)  # um, ZYX
    np.testing.assert_allclose(mesh.points.min(axis=0), lower, atol=10)
    np.testing.assert_allclose(mesh.points.max(axis=0), upper, atol=10)
    assert len(mesh.cells[0].data) == ALLEN_FACE_COUNT


def test_atlas_assets_conversion_writes_only_the_id_file(
    tmp_path, monkeypatch
):
    """Conversion writes `<id>` alone -- no `<id>.index`, no `info`."""
    mesh_dir = tmp_path / "mesh"
    mesh_dir.mkdir()
    mesh_file = mesh_dir / ALLEN_ID
    exists, cat = _mesh_fakes()

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=exists,
            get_impl=lambda remote, local: None,
            cat_impl=cat,
        ),
    )

    struct_dict = StructuresDict(
        structures_list,
        remote_root=ATLAS_ASSETS_ROOT,
        dequantize_meshes=True,
    )
    struct_dict["root"]._download_mesh(mesh_file)

    assert [p.name for p in mesh_dir.iterdir()] == [ALLEN_ID]


@pytest.mark.parametrize("remote_root", [None, "s3://brainglobe/atlas-rc2"])
def test_brainglobe_mesh_downloads_without_dequantizing(
    tmp_path, monkeypatch, remote_root
):
    """A BrainGlobe mesh is fetched as-is: its Draco header is absolute.

    `cat_impl` is left unset, so the fake raises if the conversion branch
    is taken. The format, not the remote root, picks the branch, so a
    BrainGlobe mirror on another root is fetched as-is too.
    """
    mesh_file = tmp_path / "997"

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=True,
            get_impl=lambda remote, local: local.write_bytes(_draco_bytes()),
        ),
    )

    struct_dict = StructuresDict(structures_list, remote_root=remote_root)
    struct_dict["root"]._download_mesh(mesh_file)

    assert mesh_file.read_bytes() == _draco_bytes()


def test_interrupted_conversion_removes_the_partial_file(
    tmp_path, monkeypatch
):
    """A `cat` failure mid-conversion leaves no `<id>` behind.

    Extends the corrupt-file guarantee that
    `test_download_mesh_removes_corrupt_file_on_error` asserts for the
    BrainGlobe branch to the conversion branch.
    """
    mesh_file = tmp_path / ALLEN_ID
    mesh_file.write_bytes(b"stale partial file")

    def failing_cat(path):
        raise ConnectionError("network dropped mid-conversion")

    monkeypatch.setattr(
        structure_class.s3fs,
        "S3FileSystem",
        _fake_s3_factory(
            exists=True,
            get_impl=lambda remote, local: None,
            cat_impl=failing_cat,
        ),
    )

    struct_dict = StructuresDict(
        structures_list,
        remote_root=ATLAS_ASSETS_ROOT,
        dequantize_meshes=True,
    )

    with pytest.raises(ConnectionError):
        struct_dict["root"]._download_mesh(mesh_file)

    assert not mesh_file.exists()
