"""
Unit tests for pure helper functions in wrapup.py.

Tests for overwrite and early-exit behaviour in wrapup_atlas_from_data.

These tests verify that atlas generation fails early when output already
exists, and that the overwrite flag correctly replaces existing output.
"""

import json

import meshio
import ngff_zarr as nz
import numpy as np
import pandas as pd
import pytest

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation import __version__ as ATLAS_VERSION
from brainglobe_atlasapi.atlas_generation.wrapup import (
    _build_transformations,
    _merge_resolutions_list,
    _save_coordinate_space_manifest,
    _save_if_not_exists,
    _save_terminology_csv,
    wrapup_atlas_from_data,
)

ATLAS_NAME = "test_atlas"
RESOLUTION = (25, 25, 25)
MINOR_VERSION = "0"
ROOT_ID = 999

# --- _merge_resolutions_list ---


def test_merge_resolutions_list_combines_and_sorts():
    """Test that merging two lists combines and sorts by resolution."""
    existing = [(100, 100, 100), (50, 50, 50)]
    new = [(25, 25, 25)]
    result = _merge_resolutions_list(existing, new)
    assert result == [(25, 25, 25), (50, 50, 50), (100, 100, 100)]


def test_merge_resolutions_list_deduplicates():
    """Test that duplicate resolutions are removed after merging."""
    existing = [(100, 100, 100)]
    new = [(100, 100, 100)]
    result = _merge_resolutions_list(existing, new)
    assert result == [(100, 100, 100)]


def test_merge_resolutions_list_empty_new():
    """Test that merging with an empty new list returns existing."""
    existing = [(10, 10, 10)]
    result = _merge_resolutions_list(existing, [])
    assert result == [(10, 10, 10)]


# --- _build_transformations ---


def test_build_transformations_divides_by_1000():
    """Test that resolution in microns is converted to mm (divide by 1000)."""
    result = _build_transformations([(10, 10, 10)])
    assert result == [[{"type": "scale", "scale": [0.01, 0.01, 0.01]}]]


def test_build_transformations_multiple_resolutions():
    """Test that multiple resolutions produce multiple transformations."""
    result = _build_transformations([(10, 10, 10), (20, 20, 20)])
    assert len(result) == 2
    assert result[0] == [{"type": "scale", "scale": [0.01, 0.01, 0.01]}]
    assert result[1] == [{"type": "scale", "scale": [0.02, 0.02, 0.02]}]


def test_build_transformations_anisotropic():
    """Test that anisotropic resolutions are handled per-axis."""
    result = _build_transformations([(10, 20, 30)])
    assert result == [[{"type": "scale", "scale": [0.01, 0.02, 0.03]}]]


# --- _save_terminology_csv ---


@pytest.fixture
def simple_structures():
    """Provide a simple list of structures for testing."""
    return [
        {
            "id": 999,
            "name": "root",
            "acronym": "root",
            "rgb_triplet": [255, 255, 255],
            "structure_id_path": [999],
        },
        {
            "id": 1,
            "name": "brain",
            "acronym": "br",
            "rgb_triplet": [100, 150, 200],
            "structure_id_path": [999, 1],
        },
    ]


def test_save_terminology_csv_creates_file(tmp_path, simple_structures):
    """Test that _save_terminology_csv creates the output CSV file."""
    csv_path = tmp_path / "terminology.csv"
    _save_terminology_csv(simple_structures, csv_path)
    assert csv_path.exists()


def test_save_terminology_csv_columns(tmp_path, simple_structures):
    """Test that the CSV has the expected column names in the correct order."""
    csv_path = tmp_path / "terminology.csv"
    _save_terminology_csv(simple_structures, csv_path)
    df = pd.read_csv(csv_path)
    assert list(df.columns) == [
        "identifier",
        "parent_identifier",
        "annotation_value",
        "name",
        "abbreviation",
        "color_hex_triplet",
        "root_identifier_path",
    ]


def test_save_terminology_csv_root_has_no_parent(tmp_path, simple_structures):
    """Test that the root structure has a NaN parent_identifier."""
    csv_path = tmp_path / "terminology.csv"
    _save_terminology_csv(simple_structures, csv_path)
    df = pd.read_csv(csv_path)
    root_row = df[df["identifier"] == 999].iloc[0]
    assert pd.isna(root_row["parent_identifier"])


def test_save_terminology_csv_child_has_correct_parent(
    tmp_path, simple_structures
):
    """Test that a child structure has the correct parent_identifier."""
    csv_path = tmp_path / "terminology.csv"
    _save_terminology_csv(simple_structures, csv_path)
    df = pd.read_csv(csv_path)
    child_row = df[df["identifier"] == 1].iloc[0]
    assert child_row["parent_identifier"] == 999


def test_save_terminology_csv_color_hex_format(tmp_path, simple_structures):
    """Test that RGB triplets are correctly formatted as hex color strings."""
    structures = [
        {
            "id": 1,
            "name": "a",
            "acronym": "a",
            "rgb_triplet": [255, 0, 128],
            "structure_id_path": [1],
        }
    ]
    csv_path = tmp_path / "terminology.csv"
    _save_terminology_csv(structures, csv_path)
    df = pd.read_csv(csv_path)
    assert df["color_hex_triplet"].iloc[0] == "#FF0080"


# --- _save_coordinate_space_manifest ---


def test_save_coordinate_space_manifest_creates_file(tmp_path):
    """Test that _save_coordinate_space_manifest creates the JSON file."""
    metadata = {"axes": ["x", "y", "z"], "units": "um"}
    path = tmp_path / "manifest.json"
    _save_coordinate_space_manifest(metadata, path)
    assert path.exists()


def test_save_coordinate_space_manifest_round_trips_json(tmp_path):
    """Test that metadata is preserved exactly when written and read back."""
    metadata = {"name": "test-space", "version": "1.0"}
    path = tmp_path / "manifest.json"
    _save_coordinate_space_manifest(metadata, path)
    with open(path) as f:
        result = json.load(f)
    assert result == metadata


# --- _save_if_not_exists ---


def test_save_if_not_exists_calls_save_fn_when_dir_absent(tmp_path):
    """Test that save_fn is called with correct args when dest dir absent."""
    dest_dir = tmp_path / "output"
    called_with = {}

    def fake_save(stacks, path, transformations):
        called_with["stacks"] = stacks
        called_with["path"] = path
        path.mkdir(parents=True)

    _save_if_not_exists(
        stacks=["dummy"],
        dest_dir=dest_dir,
        label="test",
        transformations=[],
        save_fn=fake_save,
    )
    assert called_with["stacks"] == ["dummy"]
    assert called_with["path"] == dest_dir


def test_save_if_not_exists_skips_when_dir_exists(tmp_path):
    """Test that save_fn is not called when dest_dir already exists."""
    dest_dir = tmp_path / "output"
    dest_dir.mkdir()
    calls = []

    def fake_save(stacks, path, transformations):
        calls.append(1)

    _save_if_not_exists(
        stacks=["dummy"],
        dest_dir=dest_dir,
        label="test",
        transformations=[],
        save_fn=fake_save,
    )
    assert calls == []


## --- end to end tests ---


@pytest.fixture(scope="module")
def structures_list():
    """Provide a simple list of structures for testing."""
    return [
        {
            "id": ROOT_ID,
            "acronym": "root",
            "name": "root",
            "rgb_triplet": [255, 255, 255],
            "structure_id_path": [ROOT_ID],
        },
        {
            "id": 1,
            "acronym": "reg1",
            "name": "Region 1",
            "rgb_triplet": [100, 150, 200],
            "structure_id_path": [ROOT_ID, 1],
        },
    ]


@pytest.fixture(scope="module")
def fake_volumes():
    """Create minimal fake volumes that satisfy all validators.

    Shape (15, 15, 15) gives enough room for the symmetry check (±5 from
    centre along the x-axis requires at least 11 voxels).
    """
    shape = (15, 15, 15)
    reference = np.full(shape, 200, dtype=np.uint16)  # all > 128
    # Root fills the whole volume; one corner voxel marks Region 1 so it
    # survives filter_structures_not_present_in_annotation without disturbing
    # the symmetry-check voxels at [7,7,2] and [7,7,12].
    annotation = np.full(shape, ROOT_ID, dtype=np.uint32)
    annotation[0, 0, 0] = 1
    # Different from reference so validate_additional_references passes.
    additional_ref = np.full(shape, 50, dtype=np.uint16)
    return reference, annotation, additional_ref


@pytest.fixture(scope="module")
def root_mesh_file(tmp_path_factory):
    """Write a minimal surface mesh for the root structure as an .obj file."""
    mesh_path = tmp_path_factory.mktemp("meshes") / f"{ROOT_ID}.obj"
    points = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
    )
    cells = [
        ("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    ]
    meshio.write(str(mesh_path), meshio.Mesh(points=points, cells=cells))
    return mesh_path


@pytest.fixture(scope="module")
def wrapup_dir(
    tmp_path_factory, structures_list, fake_volumes, root_mesh_file
):
    """Run wrapup_atlas_from_data once; return the brainglobe-atlasapi dir."""
    working_dir = tmp_path_factory.mktemp("wrapup_e2e")
    reference, annotation, additional_ref = fake_volumes

    wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=MINOR_VERSION,
        citation="unpublished",
        atlas_link="https://example.com",
        species="Mus musculus",
        resolution=RESOLUTION,
        orientation="asr",
        root_id=ROOT_ID,
        reference_stack=reference,
        annotation_stack=annotation,
        structures_list=structures_list,
        meshes_dict={ROOT_ID: root_mesh_file, 1: root_mesh_file},
        scale_meshes=False,
        working_dir=working_dir,
        hemispheres_stack=None,
        atlas_packager="Test Packager",
        additional_references=[("secondary", additional_ref)],
    )

    return working_dir / "brainglobe-atlasapi"


@pytest.fixture(scope="module")
def atlas_version():
    """Provide the atlas version string for verification."""
    return f"{ATLAS_VERSION}.{MINOR_VERSION}".replace(".", "_")


@pytest.fixture(scope="module")
def template_stub(atlas_version):
    """Provide the path stub for the template zarr in the atlas directory."""
    return (
        f"templates/{ATLAS_NAME}-template/{atlas_version}"
        f"/{descriptors.V2_TEMPLATE_NAME}"
    )


@pytest.fixture(scope="module")
def annotation_dir(atlas_version):
    """Provide the path stub for the annotation directory in the atlas."""
    return f"annotation-sets/{ATLAS_NAME}-annotation/{atlas_version}"


@pytest.fixture(scope="module")
def atlas_dir(atlas_version):
    """Provide the path stub for the atlas directory itself."""
    return f"atlases/{ATLAS_NAME}_{RESOLUTION[0]}um/{atlas_version}"


def test_template_zarr_exists(wrapup_dir, template_stub):
    """Test that the template zarr file exists at the expected location."""
    assert (wrapup_dir / template_stub).exists()


def test_annotation_zarr_exists(wrapup_dir, annotation_dir):
    """Test that the annotation zarr file exists at the expected location."""
    assert (
        wrapup_dir / annotation_dir / descriptors.V2_ANNOTATION_NAME
    ).exists()


def test_hemispheres_zarr_exists(wrapup_dir, annotation_dir):
    """Test that the hemispheres zarr file exists at the expected location."""
    assert (
        wrapup_dir / annotation_dir / descriptors.V2_HEMISPHERES_NAME
    ).exists()


def test_mesh_directory_exists(wrapup_dir, annotation_dir):
    """Test that the mesh directory exists at the expected location."""
    assert (
        wrapup_dir / annotation_dir / descriptors.V2_MESHES_DIRECTORY
    ).exists()


def test_root_mesh_file_exists(wrapup_dir, annotation_dir):
    """Test that the root mesh file exists at the expected location."""
    assert (
        wrapup_dir
        / annotation_dir
        / descriptors.V2_MESHES_DIRECTORY
        / str(ROOT_ID)
    ).exists()


def test_terminology_csv_exists(wrapup_dir, atlas_version):
    """Test that the terminology CSV file exists at the expected location."""
    assert (
        wrapup_dir
        / f"terminologies/{ATLAS_NAME}-terminology/{atlas_version}"
        / descriptors.V2_TERMINOLOGY_NAME
    ).exists()


def test_coordinate_space_manifest_exists(wrapup_dir, atlas_version):
    """Test that the coordinate space file exists at the expected location."""
    assert (
        wrapup_dir
        / f"coordinate-spaces/{ATLAS_NAME}-coordinate-space/{atlas_version}"
        / "manifest.json"
    ).exists()


def test_atlas_manifest_exists(wrapup_dir, atlas_dir):
    """Test that the atlas manifest file exists at the expected location."""
    assert (wrapup_dir / atlas_dir / "manifest.json").exists()


def test_additional_reference_zarr_exists(wrapup_dir, atlas_version):
    """Test that the additional reference zarr is at the expected location."""
    assert (
        wrapup_dir
        / f"templates/{ATLAS_NAME}-secondary-template/{atlas_version}"
        / descriptors.V2_TEMPLATE_NAME
    ).exists()


def test_template_zarr_shape_and_dtype(wrapup_dir, template_stub):
    """Template zarr is readable and matches the input shape and dtype."""
    ms = nz.from_ngff_zarr(wrapup_dir / template_stub)
    data = ms.images[0].data.compute()
    assert data.shape == (15, 15, 15)
    assert data.dtype == descriptors.REFERENCE_DTYPE


def test_annotation_zarr_shape_and_dtype(wrapup_dir, annotation_dir):
    """Annotation zarr is readable and uses the annotation dtype."""
    ms = nz.from_ngff_zarr(
        wrapup_dir / annotation_dir / descriptors.V2_ANNOTATION_NAME
    )
    data = ms.images[0].data.compute()
    assert data.shape == (15, 15, 15)
    assert data.dtype == descriptors.ANNOTATION_DTYPE


def test_annotation_zarr_values(wrapup_dir, annotation_dir):
    """Annotation contains root and Region 1 labels; root is dominant."""
    ms = nz.from_ngff_zarr(
        wrapup_dir / annotation_dir / descriptors.V2_ANNOTATION_NAME
    )
    data = ms.images[0].data.compute()
    unique = set(np.unique(data).tolist())
    assert unique == {ROOT_ID, 1}
    assert np.sum(data == ROOT_ID) > np.sum(data == 1)


def test_hemispheres_zarr_two_halves(wrapup_dir, annotation_dir):
    """Auto-generated hemispheres volume has exactly two unique labels."""
    ms = nz.from_ngff_zarr(
        wrapup_dir / annotation_dir / descriptors.V2_HEMISPHERES_NAME
    )
    data = ms.images[0].data.compute()
    assert set(np.unique(data)) == {1, 2}


def test_additional_reference_zarr_shape(wrapup_dir, atlas_version):
    """Additional reference zarr has the same spatial shape as the template."""
    ms = nz.from_ngff_zarr(
        wrapup_dir
        / f"templates/{ATLAS_NAME}-secondary-template/{atlas_version}"
        / descriptors.V2_TEMPLATE_NAME
    )
    data = ms.images[0].data.compute()
    assert data.shape == (15, 15, 15)


def test_terminology_csv_columns(wrapup_dir, atlas_version):
    """Terminology CSV has all required columns."""
    csv_path = (
        wrapup_dir
        / f"terminologies/{ATLAS_NAME}-terminology/{atlas_version}"
        / descriptors.V2_TERMINOLOGY_NAME
    )
    df = pd.read_csv(csv_path)
    expected_columns = [
        "identifier",
        "parent_identifier",
        "annotation_value",
        "name",
        "abbreviation",
        "color_hex_triplet",
        "root_identifier_path",
    ]
    assert list(df.columns) == expected_columns


def test_terminology_csv_row_count(wrapup_dir, atlas_version):
    """Terminology CSV has one row per structure."""
    csv_path = (
        wrapup_dir
        / f"terminologies/{ATLAS_NAME}-terminology/{atlas_version}"
        / descriptors.V2_TERMINOLOGY_NAME
    )
    df = pd.read_csv(csv_path)
    assert len(df) == 2  # root + Region 1


def test_terminology_csv_root_has_no_parent(wrapup_dir, atlas_version):
    """Root structure has a NaN parent_identifier."""
    csv_path = (
        wrapup_dir
        / f"terminologies/{ATLAS_NAME}-terminology/{atlas_version}"
        / descriptors.V2_TERMINOLOGY_NAME
    )
    df = pd.read_csv(csv_path)
    root_row = df[df["identifier"] == ROOT_ID].iloc[0]
    assert pd.isna(root_row["parent_identifier"])


def test_terminology_csv_child_has_correct_parent(wrapup_dir, atlas_version):
    """Region 1 has root as its parent."""
    csv_path = (
        wrapup_dir
        / f"terminologies/{ATLAS_NAME}-terminology/{atlas_version}"
        / descriptors.V2_TERMINOLOGY_NAME
    )
    df = pd.read_csv(csv_path)
    child_row = df[df["identifier"] == 1].iloc[0]
    assert child_row["parent_identifier"] == ROOT_ID


def test_coordinate_space_manifest_is_valid_json(wrapup_dir, atlas_version):
    """Coordinate space manifest can be parsed as JSON."""
    path = (
        wrapup_dir
        / f"coordinate-spaces/{ATLAS_NAME}-coordinate-space/{atlas_version}"
        / "manifest.json"
    )
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_atlas_manifest_required_fields(wrapup_dir, atlas_dir):
    """Atlas manifest contains all fields required by METADATA_TEMPLATE."""
    with open(wrapup_dir / atlas_dir / "manifest.json") as f:
        manifest = json.load(f)

    for key in descriptors.METADATA_TEMPLATE:
        assert key in manifest, f"Missing key in manifest: {key}"


def test_atlas_manifest_resolution(wrapup_dir, atlas_dir):
    """Atlas manifest records the correct resolution."""
    with open(wrapup_dir / atlas_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert tuple(manifest["resolution"]) == tuple(float(r) for r in RESOLUTION)


def test_atlas_manifest_shape(wrapup_dir, atlas_dir):
    """Atlas manifest records the correct volume shape."""
    with open(wrapup_dir / atlas_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert tuple(manifest["shape"]) == (15, 15, 15)


def test_atlas_manifest_additional_references(wrapup_dir, atlas_dir):
    """Atlas manifest lists the additional reference."""
    with open(wrapup_dir / atlas_dir / "manifest.json") as f:
        manifest = json.load(f)
    assert "additional_references" in manifest
    assert len(manifest["additional_references"]) == 1
    assert (
        manifest["additional_references"][0]["name"]
        == f"{ATLAS_NAME}-secondary-template"
    )
