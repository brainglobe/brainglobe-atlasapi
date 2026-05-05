"""
Unit tests for pure helper functions in wrapup.py.

Tests for overwrite and early-exit behaviour in wrapup_atlas_from_data.

These tests verify that atlas generation fails early when output already
exists, and that the overwrite flag correctly replaces existing output.
"""

import json

import numpy as np
import pandas as pd
import pytest

from brainglobe_atlasapi.atlas_generation.wrapup import (
    _build_transformations,
    _merge_resolutions_list,
    _save_coordinate_space_manifest,
    _save_if_not_exists,
    _save_terminology_csv,
    wrapup_atlas_from_data,
)
from brainglobe_atlasapi.descriptors import ATLAS_VERSION

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


@pytest.fixture
def minimal_valid_inputs(tmp_path):
    """
    Return a minimal set of valid inputs required to run wrapup
    past the overwrite logic without performing actual atlas gen.
    """
    return dict(
        atlas_name="test_mouse",
        atlas_minor_version=0,
        citation="unpublished",
        atlas_link="http://example.com",
        species="Mouse (Mus musculus)",
        resolution=(25, 25, 25),
        orientation="asr",
        root_id=0,
        reference_stack=np.zeros((2, 2, 2)),
        annotation_stack=np.zeros((2, 2, 2), dtype=np.uint32),
        structures_list=[
            {
                "id": 0,
                "acronym": "root",
                "name": "root",
                "rgb_triplet": [255, 255, 255],
                "structure_id_path": [0],
            }
        ],
        meshes_dict={},
        working_dir=tmp_path,
    )


@pytest.fixture
def expected_atlas_dir(minimal_valid_inputs):
    """Return the path where wrapup writes the atlas manifest."""
    atlas_version = f"{ATLAS_VERSION}.0".replace(".", "_")
    return (
        minimal_valid_inputs["working_dir"]
        / "brainglobe-atlasapi"
        / "atlases"
        / "test_mouse_25um"
        / atlas_version
    )


def test_wrapup_fails_if_output_exists(
    expected_atlas_dir, minimal_valid_inputs
):
    """Fail early if atlas output already exists and overwrite=False."""
    atlas_dir = expected_atlas_dir
    atlas_dir.mkdir(parents=True)

    kwargs = minimal_valid_inputs

    with pytest.raises(FileExistsError, match="Atlas output already exists"):
        wrapup_atlas_from_data(**kwargs, overwrite=False)


def test_wrapup_overwrites_existing_output(
    monkeypatch, expected_atlas_dir, minimal_valid_inputs
):
    """Overwrite existing atlas output when overwrite=True."""
    atlas_dir = expected_atlas_dir
    atlas_dir.mkdir(parents=True)
    (atlas_dir / "old_file.txt").write_text("old")

    from brainglobe_atlasapi.atlas_generation import wrapup

    # Stub out heavy I/O that runs after the overwrite check.
    class _FakeImage:
        data = type("arr", (), {"shape": (2, 2, 2)})()

    class _FakeMultiscale:
        images = [_FakeImage()]

    monkeypatch.setattr(
        wrapup, "_save_template_data", lambda *a, **kw: _FakeMultiscale()
    )
    monkeypatch.setattr(
        wrapup, "_save_annotation_data", lambda *a, **kw: (None, None)
    )
    monkeypatch.setattr(
        wrapup, "_save_additional_references", lambda *a, **kw: None
    )
    monkeypatch.setattr(wrapup, "get_all_validation_functions", lambda: [])
    monkeypatch.setattr(
        wrapup,
        "BrainGlobeAtlas",
        lambda *args, **kwargs: None,
    )

    kwargs = minimal_valid_inputs
    wrapup_atlas_from_data(**kwargs, overwrite=True)

    assert atlas_dir.exists()
    assert not (atlas_dir / "old_file.txt").exists()
