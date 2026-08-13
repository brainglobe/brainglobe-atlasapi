"""Tests for reading atlases that follow the atlas-assets specification."""

import copy
import json
from pathlib import Path

import DracoPy
import numpy as np
import pytest
import s3fs
import zarr

from brainglobe_atlasapi import bg_atlas, descriptors
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.structure_class import Structure

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture()
def atlas_assets_manifest():
    """Return a real atlas-assets atlas manifest.

    Returns
    -------
    dict
        Parsed manifest contents.
    """
    return json.loads((DATA_DIR / "manifest_atlas_assets.json").read_text())


@pytest.fixture()
def ome_attrs():
    """Return the OME attributes block of an atlas-assets zarr.

    Returns
    -------
    dict
        Parsed ``attributes.ome`` contents.
    """
    return json.loads((DATA_DIR / "ome_attrs_atlas_assets.json").read_text())


@pytest.mark.parametrize(
    "atlas_name, expected",
    [
        ("allen-adult-mouse-ccf-atlas", "Mus musculus"),
        ("allen-dev-mouse-e11pt5-nissl-atlas", "Mus musculus"),
        ("hmba-adult-human-homba-atlas", "Homo sapiens"),
        ("hmba-adult-marmoset-homba-atlas", "Callithrix jacchus"),
        ("hmba-adult-rhesusmacaque-homba-atlas", "Macaca mulatta"),
        (
            "hmba-adult-cynomolgusmacaque-homba-atlas",
            "Macaca fascicularis",
        ),
        ("allen-dev-P4-mouse-atlas", "Mus musculus"),
        ("some-unknown-organism-atlas", None),
    ],
)
def test_species_from_name(atlas_name, expected):
    """Species is matched by vocabulary, not by segment position.

    Parameters
    ----------
    atlas_name : str
        Atlas name under test.
    expected : str or None
        Expected binomial name, or None when no segment matches.
    """
    assert bg_atlas._species_from_name(atlas_name) == expected


def test_orientation_from_axes(ome_attrs):
    """Anatomical axis directions map to brainglobe-space letters.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert bg_atlas._orientation_from_axes(ome_attrs) == "asl"


def test_pyramid_level_from_attrs(ome_attrs):
    """Resolution in micrometres selects the matching dataset.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert bg_atlas._pyramid_level_from_attrs(ome_attrs, 10) == (0, "s0")
    assert bg_atlas._pyramid_level_from_attrs(ome_attrs, 25) == (1, "s1")


def test_pyramid_level_from_attrs_invalid(ome_attrs):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._pyramid_level_from_attrs(ome_attrs, 37)
    assert "37" in str(error.value)


def test_brainglobe_manifest_passes_through():
    """A BrainGlobe manifest is returned unchanged.

    ``_normalize_manifest`` returns the same object for a BrainGlobe
    manifest (``raw is result``), so comparing the result against
    ``raw`` after the call would pass even if the function mutated
    ``raw`` in place. Comparing against an untouched deep copy, taken
    before the call, actually exercises that guarantee.
    """
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    untouched = copy.deepcopy(raw)
    result = bg_atlas._normalize_manifest(raw, None, Path("."))
    assert result == untouched


def test_resolution_required_for_multi_scale(atlas_assets_manifest):
    """Omitting resolution on a multi-scale atlas fails loudly.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(atlas_assets_manifest, None, Path("."))
    assert "resolution=" in str(error.value)
    assert "100" in str(error.value)


def test_resolution_defaulted_for_single_scale(atlas_assets_manifest):
    """A single-scale atlas-assets manifest defaults resolution.

    The zarr read that follows resolution selection needs a real store
    on disk, which this test does not provide, so it can only exercise
    the scales-defaulting decision itself: a single-scale manifest with
    resolution=None must not raise the "multiple resolutions"
    ValueError, proving resolution was defaulted rather than left
    unset. Execution then reaches the (missing) zarr store and fails
    for that unrelated reason instead.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    atlas_assets_manifest["annotation_sets"][0]["scales"] = [25]
    with pytest.raises(FileNotFoundError):
        bg_atlas._normalize_manifest(atlas_assets_manifest, None, Path("."))


def test_resolution_must_be_an_available_scale(atlas_assets_manifest):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(atlas_assets_manifest, 37, Path("."))
    assert "37" in str(error.value)


def test_resolution_rejected_when_it_contradicts_manifest():
    """A resolution disagreeing with a BrainGlobe manifest fails."""
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(raw, 10, Path("."))
    assert "25.0" in str(error.value)


ATLAS_ASSETS_NAME = "allen-adult-mouse-ccf-atlas"

_ANNOTATION_SET = (
    f"{descriptors.ATLAS_ASSETS_REMOTE_ROOT}/annotation-sets"
    f"/allen-adult-mouse-annotation/2017"
)

ATLAS_ASSETS_SPATIAL_ASSETS = {
    # 4-D (channel, z, y, x), so this also exercises the [-3:] axis guard.
    "annotation": f"{_ANNOTATION_SET}/{descriptors.V3_ANNOTATION_NAME}",
    "annotation_masks": (
        f"{_ANNOTATION_SET}/{descriptors.V3_ANNOTATION_MASKS_NAME}"
    ),
    "template": (
        f"{descriptors.ATLAS_ASSETS_REMOTE_ROOT}/templates"
        f"/allen-adult-mouse-stpt-template/2015"
        f"/{descriptors.V3_TEMPLATE_NAME}"
    ),
}

# The 2017 annotation set's precomputed `info` declares
# size = [1140, 800, 1320] voxels at resolution = [10000] * 3 nm, in
# precomputed XYZ. Vertices are implicitly in the enclosing volume's
# units, which is the assumption this test exists to pin.
_ANNOTATION_EXTENT_NM = np.array([1140, 800, 1320]) * 10000.0

# The bucket's zarrs are still pre-release OME-Zarr: version "0.6.dev3",
# `coordinateSystems` at the `ome` level rather than per-multiscale, and
# transform `input`/`output` as bare strings rather than `{"path": ...}` /
# `{"name": ...}` objects. ngff-zarr rejects the version string outright.
# Every test that builds a BrainGlobeAtlas from this bucket therefore
# cannot pass until it is republished against released OME-Zarr 0.6. They
# are strict xfails so the suite stays green now and fails loudly, as
# XPASS, the moment that happens.
blocked_on_ome_version = pytest.mark.xfail(
    strict=True,
    reason="atlas-assets bucket declares OME-Zarr 0.6.dev3; see "
    "descriptors.ATLAS_ASSETS_REMOTE_ROOT",
)


@blocked_on_ome_version
@pytest.mark.slow
def test_load_atlas_assets_atlas():
    """A spec-conformant atlas loads at a requested resolution."""
    atlas = BrainGlobeAtlas(ATLAS_ASSETS_NAME, resolution=100)

    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.orientation == "asl"
    assert len(atlas.shape) == 3
    assert atlas.metadata["species"] == "Mus musculus"
    assert atlas.metadata["citation"] is None
    assert atlas.structures["root"]["id"] == 997


@blocked_on_ome_version
@pytest.mark.slow
def test_atlas_assets_cache_is_namespaced():
    """Downloads land under the configured root's directory."""
    atlas = BrainGlobeAtlas(ATLAS_ASSETS_NAME, resolution=100)
    assert atlas.brainglobe_dir.name == "allen"


@blocked_on_ome_version
@pytest.mark.slow
def test_atlas_assets_hemispheres_are_synthesized():
    """No hemispheres asset exists, so hemispheres are synthesized."""
    atlas = BrainGlobeAtlas(ATLAS_ASSETS_NAME, resolution=100)
    hemispheres = atlas.hemispheres
    assert hemispheres.shape == tuple(atlas.shape)
    assert set(np.unique(hemispheres)) <= {1, 2}


@pytest.mark.slow
def test_orientation_is_consistent_across_an_atlas_assets_atlas():
    """Every spatial asset of one atlas declares the same axis directions.

    An atlas is returned in the orientation it is stored in, so the only
    orientation requirement is internal: the annotation and the template
    must agree, because normalisation derives the atlas-wide orientation
    from the annotation alone and `core.Atlas` then applies it to both.

    Deliberately reads the zarr metadata with `zarr` rather than through
    `BrainGlobeAtlas`, so it exercises the live bucket without depending
    on the OME version that currently blocks the tests below.
    """
    # skip_instance_cache: an earlier test in this file builds a
    # BrainGlobeAtlas, which creates its own S3FileSystem and then raises.
    # Reusing that cached instance here inherits its torn-down session.
    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    origins = {}
    for name, url in ATLAS_ASSETS_SPATIAL_ASSETS.items():
        store = zarr.storage.FsspecStore(fs, path=url[len("s3://") :])
        attrs = zarr.open_group(store, mode="r").attrs["ome"]
        origins[name] = bg_atlas._orientation_from_axes(attrs)

    assert set(origins.values()) == {"asl"}, origins


@blocked_on_ome_version
@pytest.mark.slow
def test_atlas_assets_structure_mask():
    """A structure mask lies inside the annotation's non-zero extent."""
    atlas = BrainGlobeAtlas(ATLAS_ASSETS_NAME, resolution=100)
    mask = atlas.get_structure_mask("root")
    assert mask.shape == tuple(atlas.shape)
    assert mask.max() == 1


@pytest.mark.slow
def test_atlas_assets_legacy_mesh_is_nanometres_in_xyz(tmp_path):
    """Real Allen legacy vertices land inside the declared volume extent.

    Drives `Structure._download_mesh` directly rather than through
    `BrainGlobeAtlas`, which cannot be constructed from this bucket while
    gap 1 (OME-Zarr 0.6.dev3) stands. This is therefore the only check
    that exercises the conversion against real Allen bytes.
    """
    # `_download_mesh` derives the remote path from the last six segments
    # of the local path, so the local tree has to mirror the remote one.
    mesh_dir = (
        tmp_path
        / "annotation-sets"
        / "allen-adult-mouse-annotation"
        / "2017"
        / descriptors.V3_MESHES_DIRECTORY
    )
    mesh_dir.mkdir(parents=True)
    mesh_file = mesh_dir / "997"

    struct = Structure(id=997, acronym="root", mesh_filename=mesh_file)
    struct.remote_root = descriptors.ATLAS_ASSETS_REMOTE_ROOT
    struct._download_mesh(mesh_file)

    points = DracoPy.decode(mesh_file.read_bytes()).points
    lower, upper = points.min(axis=0), points.max(axis=0)

    # Inside the declared box on every axis, with 1% slack for the
    # surface extraction overshooting the voxel grid at the boundary.
    slack = 0.01 * _ANNOTATION_EXTENT_NM
    assert np.all(lower >= -slack), lower
    assert np.all(upper <= _ANNOTATION_EXTENT_NM + slack), upper
    # ...and actually filling it, which is what rules out a unit or axis
    # mix-up: a micrometre mesh would be 1000x too small, and a ZYX mesh
    # would overflow the 1140-voxel first axis with the 1320-voxel one.
    assert np.all(upper >= 0.9 * _ANNOTATION_EXTENT_NM), upper
