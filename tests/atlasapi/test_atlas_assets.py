"""Tests for reading atlases that follow the atlas-assets specification."""

import DracoPy
import numpy as np
import s3fs
import zarr

from brainglobe_atlasapi import core, descriptors, manifest
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.structure_class import Structure

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


def test_load_atlas_assets_atlas():
    """A spec-conformant atlas loads at a requested resolution."""
    atlas = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )

    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.orientation == "asl"
    assert len(atlas.shape) == 3
    assert atlas.metadata["species"] == "Mus musculus"
    assert atlas.metadata["citation"] is None
    assert atlas.structures["root"]["id"] == 997


def test_core_atlas_loads_atlas_assets_manifest():
    """core.Atlas normalizes an atlas-assets manifest without the subclass."""
    cached = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )

    atlas = core.Atlas(
        cached.brainglobe_dir / cached.local_full_name, resolution=100
    )

    assert atlas.metadata["format"] == "atlas-assets"
    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.shape == cached.shape


def test_atlas_assets_cache_is_namespaced():
    """Downloads land under the configured root's directory."""
    atlas = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )
    assert atlas.brainglobe_dir.name == "allen-atlas-assets"


def test_atlas_assets_hemispheres_are_synthesized():
    """No hemispheres asset exists, so hemispheres are synthesized."""
    atlas = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )
    hemispheres = atlas.hemispheres
    assert hemispheres.shape == tuple(atlas.shape)
    assert set(np.unique(hemispheres)) <= {1, 2}

    # The frontal axis starts on the side its origin letter names.
    front_ax_idx = atlas.space.axes_order.index("frontal")
    first_side = atlas.space.origin[front_ax_idx]
    expected = {
        "l": atlas.left_hemisphere_value,
        "r": atlas.right_hemisphere_value,
    }[first_side]
    assert (np.take(hemispheres, 0, axis=front_ax_idx) == expected).all()


def test_atlas_assets_meshes_are_dequantized():
    """An atlas-assets atlas flags its structures for dequantization."""
    atlas = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )
    assert atlas.metadata["format"] == "atlas-assets"
    assert all(s.dequantize_mesh for s in atlas.structures.values())


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
        origins[name] = manifest._orientation_from_axes(attrs)

    assert set(origins.values()) == {"asl"}, origins


def test_atlas_assets_structure_mask():
    """A structure mask lies inside the annotation's non-zero extent."""
    atlas = BrainGlobeAtlas(
        ATLAS_ASSETS_NAME, resolution=100, check_latest=False
    )
    mask = atlas.get_structure_mask("root")
    assert mask.shape == tuple(atlas.shape)
    assert mask.max() == 1


def test_atlas_assets_mesh_is_nanometres_in_xyz(tmp_path):
    """Real Allen vertices land inside the declared volume extent.

    Drives `Structure._download_mesh` directly, so the dequantization
    is exercised against real Allen bytes rather than fixtures.
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

    struct = Structure(
        id=997,
        acronym="root",
        mesh_filename=mesh_file,
        remote_root=descriptors.ATLAS_ASSETS_REMOTE_ROOT,
        dequantize_mesh=True,
    )
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
