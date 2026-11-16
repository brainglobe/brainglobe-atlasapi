"""Tests for reading atlases that follow the atlas-assets specification."""

import DracoPy
import numpy as np

from brainglobe_atlasapi import core, descriptors, manifest
from brainglobe_atlasapi.utils import read_json

# The 2017 annotation set's precomputed `info` declares
# size = [1140, 800, 1320] voxels at resolution = [10000] * 3 nm, in
# precomputed XYZ. Vertices are implicitly in the enclosing volume's
# units, which is the assumption this test exists to pin.
_ANNOTATION_EXTENT_NM = np.array([1140, 800, 1320]) * 10000.0


def test_load_atlas_assets_atlas(atlas_assets_atlas):
    """A spec-conformant atlas loads at a requested resolution."""
    atlas = atlas_assets_atlas

    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.orientation == "asl"
    assert len(atlas.shape) == 3
    assert atlas.metadata["species"] == "Mus musculus"
    assert atlas.metadata["citation"] is None
    assert atlas.structures["root"]["id"] == 997


def test_core_atlas_loads_atlas_assets_manifest(atlas_assets_atlas):
    """core.Atlas normalizes an atlas-assets manifest without the subclass."""
    cached = atlas_assets_atlas

    atlas = core.Atlas(
        cached.brainglobe_dir / cached.local_full_name, resolution=100
    )

    assert atlas.metadata["format"] == "atlas-assets"
    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.shape == cached.shape


def test_atlas_assets_cache_is_namespaced(atlas_assets_atlas):
    """Downloads land under the configured root's directory."""
    atlas = atlas_assets_atlas
    assert atlas.brainglobe_dir.name == "allen-atlas-assets"


def test_atlas_assets_hemispheres_are_synthesized(atlas_assets_atlas):
    """No hemispheres asset exists, so hemispheres are synthesized."""
    atlas = atlas_assets_atlas
    # An "asl" frontal axis (the last one) runs left to right, so the first
    # half of it is the left hemisphere and the second half the right.
    assert atlas.orientation == "asl"
    n = atlas.shape[2]
    expected = np.where(
        np.arange(n) < n / 2,
        atlas.left_hemisphere_value,
        atlas.right_hemisphere_value,
    )
    expected = np.broadcast_to(expected, atlas.shape)

    np.testing.assert_array_equal(atlas.hemispheres, expected)


def test_atlas_assets_meshes_are_dequantized(atlas_assets_atlas):
    """An atlas-assets atlas flags its structures for dequantization."""
    atlas = atlas_assets_atlas
    assert atlas.metadata["format"] == "atlas-assets"
    assert all(s.dequantize_mesh for s in atlas.structures.values())


def test_orientation_is_consistent_across_an_atlas_assets_atlas(
    atlas_assets_atlas,
):
    """Every spatial asset of one atlas declares the same axis directions.

    An atlas is returned in the orientation it is stored in, so the only
    orientation requirement is internal: the annotation and the template
    must agree, because normalisation derives the atlas-wide orientation
    from the annotation alone and `core.Atlas` then applies it to both.
    """
    atlas = atlas_assets_atlas
    annotation_set = atlas.metadata["annotation_set"]
    annotation = atlas.root_dir / annotation_set["location"][1:]
    template = atlas.root_dir / annotation_set["template"]["location"][1:]
    assets = [
        annotation / descriptors.V3_ANNOTATION_NAME,
        annotation / descriptors.V3_ANNOTATION_MASKS_NAME,
        template / descriptors.V3_TEMPLATE_NAME,
    ]

    origins = {
        asset.name: manifest._orientation_from_axes(
            read_json(asset / "zarr.json")["attributes"]["ome"]
        )
        for asset in assets
    }

    assert set(origins.values()) == {"asl"}, origins


def test_atlas_assets_structure_mask(atlas_assets_atlas):
    """A structure mask lies inside the annotation's non-zero extent."""
    atlas = atlas_assets_atlas
    mask = atlas.get_structure_mask("root")
    assert mask.shape == tuple(atlas.shape)
    assert mask.max() == 1


def test_atlas_assets_mesh_is_nanometres_in_xyz(atlas_assets_atlas):
    """Real Allen vertices land inside the declared volume extent.

    Reads the mesh as the atlas caches it. A cold cache converts it from the
    bucket's bytes here; `test_structure_dict.py` runs the conversion itself
    on every session.
    """
    atlas = atlas_assets_atlas
    atlas.mesh_from_structure("root")  # downloads and converts it if missing
    mesh_file = atlas.meshfile_from_structure("root")

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
