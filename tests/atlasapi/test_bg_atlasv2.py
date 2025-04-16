from pathlib import Path

import numpy as np
import pytest

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.bg_atlasv2 import BrainGlobeAtlasV2

MOCK_V2_DIRECTORY = Path.home() / ".brainglobe-tests" / ".brainglobe_v2"


@pytest.fixture(scope="module")
def v1_atlas():
    return BrainGlobeAtlas("allen_mouse_100um")


@pytest.fixture(scope="module")
def v2_atlas():
    return BrainGlobeAtlasV2(
        "allen_mouse_100um",
        brainglobe_dir=MOCK_V2_DIRECTORY,
        interm_download_dir=MOCK_V2_DIRECTORY,
        check_latest=False,
    )


def test_bg_atlasv2_init(v1_atlas, v2_atlas):
    assert isinstance(v2_atlas, BrainGlobeAtlasV2)
    assert v2_atlas.atlas_name == v1_atlas.atlas_name
    assert v2_atlas.brainglobe_dir == MOCK_V2_DIRECTORY
    assert v2_atlas.interm_download_dir == MOCK_V2_DIRECTORY


def test_local_version(v1_atlas, v2_atlas):
    assert v2_atlas.local_version == v1_atlas.local_version


def test_remote_version(v1_atlas, v2_atlas):
    assert v2_atlas.remote_version == v1_atlas.remote_version


def test_local_full_name(v1_atlas, v2_atlas):
    assert v2_atlas.local_full_name == "allen_mouse_100um_v1.2.json"


@pytest.mark.xfail(reason="Remote version not implemented yet")
def test_remote_url(v1_atlas, v2_atlas):
    assert v2_atlas.remote_url == v1_atlas.remote_url


def test_check_latest_version(v1_atlas, v2_atlas):
    assert v2_atlas.check_latest_version() == v1_atlas.check_latest_version()


def test_repr(v1_atlas, v2_atlas):
    assert repr(v2_atlas) == repr(v1_atlas)


def test_str(v1_atlas, v2_atlas):
    assert str(v2_atlas) == str(v1_atlas)


def test_reference(v1_atlas, v2_atlas):
    assert np.array_equal(v2_atlas.reference, v1_atlas.reference)


def test_annotation(v1_atlas, v2_atlas):
    assert np.array_equal(v2_atlas.annotation, v1_atlas.annotation)


def test_atlas_name(v1_atlas, v2_atlas):
    assert v2_atlas.atlas_name == v1_atlas.atlas_name


def test_resolution(v1_atlas, v2_atlas):
    assert v2_atlas.resolution == v1_atlas.resolution


def test_orientation(v1_atlas, v2_atlas):
    assert v2_atlas.orientation == v1_atlas.orientation


def test_shape(v1_atlas, v2_atlas):
    assert v2_atlas.shape == v1_atlas.shape


def test_shape_um(v1_atlas, v2_atlas):
    assert v2_atlas.shape_um == v1_atlas.shape_um


def test_structures(v1_atlas, v2_atlas):
    assert len(v2_atlas.hierarchy) == len(v1_atlas.hierarchy)


def test_lookup_df(v1_atlas, v2_atlas):
    assert (v2_atlas.lookup_df == v1_atlas.lookup_df).to_numpy().all()


def test_hemispheres(v1_atlas, v2_atlas):
    assert np.array_equal(v2_atlas.hemispheres, v1_atlas.hemispheres)


@pytest.mark.parametrize("coords", [(0, 0, 0), (50, 50, 100)])
def test_hemisphere_from_coords(v1_atlas, v2_atlas, coords):
    assert v2_atlas.hemisphere_from_coords(
        coords
    ) == v1_atlas.hemisphere_from_coords(coords)
    assert v2_atlas.hemisphere_from_coords(
        coords, microns=True
    ) == v1_atlas.hemisphere_from_coords(coords, microns=True)
    assert v2_atlas.hemisphere_from_coords(
        coords, as_string=True
    ) == v1_atlas.hemisphere_from_coords(coords, as_string=True)


@pytest.mark.parametrize("coords", [(20, 20, 20), (50, 50, 80)])
def test_structure_from_coords(v1_atlas, v2_atlas, coords):
    assert v2_atlas.structure_from_coords(
        coords
    ) == v1_atlas.structure_from_coords(coords)
    assert v2_atlas.structure_from_coords(
        coords, microns=True
    ) == v1_atlas.structure_from_coords(coords, microns=True)
    assert v2_atlas.structure_from_coords(
        coords, as_acronym=True
    ) == v1_atlas.structure_from_coords(coords, as_acronym=True)


def test_mesh_from_structure(v1_atlas, v2_atlas):
    structure = "HY"
    v2_mesh = v2_atlas.mesh_from_structure(structure)
    v1_mesh = v1_atlas.mesh_from_structure(structure)

    assert len(v2_mesh.points) == len(v1_mesh.points)


@pytest.mark.xfail(reason="Meshes are stored in different locations")
def test_meshfile_from_structure(v1_atlas, v2_atlas):
    structure = "root"
    assert v2_atlas.meshfile_from_structure(
        structure
    ) == v1_atlas.meshfile_from_structure(structure)


def test_root_mesh(v1_atlas, v2_atlas):
    v2_mesh = v2_atlas.root_mesh()
    v1_mesh = v1_atlas.root_mesh()

    assert len(v2_mesh.points) == len(v1_mesh.points)


@pytest.mark.xfail(reason="Meshes are stored in different locations")
def test_root_meshfile(v1_atlas, v2_atlas):
    assert v2_atlas.root_meshfile() == v1_atlas.root_meshfile()


def test_get_structure_ancestors(v1_atlas, v2_atlas):
    structure = "HY"
    assert v2_atlas.get_structure_ancestors(
        structure
    ) == v1_atlas.get_structure_ancestors(structure)


def test_get_structure_descendants(v1_atlas, v2_atlas):
    structure = "HY"
    assert v2_atlas.get_structure_descendants(
        structure
    ) == v1_atlas.get_structure_descendants(structure)


def test_get_structure_mask(v1_atlas, v2_atlas):
    structure = "HY"
    assert np.array_equal(
        v2_atlas.get_structure_mask(structure),
        v1_atlas.get_structure_mask(structure),
    )
