import pytest

import numpy as np

from brainatlas_api.bg_atlas import TestAtlas


@pytest.fixture()
def atlas():
    return TestAtlas()


def test_initialization(atlas):
    assert atlas.metadata == {
        "name": "test",
        "citation": "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007",
        "atlas_link": "www.brain-map.org.com",
        "symmetric": True,
        "resolution": [100, 100, 100],
        "shape": [132, 80, 114],
    }


@pytest.mark.parametrize(
    "attribute, val",
    [
        ("shape", [132, 80, 114]),
        ("resolution", [100, 100, 100]),
        ("name", "test"),
        ("symmetric", True),
    ],
)
def test_attributes(atlas, attribute, val):
    assert getattr(atlas, attribute) == val


@pytest.mark.parametrize(
    "stack_name, val",
    [
        ("reference", [[[146, 155], [153, 157]], [[148, 150], [153, 153]]]),
        ("annotated", [[[59, 362], [59, 362]], [[59, 362], [59, 362]]]),
        ("hemispheres", [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]),
    ],
)
def test_stacks(atlas, stack_name, val):
    loaded_stack = getattr(atlas, stack_name)
    assert np.allclose(loaded_stack[65:67, 39:41, 57:59], val)


def test_maps(atlas):
    assert atlas.acronym_to_id_map == {"root": 997, "grey": 8, "CH": 567}

    assert atlas.id_to_acronym_map == {997: "root", 8: "grey", 567: "CH"}


@pytest.mark.parametrize(
    "coords", [[39.0, 36.0, 57.0], (39, 36, 57), np.array([39.0, 36.0, 57.0])]
)
def test_data_from_coords(atlas, coords):
    assert atlas.get_region_id_from_coords(coords) == 997
    assert atlas.get_region_name_from_coords(coords) == "root"
    assert atlas.get_hemisphere_from_coords(coords) == 0


def test_meshfile_from_id(atlas):
    assert (
        atlas.get_mesh_file_from_acronym("CH")
        == atlas.root_dir / "meshes/567.obj"
    )


def test_mesh_from_id(atlas):
    # TODO will change depeding on mesh loading package
    vert, vnorms, faces, fnorms = atlas.get_mesh_from_id(567)
    assert np.allclose(vert[0], [8019.52, 3444.48, 507.104])
    assert np.allclose(faces[0], [0, 1, 2])
