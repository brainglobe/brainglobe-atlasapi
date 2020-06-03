import pytest

import numpy as np

from brainatlas_api.bg_atlas import ExampleAtlas


@pytest.fixture()
def atlas():
    return ExampleAtlas()


def test_initialization(atlas):
    assert atlas.metadata == {
        "name": "example_mouse",
        "citation": "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007",
        "atlas_link": "http://www.brain-map.org.com",
        "symmetric": True,
        "resolution": [100, 100, 100],
        "species": "Mus musculus",
        "version": "0.2",
        "shape": [132, 80, 114],
        "trasform_to_bg": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
    }


@pytest.mark.parametrize(
    "stack_name, val",
    [
        ("reference", [[[146, 155], [153, 157]], [[148, 150], [153, 153]]]),
        ("annotation", [[[59, 362], [59, 362]], [[59, 362], [59, 362]]]),
        ("hemispheres", [[[0, 0], [0, 0]], [[1, 1], [1, 1]]]),
    ],
)
def test_stacks(atlas, stack_name, val):
    loaded_stack = getattr(atlas, stack_name)
    assert np.allclose(loaded_stack[65:67, 39:41, 57:59], val)


def test_structures(atlas):
    assert {s["acronym"]: k for k, s in atlas.structures.items()} == {
        "root": 997,
        "grey": 8,
        "CH": 567,
    }
    assert atlas._get_from_structure([997, 8, 567], "acronym") == [
        "root",
        "grey",
        "CH",
    ]


@pytest.mark.parametrize(
    "coords", [[39.0, 36.0, 57.0], (39, 36, 57), np.array([39.0, 36.0, 57.0])]
)
def test_data_from_coords(atlas, coords):
    assert atlas.structure_from_coords(coords) == 997
    assert atlas.structure_from_coords(coords, as_acronym=True) == "root"
    assert atlas.hemisphere_from_coords(coords) == 0


def test_meshfile_from_id(atlas):
    assert (
        atlas.meshfile_from_structure("CH")
        == atlas.root_dir / "meshes/567.obj"
    )


def test_mesh_from_id(atlas):
    # TODO will change depending on mesh loading package
    mesh = atlas.structures[567]["mesh"]
    assert np.allclose(mesh.points[0], [8019.52, 3444.48, 507.104])
    assert np.allclose(mesh.cells[0].data[0], [0, 1, 2])
