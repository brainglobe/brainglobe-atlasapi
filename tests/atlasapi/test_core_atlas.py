import contextlib
from io import StringIO

import numpy as np
import pandas as pd
import pytest
import tifffile

from bg_atlasapi.core import AdditionalRefDict


def test_initialization(atlas):
    assert atlas.metadata == {
        "name": "example_mouse",
        "citation": (
            "Wang et al 2020, " "https://doi.org/10.1016/j.cell.2020.04.007"
        ),
        "atlas_link": "http://www.brain-map.org",
        "species": "Mus musculus",
        "symmetric": True,
        "resolution": [100.0, 100.0, 100.0],
        "orientation": "asr",
        "version": atlas.metadata["version"],  # no target value for version
        "shape": [132, 80, 114],
        "trasform_to_bg": [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "additional_references": [],
    }

    assert atlas.orientation == "asr"
    assert atlas.shape == (132, 80, 114)
    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.shape_um == (13200.0, 8000.0, 11400.0)


def test_additional_ref_dict(temp_path):
    fake_data = dict()
    for k in ["1", "2"]:
        stack = np.ones((10, 20, 30)) * int(k)
        fake_data[k] = stack
        tifffile.imwrite(temp_path / f"{k}.tiff", stack)

    add_ref_dict = AdditionalRefDict(fake_data.keys(), temp_path)

    for k, stack in add_ref_dict.items():
        assert add_ref_dict[k] == stack

    with pytest.warns(UserWarning, match="No reference named 3"):
        assert add_ref_dict["3"] is None


@pytest.mark.parametrize(
    "stack_name, val",
    [
        ("reference", [[[146, 155], [153, 157]], [[148, 150], [153, 153]]]),
        ("annotation", [[[59, 362], [59, 362]], [[59, 362], [59, 362]]]),
        ("hemispheres", [[[2, 1], [2, 1]], [[2, 1], [2, 1]]]),
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
    res = atlas.resolution
    assert atlas.structure_from_coords(coords) == 997
    assert atlas.structure_from_coords(coords, as_acronym=True) == "root"
    assert (
        atlas.structure_from_coords(
            [c * r for c, r in zip(coords, res)], microns=True, as_acronym=True
        )
        == "root"
    )
    assert atlas.hemisphere_from_coords(coords) == atlas.right_hemisphere_value
    assert atlas.hemisphere_from_coords(coords, as_string=True) == "right"
    assert (
        atlas.hemisphere_from_coords(
            [c * r for c, r in zip(coords, res)], microns=True, as_string=True
        )
        == "right"
    )


def test_meshfile_from_id(atlas):
    assert (
        atlas.meshfile_from_structure("CH")
        == atlas.root_dir / "meshes/567.obj"
    )
    assert atlas.root_meshfile() == atlas.root_dir / "meshes/997.obj"


def test_mesh_from_id(atlas):
    mesh = atlas.structures[567]["mesh"]
    assert np.allclose(mesh.points[0], [8019.52, 3444.48, 507.104])

    mesh = atlas.mesh_from_structure(567)
    assert np.allclose(mesh.points[0], [8019.52, 3444.48, 507.104])

    mesh = atlas.root_mesh()
    assert np.allclose(mesh.points[0], [7896.56, 3384.15, 503.781])


def test_lookup_df(atlas):
    df_lookup = atlas.lookup_df
    df = pd.DataFrame(
        dict(
            acronym=["root", "grey", "CH"],
            id=[997, 8, 567],
            name=["root", "Basic cell groups and regions", "Cerebrum"],
        )
    )

    assert all(df_lookup == df)


def test_hierarchy(atlas):
    hier = atlas.hierarchy
    temp_stdout = StringIO()
    with contextlib.redirect_stdout(temp_stdout):
        print(hier)
    output = temp_stdout.getvalue().strip()
    assert output == "root (997)\n└── grey (8)\n    └── CH (567)"

    assert {k: v.tag for k, v in hier.nodes.items()} == {
        997: "root (997)",
        8: "grey (8)",
        567: "CH (567)",
    }


def test_descendants(atlas):
    anc = atlas.get_structure_ancestors("CH")
    assert anc == ["root", "grey"]

    desc = atlas.get_structure_descendants("root")
    assert desc == ["grey", "CH"]
