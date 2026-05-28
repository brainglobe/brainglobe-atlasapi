"""Test the core Atlas class."""

import contextlib
from io import StringIO

import ngff_zarr as nz
import numpy as np
import pandas as pd
import pytest

from brainglobe_atlasapi import core
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.core import AdditionalRefDict


def test_initialization(atlas):
    """Test Atlas class initialization."""
    assert atlas.metadata["name"] == "example_mouse"
    assert (
        atlas.metadata["citation"]
        == "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"
    )
    assert atlas.metadata["atlas_link"] == "http://www.brain-map.org"
    assert atlas.metadata["species"] == "Mus musculus"
    assert atlas.metadata["symmetric"] is True
    assert atlas.metadata["resolution"] == [100.0, 100.0, 100.0]
    assert atlas.metadata["orientation"] == "asr"
    assert atlas.metadata["shape"] == [132, 80, 114]
    assert atlas.metadata["additional_references"] == []

    assert atlas.orientation == "asr"
    assert atlas.shape == (132, 80, 114)
    assert atlas.resolution == (100.0, 100.0, 100.0)
    assert atlas.shape_um == (13200.0, 8000.0, 11400.0)


def test_additional_ref_dict(atlas):
    """Test AdditionalRefDict class functionality."""
    fake_data = [
        {
            "name": "allen-adult-mouse-stpt-template",
            "version": "2015",
            "location": "/templates/allen-adult-mouse-stpt-template/2015",
        }
    ]

    data_path = get_brainglobe_dir() / "brainglobe-atlasapi"
    add_ref_dict = AdditionalRefDict(
        fake_data, data_path, resolution=(100.0, 100.0, 100.0)
    )

    assert list(add_ref_dict) == ["allen-adult-mouse-stpt-template"]
    assert np.all(
        add_ref_dict["allen-adult-mouse-stpt-template"] == atlas.template
    )

    with pytest.warns(UserWarning, match="No reference named 3"):
        assert add_ref_dict["3"] is None


def test_addition_ref_dict_keys_only(temp_path):
    """Initialize AdditionalRefDict with keys only.

    Parameters
    ----------
    temp_path : Path
        Temporary path for test files.
    """
    fake_data = [
        {
            "name": "allen-adult-mouse-stpt-template",
            "version": "2015",
            "location": "/templates/allen-adult-mouse-stpt-template/2015",
        },
        {
            "name": "another-template",
            "version": "2020",
            "location": "/templates/another-template/2020",
        },
    ]
    add_ref_dict = AdditionalRefDict(
        fake_data, temp_path, resolution=(100.0, 100.0, 100.0)
    )

    assert list(add_ref_dict) == [
        "allen-adult-mouse-stpt-template",
        "another-template",
    ]


@pytest.mark.parametrize(
    "stack_name, val",
    [
        ("template", [[[155, 146], [157, 153]], [[151, 148], [154, 153]]]),
        ("annotation", [[[59, 59], [59, 59]], [[59, 59], [59, 59]]]),
        ("hemispheres", [[[2, 1], [2, 1]], [[2, 1], [2, 1]]]),
    ],
)
def test_stacks(atlas, stack_name, val):
    """Test loading and accessing different atlas stacks.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    stack_name : str
        The name of the stack to test (e.g., "reference", "annotation").
    val : np.ndarray
        Expected values for a specific region within the stack.
    """
    loaded_stack = getattr(atlas, stack_name)
    assert np.allclose(loaded_stack[65:67, 39:41, 56:58], val)


def test_structures(atlas):
    """Test the structures dictionary and retrieval methods.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
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
    """Test retrieving structure and hemisphere information from coordinates.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    coords : list, tuple, or np.ndarray
        Coordinates to query.
    """
    res = atlas.resolution
    assert atlas.structure_from_coords(coords) == 997
    assert atlas.structure_from_coords(coords, as_acronym=True) == "root"
    assert (
        atlas.structure_from_coords(
            [c * r for c, r in zip(coords, res)], microns=True, as_acronym=True
        )
        == "root"
    )
    assert atlas.hemisphere_from_coords(coords) == atlas.left_hemisphere_value
    assert atlas.hemisphere_from_coords(coords, as_string=True) == "left"
    assert (
        atlas.hemisphere_from_coords(
            [c * r for c, r in zip(coords, res)], microns=True, as_string=True
        )
        == "left"
    )


def test_data_from_coords_out_of_brain(
    atlas,
):
    """Test querying coordinates outside the atlas boundaries.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    coords = (1, 1, 1)
    key_error_string = "Outside atlas"

    assert atlas.structure_from_coords(coords) == 0
    assert atlas.structure_from_coords(coords, microns=True) == 0

    assert (
        atlas.structure_from_coords(coords, as_acronym=True)
        == key_error_string
    )
    assert (
        atlas.structure_from_coords(coords, microns=True, as_acronym=True)
        == key_error_string
    )


def test_meshfile_from_id(atlas):
    """Test retrieving mesh file paths from structure IDs.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    mesh_root_path = (
        atlas.root_dir
        / atlas.metadata["annotation_set"]["location"][1:]
        / "annotation.precomputed"
    )
    assert atlas.meshfile_from_structure("CH") == mesh_root_path / "567"
    assert atlas.root_meshfile() == mesh_root_path / "997"


def test_mesh_from_id(atlas):
    """Test loading mesh objects from structure IDs.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    mesh = atlas.structures[567]["mesh"]
    assert np.allclose(mesh.points[0], [8019.52, 3444.48, 507.104])

    mesh = atlas.mesh_from_structure(567)
    assert np.allclose(mesh.points[0], [8019.52, 3444.48, 507.104])

    mesh = atlas.root_mesh()
    assert np.allclose(mesh.points[0], [7896.56, 3384.15, 503.781])


def test_lookup_df(atlas):
    """Test the lookup DataFrame property.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
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
    """Test the hierarchy property and its string representation.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
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
    """Test retrieving ancestors and descendants of a structure.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    anc = atlas.get_structure_ancestors("CH")
    assert anc == ["root", "grey"]

    desc = atlas.get_structure_descendants("root")
    assert desc == ["grey", "CH"]


def test_odd_hemisphere_size(atlas):
    """Test hemisphere stack handling with an odd Z-axis size.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    atlas.metadata["shape"] = [132, 80, 115]
    assert atlas.hemispheres.shape == (132, 80, 115)
    assert (atlas.hemispheres[:, :, 57] == 2).all()
    assert (atlas.hemispheres[:, :, 58] == 1).all()


def test_even_hemisphere_size(atlas):
    """Test hemisphere stack handling with an even Z-axis size.

    Parameters
    ----------
    atlas : brainglobe_atlasapi.core.Atlas
        The atlas fixture.
    """
    assert atlas.hemispheres.shape == (132, 80, 114)
    assert (atlas.hemispheres[:, :, 56] == 2).all()
    assert (atlas.hemispheres[:, :, 57] == 1).all()


def test_get_structure_mask_raises_without_4d_array(atlas):
    """get_structure_mask raises FileNotFoundError when annotations.ome.zarr
    is absent (the example_mouse atlas pre-dates the 4D array feature).
    """
    with pytest.raises(FileNotFoundError, match="4D mask array"):
        atlas.get_structure_mask("grey")


@pytest.mark.parametrize(
    "atlas_fixture",
    [
        pytest.param("asymmetric_atlas", id="asymmetric"),
        pytest.param("atlas", id="symmetric"),
    ],
)
def test_hemispheres_reads_tiff(atlas_fixture, request, mocker):
    """Read TIFF for asymmetric atlas hemispheres.

    Parameters
    ----------
    atlas_fixture : str
        Name of the atlas fixture to use ("asymmetric_atlas" or "atlas").
    request : pytest.FixtureRequest
        Pytest request fixture to get fixture values.
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture.
    """
    atlas = request.getfixturevalue(atlas_fixture)
    mock_hemispheres = np.zeros(atlas.metadata["shape"], dtype=np.uint8)
    mock_hemispheres_multiscale = nz.to_multiscales(
        mock_hemispheres, scale_factors=1
    )
    mock_hemispheres_multiscale.metadata.datasets[0].path = "0"
    mocker.patch(
        "brainglobe_atlasapi.core.nz.from_ngff_zarr",
        return_value=mock_hemispheres_multiscale,
    )
    _ = atlas.hemispheres
    if atlas.metadata["symmetric"]:
        core.nz.from_ngff_zarr.assert_not_called()
    elif atlas.metadata["symmetric"] is False:
        core.nz.from_ngff_zarr.assert_called_once()


def test_get_structures_at_hierarchy_level_as_acronym(atlas):
    """Test basic usage returning acronyms."""
    result = atlas.get_structures_at_hierarchy_level(
        "root", 1, as_acronym=True
    )
    assert result == ["grey"]


def test_get_structures_at_hierarchy_level_as_id(atlas):
    """Test basic usage returning IDs."""
    result = atlas.get_structures_at_hierarchy_level("root", 1)
    assert result == [8]


def test_get_structures_at_hierarchy_level_numeric_input(atlas):
    """Test that numeric structure IDs work."""
    result = atlas.get_structures_at_hierarchy_level(997, 1)
    assert result == [8]


def test_get_structures_at_hierarchy_level_zero(atlas):
    """Test that level 0 returns root."""
    result = atlas.get_structures_at_hierarchy_level(
        "root", 0, as_acronym=True
    )
    assert result == ["root"]


def test_get_structures_at_hierarchy_level_multiple_levels(atlas):
    """Test querying at different hierarchy levels."""
    result = atlas.get_structures_at_hierarchy_level(
        "root", 2, as_acronym=True
    )
    assert result == ["CH"]


def test_get_structures_at_hierarchy_level_none(atlas):
    """Test that hierarchy_level=None returns all structures in BFS order."""
    result = atlas.get_structures_at_hierarchy_level(
        "root", None, as_acronym=True
    )
    assert result == ["root", "grey", "CH"]


def test_get_structures_at_hierarchy_level_invalid_structure(atlas):
    """Test error handling for invalid structure."""
    with pytest.raises(KeyError, match=r"not found"):
        atlas.get_structures_at_hierarchy_level("INVALID", 1)


def test_get_structures_at_hierarchy_level_negative_level(atlas):
    """Test error handling for negative hierarchy level."""
    with pytest.raises(ValueError, match=r"must be non-negative"):
        atlas.get_structures_at_hierarchy_level("root", -1)


def test_get_structures_at_hierarchy_level_wrong_type(atlas):
    """Test error handling for wrong hierarchy_level type."""
    with pytest.raises(ValueError, match=r"must be an int or None"):
        atlas.get_structures_at_hierarchy_level("root", "2")


def test_get_structures_at_hierarchy_level_too_deep(atlas):
    """Test error handling when hierarchy level exceeds structure depth."""
    with pytest.raises(ValueError, match=r"no descendants at hierarchy level"):
        atlas.get_structures_at_hierarchy_level("root", 10)


def test_get_structures_at_hierarchy_level_leaf_node(atlas):
    """Test that querying a leaf node with hierarchy_level=0 returns root."""
    # CH is the deepest node in the test atlas
    result = atlas.get_structures_at_hierarchy_level("CH", 0)
    assert result == [997]  # root ID


@pytest.fixture(scope="module")
def atlas_with_masks(tmp_path_factory):
    """Create a minimal 3-structure atlas with annotations.ome.zarr via wrapup.

    Structure tree: root (999) → region_a (1) → leaf_b (2)
    Post-order mapping: leaf_b→0, region_a→1, root→2

    Annotation (5x5x5):
      - [0,0,0] = 1  (region_a)
      - [0,0,1] = 2  (leaf_b)
      - rest   = 999 (root)
    """
    import meshio

    working_dir = tmp_path_factory.mktemp("atlas_masks")
    shape = (15, 15, 15)
    reference = np.full(shape, 200, dtype=np.uint16)
    annotation = np.full(shape, 999, dtype=np.uint32)
    annotation[0, 0, 0] = 1
    annotation[0, 0, 1] = 2

    structures_list = [
        {
            "id": 999,
            "acronym": "root",
            "name": "root",
            "rgb_triplet": [255, 255, 255],
            "structure_id_path": [999],
        },
        {
            "id": 1,
            "acronym": "region_a",
            "name": "Region A",
            "rgb_triplet": [100, 150, 200],
            "structure_id_path": [999, 1],
        },
        {
            "id": 2,
            "acronym": "leaf_b",
            "name": "Leaf B",
            "rgb_triplet": [200, 100, 50],
            "structure_id_path": [999, 1, 2],
        },
    ]

    mesh_path = working_dir / "root.obj"
    points = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
    )
    cells = [
        ("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    ]
    meshio.write(str(mesh_path), meshio.Mesh(points=points, cells=cells))
    meshes = {999: mesh_path, 1: mesh_path, 2: mesh_path}

    wrapup_atlas_from_data(
        atlas_name="mask_test",
        atlas_minor_version="0",
        citation="unpublished",
        atlas_link="https://example.com",
        species="Mus musculus",
        resolution=(25, 25, 25),
        orientation="asr",
        root_id=999,
        reference_stack=reference,
        annotation_stack=annotation,
        structures_list=structures_list,
        meshes_dict=meshes,
        working_dir=working_dir,
    )

    manifests = list(
        (working_dir / "brainglobe-atlasapi" / "atlases").glob(
            "**/manifest.json"
        )
    )
    assert len(manifests) == 1
    return core.Atlas(manifests[0])


def test_get_structure_mask_returns_binary_uint8(atlas_with_masks):
    """get_structure_mask returns a uint8 array with values 0 or 1."""
    mask = atlas_with_masks.get_structure_mask("leaf_b")
    assert mask.dtype == np.uint8
    assert set(np.unique(mask)).issubset({0, 1})


def test_get_structure_mask_leaf_correct_voxels(atlas_with_masks):
    """Leaf mask has exactly the voxels assigned to that structure."""
    mask = atlas_with_masks.get_structure_mask("leaf_b")
    assert mask[0, 0, 1] == 1
    assert mask.sum() == 1


def test_get_structure_mask_parent_includes_children(atlas_with_masks):
    """Parent mask includes its own voxels plus all descendant voxels."""
    mask = atlas_with_masks.get_structure_mask("region_a")
    assert mask[0, 0, 0] == 1
    assert mask[0, 0, 1] == 1
    assert mask.sum() == 2


def test_get_structure_mask_raises_for_unknown_structure(atlas_with_masks):
    """get_structure_mask raises KeyError for an unknown structure."""
    with pytest.raises(KeyError):
        atlas_with_masks.get_structure_mask("nonexistent_region")
