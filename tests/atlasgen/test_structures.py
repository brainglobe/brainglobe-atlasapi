import pytest

from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
    get_structure_children,
    get_structure_terminal_nodes,
)


@pytest.mark.parametrize(
    "structure_transform, expected_error_message",
    [
        pytest.param(
            lambda structures: structures[0].update(
                {"key_mismatch": structures[0].pop("name")}
            ),
            "Inconsistencies found for structure {'id': 101",
            id="key mismatch",
        ),
        pytest.param(
            lambda structures: structures[0].update(
                {"name": list(structures[0]["name"])}
            ),
            "Inconsistencies found for structure {'id': 101",
            id="value type mismatch",
        ),
    ],
)
def test_check_struct_consistency(
    structures, structure_transform, expected_error_message
):
    """Test structure consistency check with various modifications."""
    structure_transform(structures)
    with pytest.raises(AssertionError, match=expected_error_message):
        check_struct_consistency(structures)


@pytest.mark.parametrize(
    ["struct_index", "use_tree", "expected_children"],
    [
        pytest.param(0, False, [1, 5], id="o (101) parent"),
        pytest.param(2, False, None, id="aon (5) parent"),
        pytest.param(3, False, [101, 1, 5], id="root (999) parent"),
        pytest.param(0, True, [5, 1], id="o (101) parent (use_tree)"),
        pytest.param(2, True, None, id="aon (5) parent (use_tree)"),
        pytest.param(3, True, [101, 5, 1], id="root (999) parent (use_tree)"),
    ],
)
def test_get_structure_children(
    struct_index, use_tree, expected_children, structures
):
    """Verifies correct retrieval of children from a structures fixture tree.

    Tree:

    root (999)
    └── o (101)
      ├── aon (5)
      └── on (1)
    """
    region = structures[struct_index]
    children = get_structure_children(
        structures=structures, region=region, use_tree=use_tree
    )
    assert children == expected_children


@pytest.mark.xfail  # TODO: remove after get_structure_children bug fix
@pytest.mark.parametrize(
    ["use_tree"],
    [
        pytest.param(True, id="use_tree=True"),
        pytest.param(False, id="use_tree=False"),
    ],
)
def test_get_structure_children_no_children(structures, use_tree):
    """Test retrieval of children from structures tree without children."""
    region = structures[0]
    structures = [region, structures[3]]
    children = get_structure_children(
        structures=structures, region=region, use_tree=use_tree
    )
    assert children is None


@pytest.mark.parametrize(
    ["dropped_key", "error"],
    [
        pytest.param("id", ValueError),
        pytest.param("structure_id_path", ValueError),
        pytest.param("name", None),
    ],
)
def test_get_structure_children_incomplete_region(
    dropped_key, error, structures
):
    """Error should be raised when region data is incomplete.

    Error if 'id' or 'structure_id_path' is missing but not when other keys
    (e.g. 'name') are missing."""
    region = structures[0].copy()
    region.pop(dropped_key)
    if error is not None:
        with pytest.raises(
            ValueError, match="need both 'id' and 'structure_id_path'"
        ):
            get_structure_children(structures=structures, region=region)

    else:
        get_structure_children(structures=structures, region=region)


@pytest.mark.parametrize(
    [
        "structures_transform",
        "region_transform",
        "use_tree",
        "expected_error",
        "expected_message",
    ],
    [
        pytest.param(
            lambda structures: {s["id"]: s for s in structures},
            lambda structures: structures[0],
            False,
            ValueError,
            "structures should be a list",
            id="structures is dict",
        ),
        pytest.param(
            lambda structures: structures,
            lambda structures: [r_value for r_value in structures[0].values()],
            False,
            ValueError,
            "region should be a dictionary with a structures data",
            id="region is list",
        ),
        pytest.param(
            lambda structures: [
                [r_value for r_value in s.values()] for s in structures
            ],
            lambda structures: structures[0],
            False,
            ValueError,
            "structures should be a list of dictionaries",
            id="structures is list of lists",
        ),
    ],
)
def test_get_structure_children_errors(
    structures_transform,
    region_transform,
    expected_error,
    use_tree,
    expected_message,
    structures,
):
    """Test correct raising of ValueError for invalid inputs."""
    with pytest.raises(expected_error, match=expected_message):
        get_structure_children(
            structures=structures_transform(structures),
            region=region_transform(structures),
            use_tree=use_tree,
        )


@pytest.mark.parametrize(
    "structure_id_path, expected_terminal_nodes",
    [
        pytest.param(None, [5, 1], id="original"),
        pytest.param([999, 101, 1, 2], [5, 2], id="terminal node added to 2"),
        pytest.param([999, 101, 1], [5, 1, 2], id="terminal node added to 1"),
        pytest.param([999, 1, 5, 2], [2, 1], id="terminal node added to 5"),
        pytest.param([999, 5], [5, 1], id="terminal node added to root"),
    ],
)
def test_get_structure_terminal_nodes(
    structure_id_path, expected_terminal_nodes, structures
):
    """Test get_structure_terminal_nodes with various structures."""
    if structure_id_path is not None:
        structures += [
            {
                "id": 2,
                "acronym": "tn",
                "name": "terminal node",
                "rgb_triplet": [255, 255, 255],
                "structure_id_path": structure_id_path,
            },
        ]
    region = structures[0]
    terminal_nodes = get_structure_terminal_nodes(
        structures=structures, region=region
    )
    assert terminal_nodes == expected_terminal_nodes


def test_get_structure_terminal_nodes_without_leaves(capsys, structures):
    """Test region without any terminal nodes (leaves)."""
    region = structures[1]
    terminal_nodes = get_structure_terminal_nodes(
        structures=structures, region=region
    )
    captured = capsys.readouterr()
    assert "doesnt seem to contain any other regions" in captured.out
    assert terminal_nodes is None
