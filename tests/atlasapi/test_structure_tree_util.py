"""Test the structure tree utility functions."""

import pytest

from brainglobe_atlasapi.structure_tree_util import (
    get_structures_tree,
    postorder_depth_first_search,
    preorder_depth_first_search,
)

STRUCTURES_LIST = [
    {
        "acronym": "root",
        "id": 997,
        "name": "root",
        "structure_id_path": [997],
        "rgb_triplet": [255, 255, 255],
        "mesh_filename": None,
    },
    {
        "acronym": "grey",
        "id": 8,
        "name": "Basic cell groups and regions",
        "structure_id_path": [997, 8],
        "rgb_triplet": [191, 218, 227],
        "mesh_filename": None,
    },
    {
        "acronym": "CH",
        "id": 567,
        "name": "Cerebrum",
        "structure_id_path": [997, 8, 567],
        "rgb_triplet": [176, 240, 255],
        "mesh_filename": None,
    },
    {
        "acronym": "BS",
        "id": 343,
        "name": "Brain stem",
        "structure_id_path": [997, 8, 343],
        "rgb_triplet": [255, 112, 128],
    },
    {
        "acronym": "IB",
        "id": 1129,
        "name": "Interbrain",
        "structure_id_path": [997, 8, 343, 1129],
        "rgb_triplet": [255, 112, 128],
    },
    {
        "acronym": "VS",
        "id": 73,
        "name": "ventricular systems",
        "structure_id_path": [997, 73],
        "rgb_triplet": [170, 170, 170],
    },
]


def mock_tree():
    """Create a tree structure from a list of structures."""
    return get_structures_tree(STRUCTURES_LIST)


def test_preorder_dfs():
    """Test the preorder depth-first search traversal of the tree."""
    tree = mock_tree()

    preorder = [node.identifier for node in preorder_depth_first_search(tree)]
    expected_preorder = [
        997,  # root
        8,  # grey
        567,  # CH
        343,  # BS
        1129,  # IB
        73,  # VS
    ]

    assert preorder == expected_preorder


def test_postorder_dfs():
    """Post-order visits every child before its parent."""
    tree = mock_tree()
    postorder = [
        node.identifier for node in postorder_depth_first_search(tree)
    ]
    # Leaves first: CH (567), IB (1129), BS (343), grey (8), VS (73),
    # root (997)
    expected = [567, 1129, 343, 8, 73, 997]
    assert postorder == expected


def test_postorder_dfs_parent_always_after_children():
    """For every node, all descendants appear before it in post-order."""
    tree = mock_tree()
    order = [node.identifier for node in postorder_depth_first_search(tree)]
    for node_id in order:
        idx = order.index(node_id)
        for child in tree.children(node_id):
            assert order.index(child.identifier) < idx


def test_root_found_without_a_root_acronym():
    """The root is the shared head of the paths, whatever it is called.

    Only one of the atlas-assets terminologies calls its root "root";
    the others use "brain", "mouse" or "NS".
    """
    renamed = [dict(s) for s in STRUCTURES_LIST]
    renamed[0]["acronym"] = "NS"
    renamed[0]["name"] = "nervous system"

    tree = get_structures_tree(renamed)

    assert tree.root == 997
    assert set(tree.nodes) == {s["id"] for s in renamed}


def test_extra_top_level_structures_are_outvoted():
    """A stray parentless entry does not become the root or join the tree.

    Some terminologies carry labels such as "unassigned" alongside the
    real root, each its own one-element path.
    """
    with_orphan = [dict(s) for s in STRUCTURES_LIST] + [
        {
            "acronym": "unassigned",
            "id": 0,
            "name": "unassigned",
            "structure_id_path": [0],
            "rgb_triplet": [0, 0, 0],
        }
    ]

    tree = get_structures_tree(with_orphan)

    assert tree.root == 997
    assert 0 not in tree.nodes


def test_no_paths_raises():
    """A list with no usable paths reports that, rather than KeyError."""
    pathless = [{"acronym": "x", "id": 1, "structure_id_path": []}]

    with pytest.raises(ValueError, match="root structure"):
        get_structures_tree(pathless)
