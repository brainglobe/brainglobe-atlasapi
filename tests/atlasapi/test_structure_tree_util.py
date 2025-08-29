"""Test the structure tree utility functions."""

from brainglobe_atlasapi.structure_tree_util import (
    get_structures_tree,
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
