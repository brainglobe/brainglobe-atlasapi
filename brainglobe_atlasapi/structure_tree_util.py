"""Utility functions for working with structure trees."""

from treelib import Tree


def child_ids(structure, structure_list):
    """
    Return a list of IDs of the children of a given structure.

    Parameters
    ----------
    structure : dict
        The structure to find the children of.
    structure_list : list
        A list of structures to search within.

    Returns
    -------
    list
        A list of child IDs.
    """
    return [
        s["id"]
        for s in structure_list
        if len(s["structure_id_path"]) > 1
        and s["structure_id_path"][-2] == structure
    ]


def get_structures_tree(structures_list):
    """
    Create a `tree` graph with the hierarchical organisation of all
    structures.
    """

    def add_descendants_to_tree(
        structures_list, id_to_acronym_map, tree, structure_id, parent_id
    ):
        """
        Recursively goes through all the the descendants of a region and adds
        them to the tree.
        """
        tree.create_node(
            tag=f"{id_to_acronym_map[structure_id]} ({structure_id})",
            identifier=structure_id,
            parent=parent_id,
        )
        descendants = child_ids(structure_id, structures_list)

        if len(descendants):
            for child in descendants:
                add_descendants_to_tree(
                    structures_list,
                    id_to_acronym_map,
                    tree,
                    child,
                    structure_id,
                )

    # Create a Tree structure and initialise with root
    id_to_acronym_map = {s["id"]: s["acronym"] for s in structures_list}
    acronym_to_id_map = {v: k for k, v in id_to_acronym_map.items()}

    root = acronym_to_id_map["root"]
    tree = Tree()
    tree.create_node(tag=f"root ({root})", identifier=root)

    # Recursively iterate through hierarchy#
    for child in child_ids(root, structures_list):
        add_descendants_to_tree(
            structures_list, id_to_acronym_map, tree, child, root
        )

    return tree
