from treelib import Tree

# TODO evaluate whether we want this as a method in StructureDict


def child_ids(structure, structure_list):
    return [
        s["id"]
        for s in structure_list
        if len(s["structure_id_path"]) > 1
        and s["structure_id_path"][-2] == structure
    ]


def get_structures_tree(structures_list):
    """Creates a 'tree' graph with the hierarchical organisation of all structures
    """

    def add_descendants_to_tree(
        structures_list, id_to_acronym_map, tree, structure_id, parent_id
    ):
        """
            Recursively goes through all the the descendants of a region and adds them to the tree
        """
        tree.create_node(
            tag=id_to_acronym_map[structure_id],
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
    tree.create_node(tag="root", identifier=root)

    # Recursively iterate through hierarchy#
    for child in child_ids(root, structures_list):
        add_descendants_to_tree(
            structures_list, id_to_acronym_map, tree, child, root
        )

    return tree
