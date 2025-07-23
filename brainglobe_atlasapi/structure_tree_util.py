from collections import deque
from typing import Optional

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
    """
    Creates a 'tree' graph with the hierarchical organisation of all
    structures
    """

    def add_descendants_to_tree(
        structures_list, id_to_acronym_map, tree, structure_id, parent_id
    ):
        """
        Recursively goes through all the the descendants of a region and adds
        them to the tree
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


def postorder_tree(tree: Tree, node_id: Optional[int] = None):
    """
    Yields node identifiers in a post-order depth first traversal of the tree.
    """
    if node_id is None:
        node_id = tree.root

    def _postorder(node_id):
        for child in tree.children(node_id):
            yield from _postorder(child.identifier)
        yield node_id

    yield from _postorder(node_id)


def postorder_tree_iterative(tree):
    """
    Yields nodes in a post-order depth first traversal of the tree.
    """

    class StackFrame:
        def __init__(self, node, progress=0):
            self.node = node
            self.identifier = node.identifier
            self.progress = progress

    root_node = tree.nodes[tree.root]

    root_frame = StackFrame(root_node)
    stack = deque([root_frame])

    while len(stack) > 0:
        current_frame = stack.pop()

        if current_frame.progress == 0:
            # First time visiting this node, push it back with
            # progress incremented
            current_frame.progress += 1
            stack.append(current_frame)

            # Push all children onto the stack
            for child in tree.children(current_frame.identifier):
                stack.append(StackFrame(child))
        else:
            # Second time visiting this node, yield the node
            yield current_frame.node
