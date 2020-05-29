from brainatlas_api.atlas_gen.descriptors import (
    STRUCTURE_TEMPLATE as STEMPLATE,
)
from brainatlas_api.structures.structure_tree import StructureTree


def check_struct_consistency(structures):
    """Ensures internal consistency of the structures list
    Parameters
    ----------
    structures

    Returns
    -------

    """
    assert type(structures) == list
    assert type(structures[0]) == dict

    # Check that all structures have the correct keys and value types:
    for struct in structures:
        try:
            assert struct.keys() == STEMPLATE.keys()
            assert [
                isinstance(v, type(STEMPLATE[k])) for k, v in struct.items()
            ]
        except AssertionError:
            raise AssertionError(
                f"Inconsistencies found for structure {struct}"
            )


def get_structure_children(structures, region, use_tree=False):
    """
        Given a list of dictionaries with structures data, 
        and a structure from the list, this function returns
        the structures in the list that are children of 
        the given structure (region).
        If use_tree is true it creates a StructureTree and uses that. 
    """
    if not isinstance(structures, list):
        raise ValueError("structures should be a list")
    if not isinstance(structures[0], dict):
        raise ValueError("structures should be a list of dictionaries")
    if not isinstance(region, dict):
        raise ValueError(
            "region should be a dictionary with a structures data"
        )

    if "id" not in region.keys() or "structure_id_path" not in region.keys():
        raise ValueError(
            'Incomplete structures dicts, need both "id" and "structure_id_path"'
        )

    if not use_tree:
        sub_region_ids = []
        for subregion in structures:
            if region["id"] in subregion["structure_id_path"]:
                sub_region_ids.append(subregion["id"])
    else:
        tree = StructureTree(structures).get_structures_tree()
        sub_region_ids = [l.identifier for l in tree.children(region["id"])]

    if sub_region_ids == []:
        print(f'{region["acronym"]} doesnt seem to contain any other regions')
        return None
    else:
        return sub_region_ids


def get_structure_terminal_nodes(structures, region):
    """
        Given a list of dictionaries with structures data, 
        and a structure from the list, this function returns
        the structures in the list that are children of 
        the given structure (region) that are leafs of the 
        struture tree
    """

    tree = StructureTree(structures).get_structures_tree()

    sub_region_ids = [
        l.identifier for l in tree.subtree(region["id"]).leaves()
    ]

    if not sub_region_ids:
        print(f'{region["acronym"]} doesnt seem to contain any other regions')
        return None
    else:
        return sub_region_ids
