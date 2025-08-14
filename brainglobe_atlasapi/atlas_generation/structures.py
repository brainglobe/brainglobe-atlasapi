"""Handle structure information for atlas generation."""

from brainglobe_atlasapi.descriptors import STRUCTURE_TEMPLATE as STEMPLATE
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def check_struct_consistency(structures):
    """Ensure internal consistency of the structures list.

    Checks that each structure dictionary in the list has all the required
    keys and that their values are of the correct types, as defined by
    `STRUCTURE_TEMPLATE`.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If any structure dictionary is missing required keys, or if
        the types of values do not match the expected template.
    """
    assert isinstance(structures, list)
    assert isinstance(structures[0], dict)

    # Check that all structures have the correct keys and value types:
    for struct in structures:
        try:
            assert struct.keys() == STEMPLATE.keys()
            assert all(
                isinstance(v, type(STEMPLATE[k])) for k, v in struct.items()
            )
        except AssertionError:
            raise AssertionError(
                f"Inconsistencies found for structure {struct}"
            )


def get_structure_children(structures, region, use_tree=False):
    """Get the direct and indirect children of a given brain region.

    Given a list of dictionaries with structures data and a specific
    region from that list, this function returns the IDs of all structures
    that are children (direct or indirect) of the given `region`.
    It can optionally use a `StructureTree` for more efficient traversal.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties.
    region : dict
        A dictionary representing the parent brain region, which must contain
        'id' and 'structure_id_path' keys.
    use_tree : bool, optional
        If True, a `StructureTree` will be constructed and used to find
        children. If False, a simpler list comprehension will be used.
        By default, False.

    Returns
    -------
    list of int or None
        A list of integer IDs for all child structures, or None if the
        region contains no other regions.

    Raises
    ------
    ValueError
        If `structures` is not a list of dictionaries, or if `region` is not
        a dictionary, or if `region` is missing 'id' or 'structure_id_path'.
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
            "Incomplete structures dicts, "
            "need both 'id' and 'structure_id_path'"
        )

    if not use_tree:
        sub_region_ids = []
        for subregion in structures:
            if region["id"] in subregion["structure_id_path"]:
                if subregion["id"] is not region["id"]:
                    sub_region_ids.append(subregion["id"])
    else:
        tree = get_structures_tree(structures)
        sub_region_ids = [
            n.identifier
            for k, n in tree.subtree(region["id"]).nodes.items()
            if n.identifier is not region["id"]
        ]

    if sub_region_ids == []:
        print(f"{region['acronym']} doesnt seem to contain any other regions")
        return None
    else:
        return sub_region_ids


def get_structure_terminal_nodes(structures, region):
    """Get the terminal (leaf) child nodes of a given brain region.

    Given a list of dictionaries with structures data and a specific
    region from that list, this function returns the IDs of all structures
    that are children of the given `region` and are also leaf nodes in the
    structure tree (i.e., they have no further children).

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties.
    region : dict
        A dictionary representing the parent brain region, which must contain
        an 'id' key.

    Returns
    -------
    list of int or None
        A list of integer IDs for all terminal child structures, or None if
        the region contains no terminal child regions.
    """
    tree = get_structures_tree(structures)

    sub_region_ids = [
        n.identifier
        for n in tree.subtree(region["id"]).leaves()
        if n.identifier is not region["id"]
    ]

    if not sub_region_ids:
        print(f"{region['acronym']} doesnt seem to contain any other regions")
        return None
    else:
        return sub_region_ids
