"""Convert structure JSON files to CSV format."""

from pathlib import Path

import pandas as pd


def structure_id_path_to_string(structure_id_path):
    """Convert a structure ID path list to a string.

    Given a path (as a list of structure IDs) to a specific structure,
    return it as a string of "/" separated structure IDs.

    Parameters
    ----------
    structure_id_path : list of int
        A list of integers defining the hierarchical path to a region,
        where the last element is the ID of the region itself.

    Returns
    -------
    str
        A string with structure IDs separated by "/".
        Example: "/997/8/567/"
    """
    path_string = "/"
    for element in structure_id_path:
        path_string = path_string + str(element) + "/"
    return path_string


def get_parent_id(structure_id_path, root=997):
    """Get the parent ID of a given structure.

    Given a path (as a list of structure IDs) to a specific structure,
    return the ID of its parent structure.

    Parameters
    ----------
    structure_id_path : list of int
        A list of integers defining the hierarchical path to a region,
        where the last element is the ID of the region itself.
    root : int, optional
        The ID value for the root (whole brain) structure, which has no parent.
        By default, 997.

    Returns
    -------
    int or None
        The ID of the parent structure, or None if the input structure is
        the root (has no parent).
    """
    if structure_id_path == [root]:
        return None
    else:
        return int(structure_id_path[-2])


def convert_structure_json_to_csv(
    structure_json_path, destination_path=None, root=997
):
    """Convert an atlas structure JSON file to CSV format.

    This function converts a JSON dictionary of atlas structures into a CSV
    format, which can be useful for compatibility with tools like cellfinder
    and for easier data browsing.

    Parameters
    ----------
    structure_json_path : str or Path
        Path to the input JSON file containing the structure data.
    destination_path : str or Path, optional
        Where to save the resulting CSV file. If None, the CSV file will be
        saved in the same directory as the JSON file with a `.csv` extension.
    root : int, optional
        The ID value for the root (whole brain) structure, used when
        determining parent IDs. By default, 997.
    """
    structure_json_path = Path(structure_json_path)

    df = pd.read_json(structure_json_path, encoding="utf-8")
    df = df.drop(columns=["rgb_triplet"])
    df["parent_structure_id"] = df["structure_id_path"].apply(
        get_parent_id, root=root
    )
    df["structure_id_path"] = df["structure_id_path"].apply(
        structure_id_path_to_string
    )
    df = df.sort_values("name")

    if destination_path is None:
        destination_path = structure_json_path.with_suffix(".csv")

    df.to_csv(destination_path, index=False, encoding="utf-8")
