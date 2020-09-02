from pathlib import Path
import pandas as pd


def structure_id_path_to_string(structure_id_path):
    """
    Given a path (as a list of structure ids) to a specific structure,
    return as a string of "/" separated structure ids
    Parameters
    ----------
    structure_id_path : list
        list of ints defining the path to a region (which is the last element)

    Returns
    -------
    str:
        "/" separated string of structure ids

    """

    path_string = "/"
    for element in structure_id_path:
        path_string = path_string + str(element) + "/"
    return path_string


def get_parent_id(structure_id_path, root=997):
    """
    Given a path (as a list of structure ids) to a specific structure,
    return the id of the parent structure

    Parameters
    ----------
    structure_id_path : list
        list of ints defining the path to a region (which is the last element)

    root : int (optional)
        Value for the root (whole brain) structure that has no parent.

    Returns
    -------
    int or None :
        id of the parent structure (or None if no parent)
    """

    if structure_id_path == [root]:
        return None
    else:
        return int(structure_id_path[-2])


def convert_structure_json_to_csv(
    structure_json_path, destination_path=None, root=997
):
    """
    Converts an atlas structure json dictionary to csv. For cellfinder
    compatibility and ease of browsing.

    Parameters
    ----------
    structure_json_path : str or Path object
        path to the json file
    destination_path : str or Path object (optional)
        Where to save the resulting csv file. Defaults to the same directory
        as the json file.
    """

    structure_json_path = Path(structure_json_path)

    df = pd.read_json(structure_json_path)
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

    df.to_csv(destination_path, index=False)
