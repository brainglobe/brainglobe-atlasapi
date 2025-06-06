"""
Code useful for dealing with volumetric data
(e.g. allen annotation volume for the mouse atlas)
extracting surfaces from volumetric data ....
"""

import numpy as np


def create_masked_array(volume, label, greater_than=False):
    """Create a binary masked array from a volumetric dataset.

    Given a 2D or 3D NumPy array and a label value (or list of labels),
    this function generates a binary array. The output array will have
    values of 1 where the `volume` matches the `label` (or is contained
    within the `label` list) and 0 otherwise. If `greater_than` is True,
    all voxels with values strictly greater than `label` will be set to 1.

    Parameters
    ----------
    volume : np.ndarray
        The input 2D or 3D NumPy array.
    label : int, float, or list of int
        The value(s) to match in the `volume`. If `greater_than` is True,
        this should be a single numerical value.
    greater_than : bool, optional
        If True, all voxels with values strictly greater than `label`
        will be set to 1. If False, voxels equal to `label` (or in the
        list of `label`s) will be set to 1. By default, False.

    Returns
    -------
    np.ndarray
        A binary NumPy array with the same shape as `volume`, where
        matching (or greater than) voxels are 1 and others are 0.

    Raises
    ------
    ValueError
        If `volume` is not a NumPy array.
    """
    if not isinstance(volume, np.ndarray):
        raise ValueError(
            f"Argument volume should be a numpy array not {type(volume)}"
        )

    arr = np.zeros_like(volume)

    if not isinstance(label, list) and not np.all(np.isin(label, volume)):
        print(f"Label {label} is not in the array, returning empty mask")
        return arr
    # elif isinstance(label, list):
    #     if not np.any(np.isin(volume, label)):
    #         print(f"Label is not in the array, returning empty mask")
    #         return arr

    if not greater_than:
        if not isinstance(label, list):
            arr[volume == label] = 1
        else:
            arr[np.isin(volume, label)] = 1
    else:
        arr[volume > label] = 1
    return arr
