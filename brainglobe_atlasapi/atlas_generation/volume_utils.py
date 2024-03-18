"""
    Code useful for dealing with volumetric data
    (e.g. allen annotation volume for the mouse atlas)
    extracting surfaces from volumetric data ....
"""

import numpy as np


def create_masked_array(volume, label, greater_than=False):
    """
    Given a 2d o 3d numpy array and a
    label value, creates a masked binary
    array which is 1 when volume == label
    and 0 otherwise

    Parameters
    ----------
    volume: np.ndarray
        (2d or 3d array)
    label: int, float or list of int.
        the masked array will be 1 where volume == label
    greater_than: bool
        if True, all voxels with value > label will be set to 1
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
