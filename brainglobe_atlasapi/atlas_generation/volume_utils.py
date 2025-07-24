"""
Code useful for dealing with volumetric data
(e.g. allen annotation volume for the mouse atlas)
extracting surfaces from volumetric data ....
"""

import numpy as np
import zarr


def create_masked_array(volume, label, greater_than=False) -> np.ndarray[bool]:
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
    if not isinstance(volume, (np.ndarray, zarr.Array)):
        raise ValueError(
            f"Argument volume should be an np.ndarray or a zarr.Array"
            f" object not {type(volume)}"
        )

    # if not isinstance(label, list) and not np.all(np.isin(label, volume)):
    #     print(f"Label {label} is not in the array, returning empty mask")
    #     return np.zeros_like(volume, dtype=bool)
    # # elif isinstance(label, list):
    # #     if not np.any(np.isin(volume, label)):
    # #         print(f"Label is not in the array, returning empty mask")
    # #         return arr

    if not greater_than:
        if not isinstance(label, list):
            mask = volume == label
        else:
            mask = np.isin(volume, label, kind="table")
    else:
        mask = volume > label

    return mask
