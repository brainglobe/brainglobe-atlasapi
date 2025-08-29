"""
Code useful for dealing with volumetric data
(e.g. allen annotation volume for the mouse atlas)
extracting surfaces from volumetric data ....
"""

import numpy as np
import zarr


def create_masked_array(volume, label, greater_than=False) -> np.ndarray[bool]:
    """
    Given a 2d or 3d numpy array and a
    label value, creates a masked binary
    array which is True when volume == label
    and False otherwise

    Parameters
    ----------
    volume: np.ndarray
        (2d or 3d array)
    label: int, float or list of int.
        the masked array will be 1 where volume == label
    greater_than: bool
        if True, all voxels with value > label will be set to True
    """
    if not isinstance(volume, (np.ndarray, zarr.Array)):
        raise ValueError(
            f"Argument volume should be an np.ndarray or a zarr.Array"
            f" object not {type(volume)}"
        )

    if not greater_than:
        if not isinstance(label, list):
            mask = volume == label
        else:
            mask = np.isin(volume, label)
    else:
        mask = volume > label

    return mask
