"""
Code useful for dealing with volumetric data
(e.g. allen annotation volume for the mouse atlas)
extracting surfaces from volumetric data ....
"""

import numpy as np
import zarr
from numba import njit, prange
from numba.typed import Dict


def create_masked_array(volume, label, greater_than=False):
    """Create a binary masked array from a volumetric dataset.

    Given a 2D or 3D NumPy array and a label value (or list of labels),
    this function generates a binary array. The output array will have
    values of True where the `volume` matches the `label` (or is contained
    within the `label` list) and False otherwise. If `greater_than` is True,
    all voxels with values strictly greater than `label` will be set to True.

    Parameters
    ----------
    volume : np.ndarray
        The input 2D or 3D NumPy array.
    label : int, float, or list of int
        The value(s) to match in the `volume`. If `greater_than` is True,
        this should be a single numerical value.
    greater_than : bool, optional
        If True, all voxels with values strictly greater than `label`
        will be set to True. If False, voxels equal to `label` (or in the
        list of `label`s) will be set to True. By default, False.

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


@njit(parallel=True, cache=True)
def create_masked_array_numba(
    flat_vol: np.ndarray, lut: np.ndarray, out: np.ndarray, mapping: Dict
) -> None:
    """
    Create a binary masked array from a flattened volumetric dataset.

    Given a flattened 1D NumPy array representing a volumetric dataset,
    a lookup table (LUT), and a mapping dictionary, this function generates
    a binary array. The output array will have values of 1 where the `flat_vol`
    matches the LUT values based on the provided mapping, and 0 otherwise.

    Parameters
    ----------
    flat_vol : np.ndarray
        The input 1D NumPy array representing the flattened volumetric dataset.
    lut : np.ndarray
        The lookup table containing values to match against the `flat_vol`.
    out : np.ndarray
        The output 1D NumPy array where the binary mask will be stored.
    mapping : Dict
        A Numba typed dictionary mapping `flat_vol` values to indices in LUT.
    """
    n = lut.shape[0]
    for i in prange(flat_vol.shape[0]):
        v = flat_vol[i]
        if v in mapping:
            mapped_v = mapping[v]
            # Excludes any mapped values that are out of bounds of the LUT
            out[i] = mapped_v < n and lut[mapped_v]
        else:
            # Catch any values that are not in the mapping and set them to 0
            # Mostly for background values (0)
            out[i] = 0
