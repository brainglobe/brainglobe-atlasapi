"""
    Code useful for dealing with volumetric data (e.g. allen annotation volume for the mouse atlas)
    extracting surfaces from volumetric data ....
"""
try:
    from vtkplotter import Volume
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Mesh generation with these utils requires vtkplotter\n"
        + '   please install with "pip install vtkplotter -U"'
    )

from brainio import brainio

import os
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


# ----------------------------- vtkplotter utils ----------------------------- #
# This stuff is outdated, use the functions in mesh_utils.py
# to extract meshes from volumes


def load_labelled_volume(data, vmin=0, alpha=1, **kwargs):
    """
        Load volume image from .nrrd file. 
        It assume that voxels with value = 0 are empty while voxels with values > 0
        are labelles (e.g. to indicate the location of a brain region in a reference atlas)

        :param data: str, path to file with volume data or 3d numpy array
        :param vmin: float, values below this numner will be assigned an alpha=0 and not be visualized
        :param **kwargs: kwargs to pass to the Volume class from vtkplotter
        :param alpha: float in range [0, 1], transparency [for the part of volume with value > vmin]
    """
    # Load/check volumetric data
    if isinstance(data, str):  # load from file
        if not os.path.isfile(data):
            raise FileNotFoundError(f"Volume data file {data} not found")

        try:
            data = brainio.load_any(data)
        except Exception as e:
            raise ValueError(
                f"Could not load volume data from file: {data} - {e}"
            )

    elif not isinstance(data, np.ndarray):
        raise ValueError(
            f"Data should be a filepath or np array, not: {data.__type__}"
        )

    # Create volume and set transparency range
    vol = Volume(data, alpha=alpha, **kwargs)

    otf = vol.GetProperty().GetScalarOpacity()
    otf.RemoveAllPoints()
    otf.AddPoint(vmin, 0)  # set to transparent
    otf.AddPoint(vmin + 0.1, alpha)  # set to opaque
    otf.AddPoint(data.max(), alpha)

    return vol
