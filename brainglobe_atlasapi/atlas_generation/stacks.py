"""Functions for handling and processing image stacks related to
atlas generation.
"""

from pathlib import Path
from typing import Dict, List

import numpy.typing as npt
import tifffile
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale

from brainglobe_atlasapi import descriptors

BG_OME_ZARR_AXES = [
    {
        "name": "z",
        "type": "space",
        "unit": "millimeter",
        "orientation": {
            "type": "anatomical",
            "value": "anterior-to-posterior",
        },
    },
    {
        "name": "y",
        "type": "space",
        "unit": "millimeter",
        "orientation": {
            "type": "anatomical",
            "value": "superior-to-inferior",
        },
    },
    {
        "name": "x",
        "type": "space",
        "unit": "millimeter",
        "orientation": {
            "type": "anatomical",
            "value": "right-to-left",
        },
    },
]


def write_stack(stack, filename):
    """Write an image stack to a TIFF file.

    Parameters
    ----------
    stack : np.ndarray
        The image stack to be saved.
    filename : str or Path
        The path and filename where the stack will be saved.
    """
    tifffile.imwrite(str(filename), stack)


def write_multiscale_ome_zarr(
    images: List[npt.NDArray],
    output_path: Path,
    transformations: List[List[dict]],
    axes: List[dict] = BG_OME_ZARR_AXES,
):
    """Write a multiscale image pyramid to an OME-Zarr file.


    Parameters
    ----------
    images : list of np.ndarray
        A list of image stacks representing different levels of the pyramid.
    output_path : Path
        The path where the OME-Zarr file will be saved.
    transformations : list of lists of dicts
        A list of transformations to be saved alongside the images.
    axes : list of dicts, optional
        A list of axis descriptors to be included in the OME-Zarr metadata.
        If not provided, defaults to BG_OME_ZARR_AXES.
    """
    zarr_loc = parse_url(output_path, mode="w")
    assert zarr_loc is not None
    store = zarr_loc.store
    root = zarr.group(store=store)

    write_multiscale(
        pyramid=images,
        group=root,
        axes=axes,
        coordinate_transformations=transformations,
    )

    store.close()


def save_reference(stack, output_dir):
    """Save the reference image stack.

    Ensures the stack is of the correct data type before saving.

    Parameters
    ----------
    stack : np.ndarray
        The reference image stack.
    output_dir : Path
        The directory where the reference image will be saved.
    """
    if stack.dtype != descriptors.REFERENCE_DTYPE:
        stack = stack.astype(descriptors.REFERENCE_DTYPE)
    write_stack(stack, output_dir / descriptors.REFERENCE_FILENAME)


def _save_as_ome_zarr(
    stack: npt.NDArray,
    dtype: npt.DTypeLike,
    output_path: Path,
    transformations: List[List[Dict]],
) -> None:
    if stack.dtype != dtype:
        stack = stack.astype(dtype)
    assert (
        len(transformations) == 1
    ), "Currently only one resolution level is supported."
    write_multiscale_ome_zarr(
        images=[stack],
        output_path=output_path,
        transformations=transformations,
    )


def save_template(
    stack: npt.NDArray, output_dir: Path, transformations: List[List[Dict]]
):
    """Save the template image stack along with its transformations.

    Parameters
    ----------
    stack : np.ndarray
        The template image stack.
    output_dir : Path
        The directory where the template image will be saved.
    transformations : list of lists of dicts
        A list of transformations to be saved alongside the template.
    """
    _save_as_ome_zarr(
        stack,
        descriptors.REFERENCE_DTYPE,
        output_dir / descriptors.V2_TEMPLATE_NAME,
        transformations,
    )


def save_annotation(
    stack: npt.NDArray, output_dir: Path, transformations: List[List[Dict]]
):
    """Save the annotation image stack.

    Parameters
    ----------
    stack : np.ndarray
        The annotation image stack.
    output_dir : Path
        The directory where the annotation image will be saved.
    transformations : list of lists of dicts
        A list of transformations to be saved alongside the annotation.
    """
    _save_as_ome_zarr(
        stack,
        descriptors.ANNOTATION_DTYPE,
        output_dir / descriptors.V2_ANNOTATION_NAME,
        transformations,
    )


def save_hemispheres(
    stack: npt.NDArray, output_dir: Path, transformations: List[List[Dict]]
):
    """Save the hemispheres image stack.

    Parameters
    ----------
    stack : np.ndarray
        The hemispheres image stack.
    output_dir : Path
        The directory where the hemispheres image will be saved.
    transformations : list of lists of dicts
        A list of transformations to be saved alongside the hemispheres.
    """
    _save_as_ome_zarr(
        stack,
        descriptors.HEMISPHERES_DTYPE,
        output_dir / descriptors.V2_HEMISPHERES_NAME,
        transformations,
    )
