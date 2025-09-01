"""Functions for handling and processing image stacks related to
atlas generation.
"""

import tifffile

from brainglobe_atlasapi import descriptors


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


def save_secondary_reference(stack, name, output_dir):
    """Save a secondary reference image stack with a given name.

    Ensures the stack is of the correct data type before saving.

    Parameters
    ----------
    stack : np.ndarray
        The secondary reference image stack.
    name : str
        The base name for the output file (e.g., "my_secondary_reference").
    output_dir : Path
        The directory where the secondary reference image will be saved.
    """
    if stack.dtype != descriptors.REFERENCE_DTYPE:
        stack = stack.astype(descriptors.REFERENCE_DTYPE)
    write_stack(stack, output_dir / f"{name}.tiff")


def save_annotation(stack, output_dir):
    """Save the annotation image stack.

    Ensures the stack is of the correct data type before saving.

    Parameters
    ----------
    stack : np.ndarray
        The annotation image stack.
    output_dir : Path
        The directory where the annotation image will be saved.
    """
    if stack.dtype != descriptors.ANNOTATION_DTYPE:
        stack = stack.astype(descriptors.ANNOTATION_DTYPE)
    write_stack(stack, output_dir / descriptors.ANNOTATION_FILENAME)


def save_hemispheres(stack, output_dir):
    """Save the hemispheres image stack.

    Ensures the stack is of the correct data type before saving.

    Parameters
    ----------
    stack : np.ndarray
        The hemispheres image stack.
    output_dir : Path
        The directory where the hemispheres image will be saved.
    """
    if stack.dtype != descriptors.HEMISPHERES_DTYPE:
        stack = stack.astype(descriptors.HEMISPHERES_DTYPE)
    write_stack(stack, output_dir / descriptors.HEMISPHERES_FILENAME)
