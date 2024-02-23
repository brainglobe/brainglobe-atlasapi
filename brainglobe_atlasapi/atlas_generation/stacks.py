import tifffile

from brainglobe_atlasapi import descriptors


def write_stack(stack, filename):
    """
    Parameters
    ----------
    stack
    filename

    """
    tifffile.imsave(str(filename), stack)


def save_reference(stack, output_dir):
    """
    Parameters
    ----------
    stack
    output_dir
    """
    if stack.dtype != descriptors.REFERENCE_DTYPE:
        stack = stack.astype(descriptors.REFERENCE_DTYPE)
    write_stack(stack, output_dir / descriptors.REFERENCE_FILENAME)


def save_secondary_reference(stack, name, output_dir):
    """
    Parameters
    ----------
    stack
    name
    output_dir
    """
    if stack.dtype != descriptors.REFERENCE_DTYPE:
        stack = stack.astype(descriptors.REFERENCE_DTYPE)
    write_stack(stack, output_dir / f"{name}.tiff")


def save_annotation(stack, output_dir):
    """
    Parameters
    ----------
    stack
    output_dir
    """
    if stack.dtype != descriptors.ANNOTATION_DTYPE:
        stack = stack.astype(descriptors.ANNOTATION_DTYPE)
    write_stack(stack, output_dir / descriptors.ANNOTATION_FILENAME)


def save_hemispheres(stack, output_dir):
    """
    Parameters
    ----------
    stack
    output_dir
    """
    if stack.dtype != descriptors.HEMISPHERES_DTYPE:
        stack = stack.astype(descriptors.HEMISPHERES_DTYPE)
    write_stack(stack, output_dir / descriptors.HEMISPHERES_FILENAME)
