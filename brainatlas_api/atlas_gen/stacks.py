import tifffile
import numpy as np
from brainatlas_api.atlas_gen import descriptors


def write_stack(stack, filename):
    """
    Parameters
    ----------
    stack
    filename

    """
    tifffile.imsave(str(filename), stack)


def save_anatomy(stack, output_dir):
    """
    Parameters
    ----------
    stack
    output_dir
    """
    if stack.dtype != np.uint16:
        stack = stack.astype(np.uint16)
    write_stack(stack, output_dir / descriptors.REFERENCE_FILENAME)


def save_annotation(stack, output_dir):
    """
    Parameters
    ----------
    stack
    output_dir
    """
    if stack.dtype != np.int32:
        stack = stack.astype(np.int32)
    write_stack(stack, output_dir / descriptors.ANNOTATION_FILENAME)
