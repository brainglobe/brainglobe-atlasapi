import tifffile
from brainatlas_api import descriptors


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
    if stack.dtype != descriptors.REFERENCE_DTYPE:
        stack = stack.astype(descriptors.REFERENCE_DTYPE)
    write_stack(stack, output_dir / descriptors.REFERENCE_FILENAME)


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
