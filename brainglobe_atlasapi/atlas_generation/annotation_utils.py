"""Helper functions to read annotation metadata from common formats."""

from pathlib import Path

import numpy as np
from scipy.ndimage import generic_filter


def split_label_text(name: str, acronym_length=1) -> str:
    """Split label text into name + acronym.

    If the label text ends with ')', extract the acronym inside parentheses.
    Otherwise, you can specify  how many first letters
    you would like to use as acronym
    """
    if name.endswith(")"):
        name, acronym = name.split("(")
        name = name[:-1]  # ignore trailing space
        acronym = acronym[:-1]  # ignore trailing )
    else:
        if acronym_length > len(name):
            raise ValueError(
                "Acronym length cannot be longer than the name itself."
            )
        else:
            acronym = name[:acronym_length]
    return name, acronym


def read_itk_labels(path: Path, acronym_length=1) -> dict:
    """Turn ITK label data from a file into a list of dictionaries."""
    labels = []
    with open(path) as labels_file:
        for line in labels_file:
            if not line.startswith("#"):
                raw_values = line.split(maxsplit=7)
                id = int(raw_values[0])
                rgb = list((int(r) for r in raw_values[1:4]))
                if raw_values[7][-1] == "\n":
                    raw_values[7] = raw_values[7][:-1]
                label_text = raw_values[7][1:-1]
                if label_text != "Clear Label":
                    name, acronym = split_label_text(
                        label_text, acronym_length
                    )
                    labels.append(
                        {
                            "id": id,
                            "name": name,
                            "rgb_triplet": rgb,
                            "acronym": acronym,
                        }
                    )
    return labels


ITK_SNAP_HEADER = """################################################
# ITK-SnAP Label Description File
# File format:
# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL
# Fields:
#    IDX:   Zero-based index
#    -R-:   Red color component (0..255)
#    -G-:   Green color component (0..255)
#    -B-:   Blue color component (0..255)
#    -A-:   Label transparency (0.00 .. 1.00)
#    VIS:   Label visibility (0 or 1)
#    IDX:   Label mesh visibility (0 or 1)
#  LABEL:   Label description
################################################
"""

ITK_CLEAR_LABEL = '0 0 0 0 0 0 0 "Clear Label"\n'


def write_itk_labels(path: Path, labels):
    """Write ITK label data to a file."""
    with open(path, "w") as labels_file:
        labels_file.write(ITK_SNAP_HEADER)
        labels_file.write(ITK_CLEAR_LABEL)
        for label in labels:
            labels_file.write(
                f"{label['id']} "
                f"{label['rgb_triplet'][0]} "
                f"{label['rgb_triplet'][1]} "
                f"{label['rgb_triplet'][2]} 1.00 1 1 "
                f'"{label["name"] + " (" + label["acronym"] + ")"}"\n'
            )


def modal_filter_ignore_zeros(window):
    """
    Compute the mode of the window, ignoring zero values.

    Parameters
    ----------
    window : numpy.ndarray
        The input window of values.

    Returns
    -------
    int or float
        The most common non-zero value in the window, or 0 if all values
        are zero.
    """
    # Remove zeros from the window
    non_zero_values = window[window != 0]
    if len(non_zero_values) == 0:
        return 0  # If all values are zero, return 0
    # Compute the mode (most common value)
    values, counts = np.unique(non_zero_values, return_counts=True)
    return values[np.argmax(counts)]


def apply_modal_filter(image, filter_size=3):
    """Apply a modal filter to the image, ignoring zero neighbors.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a 2D NumPy array.
    filter_size : int
        Size of the filtering window (must be odd).

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    # Apply the modal filter using a sliding window
    filtered_image = generic_filter(
        image, function=modal_filter_ignore_zeros, size=filter_size
    )
    return filtered_image
