"""Helper functions to read annotation metadata from common formats,
or manipulate annotations."""

from pathlib import Path

import numpy as np
from scipy.ndimage import generic_filter


def split_label_text(name: str) -> str:
    if name.endswith(")"):
        name, acronym = name.split("(")
        name = name[:-1]  # ignore trailing space
        acronym = acronym[:-1]  # ignore trailing )
    else:
        acronym = name[0]
    return name, acronym


def read_itk_labels(path: Path) -> dict:
    labels = []
    with open(path) as labels_file:
        for line in labels_file:
            if not line.startswith("#"):
                raw_values = line.split(maxsplit=7)
                id = int(raw_values[0])
                rgb = tuple((int(r) for r in raw_values[1:4]))
                if raw_values[7][-1] == "\n":
                    raw_values[7] = raw_values[7][:-1]
                label_text = raw_values[7][1:-1]
                if label_text != "Clear Label":
                    name, acronym = split_label_text(label_text)
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
    with open(path, "w") as labels_file:
        labels_file.write(ITK_SNAP_HEADER)
        labels_file.write(ITK_CLEAR_LABEL)
        for label in labels:
            labels_file.write(
                f"{label['id']} "
                f"{label['rgb_triplet'][0]} "
                f"{label['rgb_triplet'][1]} "
                f"{label['rgb_triplet'][2]} 1.00 1 1 "
                f"\"{label['name'] + ' (' + label['acronym']+')'}\"\n"
            )


def modal_filter_ignore_zeros(window):
    """
    Compute the mode of the window ignoring zero values.
    """
    # Remove zeros from the window
    non_zero_values = window[window != 0]
    if len(non_zero_values) == 0:
        return 0  # If all values are zero, return 0
    # Compute the mode (most common value)
    values, counts = np.unique(non_zero_values, return_counts=True)
    return values[np.argmax(counts)]


def apply_modal_filter(image, filter_size=3):
    """
    Apply a modal filter to the image, ignoring zero neighbors.

    Parameters:
        image (ndarray): Input image as a 2D NumPy array.
        filter_size (int): Size of the filtering window (must be odd).

    Returns:
        ndarray: Filtered image.
    """
    # Apply the modal filter using a sliding window
    filtered_image = generic_filter(
        image, function=modal_filter_ignore_zeros, size=filter_size
    )
    return filtered_image


# TODO turn into test
if __name__ == "__main__":
    path = Path.home() / "Downloads" / "corrected_LabelMainBrainAreas_SW.txt"
    labels = read_itk_labels(path)
    [print(label) for label in labels]
    write_itk_labels(
        Path.home() / "Downloads" / "test-writing.txt", labels=labels
    )
