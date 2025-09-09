"""A one-off script to run a modal filter on some
annotations from ITK snap inbetween manual
improvement iterations.
"""

from pathlib import Path

import numpy as np
from brainglobe_template_builder.io import (
    save_as_asr_nii,
)
from brainglobe_utils.IO.image import load_nii
from skimage.filters.rank import modal
from skimage.morphology import square

resources_path = Path("D:/UCL/Postgraduate_programme/templates/Version3")
annotation_path = (
    resources_path / "pouch_peripodial_hinge_notum_refined.nii.gz"
)
filtered_path = (
    resources_path / "pouch_peripodial_hinge_notum_refined_filtered.nii.gz"
)
target_isotropic_resolution = 2
if __name__ == "__main__":
    # Load the annotation image
    annotation_image = load_nii(annotation_path, as_array=True)
    annotation_image = annotation_image.astype(np.uint16)

    filtered_annotation_image = np.zeros_like(annotation_image)
    footprint = square(5)  # Define a square footprint for the modal filter
    for i in range(annotation_image.shape[0]):
        # Apply the modal filter to each slice of the annotation image
        filtered_stack = modal(annotation_image[i], footprint=footprint)
        print(annotation_image[i].max())
        filtered_annotation_image[i] = filtered_stack
        print(filtered_stack.max())

    # Save the filtered annotation image
    vox_sizes = [
        target_isotropic_resolution,
    ] * 3
    save_as_asr_nii(filtered_annotation_image, vox_sizes, filtered_path)
