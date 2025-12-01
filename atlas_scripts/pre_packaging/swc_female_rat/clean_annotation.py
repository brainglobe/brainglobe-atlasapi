"""
Clean WHS annotation after mapping it into SWC template space.

Expected inputs (in the working directory):
- template_sharpen_shapeupdate_orig-asr_aligned.nii.gz
- WHS_SD_annotation_template_space.nii.gz

Outputs:
- WHS_SD_annotation_template_space_cleaned.nii.gz
- clean_mask.nii.gz  (for inspection/debugging)
"""

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    binary_opening,
    label,
)

# ------------------------------------------------------------------
# 1. LOAD TEMPLATE + ANNOTATION (NIfTI)
# ------------------------------------------------------------------
tmpl_img = nib.load("template_sharpen_shapeupdate_orig-asr_aligned.nii.gz")
ann_img = nib.load("WHS_SD_annotation_template_space.nii.gz")

tmpl = tmpl_img.get_fdata()
ann = ann_img.get_fdata()

affine = tmpl_img.affine  # keep affine for saving

print("Loaded volumes:", tmpl.shape, ann.shape)

# ------------------------------------------------------------------
# 2. CREATE INITIAL MASK (adjust threshold if needed)
# ------------------------------------------------------------------
# Threshold tuned for this template; adjust if necessary.
mask = tmpl > 0.40

# ------------------------------------------------------------------
# 3. MORPHOLOGICAL CLEANUP
# ------------------------------------------------------------------
mask = binary_closing(mask, iterations=2)
mask = binary_opening(mask, iterations=2)
mask = binary_fill_holes(mask)

# ------------------------------------------------------------------
# 4. KEEP ONLY LARGEST CONNECTED COMPONENT
# ------------------------------------------------------------------
labels, num = label(mask)
print(f"Connected components found: {num}")

if num == 0:
    raise RuntimeError("No connected components found in mask.")

sizes = [(labels == i).sum() for i in range(1, num + 1)]
largest = int(np.argmax(sizes)) + 1
clean_mask = labels == largest

print(f"Keeping component {largest} (size = {sizes[largest - 1]})")

# ------------------------------------------------------------------
# 5. APPLY MASK TO THE ANNOTATION
# ------------------------------------------------------------------
ann_clean = ann.copy()
ann_clean[~clean_mask] = 0

# ------------------------------------------------------------------
# 6. REMOVE MANUALLY UNWANTED REGION IDS
# ------------------------------------------------------------------
to_remove = [41, 42, 45, 504, 119, 120, 121, 85, 162]

for rid in to_remove:
    ann_clean[ann_clean == rid] = 0

print("Removed region IDs:", to_remove)

# ------------------------------------------------------------------
# 7. SAVE CLEANED ANNOTATION
# ------------------------------------------------------------------
clean_img = nib.Nifti1Image(
    ann_clean.astype(np.int32),
    affine,
)
nib.save(clean_img, "WHS_SD_annotation_template_space_cleaned.nii.gz")

print(
    "Saved cleaned annotation -> "
    "WHS_SD_annotation_template_space_cleaned.nii.gz"
)

# ------------------------------------------------------------------
# 8. (OPTIONAL) SAVE CLEANED MASK FOR DEBUGGING
# ------------------------------------------------------------------
mask_img = nib.Nifti1Image(
    clean_mask.astype(np.uint8),
    affine,
)
nib.save(mask_img, "clean_mask.nii.gz")

print("Saved mask -> clean_mask.nii.gz")
