"""
Preprocess WHS Sprague Dawley rat template and annotation volumes.

Steps (for reference):
- Download template:
  https://www.nitrc.org/frs/download.php/12263/MBAT_WHS_SD_rat_atlas_v4_pack.zip

- Download annotation:
  https://www.nitrc.org/frs/download.php/13400/MBAT_WHS_SD_rat_atlas_v4.01.zip//?i_agree=1&download_now=1

- Unzip both archives and locate:
  - WHS_SD_rat_T2star_v1.01.nii.gz
  - WHS_SD_rat_atlas_v4.01.nii.gz
  - WHS_SD_rat_atlas_v4.01_labels.ilf

This script:
- Reorients template and annotation volumes to ASR orientation.
- Applies the same transform to both volumes.
- Zeros template voxels outside the annotated brain.
- Saves:
  - WHS_SD_T2star_clean_ASR.nii.gz
  - WHS_SD_annotation_ASR.nii.gz
"""

import nibabel as nib
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
)

# Load images
tmpl_img = nib.load("WHS_SD_rat_T2star_v1.01.nii.gz")
ann_img = nib.load("WHS_SD_rat_atlas_v4.01.nii.gz")

tmpl = tmpl_img.get_fdata()
ann = ann_img.get_fdata()

# Desired orientation: Anterior–Superior–Right
target_orient = axcodes2ornt(("A", "S", "R"))

# Determine current orientation from affine
current_orient = io_orientation(tmpl_img.affine)

# Build orientation transform from current to target
transform_orient = nib.orientations.ornt_transform(
    current_orient, target_orient
)

# Apply orientation transform to data
tmpl_asr = apply_orientation(tmpl, transform_orient)
ann_asr = apply_orientation(ann, transform_orient)

# Compute new affine in target orientation
new_affine = tmpl_img.affine @ nib.orientations.inv_ornt_aff(
    transform_orient, tmpl.shape
)

# Zero template voxels outside the annotation mask
tmpl_clean_asr = tmpl_asr.copy()
tmpl_clean_asr[ann_asr == 0] = 0

# Save outputs
tmpl_clean_img = nib.Nifti1Image(tmpl_clean_asr, affine=new_affine)
nib.save(tmpl_clean_img, "WHS_SD_T2star_clean_ASR.nii.gz")

ann_asr_img = nib.Nifti1Image(ann_asr, affine=new_affine)
nib.save(ann_asr_img, "WHS_SD_annotation_ASR.nii.gz")
