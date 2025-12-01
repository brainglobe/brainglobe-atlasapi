#!/usr/bin/env bash

set -euo pipefail

# Fixed (target) image: SWC template aligned in ASR space
FIXED="template_sharpen_shapeupdate_orig-asr_aligned.nii.gz"

# Moving image: WHS SD T2* template in ASR space (cleaned outside-brain voxels)
MOVING="WHS_SD_T2star_clean_ASR.nii.gz"

if [[ ! -f "${FIXED}" ]]; then
  echo "ERROR: Fixed image '${FIXED}' not found." >&2
  exit 1
fi

if [[ ! -f "${MOVING}" ]]; then
  echo "ERROR: Moving image '${MOVING}' not found." >&2
  exit 1
fi

echo "Running antsRegistration..."
echo "  Fixed : ${FIXED}"
echo "  Moving: ${MOVING}"

antsRegistration \
  --dimensionality 3 --float 1 --verbose 1 \
  --output [W2T_,W2T_warped.nii.gz] \
  --winsorize-image-intensities [0.01,0.99] \
  --use-histogram-matching 1 \
  --initial-moving-transform ["${FIXED}","${MOVING}",0] \
  \
  --transform Rigid[0.2] \
  --metric MI["${FIXED}","${MOVING}",1,32,Regular,0.3] \
  --convergence [400x200x100,1e-6,10] \
  --shrink-factors 6x4x2 \
  --smoothing-sigmas 3x2x1vox \
  \
  --transform Affine[0.2] \
  --metric MI["${FIXED}","${MOVING}",1,32,Regular,0.3] \
  --convergence [400x200x100,1e-6,10] \
  --shrink-factors 6x4x2 \
  --smoothing-sigmas 3x2x1vox \
  \
  --transform SyN[0.1,2,0] \
  --metric CC["${FIXED}","${MOVING}",1,4] \
  --convergence [60x40x20,1e-6,10] \
  --shrink-factors 6x4x2 \
  --smoothing-sigmas 3x2x1vox

echo "antsRegistration finished."
