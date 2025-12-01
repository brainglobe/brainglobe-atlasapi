#!/usr/bin/env bash

set -euo pipefail

INPUT_ANN="WHS_SD_annotation_ASR.nii.gz"
REFERENCE="template_sharpen_shapeupdate_orig-asr_aligned.nii.gz"
WARP="W2T_1Warp.nii.gz"
AFFINE="W2T_0GenericAffine.mat"
OUTPUT_ANN="WHS_SD_annotation_template_space.nii.gz"

for f in "${INPUT_ANN}" "${REFERENCE}" "${WARP}" "${AFFINE}"; do
  if [[ ! -f "${f}" ]]; then
    echo "ERROR: Required file '${f}' not found." >&2
    exit 1
  fi
done

echo "Applying transforms to annotation..."
echo "  Input annotation: ${INPUT_ANN}"
echo "  Reference image : ${REFERENCE}"
echo "  Warp field      : ${WARP}"
echo "  Affine transform: ${AFFINE}"

antsApplyTransforms \
  -d 3 \
  -i "${INPUT_ANN}" \
  -r "${REFERENCE}" \
  -t "${WARP}" \
  -t "${AFFINE}" \
  -n MultiLabel \
  -o "${OUTPUT_ANN}"

echo "Saved transformed annotation -> ${OUTPUT_ANN}"
