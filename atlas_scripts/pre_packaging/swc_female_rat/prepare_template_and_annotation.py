"""
Preprocess WHS Sprague Dawley rat template and annotation volumes.

This script downloads the original WHS SD rat atlas files and preprocesses them
for use in the SWC female rat atlas pipeline.

Original source files:
The original files can be obtained from NITRC:
- Template: https://www.nitrc.org/frs/download.php/12263/MBAT_WHS_SD_rat_atlas_v4_pack.zip
- Annotation: https://www.nitrc.org/frs/download.php/13400/MBAT_WHS_SD_rat_atlas_v4.01.zip

For convenience, the files are also available on the Brainglobe GIN repository:
- Template: https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/raw/master/pre-packaging/WHS_SD_rat_T2star_v1.01.nii.gz
- Annotation: https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/raw/master/pre-packaging/WHS_SD_rat_atlas_v4.01.nii.gz

This script:
- Downloads the template and annotation files from the GIN repository
- Reorients template and annotation volumes to ASR orientation
- Applies the same orientation transform to both volumes
- Zeros template voxels outside the annotated brain mask
- Saves the processed files:
  - WHS_SD_T2star_clean_ASR.nii.gz
  - WHS_SD_annotation_ASR.nii.gz

The output files are saved to the pre-packaging directory and serve as inputs
for subsequent ANTs registration and transform application scripts.
"""

import zipfile
from pathlib import Path

import nibabel as nib
import pooch
from nibabel.orientations import (
    apply_orientation,
    axcodes2ornt,
    io_orientation,
)

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.config import DEFAULT_WORKDIR

ATLAS_NAME = "swc_female_rat"
WHS_SD_TEMPLATE_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/pre-packaging/WHS_SD_rat_T2star_v1.01.nii.gz"
)
WHS_SD_ANNOTATION_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/pre-packaging/WHS_SD_rat_atlas_v4.01.nii.gz"
)

BASH_SCRIPTS_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/pre-packaging/ants_scripts.zip"
)

# Known hashes for the downloaded files. Modify if the files are updated.
WHS_SD_TEMPLATE_KNOWN_HASH = (
    "5369827d136c7f1cca637a41b7bb322a54c731018db7897325535c39bd10a050"
)
WHS_SD_ANNOTATION_KNOWN_HASH = (
    "40d45344e9b5ef6b2c22ad9c834d69f34ed2a57b4e4300b6570813769c694cde"
)
BASH_SCRIPTS_KNOWN_HASH = (
    "16b3b81321cf120e5d42ebe80ef0be8e17f0ba0d4d71bcdaeceda3f8f0e522b5"
)


def download_atlas_files(
    download_dir_path: Path,
    file_url: str,
    filename: str,
    known_hash: str = None,
):
    """Download an atlas file from a URL and save it to a directory."""
    utils.check_internet_connection()
    file_path = pooch.retrieve(
        url=file_url,
        known_hash=known_hash,
        path=download_dir_path,
        fname=filename,
        progressbar=True,
    )
    return Path(file_path)


def prepare_template_and_annotation(download_dir_path: Path):
    """Prepare template and annotation files for the SWC female rat atlas."""
    print("Downloading template file...")
    template_path = download_atlas_files(
        download_dir_path,
        WHS_SD_TEMPLATE_URL,
        "WHS_SD_rat_T2star_v1.01.nii.gz",
        WHS_SD_TEMPLATE_KNOWN_HASH,
    )

    print("Downloading annotation file...")
    annotation_path = download_atlas_files(
        download_dir_path,
        WHS_SD_ANNOTATION_URL,
        "WHS_SD_rat_atlas_v4.01.nii.gz",
        WHS_SD_ANNOTATION_KNOWN_HASH,
    )

    print("Downloading bash scripts...")
    bash_scripts_path = download_atlas_files(
        download_dir_path,
        BASH_SCRIPTS_URL,
        "ants_scripts.zip",
        BASH_SCRIPTS_KNOWN_HASH,
    )
    with zipfile.ZipFile(bash_scripts_path, "r") as zip_ref:
        zip_ref.extractall(download_dir_path)

    print("Loading volumes...")
    # Load the template and annotation files
    template_img = nib.load(template_path)
    annotation_img = nib.load(annotation_path)

    template = template_img.get_fdata()
    annotation = annotation_img.get_fdata()

    print("Reorienting volumes to ASR orientation...")
    # Desired orientation: Anterior–Superior–Right
    target_orient = axcodes2ornt(("A", "S", "R"))

    # Determine current orientation from affine
    current_orient = io_orientation(template_img.affine)

    # Build orientation transform from current to target
    transform_orient = nib.orientations.ornt_transform(
        current_orient, target_orient
    )

    # Apply orientation transform to data
    template_asr = apply_orientation(template, transform_orient)
    annotation_asr = apply_orientation(annotation, transform_orient)

    # Compute new affine in target orientation
    new_affine = template_img.affine @ nib.orientations.inv_ornt_aff(
        transform_orient, template.shape
    )

    print("Cleaning template (zeroing voxels outside brain)...")
    # Zero template voxels outside the annotation mask
    template_clean_asr = template_asr.copy()
    template_clean_asr[annotation_asr == 0] = 0

    print("Saving output files...")
    # Save outputs
    template_output_path = download_dir_path / "WHS_SD_T2star_clean_ASR.nii.gz"
    template_clean_img = nib.Nifti1Image(template_clean_asr, affine=new_affine)
    nib.save(template_clean_img, template_output_path)

    annotation_output_path = download_dir_path / "WHS_SD_annotation_ASR.nii.gz"
    annotation_asr_img = nib.Nifti1Image(annotation_asr, affine=new_affine)
    nib.save(annotation_asr_img, annotation_output_path)

    print("Done!")
    return template_output_path, annotation_output_path


if __name__ == "__main__":
    print("Preparing template and annotation files...")
    # Generated atlas path:
    pre_packaging_dir = DEFAULT_WORKDIR / ATLAS_NAME / "pre-packaging"
    pre_packaging_dir.mkdir(exist_ok=True, parents=True)
    prepare_template_and_annotation(pre_packaging_dir)
