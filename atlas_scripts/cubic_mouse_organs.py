"""Package the CUBIC Mouse Organs Atlases.

This script generates the CUBIC mouse organs atlases,
based on data published by Yoshida et al. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
# remember to add {ATLAS_NAME}_{RESOLUTION}um to:
# brainglobe_atlasapi/atlas_names.py
ATLAS_NAME = "cubic_mouse_organs"
CITATION = "https://doi.org/10.1016/j.cell.2025.12.057"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://drive.google.com/drive/folders/11QnUYaTD2blipXxWAsCk-KOxZeccXKWM"
)
ORIENTATION = "psr"

ROOT_ID = 999
RESOLUTION = 50
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

HEART_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1xlz83fEmEUUHEu7O5BtX8dbYg_hzDHFm"
HEART_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=15bncKMJh3LfneEM4LgBm3zBP4eK51RC8"
HEART_LABELS_URL = "https://drive.google.com/uc?export=download&id=1HE-oL6I75Bkz3SkAlcFK0kZoq-lIjnwy"

HEART_REFERENCE_FNAME = "03_Heart_average_atlas_50um.tif"
HEART_ANNOTATION_FNAME = "03_Heart_atlas_segmentation.tif"
HEART_LABELS_FNAME = "heart_ID_list.xlsx"

LUNGS_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1Aceo0-W4Kz2kONEgrMjwHCUp4keL_DPo"
LUNGS_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=1xoNJzxWLsPYlH8ADUqQPP4v_yHhF5iDl"
LUNGS_LABELS_URL = "https://drive.google.com/uc?export=download&id=11xyChZ8nJrwoNbZaalOq5E2lr1eeUujF"

LUNGS_REFERENCE_FNAME = "04_Lung_average_atlas_100um.tif"
LUNGS_ANNOTATION_FNAME = "04_Lung_atlas_segmentation.tif"
LUNGS_LABELS_FNAME = "lungs_ID_list.xlsx"

# LIVER_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1xlz83fEmEUUHEu7O5BtX8dbYg_hzDHFm"
# LIVER_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=15bncKMJh3LfneEM4LgBm3zBP4eK51RC8"
# LIVER_LABELS_URL = "https://drive.google.com/uc?export=download&id=1HE-oL6I75Bkz3SkAlcFK0kZoq-lIjnwy"

# LIVER_REFERENCE_FNAME = "11_Neonatal_body_atlas_density_img_100um.tif"
# LIVER_ANNOTATION_FNAME = "11_Neonatal_body_atlas_segmentation.tif"
# LIVER_LABELS_FNAME = "liver_ID_list.xlsx"

KIDNEYS_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1cZkTosDm1e0DFMdbhJ6FIqK8KhV-wfhW"
KIDNEYS_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=1pESRqCI1TuDcaMLOaTLjtfneoW_qJuBK"
KIDNEYS_LABELS_URL = "https://drive.google.com/uc?export=download&id=13K8MhUmGUSZfhBwb6x9-iHI1WhkYrxhL"

KIDNEYS_REFERENCE_FNAME = "07_Kidney_average_atlas_50um.tif"
KIDNEYS_ANNOTATION_FNAME = "07_Kidney_segmentation.tif"
KIDNEYS_LABELS_FNAME = "kidneys_ID_list.xlsx"

REFERENCE_URLS = [
    HEART_REFERENCE_URL,
    LUNGS_REFERENCE_URL,
    KIDNEYS_REFERENCE_URL,
]
REFERENCE_FNAMES = [
    HEART_REFERENCE_FNAME,
    LUNGS_REFERENCE_FNAME,
    KIDNEYS_REFERENCE_FNAME,
]

ANNOTATION_URLS = [
    HEART_ANNOTATION_URL,
    LUNGS_ANNOTATION_URL,
    KIDNEYS_ANNOTATION_URL,
]
ANNOTATION_FNAMES = [
    HEART_ANNOTATION_FNAME,
    LUNGS_ANNOTATION_FNAME,
    KIDNEYS_ANNOTATION_FNAME,
]

LABELS_URLS = [HEART_LABELS_URL, LUNGS_LABELS_URL, KIDNEYS_LABELS_URL]
LABELS_FNAMES = [HEART_LABELS_FNAME, LUNGS_LABELS_FNAME, KIDNEYS_LABELS_FNAME]


def generate_pseudorandom_rgbs(n_rgbs: int, seed: int = 0):
    """Generate a list of n_rgbs RGB triplets given a seed."""
    rng = np.random.default_rng(seed)
    # n_rgbs RGB values, each channel between 0 and 255 inclusive
    rgb_values = rng.integers(0, 256, size=(n_rgbs, 3)).tolist()
    return rgb_values


def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = (
        (not reference_path.exists())
        or (not annotation_path.exists())
        or (not labels_path.exists())
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(reference_path):
        pooch.retrieve(
            url=REFERENCE_URL,
            known_hash="bbd6944c0c6e92cf83049259ca3b48c496cf10d8ef82a718f337ecdcfc3c59ee",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="be03ef141f07e633f44524a685bd42161f0582e26f7e0c804144e194083fbd26",
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="16ec41f94b6a7c8e34f802c5f242f9d4e0338b927090859ebfc8b227c59b9016",
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    reference = load_any(reference_path)
    ref_min = np.min(reference)
    ref_max = np.max(reference)
    reference = (reference - ref_min) / (ref_max - ref_min) * 65535
    reference = reference.astype(np.uint16)
    annotation = load_any(annotation_path)
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Return a list of dictionaries with information about the atlas.

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    The expected format for each dictionary is:

    .. code-block:: python

        {
            "id": int,
            "name": str,
            "acronym": str,
            "structure_id_path": list[int],
            "rgb_triplet": list[int, int, int],
        }

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    # TODO: Requires installling openpyxl, which I'm not sure is in the pyproject.toml?
    labels_df = pd.read_excel(
        DOWNLOAD_DIR_PATH / LABELS_FNAME, engine="openpyxl"
    )
    labels_df = labels_df.iloc[1].dropna()
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    rgbs = generate_pseudorandom_rgbs(labels_df.shape[0], 42)

    for id, name in labels_df.items():
        if id == "ID" or name == "none":
            continue
        structures.append(
            {
                "id": int(id),
                "name": name,
                "acronym": name,
                "structure_id_path": [999, int(id)],
                "rgb_triplet": rgbs[int(id)],
            }
        )

    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures


def retrieve_or_construct_meshes(annotated_volume, structures):
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    If the atlas is packaged with mesh files, download and use them. Otherwise,
    construct the meshes using available helper functions.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dict = construct_meshes_from_annotation(
        save_path=DOWNLOAD_DIR_PATH,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=False,
        parallel=True,
        verbosity=0,
        num_threads=-1,
    )

    structures_with_mesh = [s for s in structures if s["id"] in meshes_dict]

    return meshes_dict, structures_with_mesh


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    This function should be edited only if the atlas includes additional
    reference images. The dictionary should map the name of each additional
    reference image to its corresponding image stack data.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    if RESOLUTION is None:
        raise ValueError("RESOLUTION must be set before running this script.")

    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(bg_root_dir.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {bg_root_dir}. "
        )
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
        annotated_volume, structures
    )

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
    )
