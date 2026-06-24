"""Template script for generating a BrainGlobe atlas.

Use this script as a starting point to package a new BrainGlobe atlas by
filling in the required functions and metadata.
"""

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pooch

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
ATLAS_NAME = "hoops_dragon"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1007/s00429-021-02282-z"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Ctenophorus decresii"

# The URL for the data files
ATLAS_LINK = "https://osf.io/ujenq"

ATLAS_PACKAGER = "Jung Woo Kim"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "ila"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 999

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 50

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://osf.io/u6nmx/download"
ANNOTATION_URL = "https://osf.io/5frjw/download"
LABELS_URL = "https://osf.io/ewqgp/download"

REFERENCE_FNAME = "lizard_model.mnc"
ANNOTATION_FNAME = "lizard_segmentation_bilateral.mnc"
LABELS_FNAME = "BilateralRegionIDs.csv"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

REFERENCE_PATH = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
ANNOTATION_PATH = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
LABELS_PATH = DOWNLOAD_DIR_PATH / LABELS_FNAME


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

    needs_download = (
        (not REFERENCE_PATH.exists())
        or (not ANNOTATION_PATH.exists())
        or (not LABELS_PATH.exists())
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(REFERENCE_PATH):
        pooch.retrieve(
            url=REFERENCE_URL,
            known_hash="1fbc4d1be3b8a6da5513d1ac44abd11d05eebe67dd3c2e4662bbe81008c7c4e5",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(ANNOTATION_PATH):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="847de51f6de3f0cca6740ac7e393501a43af2ac85414b302f99e7a904df69df1",
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
        )

    if should_fetch(LABELS_PATH):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="bba5569843825a905eaac1c6432f53a6c6302ec30fee65f520fc1eb4bc3eb084",
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
    # Requires h5py package **************************************************
    reference_file = nib.load(REFERENCE_PATH)
    annotation_file = nib.load(ANNOTATION_PATH)

    # Rescale reference from float to uint16
    reference = reference_file.get_fdata()
    ref_min = reference.min()
    ref_max = reference.max()
    reference = (reference - ref_min) / (ref_max - ref_min) * 65535
    reference = reference.astype(np.uint16)

    # Collapse hemisphere-specific IDs into one
    annotation = annotation_file.get_fdata()
    annotation = np.asarray(annotation)
    annotation = np.where(annotation < 1000, annotation, annotation - 1000)
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
    labels = pd.read_csv(LABELS_PATH)
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]
    rgbs = generate_pseudorandom_rgbs(labels.shape[0], 1337)

    for index, row in labels.iterrows():
        id = int(row["left label"])
        name = row["Structure"].strip('"')
        acronym = row["abbreviation"].strip()
        structure_id_path = [ROOT_ID, id]
        structures.append(
            {
                "id": id,
                "name": name,
                "acronym": acronym,
                "structure_id_path": structure_id_path,
                "rgb_triplet": rgbs[index],
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
        save_path=Path(BG_ROOT_DIR),
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=False,
        parallel=True,
        verbosity=0,
    )
    return meshes_dict


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
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
        )
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)

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
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
    )
