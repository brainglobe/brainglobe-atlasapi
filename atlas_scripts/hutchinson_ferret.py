import re
from pathlib import Path

import numpy as np
import pooch
import json
import SimpleITK as sitk

from pyarrow import csv

from brainglobe_atlasapi import utils
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.utils import atlas_name_from_repr

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
ATLAS_NAME = "hutchinson_ferret"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1016/j.neuroimage.2017.03.009"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mustela putorius furo"

# The URL for the data files
ATLAS_LINK = "https://scalablebrainatlas.incf.org/templates/HSRetal17/source/evT2_template.nii.gz"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "lpi" #CHECK LATER

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = None  # CHECK LATER

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 500

# --- Script toggles (no CLI args) ---
# If True, do not re-download files that already exist on disk.
SKIP_DOWNLOADS_IF_PRESENT = True
TEMPLATE_URL = "https://scalablebrainatlas.incf.org/templates/HSRetal17/source/evT2_template.nii.gz"
ANNOTATION_URL = "https://scalablebrainatlas.incf.org/templates/HSRetal17/source/evDTI_SEGMENTATION.nii.gz"
LABELS_URL = "https://scalablebrainatlas.incf.org/services/labelmapper.php?template=HSRetal17&to=all&format=json"

WHOLE_BRAIN_MESH_URL = (
    "https://scalablebrainatlas.incf.org/templates/HSRetal17/wholebrain.x3d"
)

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TEMPLATE_FNAME = "evT2_template.nii.gz"
ANNOTATION_FNAME = "evDTI_SEGMENTATION.nii.gz"
LABELS_FNAME = "labelmapper.json"

WHOLE_BRAIN_MESH_FNAME = "wholebrain.x3d"

ATLAS_PACKAGER = "Jung Woo Kim"

def hex_to_rgb(hex):
    """Convert a hexadecimal color string to an RGB triplet.

    Parameters
    ----------
    hex : str
        The hexadecimal color string (e.g., "RRGGBB").

    Returns
    -------
    list
        A list of three integers representing the RGB color (0-255).
    """
    rgb = []
    for i in (0, 2, 4):
        intvalue = int(hex[i : i + 2], 16)
        rgb.append(intvalue)

    return rgb

def download_resources():
    """Download the necessary resources for the atlas (with Pooch)."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    template_path = DOWNLOAD_DIR_PATH / TEMPLATE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = (
        (not template_path.exists())
        or (not annotation_path.exists())
        or (not labels_path.exists())
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(template_path):
        pooch.retrieve(
            TEMPLATE_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=TEMPLATE_FNAME,
            known_hash="05f9a5cd5dc4c35d8f5eddf395fd3d79b6a40faeb99b08694adb59ebdb0fefbe",
            progressbar=True,
        )
    if should_fetch(annotation_path):
        pooch.retrieve(
            ANNOTATION_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            known_hash="f167a9aa8b67ec547e82609275530720153da6f37f2ae70128f54819af4b31f4",
            progressbar=True,
        )
    if should_fetch(labels_path):
        pooch.retrieve(
            LABELS_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            known_hash="286f4cab8cd1a355574f221215d0075963c2f6a40c9b46c727804181637f4da1",
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
    template_path = DOWNLOAD_DIR_PATH / TEMPLATE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    
    reference = load_any(template_path, as_numpy=True)
    annotation = load_any(annotation_path, as_numpy=True)
    
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 0 marks the left hemisphere
    and 1 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information(annotation_volume: np.ndarray):
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
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME
    
    # Filter structures to those actually present.
    present_ids = set(map(int, np.unique(annotation_volume)))
    
    structures_by_id: dict[int, dict] = {
        ROOT_ID: {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    }
    
    with open(labels_path) as f:
        labels_data = json.load(f)

    for label in labels_data:
        if label[0] not in present_ids:
            continue
        if label[0] == 0 or label[2] == "Clear Label":
            continue
        id = label[0]
        hex_colour = label[1]
        acronym = label[2]
        name = label[3]
        rgb_colour = hex_to_rgb(hex_colour)
        if id not in structures_by_id:
            structures_by_id[id] = {
                "id": id,
                "name": name,
                "acronym": acronym,
                "structure_id_path": [ROOT_ID, id],
                "rgb_triplet": rgb_colour,
            }
    
    # Sort structures by depth of hierarchy, then ID. 
    structures = list(structures_by_id.values())
    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures


def retrieve_or_construct_meshes(annotated_volume, structures, working_dir):
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
    # Construct meshes from the annotation volume.
    # Requires atlas generation extras: vedo + PyMCubes.
    meshes_dict = construct_meshes_from_annotation(
        save_path=Path(working_dir),
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
    
    # ADD DEC AND OTHER IMAGES AVAILABLE ON THE WEBSITE
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
    structures = retrieve_structure_information(annotated_volume)
    meshes_dict = retrieve_or_construct_meshes(
        annotated_volume=annotated_volume,
        structures=structures,
        working_dir=bg_root_dir,
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
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )
