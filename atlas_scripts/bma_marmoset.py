"""Package the BRAIN/Minds Marmoset Brain Atlas.

This script generates the BRAIN/Minds marmoset brain atlas,
based on data from BRAIN/Minds. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

from pathlib import Path

import pooch
import SimpleITK as sitk

from brainglobe_atlasapi import utils
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
ATLAS_NAME = "bma_marmoset"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1038/s41597-026-06601-z"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Callithrix jacchus"

# The URL for the data files
ATLAS_LINK = "https://figshare.com/articles/dataset/The_Brain_MINDS_3D_Digital_Marmoset_Brain_Atlas_Version_2_0/29992687/5"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = None

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 50

ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://ndownloader.figshare.com/files/58252147"
ANNOTATION_URL = "https://ndownloader.figshare.com/files/58616818"
LABELS_URL = "https://ndownloader.figshare.com/files/58252051"
EX_VIVO_REFERENCE_URL = "https://ndownloader.figshare.com/files/58252144"
MYELIN_REFERENCE_URL = "https://ndownloader.figshare.com/files/58252168"
NISSL_REFRENCE_URL = "https://ndownloader.figshare.com/files/58252156"


REFERENCE_FNAME = "BMA2.0_avg_invivo_T2WI.nii.gz"
ANNOTATION_FNAME = "BMA2.0_regions_label_50mu.nii.gz"
LABELS_FNAME = "BMA2.0_regions_list.ctbl"
EX_VIVO_REFERENCE_FNAME = "BMA2.0_avg_exvivo_T2WI.nii.gz"
MYELIN_REFERENCE_FNAME = "BMA2.0_avg_myelin.nii.gz"
NISSL_REFRENCE_FNAME = "BMA2.0_avg_nissl.nii.gz"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"


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
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash=None,
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
    reference_path = DOWNLOAD_DIR_PATH / "atlasVolume/atlasVolume.mhd"
    ref_image = sitk.ReadImage(reference_path)
    reference = sitk.GetArrayFromImage(ref_image)
    annotation_path = DOWNLOAD_DIR_PATH / "annotation.mhd"
    ann_image = sitk.ReadImage(annotation_path)
    annotation = sitk.GetArrayFromImage(ann_image)
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
    return None


def retrieve_or_construct_meshes():
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
    meshes_dict = {}
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
    meshes_dict = retrieve_or_construct_meshes()

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
