"""Template script for generating a BrainGlobe atlas.

Use this script as a starting point to package a new BrainGlobe atlas by
filling in the required functions and metadata.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

### Metadata ###
__version__ = 0

ATLAS_NAME = "gunturkun_pigeon"
CITATION = "https://doi.org/10.1007/s00429-012-0400-y"
SPECIES = "Columba livia"

ATLAS_LINK = "https://ruhr-uni-bochum.sciebo.de/s/el9oeWDkMtczWDx"
ATLAS_DOWNLOAD_URL = "https://ruhr-uni-bochum.sciebo.de/public.php/dav/files/el9oeWDkMtczWDx/Full_package/?accept=zip"
ATLAS_DOWNLOAD_FNAME = "Full_package.zip"

SOURCE_ORIENTATION = "ipl"
SOURCE_RESOLUTION = (100, 80, 80)  # in microns

ORIENTATION = "asr"
RESOLUTION = (80, 100, 80)

ROOT_ID = 999

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

NON_STRUCTURAL_DIRS = [
    "Brainsurface",
    "CT",
    "T2",
    "T2star",
]

REGION_INDICES = {
    "auditory1": {
        1: "An",
        2: "La",
        3: "Mc",
        4: "MLD",
        5: "Ov",
        6: "Field L2",
    },
    "auditory2": {
        1: "OS",
        2: "LLv",
        3: "LLd",
    },
    "arcopallium": {
        1: "S",
        2: "GP",
        3: "TnA",
    },
    "Olfactory": {1: "BO", 2: "CPP", 3: "CPi"},
    "GLd-and-rotundus": {1: "Rt", 2: "GLd", 3: "GLd"},
    "visual-Wulst_HA_HI_HD-until-A13": {
        1: "HA",
        2: "HI - HD",
    },
    "nBOR-Lentiformis-mesencephali": {
        1: "nBOR",
        2: "LM",
    },
    "SLu-Ipc-Imc-left": {
        1: "Imc",
        2: "Ipc",
        3: "SLu",
    },
    "PrV-and-Basalis": {
        1: "PrV",
        2: "Bas",
    },
    "Wulst_HA_HI_HD-frontal-from-A13": {
        1: "HA",
        2: "HI - HD",
    },
    "GC_DLP_DIVA": {
        1: "GC",
        2: "DLP",
        3: "DIVA",
    },
}


def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    atlas_download_path = DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        else:
            return False

    if should_fetch(atlas_download_path):
        pooch.retrieve(
            url=ATLAS_DOWNLOAD_URL,
            known_hash="0db28c1b3de1e354323740dfc933d9b172b8b0fac3b2b4bac163c26274035375",
            path=DOWNLOAD_DIR_PATH,
            fname=ATLAS_DOWNLOAD_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )


def retrieve_reference():
    """
    Retrieve the reference volume.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    numpy.ndarray
        The reference volume.
    """
    reference = load_any(
        DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip") / "T2/T2.nii.gz"
    )

    # Remove the superior-most slice of reference, volume, as annotations are in (256, 308, 199)
    # but the reference is in (256, 308, 200).
    reference = np.delete(reference, 199, axis=2).squeeze()
    dmin = np.min(reference)
    dmax = np.max(reference)
    dscale = (2**16 - 1) / (dmax - dmin)
    reference = (reference - dmin) * dscale
    reference = reference.astype(np.uint16)
    return reference


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
    hemisphere_dir = (
        DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip") / "Brainsurface"
    )
    left = nib.load(hemisphere_dir / "brainsurface_left.hdr")
    left_hemisphere = left.get_fdata()

    hemispheres_stack = np.where(left_hemisphere == 0, 2, 1)
    return hemispheres_stack


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
    structures_by_acronym = {
        "root": {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    }

    startpath = str(DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip"))
    for root, dirs, files in os.walk(startpath):
        current_dir = root.replace(startpath, "").strip(os.sep)
        if current_dir in NON_STRUCTURAL_DIRS:
            continue
        level = root.replace(startpath, "").count(os.sep)
        structure_name_path = ["root"]
        for i in range(level):
            structure_name_path.append(root.split(os.sep)[-(level - i)])
        print(structure_name_path)

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
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
        )
    download_resources()
    reference_volume = retrieve_reference()
    annotated_volume = None
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes()

    quit()
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
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )
