"""Package the BRAIN/Minds Marmoset Brain Atlas.

This script generates the BRAIN/Minds marmoset brain atlas,
based on data from BRAIN/Minds. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

import json
import re
from pathlib import Path

import numpy as np
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
ATLAS_NAME = "bma2.0_marmoset"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1038/s41597-026-06601-z"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Callithrix jacchus"

# The URL for the data files
ATLAS_LINK = "https://figshare.com/articles/dataset/The_Brain_MINDS_3D_Digital_Marmoset_Brain_Atlas_Version_2_0/29992687/5"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "lpi"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 1

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 100

ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://ndownloader.figshare.com/files/58252144"
ANNOTATION_URL = "https://ndownloader.figshare.com/files/58616815"
LABELS_URL = "https://ndownloader.figshare.com/files/58252051"
IN_VIVO_REFERENCE_URL = "https://ndownloader.figshare.com/files/58252147"
MYELIN_REFERENCE_URL = "https://ndownloader.figshare.com/files/58252168"
NISSL_REFERENCE_URL = "https://ndownloader.figshare.com/files/58252156"
HIERARCHY_URL = "https://dataportal.brainminds.jp/ZAViewer_BMA_2019/regionTree.json?ver=20230203"

# TODO Add DWI in vivo MRI reference? It's ~7GB


REFERENCE_FNAME = "BMA2.0_avg_exvivo_T2WI.nii.gz"
ANNOTATION_FNAME = "BMA2.0_regions_label.nii.gz"
LABELS_FNAME = "BMA2.0_regions_list.ctbl"
IN_VIVO_REFERENCE_FNAME = "BMA2.0_avg_invivo_T2WI.nii.gz"
MYELIN_REFERENCE_FNAME = "BMA2.0_avg_myelin.nii.gz"
NISSL_REFERENCE_FNAME = "BMA2.0_avg_nissl.nii.gz"
HIERARCHY_FNAME = "regionTree.json"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

REFERENCE_PATH = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
ANNOTATION_PATH = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
LABELS_PATH = DOWNLOAD_DIR_PATH / LABELS_FNAME
IN_VIVO_REFERENCE_PATH = DOWNLOAD_DIR_PATH / IN_VIVO_REFERENCE_FNAME
MYELIN_REFERENCE_PATH = DOWNLOAD_DIR_PATH / MYELIN_REFERENCE_FNAME
NISSL_REFERENCE_PATH = DOWNLOAD_DIR_PATH / NISSL_REFERENCE_FNAME
HIERARCHY_PATH = DOWNLOAD_DIR_PATH / HIERARCHY_FNAME


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
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    needs_download = (
        (not REFERENCE_PATH.exists())
        or (not ANNOTATION_PATH.exists())
        or (not LABELS_PATH.exists())
        or (not IN_VIVO_REFERENCE_PATH.exists())
        or (not MYELIN_REFERENCE_PATH.exists())
        or (not NISSL_REFERENCE_PATH.exists())
        or (not HIERARCHY_PATH.exists())
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
            known_hash="9edd6684945e68aa25a968b444acd2c2a02eea6e85a0971df46b224cc0d9e286",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(ANNOTATION_PATH):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="e0b16851bf4cca255e668ec4708855ab0bf95215b4c0dbceacf6b912c9c9f4ea",
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
        )

    if should_fetch(LABELS_PATH):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="d3e6d90ae4ddc75adac65c7344f50fb315f336cecc2cc23613831d7caf93a79a",
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            progressbar=True,
        )

    if should_fetch(IN_VIVO_REFERENCE_PATH):
        pooch.retrieve(
            url=IN_VIVO_REFERENCE_URL,
            known_hash="308f052a0406e1c67b20e208035e067acb569f9de7b4da1fbb2a92a17bd8cb58",
            path=DOWNLOAD_DIR_PATH,
            fname=IN_VIVO_REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(MYELIN_REFERENCE_PATH):
        pooch.retrieve(
            url=MYELIN_REFERENCE_URL,
            known_hash="2764f4483a8770c9c9d5b3e5b0aa474c9eed13f60a04f1130b3246ed86cc9424",
            path=DOWNLOAD_DIR_PATH,
            fname=MYELIN_REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(NISSL_REFERENCE_PATH):
        pooch.retrieve(
            url=NISSL_REFERENCE_URL,
            known_hash="c90c8ae5aa626d069e471e817cb7ea3706608975d3a7cbd401568ba8959b94c4",
            path=DOWNLOAD_DIR_PATH,
            fname=NISSL_REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(HIERARCHY_PATH):
        pooch.retrieve(
            url=HIERARCHY_URL,
            known_hash="f74d80d2b982b9d502fc30e8c9d5fdc3830c452408c042ade4a5782d4c77449b",
            path=DOWNLOAD_DIR_PATH,
            fname=HIERARCHY_FNAME,
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
    reference = load_any(REFERENCE_PATH)
    annotation = load_any(ANNOTATION_PATH)
    annotation_array = np.asarray(annotation)
    annotation_array = np.where(
        annotation_array < 10000, annotation_array, annotation_array - 10000
    )
    return reference, annotation_array


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


def retrieve_structure_information(annotation_volume):
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
    # Filter structures to those actually present.
    present_ids = set(map(int, np.unique(annotation_volume)))

    # print(present_ids)

    # .ctbl label file format:
    # Index Hemisphere:_Name_(Acronym) R G B A
    # OR
    # Index Hemisphere:_Name R G B A

    # Use regex parsing to avoid pandas whitespace/quoting edge-cases.
    line_re = re.compile(
        r"^(\d+)\s+[LR]H:_((\S+)_\((\S+)\)|(\w+\-*\w))\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$"
    )

    # Use the name and acronym used within the label files,
    # and then change them back to "root" later
    structures_by_acronym: dict[str, dict] = {
        "WHOLE": {
            "id": ROOT_ID,
            "name": "WHOLE BRAIN",
            "acronym": "WHOLE",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    }

    # Open BMA2.0 regions list file to get structure information
    with open(LABELS_PATH, "r") as f:
        labels_data = f.read().splitlines()
        for key, label in enumerate(labels_data):
            if not label.strip() or label.lstrip().startswith("#"):
                continue
            m = line_re.match(label)

            # Skip malformed lines
            if not m:
                continue

            # Skip background, root and hemisphere specific labels
            if int(m.group(1)) <= 1 or int(m.group(1)) > 9999:
                continue

            id = int(m.group(1))
            if not m.group(5):
                acronym = m.group(4)
                name = m.group(3).replace("_", " ")
            else:
                acronym = m.group(5)
                name = m.group(5)
                
            rgb_colour = [int(m.group(6)), int(m.group(7)), int(m.group(8))]
            
            # Fix weird unicode error for this region name
            if id == 570:
                name = "Intercalated Nucleus"

            if acronym not in structures_by_acronym:
                structures_by_acronym[acronym] = {
                    "id": id,
                    "name": name,
                    "acronym": acronym,
                    "structure_id_path": [],
                    "rgb_triplet": rgb_colour,
                }

    # Open regionTree file to get hierarchy information
    with open(HIERARCHY_PATH) as f:
        tree_data = json.load(f)
        regions = tree_data["regions"]

        # Loop through regionTree to add missing regions
        for region in regions:
            id = int(region["id"])
            parent = region["parent"]
            acronym = region["abb"]
            hex_colour = region.get("color", "#7F7F7F").lstrip("#")
            acronym = region["abb"]
            name = region["name"]
            rgb_colour = hex_to_rgb(hex_colour)

            if acronym in structures_by_acronym:
                structures_by_acronym[acronym]["parent"] = parent
            elif acronym not in structures_by_acronym:
                structures_by_acronym[acronym] = {
                    "id": id + 1000,
                    "name": name,
                    "acronym": acronym,
                    "structure_id_path": [],
                    "rgb_triplet": rgb_colour,
                    "parent": parent,
                }

    ancestry = []
    # Loop through structures to ensure all have structure_id_path
    for acronym, structure in structures_by_acronym.items():
        if acronym == "WHOLE":
            continue
        ancestry.append(
            (
                structure["id"],
                (
                    structures_by_acronym[structure["parent"]]["id"]
                    if "parent" in structure
                    else 1
                ),
            )
        )

    # Code to try and recursively find full hierarchy
    parents = set()
    children = {}
    for child, parent in ancestry:
        parents.add(parent)
        children[child] = parent

    # Recursively determine parents until child has no parent
    def ancestors(parent):
        return (ancestors(children[parent]) if parent in children else []) + [
            parent
        ]
    
    # TODO FIX THE REMAINING STRAGGLERS, CEREBELLUM ETC

    # For each structure find all its ancestors, and remove parent key
    for acronym, structure in structures_by_acronym.items():
        structure.pop("parent", None)
        structure["structure_id_path"] = ancestors(structure["id"])
    


    # Change back the root structure details
    structures_by_acronym["WHOLE"]["name"] = "root"
    structures_by_acronym["WHOLE"]["acronym"] = "root"

    # Return root_id alongside structures.
    # Sort structures by depth of hierarchy, then ID.
    structures = list(structures_by_acronym.values())
    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    
    # Check missing childless children, print them out
    for childless in (set(children.keys()) - parents):
        if childless not in present_ids:
            print(structures[childless])
    
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
    # Construct meshes from the annotation volume.
    # Requires atlas generation extras: vedo + PyMCubes.
    quit()
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
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
    )
