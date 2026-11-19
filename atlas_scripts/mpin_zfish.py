"""Package the MPIN Z-Fish atlas.

The MPIN Z-Fish atlas is derived from data published by Kunst et al. (2019).
It includes volumetric data for reference, annotation, and hemispheres,
along with a hierarchical structure tree and corresponding meshes for
brain regions.
"""

from pathlib import Path

import json
import os
import requests
import numpy as np
import pooch
import SimpleITK as sitk
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

### Metadata ###
__version__ = 2
ATLAS_NAME = "mpin_zfish"
CITATION = "https://doi.org/10.1016/j.neuron.2019.04.034"
SPECIES = "Danio rerio"
ATLAS_LINK = r"https://mapzebrain.org"
ORIENTATION = "ial"
ROOT_ID = 9999
RESOLUTION = 1
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://api.mapzebrain.org/media/Brains/Standard_brain/standard_brain_fixed_SYP_T_GAD1b.nrrd"

ANNOTATION_URL = "https://api.mapzebrain.org/media/downloads/Regions/mapZebrain__regions__v2.0.1.zip"

LABELS_URL = "https://api.mapzebrain.org/api/load_region_data_by_version/"

REFERENCE_FNAME = "standard_brain_fixed_SYP_T_GAD1b.nrrd"

# TODO
T2W_REFERENCE_FNAMES = {
    150: "tpl-RBTV1_res-015_T2w.nii.gz",
    330: "tpl-RBTV1_res-033_T2w.nii.gz",
}

# TODO: Something is weird in the prosencephalon, figure out what's wrong

ANNOTATION_DNAME = "mapZebrain__regions__v2.0.1"

FNAME_TO_REGION = {
    "ventral_telencephalon_(subpallium).tif": "subpallium (ventral telencephalon)",
    "pituitary.tif": "pituitary (hypophysis)",
    "epiphysis.tif": "pineal complex (epiphysis)",
    "dorsal_telencephalon_(pallium).tif": "pallium (dorsal telencephalon)",
    "sac__spv.tif": "boundary zone between SAC and periventricular layer",
    "stratum_album_centrale.tif": "stratum album centrale (SAC)",
    "stratum_griseum_centrale.tif": "stratum griseum centrale (SGC)",
    "stratum_opticum.tif": "stratum opticum (SO)",
    "stratum_fibrosum_et_griseum_superficiale.tif": "stratum fibrosum et griseum superficiale (SFGS)",
    "stratum_marginale.tif": "stratum marginale (SM)",
    "glutamatergic_domain.tif": "glutamatergic domain of the nucleus isthmi",
    "posterior_cholinergic_domain.tif": "posterior cholinergic domain of the nucleus isthmi",
    "gabaergic_domain.tif": "gabaergic domain of the nucleus isthmi",
    "anterior_cholinergic_domain.tif": "anterior cholinergic domain of the nucleus isthmi",
    "sfgs__sgc.tif": "boundary zone between SFGS and SGC", 
    "nucleus_of_the_medial_longitudinal_fascicle_(pretectum,_basal_part).tif": "nucleus of the medial longitudinal fascicle",
    "posterior_tuberculum_(basal_part_of_prethalamus_and_thalamus).tif": "posterior tuberculum, posterior part (basal prosomere 2, dorsal posterior tuberculum)",
}
# Note: "retinal arborization field 10" does not exist in structures file. 

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

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
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal)

    return rgb

def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_DNAME

    needs_download = (
        (not reference_path.exists())
        or (not annotation_path.exists())
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
            known_hash="7bd5cdffb256ffbae3dde807845ada1c8d713630f236028dbad6401821a42eca",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="f10e87f0b82c461278d95c18b465dadd40631475a4364be6d0c66e8f80bc14e5",
            path=DOWNLOAD_DIR_PATH,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

def retrieve_reference_and_annotation(structures):
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    
    # Locate leaf nodes of structure tree, as only leaf structures need annotations
    parent_ids = {
        ancestor_id
        for structure in structures
        for ancestor_id in structure["structure_id_path"][:-1]
    }
    leaf_structures = [
        structure for structure in structures if structure["id"] not in parent_ids
    ]

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_DNAME
    
    reference_img = sitk.ReadImage(str(reference_path))
    reference = sitk.GetArrayFromImage(reference_img)
    reference -= np.min(reference)
    reference = reference / np.max(reference)
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    
    annotation = np.zeros_like(reference)
    
    def find_id(structures, name):
        for structure in structures: 
            if structure["name"] == name:
                return int(structure["id"])
        # Return -1 if the given name cannot be found in structures
        return -1
    
    for root, _, files in os.walk(annotation_path):
        for filename in files: 
            str_name = filename.removesuffix(".tif").replace("_", " ")
            if any(
                structure["name"] == str_name for structure in leaf_structures
            ):
                region_annotation = np.asarray(load_any(annotation_path / filename))
                id = find_id(leaf_structures, str_name)
                if id == -1: 
                    continue
                annotation = np.where(region_annotation > 0, id, annotation)
            elif FNAME_TO_REGION.get(filename, None):
                str_name = FNAME_TO_REGION.get(filename)
                region_annotation = np.asarray(load_any(annotation_path / filename))
                id = find_id(leaf_structures, str_name)
                if id == -1: 
                    continue
                annotation = np.where(region_annotation > 0, id, annotation)
    
    annotation = annotation.astype(np.uint32)    
    return reference, annotation


def retrieve_hemisphere_map(annotation_volume):
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    None
        Returns None as atlas is aligned to the midline
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
    
    # Use requests.post to retrieve structures .json file
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://mapzebrain.org",
        "Referer": "https://mapzebrain.org/",
        "User-Agent": "Mozilla/5.0",
    }

    payload = {
        "regionVersion": "2.0 (MECE, 2024)"
    }

    response = requests.post(
        LABELS_URL,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()
    regions_by_id = data["data"]["regions_dictionary"]
    
    structures_by_id = {
        ROOT_ID: {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID], 
            "rgb_triplet": [255, 255, 255],
        }
    }
    
    # Create a new dictionary to remove unwanted information from original dictionary
    for region_id, region in regions_by_id.items():
        if region["version"]["number"] != "2.0":
            continue
        id = int(region["id"])
        name = region["name"]
        acronym = region["short_name"]
        rgb_triplet = hex_to_rgb(region["version"]["color"].strip("#"))
        parent = region["parent"] if region["parent"] != -1 else ROOT_ID
        structures_by_id[id] = {
            "id": id,
            "name": name,
            "acronym": acronym,
            "rgb_triplet": rgb_triplet,
            "parent": parent,
        }
    
    ancestry = []
    
    # Loop through structures to add ancestry 
    for structure_id, structure in structures_by_id.items():
        if structure["acronym"] == "root":
            continue
        ancestry.append(
            (
                structure_id,
                structure["parent"],
            )
        )

    # Code to recursively find full hierarchy
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

    # For each structure find all its ancestors, and remove parent key
    for structure_id, structure in structures_by_id.items():
        if structure["acronym"] == "root":
            continue
        structures_by_id[structure_id]["structure_id_path"] = ancestors(structure["id"])
        structures_by_id[structure_id].pop("parent")

    structures = list(structures_by_id.values())
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
    structures = retrieve_structure_information()
    reference_volume, annotated_volume = retrieve_reference_and_annotation(structures)
    hemispheres_stack = retrieve_hemisphere_map(annotated_volume)
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
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        additional_references=None,
        atlas_packager=ATLAS_PACKAGER,
        scale_meshes=True,
    )
