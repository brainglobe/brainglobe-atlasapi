"""Package the MPIN Z-Fish atlas.

The MPIN Z-Fish atlas is derived from data published by Kunst et al. (2019).
It includes volumetric data for reference, annotation, and hemispheres,
along with a hierarchical structure tree and corresponding meshes for
brain regions.
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

### Metadata ###
__version__ = 2
ATLAS_NAME = "mpin_zfish"
CITATION = "https://doi.org/10.1016/j.neuron.2019.04.034"
SPECIES = "Danio rerio"
ATLAS_LINK = r"https://mapzebrain.org"
ORIENTATION = "rai"
ROOT_ID = 9999
RESOLUTION = 1
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://api.mapzebrain.org/media/Brains/Standard_brain/standard_brain_fixed_SYP_T_GAD1b.nrrd"

ANNOTATION_URL = "https://api.mapzebrain.org/media/downloads/Regions/mapZebrain__regions__v2.0.1.zip"

LABELS_URL = "https://china.scidb.cn/download?fileId=f8ab5dc5cd6bd65e0d4b9396127a37bf&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791"

REFERENCE_FNAME = "standard_brain_fixed_SYP_T_GAD1b.nrrd"

# TODO
T2W_REFERENCE_FNAMES = {
    150: "tpl-RBTV1_res-015_T2w.nii.gz",
    330: "tpl-RBTV1_res-033_T2w.nii.gz",
}

ANNOTATION_FNAME = "mapZebrain__regions__v2.0.1"

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
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME

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
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
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

    reference = np.asarray(load_any(reference_path), dtype=np.float64)
    reference -= np.min(reference)
    reference = reference / np.max(reference)
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    
    
    annotation = load_any(annotation_path)

    return reference, annotation


def retrieve_hemisphere_map(annotation_volume):
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    hemispheres_stack: numpy.ndarray
        A 3D array with the same shape as annotation, all values are 1 (L).
    """
    hemispheres_stack = np.where(annotation_volume > 0, 1, 0).astype(np.uint8)
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
    
    # Use requests.post to retrieve structures .json file
    headers = {
        "Accept": "application/json, text/plain, */*",
        "Content-Type": "application/json",
        "Origin": "https://mapzebrain.org",
        "Referer": "https://mapzebrain.org/",
        "User-Agent": "Mozilla/5.0",
    }

    response = requests.post(
        LABELS_URL,
        headers=headers,
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
    
    for region_id, region in regions_by_id.items():
        if region["version"]["number"] != "2.0":
            continue
        id = int(region["id"])
        name = region["name"]
        acronym = region["short_name"]
        rgb_triplet = hex_to_rgb(region["version"]["color"].strip("#"))
        parent = region["parent"]
        structures_by_id[id]["id"] = id
        structures_by_id[id]["name"] = name
        structures_by_id[id]["acronym"] = acronym
        structures_by_id[id]["rgb_triplet"] = rgb_triplet
        structures_by_id[id]["parent"] = parent
    
    print(structures_by_id)
    ancestry = []
    # Loop through structures to ensure all have structure_id_path
    for _, row in df.iterrows():
        if row["acronym"] == "root":
            continue
        ancestry.append(
            (
                row["id"],
                df.loc[df["acronym"] == row["ParentStruct"], "id"].item(),
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

    # For each structure find all its ancestors, and remove parent key
    for index, row in df.iterrows():
        df.at[index, "structure_id_path"] = ancestors(row["id"])

    df.drop(
        columns=[
            "red",
            "green",
            "blue",
            "StructureId",
            "ParentStruct",
        ],
        inplace=True,
    )

    structures = df.to_dict("records")

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
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
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
