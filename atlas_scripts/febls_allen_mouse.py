"""Package the FeBls Mouse Brain Atlas with Allen annotations.

This script generates the FeBls female mouse brain atlas with Allen annotations,
based on data from NITRC and the Allen Institute. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

import json
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

### Metadata ###
__version__ = 0
ATLAS_NAME = "febls_allen_mouse"
CITATION = "https://doi.org/10.1038/nature05453"
SPECIES = "Mus musculus"
ATLAS_LINK = r"https://www.nitrc.org/frs/download.php/20440/FeBls_v1.3.zip//?i_agree=1&download_now=1"
ORIENTATION = "rpi"
ROOT_ID = 999
RESOLUTION = 60
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

ATLAS_FNAME = "FeBls_v1.3.zip"

REFERENCE_FNAME = "FeBlsTemplateBrain.nii"
AMBA_REFERENCE_FNAME = "AMBA2FeBls_Warped.nii"
DSURQUE_REFERENCE_FNAME = "DSURQUE2FeBls_Warped.nii"
TMBTA_REFERENCE_FNAME = "TMBTA2FeBls_Warped.nii"


ALLEN_ANNOTATION_FNAME = "AMBAlabels4FeBls.nii"
DSURQUE_ANNOTATION_FNAME = "DSURQUElabels4FeBls.nii"
TMBTA_ANNOTATION_FNAME = "TMBTAlabels4FeBls.nii"

ALLEN_LABELS_URL = "https://atlas.brain-map.org/atlasviewer/ontologies/1.json"
ALLEN_LABELS_FNAME = "1.json"
DSURQUE_LABELS_FNAME = "DSURQUElabels4FeBls.txt"
TMBTA_LABELS_FNAME = "TMBTAlabels4FeBls.txt"

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

    atlas_path = DOWNLOAD_DIR_PATH / ATLAS_FNAME
    allen_labels_path = DOWNLOAD_DIR_PATH / ALLEN_LABELS_FNAME

    needs_download = (not atlas_path.exists()) or (
        not allen_labels_path.exists()
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(atlas_path):
        pooch.retrieve(
            url=ATLAS_LINK,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=ATLAS_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(allen_labels_path):
        pooch.retrieve(
            url=ALLEN_LABELS_URL,
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=ALLEN_LABELS_FNAME,
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
    annotation_path = DOWNLOAD_DIR_PATH / ALLEN_ANNOTATION_FNAME

    reference = np.asarray(load_any(reference_path), dtype=np.float64)
    reference -= np.min(reference)
    reference = reference / np.max(reference)
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    annotation = load_any(annotation_path)
    # TODO: Make sure the annotations are symmetrical

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
    labels_path = DOWNLOAD_DIR_PATH / ALLEN_LABELS_FNAME

    structures = []
    # Open labels file to get structure information
    with open(labels_path, "r") as f:

        labels_data = json.load(f)
        for structure in labels_data["msg"]:
            id = structure["id"]
            name = structure["name"]
            acronym = structure["acronym"]
            structure_id_path = list(
                map(int, structure["structure_id_path"].strip("/").split("/"))
            )
            rgb_triplet = hex_to_rgb(structure["color_hex_triplet"])

            structures.append(
                {
                    "id": id,
                    "name": name,
                    "acronym": acronym,
                    "structure_id_path": structure_id_path,
                    "rgb_triplet": rgb_triplet,
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
    hemispheres_stack = retrieve_hemisphere_map(annotated_volume)
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
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        additional_references=None,
        atlas_packager=ATLAS_PACKAGER,
        scale_meshes=True,
        overwrite=True,
    )
