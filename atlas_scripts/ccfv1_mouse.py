"""Package the Allen CCFv1 Mouse Brain Atlas.

This script generates the Allen CCFv1 mouse brain atlas,
based on data from the Allen Institute. It downloads the necessary
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

### Metadata ###
__version__ = 0
ATLAS_NAME = "ccfv1_mouse"
CITATION = "https://doi.org/10.1038/nature05453"
SPECIES = "Mus musculus"
ATLAS_LINK = r"https://data-proxy.ebrains.eu/deepslice?prefix=ccfv1%2F"
ORIENTATION = "rpi"
ROOT_ID = 999
RESOLUTION = 25
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://rgw.cscs.ch/ebrains:deepslice/ccfv1/template_25.nii"
ANNOTATION_URL = (
    "https://rgw.cscs.ch/ebrains:deepslice/ccfv1/annotation_25.nii"
)
LABELS_URL = "https://rgw.cscs.ch/ebrains:deepslice/ccfv1/brainstructures.csv"

REFERENCE_FNAME = "template_25.nii"
ANNOTATION_FNAME = "annotation_25.nii"
LABELS_FNAME = "brainstructures.csv"

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
            known_hash="76b036657134a1351a452903644a902d9aac98ec01a5eced5b622de0917ca2e5",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="0827c3bf2e65603acb35e513c5b5c6cc35e6ca7b53e5b3947434866e45421aed",
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="eeacc44153660b46b9e8a606ca6abc094e5b1b37608416e6f850850043761ec7",
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
    df = pd.read_csv(DOWNLOAD_DIR_PATH / LABELS_FNAME)

    df.rename(
        columns={
            "StructureName": "name",
            "Abbreviation": "acronym",
            "StructureId": "id",
        },
        inplace=True,
    )
    df["name"] = df["name"].str.strip()
    df["rgb_triplet"] = df[["red", "green", "blue"]].values.tolist()
    df["structure_id_path"] = None
    df.at[0, "id"] = ROOT_ID

    ancestry = []
    # Loop through structures to ensure all have structure_id_path
    for _, row in df.iterrows():
        if row["acronym"] == "Brain":
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
            "informaticsId",
            "ParentStruct",
        ],
        inplace=True,
    )
    df.at[0, "name"] = "root"
    df.at[0, "acronym"] = "root"

    structures = df.to_dict("records")
    print(structures)

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
    pass


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
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
        scale_meshes=True,
    )
