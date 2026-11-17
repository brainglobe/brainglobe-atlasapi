"""Package the RBT Rabbit Brain Atlas.

This script generates the RBT rabbit brain atlas,
based on data from Zhang et al. It downloads the necessary
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
ATLAS_NAME = "rbt_rabbit"
CITATION = "https://dx.doi.org/10.24272/j.issn.2095-8137.2026.045"
SPECIES = "Oryctolagus cuniculus"
ATLAS_LINK = r"https://www.scidb.cn/en/detail?dataSetId=3339eb8bbb284338ac2f3d9ee3198339"
ORIENTATION = "rpi"
ROOT_ID = 999
RESOLUTIONS = [150, 330]
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URLS = {
    150: "https://china.scidb.cn/download?fileId=a7dbf17665fbca3e765d43a5f770327c&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=37b0310cbe07dde9141341e559cb6da0&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

ANNOTATION_URLS = {
    150: "https://china.scidb.cn/download?fileId=c9f1e8ef690b2e9a865f392a7ff4bc3f&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=aea285f6cedac782311c86cb9194caed&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

T2W_REFERENCE_URLS = {
    150: "https://china.scidb.cn/download?fileId=2aa3a88857252c93bc439caacac1e317&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=ddf241764ff44990829d099e8cde18de&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

LABELS_URL = "https://china.scidb.cn/download?fileId=f8ab5dc5cd6bd65e0d4b9396127a37bf&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791"

REFERENCE_FNAMES = {
    150: "tpl-RBTV1_res-015_T1w.nii.gz",
    330: "tpl-RBTV1_res-033_T1w.nii.gz",
}

T2W_REFERENCE_FNAMES = {
    150: "tpl-RBTV1_res-015_T2w.nii.gz",
    330: "tpl-RBTV1_res-033_T2w.nii.gz",
}

ANNOTATION_FNAMES = "annotation_25.nii"
LABELS_FNAMES = "brainstructures.csv"

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
            "informaticsId": "id",
        },
        inplace=True,
    )
    df["name"] = df["name"].str.strip()
    df["rgb_triplet"] = df[["red", "green", "blue"]].values.tolist()
    df["structure_id_path"] = None

    # Make "Brain" structure a child of root
    df.at[0, "ParentStruct"] = "root"

    root_row = pd.DataFrame(
        [
            {
                "id": ROOT_ID,
                "name": "root",
                "acronym": "root",
                "structure_id_path": [ROOT_ID],
                "rgb_triplet": [255, 255, 255],
            }
        ]
    )
    df = pd.concat([df, root_row], axis=0, ignore_index=True)

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
    )
