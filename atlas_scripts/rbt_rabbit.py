"""Package the RBT Rabbit Brain Atlas.

This script generates the RBT rabbit brain atlas,
based on data from Zhang et al. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

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

### Metadata ###
__version__ = 0
ATLAS_NAME = "rbt_rabbit"
CITATION = "https://dx.doi.org/10.24272/j.issn.2095-8137.2026.045"
SPECIES = "Oryctolagus cuniculus"
ATLAS_LINK = r"https://www.scidb.cn/en/detail?dataSetId=3339eb8bbb284338ac2f3d9ee3198339"
ORIENTATION = "lip"
ROOT_ID = 999
RESOLUTIONS = [150, 330]
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URLS = {
    150: "https://china.scidb.cn/download?fileId=a7dbf17665fbca3e765d43a5f770327c&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=37b0310cbe07dde9141341e559cb6da0&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

REFERENCES_HASHES = {
    150: "d2f2487f8d0ca765b71fa410ee9298cc8247db60ee9b5999de97e917483632c9",
    330: "a7d770357389f6034ac383e26aeb0015e48a4405497c7230059004c9bfb2f6bc",
}

ANNOTATION_URLS = {
    150: "https://china.scidb.cn/download?fileId=c9f1e8ef690b2e9a865f392a7ff4bc3f&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=aea285f6cedac782311c86cb9194caed&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

ANNOTATION_HASHES = {
    150: "d85473fa7d3a7411ca88d426a1c22af4e818bc47425485fba1bcac4567af774e",
    330: "985dfb7a442279b31dd33336700d100c51942869d930e2ae9314251548e880dc",
}

T2W_REFERENCE_URLS = {
    150: "https://china.scidb.cn/download?fileId=2aa3a88857252c93bc439caacac1e317&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
    330: "https://china.scidb.cn/download?fileId=ddf241764ff44990829d099e8cde18de&traceId=3073c18b-cc1b-4c2a-bc13-67a25f824791",
}

T2W_REFERENCE_HASHES = {
    150: "6f7dea4b905ca6d74c204e57827acb37aba58dc811100a50b1f9ba854775a2a0",
    330: "53acf4867fa08a15e770259f5b102a1f73efc8e18efe5489d9597de9fef4c415",
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

ANNOTATION_FNAMES = {
    150: "tpl-RBTV1_res-015_desc-BilateralAtlas_dseg.nii.gz",
    330: "tpl-RBTV1_res-033_desc-BilateralAtlas_dseg.nii.gz",
}

LABELS_FNAME = "tpl-RBTV1_desc-BilateralAtlas_labels.txt"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"


def download_resources(resolution: int):
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[resolution]
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[resolution]
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
            url=REFERENCE_URLS[resolution],
            known_hash=REFERENCES_HASHES.get(resolution),
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAMES[resolution],
            progressbar=True,
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URLS[resolution],
            known_hash=ANNOTATION_HASHES.get(resolution),
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAMES[resolution],
            progressbar=True,
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="b7bf875291ec3723056ed487977da0a9441946fa128c7e5b2746b45a6e130735",
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            progressbar=True,
        )


def retrieve_reference_and_annotation(resolution: int):
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[resolution]
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[resolution]

    reference = np.asarray(load_any(reference_path), dtype=np.float64)
    reference -= np.min(reference)
    reference = reference / np.max(reference)
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    annotation = np.asarray(load_any(annotation_path))
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
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    # .txt label file format:
    # Index R G B A VIS MSH Name

    # Use regex parsing for consistency
    line_re = re.compile(r"\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).+\s+\"(.+)\"")

    # Use the name and acronym used within the label files,
    # and then change them back to "root" later
    structures: list[dict] = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    # Open labels file to get structure information
    with open(labels_path, "r") as f:
        labels_data = f.read().splitlines()
        for key, label in enumerate(labels_data):
            if not label.strip() or label.lstrip().startswith("#"):
                continue
            m = line_re.match(label)

            # Skip malformed lines
            if not m:
                continue

            # Skip background, root and hemisphere specific labels
            id = int(m.group(1))
            name = m.group(5)
            if id == 0:
                continue
            rgb_colour = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

            structures.append(
                {
                    "id": id,
                    "name": name,
                    "acronym": name,
                    "structure_id_path": [ROOT_ID, id],
                    "rgb_triplet": rgb_colour,
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


def retrieve_additional_references(resolution):
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
    t2w_reference_path = DOWNLOAD_DIR_PATH / T2W_REFERENCE_FNAMES[resolution]

    needs_download = not t2w_reference_path.exists()
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(t2w_reference_path):
        pooch.retrieve(
            url=T2W_REFERENCE_URLS[resolution],
            known_hash=T2W_REFERENCE_HASHES.get(resolution),
            path=DOWNLOAD_DIR_PATH,
            fname=T2W_REFERENCE_FNAMES[resolution],
            progressbar=True,
        )

    t2w_reference = np.asarray(load_any(t2w_reference_path), dtype=np.float64)
    t2w_reference -= np.min(t2w_reference)
    t2w_reference = t2w_reference / np.max(t2w_reference)
    t2w_reference = t2w_reference * 65535
    t2w_reference = t2w_reference.astype(np.uint16)

    additional_references = {
        "t2w_reference": t2w_reference,
    }
    return additional_references


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    for resolution in RESOLUTIONS:
        # Fail early if any version of this atlas already exists
        atlas_prefix = atlas_name_from_repr(ATLAS_NAME, resolution)
        existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))

        if existing:
            raise FileExistsError(
                f"Atlas output already exists in {BG_ROOT_DIR}. "
            )
        download_resources(resolution)
        reference_volume, annotated_volume = retrieve_reference_and_annotation(
            resolution
        )
        hemispheres_stack = retrieve_hemisphere_map(annotated_volume)
        structures = retrieve_structure_information()
        meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
            annotated_volume, structures
        )
        additional_references = retrieve_additional_references(resolution)

        output_filename = wrapup_atlas_from_data(
            atlas_name=ATLAS_NAME,
            atlas_minor_version=__version__,
            citation=CITATION,
            atlas_link=ATLAS_LINK,
            species=SPECIES,
            resolution=(resolution,) * 3,
            orientation=ORIENTATION,
            root_id=ROOT_ID,
            reference_stack=reference_volume,
            annotation_stack=annotated_volume,
            structures_list=structures_with_mesh,
            meshes_dict=meshes_dict,
            working_dir=BG_ROOT_DIR,
            hemispheres_stack=hemispheres_stack,
            additional_references=additional_references,
            atlas_packager=ATLAS_PACKAGER,
            scale_meshes=True,
        )
