"""Package the FeBls Mouse Brain Atlas with DSURQE annotations.

This script generates the FeBls female mouse brain atlas with DSURQE
annotations, based on data from NITRC. It downloads the necessary
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
ATLAS_NAME = "febls_dsurqe_mouse"
CITATION = "https://doi.org/10.1038/nature05453"
SPECIES = "Mus musculus"
ATLAS_LINK = r"https://www.nitrc.org/frs/download.php/20440/FeBls_v1.3.zip//?i_agree=1&download_now=1"
ORIENTATION = "lpi"
ROOT_ID = 999
RESOLUTION = 60
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

ATLAS_FNAME = "FeBls_v1.3.zip"

REFERENCE_FNAME = "FeBlsTemplateBrain.nii"
ALLEN_REFERENCE_FNAME = "AMBA2FeBls_Warped.nii"
DSURQE_REFERENCE_FNAME = "DSURQUE2FeBls_Warped.nii"
TMBTA_REFERENCE_FNAME = "TMBTA2FeBls_Warped.nii"

DSURQE_ANNOTATION_FNAME = "DSURQUElabels4FeBls.nii"

DSURQE_LABELS_URL = "https://www.mouseimaging.ca/repo/DSURQE_40micron/Dorr_2008_Steadman_2013_Ullmann_2013_Richards_2011_Qiu_2016_Egan_2015_40micron/mappings/DSURQE_40micron_itksnap_mapping.txt"
DSURQE_LABELS_FNAME = "DSURQE_40micron_itksnap_mapping.txt"
DSURQE_ADDITIONAL_LABELS_FNAME = "DSURQUElabels4FeBls.txt"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"


def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    """Scale image intensities to the full uint16 range."""
    image -= image.min()
    image /= image.max()
    return (image * np.iinfo(np.uint16).max).astype(np.uint16)


def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    atlas_path = DOWNLOAD_DIR_PATH / ATLAS_FNAME
    dsurqe_labels_path = DOWNLOAD_DIR_PATH / DSURQE_LABELS_FNAME

    needs_download = (not atlas_path.exists()) or (
        not dsurqe_labels_path.exists()
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
            known_hash="8ec8c1562a66005ee66526d1e51f4ae0ce12c7a5b4a4e58cdf91db8d864db7ab",
            path=DOWNLOAD_DIR_PATH,
            fname=ATLAS_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(dsurqe_labels_path):
        pooch.retrieve(
            url=DSURQE_LABELS_URL,
            known_hash="c19eaf378a1adebb060b6739ecc817289efedadad34f7d1bd934d809dc46c4a4",
            path=DOWNLOAD_DIR_PATH,
            fname=DSURQE_LABELS_FNAME,
            progressbar=True,
        )


def retrieve_reference_and_annotation(left_hemisphere_ids):
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference_path = DOWNLOAD_DIR_PATH / "v1.3" / "10x" / REFERENCE_FNAME
    annotation_path = (
        DOWNLOAD_DIR_PATH / "v1.3" / "10x" / DSURQE_ANNOTATION_FNAME
    )
    reference = np.asarray(load_any(reference_path), dtype=np.float64)
    reference = normalize_to_uint16(reference)
    annotation = np.asarray(load_any(annotation_path), dtype=np.uint32)
    for k, v in left_hemisphere_ids.items():
        annotation[annotation == k] = v

    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    None
        Returns None, as the atlas is midline-aligned.
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
    list[int]
        A list of structure IDs corresponding to the left hemisphere
    """
    # We use the DSURQE label files in order to get the original RGB values for each region
    labels_path = DOWNLOAD_DIR_PATH / DSURQE_LABELS_FNAME

    # And use the provided .txt labels file to add missing regions
    missing_labels_path = (
        DOWNLOAD_DIR_PATH / "v1.3" / "10x" / DSURQE_ADDITIONAL_LABELS_FNAME
    )

    structures_by_id: dict[int, dict] = {
        ROOT_ID: {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    }

    l2r_hemisphere_ids = {}

    # Open labels file to get structure information
    with open(labels_path, "r") as f:
        # .txt label file format:
        # id R G B A VIS IDX Name

        # Use regex parsing for consistency
        line_re = re.compile(
            r"^\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+).+\"(.+)\"\s*$"
        )
        labels_data = f.read().splitlines()
        for label in labels_data:
            if not label.strip() or label.lstrip().startswith("#"):
                continue
            m = line_re.match(label)

            # Skip malformed lines
            if not m:
                continue

            # Skip background, root and hemisphere specific labels
            id = int(m.group(1))
            if id == 0:
                continue

            # Fix double spaces in names
            name = " ".join(m.group(5).split())

            rgb_colour = [int(m.group(2)), int(m.group(3)), int(m.group(4))]

            structures_by_id[id] = {
                "id": id,
                "name": name,
                "acronym": name,
                "structure_id_path": [ROOT_ID, id],
                "rgb_triplet": rgb_colour,
            }
    with open(missing_labels_path, "r") as f:
        # .txt additional label file format:
        # id    Name

        # Use regex parsing for consistency
        line_re = re.compile(r"(\d+)\s+(.+)\s*$")
        labels_data = f.read().splitlines()
        for label in labels_data:
            if not label.strip() or label.lstrip().startswith("#"):
                continue
            m = line_re.match(label)

            # Skip malformed lines
            if not m:
                continue
            # Skip background, root and hemisphere specific labels
            id = int(m.group(1))
            # Fix double spaces in names
            name = " ".join(m.group(2).split())

            # Add missing right regions
            if all(s.get("name") != name for s in structures_by_id.values()):
                structures_by_id[id] = {
                    "id": id,
                    "name": name,
                    "acronym": name,
                    "structure_id_path": [ROOT_ID, id],
                    "rgb_triplet": [255, 255, 255],
                }

    # Create a mapping of left hemisphere IDs to corresponding right IDs
    for id, structure in structures_by_id.items():
        if structure["name"].endswith("left"):
            right_id = 0
            for s in structures_by_id.values():
                if s["name"] == structure["name"].replace("left", "right"):
                    right_id = s["id"]
                    break
            l2r_hemisphere_ids[id] = right_id

    # Remove all left hemisphere regions from structure list
    for i in l2r_hemisphere_ids.keys():
        structures_by_id.pop(i)

    # Rename regions to remove "right" suffix
    for id, structure in structures_by_id.items():
        structures_by_id[id]["name"] = structure["name"].removesuffix(" right")
        structures_by_id[id]["acronym"] = structure["acronym"].removesuffix(
            " right"
        )

    structures = list(structures_by_id.values())
    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures, l2r_hemisphere_ids


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
    allen_reference_path = (
        DOWNLOAD_DIR_PATH / "v1.3" / "10x" / ALLEN_REFERENCE_FNAME
    )
    allen_reference = np.asarray(
        load_any(allen_reference_path), dtype=np.float64
    )
    allen_reference = normalize_to_uint16(allen_reference)

    dsurqe_reference_path = (
        DOWNLOAD_DIR_PATH / "v1.3" / "10x" / DSURQE_REFERENCE_FNAME
    )
    dsurqe_reference = np.asarray(
        load_any(dsurqe_reference_path), dtype=np.float64
    )
    dsurqe_reference = normalize_to_uint16(dsurqe_reference)

    tmbta_reference_path = (
        DOWNLOAD_DIR_PATH / "v1.3" / "10x" / TMBTA_REFERENCE_FNAME
    )
    tmbta_reference = np.asarray(
        load_any(tmbta_reference_path), dtype=np.float64
    )
    tmbta_reference = normalize_to_uint16(tmbta_reference)

    additional_references = {
        "allen_template_warped": allen_reference,
        "dsurqe_template_warped": dsurqe_reference,
        "tmbta_template_warped": tmbta_reference,
    }
    return additional_references


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
    structures, l2r_hemisphere_ids = retrieve_structure_information()
    reference_volume, annotated_volume = retrieve_reference_and_annotation(
        l2r_hemisphere_ids
    )
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
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
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
        scale_meshes=True,
        overwrite=True,
    )
