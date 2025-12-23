"""Package the CArea mouse atlas.

This script downloads a reference template (NIfTI), an annotation volume
(NRRD), and an ITK-SNAP label description file, then wraps them into a
BrainGlobe atlas.
"""

import re
from pathlib import Path

import numpy as np
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

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
ATLAS_NAME = "carea_mouse"

# DOI of the most relevant citable document
# DOI or URL of the most relevant citable document.
# If no paper/DOI exists, use "unpublished".
CITATION = "unpublished"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mus musculus"

# The URL for the data files
ATLAS_LINK = "https://data-proxy.ebrains.eu/api/permalinks/5fbfacf0-ae2d-41fa-96b6-510c4d4bdb3e"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "rsa"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 999

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 25

# --- Script toggles (no CLI args) ---
# If True, do not re-download files that already exist on disk.
SKIP_DOWNLOADS_IF_PRESENT = True
TEMPLATE_URL = "https://data-proxy.ebrains.eu/api/permalinks/5fbfacf0-ae2d-41fa-96b6-510c4d4bdb3e"
ANNOTATION_URL = "https://data-proxy.ebrains.eu/api/permalinks/edf3f84d-221b-49e8-a451-4d597bd00c6b"
LABELS_URL = "https://data-proxy.ebrains.eu/api/permalinks/42203544-7c0c-49a3-b7d6-c2caaef42ff3"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TEMPLATE_FNAME = "population_average_nissl_template_25um.nii.gz"
ANNOTATION_FNAME = "whole_segmentations.nrrd"
LABELS_FNAME = "careas.label"


def download_resources():
    """Download the necessary resources for the atlas (with Pooch)."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    template_path = DOWNLOAD_DIR_PATH / TEMPLATE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME
    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = (
        (not template_path.exists())
        or (not annotation_path.exists())
        or (not labels_path.exists())
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(template_path):
        pooch.retrieve(
            TEMPLATE_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=TEMPLATE_FNAME,
            known_hash="3a240be8d0c6aa8550b15d1cf1fb81b315af842c32a017b940b5e555224509be",
            progressbar=True,
        )
    if should_fetch(annotation_path):
        pooch.retrieve(
            ANNOTATION_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            known_hash="3354d8cb7eafdce85bf795ab4f83698ecf52556df7253dc89055558ee793234d",
            progressbar=True,
        )
    if should_fetch(labels_path):
        pooch.retrieve(
            LABELS_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=LABELS_FNAME,
            known_hash="e5c41b1245a8b5599b3c6974da5c5678349a2d57651b8bcdd756fd5c8c23f875",
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
    template_path = DOWNLOAD_DIR_PATH / TEMPLATE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME

    template_img = sitk.ReadImage(str(template_path))
    annotation_img = sitk.ReadImage(str(annotation_path))

    # SimpleITK returns numpy arrays in (z, y, x). Keep this convention, as it
    # matches the majority of existing atlas scripts in this repo.
    reference = sitk.GetArrayFromImage(template_img)
    annotation = sitk.GetArrayFromImage(annotation_img)
    reference = reference * 65535.0
    annotation = annotation.astype(np.uint16)
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


def retrieve_structure_information(annotation_volume: np.ndarray):
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

    # Filter structures to those actually present.
    present_ids = set(map(int, np.unique(annotation_volume)))



    # ITK-SNAP label file format:
    # IDX   -R-  -G-  -B-  -A--  VIS MSH  "LABEL"
    # Use regex parsing to avoid pandas whitespace/quoting edge-cases.
    line_re = re.compile(
        r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([0-9.]+)\s+(\d+)\s+(\d+)\s+\"(.+?)\"\s*$"
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
    with open(labels_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            if not raw_line.strip() or raw_line.lstrip().startswith("#"):
                continue
            m = line_re.match(raw_line.rstrip("\n"))
            if not m:
                # Skip any malformed line rather than guessing.
                continue

            sid = int(m.group(1))
            r = int(m.group(2))
            g = int(m.group(3))
            b = int(m.group(4))
            name = m.group(8)

            if sid == 0 or name.lower() == "clear label":
                continue
            if sid not in present_ids:
                continue
            # Keep first occurrence if duplicated in label file.
            if sid not in structures_by_id:
                structures_by_id[sid] = {
                    "id": sid,
                    "name": name,
                    "acronym": name,
                    "structure_id_path": [ROOT_ID, sid],
                    "rgb_triplet": [r, g, b],
                }

    # Return root_id alongside the structures so wrapup and mesh generation use
    # the same root.
    structures = list(structures_by_id.values())
    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures


def retrieve_or_construct_meshes(annotated_volume, structures, working_dir):
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
    meshes_dict = construct_meshes_from_annotation(
        save_path=Path(working_dir),
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
    bg_root_dir = BG_ROOT_DIR
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information(annotated_volume)


    meshes_dict = retrieve_or_construct_meshes(
        annotated_volume=annotated_volume,
        structures=structures,
        working_dir=bg_root_dir,
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
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=hemispheres_stack,
        additional_references=additional_references,
    )
