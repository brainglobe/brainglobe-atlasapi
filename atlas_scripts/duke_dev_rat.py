"""Package the Duke Developmental Rat Brain Atlases.

This script generates the Duke dev rat brain atlases,
based on data published by Calabrese et al. It downloads the necessary
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

__version__ = 0

ATLAS_NAME = "duke_dev_rat"
CITATION = "https://doi.org/10.1016/j.neuroimage.2013.01.017"
SPECIES = "Rattus norvegicus"
ATLAS_LINK = (
    "https://data-proxy.ebrains.eu/api/v1/buckets/duke-dev-rat-materials/"
)
ORIENTATION = "pri"

ROOT_ID = 999
RESOLUTION = 25
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TIMEPOINTS = ["00", "02", "04", "08", "12", "18", "24", "40", "80"]

REFERENCE_FNAMES = {age: "p" + age + "_average_gre.nii" for age in TIMEPOINTS}
ANNOTATION_FNAMES = {
    age: "pnd" + age + "_average_labels.nii" for age in TIMEPOINTS
}
ANNOTATION_FNAMES["12"] = "pnd12_average_labels_fix.nii"
LABELS_FNAME = "Developmental_labels_lookup.txt"


def pooch_init(download_dir_path: Path) -> pooch.Pooch:
    """Initialize Pooch for downloading atlas data.

    Parameters
    ----------
    download_dir_path : Path
        Path to the directory where data will be downloaded.

    Returns
    -------
    pooch.Pooch
        Initialized Pooch instance.
    """
    keys = (
        list(REFERENCE_FNAMES.values())
        + list(ANNOTATION_FNAMES.values())
        + [LABELS_FNAME]
    )
    empty_registry = {key: None for key in keys}

    p = pooch.create(
        path=download_dir_path,
        base_url=ATLAS_LINK,
        registry=empty_registry,
    )

    p.load_registry(Path(__file__).parent / "hashes" / (ATLAS_NAME + ".txt"))
    return p


def fetch_animal(pooch_: pooch.Pooch, age: str):
    """Fetch reference and annotation volumes for a specific age.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.
    age : str
        The age timepoint (e.g., "00", "24").

    Returns
    -------
    tuple
        A tuple containing:
        - reference (np.ndarray): The reference volume (scaled to uint16).
        - annotations (np.ndarray): The annotation volume.

    Raises
    ------
    AssertionError
        If an unknown age timepoint is provided.
    """
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"

    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[age]
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[age]

    needs_download = (not reference_path.exists()) or (
        not annotation_path.exists()
    )
    if needs_download:
        utils.check_internet_connection()

    fetched_reference = pooch_.fetch(
        REFERENCE_FNAMES[age],
        progressbar=True,
    )

    fetched_annotation = pooch_.fetch(
        ANNOTATION_FNAMES[age],
        progressbar=True,
    )

    reference_volume = load_any(fetched_reference, as_numpy=True)
    annotation_volume = load_any(fetched_annotation, as_numpy=True)
    dmin = np.min(reference_volume)
    dmax = np.max(reference_volume)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference_volume = (reference_volume - dmin) * dscale
    reference_volume = reference_volume.astype(np.uint16)
    return reference_volume, annotation_volume


def fetch_ontology(pooch_: pooch.Pooch):
    """Fetch and parse the ontology (structure tree) from the labels file,
    and return a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties (id, acronym, name, structure_id_path, RGB color).
    """
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = not labels_path.exists()
    if needs_download:
        utils.check_internet_connection()

    labels_path = pooch_.fetch(LABELS_FNAME, progressbar=True)

    # .txt label file format:
    # Index Name R G B A

    # Use regex parsing for consistency
    line_re = re.compile(r"^(\d+)\s+(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$")

    # Use the name and acronym used within the label files,
    # and then change them back to "root" later
    structures = [
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
            name = m.group(2).replace("_", " ")
            if id == 0:
                continue
            rgb_colour = [int(m.group(3)), int(m.group(4)), int(m.group(5))]

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


def retrieve_hemisphere_map(annotation_volume: np.ndarray, age: str):
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
    # Atlas is in PRI orientation, slice from middle
    hemispheres_map = np.full(annotation_volume.shape, 2, dtype=int)
    hemispheres_map[:, hemispheres_map.shape[1] // 2 :, :] = 1

    # Fix midline misalignment for p24
    if age == "24":
        hemispheres_map[:, 325:, :] = 1
        hemispheres_map[:, :325, :] = 2

    # Fix midline misalignment for p40
    if age == "40":
        hemispheres_map[:, 330:, :] = 1
        hemispheres_map[:, :330, :] = 2

    return hemispheres_map


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

    # Fail when any timepoints already exist to avoid overwriting
    for age in TIMEPOINTS:
        atlas_prefix = atlas_name_from_repr(
            ATLAS_NAME + f"_p{age}", RESOLUTION
        )
        existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))
        if existing:
            raise FileExistsError(
                f"{atlas_prefix} output already exists in {BG_ROOT_DIR}. "
            )

    good_dog = pooch_init(DOWNLOAD_DIR_PATH)
    structures = fetch_ontology(good_dog)
    for age in TIMEPOINTS:
        atlas_name = f"{ATLAS_NAME}_p{age}"
        print("\nPackaging atlas for:", atlas_name)
        reference_volume, annotated_volume = fetch_animal(good_dog, age)
        hemispheres_stack = retrieve_hemisphere_map(annotated_volume, age)
        meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
            annotated_volume, structures
        )

        output_filename = wrapup_atlas_from_data(
            atlas_name=f"{ATLAS_NAME}_p{age}",
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
            scale_meshes=True,
            atlas_packager=ATLAS_PACKAGER,
        )
