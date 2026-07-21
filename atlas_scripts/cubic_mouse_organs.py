"""Package the CUBIC Mouse Organs Atlases.

This script generates the CUBIC mouse organs atlases,
based on data published by Yoshida et al. It downloads the necessary
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


__version__ = 0


ATLAS_NAME = "cubic_mouse"
CITATION = "https://doi.org/10.1016/j.cell.2025.12.057"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://drive.google.com/drive/folders/11QnUYaTD2blipXxWAsCk-KOxZeccXKWM"
)
ORIENTATION = "psr"

ROOT_ID = 999
RESOLUTION = 50
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

ORGANS = ["heart", "lungs", "kidneys"]

HEART_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1xlz83fEmEUUHEu7O5BtX8dbYg_hzDHFm"
HEART_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=15bncKMJh3LfneEM4LgBm3zBP4eK51RC8"
HEART_LABELS_URL = "https://drive.google.com/uc?export=download&id=1HE-oL6I75Bkz3SkAlcFK0kZoq-lIjnwy"

HEART_REFERENCE_FNAME = "03_Heart_average_atlas_50um.tif"
HEART_ANNOTATION_FNAME = "03_Heart_atlas_segmentation.tif"
HEART_LABELS_FNAME = "heart_ID_list.xlsx"


LUNGS_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1Aceo0-W4Kz2kONEgrMjwHCUp4keL_DPo"

# TODO DOUBLE CHECK IF THIS LINK WORKS ON OTHER MACHINES
LUNGS_ANNOTATION_URL = "https://drive.usercontent.google.com/download?id=1xoNJzxWLsPYlH8ADUqQPP4v_yHhF5iDl&export=download&authuser=0&confirm=t&uuid=e959d586-2aa8-4e1e-b771-d416d149f9e2&at=ABswASZBHAFAenwyMmOefZxqOznY:1784637441803"
LUNGS_LABELS_URL = "https://drive.google.com/uc?export=download&id=11xyChZ8nJrwoNbZaalOq5E2lr1eeUujF"

LUNGS_REFERENCE_FNAME = "04_Lung_average_atlas_100um.tif"
LUNGS_ANNOTATION_FNAME = "04_Lung_atlas_segmentation.tif"
LUNGS_LABELS_FNAME = "lungs_ID_list.xlsx"


KIDNEYS_REFERENCE_URL = "https://drive.google.com/uc?export=download&id=1cZkTosDm1e0DFMdbhJ6FIqK8KhV-wfhW"
KIDNEYS_ANNOTATION_URL = "https://drive.google.com/uc?export=download&id=1pESRqCI1TuDcaMLOaTLjtfneoW_qJuBK"
KIDNEYS_LABELS_URL = "https://drive.google.com/uc?export=download&id=13K8MhUmGUSZfhBwb6x9-iHI1WhkYrxhL"

KIDNEYS_REFERENCE_FNAME = "07_Kidney_average_atlas_50um.tif"
KIDNEYS_ANNOTATION_FNAME = "07_Kidney_segmentation.tif"
KIDNEYS_LABELS_FNAME = "kidneys_ID_list.xlsx"

REFERENCE_URLS = {
    "heart": HEART_REFERENCE_URL,
    "lungs": LUNGS_REFERENCE_URL,
    "kidneys": KIDNEYS_REFERENCE_URL,
}
REFERENCE_FNAMES = {
    "heart": HEART_REFERENCE_FNAME,
    "lungs": LUNGS_REFERENCE_FNAME,
    "kidneys": KIDNEYS_REFERENCE_FNAME,
}

ANNOTATION_URLS = {
    "heart": HEART_ANNOTATION_URL,
    "lungs": LUNGS_ANNOTATION_URL,
    "kidneys": KIDNEYS_ANNOTATION_URL,
}
ANNOTATION_FNAMES = {
    "heart": HEART_ANNOTATION_FNAME,
    "lungs": LUNGS_ANNOTATION_FNAME,
    "kidneys": KIDNEYS_ANNOTATION_FNAME,
}

LABELS_URLS = {
    "heart": HEART_LABELS_URL,
    "lungs": LUNGS_LABELS_URL,
    "kidneys": KIDNEYS_LABELS_URL,
}
LABELS_FNAMES = {
    "heart": HEART_LABELS_FNAME,
    "lungs": LUNGS_LABELS_FNAME,
    "kidneys": KIDNEYS_LABELS_FNAME,
}


HASHES = {
    "heart": {
        "reference": "f641df1cf9453c9b3f9a4a04fba4a644cd23eb25fd91ae2184f8d9c750644090",
        "annotation": "f396902232ee795a411cc73186183c1c0fbb6423871f9d313e881a13c6378114",
        "labels": "d5db0e54404f08761804e23228380ef910629e460a1dcc2d8e521a6ec3c4e8d2", 
    },
    "lungs": {
        "reference": "ab831a5e9764a6317fc7590ebe00a4948b3b8bece9f5eabb2dff6821d2540786",
        "annotation": "b186a4a3075e66198b08782a150e2f1c722882c1cf3eeee4a9032d5b849b7379",
        "labels": "8bc732d34b0c6f329ef0bc35b1bd9bb1f8aaef3302a6ba0641030f29a25f360d", 
    },
    "kidneys": {
        "reference": "46733b505d3f4176399b9a2e65a44e18375889680f157fbaf76495f5c8eab26c",
        "annotation": "8c001e57eaaa7f8ab426906094d66b255db6a196cd5f513ce0dc509fa98469aa",
        "labels": "010f156cb7f472d4db87ff53546e6cf53050eea7a53d459533f8b6c8a6ac886a", 
    },
}


def generate_pseudorandom_rgbs(n_rgbs: int, seed: int = 0):
    """Generate a list of n_rgbs RGB triplets given a seed."""
    rng = np.random.default_rng(seed)
    # n_rgbs RGB values, each channel between 0 and 255 inclusive
    rgb_values = rng.integers(0, 256, size=(n_rgbs, 3)).tolist()
    return rgb_values

def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    for organ in ORGANS:
        reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[organ]
        annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[organ]
        labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAMES[organ]

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
                url=REFERENCE_URLS[organ],
                known_hash=HASHES[organ]["reference"],
                path=DOWNLOAD_DIR_PATH,
                fname=REFERENCE_FNAMES[organ],
                progressbar=True,
            )

        if should_fetch(annotation_path):
            pooch.retrieve(
                url=ANNOTATION_URLS[organ],
                known_hash=HASHES[organ]["annotation"],
                path=DOWNLOAD_DIR_PATH,
                fname=ANNOTATION_FNAMES[organ],
                progressbar=True,
            )

        if should_fetch(labels_path):
            pooch.retrieve(
                url=LABELS_URLS[organ],
                known_hash=HASHES[organ]["labels"],
                path=DOWNLOAD_DIR_PATH,
                fname=LABELS_FNAMES[organ],
                progressbar=True,
            )

def pooch_init(download_dir_path: Path, timepoints: list[str]) -> pooch.Pooch:
    """Initialize Pooch for downloading atlas data.

    Parameters
    ----------
    download_dir_path : Path
        Path to the directory where data will be downloaded.
    timepoints : list[str]
        List of timepoints for which data archives are expected.

    Returns
    -------
    pooch.Pooch
        Initialized Pooch instance.
    """
    utils.check_internet_connection()

    keys = (
        list(REFERENCE_FNAMES.values())
        + list(ANNOTATION_FNAMES.values())
        + list(LABELS_FNAMES.values())
    )
    empty_registry = {key: None for key in keys}

    p = pooch.create(
        path=download_dir_path,
        base_url=ATLAS_LINK,
        registry=empty_registry,
    )

    pooch.make_registry(download_dir_path / "hashes", (ATLAS_NAME + ".txt"))
    # p.load_registry(Path(__file__).parent / "hashes" / (ATLAS_NAME + ".txt"))
    return p


def fetch_organ(pooch_: pooch.Pooch, organ: str):
    """Fetch annotation and reference volumes for a specific organ.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.
    organ : str
        The organ to be fetched.

    Returns
    -------
    tuple
        A tuple containing:
        - annotations (np.ndarray): The annotation volume.
        - reference (np.ndarray): The reference volume (scaled to uint16).

    Raises
    ------
    AssertionError
        If an unknown organ is provided.
    """
    assert organ in ORGANS, f"Unknown organ: '{organ}'"

    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[organ]
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[organ]

    needs_download = (not reference_path.exists()) or (
        not annotation_path.exists()
    )
    if needs_download:
        utils.check_internet_connection()

    fetched_reference = pooch_.fetch(
        REFERENCE_FNAMES[organ],
        progressbar=True,
    )

    fetched_annotation = pooch_.fetch(
        ANNOTATION_FNAMES[organ],
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


def fetch_ontology(pooch_: pooch.Pooch, organ: str):
    """Fetch and parse the ontology (structure tree) from the labels file,
    and return a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    Parameters
    ----------
    pooch_ : pooch.Pooch
        The initialized Pooch instance.
    organ : str
        The organ to be fetched.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a brain
        structure with its properties (id, acronym, name, structure_id_path, RGB color).
    """
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAMES[organ]
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAMES[organ]

    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAMES[organ]
    
    needs_download = not labels_path.exists()
    if needs_download:
        utils.check_internet_connection()
    
    fetched_labels = pooch_.fetch(
        LABELS_FNAMES[organ],
        progressbar=True,
    )

    labels_df = pd.read_excel(
        fetched_labels, engine="openpyxl"
    )
    labels_df = labels_df.iloc[1].dropna()
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    rgbs = generate_pseudorandom_rgbs(labels_df.shape[0], 42)

    for id, name in labels_df.items():
        if id == "ID" or name == "none":
            continue
        structures.append(
            {
                "id": int(id),
                "name": name,
                "acronym": name,
                "structure_id_path": [999, int(id)],
                "rgb_triplet": rgbs[int(id)],
            }
        )

    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures


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
    reference = load_any(reference_path)
    ref_min = np.min(reference)
    ref_max = np.max(reference)
    reference = (reference - ref_min) / (ref_max - ref_min) * 65535
    reference = reference.astype(np.uint16)
    annotation = load_any(annotation_path)
    return reference, annotation


def retrieve_hemisphere_map():
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
    return None


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
    for organ in ORGANS:
        atlas_name = f"{ATLAS_NAME}_{organ}"
        print("\nPackaging atlas for:", atlas_name)
        reference_volume, annotated_volume = fetch_organ(good_dog, organ)
        structures = fetch_ontology(good_dog, organ)
        hemispheres_stack = retrieve_hemisphere_map(annotated_volume, organ)
        meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
            annotated_volume, structures
        )

        output_filename = wrapup_atlas_from_data(
            atlas_name=atlas_name,
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
            working_dir=bg_root_dir,
            hemispheres_stack=hemispheres_stack,
            cleanup_files=False,
            compress=True,
            scale_meshes=True,
            atlas_packager=ATLAS_PACKAGER,
        )