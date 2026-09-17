"""Package the LAMBADA Developmental Mouse Brain Atlases.

This script generates the LAMBADA dev mouse brain atlases,
based on data published by de Launoit et al. It downloads the necessary
annotation and structure data, processes it to create an atlas,
and then wraps it up into the BrainGlobe atlas format.
"""

import re
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

__version__ = 0

ATLAS_NAME = "lambada_dev_mouse"
CITATION = "https://doi.org/10.1016/j.cell.2026.03.013"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://lambada.icm-institute.org/"
)
ORIENTATION = "lpi"

ROOT_ID = 999
RESOLUTION = 25
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TIMEPOINTS = ["3", "5", "7", "9", "12", "14", "21"]

# LABELS_FNAME = "Developmental_labels_lookup.txt"

DOWNLOAD_ROOT = "https://lambada.icm-institute.org/datalayer/"

REFERENCE_SUFFIXES = {
    "3": "16/LAMBADA_25um_reference_P3_v1.0.nii.gz",
    "5": "8/LAMBADA_25um_reference_P5_v1.0.nii.gz",
    "7": "26/LAMBADA_25um_reference_P7_v1.0.nii.gz",
    "9": "31/LAMBADA_25um_reference_P9_v1.0.nii.gz",
    "12": "11/LAMBADA_25um_reference_P12_v1.0.nii.gz",
    "14": "17/LAMBADA_25um_reference_P14_v1.0.nii.gz",
    "21": "54/LAMBADA_25um_reference_P21_v1.0.nii.gz",
}

ANNOTATION_SUFFIXES = {
    "3": "18/LAMBADA_25um_annotation_P3_v1.0.nii.gz",
    "5": "22/LAMBADA_25um_annotation_P5_v1.0.nii.gz",
    "7": "27/LAMBADA_25um_annotation_P7_v1.0.nii.gz",
    "9": "32/LAMBADA_25um_annotation_P9_v1.0.nii.gz",
    "12": "21/LAMBADA_25um_annotation_P12_v1.0.nii.gz",
    "14": "47/LAMBADA_25um_annotation_P14_v1.0.nii.gz",
    "21": "55/LAMBADA_25um_annotation_P21_v1.0.nii.gz",
}

VASC_BRANCH_SUFFIXES = {
    "3": "10/LAMBADA_25um_vasc_branch_points_P3_v1.0.nii.gz",
    "5": "24/LAMBADA_25um_vasc_branch_points_P5_v1.0.nii.gz",
    "7": "29/LAMBADA_25um_vasc_branch_points_P7_v1.0.nii.gz",
    "9": "34/LAMBADA_25um_vasc_branch_points_P9_v1.0.nii.gz",
    "12": "9/LAMBADA_25um_vasc_branch_points_P12_v1.0.nii.gz",
    "14": "50/LAMBADA_25um_vasc_branch_points_P14_v1.0.nii.gz",
    "21": "40/LAMBADA_25um_vasc_branch_points_P21_v1.0.nii.gz",
}

VASC_INTERR_SUFFIXES = {
    "3": "20/LAMBADA_25um_vasc_interrupted_vessels_P3_v1.0.nii.gz",
    "5": "25/LAMBADA_25um_vasc_interrupted_vessels_P5_v1.0.nii.gz",
    "7": "30/LAMBADA_25um_vasc_interrupted_vessels_P7_v1.0.nii.gz",
    "9": "37/LAMBADA_25um_vasc_interrupted_vessels_P9_v1.0.nii.gz",
    "12": "4/LAMBADA_25um_vasc_interrupted_vessels_P12_v1.0.nii.gz",
    "14": "46/LAMBADA_25um_vasc_interrupted_vessels_P14_v1.0.nii.gz",
    "21": "44/LAMBADA_25um_vasc_interrupted_vessels_P21_v1.0.nii.gz",
}

CCK_SUFFIXES = {
    "5": "65/LAMBADA_25um_mRNA_cck_P5_v1.0.nii.gz",
    "7": "5/LAMBADA_25um_mRNA_cck_P7_v1.0.nii.gz",
    "12": "7/LAMBADA_25um_mRNA_cck_P12_v1.0.nii.gz",
    "14": "41/LAMBADA_25um_mRNA_cck_P14_v1.0.nii.gz",
}

VEGFA_SUFFIXES = {
    "5": "52/LAMBADA_25um_mRNA_vegfa_P5_v1.0.nii.gz",
    "7": "13/LAMBADA_25um_mRNA_vegfa_P7_v1.0.nii.gz",
    "9": "39/LAMBADA_25um_mRNA_vegfa_P9_v1.0.nii.gz",
    "12": "15/LAMBADA_25um_mRNA_vegfa_P12_v1.0.nii.gz",
    "14": "62/LAMBADA_25um_mRNA_vegfa_P14_v1.0.nii.gz",
}

MBP_SUFFIXES = {
    "7": "14/LAMBADA_25um_protein_mbp_P7_v1.0.nii.gz",
    "9": "38/LAMBADA_25um_protein_mbp_P9_v1.0.nii.gz",
    "12": "6/LAMBADA_25um_protein_mbp_P12_v1.0.nii.gz",
    "14": "53/LAMBADA_25um_protein_mbp_P14_v1.0.nii.gz",
    "21": "57/LAMBADA_25um_protein_mbp_P21_v1.0.nii.gz",
}

GAP43_SUFFIXES = {
    "9": "35/LAMBADA_25um_mRNA_gap43_P9_v1.0.nii.gz",
    "12": "3/LAMBADA_25um_mRNA_gap43_P12_v1.0.nii.gz",
}

ALDOC_SUFFIXES = {
    "9": "36/LAMBADA_25um_mRNA_aldoc_P9_v1.0.nii.gz",
    "12": "2/LAMBADA_25um_mRNA_aldoc_P12_v1.0.nii.gz",
}

NRGN_SUFFIXES = {
    "14": "43/LAMBADA_25um_mRNA_nrgn_P14_v1.0.nii.gz",
}

ETV5_SUFFIXES = {
    "14": "42/LAMBADA_25um_mRNA_etv5_P14_v1.0.nii.gz",
}

ERG_SUFFIXES = {
    "5": "48/LAMBADA_25um_erg_P5_v1.0.nii.gz",
    "7": "51/LAMBADA_25um_erg_P7_v1.0.nii.gz",
    "9": "63/LAMBADA_25um_erg_P9_v1.0.nii.gz",
    "12": "64/LAMBADA_25um_erg_P12_v1.0.nii.gz",
    "14": "66/LAMBADA_25um_erg_P14_v1.0.nii.gz",
    "21": "45/LAMBADA_25um_erg_P21_v1.0.nii.gz",
}



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
        # + [LABELS_FNAME]
    )
    empty_registry = {key: None for key in keys}

    p = pooch.create(
        path=download_dir_path,
        base_url=DOWNLOAD_ROOT,
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
        The age timepoint (e.g., "3", "14").

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
        structure with its properties (id, acronym, name, structure_id_path,
        RGB color).
    """
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    labels_path = DOWNLOAD_DIR_PATH / LABELS_FNAME

    needs_download = not labels_path.exists()
    if needs_download:
        utils.check_internet_connection()
    
    spacecache = ReferenceSpaceCache(
    manifest=download_dir_path / "manifest.json",
    # downloaded files are stored relative to here
    resolution=resolution,
    reference_space_key="annotation/ccf_2017",
    # use the latest version of the CCF
    )
    
    # Download structures tree:
    ######################################
    oapi = OntologiesApi()  # ontologies

    # Find id of set of regions with mesh:
    select_set = (
        "Structures whose surfaces are represented by a precomputed mesh"
    )

    mesh_set_ids = [
        s["id"]
        for s in oapi.get_structure_sets()
        if s["description"] == select_set
    ]

    structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)

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

    odin = pooch_init(DOWNLOAD_DIR_PATH)
    structures = fetch_ontology(odin)
    for age in TIMEPOINTS:
        atlas_name = f"{ATLAS_NAME}_p{age}"
        print("\nPackaging atlas for:", atlas_name)
        reference_volume, annotated_volume = fetch_animal(odin, age)
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
