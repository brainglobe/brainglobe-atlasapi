"""Package the Duke Mouse Brain Atlas for BrainGlobe."""

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

__version__ = 0
ATLAS_NAME = "duke_mouse"
CITATION = "Mansour et al. 2025, https://doi.org/10.1126/sciadv.adq8089"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://civmimagespace.civm.duhs.duke.edu/"
    "tp_item_detail.php/view/item_number=DMBA/set_id=315"
)
ATLAS_PACKAGER = "Amirreza Bahramani"

ORIENTATION = "ipr"
ROOT_ID = 997
RESOLUTION = 15

SOURCE_DATA_DIR = (
    Path.home() / "brainglobe_workingdir" / ATLAS_NAME / "source_data"
)
BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_BASE_URL = "https://d3mof5o.s3.amazonaws.com/"

REFERENCE_FILES = {
    "md": ("DMBA_md", "DMBA_N06_md_M4D"),
    "ad": ("DMBA_ad", "DMBA_N01_ad_M4D"),
    "dwi": ("DMBA_dwi", "DMBA_N02_dwi_M4D"),
    "nqa": ("DMBA_nqa", "DMBA_N03_nqa_M4D"),
    "rd": ("DMBA_rd", "DMBA_N05_rd_M4D"),
    "fa": ("DMBA_fa", "DMBA_N09_fa_M4D"),
    "m0": ("DMBA_m0", "DMBA_N11_m0_M4D"),
    "m1": ("DMBA_m1", "DMBA_N12_m1_M4D"),
    "m2": ("DMBA_m2", "DMBA_N13_m2_M4D"),
    "m3": ("DMBA_m3", "DMBA_N14_m3_M4D"),
    "iso": ("DMBA_iso", "DMBA_N17_iso_M4D"),
    "mgre-unmasked": (
        "DMBA_mGRE-unmasked",
        "DMBA_N18_mGRE-unmasked_M4D",
    ),
}

ANNOTATION_STEM = "DMBA_RCCF_labels_M4D"
ANNOTATION_PATH = SOURCE_DATA_DIR / f"{ANNOTATION_STEM}.nhdr"
STRUCTURES_ZIP_PATH = SOURCE_DATA_DIR / f"{ANNOTATION_STEM}.zip"
LABEL_FILE = "DMBA_RCCF.label"

MESH_NUM_THREADS = 6


def download_resources():
    """Download the necessary resources for the atlas using Pooch."""
    downloads = [
        (
            f"{DOWNLOAD_BASE_URL}{ANNOTATION_STEM}.zip",
            STRUCTURES_ZIP_PATH,
        )
    ]

    image_files = [("", ANNOTATION_STEM), *REFERENCE_FILES.values()]
    for directory, filename in image_files:
        for suffix in (".nhdr", ".raw"):
            downloads.append(
                (
                    f"{DOWNLOAD_BASE_URL}{filename}{suffix}",
                    SOURCE_DATA_DIR / directory / f"{filename}{suffix}",
                )
            )

    missing_paths = [
        destination for _, destination in downloads if not destination.exists()
    ]
    if not missing_paths:
        print("All DMBA source files are already present.")
        return

    utils.check_internet_connection()
    for url, destination in downloads:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"Already present: {destination}")
            continue

        pooch.retrieve(
            url=url,
            known_hash=None,
            path=destination.parent,
            fname=destination.name,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """Retrieve the reference and annotation volumes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        The main reference volume and the lateral annotation volume.
    """
    directory, filename = REFERENCE_FILES["mgre-unmasked"]
    path = SOURCE_DATA_DIR / directory / f"{filename}.nhdr"
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    reference = reference.astype(np.float32, copy=False)
    reference -= reference.min()
    reference *= np.iinfo(np.uint16).max / reference.max()
    reference = np.rint(reference).astype(np.uint16)

    annotation = sitk.GetArrayFromImage(sitk.ReadImage(str(ANNOTATION_PATH)))
    annotation = annotation.astype(np.uint32, copy=False)
    return reference, annotation


def retrieve_hemisphere_map(annotation):
    """Retrieve a hemisphere map for the atlas.

    Parameters
    ----------
    annotation : numpy.ndarray
        The lateral annotation volume.

    Returns
    -------
    numpy.ndarray
        A map where 1 marks left and 2 marks right labeled voxels.
    """
    hemispheres = np.zeros(annotation.shape, dtype=np.uint8)
    hemispheres[(annotation > 0) & (annotation < 1000)] = 1
    hemispheres[annotation >= 1000] = 2
    return hemispheres


def retrieve_structure_information(annotation):
    """Return information about structures present in the annotation.

    Parameters
    ----------
    annotation : numpy.ndarray
        The bilateral annotation volume.

    Returns
    -------
    list[dict]
        Structure IDs, names, acronyms, paths, and RGB colors.
    """
    with zipfile.ZipFile(STRUCTURES_ZIP_PATH) as label_zip:
        labels = pd.read_csv(
            label_zip.open(LABEL_FILE),
            sep=r"\s+",
            header=None,
            usecols=[0, 1, 2, 3, 7],
            names=["id", "r", "g", "b", "name"],
        )

    labels = labels[
        labels["id"].isin(np.unique(annotation)) & (labels["id"] != 0)
    ]
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]
    for row in labels.itertuples(index=False):
        acronym_and_name = row.name.removesuffix("_left")
        (
            acronym,
            label_name,
        ) = acronym_and_name.split("__", maxsplit=1)
        structures.append(
            {
                "id": row.id,
                "name": label_name,
                "acronym": acronym,
                "structure_id_path": [ROOT_ID, row.id],
                "rgb_triplet": [row.r, row.g, row.b],
            }
        )

    return structures


def retrieve_or_construct_meshes(annotated_volume, structures):
    """Construct structure meshes and return their file paths.

    Parameters
    ----------
    annotated_volume : numpy.ndarray
        The bilateral annotation volume.
    structures : list[dict]
        Structure information for the atlas.

    Returns
    -------
    dict
        A mapping from structure IDs to mesh file paths.
    """
    meshes_dict = construct_meshes_from_annotation(
        save_path=BG_ROOT_DIR,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=1,
        decimate_fraction=0.2,
        smooth=False,
        parallel=True,
        num_threads=MESH_NUM_THREADS,
        verbosity=0,
    )
    return meshes_dict


def retrieve_additional_references():
    """Retrieve the additional reference images.

    Returns
    -------
    dict
        A mapping from contrast names to image stacks.
    """
    references = {}
    for name, (directory, filename) in REFERENCE_FILES.items():
        if name == "mgre-unmasked":
            continue
        path = SOURCE_DATA_DIR / directory / f"{filename}.nhdr"
        reference = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        reference = reference.astype(np.float32, copy=False)
        reference -= reference.min()
        reference *= np.iinfo(np.uint16).max / reference.max()
        references[name] = np.rint(reference).astype(np.uint16)
    return references


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

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
    annotated_volume[annotated_volume >= 1000] -= 1000
    structures = retrieve_structure_information(annotated_volume)
    print(structures)
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)

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
        working_dir=BG_ROOT_DIR,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        scale_meshes=True,
        additional_references=additional_references,
        overwrite=True
    )
    print(f"Atlas saved to {output_filename}")
