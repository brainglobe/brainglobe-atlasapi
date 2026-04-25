"""Package the Allen CCFv2 Developing Mouse Brain Atlas.

This script generates the Allen CCFv2 developing mouse brain atlas, based on data from
the Allen Institute. It downloads the necessary annotation and structure data,
processes it to create an atlas, and then wraps it up into the
BrainGlobe atlas format.
"""

from pathlib import Path

import pandas as pd
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

### Metadata ###
__version__ = 0
ATLAS_NAME = "ccfv2_dev_mouse"
CITATION = "https://doi.org/10.1038/nature05453"
SPECIES = "Mus musculus"
ATLAS_LINK = "https://download.alleninstitute.org/informatics-archive/october-2014/annotation/"
ORIENTATION = "rsa"
ROOT_ID = 15564
RESOLUTION = 25
ATLAS_PACKAGER = "Jung Woo Kim"

SKIP_DOWNLOADS_IF_PRESENT = True

REFERENCE_URL = "https://download.alleninstitute.org/informatics-archive/october-2014/annotation/atlasVolume.zip"
ANNOTATION_URL = "https://download.alleninstitute.org/informatics-archive/october-2014/annotation/P56_DevMouse2012_annotation.zip"
LABELS_URL = "https://download.alleninstitute.org/informatics-archive/october-2014/annotation/structures.csv"
AVERAGED_REFERENCE_URL = "https://download.alleninstitute.org/informatics-archive/october-2014/annotation/averageTemplate.zip"

REFERENCE_FNAME = "atlasVolume.zip"
ANNOTATION_FNAME = "P56_DevMouse2012_annotation.zip"
LABELS_FNAME = "structures.csv"
AVERAGED_REFERENCE_FNAME = "averageTemplate.zip"

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
    """
    Download the necessary resources for the atlas with Pooch.
    """
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
            known_hash="8b19e3435198a9811c53631b9e4a50ccd6a84ef5f10b4e526c5ec7749ae41484",
            path=DOWNLOAD_DIR_PATH,
            fname=REFERENCE_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(annotation_path):
        pooch.retrieve(
            url=ANNOTATION_URL,
            known_hash="69f8ab6139ed0a5eaf94646ee3b0bff812845d3b66322eea6d62238fbd079778",
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )

    if should_fetch(labels_path):
        pooch.retrieve(
            url=LABELS_URL,
            known_hash="9dd4264ff54c44be7fd019fac1d5780c679eeb83b0efd0ba4cc3ac67beed6825",
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
    reference_path = DOWNLOAD_DIR_PATH / "atlasVolume/atlasVolume.mhd"
    ref_image = sitk.ReadImage(reference_path)
    reference = sitk.GetArrayFromImage(ref_image)
    annotation_path = DOWNLOAD_DIR_PATH / "annotation.mhd"
    ann_image = sitk.ReadImage(annotation_path)
    annotation = sitk.GetArrayFromImage(ann_image)
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 0 marks the left hemisphere
    and 1 marks the right.

    Returns
    -------
    None
        None as the atlas is symmetrical.
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
    """
    df = pd.read_csv(DOWNLOAD_DIR_PATH / LABELS_FNAME)
    df.drop(
        columns=[
            "atlas_id",
            "st_level",
            "ontology_id",
            "hemisphere_id",
            "weight",
            "parent_structure_id",
            "depth",
            "graph_id",
            "graph_order",
            "neuro_name_structure_id",
            "neuro_name_structure_id_path",
            "failed",
            "sphinx_id",
            "structure_name_facet",
            "failed_facet",
        ],
        inplace=True,
    )
    df.rename(columns={"color_hex_triplet": "rgb_triplet"}, inplace=True)
    df["rgb_triplet"] = df["rgb_triplet"].apply(lambda x: hex_to_rgb(x))
    df["structure_id_path"] = (
        df["structure_id_path"]
        .str.split("/")
        .map(lambda path: [int(id) for id in path if id])
    )

    # Fix name of root (renamed from "Mus musculus")
    df.loc[df["id"] == 15564, "name"] = "root"
    df.loc[df["id"] == 15564, "acronym"] = "root"

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
    averaged_reference_path = DOWNLOAD_DIR_PATH / AVERAGED_REFERENCE_FNAME

    needs_download = not averaged_reference_path.exists()
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(averaged_reference_path):
        pooch.retrieve(
            url=AVERAGED_REFERENCE_URL,
            known_hash="f3476045fef475ce6c03dea20b38ec6afbb85d9110a6b468ac8259be4f8842ce",
            path=DOWNLOAD_DIR_PATH,
            fname=AVERAGED_REFERENCE_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )
    averaged_reference_path = (
        DOWNLOAD_DIR_PATH / "averageTemplate/atlasVolume.mhd"
    )
    ave_ref_image = sitk.ReadImage(averaged_reference_path)
    averaged_reference = sitk.GetArrayFromImage(ave_ref_image)
    additional_references = {"Averaged reference": averaged_reference}
    return additional_references


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
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
        annotated_volume, structures
    )
    additional_references = retrieve_additional_references()

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
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        additional_references=additional_references,
        atlas_packager=ATLAS_PACKAGER,
        scale_meshes=True,
    )
