"""Package the WHS_SD_Rat rat brain atlas.

Downloads the necessary files from the NITRC repository, processes them,
creates meshes, and packages the atlas in the BrainGlobe format.

Downloads zip archives from the NITRC repository and extracts zip files
containing the MBAT_WHS_SD_rat_atlas data. Uses the original Waxholm Space
(WHS) SD rat atlas files.
"""

__version__ = "2"

import json
from pathlib import Path

import numpy as np
import pooch
import xmltodict
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import (
    wrapup_atlas_from_data,
)
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "whs_sd_rat"
SPECIES = "Rattus norvegicus"
ATLAS_LINK = "https://www.nitrc.org/projects/whs-sd-atlas"
CITATION = "Kleven et al 2023, https://doi.org/10.1038/s41592-023-02034-3"

RESOLUTION = (39, 39, 39)
ROOT_ID = 10000
ORIENTATION = "lpi"

REFERENCE_FILENAME = "MBAT_WHS_SD_rat_atlas_v4_pack.zip"
REFERENCE_URL = (
    "https://www.nitrc.org/frs/download.php/12263/"
    "MBAT_WHS_SD_rat_atlas_v4_pack.zip"
)
ANNOTATION_FILENAME = "MBAT_WHS_SD_rat_atlas_v4.01.zip"
ANNOTATION_URL = (
    "https://www.nitrc.org/frs/download.php/13400/"
    "MBAT_WHS_SD_rat_atlas_v4.01.zip//?i_agree=1&download_now=1"
)


ATLAS_PACKAGER = (
    "Harry Carey, University of Oslo, Norway, harry.carey@medisin.uio.no"
)


def download_waxholm_atlas_files(
    download_dir_path, atlas_file_url, ATLAS_NAME
):
    """Download and extract atlas files from a zip archive.

    Downloads zip archives from the Waxholm/NITRC repository and extracts them.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where the atlas files will be downloaded and extracted.
    atlas_file_url : str
        The URL of the atlas zip file.
    ATLAS_NAME : str
        The name of the atlas, used for naming the downloaded file.

    Returns
    -------
    pathlib.Path
        The download directory path where files were extracted.

    Raises
    ------
    requests.exceptions.ConnectionError
        If there is no internet connection.
    """
    download_name = ATLAS_NAME + "_atlas.zip"

    pooch.retrieve(
        url=atlas_file_url,
        known_hash=None,
        path=download_dir_path,
        fname=download_name,
        progressbar=True,
        processor=pooch.Unzip(extract_dir=""),
    )

    return download_dir_path


def parse_structures_xml(root, path=None, structures=None):
    """Recursively parse the XML structure definition.

    Parameters
    ----------
    root : dict
        The current root element of the XML structure.
    path : list, optional
        The current path of structure IDs, by default None.
    structures : list, optional
        A list to accumulate parsed structures, by default None.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its name, acronym, ID, path, and RGB triplet.
    """
    structures = structures or []
    path = path or []

    rgb_triplet = [int(root["@color"][i : i + 2], 16) for i in (1, 3, 5)]
    id = int(root["@id"])
    struct = {
        "name": root["@name"],
        "acronym": root["@abbreviation"],
        "id": int(root["@id"]),
        "structure_id_path": path + [id],
        "rgb_triplet": rgb_triplet,
    }
    structures.append(struct)

    if "label" in root:
        if isinstance(root["label"], list):
            for label in root["label"]:
                parse_structures_xml(
                    label, path=path + [id], structures=structures
                )
        else:
            parse_structures_xml(
                root["label"], path=path + [id], structures=structures
            )

    return structures


def parse_structures(structures_file: Path):
    """Parse the structures XML file to extract atlas region metadata.

    Parameters
    ----------
    structures_file : pathlib.Path
        The path to the .ilf XML file containing structure definitions.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its name, acronym, ID, path, and RGB triplet.
    """
    parsed_xml = xmltodict.parse(structures_file.read_text())
    root = parsed_xml["milf"]["structure"]
    root["@abbreviation"] = "root"
    root["@color"] = "#ffffff"
    root["@id"] = "10000"
    root["@name"] = "Root"

    structures = parse_structures_xml(root)
    return structures


def create_meshes(download_dir_path, annotated_volume, structures):
    """Generate meshes for each brain region.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where meshes will be saved.
    annotated_volume : numpy.ndarray
        The 3D numpy array representing the annotated brain volume.
    structures : list
        A list of dictionaries, where each dictionary represents a structure.

    Returns
    -------
    tuple
        - dict: A dictionary where keys are structure IDs and values are paths
          to their corresponding .obj mesh files.
        - list: A filtered list of structures that successfully had a mesh
          created and verified.
    """
    meshes_dict = construct_meshes_from_annotation(
        save_path=download_dir_path,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=False,
    )

    # Filter structures to only those with meshes
    structures_with_mesh = [s for s in structures if s["id"] in meshes_dict]

    return meshes_dict, structures_with_mesh


def create_atlas(working_dir):
    """Package the WHS_SD_Rat atlas.

    Downloads the necessary raw data from the NITRC repository, processes the
    annotation and reference volumes, creates meshes for each brain region, and
    wraps up the data into the BrainGlobe atlas format.

    Downloads and extracts zip archives from the Waxholm/NITRC repository
    containing the MBAT_WHS_SD_rat_atlas data.

    Parameters
    ----------
    working_dir : pathlib.Path
        The directory where temporary and final atlas files will be stored.

    Returns
    -------
    pathlib.Path
        The path to the generated BrainGlobe atlas zip file.

    Raises
    ------
    AssertionError
        If `ORIENTATION` or `RESOLUTION` are not correctly defined, or
        if `REFERENCE_URL` is missing.
    """
    assert len(ORIENTATION) == 3, (
        "Orientation is not 3 characters, Got " + ORIENTATION
    )
    assert len(RESOLUTION) == 3, "Resolution is not correct, Got " + str(
        RESOLUTION
    )
    assert (
        REFERENCE_URL
    ), "No download link provided for atlas in REFERENCE_URL"

    # Generated atlas path:
    working_dir.mkdir(exist_ok=True, parents=True)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # Download and extract zip archives from Waxholm/NITRC
    download_waxholm_atlas_files(download_dir_path, REFERENCE_URL, ATLAS_NAME)
    atlas_files_dir = download_dir_path / "MBAT_WHS_SD_rat_atlas_v4_pack/Data"

    download_waxholm_atlas_files(
        download_dir_path, ANNOTATION_URL, ATLAS_NAME + "_annotation"
    )
    annotation_files_dir = (
        download_dir_path / "MBAT_WHS_SD_rat_atlas_v4.01/Data"
    )

    # Parse structure metadata
    structures = parse_structures(
        annotation_files_dir / "WHS_SD_rat_atlas_v4.01_labels.ilf"
    )

    # Load files
    annotation_file = annotation_files_dir / "WHS_SD_rat_atlas_v4.01.nii.gz"
    annotation_stack = load_any(annotation_file, as_numpy=True).astype(
        np.int64
    )
    reference_file = atlas_files_dir / "WHS_SD_rat_T2star_v1.01.nii.gz"
    reference_stack = load_any(reference_file, as_numpy=True)

    # Remove structures with missing annotations
    tree = get_structures_tree(structures)
    labels = set(np.unique(annotation_stack).astype(np.int32))
    existing_structures = []
    for structure in structures:
        stree = tree.subtree(structure["id"])
        ids = set(stree.nodes.keys())
        matched_labels = ids & labels
        if matched_labels:
            existing_structures.append(structure)
        else:
            node = tree.nodes[structure["id"]]
            print(
                f"{node.tag} not found in annotation volume, "
                "removing from list of structures..."
            )
    structures = existing_structures

    # Clean junk from reference file
    reference_stack *= annotation_stack > 0

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # Create meshes:
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dict, structures_with_mesh = create_meshes(
        download_dir_path, annotation_stack, structures
    )

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_stack,
        annotation_stack=annotation_stack,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_available=False,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
