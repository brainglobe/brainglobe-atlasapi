"""Package the WHS_SD_Rat atlas using the SWC female rat template.

This script builds a Waxholm Space (WHS) SD rat atlas using the SWC female rat
template as the reference image and Waxholm annotations (filtered to only
include regions that are present in the template).

Downloads individual files (.nii.gz, .ilf) from the SWC template repository on
GIN (gin.g-node.org/BrainGlobe/swc_rat_atlas_materials). Files are downloaded
directly without extraction. Uses pre-processed annotation and reference files
aligned to Waxholm space.

The script processes the files, creates meshes, and packages the atlas in the
BrainGlobe format.
"""

__version__ = "0"

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

ATLAS_NAME = "whs_sd_swc_female_rat"
SPECIES = "Rattus norvegicus"
ATLAS_LINK = "https://www.nitrc.org/projects/whs-sd-atlas"
CITATION = "Kleven et al 2023, https://doi.org/10.1038/s41592-023-02034-3"

RESOLUTION = (39, 39, 39)
ROOT_ID = 10000
ORIENTATION = "pir"

REFERENCE_FILENAME = "T2W_warped.nii.gz"
REFERENCE_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/packaging/Waxholm_space_39um/T2W_warped.nii.gz"
)
ANNOTATION_FILENAME = "WHS_SD_annotation_waxholm_space_cleaned.nii.gz"
ANNOTATION_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/packaging/Waxholm_space_39um/"
    "WHS_SD_annotation_waxholm_space_cleaned.nii.gz"
)
STRUCTURES_ILF_FILENAME = "WHS_SD_rat_atlas_v4.01_labels.ilf"
STRUCTURES_ILF_URL = (
    "https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    "raw/master/packaging/WHS_SD_rat_atlas_v4.01_labels.ilf"
)

# Known hashes for downloaded files
REFERENCE_HASH = (
    "926e6e27e18c5d5edd7787299429b36137e1d0070b4f67e1e2982c8398181940"
)
ANNOTATION_HASH = (
    "5ebc835c3220a84d061b03fb5d49d7c33f74348e2fb13d4d34668272d898d4f1"
)
STRUCTURES_ILF_HASH = None

ATLAS_PACKAGER = "Viktor Plattner, v.plattner@ucl.ac.uk"


def download_swc_template_file(
    download_dir_path, file_url, filename, known_hash=None
):
    """Download a single file from the SWC template repository.

    Downloads individual files (e.g., .nii.gz, .ilf) from the SWC template
    repository on GIN.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where the file will be downloaded.
    file_url : str
        The URL of the file to download.
    filename : str
        The filename to save the downloaded file as.
    known_hash : str, optional
        The SHA256 hash of the file for verification. If None, the file
        will be downloaded without hash verification.

    Returns
    -------
    pathlib.Path
        The path to the downloaded file.

    Raises
    ------
    requests.exceptions.ConnectionError
        If there is no internet connection.
    """
    file_path = pooch.retrieve(
        url=file_url,
        known_hash=known_hash,
        path=download_dir_path,
        fname=filename,
        progressbar=True,
    )
    return Path(file_path)


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
    """Package the WHS_SD_Rat atlas using the SWC female rat template.

    Downloads the necessary raw data from the SWC template repository,
    processes the annotation and reference volumes, creates meshes for each
    brain region,
    and wraps up the data into the BrainGlobe atlas format.

    Uses the SWC female rat template as the reference image and Waxholm
    annotations filtered to only include regions present in the template.

    Parameters
    ----------
    working_dir : pathlib.Path
        The directory where temporary and final atlas files will be stored.
        This can be the same working directory as the original WHS SD rat atlas
        script, as they generate separate atlas files.

    Returns
    -------
    pathlib.Path
        The path to the generated BrainGlobe atlas zip file.

    Raises
    ------
    AssertionError
        If `ORIENTATION` or `RESOLUTION` are not correctly defined, or
        if `REFERENCE_URL` is missing.
    ValueError
        If the reference stack has zero range (all values are identical).
    """
    assert len(ORIENTATION) == 3, (
        "Orientation is not 3 characters, Got" + ORIENTATION
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

    # Download individual files from SWC template repository
    reference_path = download_swc_template_file(
        download_dir_path,
        REFERENCE_URL,
        REFERENCE_FILENAME,
        REFERENCE_HASH,
    )
    annotation_path = download_swc_template_file(
        download_dir_path,
        ANNOTATION_URL,
        ANNOTATION_FILENAME,
        ANNOTATION_HASH,
    )
    structures_ilf_path = download_swc_template_file(
        download_dir_path,
        STRUCTURES_ILF_URL,
        STRUCTURES_ILF_FILENAME,
        STRUCTURES_ILF_HASH,
    )

    # Parse structure metadata
    structures = parse_structures(structures_ilf_path)

    # Load files
    annotation_stack = load_any(annotation_path, as_numpy=True).astype(
        np.int64
    )
    reference_stack = load_any(reference_path, as_numpy=True)

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

    # Normalize reference stack to uint16 range
    dmin = np.min(reference_stack)
    dmax = np.max(reference_stack)
    drange = dmax - dmin
    if drange == 0:
        raise ValueError(
            "Reference stack has zero range (all values are identical)"
        )
    dscale = (2**16 - 1) / drange  # Scale to full uint16 range
    reference_stack = (reference_stack - dmin) * dscale
    reference_stack = reference_stack.astype(np.uint16)

    # Create hemispheres stack
    hemispheres_stack = np.full(reference_stack.shape, 2, dtype=np.uint8)
    hemispheres_stack[:244] = 1

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
        hemispheres_stack=hemispheres_stack,
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
