"""Package the swc_female_rat rat brain atlas (50 µm) in BrainGlobe format.

This version:
- Downloads template, annotation, and structures files from remote URLs
- Uses annotation labels with Waxholm IDs
"""

__version__ = "0"

import json
from pathlib import Path

import numpy as np
import pooch
import xmltodict
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# -------------------------------------------------------------------------
# Atlas metadata
# -------------------------------------------------------------------------

ATLAS_NAME = "swc_female_rat"
SPECIES = "Rattus norvegicus"
ATLAS_LINK = ""
CITATION = "unpublished"
ORIENTATION = "asr"
RESOLUTION = (50, 50, 50)
ROOT_ID = 10000  # Waxholm root
ATLAS_PACKAGER = "Viktor Plattner, v.plattner@ucl.ac.uk"

# File names on GIN.
TEMPLATE_FILENAME = "template_sharpen_shapeupdate_orig-asr_aligned.nii.gz"
ANNOTATION_FILENAME = "WHS_SD_annotation_template_space_cleaned.nii.gz"
STRUCTURES_ILF_FILENAME = "WHS_SD_rat_atlas_v4.01_labels.ilf"

# Links to the atlas files (GIN repository).
TEMPLATE_URL = (
    f"https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    f"raw/master/packaging/{RESOLUTION[1]}um/{TEMPLATE_FILENAME}"
)
ANNOTATION_URL = (
    f"https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    f"raw/master/packaging/{RESOLUTION[1]}um/{ANNOTATION_FILENAME}"
)
STRUCTURES_ILF_URL = (
    f"https://gin.g-node.org/BrainGlobe/swc_rat_atlas_materials/"
    f"raw/master/packaging/{STRUCTURES_ILF_FILENAME}"
)

# Hashes for the atlas files. Modify these if the files are updated.
TEMPLATE_HASH = (
    "7aed7300fdec07c601f376ffe7b77da059fcb6a46e568e99125e473e08e75c8a"
)
ANNOTATION_HASH = (
    "bf7ace23df27a1037d494cac030d07e2a45bc26df5cf611b5563598a581116b8"
)
STRUCTURES_ILF_HASH = (
    "dd0de0cfb3ae22a8e5666d11df846afa465aa5c5c2dd02af68720488633d2a65"
)

# -------------------------------------------------------------------------
# Define function to download atlas files
# -------------------------------------------------------------------------


def download_atlas_files(
    download_dir_path: Path,
    atlas_file_url: str,
    filename: str,
    known_hash: str = None,
):
    """Download atlas files."""
    utils.check_internet_connection()
    file_path = pooch.retrieve(
        url=atlas_file_url,
        known_hash=known_hash,
        path=download_dir_path,
        fname=filename,
        progressbar=True,
    )
    return Path(file_path)


# -------------------------------------------------------------------------
# Define function to load structures from XML
# -------------------------------------------------------------------------


def _parse_structures_xml(node: dict, path=None, structures=None):
    """Recursively parse the Waxholm ILF XML tree into a list of structures."""
    structures = structures or []
    path = path or []

    # Colour in the ILF is stored as a hex string like "#RRGGBB"
    rgb_triplet = tuple(int(node["@color"][i : i + 2], 16) for i in (1, 3, 5))
    sid = int(node["@id"])

    struct = {
        "name": node["@name"],
        "acronym": node["@abbreviation"],
        "id": sid,
        "structure_id_path": path + [sid],
        "rgb_triplet": list(rgb_triplet),
    }
    structures.append(struct)

    # Recurse into children (labels)
    if "label" in node:
        if isinstance(node["label"], list):
            for child in node["label"]:
                _parse_structures_xml(
                    child, path=path + [sid], structures=structures
                )
        else:
            _parse_structures_xml(
                node["label"], path=path + [sid], structures=structures
            )

    return structures


def load_structures_from_ilf(ilf_path: Path, root_id: int):
    """Parse the Waxholm .ilf file to extract region metadata."""
    root = xmltodict.parse(ilf_path.read_text())["milf"]["structure"]

    # Normalise the top node to a BrainGlobe-style root
    root["@abbreviation"] = "root"
    root["@color"] = "#ffffff"
    root["@id"] = str(root_id)
    root["@name"] = "Root"

    structures = _parse_structures_xml(root)
    return structures


# -------------------------------------------------------------------------
# Define function to create meshes
# -------------------------------------------------------------------------


def create_meshes(work_dir, annotated_volume, structures):
    """Generate meshes for each brain region."""
    meshes_dict = construct_meshes_from_annotation(
        save_path=work_dir,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=False,
    )

    # Filter structures to only those with meshes
    structures_with_mesh = [s for s in structures if s["id"] in meshes_dict]

    return meshes_dict, structures_with_mesh


# -------------------------------------------------------------------------
# Define function to create atlas
# -------------------------------------------------------------------------


def create_atlas(
    working_dir: Path,
):
    """Package the swc_female_rat atlas."""
    assert (
        len(ORIENTATION) == 3
    ), f"Orientation is not 3 characters, got {ORIENTATION}"
    assert (
        len(RESOLUTION) == 3
    ), f"Resolution is not length 3, got {RESOLUTION}"

    working_dir.mkdir(exist_ok=True, parents=True)

    # Download atlas files
    template_path = download_atlas_files(
        working_dir, TEMPLATE_URL, TEMPLATE_FILENAME, TEMPLATE_HASH
    )
    annotation_path = download_atlas_files(
        working_dir, ANNOTATION_URL, ANNOTATION_FILENAME, ANNOTATION_HASH
    )
    structures_ilf_path = download_atlas_files(
        working_dir,
        STRUCTURES_ILF_URL,
        STRUCTURES_ILF_FILENAME,
        STRUCTURES_ILF_HASH,
    )

    # Parse structure metadata (Waxholm hierarchy)
    structures = load_structures_from_ilf(structures_ilf_path, ROOT_ID)

    # Load volumes (template + annotations)
    annotation_stack = load_any(annotation_path, as_numpy=True).astype(
        np.int64
    )
    reference_stack = load_any(template_path, as_numpy=True)

    # Filter structures to those that actually appear in annotations
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

    # Mask reference to brain voxels only
    reference_stack *= annotation_stack > 0

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

    # Hemispheres stack
    hemispheres_stack = None

    # Save regions list json
    with open(working_dir / "structures.json", "w") as f:
        json.dump(structures, f, indent=2)

    # Create meshes
    print(f"Saving atlas data at {working_dir}")
    meshes_dict, structures_with_mesh = create_meshes(
        working_dir, annotation_stack, structures
    )

    # Wrap up into BrainGlobe atlas zip
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


# -------------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------------

if __name__ == "__main__":
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
