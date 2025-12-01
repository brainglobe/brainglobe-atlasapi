"""Package the swc_female_rat rat brain atlas (50 µm) in BrainGlobe format.

This version assumes:
- template, annotation, and structures files are all local
- annotation labels use the Waxholm IDs (same hierarchy)
"""

__version__ = "1"

import json
import time
from pathlib import Path

import numpy as np
import xmltodict
from brainglobe_utils.IO.image import load_any
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# -------------------------------------------------------------------------
# Paths to the template, annotation and structures files
# -------------------------------------------------------------------------


TEMPLATE_PATH = Path(
    "/ceph/akrami/_projects/rat_atlas/atlas/swc-female-rat-atlas_v2"
    "/50um/template_space/template_sharpen_shapeupdate_orig-asr_aligned.nii.gz"
)
ANNOTATION_PATH = Path(
    "/ceph/akrami/_projects/rat_atlas/atlas/swc-female-rat-atlas_v2"
    "/50um/template_space/WaxholmLabels_cleaned.nii.gz"
)

STRUCTURES_ILF_PATH = Path(
    "/ceph/akrami/_projects/rat_atlas/atlas/swc-female-rat-atlas_v2"
    "/50um/template_space/WHS_SD_rat_atlas_v4.01_labels.ilf"
)

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


# -------------------------------------------------------------------------
# Load structures from XML
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
# Create meshes
# -------------------------------------------------------------------------


def create_meshes(work_dir, tree, annotated_volume, labels, root_id):
    """Generate meshes for each brain region."""
    meshes_dir_path = work_dir / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    for key, node in tree.nodes.items():
        node.data = Region(key in labels)

    closing_n_iters = 2
    decimate_fraction = 0.2
    smooth = False  # smooth meshes after creation

    start = time.time()
    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        description="Creating meshes",
    ):
        create_region_mesh(
            (
                meshes_dir_path,
                node,
                tree,
                labels,
                annotated_volume,
                root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )

    print(
        "Finished mesh extraction in:",
        round((time.time() - start) / 60, 2),
        "minutes",
    )
    return meshes_dir_path


def create_mesh_dict(structures, meshes_dir_path):
    """Map structure IDs to their mesh file paths."""
    meshes_dict = {}
    structures_with_mesh = []

    for s in structures:
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'

        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it.")
            continue
        if mesh_path.stat().st_size < 512:
            print(f"obj file for {s} is too small, ignoring it.")
            continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )
    return meshes_dict, structures_with_mesh


# -------------------------------------------------------------------------
# Create atlas
# -------------------------------------------------------------------------


def create_atlas(
    working_dir: Path,
    template_path: Path = TEMPLATE_PATH,
    annotation_path: Path = ANNOTATION_PATH,
    structures_ilf_path: Path = STRUCTURES_ILF_PATH,
):
    """Package the swc_female_rat atlas from local files."""
    assert (
        len(ORIENTATION) == 3
    ), f"Orientation is not 3 characters, got {ORIENTATION}"
    assert (
        len(RESOLUTION) == 3
    ), f"Resolution is not length 3, got {RESOLUTION}"

    working_dir.mkdir(exist_ok=True, parents=True)

    # 1) Parse structure metadata (Waxholm hierarchy)
    structures = load_structures_from_ilf(structures_ilf_path, ROOT_ID)

    # 2) Load volumes (template + annotations)
    annotation_stack = load_any(annotation_path, as_numpy=True).astype(
        np.int64
    )
    reference_stack = load_any(template_path, as_numpy=True)

    # 3) Filter structures to those that actually appear in annotations
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
    tree = get_structures_tree(structures)

    # 4) Mask reference to brain voxels only
    reference_stack *= annotation_stack > 0

    dmin = np.min(reference_stack)
    dmax = np.max(reference_stack)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange  # Scale to full uint16 range (0-65535)
    reference_stack = (reference_stack - dmin) * dscale
    reference_stack = reference_stack.astype(np.uint16)

    # 5) (Optional) Hemispheres stack
    hemispheres_stack = None

    # 6) Save regions list json (for debugging / inspection)
    with open(working_dir / "structures.json", "w") as f:
        json.dump(structures, f, indent=2)

    # 7) Create meshes
    print(f"Saving atlas data at {working_dir}")
    meshes_dir_path = create_meshes(
        working_dir, tree, annotation_stack, labels, ROOT_ID
    )

    meshes_dict, structures_with_mesh = create_mesh_dict(
        structures, meshes_dir_path
    )

    # 8) Wrap up into BrainGlobe atlas zip
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
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
