"""Package the Danionella cerebrum mixed reference atlas.

This script packages the dc_mixed_hhg6@1.0 reference, MECE segmentation,
and same-space confocal reflectance reference.
"""

import json
import re
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata ###

__version__ = 1
ATLAS_NAME = "danionella_cerebrum_mixed"
CITATION = (
    "Kadobianskyi et al. 2026, bioRxiv, "
    "https://doi.org/10.64898/2026.03.09.710483"
)
SPECIES = "Danionella cerebrum"
ATLAS_LINK = "https://gin.g-node.org/danionella/dc_atlas"
ORIENTATION = "las"
ROOT_ID = 9999
RESOLUTION = 2.5
ATLAS_PACKAGER = "Amirreza Bahramani"

GIN_RAW_BASE_URL = f"{ATLAS_LINK}/raw/master"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

REFERENCE_FNAME = "reference.nii.gz"
ANNOTATION_FNAME = "segmentation.nii.gz"
HIERARCHY_FNAME = "dc_labels.json"

# Source metadata registers this structural image to dc_mixed_hhg6@1.0:
# https://gin.g-node.org/danionella/dc_atlas/src/master/datasets/structural/fiber_tracts/confocalreflectance_jlab_01/dataset.json
CONFOCAL_REFERENCE_NAME = "structural_confocal_reflectance"
CONFOCAL_REFERENCE_FNAME = f"{CONFOCAL_REFERENCE_NAME}.nii.gz"

SOURCE_FILES = {
    REFERENCE_FNAME: "atlases/dc_mixed_hhg6/1.0/template/reference.nii.gz",
    ANNOTATION_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/segmentation/MECE/1.0/segmentation.nii.gz"
    ),
    HIERARCHY_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/segmentation/MECE/1.0/dc_labels.json"
    ),
    CONFOCAL_REFERENCE_FNAME: (
        "datasets/structural/fiber_tracts/confocalreflectance_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/confocalreflectance_jlab_01.nii.gz"
    ),
}


def download_resources():
    """Download the source volumes and hierarchy."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    for fname, source_path in SOURCE_FILES.items():
        pooch.retrieve(
            url=f"{GIN_RAW_BASE_URL}/{source_path}",
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=fname,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """Load the reference and annotation volumes."""
    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME

    reference = load_any(reference_path, as_numpy=True)
    annotation = load_any(annotation_path, as_numpy=True)

    reference = np.maximum(reference, 0)
    maximum = float(np.max(reference))
    reference *= np.iinfo(np.uint16).max / maximum
    reference = np.rint(reference)
    reference = reference.astype(np.uint16)

    return reference, annotation.astype(np.uint32)


def retrieve_hemisphere_map():
    """Return no map; source labels are bilateral and no map is provided."""
    return None


def retrieve_structure_information():
    """Convert the source hierarchy to BrainGlobe structure dictionaries."""
    hierarchy_path = DOWNLOAD_DIR_PATH / HIERARCHY_FNAME

    with open(hierarchy_path) as hierarchy_file:
        source_hierarchy = json.load(hierarchy_file)

    structures = []

    def add_structure(node, structure_id_path):
        structure_id = int(node["id"])

        name = re.sub(r"\s*/\s*", " / ", node["name"].strip())
        name = re.sub(r"\s+", " ", name)
        if structure_id == ROOT_ID:
            name = "root"
            acronym = "root"
            rgb_triplet = [255, 255, 255]
        elif node.get("is_segmentation"):
            acronym = re.sub(r"\s*/\s*", "/", node["abbreviation"].strip())
            rgb_triplet = node["rgb"]
        else:
            # Source parent nodes do not have abbreviations, so use their full
            # names as acronyms.
            acronym = name
            rgb_triplet = node["rgb"]

        current_path = [*structure_id_path, structure_id]
        structures.append(
            {
                "id": structure_id,
                "name": name,
                "acronym": acronym,
                "structure_id_path": current_path,
                "rgb_triplet": rgb_triplet,
            }
        )

        for child in node.get("children", []):
            add_structure(child, current_path)

    add_structure(source_hierarchy, [])

    return structures


def retrieve_or_construct_meshes(annotated_volume, structures):
    """Construct meshes for all structures from the annotation volume."""
    return construct_meshes_from_annotation(
        save_path=DOWNLOAD_DIR_PATH,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=True,
        parallel=True,
        num_threads=6,
        verbosity=0,
    )


def retrieve_additional_references():
    """Load and convert the same-space confocal reflectance image."""
    reference = load_any(
        DOWNLOAD_DIR_PATH / CONFOCAL_REFERENCE_FNAME, as_numpy=True
    )

    reference = np.maximum(reference, 0)
    maximum = float(np.max(reference))
    reference *= np.iinfo(np.uint16).max / maximum
    reference = np.rint(reference)
    reference = reference.astype(np.uint16)

    return {CONFOCAL_REFERENCE_NAME: reference}


if __name__ == "__main__":
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)
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
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        scale_meshes=True,
        additional_references=additional_references,
        overwrite=True
    )

    print(f"Atlas packaged at: {output_filename}")
