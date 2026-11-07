"""Package the SIGMA v2 Wistar rat anatomical atlases for BrainGlobe.

The in-vivo and ex-vivo variants are packaged as separate atlases.
The ex-vivo atlas uses the full-head registration reference and its
matching Waxholm v4 anatomical annotation.
The in-vivo atlas also includes GM, WM, and CSF probability maps,
encoded as uint16 with probability = stored value / 65535.
"""

import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

# BrainGlobe package revision; distinct from SIGMA source version 2.0.
__version__ = 0
CITATION = (
    "Barrière et al. 2019, https://doi.org/10.1038/s41467-019-13575-7; "
    "Kleven et al. 2023, https://doi.org/10.1038/s41592-023-02034-3"
)
SPECIES = "Rattus norvegicus"
ATLAS_LINK = "https://zenodo.org/records/10635831"
ATLAS_PACKAGER = "Amirreza Bahramani"

# Source arrays increase toward RAS; BrainGlobe specifies the origin ends.
ORIENTATION = "lpi"
# Encompass the source Brain, spinal cord, and inner ear branches.
ROOT_ID = 10000
RESOLUTIONS = {"exvivo": 90, "invivo": 150}  # µm, isotropic
MESH_NUM_THREADS = 8
PROBABILITY_SCALE = 65535
TISSUE_PROBABILITY_MAPS = {
    "gray_matter_probability": "gm",
    "white_matter_probability": "wm",
    "csf_probability": "csf",
}

# Zenodo lists CC BY 4.0, but NITRC lists non-commercial terms and the
# linked GitHub repository contains GPL v3. Resolve scope before release.

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir"
# Share the SIGMA source archive between both anatomical atlas variants.
SOURCE_DATA_DIR = BG_ROOT_DIR / "sigma_rat" / "source_data"
ARCHIVE_FILENAME = "sigma_wistar_rat_brain_templatesandatlases_version_2.0.zip"
ARCHIVE_URL = f"{ATLAS_LINK}/files/{ARCHIVE_FILENAME}"
ARCHIVE_HASH = "md5:a47ba53555cb33fed270b94071940f38"
EXTRACTED_DATA_DIR = (
    SOURCE_DATA_DIR / "extracted" / ARCHIVE_FILENAME.removesuffix(".zip")
)
ANATOMICAL_ATLAS_DIR = (
    EXTRACTED_DATA_DIR / "SIGMA_Rat_Brain_Atlases" / "SIGMA_Anatomical_Atlas"
)
INVIVO_ANATOMY_DIR = (
    EXTRACTED_DATA_DIR
    / "SIGMA_Rat_Anatomical_Imaging"
    / "SIGMA_Rat_Anatomical_InVivo_Template"
    / "anatomy"
)
WHS_ARCHIVE_FILENAME = "MBAT_WHS_SD_rat_atlas_v4.zip"
WHS_ARCHIVE_URL = (
    f"https://www.nitrc.org/frs/download.php/12262/{WHS_ARCHIVE_FILENAME}"
)
WHS_ARCHIVE_HASH = (
    "sha256:5a9d913b430a38123dba5ddfde7e01b1326280464e8c93d2ffad54e457aa5b72"
)
WHS_HIERARCHY_MEMBER = (
    "MBAT_WHS_SD_rat_atlas_v4/Data/WHS_SD_rat_atlas_v4_labels.ilf"
)
WHS_HIERARCHY_PATH = SOURCE_DATA_DIR / "waxholm_v4" / WHS_HIERARCHY_MEMBER


def download_resources(variant):
    """Download the selected SIGMA variant and the Waxholm v4 hierarchy."""
    source_variant = {"exvivo": "ExVivo", "invivo": "InVivo"}[variant]
    archive_root = ARCHIVE_FILENAME.removesuffix(".zip")
    annotation_dir = (
        f"{archive_root}/SIGMA_Rat_Brain_Atlases/"
        f"SIGMA_Anatomical_Atlas/{source_variant}_Atlas"
    )
    members = [
        f"{annotation_dir}/SIGMA_{source_variant}_Anatomical_Brain_Atlas.nii.gz",
        f"{annotation_dir}/SIGMA_{source_variant}_Anatomical_Brain_Atlas.txt",
    ]
    if variant == "exvivo":
        members.append(
            f"{annotation_dir}/WHS-to-SIGMA_Transformations/reference_SIGMA.nii.gz"
        )
    else:
        anatomy_dir = (
            f"{archive_root}/SIGMA_Rat_Anatomical_Imaging/"
            "SIGMA_Rat_Anatomical_InVivo_Template/anatomy"
        )
        members.extend(
            f"{anatomy_dir}/SIGMA_InVivo_Anatomical_Brain_{suffix}.nii.gz"
            for suffix in ("template", *TISSUE_PROBABILITY_MAPS.values())
        )
    pooch.retrieve(
        url=ARCHIVE_URL,
        known_hash=ARCHIVE_HASH,
        fname=ARCHIVE_FILENAME,
        path=SOURCE_DATA_DIR,
        processor=pooch.Unzip(members=members, extract_dir="extracted"),
        progressbar=True,
    )
    pooch.retrieve(
        url=WHS_ARCHIVE_URL,
        known_hash=WHS_ARCHIVE_HASH,
        fname=WHS_ARCHIVE_FILENAME,
        path=SOURCE_DATA_DIR,
        processor=pooch.Unzip(
            members=[WHS_HIERARCHY_MEMBER], extract_dir="waxholm_v4"
        ),
        progressbar=True,
    )


def retrieve_reference_and_annotation(variant):
    """Return the MRI, annotation, and additional tissue-prior images."""
    source_variant = {"exvivo": "ExVivo", "invivo": "InVivo"}[variant]
    annotation_dir = ANATOMICAL_ATLAS_DIR / f"{source_variant}_Atlas"
    if variant == "exvivo":
        # The registration target has the same affine as the annotation.
        reference_path = (
            annotation_dir
            / "WHS-to-SIGMA_Transformations"
            / "reference_SIGMA.nii.gz"
        )
    else:
        reference_path = (
            INVIVO_ANATOMY_DIR
            / "SIGMA_InVivo_Anatomical_Brain_template.nii.gz"
        )
    annotation_path = (
        annotation_dir
        / f"SIGMA_{source_variant}_Anatomical_Brain_Atlas.nii.gz"
    )
    reference = load_any(reference_path, as_numpy=True)
    annotation = load_any(annotation_path, as_numpy=True).astype(np.uint32)

    reference = reference - reference.min()
    reference = reference / reference.max()
    reference = np.rint(reference * np.iinfo(np.uint16).max)
    reference = reference.astype(np.uint16)

    additional_references = {}
    if variant == "invivo":
        for name, tissue in TISSUE_PROBABILITY_MAPS.items():
            path = (
                INVIVO_ANATOMY_DIR
                / f"SIGMA_InVivo_Anatomical_Brain_{tissue}.nii.gz"
            )
            probability = load_any(path, as_numpy=True)
            # Source maxima exceed 1 by float32 rounding (about 1.2e-7).
            # Fixed scaling preserves a common probability scale for all maps.
            probability = np.clip(probability, 0, 1)
            probability = np.rint(probability * PROBABILITY_SCALE)
            additional_references[name] = probability.astype(np.uint16)

    return reference, annotation, additional_references


def retrieve_hemisphere_map(annotation, variant):
    """Return a hemisphere map with 1 for left and 2 for right."""
    # The first source axis increases left to right. Split at physical x=0:
    # index 129 for ex-vivo and 62.93 for in-vivo; x>=0 belongs to the right.
    left_stop = {"exvivo": 129, "invivo": 63}[variant]
    hemispheres = np.full(annotation.shape, 2, dtype=np.uint8)
    hemispheres[:left_stop] = 1
    return hemispheres


def retrieve_structure_information(variant):
    """Combine SIGMA names/colors with Waxholm v4 acronyms and parents."""
    source_variant = {"exvivo": "ExVivo", "invivo": "InVivo"}[variant]
    annotation_dir = ANATOMICAL_ATLAS_DIR / f"{source_variant}_Atlas"
    # The source ITK-SNAP table has 14 header lines and quoted region names.
    label_table = np.loadtxt(
        annotation_dir / f"SIGMA_{source_variant}_Anatomical_Brain_Atlas.txt",
        dtype=str,
        skiprows=14,
        quotechar='"',
    )
    sigma_labels = {int(row[0]): row for row in label_table[1:]}
    hierarchy = ET.parse(WHS_HIERARCHY_PATH).getroot().find("structure")
    structures = [
        {
            "id": ROOT_ID,
            "name": "Root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]
    pending = deque((node, [ROOT_ID]) for node in hierarchy)
    while pending:
        node, parent_path = pending.popleft()
        structure_id = int(node.attrib["id"])
        id_path = parent_path + [structure_id]
        name = node.attrib["name"]
        rgb_triplet = list(bytes.fromhex(node.attrib["color"].lstrip("#")))
        if structure_id in sigma_labels:
            row = sigma_labels[structure_id]
            name = str(row[-1])
            rgb_triplet = [int(channel) for channel in row[1:4]]
        structures.append(
            {
                "id": structure_id,
                "name": name,
                "acronym": node.attrib["abbreviation"],
                "structure_id_path": id_path,
                "rgb_triplet": rgb_triplet,
            }
        )
        pending.extend((child, id_path) for child in node)
    return structures


def retrieve_or_construct_meshes(annotation, structures, working_dir):
    """Construct meshes for annotated regions and their anatomical parents."""
    working_dir.mkdir(parents=True, exist_ok=True)
    return construct_meshes_from_annotation(
        save_path=working_dir,
        volume=annotation,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0,
        smooth=False,
        parallel=True,
        num_threads=MESH_NUM_THREADS,
        verbosity=1,
    )


if __name__ == "__main__":
    for variant, resolution in RESOLUTIONS.items():
        atlas_name = f"sigma_{variant}_rat"
        working_dir = BG_ROOT_DIR / atlas_name
        print(f"Packaging {atlas_name} at {resolution} µm")
        download_resources(variant)
        reference, annotation, additional_references = (
            retrieve_reference_and_annotation(variant)
        )
        hemispheres = retrieve_hemisphere_map(annotation, variant)
        structures = retrieve_structure_information(variant)
        meshes = retrieve_or_construct_meshes(
            annotation, structures, working_dir
        )
        additional_metadata = {"source_version": "2.0"}
        if variant == "invivo":
            additional_metadata["tissue_probability_scale"] = PROBABILITY_SCALE
        output_dir = wrapup_atlas_from_data(
            atlas_name=atlas_name,
            atlas_minor_version=__version__,
            citation=CITATION,
            atlas_link=ATLAS_LINK,
            species=SPECIES,
            resolution=(resolution,) * 3,
            orientation=ORIENTATION,
            root_id=ROOT_ID,
            reference_stack=reference,
            annotation_stack=annotation,
            structures_list=structures,
            meshes_dict=meshes,
            working_dir=working_dir,
            atlas_packager=ATLAS_PACKAGER,
            hemispheres_stack=hemispheres,
            scale_meshes=True,
            additional_references=additional_references,
            additional_metadata=additional_metadata,
        )
        print(f"Atlas generated at {output_dir}")
