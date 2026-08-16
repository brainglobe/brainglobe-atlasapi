"""Atlas generation script for the VFB's JRC2018Unisex VNC Fly atlas."""

import json
from pathlib import Path

import numpy as np
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata
__version__ = 0

ATLAS_NAME = "vfb_jrc2018u_vnc_fly"

CITATION = (
    "Bogovic et al. 2020, https://doi.org/10.1371/journal.pone.0236495; "
    "Court et al. 2020, https://doi.org/10.1016/j.neuron.2020.08.005"
)

SPECIES = "Drosophila melanogaster"

ATLAS_LINK = (
    "https://www.virtualflybrain.org/term/"
    "adult-vnc-neuropils-court2020-court2020/"
)

ORIENTATION = "lps"
RESOLUTION = 0.4  # microns

ROOT_ID = 999

# First 21 colours from VFB's viewer palette, assigned in structure-ID order:
# https://github.com/VirtualFlyBrain/geppetto-vfb/blob/master/components/configuration/VFBMain/colours.json
VFB_COLOR_PALETTE = [
    [91, 91, 91],
    [0, 255, 0],
    [255, 0, 255],
    [0, 0, 255],
    [0, 132, 246],
    [0, 141, 70],
    [167, 97, 62],
    [79, 0, 106],
    [0, 255, 246],
    [62, 123, 141],
    [237, 167, 255],
    [211, 255, 149],
    [185, 79, 255],
    [229, 26, 88],
    [132, 132, 0],
    [0, 255, 149],
    [97, 0, 44],
    [246, 132, 18],
    [202, 255, 0],
    [44, 62, 0],
    [0, 53, 193],
]

# From VFB individual ROI pub_syn records, with capitalization checked against
# Court et al. 2020 Table 1, on 2026-05-31:
# https://solr.virtualflybrain.org/solr/vfb_json/select
ACRONYM_OVERRIDES = {
    1: "DLT",
    2: "DLV",
    3: "DMT",
    4: "MDT",
    5: "VLT",
    6: "ITD",
    7: "ITD-CFF",
    8: "ITD-HC",
    9: "ITD-HT",
    10: "VTV",
    11: "ANm",
    12: "AMNp",
    13: "HTct",
    14: "IntTct",
    15: "LTct",
    16: "MesoNm",
    17: "MetaNm",
    18: "mVAC",
    19: "NTct",
    20: "ProNm",
    21: "WTct",
}

ATLAS_PACKAGER = "Amirreza Bahramani"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
SOURCE_DATA_DIR = BG_ROOT_DIR / "source_data"
ROI_VOLUMES_DIR = SOURCE_DATA_DIR / "roi_volumes"
MESHES_DIR = SOURCE_DATA_DIR / "meshes"


def pooch_init(temp_download_dir):
    """Initialize Pooch with the VNC atlas registry."""
    dawg = pooch.create(path=temp_download_dir, base_url="")
    dawg.load_registry(
        Path(__file__).parent / "hashes" / f"{ATLAS_NAME}.txt"
    )
    return dawg


def download_resources():
    """Download the VFB template, ROI volumes, meshes, and metadata."""
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ROI_VOLUMES_DIR.mkdir(exist_ok=True)
    MESHES_DIR.mkdir(exist_ok=True)
    source_pooch = pooch_init(SOURCE_DATA_DIR)
    roi_pooch = pooch_init(ROI_VOLUMES_DIR)
    mesh_pooch = pooch_init(MESHES_DIR)

    term_info_response_path = Path(
        source_pooch.fetch(
            "jrc2018u_vnc_term_info_response.json", progressbar=True
        )
    )
    with open(term_info_response_path, encoding="utf-8") as f:
        response = json.load(f)

    docs = response["response"]["docs"]
    if not docs:
        raise RuntimeError(
            "No VFB term-info record found"
        )

    term_info = docs[0]["term_info"]
    if isinstance(term_info, list):
        term_info = term_info[0]
    term_info = json.loads(term_info)

    reference_path = Path(
        source_pooch.fetch("jrc2018u_vnc_template.nrrd", progressbar=True)
    )

    mesh_paths = {
        ROOT_ID: Path(mesh_pooch.fetch(f"{ROOT_ID}.obj", progressbar=True))
    }

    domain_names = {}
    roi_paths = {}
    domains = sorted(
        term_info["template_domains"],
        key=lambda domain: int(domain["index"][0]),
    )
    for domain in domains:
        vfb_index = int(domain["index"][0])
        if vfb_index == 0:
            continue

        structure_id = vfb_index
        domain_names[structure_id] = domain["anatomical_individual"]["label"]
        roi_paths[structure_id] = Path(
            roi_pooch.fetch(f"{structure_id}.nrrd", progressbar=True)
        )
        mesh_paths[structure_id] = Path(
            mesh_pooch.fetch(f"{structure_id}.obj", progressbar=True)
        )

    return reference_path, domain_names, roi_paths, mesh_paths


def retrieve_reference_and_annotation(reference_path, roi_paths):
    """Load the VNC reference and combine ROI masks into one annotation."""
    reference = sitk.GetArrayFromImage(sitk.ReadImage(str(reference_path))).transpose(2, 1, 0)
    annotation = np.zeros(reference.shape, dtype=np.uint16)

    roi_sizes = []
    for structure_id, roi_path in roi_paths.items():
        roi_mask = sitk.GetArrayFromImage(sitk.ReadImage(str(roi_path))).transpose(2, 1, 0)
        if roi_mask.shape != reference.shape:
            raise ValueError(
                f"ROI {structure_id} has shape {roi_mask.shape}, "
                f"but reference has shape {reference.shape}"
            )

        roi_sizes.append((np.count_nonzero(roi_mask), structure_id))

    # Some VNC domains overlap. Write broad masks first so smaller domains
    # remain visible in the single-label BrainGlobe annotation image.
    for _, structure_id in sorted(roi_sizes, reverse=True):
        mask_voxels = sitk.GetArrayFromImage(sitk.ReadImage(str(roi_paths[structure_id]))).transpose(2, 1, 0) > 0
        annotation[mask_voxels] = structure_id

    return reference, annotation


def retrieve_hemisphere_map():
    """Return no hemisphere map because this atlas is treated as symmetric."""
    return None


def retrieve_structure_information(domain_names, annotation):
    """Return a flat root-plus-domain structure list."""
    annotation_structure_ids = set(np.unique(annotation))
    structures = [
        {
            "id": ROOT_ID,
            "name": "JRC2018UnisexVNC",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    for color, (structure_id, domain_label) in zip(
        VFB_COLOR_PALETTE, sorted(domain_names.items())
    ):
        if structure_id not in annotation_structure_ids:
            continue

        domain_name = domain_label.split(" on ", maxsplit=1)[0]
        domain_name = domain_name.replace("\\'", "'")
        structures.append(
            {
                "id": structure_id,
                "name": domain_name,
                "acronym": ACRONYM_OVERRIDES[structure_id],
                "structure_id_path": [ROOT_ID, structure_id],
                "rgb_triplet": color,
            }
        )

    return structures


def retrieve_or_construct_meshes(mesh_paths, structures):
    """Return the downloaded VFB mesh paths."""
    structure_ids = {structure["id"] for structure in structures}
    return {
        structure_id: mesh_path
        for structure_id, mesh_path in mesh_paths.items()
        if structure_id in structure_ids
    }


def retrieve_additional_references():
    """Return no additional reference images."""
    return {}


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    reference_path, domain_names, roi_paths, mesh_paths = download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation(
        reference_path, roi_paths
    )
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information(domain_names, annotated_volume)
    meshes_dict = retrieve_or_construct_meshes(mesh_paths, structures)

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
        scale_meshes=False,
        additional_references=additional_references,
    )
    print(f"Atlas packaged: {output_filename}")
