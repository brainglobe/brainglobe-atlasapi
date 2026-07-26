"""Atlas generation script for the VFB's JRC2018Unisex VNC Fly atlas."""

import json
from pathlib import Path

import numpy as np
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

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

ATLAS_FILE_URL = (
    "https://www.virtualflybrain.org/data/VFB/i/0020/0000/"
    "VFB_00200000/volume.nrrd"
)

VFB_TEMPLATE_ID = "VFB_00200000"

VFB_SOLR_TERM_INFO_URL = (
    "https://solr.virtualflybrain.org/solr/vfb_json/select"
    f"?q=id:{VFB_TEMPLATE_ID}&fl=id,term_info&rows=1&wt=json"
)

ORIENTATION = "lps"
RESOLUTION = 0.4  # microns

ROOT_ID = 999

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
REFERENCE_PATH = SOURCE_DATA_DIR / "jrc2018u_vnc_template.nrrd"
DOMAIN_METADATA_PATH = SOURCE_DATA_DIR / "jrc2018u_vnc_domain_metadata.json"


def _retrieve(url, download_dir, file_name):
    return Path(
        pooch.retrieve(
            url=url,
            known_hash=None,
            path=download_dir,
            fname=file_name,
            progressbar=True,
        )
    )


def _load_nrrd_array(nrrd_path):
    """Read a VFB NRRD into the x, y, z array order used by this script."""
    return sitk.GetArrayFromImage(sitk.ReadImage(str(nrrd_path))).transpose(
        2, 1, 0
    )


def download_resources():
    """Download the VFB template, ROI volumes, meshes, and metadata."""
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ROI_VOLUMES_DIR.mkdir(exist_ok=True)
    MESHES_DIR.mkdir(exist_ok=True)
    vfb_http_prefix = "http://www.virtualflybrain.org"
    vfb_https_prefix = "https://www.virtualflybrain.org"

    term_info_response_path = _retrieve(
        VFB_SOLR_TERM_INFO_URL,
        SOURCE_DATA_DIR,
        "jrc2018u_vnc_term_info_response.json",
    )
    with open(term_info_response_path, encoding="utf-8") as f:
        response = json.load(f)

    docs = response["response"]["docs"]
    if not docs:
        raise RuntimeError(
            f"No VFB term-info record found for {VFB_TEMPLATE_ID}"
        )

    term_info = docs[0]["term_info"]
    if isinstance(term_info, list):
        term_info = term_info[0]
    term_info = json.loads(term_info)

    term_info_path = SOURCE_DATA_DIR / "jrc2018u_vnc_term_info.json"
    with open(term_info_path, "w", encoding="utf-8") as f:
        json.dump(term_info, f, indent=2)

    template_channel = term_info["template_channel"]
    reference_url = (
        template_channel.get("image_nrrd") or ATLAS_FILE_URL
    ).replace(
        vfb_http_prefix,
        vfb_https_prefix,
    )
    _retrieve(
        reference_url,
        SOURCE_DATA_DIR,
        REFERENCE_PATH.name,
    )

    _retrieve(
        template_channel["image_obj"].replace(
            vfb_http_prefix,
            vfb_https_prefix,
        ),
        MESHES_DIR,
        f"{ROOT_ID}.obj",
    )

    domain_metadata = []
    domains = sorted(
        term_info["template_domains"],
        key=lambda domain: int(domain["index"][0]),
    )
    for domain in domains:
        vfb_index = int(domain["index"][0])
        if vfb_index == 0:
            continue

        structure_id = vfb_index
        vfb_id = domain["anatomical_individual"]["short_form"]

        roi_path = _retrieve(
            domain["image_nrrd"].replace(vfb_http_prefix, vfb_https_prefix),
            ROI_VOLUMES_DIR,
            f"{structure_id}.nrrd",
        )
        mesh_path = _retrieve(
            domain["image_obj"].replace(vfb_http_prefix, vfb_https_prefix),
            MESHES_DIR,
            f"{structure_id}.obj",
        )

        domain_metadata.append(
            {
                "id": structure_id,
                "vfb_id": vfb_id,
                "label": domain["anatomical_individual"]["label"],
                "type_id": domain["anatomical_type"]["short_form"],
                "type_label": domain["anatomical_type"]["label"],
                "nrrd": str(roi_path),
                "obj": str(mesh_path),
            }
        )

    with open(DOMAIN_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(domain_metadata, f, indent=2)


def retrieve_reference_and_annotation():
    """Load the VNC reference and combine ROI masks into one annotation."""
    reference = _load_nrrd_array(REFERENCE_PATH)
    annotation = np.zeros(reference.shape, dtype=np.uint16)

    roi_sizes = []
    for roi_path in ROI_VOLUMES_DIR.glob("*.nrrd"):
        roi_mask = _load_nrrd_array(roi_path)
        if roi_mask.shape != reference.shape:
            raise ValueError(
                f"ROI {roi_path.stem} has shape {roi_mask.shape}, "
                f"but reference has shape {reference.shape}"
            )

        roi_sizes.append((np.count_nonzero(roi_mask), int(roi_path.stem)))

    # Some VNC domains overlap. Write broad masks first so smaller domains
    # remain visible in the single-label BrainGlobe annotation image.
    for _, structure_id in sorted(roi_sizes, reverse=True):
        roi_path = ROI_VOLUMES_DIR / f"{structure_id}.nrrd"
        mask_voxels = _load_nrrd_array(roi_path) > 0
        annotation[mask_voxels] = structure_id

    return reference, annotation


def retrieve_hemisphere_map():
    """Return no hemisphere map because this atlas is treated as symmetric."""
    return None


def retrieve_structure_information():
    """Return a flat root-plus-domain structure list."""
    with open(DOMAIN_METADATA_PATH, encoding="utf-8") as f:
        domain_metadata = json.load(f)

    structures = [
        {
            "id": ROOT_ID,
            "name": "JRC2018UnisexVNC",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    domains = sorted(domain_metadata, key=lambda domain: int(domain["id"]))
    for domain in domains:
        structure_id = int(domain["id"])
        domain_name = domain["label"].split(" on ", maxsplit=1)[0]
        domain_name = domain_name.replace("\\'", "'")
        color_value = (structure_id * 2654435761) % (2**32)
        structures.append(
            {
                "id": structure_id,
                "name": domain_name,
                "acronym": ACRONYM_OVERRIDES.get(structure_id, domain_name),
                "structure_id_path": [ROOT_ID, structure_id],
                "rgb_triplet": [
                    50 + ((color_value >> shift) % 180) for shift in (16, 8, 0)
                ],
            }
        )

    return structures


def retrieve_or_construct_meshes(structures):
    """Return the downloaded VFB mesh paths."""
    with open(DOMAIN_METADATA_PATH, encoding="utf-8") as f:
        domain_metadata = json.load(f)

    structure_ids = {structure["id"] for structure in structures}
    root_mesh_path = MESHES_DIR / f"{ROOT_ID}.obj"
    if not root_mesh_path.is_file():
        raise FileNotFoundError(f"Missing root mesh: {root_mesh_path}")

    meshes_dict = {ROOT_ID: root_mesh_path}
    domains = sorted(domain_metadata, key=lambda domain: int(domain["id"]))
    for domain in domains:
        structure_id = int(domain["id"])
        if structure_id not in structure_ids:
            continue

        mesh_path = Path(domain["obj"])
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Missing ROI mesh: {mesh_path}")

        meshes_dict[structure_id] = mesh_path

    return meshes_dict


def retrieve_additional_references():
    """Return no additional reference images."""
    return {}


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))
    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
            "Move it or delete it before running this script again."
        )

    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    annotation_structure_ids = {
        int(structure_id) for structure_id in np.unique(annotated_volume)
    }
    structures = [
        structure
        for structure in structures
        if structure["id"] == ROOT_ID
        or structure["id"] in annotation_structure_ids
    ]
    meshes_dict = retrieve_or_construct_meshes(structures)

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
