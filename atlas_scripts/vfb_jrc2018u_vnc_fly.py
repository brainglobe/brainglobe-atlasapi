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

VFB_TEMPLATE_ID = "VFB_00200000"

VFB_SOLR_TERM_INFO_URL = (
    "https://solr.virtualflybrain.org/solr/vfb_json/select"
    f"?q=id:{VFB_TEMPLATE_ID}&fl=id,term_info&rows=1&wt=json"
)

ORIENTATION = "lps"
RESOLUTION = 0.4  # microns

ROOT_ID = 999

TERM_INFO_HASH = (
    "5ac8d30fcde612fa62cf31d6c660127b49ba02fd26011ca64515dbe138ed94cd"
)
REFERENCE_HASH = (
    "48d6daa3d9c0f6bac01a12eb464344e5102233ecc2f833c6c3748eb5e30d617b"
)
ROOT_MESH_HASH = (
    "558fa380d74b1bb3b6f83c0a388d328276abf1ffcfaeb523d10030eda0af324a"
)

# The item at index zero is unused so each hash is indexed by structure ID.
ROI_HASHES = (
    None,
    "467d47d4677b0dd0185dc8bfa176a5df19c30d43166190d921d53641f8f01966",
    "7e6f26b2715f81f70e1e9085192759451c4bef9756c54434848c0c8e2acc47df",
    "ae248267adc478bf85ad3af0dc715768fc1c04b874f0578d2025efd07c4edbaf",
    "c8f6e73107d3dcb6ce880b9f88a99e5c604506d1bd4262307cdb4749ded27090",
    "21cd1c63866701a9c52eb7c22d9ddef5dcfafdda5c1dc008936af29f2a830f63",
    "a947b6a71251681107ae22a9a0e9be2ca5167957debe90762c261b1ce5a8c3d6",
    "c251844054ee93cc419fcec46d1fc8825ffe876773e4ccbbf74915b256078307",
    "ebd9ae4ab6c8226e49ec8fa6f4a54454921c4b19c6a7a9543dadd3aeef1c6bee",
    "266fdce22e036ea25b72eae1ec1a7d8bf58d12313cb8089b291a361ecce6fe92",
    "131b7375d436d5ff5f4b6965d8fec827f4d31f00aed88d4776b5d45db50257e9",
    "a6b2cef70012945d616572209d737d06736ca7174d95ca7cd926035b093aedbe",
    "81ac65ee91dc3564b8eb6e502bb76f8c7a68bb075a15253e8769f296ce607fdc",
    "7f38d643f78778f1344c8b22e4dd7f3148a5276c567927a77d1d1d349d6a0adf",
    "d66ab04f889e60e9587b93ea60e3d9676519a937beef5b98184bebd294a2d94e",
    "fa396668d77ff5c34d28300ea5e7501ef434c680a4e2ce2837b7742da6c07321",
    "5e3949370f27ebf83c5620d487f8537f26249fbef8bbc1d0ae783bbe29ee39e9",
    "918ba24b96cd2f973bbde8f4377970baabb98ab27c957f80b3b276f8a1a14ab7",
    "0d0f89e73fa79d709756a76bec72b0d3d4dcf2408312cc4f5c8acf6b63307685",
    "bfa2c1fb7d5e44426bee569d504c80e7f1695eeefa5e8afc683a908516cafc1f",
    "d3fa3913db596058bd405912e56b0dea3f88b6b4d14fb618ec56a7d40a182d0c",
    "8a9fe859408a9ff88b69add5cb5527c5d0f9da97ae188c5534cb0ad8363c87ed",
)

MESH_HASHES = (
    None,
    "00a58fa0dde09778a859056eb11397cf6aabd65be8146ee47578231afdab7342",
    "9033876d2dc3819f43bd70d7a236b49d6d39de1621f25b307697e31be38922c8",
    "ed1446ea03f30110cb36a2be675e0f5be3f1a6d40d42384064829a5fa1d60028",
    "a2e1c1d16c9eda0a190f3ceed398c8cf1b05fadbc2d63a196540963e8dee5088",
    "3761054ba9626dc0fb3390522d45f634216d4420404eca8f7ef675570fb7a3d6",
    "ec2d92b947f7e9e5c8ff7db35ea6d9504c1f4b1a9dad25e46a7a2cc09a433d85",
    "26a8637ea7aab50b921d31719e4b2ba5c22970fb5f4a7e0fdca60066d9ca5c3c",
    "0a31cbd2c6f62898ef11421cfd8c5728e3b593fb1ca02b9c9504d5d1baafa02f",
    "95fdf161bec9435b94ea6bce2159fea039c1140469b0ec3ce1841d5f920d0bd3",
    "74d5637e49a985809a4614edd28a017545753c926b98495d496a85664dd483a5",
    "d719496aeb8ddd12f7e048ce26f854862686989f8013702b5f2115d6bbc14cde",
    "5ddcd903dfd649848d3e836de5a9949d90c701d5ac4868200f0dbec3789c3ce0",
    "f315bdc530428bb0d3563ff37d0a62ca1aae4333c58c5a96ce187ac8e79b352d",
    "575b4b6680b16f0ae86033779e3a6e1805e3f9f71019ac56bf137b45df7f35b4",
    "202b571e2570073bd065cdbdb501e82d5fda463ec04543c097cf3787efd51ad7",
    "d5a03607718fc66634d6831f66be991b6c91b7b64145391e5c0b3af00fe7c2ac",
    "71c2dbdf408b48d6f4faf654a12bd6a7ca87516629c1ffa2e802b338cfb767ec",
    "4ee8a9bc6224716d33782c6aedd966b8a055bcc1eb0f6bfc483b4a4c10fd596a",
    "6207e0d4ab6974ebb0aa0e38b52c9cb4cd3fc29495cd4768f84759cc2d3f3e57",
    "6009f9fa51116680642e034a945e900e6a0eb88578b5896a2354a6650bd3ad9b",
    "9ad79819aef2454eb9960255907d7ef6b39bce26f63cbeb4048dbaa69c99001b",
)

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

    term_info_response_path = Path(
        pooch.retrieve(
            url=VFB_SOLR_TERM_INFO_URL,
            known_hash=TERM_INFO_HASH,
            path=SOURCE_DATA_DIR,
            fname="jrc2018u_vnc_term_info_response.json",
            progressbar=True,
        )
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

    template_channel = term_info["template_channel"]
    # VFB currently returns HTTP asset URLs, but serves the same files over
    # HTTPS. Normalize each source URL where it is downloaded.
    reference_path = Path(
        pooch.retrieve(
            url=template_channel["image_nrrd"].replace(
                "http://www.virtualflybrain.org",
                "https://www.virtualflybrain.org",
            ),
            known_hash=REFERENCE_HASH,
            path=SOURCE_DATA_DIR,
            fname=REFERENCE_PATH.name,
            progressbar=True,
        )
    )

    mesh_paths = {
        ROOT_ID: Path(
            pooch.retrieve(
                url=template_channel["image_obj"].replace(
                    "http://www.virtualflybrain.org",
                    "https://www.virtualflybrain.org",
                ),
                known_hash=ROOT_MESH_HASH,
                path=MESHES_DIR,
                fname=f"{ROOT_ID}.obj",
                progressbar=True,
            )
        )
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
            pooch.retrieve(
                url=domain["image_nrrd"].replace(
                    "http://www.virtualflybrain.org",
                    "https://www.virtualflybrain.org",
                ),
                known_hash=ROI_HASHES[structure_id],
                path=ROI_VOLUMES_DIR,
                fname=f"{structure_id}.nrrd",
                progressbar=True,
            )
        )
        mesh_paths[structure_id] = Path(
            pooch.retrieve(
                url=domain["image_obj"].replace(
                    "http://www.virtualflybrain.org",
                    "https://www.virtualflybrain.org",
                ),
                known_hash=MESH_HASHES[structure_id],
                path=MESHES_DIR,
                fname=f"{structure_id}.obj",
                progressbar=True,
            )
        )

    return reference_path, domain_names, roi_paths, mesh_paths


def retrieve_reference_and_annotation(reference_path, roi_paths):
    """Load the VNC reference and combine ROI masks into one annotation."""
    reference = _load_nrrd_array(reference_path)
    annotation = np.zeros(reference.shape, dtype=np.uint16)

    roi_sizes = []
    for structure_id, roi_path in roi_paths.items():
        roi_mask = _load_nrrd_array(roi_path)
        if roi_mask.shape != reference.shape:
            raise ValueError(
                f"ROI {structure_id} has shape {roi_mask.shape}, "
                f"but reference has shape {reference.shape}"
            )

        roi_sizes.append((np.count_nonzero(roi_mask), structure_id))

    # Some VNC domains overlap. Write broad masks first so smaller domains
    # remain visible in the single-label BrainGlobe annotation image.
    for _, structure_id in sorted(roi_sizes, reverse=True):
        mask_voxels = _load_nrrd_array(roi_paths[structure_id]) > 0
        annotation[mask_voxels] = structure_id

    return reference, annotation


def retrieve_hemisphere_map():
    """Return no hemisphere map because this atlas is treated as symmetric."""
    return None


def retrieve_structure_information(domain_names):
    """Return a flat root-plus-domain structure list."""
    structures = [
        {
            "id": ROOT_ID,
            "name": "JRC2018UnisexVNC",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    for structure_id, domain_label in sorted(domain_names.items()):
        domain_name = domain_label.split(" on ", maxsplit=1)[0]
        domain_name = domain_name.replace("\\'", "'")
        color_value = (structure_id * 2654435761) % (2**32)
        structures.append(
            {
                "id": structure_id,
                "name": domain_name,
                "acronym": ACRONYM_OVERRIDES[structure_id],
                "structure_id_path": [ROOT_ID, structure_id],
                "rgb_triplet": [
                    50 + ((color_value >> shift) % 180) for shift in (16, 8, 0)
                ],
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
    structures = retrieve_structure_information(domain_names)
    annotation_structure_ids = {
        int(structure_id) for structure_id in np.unique(annotated_volume)
    }
    structures = [
        structure
        for structure in structures
        if structure["id"] == ROOT_ID
        or structure["id"] in annotation_structure_ids
    ]
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
