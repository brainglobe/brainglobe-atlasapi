"""Atlas generation script for the VFB's JRC2018Unisex Neuropils Fly atlas."""

import json
from pathlib import Path
from urllib.parse import urlsplit

import brainglobe_space as bgs
import meshio as mio
import numpy as np
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata
__version__ = 0

ATLAS_NAME = "vfb_jrc2018u_neuropils_fly"

CITATION = "Bogovic et al. 2020, https://doi.org/10.1371/journal.pone.0236495"

SPECIES = "Drosophila melanogaster"

ATLAS_LINK = (
    "https://www.virtualflybrain.org/blog/2022/01/01/"
    "jrc-2018-templates-rois-jrc2018/"
)

VFB_TEMPLATE_ID = "VFB_00101567"

VFB_SOLR_TERM_INFO_URL = (
    "https://solr.virtualflybrain.org/solr/vfb_json/select"
    f"?q=id:{VFB_TEMPLATE_ID}&fl=id,term_info&rows=1&wt=json"
)

SOURCE_ORIENTATION = "lps"
SOURCE_RESOLUTION = (0.5189161, 0.5189161, 1.0)  # Exact VFB NRRD spacing

ROOT_ID = 999
ORIENTATION = "asr"
RESOLUTION = (0.519, 1.0, 0.519)  # microns

ATLAS_PACKAGER = "Amirreza Bahramani"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
SOURCE_DATA_DIR = BG_ROOT_DIR / "source_data"
ROI_VOLUMES_DIR = SOURCE_DATA_DIR / "roi_volumes"
MESHES_DIR = SOURCE_DATA_DIR / "meshes"
ASR_MESHES_DIR = SOURCE_DATA_DIR / "asr_meshes"
REFERENCE_PATH = SOURCE_DATA_DIR / "jrc2018u_template.nrrd"
DOMAIN_METADATA_PATH = SOURCE_DATA_DIR / "jrc2018u_domain_metadata.json"


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


def _source_space(source_shape):
    physical_shape = tuple(
        size * resolution
        for size, resolution in zip(source_shape, SOURCE_RESOLUTION)
    )
    return bgs.AnatomicalSpace(SOURCE_ORIENTATION, shape=physical_shape)


def _map_stack_to_asr(stack):
    mapped_stack = _source_space(stack.shape).map_stack_to(
        ORIENTATION, stack, copy=False
    )
    return np.ascontiguousarray(mapped_stack)


def _map_mesh_to_asr(mesh_path, output_path, source_shape):
    mesh = mio.read(mesh_path)
    mesh.points = _source_space(source_shape).map_points_to(
        ORIENTATION, mesh.points
    )
    mio.write(output_path, mesh)
    return output_path


def download_resources():
    """Download the VFB template, ROI volumes, meshes, and metadata."""
    SOURCE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ROI_VOLUMES_DIR.mkdir(exist_ok=True)
    MESHES_DIR.mkdir(exist_ok=True)

    term_info_response_path = _retrieve(
        VFB_SOLR_TERM_INFO_URL,
        SOURCE_DATA_DIR,
        "jrc2018u_term_info_response.json",
    )
    with open(term_info_response_path, encoding="utf-8") as f:
        response = json.load(f)

    term_info = json.loads(response["response"]["docs"][0]["term_info"][0])

    term_info_path = SOURCE_DATA_DIR / "jrc2018u_term_info.json"
    with open(term_info_path, "w", encoding="utf-8") as f:
        json.dump(term_info, f, indent=2)

    template_channel = term_info["template_channel"]
    # VFB metadata may use a clear-text scheme; force HTTPS for downloads.
    reference_url = template_channel["image_nrrd"]
    reference_url = urlsplit(reference_url)._replace(scheme="https").geturl()
    _retrieve(
        reference_url,
        SOURCE_DATA_DIR,
        REFERENCE_PATH.name,
    )

    root_mesh_url = urlsplit(template_channel["image_obj"])
    root_mesh_url = root_mesh_url._replace(scheme="https").geturl()
    _retrieve(
        root_mesh_url,
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

        roi_url = urlsplit(domain["image_nrrd"])
        roi_url = roi_url._replace(scheme="https").geturl()
        roi_path = _retrieve(
            roi_url,
            ROI_VOLUMES_DIR,
            f"{structure_id}.nrrd",
        )

        mesh_url = urlsplit(domain["image_obj"])
        mesh_url = mesh_url._replace(scheme="https").geturl()
        mesh_path = _retrieve(
            mesh_url,
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
    """Load the reference and combine ROI masks into one annotation."""
    reference = sitk.GetArrayFromImage(
        sitk.ReadImage(str(REFERENCE_PATH))
    ).transpose(2, 1, 0)
    annotation = np.zeros(reference.shape, dtype=np.uint16)

    roi_paths = sorted(
        ROI_VOLUMES_DIR.glob("*.nrrd"),
        key=lambda path: int(path.stem),
    )
    for roi_path in roi_paths:
        structure_id = int(roi_path.stem)
        roi_mask = sitk.GetArrayFromImage(
            sitk.ReadImage(str(roi_path))
        ).transpose(2, 1, 0)
        mask_voxels = roi_mask > 0
        annotation[mask_voxels] = structure_id

    return _map_stack_to_asr(reference), _map_stack_to_asr(annotation)


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
            "name": "JRC2018Unisex adult brain",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    domains = sorted(domain_metadata, key=lambda domain: int(domain["id"]))
    for domain in domains:
        structure_id = int(domain["id"])
        domain_acronym = domain["label"].split(" on ", maxsplit=1)[0]
        domain_acronym = domain_acronym.replace("\\'", "'")
        color_value = (structure_id * 2654435761) % (2**32)
        structures.append(
            {
                "id": structure_id,
                "name": domain["type_label"],
                "acronym": domain_acronym,
                "structure_id_path": [ROOT_ID, structure_id],
                "rgb_triplet": [
                    50 + ((color_value >> shift) % 180) for shift in (16, 8, 0)
                ],
            }
        )

    return structures


def retrieve_or_construct_meshes():
    """Return the VFB mesh files mapped into BrainGlobe ASR space."""
    ASR_MESHES_DIR.mkdir(exist_ok=True)

    with open(DOMAIN_METADATA_PATH, encoding="utf-8") as f:
        domain_metadata = json.load(f)

    source_shape = (
        sitk.GetArrayFromImage(sitk.ReadImage(str(REFERENCE_PATH)))
        .transpose(2, 1, 0)
        .shape
    )
    root_mesh_path = MESHES_DIR / f"{ROOT_ID}.obj"
    meshes_dict = {
        ROOT_ID: _map_mesh_to_asr(
            root_mesh_path, ASR_MESHES_DIR / f"{ROOT_ID}.obj", source_shape
        )
    }
    domains = sorted(domain_metadata, key=lambda domain: int(domain["id"]))
    for domain in domains:
        structure_id = int(domain["id"])
        mesh_path = Path(domain["obj"])
        meshes_dict[structure_id] = _map_mesh_to_asr(
            mesh_path, ASR_MESHES_DIR / f"{structure_id}.obj", source_shape
        )

    return meshes_dict


def retrieve_additional_references():
    """Return no additional reference images."""
    return {}


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes()

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=False,
        additional_references=additional_references,
    )
    print(f"Atlas packaged: {output_filename}")
