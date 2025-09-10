"""Atlas generation script for the Allen Segmentation version of DeMBA."""

import shutil
from pathlib import Path

import pooch
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from brainglobe_utils.IO.image import load_any
from scipy.ndimage import zoom

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

VERSION = 0
NAME = "demba_allen_seg_dev_mouse"
CITATION = "https://doi.org/10.1101/2024.06.14.598876"
ATLAS_LINK = "https://doi.org/10.25493/V3AH-HK7"
SPECIES = "Mus musculus"
ORIENTATION = "rsa"
ROOT_ID = 997

TEMPLATE_KEYS = ["acronym", "id", "name", "structure_id_path", "rgb_triplet"]

data_file_url = (
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/"
    "buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_segmentations",
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/"
    "buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_volumes",
)
resolution_to_modalities = {
    25: ["stpt", "mri", "lsfm", "allen_stpt"],
    #  the mesh generation is far too slow with 10 um
    #  10: ["allen_stpt"],
    20: ["stpt", "allen_stpt"],
}


def download_resources(download_dir_path, atlas_file_url, atlas_name):
    """
    Slight issue that the hash seems different each time.
    I think the files are being zipped on the server each time we request
    and it's changing the hash somehow (Maybe date and time is encoded in
    the file when zipped).
    """
    utils.check_internet_connection()

    download_name = atlas_name
    destination_path = download_dir_path / download_name
    for url in atlas_file_url:

        pooch.retrieve(
            url=url,
            known_hash=None,
            path=destination_path,
            progressbar=True,
            processor=pooch.Unzip(extract_dir="."),
        )
    return destination_path


def get_reference_and_annotation_paths(download_dir_path, age, modality):
    """
    Determine the reference and annotation paths based on the modality.

    Returns
    -------
        tuple: A tuple containing the reference path and annotation path.
    """
    base_path = f"{download_dir_path}/{NAME}/"
    if modality == "stpt":
        reference_path = (
            f"{base_path}DeMBA_templates/DeMBA_P{age}_brain.nii.gz"
        )
        annotation_path = (
            f"{base_path}AllenCCFv3_segmentations/20um/2022/"
            f"DeMBA_P{age}_segmentation_2022_20um.nii.gz"
        )
        volume_resolution = 20
    elif modality == "allen_stpt":
        reference_path = (
            f"{base_path}allen_stpt_10um/DeMBA_P{age}_AllenSTPT_10um.nii.gz"
        )
        annotation_path = (
            f"{base_path}AllenCCFv3_segmentations/10um/2022/"
            f"DeMBA_P{age}_segmentation_2022_10um.nii.gz"
        )
        volume_resolution = 10

    elif modality == "mri":
        reference_path = f"{base_path}mri_volumes/DeMBA_P{age}_mri.nii.gz"
        annotation_path = (
            f"{base_path}AllenCCFv3_segmentations/20um/2022/"
            f"DeMBA_P{age}_segmentation_2022_20um.nii.gz"
        )
        volume_resolution = 25

    elif modality == "lsfm":
        reference_path = f"{base_path}lsfm_volumes/DeMBA_P{age}_lsfm.nii.gz"
        annotation_path = (
            f"{base_path}AllenCCFv3_segmentations/20um/2022/"
            f"DeMBA_P{age}_segmentation_2022_20um.nii.gz"
        )
        volume_resolution = 25
    else:
        raise ValueError(f"Unknown modality: {modality}")
    return reference_path, annotation_path, volume_resolution


def retrieve_reference_and_annotation(
    download_dir_path, age, resolution, modality
):
    """
    Retrieve the desired reference and annotation as two numpy arrays.
    we unfortunately did not provide 25um segmentations so we will just
    downsample the 20um ones.

    Returns
    -------
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    reference_path, annotation_path, volume_resolution = (
        get_reference_and_annotation_paths(download_dir_path, age, modality)
    )
    annotation = load_any(annotation_path)
    reference = load_any(reference_path)
    zoom_factors = tuple(volume_resolution / resolution for _ in range(3))
    reference = zoom(reference, zoom_factors, order=1)
    if annotation.shape != reference.shape:

        zoom_factors = tuple(
            ref_dim / ann_dim
            for ref_dim, ann_dim in zip(reference.shape, annotation.shape)
        )
        annotation = zoom(annotation, zoom_factors, order=0)
    return reference, annotation


def retrieve_additional_references(
    download_dir_path, age, resolution, modalities
):
    """
    Retrieve the additional references which are MRI LSFM and Allen STPT.
    For the 10um Allen stpt is the main and only reference.
    This is because it is the only volume with 10um resolution.
    """
    additional_references = {}
    for modality in modalities:
        reference_path, _, volume_resolution = (
            get_reference_and_annotation_paths(
                download_dir_path, age, modality
            )
        )
        ref = load_any(reference_path)
        zoom_factors = tuple(volume_resolution / resolution for _ in range(3))
        ref = zoom(ref, zoom_factors, order=1)
        additional_references[modality] = ref
    return additional_references


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.
    Atlas is symmetrical, so hemisphere map is None.

    Returns
    -------
        None: Atlas is symmetrical
    """
    return None


def retrieve_structure_information(download_path):
    """Use the allen ccf 2022 ontology via the allen ontology api."""
    spacecache = ReferenceSpaceCache(
        manifest=download_path / "manifest.json",
        # downloaded files are stored relative to here
        resolution=resolution,
        reference_space_key="annotation/ccf_2022",
        # use the latest version of the CCF
    )
    # Download structures tree and meshes:
    ######################################
    oapi = OntologiesApi()  # ontologies
    struct_tree = spacecache.get_structure_tree()  # structures tree

    # Find id of set of regions with mesh:
    select_set = (
        "Structures whose surfaces are represented by a precomputed mesh"
    )
    mesh_set_ids = [
        s["id"]
        for s in oapi.get_structure_sets()
        if s["description"] == select_set
    ]
    structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)
    return structs_with_mesh


age_specific_root_dir = None

if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / NAME
    bg_root_dir.mkdir(exist_ok=True)
    download_resources(
        download_dir_path=bg_root_dir,
        atlas_name=NAME,
        atlas_file_url=data_file_url,
    )
    for age in range(4, 57):
        if age != 4:
            shutil.rmtree(age_specific_root_dir)
        age_specific_root_dir = bg_root_dir / f"P{age}"
        age_specific_root_dir.mkdir(exist_ok=True)
        for resolution, modalities in resolution_to_modalities.items():
            if resolution != list(resolution_to_modalities.keys())[0]:
                shutil.rmtree(age_specific_root_dir)
            age_specific_root_dir.mkdir(exist_ok=True)
            reference_volume, annotated_volume = (
                retrieve_reference_and_annotation(
                    bg_root_dir, age, resolution, modalities[0]
                )
            )
            if len(modalities) > 1:
                additional_references = retrieve_additional_references(
                    bg_root_dir, age, resolution, modalities[1:]
                )
            hemispheres_stack = retrieve_hemisphere_map()
            structures = retrieve_structure_information(bg_root_dir)
            meshes_dict = construct_meshes_from_annotation(
                age_specific_root_dir,
                annotated_volume,
                structures,
                decimate_fraction=0.5,
                closing_n_iters=1,
            )
            current_name = f"{NAME}_p{age}"

            structures = [
                {k: s[k] for k in TEMPLATE_KEYS if k in s} for s in structures
            ]
            output_filename = wrapup_atlas_from_data(
                atlas_name=current_name,
                atlas_minor_version=VERSION,
                citation=CITATION,
                atlas_link=ATLAS_LINK,
                species=SPECIES,
                resolution=(resolution,) * 3,
                orientation=ORIENTATION,
                root_id=ROOT_ID,
                reference_stack=reference_volume,
                annotation_stack=annotated_volume,
                structures_list=structures,
                meshes_dict=meshes_dict,
                working_dir=bg_root_dir,
                hemispheres_stack=None,
                cleanup_files=False,
                compress=True,
                scale_meshes=True,
                additional_references=additional_references,
            )
        meshes_dict = None
