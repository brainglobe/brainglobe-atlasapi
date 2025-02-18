from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any
from rich.progress import track
from scipy.ndimage import zoom

from brainglobe_atlasapi import BrainGlobeAtlas, utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


@dataclass
class AtlasMetadata:
    version: int
    name: str
    citation: str
    species: str
    atlas_link: tuple[str, str]
    orientation: str
    root_id: int


# Define the atlas metadata
METADATA = AtlasMetadata(
    version=0,
    name="demba_dev_mouse",
    citation="https://doi.org/10.1101/2024.06.14.598876",
    atlas_link="https://doi.org/10.25493/V3AH-HK7",
    species="Mus musculus",
    orientation="rsa",
    root_id=997,
)

data_file_url = (
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/"
    "buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_segmentations",
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/"
    "buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_volumes",
)
resolution_to_modalities = {
    25: ["stpt", "mri", "lsfm", "allen_stpt"],
    10: ["allen_stpt"],
    20: ["stpt", "allen_stpt"],
}


def download_resources(download_dir_path, atlas_file_url, atlas_name):
    """
    Slight issue that the hash seems different each time. I think the
    files are being zipped on the server each time we request and it's
    changing the hash somehow (Maybe date and time is encoded in the
    file when zipped)
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

    Returns:
        tuple: A tuple containing the reference path and annotation path.
    """
    base_path = f"{download_dir_path}/{METADATA.name}/"
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
    we unfortunately did not provide 25um segmentations so
    we will just downsample the 20um ones
    Returns:
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
    The additional references are MRI LSFM and Allen STPT.
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

    Returns:
        None: Atlas is symmetrical
    """


def retrieve_structure_information():
    """
    Retrieve the structures tree and meshes for the Allen mouse brain atlas.

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    # Since this atlas inherits from the allen can we not simply get the data
    # from the bgapi?
    print("determining structures")
    allen_atlas = BrainGlobeAtlas("allen_mouse_25um")
    allen_structures = allen_atlas.structures_list
    allen_structures = [
        {
            "id": i["id"],
            "name": i["name"],
            "acronym": i["acronym"],
            "structure_id_path": i["structure_id_path"],
            "rgb_triplet": i["rgb_triplet"],
        }
        for i in allen_structures
    ]
    return allen_structures


def retrieve_or_construct_meshes(
    structures, annotated_volume, download_dir_path
):
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. We construct the meshes ourselves for this atlas, as the
    original data does not provide precomputed meshes.
    """
    print("constructing meshes")
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)
    tree = get_structures_tree(structures)
    labels = np.unique(annotated_volume).astype(np.int32)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2
    decimate_fraction = 0.2
    smooth = False
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
                METADATA.root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )
    # Create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )
    return meshes_dict


if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / METADATA.name
    bg_root_dir.mkdir(exist_ok=True)
    download_resources(
        download_dir_path=bg_root_dir,
        atlas_name=METADATA.name,
        atlas_file_url=data_file_url,
    )
    meshes_dict = None

    for age in range(4, 57):
        age_specific_root_dir = bg_root_dir / f"P{age}"
        age_specific_root_dir.mkdir(exist_ok=True)
        for resolution, modalities in resolution_to_modalities.items():
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
            structures = retrieve_structure_information()
            if meshes_dict is None:
                if resolution != 25:
                    raise (
                        """"
                        The order or resolutions is wrong,
                        25um should be first since its the most
                        efficient to produce (10um is far too slow)
                        """
                    )
            # generate pixel-scale mesh files only once, for 25um, and
            # re-use them and the meshes_dict
            if meshes_dict is None:
                if resolution != 25:
                    raise (
                        """"
                        The order or resolutions is wrong,
                        25um should be first since its the most
                        efficient to produce (10um is far too slow)
                        """
                    )
                meshes_dict = retrieve_or_construct_meshes(
                    structures, annotated_volume, age_specific_root_dir
                )
            current_name = f"{METADATA.name}_p{age}_{modalities[0]}"
            output_filename = wrapup_atlas_from_data(
                atlas_name=current_name,
                atlas_minor_version=METADATA.version,
                citation=METADATA.citation,
                atlas_link=METADATA.atlas_link,
                species=METADATA.species,
                resolution=(resolution,) * 3,
                orientation=METADATA.orientation,
                root_id=METADATA.root_id,
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
