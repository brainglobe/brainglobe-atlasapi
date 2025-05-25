import json
import shutil
import zipfile
from pathlib import Path

import nrrd
import numpy as np
import requests
from brainglobe_utils.IO.image import load_any
from scipy.ndimage import zoom
from tqdm import tqdm

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

VERSION = 0
NAME = "ccfv3augmented_mouse"
CITATION = "https://doi.org/10.1101/2024.11.06.622212"
ATLAS_LINK = "https://zenodo.org/api/records/14034334/files-archive"
SPECIES = "Mus musculus"
ORIENTATION = "asr"
ROOT_ID = 997
ATLAS_PACKAGER = "Harry Carey"


def download_resources(download_dir_path, atlas_file_url, atlas_name):
    """
    Download the necessary resources for the atlas.

    Pooch for some reason refused to download the entire file.
    """
    if "download=1" not in atlas_file_url:
        atlas_file_url = atlas_file_url + "?download=1"
    file_path = download_dir_path / f"{atlas_name}-files-archive.zip"

    if not file_path.exists():
        # Pooch for some reason refused to download the entire file.
        response = requests.get(atlas_file_url, stream=True)
        response.raise_for_status()
        # Get file size in bytes (21.5GB)
        total_size = int(21.5 * 1024 * 1024 * 1024)
        with tqdm(
            total=total_size, unit="B", unit_scale=True, desc="Downloading"
        ) as pbar:
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):  # 1MB chunks
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(download_dir_path)
    zipped_template_dir = download_dir_path / "average_nissl_template.zip"
    if not (download_dir_path / "nissl_average_full.nii.gz").exists():
        with zipfile.ZipFile(zipped_template_dir, "r") as zip_ref:
            zip_ref.extractall(download_dir_path)


def retrieve_reference_and_annotation(download_path, resolution):
    """
    Retrieve the desired reference and annotation as two numpy arrays.
    When the annotation is 10 microns the authors only provided a hemi
    segmentation. I checked with the lead author and he confirmed this was
    accidental. Here I correct this by mirroring if the resolution is 10um.
    The developers only provided a 10um population average template so I
    downsample here.
    """
    reference_path = download_path / "nissl_average_full.nii.gz"
    reference = load_any(reference_path)
    if resolution == 25:
        reference = zoom(reference, 0.4, order=1)
    reference = (reference * 65535).astype(np.uint16)
    reference = reference.transpose(1, 2, 0)[::-1, ::-1]

    annotation, header = nrrd.read(
        download_path / f"annotv3a_bbp_{resolution}.nrrd"
    )
    if resolution == 10:
        # mirror the volume
        new_annot = np.zeros(reference.shape)
        new_annot[:, :, 570:] = annotation
        new_annot[:, :, :570] = annotation[:, :, ::-1]
        annotation = new_annot
    return reference, annotation


def retrieve_additional_references(download_path, resolution):
    """
    This returns the single animal nissl and converts it
    to 16 bit unsigned int which bg requires.
    """

    additional_reference_path = (
        download_path / f"arav3a_bbp_nisslCOR_{resolution}.nrrd"
    )
    reference, header = nrrd.read(additional_reference_path)
    reference = reference * 65535
    reference = reference.astype(np.uint16)
    return {"single_animal_nissl": reference}


def retrieve_hemisphere_map():
    """
    At least one of the templates is technically not
    symmetrical as in pixels differ in left vs right
    hemisphere. But it is registered to a symmetrical
    template meaning there should not be morphological
    differences.
    """
    return None


def hex_to_rgb(hex_string):
    """Convert hex color string to RGB list."""
    hex_string = hex_string.lstrip("#")
    return [int(hex_string[i : i + 2], 16) for i in (0, 2, 4)]


def flatten_structure_tree(node, structure_id_path=None):
    """Recursively flatten the tree structure."""
    if structure_id_path is None:
        structure_id_path = []

    # Create current path
    current_path = structure_id_path + [node["id"]]

    # Create entry for current node
    entry = {
        "id": node["id"],
        "name": node["name"],
        "acronym": node["acronym"],
        "structure_id_path": current_path,
        "rgb_triplet": hex_to_rgb(node["color_hex_triplet"]),
    }

    # Start with current node's data
    entries = [entry]

    # Recursively process children
    if "children" in node:
        for child in node["children"]:
            entries.extend(flatten_structure_tree(child, current_path))

    return entries


def retrieve_structure_information(download_path):
    """
    The authors provide a json which is nested so I unest it.
    """
    with open(download_path / "hierarchy_bbp_atlas_pipeline.json") as f:
        metadata_json = json.load(f)

    # Get the root of the hierarchy
    root = metadata_json["msg"][0]

    # Flatten the tree structure
    flattened_structures = flatten_structure_tree(root)

    return flattened_structures


def retrieve_or_construct_meshes(download_path, annotated_volume, structures):
    """
    I use the construct_meshes_from_annotation helper function introduced in
    this pull request
    """
    meshes_dict = construct_meshes_from_annotation(
        download_path, annotated_volume, structures, ROOT_ID
    )
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / NAME
    bg_root_dir.mkdir(exist_ok=True)
    # download_resources(
    #     download_dir_path=bg_root_dir,
    #     atlas_name=NAME,
    #     atlas_file_url=ATLAS_LINK,
    # )
    for resolution in [10, 25]:
        reference_volume, annotated_volume = retrieve_reference_and_annotation(
            bg_root_dir, resolution=resolution
        )
        additional_references = retrieve_additional_references(
            bg_root_dir, resolution=resolution
        )

        hemispheres_stack = retrieve_hemisphere_map()
        structures = retrieve_structure_information(bg_root_dir)
        meshes_dict = retrieve_or_construct_meshes(
            bg_root_dir, annotated_volume, structures
        )
        reference_volume = (reference_volume * 65535).astype(np.uint16)
        output_filename = wrapup_atlas_from_data(
            atlas_name=NAME,
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
            additional_metadata={"atlas_packager": ATLAS_PACKAGER},
        )
        # its important we clear the mesh folder between loops in
        # case one resolution creates more regions than the other
        shutil.rmtree(bg_root_dir / "meshes")
