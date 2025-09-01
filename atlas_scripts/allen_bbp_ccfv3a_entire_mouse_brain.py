"""
Generates the CCFv3a Augmented Mouse Brain Atlas for BrainGlobe.

It handles downloading, processing, and packaging the atlas data,
including reference volumes, annotations, and structure information,
from a Zenodo archive.
"""

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
    Download necessary resources for the atlas.

    Parameters
    ----------
    download_dir_path : Path
        Path to the directory where files will be downloaded.
    atlas_file_url : str
        URL to the Zenodo archive containing atlas files.
    atlas_name : str
        Name of the atlas, used for naming the downloaded zip file.

    Notes
    -----
    `pooch` was initially attempted for downloading, but it failed to download
    the entire file. Therefore, `requests` is used instead.
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
    Retrieve reference and annotation as NumPy arrays.

    Parameters
    ----------
    download_path : Path
        Path to the directory containing downloaded atlas files.
    resolution : int
        The desired resolution of the atlas in micrometers (e.g., 10 or 25).

    Returns
    -------
    numpy.ndarray
        The reference volume.
    numpy.ndarray
        The annotation volume.

    Notes
    -----
    For the 10-micron resolution, the provided annotation is only a
    hemi-segmentation. This function corrects it by mirroring the existing
    hemi-segmentation to create a full volume. The developers provided only
    a 10um population average template, which is downsampled when a
    25um resolution is requested.
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
    Retrieve additional reference volumes.

    Parameters
    ----------
    download_path : Path
        Path to the directory containing downloaded atlas files.
    resolution : int
        The desired resolution of the atlas in micrometers (e.g., 10 or 25).

    Returns
    -------
    dict of numpy.ndarray
        A dictionary containing additional reference volumes,
        e.g., {"single_animal_nissl": reference_volume}.

    Notes
    -----
    The single animal Nissl reference volume is converted to a 16-bit
    unsigned integer array as required by BrainGlobe.
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
    Retrieve hemisphere map for the atlas.

    Returns
    -------
    None
        A hemisphere map is not provided or required for this atlas,
        as the reference template is symmetrical.
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
    """Unnest the JSON structure provided by the author."""
    with open(download_path / "hierarchy_bbp_atlas_pipeline.json") as f:
        metadata_json = json.load(f)

    # Get the root of the hierarchy
    root = metadata_json["msg"][0]

    # Flatten the tree structure
    flattened_structures = flatten_structure_tree(root)

    return flattened_structures


def retrieve_or_construct_meshes(download_path, annotated_volume, structures):
    """
    Construct meshes for atlas structures.

    Parameters
    ----------
    download_path : Path
        Path to the directory containing downloaded atlas files.
    annotated_volume : numpy.ndarray
        The annotation volume used to generate meshes.
    structures : list of dict
        A list of dictionaries, where each dictionary represents a structure
        and contains its metadata (e.g., ID, name, acronym, color).

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to
        the generated mesh files.
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
    download_resources(
        download_dir_path=bg_root_dir,
        atlas_name=NAME,
        atlas_file_url=ATLAS_LINK,
    )
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
