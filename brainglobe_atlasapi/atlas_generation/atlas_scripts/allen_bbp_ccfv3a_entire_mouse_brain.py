import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Union

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

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)


@dataclass
class AtlasMetadata:
    """
    Holds metadata describing a BrainGlobe atlas.

    Attributes:
        version (int): The minor version of the atlas (the first number after
            the decimal point).
        name (str): Atlas name, following "FirstAuthor_SpeciesCommonName" or
            "Institution_SpeciesCommonName".
        citation (str): A DOI of the most relevant citable document.
        species (str): The scientific name of the species.
        atlas_link (Union[str, Tuple[str, ...]]): URL(s) for the data files.
        orientation (str): The **original** atlas orientation in BrainGlobe
            convention.
        root_id (int): The ID of the highest atlas level. This is commonly
            called root or brain.
        resolution (Union[int, float]): The atlas resolution in microns.
    """

    version: int
    name: str
    citation: str
    species: str
    atlas_link: Union[str, Tuple[str, ...]]
    orientation: str
    root_id: int
    resolution: Union[int, float]


METADATA = AtlasMetadata(
    version=0,
    name="allen_bbp_ccfv3a_entire_mouse_brain",
    citation="https://doi.org/10.1101/2024.11.06.622212",
    atlas_link="https://zenodo.org/api/records/14034334/files-archive",
    species="Mus musculus",
    orientation="asr",
    root_id=997,
    resolution=25,
)


def download_resources(download_dir_path, atlas_file_url, atlas_name):
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
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

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    reference_path = download_path / f"arav3a_bbp_nisslCOR_{resolution}.nrrd"
    reference, header = nrrd.read(reference_path)
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
    """This function only needs editing if the atlas has additional reference
    images. It should return a dictionary that maps the name of each
    additional reference image to an image stack containing its data.
    """
    additional_reference_path = download_path / "nissl_average_full.nii.gz"
    reference = load_any(additional_reference_path)
    if resolution == 25:
        reference = zoom(reference, 0.4, order=1)
    return {"population_average_nissl": reference}


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
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
    Return a pandas DataFrame with flattened atlas structure information.
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
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    meshes_dict = construct_meshes_from_annotation(
        download_path, annotated_volume, structures, METADATA.root_id
    )
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / METADATA.name
    bg_root_dir.mkdir(exist_ok=True)
    download_resources(
        download_dir_path=bg_root_dir,
        atlas_name=METADATA.name,
        atlas_file_url=METADATA.atlas_link,
    )
    for resolution in [25, 10]:
        reference_volume, annotated_volume = retrieve_reference_and_annotation(
            bg_root_dir, resolution=resolution
        )
        additional_references = retrieve_additional_references(
            bg_root_dir, resolution=resolution
        )
        hemispheres_stack = retrieve_hemisphere_map()
        if resolution == 25:
            structures = retrieve_structure_information(bg_root_dir)
            meshes_dict = retrieve_or_construct_meshes(
                bg_root_dir, annotated_volume, structures
            )

        output_filename = wrapup_atlas_from_data(
            atlas_name=METADATA.name,
            atlas_minor_version=METADATA.version,
            citation=METADATA.citation,
            atlas_link=METADATA.atlas_link,
            species=METADATA.species,
            resolution=(METADATA.resolution,) * 3,
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
