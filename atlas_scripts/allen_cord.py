"""Script to package the Allen Cord Atlas."""

__version__ = "1"

import json
import time
from pathlib import Path
from random import choices

import numpy as np
import pandas as pd
import pooch
import tifffile
from loguru import logger
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

TEST = False

ATLAS_NAME = "allen_cord"
SPECIES = "Mus musculus"
ATLAS_LINK = "https://data.mendeley.com/datasets/4rrggzv5d5/1"
CITATION = "Fiederling et al. 2021, https://doi.org/10.1101/2021.05.06.443008"
ORIENTATION = "asr"
RESOLUTION = (20, 10, 10)
ROOT_ID = 250
ATLAS_FILE_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/4rrggzv5d5-1.zip"
ATLAS_PACKAGER = "MetaCell LLC, Ltd."


def download_atlas_files(download_dir_path: Path, atlas_file_url: str) -> Path:
    """Download and extract the Allen Cord Atlas files.

    Parameters
    ----------
    download_dir_path : Path
        The path to the directory where the files will be downloaded.
    atlas_file_url : str
        The URL of the atlas zip file.

    Returns
    -------
    Path
        The path to the extracted atlas files directory.
    """
    utils.check_internet_connection()

    pooch.retrieve(
        url=atlas_file_url,
        known_hash="4e8d592c78d1613827fa7bc524f215dc0fe7c7e5049fb31be6d3e4b3822852f7",
        path=download_dir_path,
        progressbar=True,
        processor=pooch.Unzip(extract_dir=""),
    )

    atlas_files_dir = download_dir_path / "SC_P56_Atlas_10x10x20_v5_2020"

    return atlas_files_dir


def parse_structures(structures_file, root_id):
    """
    Parse a CSV file containing structure information and organizes it
    into a hierarchical structure.

    Parameters
    ----------
    structures_file : str
        Path to the CSV file containing structure data.
    root_id : int
        The ID of the root structure for the hierarchy.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its ID, name, acronym, parent, RGB triplet, and structure ID path.
    """
    df = pd.read_csv(structures_file)
    df = df.rename(columns={"parent_ID": "parent_structure_id"})
    df = df.drop(
        columns=[
            "output_id",
            "parent_acronym",
            "children_acronym",
            "children_IDs",
        ]
    )

    df["rgb_triplet"] = df.apply(lambda x: [x.red, x.green, x.blue], axis=1)
    df["structure_id_path"] = df.apply(lambda x: [x.id], axis=1)

    df = df.drop(columns=["red", "green", "blue"])

    structures = df.to_dict("records")
    structures = create_structure_hierarchy(structures, df, root_id)
    return structures


def create_structure_hierarchy(structures, df, root_id):
    """
    Create a hierarchical structure ID path for each structure in the list.

    This function iterates through the given structures and constructs
    a full path of parent structure IDs for each, leading up to the root_id.
    It also renames the root structure.

    Parameters
    ----------
    structures : list
        A list of dictionaries, where each dictionary represents a structure
        and must contain 'id', 'parent_structure_id', and 'structure_id_path'
        keys.
    df : pandas.DataFrame
        A DataFrame containing the original structure data, used to look up
        parent IDs by structure ID. Must contain 'id' and
        'parent_structure_id' columns.
    root_id : int
        The ID of the root structure in the hierarchy.

    Returns
    -------
    list
        The modified list of structures, with updated 'structure_id_path'
        and the 'parent_structure_id' key removed.
    """
    for structure in structures:
        if structure["id"] != root_id:
            parent_id = structure["parent_structure_id"]
            while True:
                structure["structure_id_path"] = [parent_id] + structure[
                    "structure_id_path"
                ]
                if parent_id != root_id:
                    parent_id = int(
                        df[df["id"] == parent_id]["parent_structure_id"]
                    )
                else:
                    break
        else:
            structure["name"] = "root"
            structure["acronym"] = "root"

        del structure["parent_structure_id"]

    return structures


def create_meshes(download_dir_path, structures, annotated_volume, root_id):
    """
    Generate 3D meshes for brain regions from an annotated volume.

    This function iterates through the structure hierarchy, creates a mesh for
    each region that has corresponding labels in the annotated volume,
    and saves these meshes to a specified directory.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The base directory where meshes will be stored (a 'meshes'
        subdirectory will be created within it).
    structures : list
        A list of dictionaries, where each dictionary represents a structure
        with its ID and other metadata. This list is used to build the
        structure tree.
    annotated_volume : numpy.ndarray
        A 3D NumPy array representing the annotated brain volume, where each
        voxel contains a label corresponding to a brain region.
    root_id : int
        The ID of the root structure in the hierarchy, used for building
        the structure tree.

    Returns
    -------
    pathlib.Path
        The path to the directory where the generated meshes are saved.
    """
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
    smooth = False  # smooth meshes after creation
    start = time.time()

    # check how many regions to create the meshes for
    nodes = list(tree.nodes.values())
    if TEST:
        logger.info(
            "Creating atlas in test mode: selecting 10 "
            "random regions for mesh creation"
        )
        nodes = choices(nodes, k=10)

    print(f"Creating {len(nodes)} meshes")
    for node in track(
        nodes,
        total=len(nodes),
        description="Creating meshes",
    ):
        create_region_mesh(
            (
                meshes_dir_path,
                node,
                tree,
                labels,
                annotated_volume,
                root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    return meshes_dir_path


def create_mesh_dict(structures, meshes_dir_path):
    """
    Create a dictionary of structure IDs to mesh file paths for structures
    that have a valid mesh file.

    This function iterates through a list of structures, checks if a
    corresponding mesh file exists for each structure ID in the specified
    directory, and verifies that the mesh file is not empty. Only structures
    with valid, non-empty mesh files are included in the output.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries, where each dictionary represents a structure
        and must contain an 'id' key.
    meshes_dir_path : pathlib.Path
        The path to the directory where the mesh (.obj) files are stored.

    Returns
    -------
    tuple
        A tuple containing:
        - meshes_dict (dict): A dictionary where keys are structure IDs (int)
          and values are pathlib.Path objects pointing to the corresponding
          mesh files.
        - structures_with_mesh (list): A filtered list of structure
          dictionaries that have associated valid mesh files.
    """
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
    return meshes_dict, structures_with_mesh


def create_atlas(working_dir):
    """
    Package the Allen Cord Atlas from source files.

    This function orchestrates the entire atlas generation process,
    including downloading raw data, parsing structure metadata,
    generating 3D meshes for brain regions, and finally packaging
    all components into a BrainGlobe atlas file.

    Parameters
    ----------
    working_dir : Path
        The path to the directory where intermediate files will be stored
        and the final atlas will be saved.

    Returns
    -------
    Path
        The path to the generated BrainGlobe atlas file.
    """
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # Download atlas files from Mendeley
    atlas_files_dir = download_atlas_files(download_dir_path, ATLAS_FILE_URL)

    # Load files
    structures_file = atlas_files_dir / "Atlas_Regions.csv"
    reference_file = atlas_files_dir / "Template.tif"
    annotations_file = atlas_files_dir / "Annotation.tif"
    segments_file = atlas_files_dir / "Segments.csv"

    annotated_volume = tifffile.imread(annotations_file)
    template_volume = tifffile.imread(reference_file)

    atlas_segments = pd.read_csv(segments_file)
    atlas_segments = dict(atlas_segments=atlas_segments.to_dict("records"))

    # Parse structure metadata
    structures = parse_structures(structures_file, ROOT_ID)

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # Create meshes:
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = create_meshes(
        download_dir_path, structures, annotated_volume, ROOT_ID
    )
    meshes_dict, structures_with_mesh = create_mesh_dict(
        structures, meshes_dir_path
    )

    # Set black meshes to white
    for structure in structures_with_mesh:
        if structure["rgb_triplet"] == [0, 0, 0]:
            structure["rgb_triplet"] = [255, 255, 255]

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_metadata=atlas_segments,
    )

    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    # package atlas
    print(f'Creating atlas and saving it at "{bg_root_dir}"')
    create_atlas(bg_root_dir)
