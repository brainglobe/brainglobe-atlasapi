__version__ = "1"

import json
import multiprocessing as mp
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
    inspect_meshes_folder,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

PARALLEL = True
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
    utils.check_internet_connection()

    pooch.retrieve(
        url=atlas_file_url,
        known_hash="4e8d592c78d1613827fa7bc524f215dc0fe7c7e5049fb31be6d3e4b3822852f7",
        path=download_dir_path,
        progressbar=True,
        processor=pooch.Unzip(extract_dir="."),
    )

    atlas_files_dir = download_dir_path / "SC_P56_Atlas_10x10x20_v5_2020"

    return atlas_files_dir


def parse_structures(structures_file, root_id):
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

    if PARALLEL:
        print(
            f"Creating {tree.size()} meshes in parallel with "
            f"{mp.cpu_count() - 2} CPU cores"
        )
        pool = mp.Pool(mp.cpu_count() - 2)

        try:
            pool.map(
                create_region_mesh,
                [
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
                    for node in nodes
                ],
            )
        except mp.pool.MaybeEncodingError:
            pass
    else:
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

    if TEST:
        # create visualization of the various meshes
        inspect_meshes_folder(meshes_dir_path)

    return meshes_dir_path


def create_mesh_dict(structures, meshes_dir_path):
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

    # generate atlas
    print(f'Creating atlas and saving it at "{bg_root_dir}"')
    create_atlas(bg_root_dir)
