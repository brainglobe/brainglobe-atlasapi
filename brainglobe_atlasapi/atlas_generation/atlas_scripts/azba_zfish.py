#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate a Brainglobe compatible atlas object
for the Adult Zebrafish Brain Atlas (AZBA)

@author: Kailyn Fields, kailyn.fields@wayne.edu
"""

__version__ = "1"

import csv
import multiprocessing as mp
import tarfile
import time
from pathlib import Path

import numpy as np
import tifffile
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

PARALLEL = False  # Disable for debugging mesh creation


def create_atlas(working_dir, resolution):
    # metadata
    ATLAS_NAME = "azba_zfish"
    SPECIES = "Danio rerio"
    ATLAS_LINK = "http://www.azba.wayne.edu"
    CITATION = "Kenney et al. 2021, https://doi.org/10.7554/elife.69988"
    ATLAS_FILE_URL = "http://www.azba.wayne.edu/2021-08-22_AZBA.tar.gz"
    ORIENTATION = "las"
    ROOT_ID = 9999
    ATLAS_PACKAGER = "Kailyn Fields, kailyn.fields@wayne.edu"
    ADDITIONAL_METADATA = {}

    # setup folder for downloading
    working_dir = working_dir / ATLAS_NAME
    working_dir.mkdir(exist_ok=True)
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / f"{ATLAS_NAME}"

    # download atlas files
    utils.check_internet_connection()
    destination_path = download_dir_path / "atlas_download"
    utils.retrieve_over_http(ATLAS_FILE_URL, destination_path)

    # unpack the atlas download folder
    tar = tarfile.open(destination_path)
    tar.extractall(path=atlas_path)
    tar.close()
    destination_path.unlink()

    print("Atlas files download completed")

    # paths
    structures_file = atlas_path / "2021-08-22_AZBA_labels.csv"
    annotations_file = atlas_path / "2021-08-22_AZBA_segmentation.tif"
    reference_topro = atlas_path / "20180219_AZBA_topro_average_2020.tif"
    reference_file = atlas_path / "20180628_AZBA_AF_average.tif"
    meshes_dir_path = atlas_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    # adding topro image as additional reference file,
    # main reference file is autofl
    topro = tifffile.imread(reference_topro)
    ADDITIONAL_REFERENCES = {"TO-PRO": topro}

    # open structures.csv and prep for dictionary parsing
    print("Creating structure tree")
    zfishFile = open(structures_file)
    zfishDictReader = csv.DictReader(zfishFile)

    # empty list to populate with dictionaries
    hierarchy = []

    # parse through csv file and populate hierarchy list
    for row in zfishDictReader:
        hierarchy.append(row)

    # make string to int and list of int conversions in
    # 'id', 'structure_id_path', and 'rgb_triplet' key values
    for i in range(0, len(hierarchy)):
        hierarchy[i]["id"] = int(hierarchy[i]["id"])
    for j in range(0, len(hierarchy)):
        hierarchy[j]["structure_id_path"] = list(
            map(int, hierarchy[j]["structure_id_path"].split("/"))
        )
    for k in range(0, len(hierarchy)):
        try:
            hierarchy[k]["rgb_triplet"] = list(
                map(int, hierarchy[k]["rgb_triplet"].split("/"))
            )
        except ValueError:
            hierarchy[k]["rgb_triplet"] = [255, 255, 255]

    # remove clear label (id 0) from hierarchy.
    # ITK-Snap uses this to label unlabeled areas,
    # but this convention interferes with the root mask generation
    # and is unnecessary for this application
    hierarchy.remove(hierarchy[1])

    # use tifffile to read annotated file
    annotated_volume = tifffile.imread(annotations_file)

    print(f"Saving atlas data at {atlas_path}")
    tree = get_structures_tree(hierarchy)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # mesh creation
    closing_n_iters = 2
    start = time.time()

    decimate_fraction = 0.3
    smooth = True

    if PARALLEL:
        print("Multiprocessing mesh creation...")
        pool = mp.Pool(int(mp.cpu_count() / 2))

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
                        ROOT_ID,
                        closing_n_iters,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            pass

    else:
        print("Multiprocessing disabled")
        # nodes = list(tree.nodes.values())
        # nodes = choices(nodes, k=10)
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
                    ROOT_ID,
                    closing_n_iters,
                    decimate_fraction,
                    smooth,
                )
            )

    print(
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in hierarchy:
        # check if a mesh was created
        mesh_path = meshes_dir_path / f"{s['id']}.obj"
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it.")
            continue
        else:
            # check that the mesh actually exists and isn't empty
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue
        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    # import reference file with tifffile so
    # it can be read in wrapup_atlas_from_data
    reference = tifffile.imread(reference_file)
    # inspect_meshes_folder(meshes_dir_path)
    # wrap up atlas file
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(resolution,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference,
        annotation_stack=annotations_file,
        structures_list=hierarchy,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
        additional_references=ADDITIONAL_REFERENCES,
    )

    return output_filename


if __name__ == "__main__":
    resolution = 4

    # generated atlas path
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir, resolution)
