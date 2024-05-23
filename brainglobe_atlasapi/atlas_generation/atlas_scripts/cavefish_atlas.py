__version__ = "0"

import dataclasses
import json
import csv
import tifffile
import multiprocessing as mp
import time
import tarfile
import zipfile
#from os import listdir, path
# p = Path('.')
from pathlib import Path
from typing import Tuple
import glob as glob
import os

import numpy as np
import pandas as pd
from brainglobe_atlasapi import utils
from brainglobe_atlasapi.structure_tree_util import get_structures_tree
from rich.progress import track
# from skimage import io

from brainglobe_atlasapi.atlas_generation.mesh_utils import Region, create_region_mesh
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

PARALLEL = True

def create_atlas (working_dir, resolution):
    ATLAS_NAME = "asty_atlas"
    SPECIES = "Astyanax mexicanus"
    ATLAS_LINK = "https://a-cavefishneuroevoluti.vev.site/lab-website"
    CITATION = "Kozol et al. 2023, https://elifesciences.org/articles/80777"
    ATLAS_FILE_URL = "https://cdn.vev.design/private/30dLuULhwBhk45Fm8dHoSpD6uG12/8epecj-asty-atlas.zip"
    ORIENTATION = "las"
    ROOT_ID = 999
    ATLAS_PACKAGER = "Robert Kozol, kozolrobert@gmail.com"
    ADDITIONAL_METADATA = {}

    # setup folder for downloading

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"

    # download atlas files
    utils.check_internet_connection()
    destination_path = download_dir_path / "atlas_download"
    utils.retrieve_over_http(ATLAS_FILE_URL, destination_path)

    # unpack the atlas download folder
    with zipfile.ZipFile(destination_path, 'r') as zip:
        zip.extractall(path=atlas_path)

    destination_path.unlink()

    structures_file = atlas_path / "asty_atlas/SPF2_25_Region_atlas_list.csv"
    annotations_volume = atlas_path / "asty_atlas/SPF2_regions_SP2c_1iWarp_25.tif"
    #reference_cartpt = atlas_path / "SPF2_carpt_ref.tif" #ADDITIONAL REFERENCE
    reference_file = atlas_path / "asty_atlas/SPF2_terk_ref.tif"
    meshes_dir_path = atlas_path / "asty_atlas/meshes"
    try:
        os.mkdir(meshes_dir_path)
    except:
        'mesh folder already exists'

    #cartpt = tifffile.imread(reference_cartpt)
    #ADDITIONAL_REFERENCES = {"cartpt": cartpt}

    # create dictionaries
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

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(resolution,) * 3,  # if isotropic - highly recommended
        orientation=ORIENTATION,
        root_id=999,
        reference_stack=reference_file,
        annotation_stack=annotations_volume,
        structures_list=structures_file,
        meshes_dir_path=meshes,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
    )

    return output_filename

res = 2
home = str(Path.home())
bg_root_dir = Path.home() / "bg-atlasgen"
bg_root_dir.mkdir(exist_ok=True, parents=True)

create_atlas (bg_root_dir, res)
