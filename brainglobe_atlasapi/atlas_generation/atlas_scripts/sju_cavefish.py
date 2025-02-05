__version__ = "0"

import csv
import time
from pathlib import Path

import numpy as np
import pooch
import tifffile
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "sju_cavefish"
SPECIES = "Astyanax mexicanus"
ATLAS_LINK = "https://a-cavefishneuroevoluti.vev.site/lab-website"
CITATION = "Kozol et al. 2023, https://doi.org/10.7554/eLife.80777"
ATLAS_FILE_URL = "https://cdn.vev.design/private/30dLuULhwBhk45Fm8dHoSpD6uG12/35s9sm-asty-atlas.zip"
ORIENTATION = "sla"
ROOT_ID = 999
ATLAS_PACKAGER = "Robert Kozol, kozolrobert@gmail.com"
ADDITIONAL_METADATA = {}
RESOLUTION = 2, 2, 2


def create_atlas(working_dir, resolution):
    # setup folder for downloading

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"

    # download atlas files
    utils.check_internet_connection()
    pooch.retrieve(
        ATLAS_FILE_URL,
        known_hash="49f82a3b9fc107cf5d6e02c0ca8d34b9c13ddeff2305d30eced07fddc87a175a",
        processor=pooch.Unzip(extract_dir=atlas_path),
        progressbar=True,
    )

    structures_file = atlas_path / "asty_atlas/SPF2_25_Region_atlas_list.csv"
    annotations_file = (
        atlas_path / "asty_atlas/SPF2_regions_SP2c_1iWarp_25.tif"
    )
    reference_file = atlas_path / "asty_atlas/SPF2_terk_ref.tif"
    meshes_dir_path = atlas_path / "asty_atlas/meshes"

    # additional references (not in remote):
    reference_cartpt = atlas_path / "asty_atlas/SPF2_cartpt_ref.tif"

    Path(meshes_dir_path).mkdir(exist_ok=True)

    # create dictionaries
    print("Creating structure tree")
    with open(
        structures_file, mode="r", encoding="utf-8-sig"
    ) as cavefish_file:
        cavefish_dict_reader = csv.DictReader(cavefish_file)

        # empty list to populate with dictionaries
        hierarchy = []

        # parse through csv file and populate hierarchy list
        for row in cavefish_dict_reader:
            hierarchy.append(row)

    # make string to int and list of int conversions in
    # 'id', 'structure_id_path', and 'rgb_triplet' key values
    for i in range(0, len(hierarchy)):
        hierarchy[i]["id"] = int(hierarchy[i]["id"])
        hierarchy[i]["structure_id_path"] = list(
            map(int, hierarchy[i]["structure_id_path"].split("/"))
        )
        try:
            hierarchy[i]["rgb_triplet"] = list(
                map(int, hierarchy[i]["rgb_triplet"].split("/"))
            )
        except ValueError:
            hierarchy[i]["rgb_triplet"] = [255, 255, 255]

    # remove clear label (id 0) from hierarchy.
    # ITK-Snap uses this to label unlabeled areas,
    # but this convention interferes with the root mask generation
    # and is unnecessary for this application
    hierarchy.remove(hierarchy[1])

    # Set root mesh to white
    hierarchy[0]["rgb_triplet"] = [255, 255, 255]

    # use tifffile to read annotated file
    annotated_volume = tifffile.imread(annotations_file).astype(np.uint8)
    reference_volume = tifffile.imread(reference_file)

    # additional reference
    cartpt_volume = tifffile.imread(reference_cartpt)
    cartpt_volume -= np.min(
        cartpt_volume
    )  # shift cartpt to a non-negative range before converting to UINT16
    cartpt_volume = cartpt_volume.astype(np.uint16)
    ADDITIONAL_REFERENCES = {"cartpt": cartpt_volume}

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
    decimate_fraction = 0.3
    smooth = True

    start = time.time()

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
        resolution=resolution,
        orientation=ORIENTATION,
        root_id=999,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=hierarchy,
        meshes_dict=meshes_dict,
        scale_meshes=True,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
        additional_references=ADDITIONAL_REFERENCES,
    )

    return output_filename


if __name__ == "__main__":
    home = str(Path.home())
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, RESOLUTION)
