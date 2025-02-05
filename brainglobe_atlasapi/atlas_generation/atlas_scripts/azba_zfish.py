"""
Script to generate a Brainglobe compatible atlas object
for the Adult Zebrafish Brain Atlas (AZBA)

@author: Kailyn Fields, kailyn.fields@wayne.edu
"""

__version__ = "2"

import csv
import time

import numpy as np
import pooch
import tifffile
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "azba_zfish"
SPECIES = "Danio rerio"
ATLAS_LINK = "http://www.azba.wayne.edu"
CITATION = "Kenney et al. 2021, https://doi.org/10.7554/elife.69988"
ATLAS_FILE_URL = "http://www.azba.wayne.edu/2021-08-22_AZBA.tar.gz"
ORIENTATION = "las"
ROOT_ID = 9999
ATLAS_PACKAGER = "Kailyn Fields, kailyn.fields@wayne.edu"
ADDITIONAL_METADATA = {}
RESOLUTION = 4


def create_atlas(working_dir, resolution):

    download_path = working_dir / "downloads"
    download_path.mkdir(exist_ok=True, parents=True)

    atlas_path = pooch.retrieve(
        url=ATLAS_FILE_URL,
        known_hash="a14b09b88979bca3c06fa96d525e6c1ba8906fe08689239433eb72d8d3e2ba44",
        path=download_path,
        progressbar=True,
        processor=pooch.Untar(extract_dir="."),
    )

    print("Atlas files download completed")

    # paths
    structures_file = download_path / "2021-08-22_AZBA_labels.csv"
    annotations_file = download_path / "2021-08-22_AZBA_segmentation.tif"
    reference_topro = download_path / "20180219_AZBA_topro_average_2020.tif"
    reference_file = download_path / "20180628_AZBA_AF_average.tif"
    meshes_dir_path = download_path / "meshes"
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
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    # generated atlas path
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    create_atlas(bg_root_dir, RESOLUTION)
