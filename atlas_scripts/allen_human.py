"""Module to package the Allen Human Reference Atlas."""

__version__ = "1"

import json
import time

import brainglobe_space as bgs
import numpy as np
import pandas as pd
import pooch
import treelib
import urllib3
from allensdk.core.structure_tree import StructureTree
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi import BrainGlobeAtlas, utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

RES_UM = 500
VERSION = 1
ATLAS_NAME = "allen_human"
SPECIES = "Homo sapiens"
ATLAS_LINK = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/"
CITATION = "Ding et al 2016, https://doi.org/10.1002/cne.24080"
ORIENTATION = "rpi"

### Settings


def prune_tree(tree):
    """
    Prunes the input tree based on the 'has_label' attribute of its nodes.

    A node is removed only when neither it nor any of its descendants has a
    label in the annotation volume.

    Parameters
    ----------
    tree : treelib.Tree
        The tree to be pruned, where each node's data contains a 'has_label'
        boolean attribute indicating if the region has a corresponding label
        in the annotation volume.

    Returns
    -------
    treelib.Tree
        The pruned tree.
    """
    nodes = tree.nodes.copy()
    for node in nodes.values():
        if node.identifier == tree.root:
            continue

        try:
            subtree = tree.subtree(node.identifier)
        except treelib.exceptions.NodeIDAbsentError:
            continue

        if not any(
            descendant.data.has_label for descendant in subtree.nodes.values()
        ):
            tree.remove_node(node.identifier)
    return tree


def download_atlas_files(download_dir_path, atlas_file_url):
    """
    Download the annotation file and anatomy template image for
    the Allen Human Reference Atlas.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The path to the directory where the files should be downloaded.
    atlas_file_url : str
        The URL for the full annotation NIfTI file (gzipped).

    Returns
    -------
    pathlib.Path
        The path to the directory where the files were downloaded.
    """
    utils.check_internet_connection()

    data_fld = download_dir_path

    # downloading and un-compressing full annotation file

    print("Downloading annotation file...")
    pooch.retrieve(
        url=atlas_file_url,
        known_hash="2b05581e39c44f2623d9b0a69f64e3df0823c20d054abef92973812313335dc3",
        path=download_dir_path,
        progressbar=True,
        processor=pooch.Decompress(name="annotation_full.nii"),
    )

    print("Download and decompression completed.")

    return data_fld


def create_atlas(working_dir):
    """
    Package the Allen Human Reference Atlas.

    This function downloads the necessary annotation and anatomy files,
    constructs the hierarchical structure tree, creates meshes for
    the brain regions, and finally packages all data into a BrainGlobe atlas.

    Parameters
    ----------
    working_dir : pathlib.Path
        The directory where all downloaded files and generated atlas data
        will be stored.

    Returns
    -------
    pathlib.Path
        The path to the generated BrainGlobe atlas file (e.g., a .zip file).
    """
    # ------------------ #
    #   PREP FILEPATHS   #
    # ------------------ #

    annotation_full_url = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/annotation_full.nii.gz"

    atlas_files_dir = download_atlas_files(working_dir, annotation_full_url)

    annotations_image = atlas_files_dir / "annotation_full.nii"

    # Temporary folder for nrrd files download:
    temp_path = working_dir
    temp_path.mkdir(exist_ok=True)

    # Temporary folder for files before compressing:
    uncompr_atlas_path = temp_path / ATLAS_NAME
    uncompr_atlas_path.mkdir(exist_ok=True)

    old_atlas = BrainGlobeAtlas("allen_human_500um")

    # ---------------- #
    #   GET TEMPLATE   #
    # ---------------- #

    template_metadata = old_atlas.metadata["template"]
    template_name = template_metadata["name"]
    template_version = template_metadata["version"]
    
    # Rotate the template to match the orientation of the annotation volume
    # template must be in the orientation the script declares
    template = bgs.AnatomicalSpace(
        old_atlas.orientation, shape=old_atlas.template.shape
    ).map_stack_to(ORIENTATION, old_atlas.template)

    template_info = {
        "name": template_name,
        "version": template_version,
        "use_existing": True,
    }

    # ---------------- #
    #   GET SPACE      #
    # ---------------- #

    space_metadata = old_atlas.metadata["coordinate_space"]
    space_name = space_metadata["name"]
    space_version = space_metadata["version"]

    space_info = {
        "name": space_name,
        "version": space_version,
        "use_existing": True,
    }

    # ---------------- #
    #   GET ANNOTATION #
    # ---------------- #

    annotation = load_nii(annotations_image)  # shape (394, 466, 378)
    annotation = np.asanyarray(annotation.dataobj).astype(
        np.uint32, copy=False
    )

    # ------------------------ #
    #   STRUCTURES HIERARCHY   #
    # ------------------------ #
    # Download structure tree
    #########################

    # RMA query to fetch structures for the structure graph
    query_url = "https://api.brain-map.org/api/v2/data/query.json?criteria=model::Structure"
    query_url += ",rma::criteria,[graph_id$eq%d]" % 16
    query_url += (
        ",rma::options[order$eq'structures.graph_order'][num_rows$eqall]"
    )

    http = urllib3.PoolManager()
    r = http.request("GET", query_url)
    data = json.loads(r.data.decode("utf-8"))["msg"]
    structures = pd.DataFrame(data)

    # Create empty list and collect all regions
    # traversing the regions hierarchy:
    regions_list = []

    for i, region in structures.iterrows():
        if i == 0:
            acronym = "root"
        else:
            acronym = region["acronym"]

        regions_list.append(
            {
                "name": region["name"],
                "acronym": acronym,
                "id": region["id"],
                "rgb_triplet": StructureTree.hex_to_rgb(
                    region["color_hex_triplet"]
                ),
                "structure_id_path": StructureTree.path_to_list(
                    region["structure_id_path"]
                ),
            }
        )
    ROOT_ID = regions_list[0]["id"]

    # ----------------- #
    #   CREATE MESHES   #
    # ----------------- #
    print(f"Saving atlas data at {uncompr_atlas_path}")

    tree = get_structures_tree(regions_list)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )

    # Mark which tree elements are in the annotation volume
    labels = np.unique(annotation).astype(np.int32)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # Remove nodes for which no mesh can be created
    tree = prune_tree(tree)
    print(
        f"After pruning: # of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )

    # Mesh creation
    closing_n_iters = 2
    decimate_fraction = 0.2
    smooth = False  # smooth meshes after creation
    start = time.time()
    annotated_volume = annotation

    print("Starting mesh creation")

    pruned_list = [
        region for region in regions_list if region["id"] in tree.nodes
    ]

    meshes_dict = construct_meshes_from_annotation(
        uncompr_atlas_path,
        annotated_volume,
        pruned_list,
        closing_n_iters,
        decimate_fraction,
        smooth,
    )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Retain every structure in the annotation-backed hierarchy, regardless
    # of whether mesh extraction succeeded. Missing meshes are reported by
    # atlas validation and must not remove structures from the terminology.
    structures_to_keep = [
        structure
        for structure in regions_list
        if structure["id"] in tree.nodes
    ]

    print(
        f"Retaining {len(structures_to_keep)} structures, "
        f"{len(meshes_dict)} of which have meshes"
    )

    # ----------- #
    #   WRAP UP   #
    # ----------- #

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=VERSION,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RES_UM,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=template,
        template_info=template_info,
        coordinate_space_info=space_info,
        annotation_stack=annotated_volume,
        structures_list=structures_to_keep,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        overwrite=True,
    )

    return output_filename


if __name__ == "__main__":
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)
    create_atlas(bg_root_dir)
