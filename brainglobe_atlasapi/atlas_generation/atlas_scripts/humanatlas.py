import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import treelib
import urllib3
from allensdk.core.structure_tree import StructureTree
from brainio import brainio
from rich.progress import track

# import sys
# sys.path.append("./")
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
    inspect_meshes_folder,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def prune_tree(tree):
    nodes = tree.nodes.copy()
    for key, node in nodes.items():
        if node.tag == "root":
            continue
        if node.data.has_label:
            try:
                children = tree.children(node.identifier)
            except treelib.exceptions.NodeIDAbsentError:
                continue

            if children:
                for child in children:
                    try:
                        tree.remove_node(child.identifier)
                    except treelib.exceptions.NodeIDAbsentError:
                        pass
        else:
            # Remove if none of the children has mesh
            try:
                subtree = tree.subtree(node.identifier)
            except treelib.exceptions.NodeIDAbsentError:
                continue
            else:
                if not np.any(
                    [c.data.has_label for _, c in subtree.nodes.items()]
                ):
                    tree.remove_node(node.identifier)
    return tree


if __name__ == "__main__":
    PARALLEL = False  # disable parallel mesh extraction for easier debugging
    TEST = False

    # ----------------- #
    #   PREP METADATA   #
    # ----------------- #
    RES_UM = 500
    VERSION = 1
    ATLAS_NAME = "allen_human"
    SPECIES = "Homo sapiens"
    ATLAS_LINK = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/"
    CITATION = "Ding et al 2016, https://doi.org/10.1002/cne.24080"
    ORIENTATION = "ipr"

    # ------------------ #
    #   PREP FILEPATHS   #
    # ------------------ #

    data_fld = Path(
        r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy"
        r"\Atlases\atlasesforbrainrender\AllenHuman"
    )

    annotations_image = data_fld / "annotation.nii"
    anatomy_image = (
        data_fld
        / "mni_icbm152_nlin_sym_09b"
        / "mni_icbm152_pd_tal_nlin_sym_09b_hires.nii"
    )

    # Generated atlas path:
    bg_root_dir = Path.home() / ".brainglobe"
    bg_root_dir.mkdir(exist_ok=True)

    # Temporary folder for nrrd files download:
    temp_path = Path(r"C:\Users\Federico\.brainglobe\humanev")
    temp_path.mkdir(exist_ok=True)

    # Temporary folder for files before compressing:
    uncompr_atlas_path = temp_path / ATLAS_NAME
    uncompr_atlas_path.mkdir(exist_ok=True)

    # ---------------- #
    #   GET TEMPLATE   #
    # ---------------- #
    annotation = brainio.load_any(annotations_image)  # shape (394, 466, 378)
    anatomy = brainio.load_any(anatomy_image)  # shape (394, 466, 378)

    # Remove weird artefact
    annotation = annotation[:200, :, :]
    anatomy = anatomy[:200, :, :]

    # show(Volume(root_annotation), axes=1)

    # ------------------------ #
    #   STRUCTURES HIERARCHY   #
    # ------------------------ #
    # Download structure tree
    #########################

    # RMA query to fetch structures for the structure graph
    query_url = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::Structure"
    query_url += ",rma::criteria,[graph_id$eq%d]" % 16
    query_url += (
        ",rma::options[order$eq'structures.graph_order'][num_rows$eqall]"
    )

    http = urllib3.PoolManager()
    r = http.request("GET", query_url)
    data = json.loads(r.data.decode("utf-8"))["msg"]
    structures = pd.read_json(json.dumps(data))

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
    meshes_dir_path = uncompr_atlas_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

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

    # tree.show(data_property='has_label')

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
    if PARALLEL:
        print("Starting mesh creation in parallel")

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
                        annotation,
                        ROOT_ID,
                        closing_n_iters,
                        decimate_fraction,
                        smooth,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            # error with returning results from pool.map but we don't care
            pass
    else:
        print("Starting mesh creation")

        for node in track(
            tree.nodes.values(),
            total=tree.size(),
            description="Creating meshes",
        ):
            if node.tag == "root":
                volume = annotation.copy()
                volume[volume > 0] = node.identifier
            else:
                volume = annotation

            create_region_mesh(
                (
                    meshes_dir_path,
                    node,
                    tree,
                    labels,
                    volume,
                    ROOT_ID,
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

    # Create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in regions_list:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            # print(f"No mesh file exists for: {s['name']}")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                # print(f"obj file for {s['name']} is too small.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    # ----------- #
    #   WRAP UP   #
    # ----------- #

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=VERSION,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RES_UM,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=anatomy,
        annotation_stack=annotation,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
    )
