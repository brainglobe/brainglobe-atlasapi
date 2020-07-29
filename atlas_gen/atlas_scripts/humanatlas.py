import json
from rich.progress import track
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from pathlib import Path
import treelib
from brainio import brainio
import urllib3
from allensdk.core.structure_tree import StructureTree

# import sys

# sys.path.append("./")
from atlas_gen.mesh_utils import create_region_mesh, Region
from atlas_gen.wrapup import wrapup_atlas_from_data
from bg_atlasapi.structure_tree_util import get_structures_tree


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

    # ---------------------------------------------------------------------------- #
    #                                 PREP METADATA                                #
    # ---------------------------------------------------------------------------- #
    RES_UM = 500
    VERSION = 1
    ATLAS_NAME = "allen_human"
    SPECIES = "Homo sapiens"
    ATLAS_LINK = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/"
    CITATION = "Ding et al 2020, https://doi.org/10.1002/cne.24080"
    ORIENTATION = "ipr"

    # ---------------------------------------------------------------------------- #
    #                                PREP FILEPATHS                                #
    # ---------------------------------------------------------------------------- #

    data_fld = Path(
        r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy\Atlases\atlasesforbrainrender\AllenHuman"
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

    # ---------------------------------------------------------------------------- #
    #                                 GET TEMPLATE                                 #
    # ---------------------------------------------------------------------------- #
    annotation = brainio.load_any(annotations_image)  # shape (394, 466, 378)
    anatomy = brainio.load_any(anatomy_image)  # shape (394, 466, 378)

    # Remove weird artefact
    annotation = annotation[:200, :, :]
    anatomy = anatomy[:200, :, :]

    # show(Volume(root_annotation), axes=1)

    # ---------------------------------------------------------------------------- #
    #                             STRUCTURES HIERARCHY                             #
    # ---------------------------------------------------------------------------- #
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

    # Create empty list and collect all regions traversing the regions hierarchy:
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

    # ---------------------------------------------------------------------------- #
    #                                CREATE MESHESH                                #
    # ---------------------------------------------------------------------------- #
    print(f"Saving atlas data at {uncompr_atlas_path}")
    meshes_dir_path = uncompr_atlas_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(regions_list)
    print(
        f"Number of brain regions: {tree.size()}, max tree depth: {tree.depth()}"
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
        f"After pruning: # of brain regions: {tree.size()}, max tree depth: {tree.depth()}"
    )

    # Mesh creation
    closing_n_iters = 2
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
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            pass  # error with returning results from pool.map but we don't care
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
                )
            )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

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
        f"In the end, {len(structures_with_mesh)} structures with mesh are kept"
    )

    # ---------------------------------------------------------------------------- #
    #                                    WRAP UP                                   #
    # ---------------------------------------------------------------------------- #

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


# ---------------------------------------------------------------------------- #
#                                   OLD CODE                                   #
# ---------------------------------------------------------------------------- #

#     # Create meshes
#     ###############
#     meshes_dir = uncompr_atlas_path / descriptors.MESHES_DIRNAME
#     meshes_dir.mkdir(exist_ok=True)

#     unique_values, unique_counts = np.unique(
#         annotation_whole, return_counts=True
#     )
#     voxel_counts = dict(zip(unique_values, unique_counts))
#     if 0 in voxel_counts:
#         del voxel_counts[0]
#     structures.set_index("id", inplace=True)

#     # Create root first
#     root = [s for s in regions_list if s["acronym"] == "root"][0]
#     root_idx = root["id"]
#     root_volume = volume_utils.create_masked_array(
#         annotation_whole, 0, greater_than=True
#     )
#     savepath = meshes_dir / f'{root["id"]}.obj'
#     if not savepath.exists():
#         root_mesh = mesh_utils.extract_mesh_from_mask(
#             root_volume, savepath, smooth=False, decimate=True
#         )
#     else:
#         root_mesh = load(str(savepath))

#     # Asses mesh extraction quality
#     # mesh_utils.compare_mesh_and_volume(root_mesh, root_volume)

#     # ? Create meshes for leaf nodes
#     start = time.time()
#     pool = mp.Pool(mp.cpu_count() - 2)
#     try:
#         pool.map(
#             create_structure_mesh,
#             [
#                 (structures, annotation_whole, meshes_dir, a)
#                 for a in voxel_counts
#             ],
#         )
#     except mp.pool.MaybeEncodingError:
#         pass  # error with returning results from pool.map but we don't care
#     print(
#         f"Creating meshes for {len(voxel_counts)} structures took: {round(time.time() - start, 3)}s"
#     )

#     # Show which regions were represented in the annotated volume
#     regions_with_mesh = [structures.loc[a, "acronym"] for a in voxel_counts]

#     tree = StructureTree(regions_list).get_structures_tree()

#     for key, node in tree.nodes.items():
#         if node.tag in regions_with_mesh:
#             has_mesh = True
#         else:
#             has_mesh = False
#         node.data = Region(has_mesh)

#     # Remove regions that are children to the ones that which
#     # were represented in the volume or were
#     # at least some of their children had a mesh
#     tree = prune_tree(tree)

#     # ? extract meshes for non leaf regions
#     id_to_acronym_map = {s["id"]: s["acronym"] for s in regions_list}
#     voxel_to_acro = {a: structures.loc[a, "acronym"] for a in voxel_counts}
#     acronym_to_voxel = {v: k for k, v in voxel_to_acro.items()}
#     non_leaf_nodes = [
#         s
#         for s in regions_list
#         if s["acronym"] != "root" and s["id"] not in voxel_counts
#     ]

#     start = time.time()
#     pool = mp.Pool(mp.cpu_count() - 2)
#     try:
#         pool.map(
#             create_nonleaf_structure_mesh,
#             [
#                 (
#                     nonleaf,
#                     meshes_dir,
#                     regions_list,
#                     id_to_acronym_map,
#                     acronym_to_voxel,
#                     annotation_whole,
#                 )
#                 for nonleaf in non_leaf_nodes
#             ],
#         )
#     except mp.pool.MaybeEncodingError:
#         pass  # error with returning results from pool.map but we don't care
#     print(
#         f"Creating meshes for {len(non_leaf_nodes)} structures took: {round(time.time() - start, 3)}s"
#     )

#     # ? Fill in more of the regions that don't have mesh yet
#     for repeat in range(4):
#         for idx, node in tree.nodes.items():
#             savepath = meshes_dir / f"{idx}.obj"
#             if not savepath.exists():
#                 region = [r for r in regions_list if r["id"] == idx][0]
#                 args = (
#                     region,
#                     meshes_dir,
#                     regions_list,
#                     id_to_acronym_map,
#                     acronym_to_voxel,
#                     annotation_whole,
#                 )
#                 create_nonleaf_structure_mesh(args)

#     # Update tree and check that everyone got a mesh
#     for idx, node in tree.nodes.items():
#         savepath = meshes_dir / f"{idx}.obj"
#         if savepath.exists():
#             node.data.has_mesh = True

#     tree.show(data_property="has_mesh")

#     print(
#         f"\n\nTotal number of structures left in tree: {tree.size()} - max depth: {tree.depth()}"
#     )

#     tree_regions = [node.identifier for k, node in tree.nodes.items()]
#     pruned_regions_list = [r for r in regions_list if r["id"] in tree_regions]

#     # save regions list json:
#     with open(uncompr_atlas_path / descriptors.STRUCTURES_FILENAME, "w") as f:
#         json.dump(pruned_regions_list, f)

#     # Wrap up, compress, and remove file:
#     #####################################
#     wrapup_atlas_from_dir(
#         uncompr_atlas_path,
#         CITATION,
#         ATLAS_LINK,
#         SPECIES,
#         (RES_UM,) * 3,
#         cleanup_files=False,
#         compress=True,
#         root=root_idx,
#     )
