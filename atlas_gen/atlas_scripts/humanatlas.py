from pathlib import Path
import json
import pandas as pd
from brainio import brainio
import numpy as np
import urllib3
import multiprocessing as mp
import time
import treelib
from vtkplotter import load

from brainatlas_api.atlas_gen import (
    save_anatomy,
    save_annotation,
    wrapup_atlas_from_dir,
    volume_utils,
    get_structure_children,
)
from atlas_gen import mesh_utils
from brainatlas_api import descriptors
from brainatlas_api.structures.structure_tree import StructureTree


def create_structure_mesh(args):
    structures, annotation_whole, meshes_dir, a = args

    lbl = structures.loc[a, "acronym"]
    volume = volume_utils.create_masked_array(
        annotation_whole, a, greater_than=False
    )

    if np.max(volume) < 1:
        print(f"No voxel data for region {lbl}")
        return None
    else:
        print(f"Creating mesh for {a}.obj")
        savepath = meshes_dir / f"{a}.obj"
        if not savepath.exists():
            mesh = mesh_utils.extract_mesh_from_mask(
                volume,
                obj_filepath=savepath,
                closing_n_iters=1,
                decimate=True,
                smooth=False,
            )
            return mesh
        else:
            return None


def create_nonleaf_structure_mesh(args):
    (
        nonleaf,
        meshes_dir,
        regions_list,
        id_to_acronym_map,
        acronym_to_voxel,
        annotation_whole,
    ) = args
    savepath = meshes_dir / f'{nonleaf["id"]}.obj'
    if not savepath.exists():
        print(f'Creating mesh for {nonleaf["id"]}.obj')

        # Get id of substructures leafs
        substructures = get_structure_children(
            regions_list, nonleaf, use_tree=True
        )

        if not substructures:
            print(f'No substructures for : {nonleaf["id"]}')
            return None

        # get the voxel ids of substructures
        substructuresids = [id_to_acronym_map[idx] for idx in substructures]
        voxel_labels = [
            acronym_to_voxel[acro]
            for acro in substructuresids
            if acro in list(acronym_to_voxel.keys())
        ]

        volume = volume_utils.create_masked_array(
            annotation_whole, voxel_labels
        )

        if volume.max() < 1:
            print(f"No voxel data for region {nonleaf['id']}")
            return None

        mesh_utils.extract_mesh_from_mask(
            volume,
            obj_filepath=savepath,
            closing_n_iters=8,
            decimate=True,
            smooth=False,
        )
    else:
        return None


class Region(object):
    def __init__(self, has_mesh):
        self.has_mesh = has_mesh


def prune_tree(tree):
    nodes = tree.nodes.copy()
    for key, node in nodes.items():
        if node.tag == "root":
            continue
        if node.data.has_mesh:
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
                    [c.data.has_mesh for _, c in subtree.nodes.items()]
                ):
                    tree.remove_node(node.identifier)
    return tree


if __name__ == "__main__":
    # Specify information about the atlas:
    RES_UM = 500
    VERSION = "0.1"
    ATLAS_NAME = f"allen_human_{RES_UM}um_v{VERSION}"
    SPECIES = "human (Homo sapiens)"
    ATLAS_LINK = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/"
    CITATION = "Ding et al 2020, https://doi.org/10.1002/cne.24080"

    data_fld = Path(
        r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy\Atlases\atlasesforbrainrender\AllenHuman"
    )

    # Generated atlas path:
    bg_root_dir = Path.home() / ".brainglobe"
    bg_root_dir.mkdir(exist_ok=True)

    # Temporary folder for nrrd files download:
    temp_path = bg_root_dir / "temp"
    temp_path.mkdir(exist_ok=True)

    # Temporary folder for files before compressing:
    uncompr_atlas_path = temp_path / ATLAS_NAME
    uncompr_atlas_path.mkdir(exist_ok=True)

    # Open reference:
    #################
    # TODO check if re-orienting is necessary

    annotation = brainio.load_any(
        data_fld / "annotation.nii"
    )  # shape (394, 466, 378)

    anatomy = brainio.load_any(
        data_fld
        / "mni_icbm152_nlin_sym_09b"
        / "mni_icbm152_pd_tal_nlin_sym_09b_hires.nii"
    )  # shape (394, 466, 378)

    # Remove weird artefact
    annotation = annotation[:197, :, :]
    anatomy = anatomy[:197, :, :]

    # These data only have one hemisphere, so mirror them
    annotation_whole = np.zeros(
        (annotation.shape[0] * 2, annotation.shape[1], annotation.shape[2]),
        annotation.dtype,
    )
    annotation_whole[: annotation.shape[0], :, :] = annotation
    annotation_whole[annotation.shape[0] :, :, :] = annotation[::-1, :, :]

    anatomy_whole = np.zeros(
        (anatomy.shape[0] * 2, anatomy.shape[1], anatomy.shape[2]),
        anatomy.dtype,
    )
    anatomy_whole[: anatomy.shape[0], :, :] = anatomy
    anatomy_whole[anatomy.shape[0] :, :, :] = anatomy[::-1, :, :]

    # Save as .tiff
    save_annotation(annotation_whole, uncompr_atlas_path)
    save_anatomy(anatomy_whole, uncompr_atlas_path)
    del anatomy_whole, annotation, anatomy

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

    # save regions list json:
    with open(uncompr_atlas_path / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(regions_list, f)

    # Create meshes
    ###############
    meshes_dir = uncompr_atlas_path / descriptors.MESHES_DIRNAME
    meshes_dir.mkdir(exist_ok=True)

    unique_values, unique_counts = np.unique(
        annotation_whole, return_counts=True
    )
    voxel_counts = dict(zip(unique_values, unique_counts))
    if 0 in voxel_counts:
        del voxel_counts[0]
    structures.set_index("id", inplace=True)

    # Create root first
    root = [s for s in regions_list if s["acronym"] == "root"][0]
    root_idx = root["id"]
    root_volume = volume_utils.create_masked_array(
        annotation_whole, 0, greater_than=True
    )
    savepath = meshes_dir / f'{root["id"]}.obj'
    if not savepath.exists():
        root_mesh = mesh_utils.extract_mesh_from_mask(
            root_volume, savepath, smooth=False, decimate=True
        )
    else:
        root_mesh = load(str(savepath))

    # Asses mesh extraction quality
    # mesh_utils.compare_mesh_and_volume(root_mesh, root_volume)

    # ? Create meshes for leaf nodes
    start = time.time()
    pool = mp.Pool(mp.cpu_count() - 2)
    try:
        pool.map(
            create_structure_mesh,
            [
                (structures, annotation_whole, meshes_dir, a)
                for a in voxel_counts
            ],
        )
    except mp.pool.MaybeEncodingError:
        pass  # error with returning results from pool.map but we don't care
    print(
        f"Creating meshes for {len(voxel_counts)} structures took: {round(time.time() - start, 3)}s"
    )

    # Show which regions were represented in the annotated volume
    regions_with_mesh = [structures.loc[a, "acronym"] for a in voxel_counts]

    tree = StructureTree(regions_list).get_structures_tree()

    for key, node in tree.nodes.items():
        if node.tag in regions_with_mesh:
            has_mesh = True
        else:
            has_mesh = False
        node.data = Region(has_mesh)

    # Remove regions that are children to the ones that which
    # were represented in the volume or were
    # at least some of their children had a mesh
    tree = prune_tree(tree)

    # ? extract meshes for non leaf regions
    id_to_acronym_map = {s["id"]: s["acronym"] for s in regions_list}
    voxel_to_acro = {a: structures.loc[a, "acronym"] for a in voxel_counts}
    acronym_to_voxel = {v: k for k, v in voxel_to_acro.items()}
    non_leaf_nodes = [
        s
        for s in regions_list
        if s["acronym"] != "root" and s["id"] not in voxel_counts
    ]

    start = time.time()
    pool = mp.Pool(mp.cpu_count() - 2)
    try:
        pool.map(
            create_nonleaf_structure_mesh,
            [
                (
                    nonleaf,
                    meshes_dir,
                    regions_list,
                    id_to_acronym_map,
                    acronym_to_voxel,
                    annotation_whole,
                )
                for nonleaf in non_leaf_nodes
            ],
        )
    except mp.pool.MaybeEncodingError:
        pass  # error with returning results from pool.map but we don't care
    print(
        f"Creating meshes for {len(non_leaf_nodes)} structures took: {round(time.time() - start, 3)}s"
    )

    # ? Fill in more of the regions that don't have mesh yet
    for repeat in range(4):
        for idx, node in tree.nodes.items():
            savepath = meshes_dir / f"{idx}.obj"
            if not savepath.exists():
                region = [r for r in regions_list if r["id"] == idx][0]
                args = (
                    region,
                    meshes_dir,
                    regions_list,
                    id_to_acronym_map,
                    acronym_to_voxel,
                    annotation_whole,
                )
                create_nonleaf_structure_mesh(args)

    # Update tree and check that everyone got a mesh
    for idx, node in tree.nodes.items():
        savepath = meshes_dir / f"{idx}.obj"
        if savepath.exists():
            node.data.has_mesh = True

    tree.show(data_property="has_mesh")

    print(
        f"\n\nTotal number of structures left in tree: {tree.size()} - max depth: {tree.depth()}"
    )

    tree_regions = [node.identifier for k, node in tree.nodes.items()]
    pruned_regions_list = [r for r in regions_list if r["id"] in tree_regions]

    # save regions list json:
    with open(uncompr_atlas_path / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(pruned_regions_list, f)

    # Wrap up, compress, and remove file:
    #####################################
    wrapup_atlas_from_dir(
        uncompr_atlas_path,
        CITATION,
        ATLAS_LINK,
        SPECIES,
        (RES_UM,) * 3,
        cleanup_files=False,
        compress=True,
        root=root_idx,
    )
