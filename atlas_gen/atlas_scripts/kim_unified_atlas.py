import json
from rich.progress import track
import pandas as pd
import numpy as np
import time
import multiprocessing as mp
from pathlib import Path
from brainio.brainio import load_any
from allensdk.core.reference_space_cache import ReferenceSpaceCache

# import sys

# sys.path.append("./")
from atlas_gen.mesh_utils import create_region_mesh, Region
from atlas_gen.wrapup import wrapup_atlas_from_data
from bg_atlasapi.structure_tree_util import get_structures_tree


if __name__ == "__main__":
    PARALLEL = True  # disable parallel mesh extraction for easier debugging

    # ---------------------------------------------------------------------------- #
    #                                 PREP METADATA                                #
    # ---------------------------------------------------------------------------- #
    RES_UM = 25
    VERSION = 1
    ATLAS_NAME = "kim_unified"
    SPECIES = "Mus musculus"
    ATLAS_LINK = "https://kimlab.io/brain-map/atlas/"
    CITATION = "Chon et al. 2019, https://doi.org/10.1038/s41467-019-13057-w"
    ORIENTATION = "als"
    ROOT_ID = 997

    # ---------------------------------------------------------------------------- #
    #                                PREP FILEPATHS                                #
    # ---------------------------------------------------------------------------- #

    paxinos_allen_directory = Path(
        r"C:\Users\Federico\Downloads\kim_atlas_materials.tar\kim_atlas_materials"
    )
    annotations_image = paxinos_allen_directory / "annotations_coronal.tif"
    structures_file = paxinos_allen_directory / "structures.csv"

    # assume isotropic
    ANNOTATIONS_RES_UM = 10

    version = "0.1"

    # Generated atlas path:
    bg_root_dir = Path.home() / ".brainglobe"
    bg_root_dir.mkdir(exist_ok=True)

    # Temporary folder for nrrd files download:
    temp_path = Path(r"C:\Users\Federico\.brainglobe\kimdev")
    temp_path.mkdir(exist_ok=True)
    downloading_path = temp_path / "downloading_path"
    downloading_path.mkdir(exist_ok=True)

    # Temporary folder for files before compressing:
    uncompr_atlas_path = temp_path / ATLAS_NAME
    uncompr_atlas_path.mkdir(exist_ok=True)

    # ---------------------------------------------------------------------------- #
    #                                 GET TEMPLATE                                 #
    # ---------------------------------------------------------------------------- #

    # Load (and possibly downsample) annotated volume:
    #########################################
    scaling_factor = ANNOTATIONS_RES_UM / RES_UM
    print(
        f"Loading: {annotations_image.name} and downscaling by: {scaling_factor}"
    )
    annotated_volume = load_any(
        annotations_image,
        x_scaling_factor=scaling_factor,
        y_scaling_factor=scaling_factor,
        z_scaling_factor=scaling_factor,
        anti_aliasing=False,
    )

    # Download template volume:
    #########################################
    spacecache = ReferenceSpaceCache(
        manifest=downloading_path / "manifest.json",
        # downloaded files are stored relative to here
        resolution=RES_UM,
        reference_space_key="annotation/ccf_2017"
        # use the latest version of the CCF
    )

    # Download
    print("Downloading template file")
    template_volume, _ = spacecache.get_template_volume()
    print("Download completed...")

    # ---------------------------------------------------------------------------- #
    #                             STRUCTURES HIERARCHY                             #
    # ---------------------------------------------------------------------------- #

    # Parse region names & hierarchy
    # ######################################
    df = pd.read_csv(structures_file)
    df = df.drop(columns=["Unnamed: 0", "parent_id", "parent_acronym"])

    # split by "/" and convert list of strings to list of ints
    df["structure_id_path"] = (
        df["structure_id_path"]
        .str.split(pat="/")
        .map(lambda x: [int(i) for i in x])
    )

    structures = df.to_dict("records")

    for structure in structures:
        structure.update({"rgb_triplet": [255, 255, 255]})
        # root doesn't have a parent
        if structure["id"] != 997:
            structure["structure_id_path"].append(structure["id"])

    # save regions list json:
    with open(uncompr_atlas_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # ---------------------------------------------------------------------------- #
    #                                CREATE MESHESH                                #
    # ---------------------------------------------------------------------------- #
    print(f"Saving atlas data at {uncompr_atlas_path}")
    meshes_dir_path = uncompr_atlas_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    # Create and prune structures tree
    tree = get_structures_tree(structures)
    drop_from_tree = [
        "fiber_tracts",
        "VentSys",
        "bas",
    ]  # stuff we don't need meshes for
    for drop in drop_from_tree:
        print("Dropping from structures tree: ", drop)
        dropped = tree.remove_subtree(
            [nid for nid, n in tree.nodes.items() if n.tag == drop][0]
        )

    print(
        f"Number of brain regions: {tree.size()}, max tree depth: {tree.depth()}"
    )

    # Create a tree marking which brain regions are shown in the annotation
    labels = np.unique(annotated_volume).astype(np.int32)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # tree.show(data_property='has_label')

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
                        annotated_volume,
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
            create_region_mesh(
                (
                    meshes_dir_path,
                    node,
                    tree,
                    labels,
                    annotated_volume,
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
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
