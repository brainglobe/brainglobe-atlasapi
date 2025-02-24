__version__ = "0"

import json
import multiprocessing as mp
import os.path
import time

import numpy as np
import pandas as pd
import pooch
import tifffile
from rich.progress import track
from scipy.ndimage import zoom

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

PARALLEL = False

ATLAS_NAME = "princeton_mouse"
SPECIES = "Mus musculus"
ATLAS_LINK = "https://brainmaps.princeton.edu/2020/09/princeton-mouse-brain-atlas-links/"
CITATION = "Pisano et al 2021, https://doi.org/10.1016/j.celrep.2021.109721"
ORIENTATION = "las"
ROOT_ID = 997
ATLAS_RES = 20
PACKAGER = "Sam Clothier. sam.clothier.18@ucl.ac.uk"


def create_atlas(working_dir, resolution):
    # Download the atlas tissue and annotation TIFFs:
    ######################################

    reference_download_url = "https://brainmaps.princeton.edu/pma_tissue"
    annotation_download_url = "https://brainmaps.princeton.edu/pma_annotations"

    # Temporary folder for nrrd files download:
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    utils.check_internet_connection()

    if not os.path.isfile(download_dir_path):
        print("Downloading tissue volume...")
        pooch.retrieve(
            url=reference_download_url,
            known_hash="efe25ee9b5c52faae82fdcf53b96ef10529414ab61e22ef10753a1545ebc8128",
            path=download_dir_path,
            progressbar=True,
        )
    if not os.path.isfile(download_dir_path):
        print("Downloading annotation stack...")
        pooch.retrieve(
            url=annotation_download_url,
            known_hash="36ac8f1d65f2ef76ea35e5f367e90fdeb904af5f2f39ed75568cfafc8ede357e",
            path=download_dir_path,
            progressbar=True,
        )
    print("Download complete.")

    reference_dest_path = (
        download_dir_path / "85c139517e1de923e63d741a8d4dc345-pma_tissue"
    )
    annotation_dest_path = (
        download_dir_path / "179d667f26ee659d1d11de70a9fc004f-pma_annotations"
    )

    template_volume = tifffile.imread(reference_dest_path)
    template_volume = np.array(template_volume)
    annotated_volume = tifffile.imread(annotation_dest_path)
    annotated_volume = np.array(annotated_volume)

    scaling = ATLAS_RES / resolution
    annotated_volume = zoom(
        annotated_volume, (scaling, scaling, scaling), order=0, prefilter=False
    )

    # Download structures tree and define regions:
    ######################################

    structures_download_url = "https://brainmaps.princeton.edu/pma_id_table"
    structures_dest_path = download_dir_path / "structures_download.csv"
    if not os.path.isfile(structures_dest_path):
        utils.retrieve_over_http(structures_download_url, structures_dest_path)

    structures = pd.read_csv(structures_dest_path)
    structures = structures.drop(
        columns=["parent_name", "parent_acronym", "voxels_in_structure"]
    )

    # create structure_id_path column
    def get_inheritance_list_from(id_val):
        inheritance_list = [id_val]

        def add_parent_id(child_id):
            if child_id != 997:  # don't look for the parent of the root area
                parent_id = structures.loc[
                    structures["id"] == child_id, "parent_structure_id"
                ].values[0]
                inheritance_list.insert(0, int(parent_id))
                add_parent_id(parent_id)

        add_parent_id(id_val)
        return inheritance_list

    structures["structure_id_path"] = structures["id"].map(
        lambda x: get_inheritance_list_from(x)
    )

    # create rgb_triplet column
    structures["rgb_triplet"] = "[255, 255, 255]"
    structures["rgb_triplet"] = structures["rgb_triplet"].map(
        lambda x: json.loads(x)
    )

    # order dataframe and convert to list of dictionaries
    # specifying parameters for each area
    structures = structures[
        ["acronym", "id", "name", "structure_id_path", "rgb_triplet"]
    ]
    structs_dict = structures.to_dict(orient="records")
    print(structs_dict)

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structs_dict, f)

    # Create region meshes:
    ######################################

    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structs_dict)

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
    if PARALLEL:
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
                        decimate_fraction,
                        smooth,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            # Error with returning results from pool.map, but we don't care
            pass
    else:
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
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Create meshes dict
    meshes_dict = dict()
    structs_with_mesh = []
    for s in structs_dict:
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

        structs_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structs_with_mesh)} structures with mesh are kept"
    )

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(resolution,) * 3,
        orientation=ORIENTATION,
        root_id=997,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structs_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=PACKAGER,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, ATLAS_RES)
