__version__ = "3"
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr
import SimpleITK as sitk
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

PARALLEL = True  # disable parallel mesh extraction for easier debugging


### Additional functions #####################################################


##############################################################################
def get_id_from_acronym(df, acronym):
    """
    Get Allen's brain atlas ID from brain region acronym(s)

    Call:
        get_id_from_acronym(df, acronym)

    Args:
        df      (pandas dataframe):
            atlas table file [see atlas.load_table()]
        acronym (string or list of strings): brain region acronym(s)

    Returns:
        ID (int or list of ints):
            brain region ID(s) corresponding to input acronym(s)
    """

    # create as list if necessary
    if not isinstance(acronym, list):
        acronym = [acronym]

    if len(acronym) > 1:
        ID_list = []
        for acro in acronym:
            ID = df["id"][df["acronym"] == acro].item()
            ID_list.append(ID)
        return ID_list
    else:
        return df["id"][df["acronym"] == acronym[0]].item()

    # return df['id'][df['acronym']  == acronym].item() # OLD VERSION


def get_acronym_from_id(df, ID):
    """
    Get Allen's brain atlas acronym from brain region ID(s)

    Call:
        get_acronym_from_ID(df, acronym)

    Args:
        df (pandas dataframe): atlas table dataframe [see atlas.load_table()]
        ID (int or list of int): brain region ID(s)

    Returns:
        acronym (string or list of strings):
        brain region acronym(s) corresponding to input ID(s)
    """

    # create as list if necessary
    if not isinstance(ID, list):
        ID = [ID]

    if len(ID) > 1:
        acronym_list = []
        for id in ID:
            acronym = df["acronym"][df["id"] == id].item()
            acronym_list.append(acronym)
        return acronym_list
    else:
        return df["acronym"][df["id"] == ID[0]].item()


def tree_traverse_child2parent(df, child_id, ids):
    parent = df["parent_id"][df["id"] == child_id].item()

    if not np.isnan(parent):
        id = df["id"][df["id"] == parent].item()
        ids.append(id)
        tree_traverse_child2parent(df, parent, ids)
        return ids
    else:
        return ids


def get_all_parents(df, key):
    """
    Get all parent IDs/acronyms in Allen's brain atlas hierarchical structure'

    Call:
        get_all_children(df, key)

    Args:
        df (pandas dataframe) : atlas table dataframe [see atlas.load_table()]
        key (int/string)      : atlas region ID/acronym

    Returns:
        parents (list) : brain region acronym corresponding to input ID
    """

    if isinstance(key, str):  # if input is acronym convert to ID
        list_parent_ids = tree_traverse_child2parent(
            df, get_id_from_acronym(df, key), []
        )
    elif isinstance(key, int):
        list_parent_ids = tree_traverse_child2parent(df, key, [])

    if isinstance(key, str):  # if input is acronym convert IDs to acronyms
        parents = []
        for id in list_parent_ids:
            parents.append(get_acronym_from_id(df, id))
    elif isinstance(key, int):
        parents = list_parent_ids.copy()

    return parents


##############################################################################

##############################################################################
# %%


def create_atlas(working_dir, resolution):
    ATLAS_NAME = "perens_stereotaxic_mri_mouse"
    SPECIES = "Mus musculus"
    ATLAS_LINK = "https://www.neuropedia.dk/resource/multimodal-3d-mouse-brain-atlas-framework-with-the-skull-derived-coordinate-system/"
    CITATION = "Perens et al. 2023, https://doi.org/10.1007/s12021-023-09623-9"
    ORIENTATION = "ial"
    ROOT_ID = 997
    ATLAS_FILE_URL = "https://www.neuropedia.dk/wp-content/uploads/Multimodal_mouse_brain_atlas_files.7z"

    # Temporary folder for  download:
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_files_dir = download_dir_path / "atlas_files"

    ## Download atlas_file
    utils.check_internet_connection()

    import urllib.request

    destination_path = download_dir_path / "atlas_download.7z"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:129.0) Gecko/20100101 Firefox/129.0",
        "Accept": (
            "text/html,"
            "application/xhtml+xml,"
            "application/xml;q=0.9,"
            "image/avif,"
            "image/webp,"
            "image/png,"
            "image/svg+xml,"
            "*/*;q=0.8"
        ),
        "Accept-Language": "en-GB,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "DNT": "1",
        "Sec-GPC": "1",
        "Host": "www.neuropedia.dk",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
        "TE": "trailers",
        "Priority": "u=0, i",
    }
    if not os.path.isdir(
        atlas_files_dir / "Multimodal_mouse_brain_atlas_files"
    ):
        req = urllib.request.Request(ATLAS_FILE_URL, headers=headers)
        with (
            urllib.request.urlopen(req) as response,
            open(destination_path, "wb") as out_file,
        ):
            data = response.read()  # a `bytes` object
            out_file.write(data)
        with py7zr.SevenZipFile(destination_path, mode="r") as z:
            z.extractall(path=atlas_files_dir)
        destination_path.unlink()

    structures_file = (
        atlas_files_dir
        / "Multimodal_mouse_brain_atlas_files"
        / "Hierarchy_tree"
        / "Annotation_info.csv"
    )
    annotations_file = (
        atlas_files_dir
        / "Multimodal_mouse_brain_atlas_files"
        / "MRI_space_oriented"
        / "mri_ano.nii.gz"
    )
    reference_file = (
        atlas_files_dir
        / "Multimodal_mouse_brain_atlas_files"
        / "MRI_space_oriented"
        / "mri_temp.nii.gz"
    )

    annotated_volume = sitk.GetArrayFromImage(
        sitk.ReadImage(str(annotations_file))
    )
    template_volume = sitk.GetArrayFromImage(
        sitk.ReadImage(str(reference_file))
    )

    print("Download completed...")

    # ------------------------ #
    #   STRUCTURES HIERARCHY   #
    # ------------------------ #

    # Parse region names & hierarchy
    # ##############################
    df = pd.read_csv(structures_file)

    # Make region hierarchy and gather colors to one list
    parents = []
    rgb = []
    for index, row in df.iterrows():
        temp_id = row["id"]
        temp_parents = get_all_parents(df, temp_id)
        parents.append(temp_parents[::-1])

        temp_rgb = [row["red"], row["green"], row["blue"]]
        rgb.append(temp_rgb)

    df = df.drop(columns=["parent_id", "red", "green", "blue"])
    df = df.assign(structure_id_path=parents)
    df = df.assign(rgb_triplet=rgb)
    df.loc[0, "structure_id_path"] = [997]

    structures = df.to_dict("records")

    for structure in structures:
        # root doesn't have a parent
        if structure["id"] != 997:
            structure["structure_id_path"].append(structure["id"])

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # Create meshes:
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures)

    labels = np.unique(annotated_volume).astype(np.int32)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2  # not used for this atlas
    decimate_fraction = 0.2  # not used for this atlas

    smooth = False
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
            # error with returning results from pool.map but we don't care
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
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    # ----------- #
    #   WRAP UP   #
    # ----------- #

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
        root_id=ROOT_ID,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    resolution = 25  # some resolution, in microns

    # Generated atlas path:
    bg_root_dir = Path.home() / "brainglobe_workingdir" / "perens_mri_mouse"
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir, resolution)
