""" Atlas for a domestic cat """

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii
from rich.progress import track
from skimage.filters.rank import modal
from skimage.morphology import ball
from vedo import Mesh, write

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.structure_tree_util import get_structures_tree
from brainglobe_atlasapi.utils import check_internet_connection

###Metadata
__version__ = 1
ATLAS_NAME = "catlas"
CITATION = "Stolzberg, Daniel et al 2017.https://doi.org/10.1002/cne.24271"
SPECIES = "Felis catus"
ATLAS_LINK = "https://github.com/CerebralSystemsLab/CATLAS"
ATLAS_FILE_URL = "https://raw.githubusercontent.com/CerebralSystemsLab/CATLAS/main/SlicerFiles/"
ORIENTATION = "lps"
ROOT_ID = 999  # Placeholder as no hierarchy is present
RESOLUTION = 500  # microns
ATLAS_PACKAGER = "Henry Crosswell"

annotated_volume = None
# temp location
working_dir = Path("F:/Users/Henry/Downloads/Setup/CATLAS-main/temp_pooch")
mesh_folder_path = Path("/CATLAS-main/SlicerFiles/CorticalAtlasModel_A/")
mesh_folder_path = working_dir + mesh_folder_path
mesh_save_folder = working_dir / "meshes"


# A CSV I made from table 1 of the paper, cerebellum added
# ALv and ALd included in catlas but not included on the table

csv_of_full_name = "~/Desktop/catlas_table1_name.csv"


def download_resources(working_dir):
    """
    Download the necessary resources for the atlas.
    If possible, please use the Pooch library to retrieve any resources.
    """
    # Setup download folder
    download_dir_path = working_dir / "download_dir"
    download_dir_path.mkdir(parents=True, exist_ok=True)
    # Setup atlas folder within download_dir
    atlas_dir_path = download_dir_path / "atlas_dir"
    atlas_dir_path.mkdir(exist_ok=True)

    check_internet_connection()
    file_path_list = []
    file_hash_list = [
        ["meanBrain.nii", "md5:84e0d950474bd6c2a4bcebecd0e02ce7"],
        ["CorticalAtlas.nii", "md5:942bbe2483c1d272434b4fd8f8df606f"],
        ["CATLAS_COLORS.txt", "md5:5a48c961ebc1bbc2adb821be173b03e4"],
        ["CorticalAtlas-Split.nii", "md5:7e883fefb60a289c70c4e5553c2c1f6a"],
        ["CATLAS_COLORS-SPLIT.txt", "md5:ff80025b82b51c263ac2d1bfa3b8ae6b"],
    ]

    for file, hash in file_hash_list:
        cached_file = pooch.retrieve(
            url=ATLAS_FILE_URL + file, known_hash=hash, path=atlas_dir_path
        )
        file_path_list.append(cached_file)

    return file_path_list


def retrieve_template_and_annotations(file_path_list):
    """
    Retrieve the desired template and annotations as two numpy arrays.
    Template is MRI image of brain
    Annotations is an annotated 'segmentation' - each label has a unique ID

    Returns:
        tuple: A tuple containing two numpy arrays.
        The first array is the template volume,
        and the second array is the reference volume.
    """
    template = load_nii(file_path_list[0], as_array=True)
    annotations = load_nii(file_path_list[1], as_array=True)
    return template, annotations


def add_heirarchy(labels_df_row):
    """
    Takes the index at a given row and adds the root_id -
    produces structural heirarchy
    """
    structure_index = labels_df_row["id"]
    structure_id = [ROOT_ID, structure_index]
    return structure_id


def add_rgb_col_and_heirarchy(labels_df):
    """
    Re-formats df columns, from individual r,g,b,a into the desired [r,g,b].
    """

    rgb_list = []
    structure_list = []
    for _, row in labels_df.iterrows():
        new_rgb_row = [row["r"], row["g"], row["b"]]
        rgb_list.append(new_rgb_row)

        structure_id = add_heirarchy(row)
        structure_list.append(structure_id)

    labels_df = labels_df.drop(columns=["r", "g", "b", "alpha"])
    labels_df["rgb_triplet"] = rgb_list
    labels_df["structure_id_path"] = structure_list
    return labels_df


def retrieve_structure_information(file_path_list, csv_of_full_name):
    """
    This function should return a pandas DataFrame
    with information about your atlas.

    The DataFrame should be in the following format:

    ╭─────┬──────────────────┬─────────┬───────────────────┬─────────────────╮
    | id  | name             | acronym | structure_id_path | rgb_triplet     |
    |     |                  |         |                   |                 |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997 | root             | root    | []                | [255, 255, 255] |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8   | grps and regions | grey    | [997]             | [191, 218, 227] |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567 | Cerebrum         | CH      | [997, 8]          | [176, 240, 255] |
    ╰─────┴──────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """

    label_df_col = ["id", "acronym", "r", "g", "b", "alpha"]
    full_name_df_col = ["acronym", "name"]
    combined_df_col = [
        "id",
        "name",
        "acronym",
        "structure_id_path",
        "rgb_triplet",
    ]
    root_col = [col for col in combined_df_col if col != "name"]

    labels_df = pd.read_csv(
        file_path_list[2],
        sep=r"\s+",
        skiprows=2,
        header=None,
        names=label_df_col,
        index_col=False,
    )

    full_name_df = pd.read_csv(
        csv_of_full_name, names=full_name_df_col, index_col=False
    )

    labels_df = add_rgb_col_and_heirarchy(labels_df)
    root_id_row = pd.DataFrame(
        [[999, "root", [], [255, 255, 255]]], columns=root_col
    )
    complete_labels_df = pd.concat(
        [root_id_row, labels_df.loc[:]]
    ).reset_index(drop=True)

    # merges both dataframes, aligning the fullnames with the acronyms,
    # fills empty spaces with NaN and sorts df to desired structure.
    structure_info_mix = pd.merge(
        complete_labels_df, full_name_df, on="acronym", how="left"
    )

    structures_df = structure_info_mix[combined_df_col]
    structures_dict = structures_df.to_dict(orient="records")

    structures_tree = get_structures_tree(structures_dict)
    return structures_tree, structures_dict


def extract_mesh_from_vtk(mesh_folder_path):
    mesh_dict = {}
    list_of_mesh_files = os.listdir(mesh_folder_path)
    for vtk_file in list_of_mesh_files:
        if vtk_file[-4:] != ".vtk":
            continue
        elif vtk_file == "A_32_Cerebelum.vtk":  # duplicate
            continue
        elif vtk_file == "A_36_pPE.vtk":  # duplicate & different acronym
            continue
        mesh = Mesh(mesh_folder_path + vtk_file)

        # Re-creating the transformations from mesh_utils
        mesh.triangulate()
        mesh.decimate_pro(0.6)
        mesh.smooth()

        index = vtk_file[2:4]
        if index[-1] == "_":
            index = vtk_file[2]

        file_name = index + ".obj"

        if not Path(mesh_save_folder / file_name).exists():
            write(mesh, str(mesh_save_folder / file_name))
        else:
            print(f"mesh already generated for file {index}")

        mesh_dict[index] = file_name

    return mesh_dict


def construct_meshes(structures_tree, annotations):
    """
    This should return a dict of ids and corresponding paths to mesh files.
    Use packaged mesh files if possible.
    Download or construct mesh files  - use helper function for this
    """
    # Generate binary mask for mesh creation
    labels = np.unique(annotations).astype(np.int_)
    for key, node in structures_tree.nodes.items():
        # Check if the node's key is in the list of labels
        is_label = key in labels
        node.data = Region(is_label)

    # Mesh creation parameters
    closing_n_iters = 5
    decimate_fraction = 0.6
    smooth = True

    meshes_dir_path = working_dir / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    # pass a smoothed version of the annotations for meshing
    smoothed_annotations = annotations.copy()
    smoothed_annotations = modal(
        smoothed_annotations.astype(np.uint8), ball(5)
    )

    # Iterate over each node in the tree and create meshes
    for node in track(
        structures_tree.nodes.values(),
        total=structures_tree.size(),
        description="Creating meshes",
    ):

        create_region_mesh(
            [
                meshes_dir_path,  # Directory where mesh files will be saved
                node,
                structures_tree,
                labels,
                smoothed_annotations,
                ROOT_ID,
                closing_n_iters,
                decimate_fraction,
                smooth,
            ]
        )


# construct_meshes(structures_tree, annotations)


#     )

#     meshes_dir_path: pathlib Path object with folder where meshes are saved
#     tree: treelib.Tree with hierarchical structures information
#     node: tree's node corresponding to the region who's mesh is being created
#     labels: list of unique label annotations in annotated volume,
#     (list(np.unique(annotated_volume)))
#     annotated_volume: 3d numpy array with annotaed volume
#     ROOT_ID: int,
#     id of root structure (mesh creation is a bit more refined for that)

#     meshes_dict = {}
#     return meshes_dict


# commenting out to unit test

### If the code above this line has been filled correctly, nothing needs to be
# edited below (unless variables need to be passed between the functions).
# bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
# bg_root_dir.mkdir(exist_ok=True)
# working_dir = bg_root_dir
# local_file_path_list = download_resources(working_dir)
# template_volume, reference_volume =
# retrieve_template_and_reference(local_file_path_list)
# hemispheres_stack = retrieve_hemisphere_map(local_file_path_list)
# structures = retrieve_structure_information(local_file_path_list)
# meshes_dict = retrieve_or_construct_meshes()

# output_filename = wrapup_atlas_from_data(
#     atlas_name=ATLAS_NAME,
#     atlas_minor_version=__version__,
#     citation=CITATION,
#     atlas_link=ATLAS_LINK,
#     species=SPECIES,
#     resolution=(RESOLUTION,) * 3,
#     orientation=ORIENTATION,
#     root_id=ROOT_ID,
#     reference_stack=template_volume,
#     annotation_stack=annotated_volume,
#     structures_list=structures,
#     meshes_dict=meshes_dict,
#     working_dir=working_dir,
#     hemispheres_stack=None,
#     cleanup_files=False,
#     compress=True,
#     scale_meshes=True,
# )
