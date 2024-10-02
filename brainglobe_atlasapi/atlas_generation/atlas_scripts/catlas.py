""" Atlas for a domestic cat """

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import pooch
import requests
from brainglobe_utils.IO.image import load_nii
from bs4 import BeautifulSoup
from vedo import Mesh, write

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import check_internet_connection

### Metadata
__version__ = 1
ATLAS_NAME = "CSL_catlas"
CITATION = "Stolzberg, Daniel et al 2017.https://doi.org/10.1002/cne.24271"
SPECIES = "Felis catus"
ATLAS_LINK = "https://github.com/CerebralSystemsLab/CATLAS"
ATLAS_FILE_URL = "https://raw.githubusercontent.com/CerebralSystemsLab/CATLAS/main/SlicerFiles/"
ORIENTATION = "lps"
ROOT_ID = 997
RESOLUTION = 500  # microns
ATLAS_PACKAGER = "Henry Crosswell"


def download_resources(working_dir):
    """
    Downloads the nifti images, labels and annotations for the atlas.
    Uses pooch hashes if they are present, if not obtains them

    Returns:
        List : list of all downloaded local filepaths
    """
    # Setup download folder
    download_dir_path = working_dir / "download_dir"
    download_dir_path.mkdir(parents=True, exist_ok=True)

    # Setup atlas folder within download_dir
    atlas_dir_path = download_dir_path / "atlas_dir"
    file_hash_jsonpath = atlas_dir_path / "hash_registry.json"

    file_path_and_hash_list = []
    file_path_list = []
    check_internet_connection()

    if file_hash_jsonpath.exists():
        print("Retrieving file paths and hash lists...")
        hash_registry_json = atlas_dir_path / "hash_registry.json"
        with open(hash_registry_json, "r") as json_file_to_edit:
            file_path_and_hash_list = json.load(json_file_to_edit)

    else:
        # If dir is not already created - it downloads the data
        print("Downloading file paths and hash lists...")
        atlas_dir_path.mkdir(exist_ok=True)
        file_names = [
            "meanBrain.nii",
            "CorticalAtlas.nii",
            "CATLAS_COLORS.txt",
        ]

        file_hash_jsonpath = download_filename_and_hash(
            file_names, atlas_dir_path
        )

        hash_registry_json = download_multiple_filenames_and_hashs(
            file_hash_jsonpath, atlas_dir_path
        )

        with open(hash_registry_json, "r") as json_file_to_edit:
            file_path_and_hash_list = json.load(json_file_to_edit)

    for file_path, _ in file_path_and_hash_list:
        file_path_list.append(file_path)
    return file_path_list


def download_filename_and_hash(file_names, atlas_dir_path):
    """
    Takes the given file names and uses pooch to download
    the files converting into a json

    Returns:
        Json.file : a nested list containing [filepath,hash]
    """
    file_and_hash_list = []
    file_hash_jsonpath = atlas_dir_path / "hash_registry.json"

    for file in file_names:
        cached_file = pooch.retrieve(
            url=ATLAS_FILE_URL + file, known_hash=None, path=atlas_dir_path
        )
        file_hash = pooch.file_hash(cached_file, alg="md5")
        file_and_hash_list.append([cached_file, file_hash])

    with open(file_hash_jsonpath, "w") as file_and_hash_json:
        json.dump(file_and_hash_list, file_and_hash_json)

    return file_hash_jsonpath


def download_multiple_filenames_and_hashs(file_hash_jsonpath, atlas_dir_path):
    """
    Opens the json file to write into, requests file names and uses these
    to download the vtk annotation files, iterating through the folder

    Returns:
        Json.file : a updated nested list containing [filepath,hash]
    """

    vtk_folder_url = (
        "https://github.com/CerebralSystemsLab/CATLAS/"
        + "blob/main/SlicerFiles/CorticalAtlasModel_A/"
    )

    vtk_file_url = (
        "https://github.com/CerebralSystemsLab/CATLAS/raw/"
        + "refs/heads/main/SlicerFiles/CorticalAtlasModel_A/"
    )

    with open(file_hash_jsonpath, "r") as json_file_to_edit:
        file_and_hash_list = json.load(json_file_to_edit)

    response = requests.get(vtk_folder_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, "html.parser")
        files = soup.find_all("a", class_="Link--primary")

        seen_files = set()
        for file in files:
            file_name = file.get_text()

            if file_name not in seen_files:
                seen_files.add(file_name)

                cached_file = pooch.retrieve(
                    url=vtk_file_url + file_name,
                    known_hash=None,
                    path=atlas_dir_path,
                )
                file_hash = pooch.file_hash(cached_file, alg="md5")
                file_and_hash_list.append([cached_file, file_hash])
            else:
                continue

    else:
        print("Incorrect URL given for VTK files")

    with open(file_hash_jsonpath, "w") as file_and_hash_json:
        json.dump(file_and_hash_list, file_and_hash_json)

    return file_hash_jsonpath


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
    This function should return a list of dictionaries
    with information about your atlas.

    The list of dictionaries should be in the following format:

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
        list of dicts: A list containing a dict of the atlas information.
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
        [[997, "root", [997], [255, 255, 255]]], columns=root_col
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

    return structures_dict


def extract_mesh_from_vtk(working_dir):
    """
    Given the path to a folder containing annotation.vtk files
    extracts the data as a vedo.mesh and saves as .obj
    to be readable by other functions

    Returns:
        dict: Key is obj id - value is obj file path
    """
    mesh_dict = {}

    atlas_dir_path = working_dir / "download_dir" / "atlas_dir"
    mesh_save_folder = atlas_dir_path / "meshes"
    mesh_save_folder.mkdir(parents=True, exist_ok=True)

    list_of_mesh_files = os.listdir(atlas_dir_path)

    for vtk_file in list_of_mesh_files:
        if not vtk_file.endswith(".vtk"):
            continue

        # Checking for duplicates
        elif vtk_file in ["A_32_Cerebelum.vtk", "A_36_pPE.vtk"]:
            continue

        mesh = Mesh(str(atlas_dir_path / vtk_file))

        # Re-creating the transformations from mesh_utils
        mesh.triangulate()
        mesh.decimate_pro(0.6)
        mesh.smooth()

        # Saves object files with a numerical index
        index = str(vtk_file)
        index = index.split("-")
        index = index[1][2:4]
        if index[-1] == "_":
            index = index[0]
        file_name = f"{index}.obj"
        file_path = str(mesh_save_folder / file_name)
        mesh_dict[index] = file_path

        if not Path(file_path).exists():
            write(mesh, file_path)
        else:
            continue
    return mesh_dict


# A CSV Table1 of paper, cerebellum added
# ALv and ALd included in catlas but not included on the table
csv_of_full_name = "~/Desktop/catlas_table1_name.csv"

bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME

# Deletes the folder since wrapup doesnt work if folder is created
if bg_root_dir.exists():
    shutil.rmtree(bg_root_dir)

bg_root_dir.mkdir(parents=True, exist_ok=True)
working_dir = bg_root_dir
local_file_path_list = download_resources(working_dir)
template_volume, reference_volume = retrieve_template_and_annotations(
    local_file_path_list
)
structures_dict = retrieve_structure_information(
    local_file_path_list, csv_of_full_name
)
meshes_dict = extract_mesh_from_vtk(working_dir)

output_filename = wrapup_atlas_from_data(
    atlas_name=ATLAS_NAME,
    atlas_minor_version=__version__,
    citation=CITATION,
    atlas_link=ATLAS_LINK,
    species=SPECIES,
    resolution=(RESOLUTION,) * 3,
    orientation=ORIENTATION,
    root_id=ROOT_ID,
    reference_stack=template_volume,
    annotation_stack=reference_volume,
    structures_list=structures_dict,
    meshes_dict=meshes_dict,
    working_dir=working_dir,
    hemispheres_stack=None,
    cleanup_files=False,
    compress=True,
    scale_meshes=True,
)
