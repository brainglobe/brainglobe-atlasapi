"""Atlas for a domestic cat"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii
from scipy.ndimage import median_filter
from vedo import Mesh, write

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    extract_mesh_from_mask,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import check_internet_connection

### Metadata
__version__ = 0
ATLAS_NAME = "csl_cat"
CITATION = "Stolzberg, Daniel et al 2017.https://doi.org/10.1002/cne.24271"
SPECIES = "Felis catus"
ATLAS_LINK = "https://github.com/CerebralSystemsLab/CATLAS"
ATLAS_FILE_URL = "https://raw.githubusercontent.com/CerebralSystemsLab/CATLAS/main/SlicerFiles/"
ORIENTATION = "lps"
ROOT_ID = 997
RESOLUTION = 500  # microns
ATLAS_PACKAGER = "Henry Crosswell"


def pooch_init(temp_download_dir, base_url):
    """
    initiate a pooch object to be used to fetch files in the future,
    using the hashes saved to the hashes folder.

    """
    hash_folder = (
        Path(__file__).parent.parent / "hashes" / (ATLAS_NAME + ".txt")
    )
    dawg = pooch.create(
        path=temp_download_dir,
        base_url=base_url,
    )

    dawg.load_registry(hash_folder)

    return dawg, hash_folder


def download_resources(working_dir):
    """
    Uses Pooch to download the nifti images, labels and annotations
    for the atlas, using the pooch hashes from the hash folder.

    Returns:
        List : list of all downloaded local filepaths
    """

    check_internet_connection()

    # Setup download folder
    temp_download_dir = working_dir / "download_dir"
    temp_download_dir.mkdir(parents=True, exist_ok=True)

    file_path_list = []

    file_paths = download_nifti_files(temp_download_dir)

    file_paths = download_mesh_files(file_paths, temp_download_dir)

    for _, file_path in file_paths:
        file_path_list.append(file_path)

    return file_path_list


def download_nifti_files(temp_download_dir):
    """
    Takes pre-assigned file names and uses pooch to download
    the files, writing the file_path to a list.

    Returns:
        file_paths :
            a list of downloaded file paths containing [filename, filepath]
    """
    file_paths = []

    file_names = [
        "meanBrain.nii",
        "CorticalAtlas.nii",
        "CATLAS_COLORS.txt",
    ]

    dawg, _ = pooch_init(temp_download_dir, ATLAS_FILE_URL)

    # Loops through the files
    for file in file_names:
        cached_file = dawg.fetch(file)
        file_paths.append([file, cached_file])

    return file_paths


def download_mesh_files(file_paths, temp_download_dir):
    """
    Retrieves the file names from the hash file,
    then downloads the mesh files using the associated hash.
    Appends this to the nifti file path list

    Returns:
        nifti_file_paths :
            List containing nifti and mesh file paths [filename, filepath]
    """

    vtk_file_url = (
        "https://github.com/CerebralSystemsLab/CATLAS/raw/"
        + "refs/heads/main/SlicerFiles/CorticalAtlasModel_A/"
    )

    dawg, hash_folder = pooch_init(temp_download_dir, vtk_file_url)

    with open(hash_folder, "r") as file:
        file_names = [line.strip().split()[0] for line in file]

    # first 3 files are already read: template, annotation and colors.txt
    # so we just need to fetch the remaining files
    for file in file_names[3:]:
        cached_file = dawg.fetch(file)
        file_paths.append([file, cached_file])

    return file_paths


def retrieve_template_and_annotations(file_path_list):
    """
    Retrieve the desired template and annotations as two numpy arrays.
    Template_volume is an MRI image of brain
    annotation_volume is a segmentation - each label has a unique ID

    Returns:
        tuple: A tuple containing two numpy arrays.
        The first array is a template volume rescale to uint16 range,
        and the second array is a reference volume.
    """

    template_volume = load_nii(file_path_list[0], as_array=True)
    annotation_volume = load_nii(file_path_list[1], as_array=True)

    dmin = np.min(template_volume)
    dmax = np.max(template_volume)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    template_volume = (template_volume - dmin) * dscale
    template_volume = template_volume.astype(np.uint16)

    return template_volume, annotation_volume


def add_hierarchy(labels_df_row):
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

        structure_id = add_hierarchy(row)
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

    full_name_df = pd.DataFrame(csv_of_full_name, columns=full_name_df_col)

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


def create_mask_for_root(template_volume, mesh_save_folder):
    """
    Create a smooth mask from the template for the whole brain,
    meshes and saves it .obj file within mesh_save_folder

    Returns:
        None
    """
    root_mask = (template_volume >= 8000).astype(np.uint8)
    root_mask = median_filter(root_mask, size=7)
    file_name = f"{ROOT_ID}.obj"
    file_path = str(mesh_save_folder / file_name)

    extract_mesh_from_mask(
        root_mask,
        obj_filepath=file_path,
        smooth=True,
        decimate_fraction=0.1,
    )
    return None


def extract_mesh_from_vtk(working_dir):
    """
    Given the path to a folder containing annotation.vtk files
    extracts the data as a vedo.mesh and saves as .obj
    to be readable by other functions

    Returns:
        dict: Key is obj id - value is obj file path
    """
    mesh_dict = {}

    temp_download_dir = working_dir / "download_dir"
    mesh_save_folder = temp_download_dir / "meshes"
    mesh_save_folder.mkdir(parents=True, exist_ok=True)

    list_of_mesh_files = os.listdir(temp_download_dir)
    for vtk_file in list_of_mesh_files:
        if not vtk_file.endswith(".vtk"):
            continue

        mesh = Mesh(str(temp_download_dir / vtk_file))

        # Re-creating the transformations from mesh_utils
        mesh.triangulate()
        mesh.decimate_pro(0.6)
        mesh.smooth()

        # Saves object files with a numerical index
        index = str(vtk_file)
        index = index.split("_")[1]
        file_name = f"{index}.obj"
        file_path = str(mesh_save_folder / file_name)
        mesh_dict[index] = file_path

        # convert mesh from mm to pixels
        # calculated by np.linalg.inv from nibabel.image.affine
        # it's an oblique view close to LPS
        nii_affine_inverse = np.array(
            [
                [1.99690761, 0.1024282, -0.0432288],
                [-0.07216534, 1.78576813, 0.89767717],
                [-0.08457204, 0.89472933, -1.78670276],
            ]
        )

        transformed_coordinates = np.zeros_like(mesh.coordinates)
        for i, c in enumerate(mesh.coordinates):
            transformed_coordinates[i] = nii_affine_inverse.dot(c)

        mesh.coordinates = transformed_coordinates

        # shift to centre of image
        # necessity for this operation found via trial and error
        mesh.coordinates += np.array([75, 96, 48])

        if not Path(file_path).exists():
            write(mesh, file_path)
        else:
            continue

    create_mask_for_root(template_volume, mesh_save_folder)
    mesh_dict[ROOT_ID] = str(mesh_save_folder / f"{ROOT_ID}.obj")
    return mesh_dict


# A list form of Table 1, manually copied from the paper.
# Cerebellum added to the list, as not included in table 1
# ALv and ALd are included in catlas text file but not included on the table,
# replaced with NaN for full name
acronym_to_region_map = [
    ["root", "root"],
    ["A1", "Primary auditory cortex"],
    ["A2", "Second auditory cortex"],
    ["AAF", "Anterior auditory field"],
    ["DP", "Posterior ectosylvian auditory cortex, dorsal division"],
    ["DZ", "Dorsal zone of auditory cortex"],
    ["EPp", "Posterior ectosylvian gyrus, posterior division"],
    ["FAES", "FAES, Field of the anterior ectosylvian sulcus"],
    ["IN", "Auditory insular cortex"],
    ["iPE", "Posterior ectosylvian auditory cortex, intermediate division"],
    ["PAF", "Posterior auditory field"],
    ["TE", "Temporal cortex"],
    ["VAF", "Ventral auditory field"],
    ["vPAF", "Posterior auditory field, ventral division"],
    ["vPE", "Posterior ectosylvian auditory cortex, ventral division"],
    ["1", "Area 1, primary somatosensory cortex"],
    ["2", "Area 2, primary somatosensory cortex"],
    ["3a", "Area 3a primary somatosensory cortex"],
    ["3b", "Area 3b primary somatosensory cortex"],
    ["5al", "Area 5a, lateral division"],
    ["5am", "Area 5a, medial division"],
    ["5bl", "Area 5b, lateral division"],
    ["5bm", "Area 5b, medial division"],
    ["5m", "Area 5, medial division"],
    ["S2", "Second somatosensory cortex"],
    ["S2m", "Second somatosensory cortex, medial division"],
    ["S3", "Third somatosensory cortex"],
    ["S4", "Fourth somatosensory cortex"],
    ["S5", "Fifth somatosensory cortex"],
    ["17", "Area 17"],
    ["18", "Area 18"],
    ["19", "Area 19"],
    ["20a", "Area 20a"],
    ["20b", "Area 20b"],
    ["21a", "Area 21a"],
    ["21b", "Area 21b"],
    ["7a", "Area 7"],
    ["7m", "Area 7"],
    ["7p", "Area 7"],
    ["AEV", "Anterior ectosylvian visual area"],
    ["ALLS", "Anterolateral lateral suprasylvian area"],
    ["AMLS", "Anteromedial lateral suprasylvian area"],
    ["CVA", "Cingulate visual area"],
    ["DLS", "Dorsolateral suprasylvian visual area"],
    ["PLLS", "Posterolateral lateral suprasylvian area"],
    ["PMLS", "Posteromedial lateral suprasylvian area"],
    ["PS", "Posterior suprasylvian visual area"],
    ["SVA", "Splenial visual area"],
    ["VLS", "Ventrolateral suprasylvian area"],
    ["4Delta", "Area praecentralis macropyramidalis"],
    ["4fu", "Area praecentralis in fundo"],
    ["4Gamma", "Area praecentralis"],
    ["4sfu", "Area praecentralis supra fundo"],
    ["6aAlpha", "Area frontalis agranularis mediopyramidalis"],
    ["6aBeta", "Area frontalis agranularis macropyramidalis"],
    ["6aGamma", "Area 6, lateral division"],
    ["6iffu", "Area 6, infra fundum"],
    ["PFdl", "Prefrontal cortex, dorsolateral division"],
    ["PFdm", "Prefrontal cortex, dorsomedial division"],
    ["PFv", "Prefrontal cortex, ventral division"],
    ["36", "Perirhinal cortex"],
    ["AId", "Agranular insular area, dorsal division"],
    ["AIv", "Agranular insular area, ventral division"],
    ["DI", "Dysgranular insular area"],
    ["GI", "Granular insular area"],
    ["CGa", "Anterior cingulate area"],
    ["CGp", "Posterior cingulate area"],
    ["PL", "Prelimbic area"],
    ["G", "Primary gustatory area"],
    ["MZ", "Multisensory zone"],
    ["Pp", "Prepyriform cortex"],
    ["RS", "Retrosplenial area"],
    ["Cbm", "Cerebellum"],
]


if __name__ == "__main__":

    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    working_dir = bg_root_dir
    temp_download_dir = bg_root_dir / "download_dir"

    bg_root_dir.mkdir(parents=True, exist_ok=True)
    local_file_path_list = download_resources(working_dir)
    template_volume, reference_volume = retrieve_template_and_annotations(
        local_file_path_list
    )
    structures_dict = retrieve_structure_information(
        local_file_path_list, acronym_to_region_map
    )

    print("Converting VTK files into .obj mesh")
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
        atlas_packager=ATLAS_PACKAGER,
    )
