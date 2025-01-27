import time
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_any
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0
PARALLEL = False
# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "nadkarni_mri_mouselemur"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1016/j.dib.2018.10.067"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Microcebus murinus"

# The URL for the data files
ATLAS_LINK = (
    "https://www.nitrc.org/frs/download.php/10867/MIRCen-Mouse"
    "LemurAtlas_V0.01.tar.gz"
)

# The orientation of the **original** atlas data, in BrainGlobe convention:
ORIENTATION = "lia"
ROOT_ID = 9999
# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 91
BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """
    # Define the path to the doggie bag
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)
    known_hash = (
        "327dd8efc73cc2c2ca34bc8b86a4afd88f083be283431e1fb3f063d901da9de3"
    )
    # Create a new doggie bag
    pooch.retrieve(
        ATLAS_LINK,
        known_hash=known_hash,
        path=DOWNLOAD_DIR_PATH,
        processor=pooch.Untar(),
    )


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    reference = load_any(
        DOWNLOAD_DIR_PATH
        / (
            "af60cf392f8cd8b22925dddedada8e58-MIRCen-MouseLemurAtlas_V0.01.tar."
            "gz.untar"
        )
        / "MIRCen-MouseLemurAtlas_V0.01"
        / "MouseLemurHeadTemplate_91mu_V0.01.nii.gz"
    )
    annotation = load_any(
        DOWNLOAD_DIR_PATH
        / (
            "af60cf392f8cd8b22925dddedada8e58-MIRCen-MouseLemurAtlas_V0.01.tar."
            "gz.untar"
        )
        / "MIRCen-MouseLemurAtlas_V0.01"
        / "MouseLemurLabels_V0.01.nii.gz"
    )
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None  # Symmetrical


def retrieve_structure_information():
    """
    This function should return a pandas DataFrame with information about your
    atlas.

    The DataFrame should be in the following format:

    ╭────┬───────────────────┬─────────┬───────────────────┬─────────────────╮
    | id | name              | acronym | structure_id_path | rgb_triplet     |
    |    |                   |         |                   |                 |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997| root              | root    | [997]             | [255, 255, 255] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8  | Basic cell groups | grey    | [997, 8]          | [191, 218, 227] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567| Cerebrum          | CH      | [997, 8, 567]     | [176, 240, 255] |
    ╰────┴───────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    col_names = ["IDX", "R", "G", "B", "A", "VIS", "MSH", "LABEL"]
    df = pd.read_csv(
        DOWNLOAD_DIR_PATH
        / (
            "af60cf392f8cd8b22925dddedada8e58-MIRCen-MouseLemurAtlas_V0.01.tar"
            ".gz.untar"
        )
        / "MIRCen-MouseLemurAtlas_V0.01"
        / "MouseLemurLabelNames.txt",
        comment="#",
        delim_whitespace=True,
        names=col_names,
        header=None,
    )
    new_df = pd.DataFrame()
    new_df["id"] = df["IDX"]
    new_df["name"] = df["LABEL"]
    new_df["acronym"] = df["LABEL"]
    new_df["rgb_triplet"] = df[["R", "G", "B"]].values.tolist()
    new_df["structure_id_path"] = new_df["id"].apply(lambda x: [ROOT_ID, x])
    new_df["rgb_triplet"] = df[["R", "G", "B"]].values.tolist()
    root = pd.DataFrame(
        {
            "id": [ROOT_ID],
            "name": ["root"],
            "acronym": ["root"],
            "structure_id_path": [[ROOT_ID]],
            "rgb_triplet": [[255, 255, 255]],
        }
    )
    new_df = pd.concat([root, new_df]).reset_index(drop=True)
    new_df = new_df[new_df["name"] != "Clear Label"].reset_index(drop=True)
    return new_df.to_dict("records")


def retrieve_or_construct_meshes(structures):
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    print("constructing meshes")

    download_dir_path = BG_ROOT_DIR / "downloads"
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
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(structures)
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
