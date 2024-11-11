__version__ = "2"

import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii
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

PARALLEL = False  # disable parallel mesh extraction for easier debugging

ATLAS_GROUP_NAME = "kim_dev_mouse"  # multiple versions of the same atlas
SPECIES = "Mus musculus"
ATLAS_LINK = "https://data.mendeley.com/datasets/2svx788ddf/1"
CITATION = (
    "Kim, Yongsoo (2022), “KimLabDevCCFv001”, Mendeley Data, "
    "V1, doi: 10.17632/2svx788ddf.1"
)
ORIENTATION = "asl"
ROOT_ID = 99999999
ANNOTATIONS_RES_UM = 10
ATLAS_FILE_URL = "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/2svx788ddf-1.zip"


def clean_up_df_entries(df):
    """
    Remove ' from string entries in the csv
    """
    df["Acronym"] = df["Acronym"].apply(lambda x: x.replace("'", ""))
    df["Name"] = df["Name"].apply(lambda x: x.replace("'", ""))
    df["ID"] = df["ID"].apply(
        lambda x: int(x)
    )  # convert from numpy to int() for dumping as json

    ints = [int(ele) for ele in df["ID"]]
    df["ID"] = ints


def get_structure_id_path_from_id(id, id_dict, root_id):
    """
    Create the structure_id_path for a region
    from a dict mapping id to parent_id
    """
    structure_id_path = [id]
    if id == root_id:
        return structure_id_path

    while True:
        parent = int(id_dict[id])
        structure_id_path.insert(0, parent)

        if parent == root_id:
            break

        id = parent

    return structure_id_path


def create_atlas(
    working_dir,
    resolution,
    reference_key,
    reference_filename,
    mesh_creation,
    existing_mesh_dir_path=None,
):

    atlas_name = f"kim_dev_mouse_{reference_key}"
    # Temporary folder for  download:
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    utils.check_internet_connection()
    destination_path = download_dir_path / "atlas_download"

    pooch.retrieve(
        url=ATLAS_FILE_URL,
        known_hash="bfc38d1992aa2300f4c12a510c50fe380b3f4cbdb3664888587f874dd1a16823",
        path=destination_path,
        progressbar=True,
        processor=pooch.Unzip(extract_dir="."),
    )

    # Set paths to volumes
    structures_file = (
        destination_path
        / "KimLabDevCCFv001"
        / "KimLabDevCCFv001_MouseOntologyStructure.csv"
    )
    annotations_file = (
        destination_path
        / "KimLabDevCCFv001"
        / "10um"
        / "KimLabDevCCFv001_Annotations_ASL_Oriented_10um.nii.gz"
    )
    template_file = (
        destination_path / "KimLabDevCCFv001" / "10um" / reference_filename
    )

    # ---------------- #
    #   GET TEMPLATE   #
    # ---------------- #

    # Load (and possibly downsample) annotated volume:
    scaling = ANNOTATIONS_RES_UM / resolution

    annotated_volume = load_nii(annotations_file, as_array=True)
    template_volume = load_nii(template_file, as_array=True)

    annotated_volume = zoom(
        annotated_volume, (scaling, scaling, scaling), order=0, prefilter=False
    )

    # ------------------------ #
    #   STRUCTURES HIERARCHY   #
    # ------------------------ #

    # Parse region names & hierarchy
    df = pd.read_csv(structures_file)
    clean_up_df_entries(df)

    new_row = pd.DataFrame(
        [["root", ROOT_ID, "root", ROOT_ID]], columns=df.columns
    )
    df = pd.concat([df, new_row], ignore_index=True)

    id_dict = dict(zip(df["ID"], df["Parent ID"]))

    assert id_dict[15564] == "[]"
    id_dict[15564] = ROOT_ID

    structures = []
    for row in range(df.shape[0]):
        entry = {
            "acronym": df["Acronym"][row],
            "id": int(df["ID"][row]),  # from np.int for JSON serialization
            "name": df["Name"][row],
            "structure_id_path": get_structure_id_path_from_id(
                int(df["ID"][row]), id_dict, ROOT_ID
            ),
            "rgb_triplet": [255, 255, 255],
        }

        structures.append(entry)

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # ----------------- #
    #   Create Meshes   #
    # ----------------- #

    print(f"Saving atlas data at {download_dir_path}")

    if mesh_creation == "copy":
        meshes_dir_path = Path(existing_mesh_dir_path)
    else:
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

    if mesh_creation == "generate":
        closing_n_iters = 2
        decimate_fraction = 0.04
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
        atlas_name=atlas_name,
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
    """
    This atlas is too large to package into a single atlas.
    Hence it is split with one atlas per reference.
    To avoid re-generating the meshes for each creation,
    the script should be run once with mesh_creation = 'generate'.
    This will generate the standard template atlas with the meshes.
    For the rest of the references, use mesh_creation = 'copy',
    and set the existing_mesh_dir_path to the previously-generated meshes.

    Note the decimate fraction is set to 0.04
    to further reduce size of this large atlas.
    """
    resolution = 10  # some resolution, in microns (10, 25, 50, 100)

    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_GROUP_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    # First create the standard template, including all meshes

    create_atlas(
        bg_root_dir,
        resolution,
        reference_key="stp",
        reference_filename="CCFv3_average_template_ASL_Oriented_u16_10um.nii.gz",
        mesh_creation="generate",
    )

    # Now get the mesh path from the previously generated atlas and use this
    # for all other atlases

    additional_references = {
        "idisco": "KimLabDevCCFv001_iDiscoLSFM2CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_a0": "KimLabDevCCFv001_P56_MRI-a02CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_adc": "KimLabDevCCFv001_P56_MRI-adc2CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_dwi": "KimLabDevCCFv001_P56_MRI-dwi2CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_fa": "KimLabDevCCFv001_P56_MRI-fa2CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_mtr": "KimLabDevCCFv001_P56_MRI-MTR2CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
        "mri_t2": "KimLabDevCCFv001_P56_MRI-T22CCF_"
        "avgTemplate_ASL_Oriented_10um.nii.gz",
    }

    existing_mesh_dir_path = bg_root_dir / "downloads" / "meshes"

    for reference_key, reference_filename in additional_references.items():
        create_atlas(
            bg_root_dir,
            resolution,
            reference_key,
            reference_filename,
            mesh_creation="copy",
            existing_mesh_dir_path=existing_mesh_dir_path,
        )
