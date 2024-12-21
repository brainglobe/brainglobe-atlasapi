import json
import multiprocessing as mp
import time
from pathlib import Path
from os import listdir, path
import pooch

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_nii
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "kim_devccf_mouse"  # multiple versions of the same atlas
SPECIES = "Mus musculus"
ATLAS_LINK = "https://kimlab.io/brain-map/DevCCF/"
CITATION = "Kronman, F.N., Liwang, J.K., Betty, R. et al. 2024, https://doi.org/10.1038/s41467-024-53254-w"
ORIENTATION = ["left", "superior", "posterior"]
ROOT_ID = 15564
RESOLUTION_UM = 20
VERSION = 1
PACKAGER = "Carlo Castoldi <castoldi[at]ipmc.cnrs.fr>"
ATLAS_FILE_URL = "https://doi.org/10.6084/m9.figshare.26377171.v1"

TIMEPOINTS = (
    "E11.5",
    "E13.5",
    "E15.5",
    "E18.5",
    "P04",
    "P14",
    "P56"
)
RESOLUTIONS = (20, 50)
MODALITIES = (
    "LSFM",     # Light Sheet Fluorescence Microscopy
    "MRI-adc",  # MRI Apparent Diffusion Coefficient
    "MRI-dwi",  # MRI Difusion Weighted Imaging
    "MRI-fa",   # MRI Fractional Anisotropy
    "MRI-MTR",  # MRI Magnetization Transfer Ratio
    "MRI-T2"    # MRI T2-weighted
)

def pooch_init(download_dir_path: Path):
    utils.check_internet_connection()

    hash = None
    registry = {a+".zip": hash for a in TIMEPOINTS}
    registry["DevCCFv1_OntologyStructure.xlsx"] = hash
    return pooch.create(
        path=download_dir_path, #/ATLAS_NAME,
        base_url="doi:10.6084/m9.figshare.26377171.v1/",
        registry=registry
    )

def fetch_animal(pooch_: pooch.Pooch, age: str):
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"
    archive = age+".zip"
    members = [
        f"{age}/{age.replace('.','-')}_DevCCF_Annotations_20um.nii.gz",
        f"{age}/{age.replace('.','-')}_LSFM_20um.nii.gz"
    ]
    annotations_path, reference_path = pooch_.fetch(archive,
                        progressbar=True,
                        processor=pooch.Unzip(extract_dir=".", members=members)
                        )
    # archive_path: Path = (pooch_.path/archive)
    # archive_path.unlink()
    annotations = load_nii(annotations_path, as_array=True)
    reference = load_nii(reference_path, as_array=True)
    return annotations, reference

def fetch_ontology(pooch_: pooch.Pooch):
    devccfv1_path = pooch_.fetch("DevCCFv1_OntologyStructure.xlsx", progressbar=True)
    xl = pd.ExcelFile(devccfv1_path)
    # xl.sheet_names # it has two excel sheets
                     # 'DevCCFv1_Ontology', 'README'
    df = xl.parse("DevCCFv1_Ontology", header=1)
    df = df[["Acronym", "ID16", "Name", "Structure ID Path16", "R", "G", "B"]]
    df.rename(columns={
        "Acronym": "acronym",
        "ID16": "id",
        "Name": "name",
        "Structure ID Path16": "structure_id_path",
        "R": "r",
        "G": "g",
        "B": "b"
    }, inplace=True)
    structures = list(df.to_dict(orient="index").values())
    for structure in structures:
        if structure["acronym"] == "mouse":
            structure["acronym"] = "root"
        structure_path = structure["structure_id_path"]
        structure["structure_id_path"] = [int(id) for id in structure_path.strip("/").split("/")]
        structure["rgb_triplet"] = [structure["r"], structure["g"], structure["b"]]
        del structure["r"]
        del structure["g"]
        del structure["b"]
    return structures

def create_meshes(output_path: str|Path,
                  structures, annotation_volume, root_id):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures)

    labels = np.unique(annotation_volume).astype(np.uint16)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False
        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2
    decimate_fraction = 0.2 # 0.04
    # What fraction of the original number of vertices is to be kept.
    smooth = False  # smooth meshes after creation
    start = time.time()
    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        # list(tree.nodes.values())[:5],
        # total=5,
        description="Creating meshes",
    ):
        output_file = output_path/f"{node.identifier}.obj"
        if output_file.exists():
            # print(f"mesh already existing: {output_file.exists()} - {output_file}")
            continue
        # root_node = tree.nodes[root_id]
        create_region_mesh(
            (
                output_path,
                # root_node,
                node,
                tree,
                labels,
                annotation_volume,
                root_id,
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
    return output_path

def create_mesh_dict(structures, meshes_dir_path):
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
    return meshes_dict, structures_with_mesh

if __name__ == "__main__":
    bg_root_dir = DEFAULT_WORKDIR/ATLAS_NAME
    download_dir_path = bg_root_dir/"downloads"
    download_dir_path.mkdir(exist_ok=True, parents=True)
    pooch_ = pooch_init(download_dir_path)
    structures = fetch_ontology(pooch_)
    # save regions list json:
    with open(bg_root_dir/"structures.json", "w") as f:
        json.dump(structures, f)

    for age in TIMEPOINTS:
        atlas_name = f"{ATLAS_NAME}_{age.replace('.', '-')}"
        annotation_volume, reference_volume = fetch_animal(pooch_, age)
        atlas_dir = bg_root_dir/atlas_name
        atlas_dir.mkdir(exist_ok=True)
        print(f"Saving atlas data at {atlas_dir}")
        # Create meshes:
        meshes_dir_path = atlas_dir/"meshes"
        create_meshes(
            meshes_dir_path,
            structures,
            annotation_volume,
            ROOT_ID
        )
        meshes_dict, structures_with_mesh = create_mesh_dict(
            structures, meshes_dir_path
        )
        # Wrap up, compress, and remove file:
        print("Finalising atlas")
        output_filename = wrapup_atlas_from_data(
            atlas_name=atlas_name,
            atlas_minor_version=VERSION,
            citation=CITATION,
            atlas_link=ATLAS_LINK,
            species=SPECIES,
            resolution=(RESOLUTION_UM,)*3,
            orientation=ORIENTATION,
            root_id=ROOT_ID,
            reference_stack=reference_volume,
            annotation_stack=annotation_volume,
            structures_list=structures_with_mesh,
            meshes_dict=meshes_dict,
            working_dir=atlas_dir,
            atlas_packager=PACKAGER,
            hemispheres_stack=None, # it is symmetric
            cleanup_files=False,
            compress=True,
            scale_meshes=True,
            # resolution_mapping=[2, 1, 0],
        )
        print("Done. Atlas generated at: ", output_filename)