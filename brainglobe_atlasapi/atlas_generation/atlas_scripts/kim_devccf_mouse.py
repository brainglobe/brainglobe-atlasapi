import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_nii
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

# from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "kim_devccf_mouse"  # multiple versions of the same atlas
SPECIES = "Mus musculus"
AGE = "P14"
ATLAS_LINK = "https://kimlab.io/brain-map/DevCCF/"
CITATION = "Kronman, F.N., Liwang, J.K., Betty, R. et al. 2024, https://doi.org/10.1038/s41467-024-53254-w"
ORIENTATION = "posterior superior left"
ROOT_ID = 15564
RESOLUTION_UM = 20
VERSION = 1
PACKAGER = "Carlo Castoldi <castoldi[at]ipmc.cnrs.fr>"
ATLAS_FILE_URL = "https://pennstateoffice365-my.sharepoint.com/personal/yuk17_psu_edu/_layouts/15/download.aspx?UniqueId=fe3d1692%2D94e4%2D4238%2Db6bc%2D95d18bcac022"
#           curl 'https://pennstateoffice365-my.sharepoint.com/personal/yuk17_psu_edu/_layouts/15/download.aspx?UniqueId=fe3d1692%2D94e4%2D4238%2Db6bc%2D95d18bcac022' -H 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0' -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8' -H 'Accept-Language: en-GB,en;q=0.5' -H 'Accept-Encoding: gzip, deflate, br, zstd' -H 'Referer: https://pennstateoffice365-my.sharepoint.com/personal/yuk17_psu_edu/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fyuk17%5Fpsu%5Fedu%2FDocuments%2FProjects%2FDevelopmental%5FAtlasing%2FDevCCF%5FTeamFileshare%2FDevCCF%5FTemplates%2FDevCCFv1%2B%2FDevCCFv1' -H 'Upgrade-Insecure-Requests: 1' -H 'Sec-Fetch-Dest: iframe' -H 'Sec-Fetch-Mode: navigate' -H 'Sec-Fetch-Site: same-origin' -H 'Connection: keep-alive' -H 'Cookie: FedAuth=77u/PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz48U1A+VjEzLDBoLmZ8bWVtYmVyc2hpcHx1cm4lM2FzcG8lM2Fhbm9uIzc1Yjc1NjA0NGUxZDI5ZDU5NjFmMjZjODc0MzFlZmQwNmU4ZWM2NzYwMTJmZGEyNjgxYzg4Zjc5ODdmMGY2YTAsMCMuZnxtZW1iZXJzaGlwfHVybiUzYXNwbyUzYWFub24jNzViNzU2MDQ0ZTFkMjlkNTk2MWYyNmM4NzQzMWVmZDA2ZThlYzY3NjAxMmZkYTI2ODFjODhmNzk4N2YwZjZhMCwxMzM3NjQ5NTA5NzAwMDAwMDAsMCwxMzM3NjY3MzA4NzYxOTQyMTMsMC4wLjAuMCwyNTgsN2NmNDhkNDUtM2RkYi00Mzg5LWE5YzEtYzExNTUyNmViNTJlLCwsZWU2YTY1YTEtMDBmNS02MDAwLWZiMDYtYTgzYjk1Yjg3NTIzLDkxYzI2NWExLTMwNDctNjAwMC1mYjA2LWFiY2IwMTQ2MTZmNSxDR1EvcVlGcGJFeVlJTHAvM2NHb3pnLDAsMCwwLCwsLDI2NTA0Njc3NDM5OTk5OTk5OTksMCwsLCwsLCwwLCwxOTI2MDksdVhlaFFKUGxlVmpOQ2Jha1VoR0Q2SXlGUVFrLFkzdXYyY1dWdUZMWElneHRSOW4vMVVValJ1UUNMd2wybzdZbE51S0RtNTBxOE16ODJVaCt6K0ZTL0NqUU0rNlZrc090RlkwY211OUJNWHllc2U3Z3dST1VMcXRDWEorV3M1NkVzcEZrdk1qTEI0VzNhbDVZeVRlZ0greURRaUFRbG8rVG82aUVmVndVODU5Z2RWME5GSUprenJSVUlCeWVDUThiMmhMRjhQNmlJTDZHYldBNXVwRnJpcTl6cml6THhheFBRUWtVRzRoUmJjMEgzaVMzMnVKekw1ckFuWnVZSFk1N2J4K2xZOGE5MUNVQ3BSQXJ3aU45NHZ5aXdaQWdlcDRXUkxwWlVWOU5NNFZwZ0xUN05zVUh2K0JBcGwxbnViY0FDVENlWElhbWNwclZ0eTBFdElpajVSeXR3UGk0OFRlSVlIQXVrRjhDUFRFVEdDcTlGQT09PC9TUD4=; FeatureOverrides_experiments=[]; MicrosoftApplicationsTelemetryDeviceId=8293cff3-d815-447c-b343-546e678a0450; MSFPC=GUID=79c2671112584c09b960ec538dca634d&HASH=79c2&LV=202411&V=4&LU=1732022021463; ai_session=O/d7EbWPiaNaTZQh0qUYpP|1732113248412|1732113248412' --output P14.zip
PARALLEL = False  # disable parallel mesh extraction for easier debugging


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

def get_annotations(devccfv1_root: str|Path,
                    age: str,
                    resolution: int):
    if not isinstance(devccfv1_root, Path):
        devccfv1_root = Path(devccfv1_root)
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"
    assert resolution in RESOLUTIONS, f"Unknown resolution in µm: '{resolution}'"
    annotations_path = devccfv1_root/age/f"{age}_DevCCF_Annotations_{resolution}um.nii.gz"
    annotations = load_nii(annotations_path, as_array=True)
    return annotations

def get_reference(devccfv1_root: str|Path,
                  age: str,
                  resolution: int,
                  modality: str):
    if not isinstance(devccfv1_root, Path):
        devccfv1_root = Path(devccfv1_root)
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"
    assert resolution in RESOLUTIONS, f"Unknown resolution: '{resolution}'"
    assert modality in MODALITIES, f"Unknown modality: '{modality}'"
    reference_path = devccfv1_root/age/f"{age}_{modality}_{resolution}um.nii.gz"
    reference = load_nii(reference_path, as_array=True)
    return reference

def get_ontology(devccfv1_root: str|Path):
    if not isinstance(devccfv1_root, Path):
        devccfv1_root = Path(devccfv1_root)
    DevCCFv1_path = devccfv1_root/"DevCCFv1_OntologyStructure.xlsx"
    xl = pd.ExcelFile(DevCCFv1_path)
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

def create_meshes(download_dir_path: str|Path,
                  structures, annotation_volume, root_id):
    if not isinstance(download_dir_path, Path):
        download_dir_path = Path(download_dir_path)
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

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
                        annotation_volume,
                        root_id,
                        closing_n_iters,
                        decimate_fraction,
                        smooth,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            pass
    else:
        for node in track(
            tree.nodes.values(),
            total=tree.size(),
            description="Creating meshes",
        ):
            # root_node = tree.nodes[root_id]
            create_region_mesh(
                (
                    meshes_dir_path,
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
    return meshes_dir_path


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
    atlas_root = "/home/castoldi/Downloads/DevCCFv1"
    # Generated atlas path:
    DEFAULT_WORKDIR = Path.home()/"brainglobe_workingdir"
    bg_root_dir = DEFAULT_WORKDIR/ATLAS_NAME
    download_dir_path = bg_root_dir/"downloads"
    download_dir_path.mkdir(exist_ok=True, parents=True)
    
    structures = get_ontology(atlas_root)
    # save regions list json:
    with open(download_dir_path/"structures.json", "w") as f:
        json.dump(structures, f)
    annotation_volume = get_annotations(atlas_root, AGE, RESOLUTION_UM)
    reference_volume = get_reference(atlas_root, AGE, RESOLUTION_UM, "LSFM")
    # Create meshes:
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = create_meshes(
        download_dir_path,
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
        atlas_name=ATLAS_NAME,
        atlas_minor_version=VERSION,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION_UM,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotation_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        atlas_packager=PACKAGER,
        hemispheres_stack=None, # it is symmetric
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        # resolution_mapping=[2, 1, 0],
    )
    print("Done. Atlas generated at: ", output_filename)