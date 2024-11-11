__version__ = "1"
__atlas__ = "admba_3d_dev_mouse"

import dataclasses
import json
import multiprocessing as mp
import time
from os import listdir, path
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pooch
from rich.progress import track
from skimage import io

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

PARALLEL = False


def download_atlas_files(
    download_dir_path, atlas_file_url, ATLAS_NAME, known_hash
):
    utils.check_internet_connection()

    try:
        download_name = ATLAS_NAME + "_atlas.zip"
    except TypeError:
        download_name = ATLAS_NAME / "_atlas.zip"
    destination_path = download_dir_path / download_name

    pooch.retrieve(
        url=atlas_file_url,
        known_hash=known_hash,
        path=destination_path,
        progressbar=True,
        processor=pooch.Unzip(extract_dir="."),
    )

    return destination_path


def parse_structures(structures_file, root_id):
    df = pd.read_csv(structures_file)
    df = df.rename(columns={"Parent": "parent_structure_id"})
    df = df.rename(columns={"Region": "id"})
    df = df.rename(columns={"RegionName": "name"})
    df = df.rename(columns={"RegionAbbr": "acronym"})
    df = df.drop(columns=["Level"])
    # get length of labels so as to generate rgb values
    no_items = df.shape[0]
    # Random values for RGB
    # could use this instead?
    rgb_list = [
        [
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        ]
        for i in range(no_items)
    ]
    rgb_list = pd.DataFrame(rgb_list, columns=["red", "green", "blue"])

    df["rgb_triplet"] = rgb_list.apply(
        lambda x: [x.red.item(), x.green.item(), x.blue.item()], axis=1
    )
    df["structure_id_path"] = df.apply(lambda x: [x.id], axis=1)
    structures = df.to_dict("records")
    structures = create_structure_hierarchy(structures, df, root_id)
    return structures


def create_structure_hierarchy(structures, df, root_id):
    for structure in structures:
        if structure["id"] != root_id:
            parent_id = structure["parent_structure_id"]
            while True:
                structure["structure_id_path"] = [parent_id] + structure[
                    "structure_id_path"
                ]
                if parent_id != root_id:
                    parent_id = int(
                        df[df["id"] == parent_id]["parent_structure_id"]
                    )
                else:
                    break
        else:
            structure["name"] = "root"
            structure["acronym"] = "root"

        del structure["parent_structure_id"]

    return structures


def create_meshes(download_dir_path, structures, annotated_volume, root_id):
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
                    node,  # root_node
                    tree,
                    labels,
                    annotated_volume,
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


@dataclasses.dataclass
class AtlasConfig:
    """Data class to configure atlas creation."""

    atlas_name: str
    species: str
    atlas_link: str
    atlas_file_url: str
    #: Input orientation in 3-letter notation using the NumPy system with
    #: origin at top left corner of first plane.
    #: Axis 0 = front to back, 1 = top to bottom, 2 = left to right.
    #: Output orientation will be ASR.
    orientation: str
    #: Resolution to match the output orientation of ASR.
    resolution: Tuple[float, float, float]
    citation: str
    root_id: int
    atlas_packager: str
    # pooch hash for remote atlas file:
    known_hash: str


def create_atlas(
    working_dir: Path = Path.home(), atlas_config: "AtlasConfig" = None
):
    assert (
        len(atlas_config.orientation) == 3
    ), f"Orientation is not 3 characters, Got {atlas_config.orientation}"
    assert (
        len(atlas_config.resolution) == 3
    ), f"Resolution is not correct, Got {atlas_config.resolution}"
    assert (
        atlas_config.atlas_file_url
    ), f"No download link provided for atlas in {atlas_config.atlas_file_url}"
    if isinstance(working_dir, str):
        working_dir = Path(working_dir)
    # Generated atlas path:
    working_dir = working_dir / atlas_config.atlas_name
    working_dir.mkdir(exist_ok=True, parents=True)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    if path.isdir(atlas_config.atlas_file_url):
        print("Setting atlas to directory: ", atlas_config.atlas_file_url)
        atlas_files_dir = atlas_config.atlas_file_url
    else:
        # Download atlas files from link provided
        print("Downloading atlas from link: ", atlas_config.atlas_file_url)
        atlas_files_dir = download_atlas_files(
            download_dir_path,
            atlas_config.atlas_file_url,
            atlas_config.atlas_name,
            atlas_config.known_hash,
        )
        ## Load files

    structures_file = atlas_files_dir / (
        [f for f in listdir(atlas_files_dir) if "region_ids_ADMBA" in f][0]
    )

    reference_file = atlas_files_dir / (
        [f for f in listdir(atlas_files_dir) if "atlasVolume.mhd" in f][0]
    )

    annotations_file = atlas_files_dir / (
        [f for f in listdir(atlas_files_dir) if "annotation.mhd" in f][0]
    )
    # segments_file = atlas_files_dir / "Segments.csv"

    annotated_volume = io.imread(annotations_file)
    template_volume = io.imread(reference_file)

    ## Parse structure metadata
    structures = parse_structures(structures_file, atlas_config.root_id)

    # save regions list json:
    with open(download_dir_path / "structures.json", "w") as f:
        json.dump(structures, f)

    # Create meshes:
    print(f"Saving atlas data at {download_dir_path}")
    meshes_dir_path = create_meshes(
        download_dir_path, structures, annotated_volume, atlas_config.root_id
    )

    meshes_dict, structures_with_mesh = create_mesh_dict(
        structures, meshes_dir_path
    )

    # Wrap up, compress, and remove file:
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=atlas_config.atlas_name,
        atlas_minor_version=__version__,
        citation=atlas_config.citation,
        atlas_link=atlas_config.atlas_link,
        species=atlas_config.species,
        resolution=atlas_config.resolution,
        orientation=atlas_config.orientation,
        root_id=atlas_config.root_id,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=atlas_config.atlas_packager,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        resolution_mapping=[2, 1, 0],
    )
    print("Done. Atlas generated at: ", output_filename)
    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / __atlas__
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    # set up E11.5 atlas settings and use as template for rest of brains
    e11_5_config = AtlasConfig(
        atlas_name="admba_3d_e11_5_mouse",
        species="Mus musculus",
        atlas_link="https://search.kg.ebrains.eu/instances/8ab25629-bdac-47d0-bc86-6f3aa3885f29",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000023_3Drecon-ADMBA-E11pt5_pub",
        orientation="rsa",
        resolution=(
            16,
            16,
            20,
        ),  # this is in ASR, the target orientation.
        # The resolution in LR is lower than SI and AP!
        citation="Young et al. 2021, https://doi.org/10.7554/eLife.61408",
        root_id=15564,
        atlas_packager="Pradeep Rajasekhar, WEHI, Australia, "
        "rajasekhardotp@wehidotedudotau; David Young, UCSF, "
        "United States, davedotyoung@ucsfdotedu",
        known_hash="30e978c0f72939c9967442e3bc18adbb1d55eba902df4cfa76d4317df5059b08",
    )

    # E13.5 atlas, with updated name and URLs
    e13_5_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_e13_5_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/bdb89f61-8dc4-4255-b4d5-50d470958b58",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000024_3Drecon-ADMBA-E13pt5_pub",
        known_hash="7ab6d7fc62a7cce26ed0691716ccd00067df47e1a90ed05ed6a017709455593b",
    )

    # E15.5 atlas
    e15_5_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_e15_5_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/Dataset/51a81ae5-d821-437a-a6d5-9b1f963cfe9b",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000025_3Drecon-ADMBA-E15pt5_pub",
        known_hash="b85be368d2460c7193d7ecfbae91acefd88e4022d1d6d4c650d5b3082c56d43b",
    )

    # E18.5 atlas
    e18_5_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_e18_5_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/633b41be-867a-4611-8570-82271aebd516",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000026_3Drecon-ADMBA-E18pt5_pub",
        known_hash="18b57e4db2c2d87d5d19295b4167de0c67a24e63061d01098724615beed0192a",
    )

    # P4 atlas, which has different resolutions
    p4_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_p4_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/eea3589f-d74b-4988-8f4c-fd9ae8e3a4b3",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000027_3Drecon-ADMBA-P4_pub",
        resolution=(16.752, 16.752, 20),
        known_hash="87dd3411737b5fdccffabfddb0ae900d358427156d397dc04668e3442f59b0f2",
    )

    # P14 atlas, which has slightly different resolutions
    p14_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_p14_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/114e50aa-156c-4283-af73-11b7f03d287e",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000028_3Drecon-ADMBA-P14_pub",
        resolution=(16.752, 16.752, 25),
        known_hash="7c81bdb68493d7a31f865a06c08f430229e04dc5f1c0b85d4434fee8c2d0ebac",
    )

    # P28 atlas, which has same resolutions as P14
    p28_config = dataclasses.replace(
        p14_config,
        resolution=(16.752, 16.752, 25),
        atlas_name="admba_3d_p28_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/3a1153f0-6779-43bd-9f02-f92700a585a4",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000029_3Drecon-ADMBA-P28_pub",
        known_hash="be108596c14957f4b25019f4d23477f9e1abbb3d3e014c112ac42eec7b3686fd",
    )

    # P56 atlas, which has different resolutions
    p56_config = dataclasses.replace(
        e11_5_config,
        atlas_name="admba_3d_p56_mouse",
        atlas_link="https://search.kg.ebrains.eu/instances/a7e99105-1ec2-42e2-a53a-7aa0f2b78135",
        atlas_file_url="https://data.kg.ebrains.eu/zip?container=https://object.cscs.ch/v1/AUTH_4791e0a3b3de43e2840fe46d9dc2b334/ext-d000030_3Drecon-ADMBA-P56_pub",
        resolution=(25, 25, 25),
        known_hash="0bad5f8e4dfa256d5f634834f1f07b58356b16a9165aebe3ff39f3133901986d",
    )

    # atlases to create
    configs = (
        e11_5_config,
        e13_5_config,
        e15_5_config,
        e18_5_config,
        p4_config,
        p14_config,
        p28_config,
        p56_config,
    )

    # create each atlas
    for config in configs:
        create_atlas(bg_root_dir, config)
