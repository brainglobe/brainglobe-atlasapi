__version__ = "0"

import random
import shutil
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import tifffile
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def create_atlas(working_dir):
    ATLAS_NAME = "prairie_vole"
    SPECIES = "Microtus ochrogaster"
    ATLAS_LINK = "https://doi.org/10.7554/eLife.87029"
    CITATION = (
        "Morgan L Gustison, Rodrigo Muñoz-Castañeda, Pavel Osten, "
        "Steven M Phelps (2024) Sexual coordination "
        "in a whole-brain map of prairie vole pair bonding eLife 12:RP87029 "
        "https://doi.org/10.7554/eLife.87029.3"
    )
    ATLAS_BASE_URL = "https://ndownloader.figshare.com/files/"
    ORIENTATION = "asr"
    ROOT_ID = 997
    ATLAS_PACKAGER = "Sacha Hadaway-Andreae"
    ADDITIONAL_METADATA = {}

    # setup folder for downloading
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"

    files = {
        "PrV_ReferenceBrain_bilateral.tif": "42917449",
        "PrV_Annotation_bilateral.tif": "42917452",
        "PrV_Atlas_Ontology.csv": "42917443",
        "PrV_results_datasets.zip": "42501327",
    }

    utils.check_internet_connection()
    pooch.create(
        path=atlas_path,
        base_url=ATLAS_BASE_URL,
        registry={},
    )

    for filename, file_id in files.items():
        url = f"{ATLAS_BASE_URL}{file_id}"
        pooch.retrieve(
            url=url,
            fname=filename,
            path=atlas_path,
            known_hash=None,
        )

    zip_file_path = atlas_path / "PrV_results_datasets.zip"
    with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
        zip_ref.extract(
            "results_datasets/neural_nomenclature_hierarchy.csv", atlas_path
        )

    if not (atlas_path / "neural_nomenclature_hierarchy.csv").exists():
        shutil.move(
            (
                atlas_path
                / "results_datasets"
                / "neural_nomenclature_hierarchy.csv"
            ),
            atlas_path,
        )
    else:
        print(
            f"File '{(atlas_path / 'neural_nomenclature_hierarchy.csv')}' "
            f"already exists, skipping move operation."
        )

    shutil.rmtree(atlas_path / r"results_datasets")

    annotations_file = atlas_path / "PrV_Annotation_bilateral.tif"
    reference_file = atlas_path / "PrV_ReferenceBrain_bilateral.tif"
    meshes_dir_path = atlas_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    nomenclature_hierarchy = pd.read_csv(
        atlas_path / "neural_nomenclature_hierarchy.csv"
    ).values
    reference_id = pd.read_csv(atlas_path / "PrV_Atlas_Ontology.csv")

    acronyms = reference_id["acronym"].values
    names = reference_id["name"].values
    ids = reference_id["id"].values

    acronym_to_id = dict(zip(acronyms, ids))
    name_to_id = dict(zip(names, ids))

    ids = list(ids)
    hierarchy = [
        {
            "acronym": "root",
            "id": 997,
            "name": "root",
            "structure_id_path": [997],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    for row in nomenclature_hierarchy:
        structure_hierarchy = []
        unique_pairs = []
        pairs = [(row[i], row[i + 1]) for i in range(0, len(row), 2)]
        pairs_seen = set()

        for pair in pairs:
            if pair not in pairs_seen:
                unique_pairs.append(pair)
                pairs_seen.add(pair)

        for acronym, name in unique_pairs:
            if acronym in acronym_to_id:
                structure_hierarchy.append(acronym_to_id[acronym])
            elif name in name_to_id:
                structure_hierarchy.append(name_to_id[name])

        structure_hierarchy = [ROOT_ID] + structure_hierarchy

        result = [
            structure_hierarchy[: i + 1]
            for i in range(len(structure_hierarchy))
        ]

        for a in result:
            structure_template = {
                "acronym": acronyms[ids.index(a[-1])],
                "id": a[-1],
                "name": names[ids.index(a[-1])],
                "structure_id_path": a,
                "rgb_triplet": [random.randint(0, 255) for _ in range(3)],
            }

            hierarchy.append(structure_template)

    hierarchy = list(
        {tuple(d["structure_id_path"]): d for d in hierarchy}.values()
    )

    # use tifffile to read annotated file
    annotated_volume = tifffile.imread(annotations_file)
    reference_volume = tifffile.imread(reference_file)

    print(f"Saving atlas data at {atlas_path}")

    tree = get_structures_tree(hierarchy)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )
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

    # Create meshes dict & hierarchy to match
    hierarchy_with_mesh = []
    meshes_dict = dict()
    structs_with_mesh = []
    for s in hierarchy:
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

        hierarchy_with_mesh.append(s)

    print(
        f"In the end, {len(structs_with_mesh)} structures with mesh are kept"
    )

    for item in hierarchy_with_mesh:
        item["id"] = int(item["id"])
        item["structure_id_path"] = [int(i) for i in item["structure_id_path"]]

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=hierarchy_with_mesh,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
        resolution=(25, 25, 25),
        meshes_dict=meshes_dict,
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir)
