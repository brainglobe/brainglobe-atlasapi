__version__ = "0"

import os
import random
import shutil
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import tifffile

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data


def create_atlas(working_dir):
    ATLAS_NAME = "..."
    SPECIES = "..."
    ATLAS_LINK = "https://elifesciences.org/articles/87029#data"
    CITATION = "..."
    ATLAS_BASE_URL = "https://ndownloader.figshare.com/files/"
    ORIENTATION = "..."
    ROOT_ID = 997
    ATLAS_PACKAGER = "..."
    ADDITIONAL_METADATA = {}

    # setup folder for downloading
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"

    files = {
        "PrV_ReferenceBrain_bilateral.tif": "42917449",
        "PrV_Annotation_bilateral.tif": "42917452",
        "PrV_Atlas_Ontology.csv": "42917443",
        "PrV_results_datasets.zip": "42501327"
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

    with zipfile.ZipFile(r"C:\Users\sacha\Downloads\atlas shit\PrV_results_datasets.zip", 'r') as zip_ref:
        file_list = zip_ref.namelist()
        print(f"Files: {file_list}")
        zip_ref.extract('results_datasets/neural_nomenclature_hierarchy.csv', atlas_path)

    shutil.move(f"{atlas_path}\\results_datasets\\neural_nomenclature_hierarchy.csv", atlas_path)
    os.rmdir(f"{atlas_path}\\results_datasets")

    annotations_file = (
            atlas_path / "PrV_Annotation_bilateral.tif"
    )
    reference_file = atlas_path / "PrV_ReferenceBrain_bilateral.tif"

    nmh = pd.read_csv('neural_nomenclature_hierarchy.csv').values
    reference_id = pd.read_csv('PrV_Atlas_Ontology.csv')

    acronyms = reference_id['acronym'].values
    names = reference_id['name'].values
    ids = reference_id['id'].values

    acronym_to_id = dict(zip(acronyms, ids))
    name_to_id = dict(zip(names, ids))

    ids = list(ids)
    full_list = []

    for row in nmh:
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

        structure_template = {
            "acronym": acronyms[ids.index(structure_hierarchy[-1])],
            "id": structure_hierarchy[-1],
            "name": names[ids.index(structure_hierarchy[-1])],
            "structure_id_path": structure_hierarchy,
            "rgb_triplet": [random.randint(0, 255) for _ in range(3)]
        }

        full_list.append(structure_template)

    # use tifffile to read annotated file
    annotated_volume = tifffile.imread(annotations_file).astype(np.uint8)
    reference_volume = tifffile.imread(reference_file)

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
        structures_list=full_list,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,

        resolution=("idk",),
        meshes_dict={"": "i dont have one"}
    )

    return output_filename


if __name__ == "__main__":
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir)
