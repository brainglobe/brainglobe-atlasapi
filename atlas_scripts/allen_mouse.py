"""Package the Allen Mouse Brain Atlas for BrainGlobe.

This script downloads data from the Allen Institute and creates a
BrainGlobe-compatible atlas, including reference/annotation volumes,
structures metadata, and meshes.

This file follows the same function-based template as `example_mouse.py`.
"""

import csv
import json

import nrrd
import numpy as np

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.utils import retrieve_over_http

# The minor version of the atlas in brainglobe_atlasapi (1.<minor>)
__version__ = 3

ATLAS_NAME = "allen_mouse"
CITATION = "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"
SPECIES = "Mus musculus"
ATLAS_LINK = "http://www.brain-map.org"
ORIENTATION = "asr"
ROOT_ID = 997
RESOLUTION = 25

BG_ROOT_DIR = DEFAULT_WORKDIR / ATLAS_NAME
ALLEN_ONTOLOGIES_URL = (
    "https://atlas.brain-map.org/atlasviewer/ontologies/1.json"
)
ALLEN_BASE_URL = "https://download.alleninstitute.org/informatics-archive/\
            current-release/mouse_ccf"
ALLEN_TEMPLATE_URL = (
    f"{ALLEN_BASE_URL}/average_template/average_template_{RESOLUTION}.nrrd"
)
ALLEN_ANNOTATION_10_URL = (
    f"{ALLEN_BASE_URL}/annotation/ccf_2022/annotation_10.nrrd"
)
ALLEN_VOXEL_DIFFS_URL = (
    f"{ALLEN_BASE_URL}/annotation/ccf_2022/voxel_count_and_differences.csv"
)
ALLEN_2017_MESH_URL_TEMPLATE = f"{ALLEN_BASE_URL}/annotation/ccf_2017/\
            structure_meshes/{{structure_id}}.obj"


def download_resources() -> None:
    """Download resources required for atlas generation.

    Downloads the reference/annotation NRRDs and the ontologies JSON.
    """
    download_dir_path = BG_ROOT_DIR / "downloading_path"
    download_dir_path.mkdir(exist_ok=True, parents=True)

    ontology_path = download_dir_path / "ontologies_1.json"
    if not ontology_path.exists():
        retrieve_over_http(ALLEN_ONTOLOGIES_URL, ontology_path)

    template_path = download_dir_path / f"average_template_{RESOLUTION}.nrrd"
    if not template_path.exists():
        retrieve_over_http(ALLEN_TEMPLATE_URL, template_path)

    # Allen CCF provides 10µm labels; for 25µm we downsample from 10µm.

    annotation_path = download_dir_path / "annotation_10.nrrd"
    if not annotation_path.exists():
        retrieve_over_http(ALLEN_ANNOTATION_10_URL, annotation_path)

    voxel_diffs_path = download_dir_path / "voxel_count_and_differences.csv"
    if not voxel_diffs_path.exists():
        retrieve_over_http(ALLEN_VOXEL_DIFFS_URL, voxel_diffs_path)


def retrieve_reference_and_annotation():
    """Retrieve the reference (template) and annotation volumes."""

    def downsample_alternating(volume: np.ndarray, pattern: list[int]):
        def make_indices(max_dim: int, pattern: list[int]):
            idx = []
            current = 0
            i = 0
            while current < max_dim:
                idx.append(current)
                current += pattern[i % len(pattern)]
                i += 1
            return np.array(idx)

        idx_z = make_indices(volume.shape[0], pattern)
        idx_y = make_indices(volume.shape[1], pattern)
        idx_x = make_indices(volume.shape[2], pattern)
        return volume[np.ix_(idx_z, idx_y, idx_x)]

    download_dir_path = BG_ROOT_DIR / "downloading_path"
    download_dir_path.mkdir(exist_ok=True, parents=True)

    template_path = download_dir_path / f"average_template_{RESOLUTION}.nrrd"
    if not template_path.exists():
        retrieve_over_http(ALLEN_TEMPLATE_URL, template_path)
    reference, _ = nrrd.read(template_path)

    if RESOLUTION == 25:
        annotation_path = download_dir_path / "annotation_10.nrrd"
        if not annotation_path.exists():
            retrieve_over_http(ALLEN_ANNOTATION_10_URL, annotation_path)
        annotation, _ = nrrd.read(annotation_path)
        annotation = downsample_alternating(annotation, [3, 2])
    else:
        annotation_url = f"{ALLEN_BASE_URL}/annotation/\
            ccf_2022/annotation_{RESOLUTION}.nrrd"
        annotation_path = download_dir_path / f"annotation_{RESOLUTION}.nrrd"
        if not annotation_path.exists():
            retrieve_over_http(annotation_url, annotation_path)
        annotation, _ = nrrd.read(annotation_path)

    annotation = annotation.astype(np.int64, copy=False)
    return reference, annotation


def retrieve_hemisphere_map():
    """Return the hemisphere map (None for symmetric atlases)."""
    return None


def retrieve_structure_information():
    """Retrieve structure metadata for the atlas."""
    download_dir_path = BG_ROOT_DIR / "downloading_path"
    download_dir_path.mkdir(exist_ok=True, parents=True)

    ontology_path = download_dir_path / "ontologies_1.json"
    if not ontology_path.exists():
        retrieve_over_http(ALLEN_ONTOLOGIES_URL, ontology_path)

    with open(ontology_path) as f:
        payload = json.load(f)

    structures_raw = payload.get("msg")
    if not isinstance(structures_raw, list):
        raise ValueError(
            "Unexpected Allen ontology response format: missing `msg` list."
        )

    structures = []
    for s in structures_raw:
        # Allen returns e.g. "/997/8/567/".
        path_string = s.get("structure_id_path")
        structure_id_path = (
            [int(p) for p in path_string.split("/") if p]
            if path_string
            else []
        )
        if not structure_id_path:
            structure_id_path = [int(s["id"])]

        hex_string = s.get("color_hex_triplet") or ""
        hex_string = hex_string.strip().lstrip("#")
        if len(hex_string) == 6:
            rgb_triplet = [int(hex_string[i : i + 2], 16) for i in (0, 2, 4)]
        else:
            rgb_triplet = [255, 255, 255]

        structures.append(
            {
                "id": int(s["id"]),
                "name": s.get("name", ""),
                "acronym": s.get("acronym", ""),
                "structure_id_path": structure_id_path,
                "rgb_triplet": rgb_triplet,
            }
        )

    return structures


def retrieve_or_construct_meshes(annotated_volume: np.ndarray, structures):
    """Construct meshes from the annotation volume.

    Reuse Allen-provided 2017 meshes for unchanged structures (as reported
    by the Allen voxel difference CSV) and generate meshes locally only for
    structures that are new or changed in the 2022 annotation.
    """
    download_dir_path = BG_ROOT_DIR / "downloading_path"
    download_dir_path.mkdir(exist_ok=True, parents=True)

    voxel_diffs_path = download_dir_path / "voxel_count_and_differences.csv"
    if not voxel_diffs_path.exists():
        retrieve_over_http(ALLEN_VOXEL_DIFFS_URL, voxel_diffs_path)

    unchanged_ids: set[int] = set()
    with open(voxel_diffs_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                structure_id = int(row.get("id") or "")
            except ValueError:
                continue

            try:
                prev_only = int(row.get("previous_only_count") or "")
                curr_only = int(row.get("current_only_count") or "")
            except ValueError:
                continue

            if prev_only == 0 and curr_only == 0:
                unchanged_ids.add(structure_id)

    meshes_dir_path = BG_ROOT_DIR / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    meshes_dict = {}
    # the root is also unchanged but treated as if its not
    # bc of course some of its children are.
    unchanged_ids.add(ROOT_ID)
    # The 545 mesh on the allen server is empty... but this only exists in 10um
    if RESOLUTION == 10:
        unchanged_ids.remove(545)
    # Fetch 2017 meshes for structures that are unchanged in 2022.
    for s in structures:
        sid = int(s["id"])
        if sid not in unchanged_ids:
            continue
        mesh_path = meshes_dir_path / f"{sid}.obj"
        if not mesh_path.exists():
            mesh_url = ALLEN_2017_MESH_URL_TEMPLATE.format(structure_id=sid)
            retrieve_over_http(mesh_url, mesh_path)
        meshes_dict[sid] = mesh_path

    # Generate meshes using the full structure tree so parent regions include
    # unchanged children, but skip regenerating meshes we fetched from 2017.
    generated_meshes_dict = construct_meshes_from_annotation(
        save_path=BG_ROOT_DIR,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=10,
        decimate_fraction=0.2,
        smooth=False,
        num_threads=1,
        skip_structure_ids=unchanged_ids,
    )
    meshes_dict.update(generated_meshes_dict)

    structures_with_mesh = [
        s for s in structures if int(s["id"]) in meshes_dict
    ]
    return meshes_dict, structures_with_mesh


def retrieve_additional_references():
    """Return additional reference images (none for this atlas)."""
    return {}


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(exist_ok=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict, structures_with_mesh = retrieve_or_construct_meshes(
        annotated_volume, structures
    )

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
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )

    print(output_filename)
