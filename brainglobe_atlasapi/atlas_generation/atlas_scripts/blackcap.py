__version__ = "0"

import glob as glob
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_nii
from rich.progress import track
from scipy import ndimage
from skimage.filters.rank import median
from skimage.morphology import ball

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def create_atlas(working_dir, resolution):
    ATLAS_NAME = "oldenburg_blackcap"
    SPECIES = "Sylvia atricapilla"
    ATLAS_LINK = "https://uol.de/en/ibu/animal-navigation"
    CITATION = "unpublished"
    ATLAS_FILE_URL = "https://uol.de/en/ibu/animal-navigation"  # noqa
    ORIENTATION = "asr"
    ROOT_ID = 999
    ATLAS_PACKAGER = "BrainGlobe Developers, hello@brainglobe.info"
    ADDITIONAL_METADATA = {}

    # setup folder for downloading

    atlas_path = Path(
        "/media/ceph-neuroinformatics/neuroinformatics/atlas-forge/BlackCap/templates/template_sym_res-50um_n-18/for_atlas/"
    )

    structures_file = atlas_path / "Label_description_BC74white_16_02_2024.txt"
    annotations_file = (
        atlas_path / "sub-BC74_res-50um_labels_aligned-to-reference.nii.gz"
    )
    reference_file = atlas_path / "reference_res-50um_image.nii.gz"
    reference_mask_file = atlas_path / "reference_res-50um_mask-4reg.nii.gz"
    meshes_dir_path = Path.home() / "blackcap-meshes"

    try:
        os.mkdir(meshes_dir_path)
    except FileExistsError:
        "mesh folder already exists"

    # Read structures file
    print("Reading structures file")
    with open(
        structures_file, mode="r", encoding="utf-8-sig"
    ) as blackcap_file:
        structure_data = pd.read_csv(
            blackcap_file,
            delimiter="\s+",
            comment="#",
            names=["IDX", "R", "G", "B", "A", "VIS", "MESH_VIS", "LABEL"],
        )

    # replace Clear Label (first row) with a root entry
    structure_data["LABEL"].iloc[0] = "root"
    structure_data["R"].iloc[0] = 255
    structure_data["G"].iloc[0] = 255
    structure_data["B"].iloc[0] = 255
    structure_data["IDX"].iloc[0] = 999

    structure_data.rename(columns={"IDX": "id"}, inplace=True)
    structure_data.rename(columns={"LABEL": "acronym"}, inplace=True)
    structure_data["name"] = structure_data.apply(
        lambda row: row["acronym"], axis=1
    )
    structure_data["rgb_triplet"] = structure_data.apply(
        lambda row: [str(row["R"]), str(row["G"]), str(row["B"])], axis=1
    )
    structure_data["structure_id_path"] = structure_data.apply(
        lambda row: [row["id"]] if row["id"] == 999 else [999, row["id"]],
        axis=1,
    )

    # drop columns we don't need
    structure_data.drop(
        columns=["A", "VIS", "MESH_VIS", "R", "G", "B"], inplace=True
    )

    structure_data_list = []
    for _, row in structure_data.iterrows():
        structure_data_list.append(
            {
                "id": row["id"],
                "rgb_triplet": row["rgb_triplet"],
                # "parent_structure_id": row["parent_structure_id"],
                "name": row["name"],
                "structure_id_path": row["structure_id_path"],
                "acronym": row["acronym"],
            }
        )

    tree = get_structures_tree(structure_data_list)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )
    print(tree)
    print(f"Saving atlas data at {atlas_path}")

    # use tifffile to read annotated file
    annotated_volume = load_nii(annotations_file, as_array=True).astype(
        np.uint8
    )

    # remove unconnected components
    label_im, nb_labels = ndimage.label(
        annotated_volume
    )  # not to be confused with our labels
    sizes = ndimage.sum(annotated_volume > 0, label_im, range(nb_labels + 1))
    mask = sizes > 1000
    annotated_volume *= mask[label_im]

    # naive forcing symmetry
    extent_LR = annotated_volume.shape[2]
    half_image = extent_LR // 2

    flipped_first_half = np.flip(
        annotated_volume[:, :, 0:half_image], axis=2
    ).copy()
    flipped_second_half = np.flip(
        annotated_volume[:, :, half_image - 1 : -1], axis=2
    ).copy()

    annotated_volume[:, :, 0:half_image] = np.minimum(
        annotated_volume[:, :, 0:half_image], flipped_second_half
    )
    annotated_volume[:, :, half_image - 1 : -1] = np.minimum(
        annotated_volume[:, :, half_image - 1 : -1], flipped_first_half
    )

    # smooth annotations
    annotated_volume = median(
        annotated_volume, ball(3), mask=annotated_volume > 0
    )

    # keep only annotations in mask
    brain_mask = load_nii(reference_mask_file, as_array=True).astype(np.uint16)
    annotated_volume *= brain_mask

    # rescale reference volume into int16 range
    reference_volume = load_nii(reference_file, as_array=True)
    dmin = np.min(reference_volume)
    dmax = np.max(reference_volume)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference_volume = (reference_volume - dmin) * dscale
    reference_volume = reference_volume.astype(np.uint16)

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # mesh creation
    closing_n_iters = 2
    decimate_fraction = 0.3
    smooth = True

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
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structure_data_list:
        # check if a mesh was created
        mesh_path = meshes_dir_path / f"{s['id']}.obj"
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it.")
            continue
        else:
            # check that the mesh actually exists and isn't empty
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue
        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=resolution,
        orientation=ORIENTATION,
        root_id=999,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structure_data_list,
        meshes_dict=meshes_dict,
        scale_meshes=True,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
    )

    return output_filename


if __name__ == "__main__":
    res = 50, 50, 50
    home = str(Path.home())
    bg_root_dir = Path.home() / "bg-atlasgen"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)
