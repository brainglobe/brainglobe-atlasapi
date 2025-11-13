"""Package the BrainGlobe atlas for the Eurasian Blackcap."""

__version__ = "4"

import csv
import os
import time
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    read_itk_labels,
)
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_meshes_from_annotated_volume,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import (
    get_structures_tree,
)


def create_atlas(working_dir, resolution):
    """
    Package the Eurasian Blackcap BrainGlobe atlas.

    This function downloads necessary data, processes anatomical structures,
    creates reference and annotation volumes, generates 3D meshes for
    brain regions, and packages them into a BrainGlobe atlas.

    Parameters
    ----------
    working_dir : Path
        The directory where temporary files and the final atlas will be saved.
    resolution : tuple
        The resolution of the atlas volumes in microns (x, y, z).

    Returns
    -------
    str
        The path to the generated atlas file.
    """
    ATLAS_NAME = "eurasian_blackcap"
    SPECIES = "Sylvia atricapilla"
    ATLAS_LINK = "https://uol.de/en/ibu/animal-navigation"
    CITATION = "unpublished"
    ATLAS_FILE_URL = "https://uol.de/en/ibu/animal-navigation"  # noqa
    ORIENTATION = "asr"
    ROOT_ID = 999
    ATLAS_PACKAGER = "BrainGlobe Developers, hello@brainglobe.info"
    ADDITIONAL_METADATA = {}

    gin_url = "https://gin.g-node.org/BrainGlobe/blackcap_materials/raw/master/blackcap_materials.zip"
    atlas_materials_paths = pooch.retrieve(
        gin_url, known_hash=None, processor=pooch.Unzip(), progressbar=True
    )

    materials_folder = Path(atlas_materials_paths[0]).parent
    hierarchy_path = materials_folder / "combined_structures_update_0825.csv"
    reference_file = materials_folder / "blackcap_male_template.nii.gz"
    structures_file = (
        materials_folder / "merged_unique_labels_color-adjusted.txt"
    )
    annotations_file = (
        materials_folder / "blackcap_male_smoothed_annotations.nii.gz"
    )
    meshes_dir_path = Path.home() / "blackcap-meshes"

    try:
        os.mkdir(meshes_dir_path)
    except FileExistsError:
        "mesh folder already exists"

    print("Reading structures files")
    structure_to_parent_map = {}
    with open(hierarchy_path, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            structure_to_parent_map[int(row[1])] = [
                int(parent) for parent in row[2].split(",")
            ]

    structure_data_list = read_itk_labels(structures_file)
    for structure in structure_data_list:
        structure_id = structure["id"]
        structure["structure_id_path"] = structure_to_parent_map[structure_id]
        structure["rgb_triplet"] = list(structure["rgb_triplet"])

    # append root and pallium structures, which don't have their own voxels
    # and are therefore not in itk file
    structure_data_list.append(
        {
            "id": 1,
            "name": "Pallium",
            "acronym": "P",
            "structure_id_path": [999, 1],
            "rgb_triplet": [0, 200, 100],
        }
    )
    structure_data_list.append(
        {
            "id": 999,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    )

    tree = get_structures_tree(structure_data_list)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )
    print(tree)

    # use tifffile to read annotated file
    annotated_volume = load_nii(annotations_file, as_array=True).astype(
        np.uint16
    )

    # rescale reference volume into int16 range
    reference_volume = load_nii(reference_file, as_array=True)
    dmin = np.min(reference_volume)
    dmax = np.max(reference_volume)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference_volume = (reference_volume - dmin) * dscale
    reference_volume = reference_volume.astype(np.uint16)

    has_label = annotated_volume > 0
    reference_volume *= has_label

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)
    for key, node in tree.nodes.items():
        if (
            key in labels or key == 1
        ):  # Pallium == 1 needs mesh but has no own voxels
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # mesh creation
    closing_n_iters = 1
    decimate_fraction = 0.6  # higher = more triangles
    smooth = False

    start = time.time()

    create_meshes_from_annotated_volume(
        meshes_dir_path,
        tree,
        annotated_volume,
        closing_n_iters=closing_n_iters,
        decimate_fraction=decimate_fraction,
        smooth=smooth,
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
        root_id=ROOT_ID,
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
    res = 25, 25, 25
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir, res)
