"""Package the BrainGlobe atlas for the Eurasian Blackcap."""

__version__ = "5"

import csv
import time
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    apply_modal_filter,
    read_itk_labels,
)
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
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

    mc_annotations_file = materials_folder / "annotations_MC.nii"

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

    # hackily add MC (region 61)to the structure data list
    structure_data_list.append(
        {
            "id": 61,
            "name": "Caudal Mesopallium",
            "acronym": "MC",
            "structure_id_path": [999, 1, 60, 61],
            "rgb_triplet": [255, 0, 4],
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

    # hackily add MC (region 61) to the annotated volume
    mc_annotations = load_nii(mc_annotations_file, as_array=True).astype(
        np.uint16
    )
    mirrored_mc_annotations = np.flip(mc_annotations, axis=2)
    mc_annotations = np.concatenate(
        (mc_annotations, mirrored_mc_annotations), axis=2
    )

    # region 61 is a child of region 60, so only overwrite pixels with value 60
    mc_annotations = apply_modal_filter(mc_annotations)
    annotated_volume[(mc_annotations == 61) & (annotated_volume == 60)] = 61
    assert np.any(annotated_volume == 61)

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

    # mesh creation
    closing_n_iters = 1
    decimate_fraction = 0.6  # higher = more triangles
    smooth = False

    start = time.time()

    meshes_dir_path = Path.home() / "blackcap-meshes"
    meshes_dict = construct_meshes_from_annotation(
        meshes_dir_path,
        annotated_volume,
        structure_data_list,
        closing_n_iters=closing_n_iters,
        decimate_fraction=decimate_fraction,
        smooth=smooth,
    )

    print(
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
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
