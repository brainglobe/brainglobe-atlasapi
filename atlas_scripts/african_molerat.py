"""Package the BrainGlobe atlas for the African Molerat."""

__version__ = "1"

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import (
    get_structures_tree,
)


def create_atlas(working_dir, resolution):
    """
    Package the African Molerat BrainGlobe atlas.

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
    ATLAS_NAME = "african_molerat"
    SPECIES = "Fukomys anselli"
    ATLAS_LINK = "https://www.malkemper-lab.com/"
    CITATION = "unpublished"
    ATLAS_FILE_URL = "https://gin.g-node.org/BrainGlobe/molerat_materials/raw/master/mole-rat_atlas_20251023.zip"
    ORIENTATION = "asr"
    ROOT_ID = 999
    ATLAS_PACKAGER = "BrainGlobe Developers, hello@brainglobe.info"
    ADDITIONAL_METADATA = {}

    atlas_path = pooch.retrieve(
        ATLAS_FILE_URL,
        known_hash=None,
        processor=pooch.Unzip(),
        progressbar=True,
    )

    materials_directory = Path(atlas_path[0]).parent
    hierarchy_path = materials_directory / "annotation_table_20251023.xlsx"

    reference_file = (
        materials_directory / "Reference_mole-rat_brain_fullmap.tif"
    )

    annotations_file = (
        materials_directory / "anotation_latest_cleaned_fullmap.tif"
    )

    print("Reading structures files")
    df = pd.read_excel(hierarchy_path, engine="openpyxl")

    # Replace missing parent IDs with root
    df["parent id"] = df["parent id"].fillna(999).astype(int)

    # Build the list of dictionaries
    structure_data_list = [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "acronym": row["acronym"],
            "rgb_triplet": [int(row["R"]), int(row["G"]), int(row["B"])],
            "structure_id_path": [int(row["parent id"]), int(row["id"])],
        }
        for _, row in df.iterrows()
    ]

    # append root which doesn't have its own voxels and
    # therefore not in itk file
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
    annotated_volume = load_any(annotations_file).astype(np.uint16)

    # rescale reference volume into int16 range
    reference_volume = load_any(reference_file).astype(np.uint16)

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # mesh creation
    closing_n_iters = 1
    decimate_fraction = 0.6  # higher = more triangles
    smooth = False

    start = time.time()

    construct_meshes_from_annotation(
        working_dir,
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

    meshes_dir_path = working_dir / "meshes"
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
    res = 20, 20, 20
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)
