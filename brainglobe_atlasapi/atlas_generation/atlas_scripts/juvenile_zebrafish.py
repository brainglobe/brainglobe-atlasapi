import time
from pathlib import Path

import numpy as np
import pandas as pd
from brainglobe_utils.IO.image import load_any, save_any
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "dreosti_juvenile_zebrafish"

# DOI of the most relevant citable document
CITATION = "unpublished"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Danio rerio"

# The URL for the data files
ATLAS_LINK = None

# The orientation of the **original** atlas data, in BrainGlobe convention:
ORIENTATION = "sal"

# The id of the highest level of the atlas.
ROOT_ID = 999

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 1


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """
    pass


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    atlas_path = Path(
        "/media/ceph/neuroinformatics/neuroinformatics/atlas-forge/JuvenileZebrafish/"
    )
    reference_image_path = atlas_path / "DAPI.tif"
    reference = load_any(reference_image_path)

    mask_path = atlas_path / "masks"
    annotation = np.zeros_like(reference, dtype=np.uint16)
    annotation[reference > 0] = ROOT_ID

    region_names_path = atlas_path / "juvenile_zebrafish_atlas_regions.csv"
    region_names_df = pd.read_csv(region_names_path)

    for _, row in region_names_df.iterrows():
        id = row["id"]
        filename = row["filename"]
        mask_file = mask_path / filename
        mask = load_any(mask_file)
        print(mask_file)
        annotation[mask > 0] = id

    save_any(annotation, Path.home() / "juvenile_zebrafish_annotation.tif")
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    This function should return a pandas DataFrame with information about your
    atlas.

    The DataFrame should be in the following format:

    ╭────┬───────────────────┬─────────┬───────────────────┬─────────────────╮
    | id | name              | acronym | structure_id_path | rgb_triplet     |
    |    |                   |         |                   |                 |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997| root              | root    | [997]             | [255, 255, 255] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8  | Basic cell groups | grey    | [997, 8]          | [191, 218, 227] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567| Cerebrum          | CH      | [997, 8, 567]     | [176, 240, 255] |
    ╰────┴───────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    atlas_path = Path(
        "/media/ceph/neuroinformatics/neuroinformatics/atlas-forge/JuvenileZebrafish/"
    )
    structures = []

    region_names_path = atlas_path / "juvenile_zebrafish_atlas_regions.csv"
    region_names_df = pd.read_csv(region_names_path)

    for _, row in region_names_df.iterrows():
        id = row["id"]
        structure = {
            "id": id,
            "acronym": row["acronym"],
            "name": row["region_name"],
            "structure_id_path": [999, id],
            "rgb_triplet": [0, 125, 125],
        }
        structures.append(structure)

    structures.append(
        {
            "id": ROOT_ID,
            "acronym": "root",
            "name": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    )
    return structures


def retrieve_or_construct_meshes(hierarchy, annotated_volume):
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    tree = get_structures_tree(hierarchy)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    meshes_dir_path = DEFAULT_WORKDIR / "tmp_meshes"
    meshes_dir_path.mkdir(exist_ok=True)

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
    for s in hierarchy:
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
    return meshes_dict


def retrieve_additional_references():
    """This function only needs editing if the atlas has additional reference
    images. It should return a dictionary that maps the name of each
    additional reference image to an image stack containing its data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(structures, annotated_volume)

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
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )
