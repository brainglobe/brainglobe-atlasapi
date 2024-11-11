__version__ = "0"

import csv
import time
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_nii
from rich.progress import track
from skimage.filters.rank import modal
from skimage.measure import label, regionprops
from skimage.morphology import ball

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "unam_axolotl"
SPECIES = "Ambystoma mexicanum"
ATLAS_LINK = "https://zenodo.org/records/4595016"
CITATION = (
    "Lazcano, I. et al. 2021, https://doi.org/10.1038/s41598-021-89357-3"
)
ORIENTATION = "lpi"
ROOT_ID = 999
ATLAS_PACKAGER = "Saima Abdus, David Perez-Suarez, Alessandro Felder"
ADDITIONAL_METADATA = {}
RESOLUTION = 40, 40, 40  # Resolution tuple


def create_atlas(working_dir, resolution):

    # setup folder for downloading
    working_dir = Path(working_dir)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(parents=True, exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"
    atlas_path.mkdir(exist_ok=True)

    # download atlas files
    utils.check_internet_connection()
    hashes = [
        "md5:3a9ba5a23c17180981b5678329915226",
        "md5:66df0da5d7eed10ff59add32741d0bf2",
    ]
    list_files = {
        "axolotl_labels_66rois_40micra.nii.gz": hashes[0],
        "axolotl_template_40micra.nii.gz": hashes[1],
    }

    for filename, hash in list_files.items():
        pooch.retrieve(
            url=f"{ATLAS_LINK}/files/{filename}",
            known_hash=hash,
            path=atlas_path,
            progressbar=True,
            processor=pooch.Decompress(name=filename[:-3]),
        )

    # download structure file
    pooch.retrieve(
        url=f"{ATLAS_LINK}/files/axolotl_label_names_66rois.csv",
        known_hash="md5:ab13eb8b8f9324a67fdd162f4e79f3c0",
        path=atlas_path,
        progressbar=True,
        fname="axolotl_label_names_66rois.csv",
    )

    structures_file = atlas_path / "axolotl_label_names_66rois.csv"
    annotations_file = atlas_path / "axolotl_labels_66rois_40micra.nii"
    reference_file = atlas_path / "axolotl_template_40micra.nii"

    annotation_image = load_nii(annotations_file, as_array=True)
    reference_image = load_nii(reference_file, as_array=True)
    dmin = np.min(reference_image)
    dmax = np.max(reference_image)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference_image = (reference_image - dmin) * dscale
    reference_image = reference_image.astype(np.uint16)

    # mask: put 1 where there is an annotation
    annotation_mask = np.zeros(annotation_image.shape)
    annotation_mask[annotation_image > 0] = 1

    # find the connected regions in the mask
    labeled_image = label(annotation_mask)
    regions = regionprops(labeled_image)

    # find the pixels belonging to the largest region
    largest_region = max(regions, key=lambda region: region.area)
    largest_mask = labeled_image == largest_region.label

    # keep only annotations in the largest connected region
    annotation_image = annotation_image * largest_mask

    hierarchy = []

    # create dictionaries # create dictionary from data read from the CSV file
    print("Creating structure tree")
    with open(
        structures_file,
        mode="r",
        encoding="utf-8",
    ) as axolotl_file:
        axolotl_dict_reader = csv.DictReader(axolotl_file)

        for row in axolotl_dict_reader:
            if "label_id" in row:
                row["id"] = row.pop("label_id")
                row["acronym"] = row.pop("Abbreviation/reference")
                row["name"] = row.pop("label_name")
                row["rgb_triplet"] = [255, 0, 0]
                row.pop("hemisphere")
                row.pop("voxels")
                row.pop("volume")
            hierarchy.append(row)

    # clean out different columns
    for element in hierarchy:
        element["id"] = int(element["id"])
        element["main_structure"] = element["main_structure"].strip()

    main_structures = set()

    for element in hierarchy:
        main_structure = element["main_structure"]
        main_structures.add(main_structure)

    # Assign unique numeric IDs to each main structure
    structure_id_map = {
        structure: idx + 1
        for idx, structure in enumerate(main_structures, start=100)
    }

    # Function to create the structure_id_path
    def create_structure_id_path(main_structure):
        structure_id = structure_id_map[main_structure]
        return [ROOT_ID, structure_id]

    for main_structure in main_structures:
        path = create_structure_id_path(main_structure)
        print(f"Main Structure: {main_structure}, Path: {path}")

    for element in hierarchy:
        structure_id_path = create_structure_id_path(element["main_structure"])

        element["structure_id_path"] = structure_id_path + [element["id"]]

    for main_structure, id_main_structure in structure_id_map.items():
        main_structure_acronym = "".join(
            [word[0].upper() for word in main_structure.split()]
        )
        create_main_structure = {
            "name": main_structure,
            "acronym": main_structure_acronym,
            "id": id_main_structure,
            "rgb_triplet": [125, 0, 125],
            "structure_id_path": [ROOT_ID, id_main_structure],
        }
        hierarchy.append(create_main_structure)

    for row in hierarchy:
        if "main_structure" in row.keys():
            row.pop("main_structure")

    root_dict = {
        "name": "root",
        "acronym": "root",
        "id": ROOT_ID,
        "rgb_triplet": [255, 255, 255],
        "structure_id_path": [999],
    }
    hierarchy.append(root_dict)

    tree = get_structures_tree(hierarchy)

    # Generate binary mask for mesh creation
    labels = np.unique(annotation_image).astype(np.int_)
    for key, node in tree.nodes.items():
        # Check if the node's key is in the list of labels
        is_label = key in labels
        node.data = Region(is_label)

    # Mesh creation parameters
    closing_n_iters = 10
    decimate_fraction = 0.6
    smooth = True

    meshes_dir_path = working_dir / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    # pass a smoothed version of the annotations for meshing
    smoothed_annotations = annotation_image.copy()
    smoothed_annotations = modal(
        smoothed_annotations.astype(np.uint8), ball(5)
    )

    # Measure duration of mesh creation
    start = time.time()

    # Iterate over each node in the tree and create meshes
    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        description="Creating meshes",
    ):

        create_region_mesh(
            [
                meshes_dir_path,  # Directory where mesh files will be saved
                node,
                tree,
                labels,
                smoothed_annotations,
                ROOT_ID,
                closing_n_iters,
                decimate_fraction,
                smooth,
            ]
        )

    # Print the duration of mesh extraction
    print(
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Create a dictionary to store mappings of structure IDs to mesh file paths
    meshes_dict = {}
    structures_with_mesh = []

    for s in hierarchy:
        # Construct the path to the mesh file using the structure ID
        mesh_path = meshes_dir_path / f"{s['id']}.obj"

        # Check if the mesh file exists
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it.")
            continue

        # Check that the mesh actually exists and isn't empty
        if mesh_path.stat().st_size < 512:
            print(f"OBJ file for {s} is too small, ignoring it.")
            continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    # Print the total number of structures that have valid meshes
    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    # Package all the provided data and parameters into an atlas format
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=resolution,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_image,
        annotation_stack=annotation_image,
        structures_list=hierarchy,
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
    home = str(Path.home())
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, RESOLUTION)
