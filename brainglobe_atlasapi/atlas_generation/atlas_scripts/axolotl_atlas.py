__version__ = "0"

import csv
import glob as glob
import time
from pathlib import Path

import numpy as np
import pooch
import tifffile
from rich.progress import track

from brainglobe_atlasapi import utils

# from skimage import io
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


from brainglobe_utils.IO.image.load import load_any



def create_atlas(working_dir, resolution):
    ATLAS_NAME = "axolotl"
    SPECIES = "Ambystoma mexicanum"
    ATLAS_LINK = "https://www.nature.com/articles/s41598-021-89357-3#citeas" 
    CITATION = "Lazcano, I. et al. 2021, https://doi.org/10.1038/s41598-021-89357-3"
    ATLAS_FILE_URL = "https://zenodo.org/records/4595016" 
    ORIENTATION = "ras" 
    ROOT_ID_LEFT = 42  #FIXME
    ROOT_ID_RIGHT = 92 #FIXME
    ATLAS_PACKAGER = "Name, name@gmail.com" #FIXME
    ADDITIONAL_METADATA = {} 

    # setup folder for downloading

    working_dir = Path(working_dir)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(parents=True, exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"
    atlas_path.mkdir(exist_ok=True)

        # download atlas files
    utils.check_internet_connection()
    pooch.retrieve(url="https://zenodo.org/records/4595016",
                known_hash=None,#FIXME 
                progressbar=True  
                ) 

    structures_file = working_dir / "axolotl_label_names_66rois(2).csv"
    annotations_file = working_dir / "axolotl_labels_66rois_40micra(2).nii.gz"
    reference_file = working_dir / "axolotl_template_40micra(3).nii.gz"

    #meshes_dir_path = atlas_path / "asty_atlas/meshes"
    # additional references (not in remote):
    #reference_cartpt = atlas_path / "asty_atlas/SPF2_cartpt_ref.tif"

    #Path(meshes_dir_path).mkdir(exist_ok=True)

    # create dictionaries # create dictionary from data read from the CSV file 
    print("Creating structure tree")
    with open(
        structures_file, mode="r", encoding='utf-8',
        ) as axolotl_file:
        axolotl_dict_reader = csv.DictReader(axolotl_file) 
        hierarchy = []
        for row in axolotl_dict_reader:
            hierarchy.append(row)    
    print(hierarchy)

    # Replace the 'label_id' key with 'id' key 
    # Define the file paths
    input_file = working_dir / "axolotl_label_names_66rois(2).csv"
    output_file = working_dir / "axolotl_label_names_66rois(v2).csv"

    # Read the CSV file and replace the key
    modified_rows = []
    with open(input_file, mode='r', encoding='utf-8-sig') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            if 'label_id' in row:
                row['id'] = row.pop('label_id')  # Replace 'label_id' with 'id'
            modified_rows.append(row)

    # Get the fieldnames from the modified rows
    fieldnames = modified_rows[0].keys()

    hierarchy = modified_rows 

    # clean out different columns 
    for element in hierarchy:
        element["id"] = int(element["id"])
        element["main_structure"]=element["main_structure"].strip()
    

    main_structures = set()

    # with open(input_file, mode='r', newline='') as infile:
    #     reader = csv.DictReader(infile)
    for element in hierarchy:
        main_structure = element['main_structure']
        main_structures.add(main_structure)

    # Assign unique numeric IDs to each main structure
    structure_id_map = {structure: idx + 1 for idx, structure in enumerate(main_structures)}

    # Print the mapping to verify
    print(structure_id_map)

    # Define the root ID
    root_id = 999

    # Function to create the structure_id_path
    def create_structure_id_path(main_structure):
        structure_id = structure_id_map[main_structure]
        return [root_id, structure_id]

    for main_structure in main_structures:
        path = create_structure_id_path(main_structure)
        print(f"Main Structure: {main_structure}, Path: {path}")

    for element in hierarchy:
        structure_id_path = create_structure_id_path(element["main_structure"])
                
        element['structure_id_path'] = structure_id_path

    output_file = working_dir/ 'updated_axolotl.csv' #FIXME

    with open(output_file, mode='w', newline='') as outfile:
        fieldnames = hierarchy[0].keys()
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for element in hierarchy:
            writer.writerow(element)

    print("CSV updated successfully!")

    # TODO Create meshes 

    # Set root mesh to white
    hierarchy[0]["rgb_triplet"] = [255, 255, 255]
    # NOTE Reviewed till here #
    # use tifffile to read annotated file
    annotated_volume = tifffile.imread(annotations_file).astype(np.uint8)
    reference_volume = tifffile.imread(reference_file)

    # additional reference
    cartpt_volume = tifffile.imread(reference_cartpt)
    cartpt_volume -= np.min(
        cartpt_volume
    )  # shift cartpt to a non-negative range before converting to UINT16
    cartpt_volume = cartpt_volume.astype(np.uint16)
    ADDITIONAL_REFERENCES = {"cartpt": cartpt_volume}

    print(f"Saving atlas data at {atlas_path}")
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
        structures_list=hierarchy,
        meshes_dict=meshes_dict,
        scale_meshes=True,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_metadata=ADDITIONAL_METADATA,
        additional_references=ADDITIONAL_REFERENCES,
    )

    return output_filename


if __name__ == "__main__":
    res = 2, 2, 2
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)
