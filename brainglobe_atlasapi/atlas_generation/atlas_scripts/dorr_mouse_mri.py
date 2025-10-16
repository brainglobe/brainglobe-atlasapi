__version__ = "0"

import random
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile
from rich.progress import track
from skimage.filters.rank import modal
from skimage.morphology import ball

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def create_atlas(working_dir):
    ATLAS_NAME = "dorr_mouse_mri"
    SPECIES = "Mus musculus"
    ATLAS_LINK = (
        "https://www.mouseimaging.ca/technologies/mouse_atlas/"
        "C57Bl6j_mouse_atlas"
    )
    CITATION = (
        "A.E. Dorr et al. 2008, https://doi.org/10.1016/"
        "j.neuroimage.2008.03.037"
    )
    ORIENTATION = "ipl"
    ROOT_ID = 1
    ANNOTATIONS_RES_UM = 32

    ### Need to change gin link ###
    # ATLAS_FILE_URL = "https://gin.g-node.org/brainglobe/kim_atlas_materials/raw/master/kim_atlas_materials.tar.gz"
    ###############################

    def get_reference_and_annotations(root_dir=working_dir):
        urls = {
            "male-female-mouse-atlas.mnc": "https://www.mouseimaging.ca/mnc/C57Bl6j_mouse_atlas/"
            "male-female-mouse-atlas.mnc",
            "c57_fixed_labels_resized.mnc": "https://www.mouseimaging.ca/mnc/C57Bl6j_mouse_atlas/"
            "c57_fixed_labels_resized.mnc",
        }

        outputs = {}

        for name, url in urls.items():
            mnc_path = root_dir / name
            utils.retrieve_over_http(url, mnc_path)

            # Load MNC
            img = nib.load(str(mnc_path))
            data = img.get_fdata()

            if data.dtype.kind == "f":
                data_min, data_max = data.min(), data.max()
                if data_max == data_min:
                    scaled = (data * 0).astype("uint16")
                else:
                    scaled = (
                        (data - data_min) / (data_max - data_min) * 65535
                    ).astype("uint16")
                data = scaled

            # Save as TIFF
            tiff_path = root_dir / f"{mnc_path.stem}.tiff"
            tifffile.imwrite(tiff_path, data)

            # Read TIFF
            arr = tifffile.imread(tiff_path)
            outputs[name] = arr

        return (
            outputs["male-female-mouse-atlas.mnc"],
            outputs["c57_fixed_labels_resized.mnc"],
        )

    def generate_brainglobe_structures(root_id=1, root_dir=working_dir):
        csv_url = f"{ATLAS_LINK}/c57_brain_atlas_labels.csv"
        csv_path = root_dir / "c57_brain_atlas_labels.csv"
        utils.retrieve_over_http(csv_url, csv_path)

        df = pd.read_csv(csv_path)

        structures = [
            {
                "acronym": "root",
                "id": root_id,
                "name": "brain",
                "structure_id_path": [root_id],
                "rgb_triplet": [255, 255, 255],
            }
        ]

        used_ids = set([root_id])

        for _, row in df.iterrows():
            name = row["Structure"].strip()
            left_id = int(row["left label"])
            right_id = int(row["right label"])

            # Generate acronym from first letters of up to first 3 words
            acronym = "".join([w[0].upper() for w in name.split()[:3]])[:5]

            random.seed(name)
            color = [random.randint(0, 255) for _ in range(3)]

            # Merge if IDs are the same
            if left_id == right_id:
                if left_id not in used_ids:
                    structures.append(
                        {
                            "acronym": acronym,
                            "id": left_id,
                            "name": name,
                            "structure_id_path": [root_id, left_id],
                            "rgb_triplet": color,
                        }
                    )
                    used_ids.add(left_id)
            else:
                # Right hemisphere
                if right_id not in used_ids:
                    structures.append(
                        {
                            "acronym": acronym + "_R",
                            "id": right_id,
                            "name": name + " right",
                            "structure_id_path": [root_id, right_id],
                            "rgb_triplet": color,
                        }
                    )
                    used_ids.add(right_id)

                # Left hemisphere
                if left_id not in used_ids:
                    structures.append(
                        {
                            "acronym": acronym + "_L",
                            "id": left_id,
                            "name": name + " left",
                            "structure_id_path": [root_id, left_id],
                            "rgb_triplet": color,
                        }
                    )
                    used_ids.add(left_id)

        return structures

    reference, annotations = get_reference_and_annotations()
    structures = generate_brainglobe_structures()

    # ----------- REMAP ANNOTATION LABELS TO MATCH STRUCTURE IDS ----------- #
    unique_ann = np.unique(annotations)
    structure_ids = [s["id"] for s in structures if s["id"] != ROOT_ID]

    unique_ann_sorted = np.sort(unique_ann[unique_ann != 0])
    structure_ids_sorted = np.sort(structure_ids)

    if len(unique_ann_sorted) != len(structure_ids_sorted):
        raise ValueError(
            f"Mismatch: {len(unique_ann_sorted)} annotation labels vs "
            f"{len(structure_ids_sorted)} structure IDs."
        )

    print(
        f"Mapping {len(unique_ann_sorted)} "
        f"annotation labels to structure IDs..."
    )

    label_mapping = dict(zip(unique_ann_sorted, structure_ids_sorted))

    remapped_annotations = np.zeros_like(annotations, dtype=np.int32)
    for old, new in label_mapping.items():
        remapped_annotations[annotations == old] = new

    annotations = remapped_annotations

    print("Remapping complete.")
    print("Unique annotation IDs after remap:", np.unique(annotations))

    # ---------------------------------------------------------------------- #

    tree = get_structures_tree(structures)

    # Generate binary mask for mesh creation
    labels = np.unique(annotations).astype(np.int_)
    for key, node in tree.nodes.items():
        is_label = key in labels
        node.data = Region(is_label)

    # Mesh creation parameters
    closing_n_iters = 10
    decimate_fraction = 0.6
    smooth = True

    meshes_dir_path = working_dir / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    # pass a smoothed version of the annotations for meshing
    smoothed_annotations = annotations.copy()
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
                meshes_dir_path,
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

    print(
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Create a dictionary to store mappings of structure IDs to mesh file paths
    meshes_dict = {}
    structures_with_mesh = []

    for s in structures:
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
        resolution=(ANNOTATIONS_RES_UM,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference,
        annotation_stack=annotations,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )

    return output_filename


if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
