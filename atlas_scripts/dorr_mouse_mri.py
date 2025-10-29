"""Atlas generation script for the Dorr MRI mouse atlas."""

__version__ = "0"

import random
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import tifffile

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

ACRONYM_MAP = {
    "amygdala": "A",
    "anterior commissure: pars anterior": "aco",
    "anterior commissure: pars posterior": "act",
    "arbor vita of cerebellum": "arb",
    "basal forebrain": "fb",
    "bed nucleus of stria terminalis": "BST",
    "cerebellar cortex": "CBX",
    "cerebellar peduncle: inferior": "icp",
    "cerebellar peduncle: middle": "mcp",
    "cerebellar peduncle: superior": "dscp",
    "cerebral aqueduct": "AQ",
    "cerebral cortex: entorhinal cortex": "CTXe",
    "cerebral cortex: frontal lobe": "CTXf",
    "cerebral cortex: occipital lobe": "CTXo",
    "cerebral cortex: parieto-temporal lobe": "CTXp",
    "cerebral peduncle": "cpd",
    "colliculus: inferior": "IC",
    "colliculus: superior": "SC",
    "corpus callosum": "cc",
    "corticospinal tract/pyramids": "cst",
    "cuneate nucleus": "CU",
    "dentate gyrus of hippocampus": "DG",
    "facial nerve (cranial nerve 7)": "VIIn",
    "fasciculus retroflexus": "fr",
    "fimbria": "fi",
    "fornix": "fxs",
    "fourth ventricle": "V4",
    "fundus of striatum": "FS",
    "globus pallidus": "GP",
    "habenular commissure": "hbc",
    "hippocampus": "HIP",
    "hypothalamus": "HY",
    "inferior olivary complex": "IO",
    "internal capsule": "int",
    "interpedunclar nucleus": "IPN",
    "lateral olfactory tract": "lotg",
    "lateral septum": "LS",
    "lateral ventricle": "VL",
    "mammillary bodies": "MBO",
    "mammilothalamic tract": "mtt",
    "medial lemniscus/medial longitudinal fasciculus": "ml",
    "medial septum": "MS",
    "medulla": "MY",
    "midbrain": "MB",
    "nucleus accumbens": "ACB",
    "olfactory bulbs": "OB",
    "olfactory tubercle": "OT",
    "optic tract": "opt",
    "periaqueductal grey": "PAG",
    "pons": "P",
    "pontine nucleus": "PRN",
    "posterior commissure": "pc",
    "pre-para subiculum": "SUB",
    "stratum granulosum of hippocampus": "DGsg",
    "stria medullaris": "sm",
    "stria terminalis": "st",
    "striatum": "STR",
    "subependymale zone / rhinocele": "SEZ",
    "superior olivary complex": "SOC",
    "thalamus": "TH",
    "third ventricle": "V3",
    "ventral tegmental decussation": "vtd",
}


def create_atlas(working_dir):
    """Create the Dorr Mouse MRI BrainGlobe atlas.

    Parameters
    ----------
    working_dir : Path
        Directory where intermediate and output files are stored.

    Returns
    -------
    Path
        Path to the generated BrainGlobe atlas archive.
    """
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

    temp_dir = working_dir / "dorr-temp"
    temp_dir.mkdir(exist_ok=True)

    def get_reference_and_annotations(root_dir=temp_dir):
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

    def generate_brainglobe_structures(root_id=1, root_dir=temp_dir):
        csv_url = f"{ATLAS_LINK}/c57_brain_atlas_labels.csv"
        csv_path = root_dir / "c57_brain_atlas_labels.csv"
        utils.retrieve_over_http(csv_url, csv_path)

        df = pd.read_csv(csv_path)

        original_structures = [
            {
                "acronym": "root",
                "id": root_id,
                "name": "brain",
                "structure_id_path": [root_id],
                "rgb_triplet": [255, 255, 255],
            }
        ]

        unified_structures = [
            {
                "acronym": "root",
                "id": root_id,
                "name": "brain",
                "structure_id_path": [root_id],
                "rgb_triplet": [255, 255, 255],
            }
        ]

        used_ids = set([root_id])
        unified_used_ids = set([root_id])
        hemisphere_pairs = {}

        for _, row in df.iterrows():
            name = row["Structure"].strip()
            left_id = int(row["left label"])
            right_id = int(row["right label"])

            # Use provided acronym map
            acronym = ACRONYM_MAP.get(name.lower(), None)
            if acronym is None:
                # fallback for unmapped structures
                acronym = "".join([w[0].upper() for w in name.split()])[:6]

            random.seed(name)
            color = [random.randint(0, 255) for _ in range(3)]

            # Original structures (keep left/right separate)
            if left_id == right_id:
                if left_id not in used_ids:
                    original_structures.append(
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
                    original_structures.append(
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
                    original_structures.append(
                        {
                            "acronym": acronym + "_L",
                            "id": left_id,
                            "name": name + " left",
                            "structure_id_path": [root_id, left_id],
                            "rgb_triplet": color,
                        }
                    )
                    used_ids.add(left_id)

                # Record hemisphere mapping
                hemisphere_pairs[right_id] = left_id

            # Unified structures (merge L/R)
            canonical_id = left_id  # Use left hemisphere ID
            if canonical_id not in unified_used_ids:
                unified_structures.append(
                    {
                        "acronym": acronym,
                        "id": canonical_id,
                        "name": name,
                        "structure_id_path": [root_id, canonical_id],
                        "rgb_triplet": color,
                    }
                )
                unified_used_ids.add(canonical_id)

        return original_structures, unified_structures, hemisphere_pairs

    reference, annotations = get_reference_and_annotations()
    original_structures, unified_structures, hemisphere_pairs = (
        generate_brainglobe_structures()
    )

    # Remap Annotation Labels to Match Structure IDS
    print("Remapping annotation labels to structure IDs...")

    unique_ann = np.unique(annotations)
    unique_ann_sorted = np.sort(unique_ann[unique_ann != 0])

    structure_ids = [
        s["id"] for s in original_structures if s["id"] != ROOT_ID
    ]
    structure_ids_sorted = np.sort(structure_ids)

    if len(unique_ann_sorted) != len(structure_ids_sorted):
        raise ValueError(
            f"Mismatch: {len(unique_ann_sorted)} annotation labels vs "
            f"{len(structure_ids_sorted)} structure IDs."
        )

    label_mapping = dict(zip(unique_ann_sorted, structure_ids_sorted))

    remapped_annotations = np.zeros_like(annotations, dtype=np.int32)
    for old_id, new_id in label_mapping.items():
        remapped_annotations[annotations == old_id] = new_id

    print("Remapping complete.")
    print(
        "Unique annotation IDs after remap:",
        len(np.unique(remapped_annotations)),
    )

    # Unify Left and Right Hemisphere Labels
    print("Unifying left and right hemisphere labels...")

    unified_annotations = remapped_annotations.copy()
    for right_id, left_id in hemisphere_pairs.items():
        if right_id in np.unique(unified_annotations):
            unified_annotations[unified_annotations == right_id] = left_id

    print("Hemisphere unification complete.")
    print(
        "Unique annotation IDs after unification:",
        len(np.unique(unified_annotations)),
    )

    # ---------------------------------------------------------------------- #

    # Mesh creation parameters
    closing_n_iters = 10
    decimate_fraction = 0.6
    smooth = True

    start = time.time()

    meshes_dict = construct_meshes_from_annotation(
        working_dir,
        unified_annotations,
        unified_structures,
        closing_n_iters,
        decimate_fraction,
        smooth,
    )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
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
        annotation_stack=unified_annotations,
        structures_list=unified_structures,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )

    # Clean up temporary directory
    shutil.rmtree(temp_dir, ignore_errors=True)

    return output_filename


if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
