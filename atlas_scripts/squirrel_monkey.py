"""Package the VALiDATe29 Squirrel Monkey Brain atlas.

This module provides functions to download, process, and package the
Squirrel Monkey atlas into the BrainGlobe atlas format. The atlas includes
multiple MRI contrasts and diffusion imaging data from 29 squirrel monkeys.
"""

__version__ = 0

import time

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

### Metadata ###

ATLAS_NAME = "validate29_squirrel_monkey"
SPECIES = "Saimiri sciureus"  # Most common species in the atlas (21/29)
ATLAS_LINK = "https://www.nitrc.org/projects/validate29"
CITATION = "Schilling et al. 2017, https://doi.org/10.1007/s12021-017-9334-0"
ORIENTATION = "ras"  # Right-Anterior-Superior as stated in publication
RESOLUTION = (300, 300, 300)  # 300 microns as specified in issue
ROOT_ID = 9999
DOWNLOAD_URL = "https://www.nitrc.org/frs/download.php/12413/VALiDATe29.zip/?i_agree=1&download_now=1"
ATLAS_PACKAGER = "Eshaan Gupta, eshaan.gupta@gmail.com"


def download_resources(download_dir_path):
    """
    Download the necessary resources for the atlas with Pooch.

    Downloads the VALiDATe29 Squirrel Monkey atlas from NITRC using
    direct download URL.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where the atlas files will be downloaded.

    Returns
    -------
    pathlib.Path
        The path to the extracted atlas files.
    """
    download_dir_path.mkdir(exist_ok=True, parents=True)

    download_name = ATLAS_NAME + "_atlas.zip"
    destination_path = download_dir_path / download_name

    # Download and extract using pooch
    pooch.retrieve(
        url=DOWNLOAD_URL,
        known_hash=None,  # Will be updated after first successful download
        path=destination_path,
        progressbar=True,
        processor=pooch.Unzip(extract_dir=""),
    )

    return destination_path


def retrieve_reference_and_annotation(download_dir_path):
    """
    Retrieve the reference and annotation volumes.

    The VALiDATe29 atlas includes multiple MRI templates and segmentation.
    This function loads the appropriate volumes.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory containing the downloaded atlas files.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    # List all available files to help identify the correct ones
    print("Scanning for atlas files...")
    for file in download_dir_path.rglob("*.nii*"):
        print(f"  Found: {file.relative_to(download_dir_path)}")

    # Common naming patterns for VALiDATe29 atlas
    # Based on NITRC description: T1, T2*, PD templates and labels
    possible_reference_paths = [
        download_dir_path / "VALiDATe29" / "template_T1.nii.gz",
        download_dir_path / "VALiDATe29" / "template_T1.nii",
        download_dir_path / "VALiDATe29" / "T1_template.nii.gz",
        download_dir_path / "VALiDATe29" / "T1_template.nii",
        download_dir_path / "template_T1.nii.gz",
        download_dir_path / "template_T1.nii",
    ]

    reference = None
    for ref_path in possible_reference_paths:
        if ref_path.exists():
            reference = load_any(ref_path, as_numpy=True)
            print(f"Loaded reference from: {ref_path}")
            break

    if reference is None:
        # Try to find any template file
        template_files = list(download_dir_path.rglob("*template*.nii*"))
        if template_files:
            reference = load_any(template_files[0], as_numpy=True)
            print(f"Loaded reference from: {template_files[0]}")
        else:
            raise FileNotFoundError(
                "Could not find reference template. "
                "Please check file structure in downloaded archive."
            )

    # Load annotation/labels
    possible_annotation_paths = [
        download_dir_path / "VALiDATe29" / "labels.nii.gz",
        download_dir_path / "VALiDATe29" / "labels.nii",
        download_dir_path / "VALiDATe29" / "atlas_labels.nii.gz",
        download_dir_path / "VALiDATe29" / "segmentation.nii.gz",
        download_dir_path / "labels.nii.gz",
        download_dir_path / "labels.nii",
    ]

    annotation = None
    for annot_path in possible_annotation_paths:
        if annot_path.exists():
            annotation = load_any(annot_path, as_numpy=True).astype(np.int64)
            print(f"Loaded annotation from: {annot_path}")
            break

    if annotation is None:
        # Try to find any label file
        label_files = list(download_dir_path.rglob("*label*.nii*"))
        if label_files:
            annotation = load_any(label_files[0], as_numpy=True).astype(
                np.int64
            )
            print(f"Loaded annotation from: {label_files[0]}")
        else:
            print(
                "Warning: No annotation file found. Creating basic annotation."
            )
            annotation = np.zeros_like(reference, dtype=np.int64)
            annotation[reference > 0] = 1  # Simple binary mask

    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Returns None to indicate symmetrical atlas, allowing automatic
    hemisphere generation.

    Returns
    -------
    np.ndarray or None
        None, indicating the atlas should be treated as symmetrical.
    """
    return None


def retrieve_structure_information(download_dir_path):
    """
    Return a list of dictionaries with information about brain structures.

    The VALiDATe29 atlas includes 81 labels:
    - 18 cortical gray matter regions
    - 57 white matter tracts
    - 6 ventricle labels

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory containing the downloaded atlas files.

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    # Basic structure list - root + brain
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        },
        {
            "id": 1,
            "name": "Brain",
            "acronym": "Brain",
            "structure_id_path": [ROOT_ID, 1],
            "rgb_triplet": [200, 200, 200],
        },
    ]

    # Look for label definition files
    possible_label_files = (
        list(download_dir_path.rglob("*label*.txt"))
        + list(download_dir_path.rglob("*label*.csv"))
        + list(download_dir_path.rglob("*structure*.txt"))
        + list(download_dir_path.rglob("*structure*.csv"))
        + list(download_dir_path.rglob("*legend*.txt"))
    )

    if possible_label_files:
        file_names = [f.name for f in possible_label_files]
        print(f"Found potential label files: {file_names}")
        # Try to parse the first label file
        label_file = possible_label_files[0]
        print(f"Attempting to parse: {label_file}")

        try:
            import pandas as pd

            # Try different delimiters
            for sep in ["\t", ",", " ", ";"]:
                try:
                    df = pd.read_csv(label_file, sep=sep, comment="#")
                    if len(df.columns) >= 2:  # At least ID and name
                        print(f"Successfully parsed with separator: '{sep}'")
                        print(f"Columns: {df.columns.tolist()}")

                        # Try to identify ID and name columns
                        id_col = None
                        name_col = None

                        for col in df.columns:
                            col_lower = str(col).lower()
                            if "id" in col_lower and id_col is None:
                                id_col = col
                            elif (
                                any(
                                    x in col_lower
                                    for x in ["name", "label", "structure"]
                                )
                                and name_col is None
                            ):
                                name_col = col

                        if id_col and name_col:
                            for idx, row in df.iterrows():
                                try:
                                    struct_id = int(row[id_col])
                                    if struct_id == 0 or struct_id == ROOT_ID:
                                        continue

                                    structures.append(
                                        {
                                            "id": struct_id,
                                            "name": str(row[name_col]),
                                            "acronym": str(
                                                row.get(
                                                    "acronym", row[name_col]
                                                )
                                            )[:10],
                                            "structure_id_path": [
                                                ROOT_ID,
                                                struct_id,
                                            ],
                                            "rgb_triplet": [
                                                int(row.get("r", 100)),
                                                int(row.get("g", 100)),
                                                int(row.get("b", 100)),
                                            ],
                                        }
                                    )
                                except (ValueError, KeyError):
                                    continue

                            if len(structures) > 2:
                                # More than just root + brain
                                num_loaded = len(structures) - 2
                                print(f"Loaded {num_loaded} structures")
                                break
                except Exception:
                    continue
        except Exception as e:
            print(f"Error parsing label file: {e}")
            print("Using default structure list.")
    else:
        print("No label definition files found. Using minimal structure list.")

    return structures


def create_meshes(download_dir_path, tree, annotated_volume, labels, root_id):
    """
    Generate meshes for each brain region.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where meshes will be saved.
    tree : Any
        The structure tree object.
    annotated_volume : numpy.ndarray
        The 3D numpy array representing the annotated brain volume.
    labels : set
        A set of unique labels found in the annotated volume.
    root_id : int
        The ID of the root structure.

    Returns
    -------
    pathlib.Path
        The path to the directory containing the generated meshes.
    """
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False
        node.data = Region(is_label)

    # Mesh creation parameters
    closing_n_iters = 2
    decimate_fraction = 0.2
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
                root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )
    return meshes_dir_path


def create_mesh_dict(structures, meshes_dir_path):
    """
    Create a dictionary mapping structure IDs to their mesh file paths.

    Parameters
    ----------
    structures : list
        A list of dictionaries, where each dictionary represents a structure.
    meshes_dir_path : pathlib.Path
        The directory where the mesh files are stored.

    Returns
    -------
    tuple
        - dict: A dictionary where keys are structure IDs and values are paths
          to their corresponding .obj mesh files.
        - list: A filtered list of structures that successfully had a mesh
          created and verified.
    """
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )
    return meshes_dict, structures_with_mesh


def retrieve_additional_references(download_dir_path):
    """
    Return a dictionary of additional reference images.

    The VALiDATe29 atlas includes multiple contrasts and diffusion maps:
    - T2* weighted
    - Proton density
    - Fractional anisotropy (FA)
    - Mean diffusivity (MD)

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory containing the downloaded atlas files.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}

    # Define possible additional reference files
    reference_maps = {
        "t2star": [
            "template_T2star.nii.gz",
            "template_T2star.nii",
            "T2star_template.nii.gz",
        ],
        "pd": ["template_PD.nii.gz", "template_PD.nii", "PD_template.nii.gz"],
        "fa": ["template_FA.nii.gz", "template_FA.nii", "FA_template.nii.gz"],
        "md": ["template_MD.nii.gz", "template_MD.nii", "MD_template.nii.gz"],
    }

    for map_name, possible_files in reference_maps.items():
        for filename in possible_files:
            # Check in multiple possible locations
            possible_paths = [
                download_dir_path / "VALiDATe29" / filename,
                download_dir_path / filename,
            ]

            for file_path in possible_paths:
                if file_path.exists():
                    try:
                        additional_references[map_name] = load_any(
                            file_path, as_numpy=True
                        )
                        print(f"Loaded {map_name} from: {file_path}")
                        break
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")

            if map_name in additional_references:
                break

    return additional_references


def create_atlas(working_dir):
    """
    Package the VALiDATe29 Squirrel Monkey atlas.

    Downloads the necessary raw data, processes the annotation and reference
    volumes, creates meshes for each brain region, and wraps up the data
    into the BrainGlobe atlas format.

    Parameters
    ----------
    working_dir : pathlib.Path
        The directory where temporary and final atlas files will be stored.

    Returns
    -------
    pathlib.Path
        The path to the generated BrainGlobe atlas zip file.
    """
    # Create working directory
    working_dir.mkdir(exist_ok=True, parents=True)

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # Download atlas files
    atlas_files_dir = download_resources(download_dir_path)

    # Load reference and annotation volumes
    reference_volume, annotated_volume = retrieve_reference_and_annotation(
        atlas_files_dir
    )

    # Get hemisphere map
    hemispheres_stack = retrieve_hemisphere_map()

    # Get structure information
    structures = retrieve_structure_information(atlas_files_dir)

    # Create structure tree
    tree = get_structures_tree(structures)
    labels = set(np.unique(annotated_volume).astype(np.int32))

    # Remove structures with missing annotations
    existing_structures = []
    for structure in structures:
        stree = tree.subtree(structure["id"])
        ids = set(stree.nodes.keys())
        matched_labels = ids & labels
        if matched_labels:
            existing_structures.append(structure)
        else:
            node = tree.nodes[structure["id"]]
            print(
                f"{node.tag} not found in annotation volume, "
                "removing from list of structures..."
            )
    structures = existing_structures
    tree = get_structures_tree(structures)

    # Create meshes
    meshes_dir_path = create_meshes(
        download_dir_path, tree, annotated_volume, labels, ROOT_ID
    )

    meshes_dict, structures_with_mesh = create_mesh_dict(
        structures, meshes_dir_path
    )

    # Get additional references
    additional_references = retrieve_additional_references(atlas_files_dir)

    # Wrap up the atlas
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )

    print(f"Atlas packaged successfully: {output_filename}")
    return output_filename


if __name__ == "__main__":
    # Generated atlas path
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    create_atlas(bg_root_dir)
