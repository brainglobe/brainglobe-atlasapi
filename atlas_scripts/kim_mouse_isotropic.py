"""Package the Kim Mouse Isotropic Brain Atlas.

This script downloads the Kim Mouse isotropic brain atlas resources, including
the raw data, annotation volumes, and structure hierarchy. It processes the
ontology data, generates meshes for anatomical structures, and packages the
atlas in the BrainGlobe format.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "kim_mouse_isotropic"

# DOI of the most relevant citable document
CITATION = (
    "Chon et al., 2019, Nature Communications "
    "(PMID: 31699990), doi: 10.1038/s41467-019-13057-w"
)

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mus musculus"

# The URL for the data files
ATLAS_LINK = "https://figshare.com/ndownloader/articles/25750983/versions/1"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "rsp"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 20

ATLAS_PACKAGER = "Pavel Vychyk / pavel.vychyk@brain.mpg.de"

# Custom globals for the retrieved file names
ONTOLOGY_FILE = "UnifiedAtlas_Label_ontology_v2.csv"
ANNOTATION_FILE = "UnifiedAtlas_Label_v2_20um-isotropic.nii"
REFERENCE_FILE = "UnifiedAtlas_template_coronal_20um-isotropic.nii"


def download_resources() -> Tuple[Path, Path]:
    """Download the atlas resource files and create required directories.

    Returns
    -------
    Tuple[Path, Path]
        The root directory and the nested download directory as Path objects.
    """
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    download_dir = bg_root_dir / "download"
    download_dir.mkdir(exist_ok=True)
    download_dir_content = {item.name for item in download_dir.iterdir()}
    if (
        ONTOLOGY_FILE not in download_dir_content
        and REFERENCE_FILE not in download_dir_content
        and ANNOTATION_FILE not in download_dir_content
    ):
        pooch.retrieve(
            url=ATLAS_LINK,
            path=(download_dir),
            progressbar=True,
            known_hash=None,
            processor=pooch.Unzip(extract_dir=download_dir),
        )
    return bg_root_dir, download_dir


def retrieve_reference_and_annotation(
    download_dir: Path,
) -> Tuple[np.ndarray, np.ndarray]:
    """Retrieve the reference and annotation volumes as two numpy arrays.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        The reference volume and the annotation volume as numpy arrays.
    """
    download_dir_content = {item.name for item in download_dir.iterdir()}
    if ANNOTATION_FILE not in download_dir_content:
        raise FileNotFoundError(
            f"{ANNOTATION_FILE} not found in {download_dir}"
        )
    if REFERENCE_FILE not in download_dir_content:
        raise FileNotFoundError(
            f"{REFERENCE_FILE} not found in {download_dir}"
        )
    # Load reference volume (nifti -> numpy array)
    img_array = load_any(download_dir / REFERENCE_FILE)
    reference = img_array.squeeze().astype(np.int32)
    # Load annotation volume (nifti -> numpy array)
    img_array = load_any(download_dir / ANNOTATION_FILE)
    annotation = img_array.squeeze().astype(np.int32)
    return reference, annotation


def retrieve_structure_information(download_dir: Path) -> List[Dict[str, Any]]:
    """Return a list with dictionaries containing information about the atlas.

    Example of the ROOT dictionary entry:
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255]
        }

    Returns
    -------
    List[Dict[str, Any]]
        List of dictionaries containing the atlas information.
    """
    download_dir_content = {item.name for item in download_dir.iterdir()}
    if ONTOLOGY_FILE not in download_dir_content:
        raise FileNotFoundError(f"{ONTOLOGY_FILE} not found in {download_dir}")
    df = pd.read_csv(download_dir / ONTOLOGY_FILE)
    int_cols = ["id", "RGB_1", "RGB_2", "RGB_3", "structure_id_path"]
    df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["id"])
    df[int_cols] = df[int_cols].astype("int32")
    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]
    id_to_row = df.set_index("id").to_dict("index")
    for _, row in df.iterrows():
        structures.append(
            {
                "id": row["id"],
                "name": row["name"],
                "acronym": row["acronym"],
                "structure_id_path": get_path_to_root_id(id_to_row, row["id"]),
                "rgb_triplet": [row["RGB_1"], row["RGB_2"], row["RGB_3"]],
            }
        )
    return structures


def retrieve_or_construct_meshes(
    download_dir: Path,
    annotation: np.ndarray,
    structures: List[Dict[str, Any]],
) -> Dict[int, str]:
    """Retrieve or construct meshes for atlas structures.

    Parameters
    ----------
    download_dir : Path
        Directory containing atlas data.
    annotation : np.ndarray
        Annotation volume as a NumPy array.
    structures : List[Dict[str, Any]]
        List of dictionaries describing atlas structures.

    Returns
    -------
    Dict[int, str]
        Mapping from structure ID to mesh file path.
    """
    meshes_dir = download_dir
    meshes_dict = construct_meshes_from_annotation(
        meshes_dir, annotation, structures, ROOT_ID
    )
    return meshes_dict


def get_path_to_root_id(
    id_to_row: Dict[int, Dict[str, Any]], current_id: int
) -> List[int]:
    """Generate the hierarchical path from a given structure ID to the root ID.

    Traces the parent IDs from `current_id` up to
    the root of the atlas hierarchy. The path includes the root ID at the start
    and the `current_id` at the end.

    Parameters
    ----------
    id_to_row : Dict[int, Dict[str, Any]]
        Dictionary mapping structure IDs to rows in the ontology DataFrame.
    current_id : int
        The starting structure ID for which the path to the root is computed.

    Returns
    -------
    List[int]
        A list of structure IDs starting from the root ID to `current_id`.
    """
    path_to_root = []
    while True:
        current_row = id_to_row.get(current_id)
        if current_row is None:
            break
        path_to_root.insert(0, current_id)
        parent_id = current_row["structure_id_path"]
        if parent_id == current_id:
            break
        current_id = parent_id

    path_to_root.insert(0, ROOT_ID)
    return path_to_root


if __name__ == "__main__":
    root_dir, download_dir = download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation(
        download_dir
    )
    structures = retrieve_structure_information(download_dir)
    meshes_dict = retrieve_or_construct_meshes(
        download_dir, annotated_volume, structures
    )

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
        working_dir=root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
