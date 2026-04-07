from pathlib import Path

import numpy as np
import pooch

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree
from brainglobe_atlasapi.utils import atlas_name_from_repr

# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
# remember to add {ATLAS_NAME}_{RESOLUTION}um to:
# brainglobe_atlasapi/atlas_names.py
ATLAS_NAME = "princeton_rat"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.21769/BioProtoc.4854"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Rattus norvegicus"

# The URL for the data files
ATLAS_LINK = "https://figshare.com/ndownloader/files/42485103"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "rai"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 10000

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 25

# --- Script toggles (no CLI args) ---
# If True, do not re-download files that already exist on disk.
SKIP_DOWNLOADS_IF_PRESENT = True
TEMPLATE_URL = "https://figshare.com/ndownloader/files/42485103"
ANNOTATION_URL = "https://figshare.com/ndownloader/files/51181751"
LABELS_URL = "https://www.nitrc.org/frs/download.php/13400/MBAT_WHS_SD_rat_atlas_v4_pack.zip"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

TEMPLATE_FNAME = "PRA.tif"
ANNOTATION_FNAME = "PRA_WHS_v4_anns.tif"
LABELS_FNAME = "WHS_SD_rat_atlas_v4_labels.ilf"

ATLAS_PACKAGER = "Jung Woo Kim"


def download_waxholm_atlas_files(
    download_dir_path, atlas_file_url, ATLAS_NAME
):
    """Download and extract atlas files from a zip archive.

    Downloads zip archives from the Waxholm/NITRC repository and extracts them.

    Parameters
    ----------
    download_dir_path : pathlib.Path
        The directory where the atlas files will be downloaded and extracted.
    atlas_file_url : str
        The URL of the atlas zip file.
    ATLAS_NAME : str
        The name of the atlas, used for naming the downloaded file.

    Returns
    -------
    pathlib.Path
        The download directory path where files were extracted.

    Raises
    ------
    requests.exceptions.ConnectionError
        If there is no internet connection.
    """
    download_name = ATLAS_NAME + "_atlas.zip"

    pooch.retrieve(
        url=atlas_file_url,
        known_hash=None,
        path=download_dir_path,
        fname=download_name,
        progressbar=True,
        processor=pooch.Unzip(extract_dir=""),
    )

    return download_dir_path


def parse_structures_xml(root, path=None, structures=None):
    """Recursively parse the XML structure definition.

    Parameters
    ----------
    root : dict
        The current root element of the XML structure.
    path : list, optional
        The current path of structure IDs, by default None.
    structures : list, optional
        A list to accumulate parsed structures, by default None.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its name, acronym, ID, path, and RGB triplet.
    """
    structures = structures or []
    path = path or []

    rgb_triplet = [int(root["@color"][i : i + 2], 16) for i in (1, 3, 5)]
    id = int(root["@id"])
    struct = {
        "name": root["@name"],
        "acronym": root["@abbreviation"],
        "id": int(root["@id"]),
        "structure_id_path": path + [id],
        "rgb_triplet": rgb_triplet,
    }
    structures.append(struct)

    if "label" in root:
        if isinstance(root["label"], list):
            for label in root["label"]:
                parse_structures_xml(
                    label, path=path + [id], structures=structures
                )
        else:
            parse_structures_xml(
                root["label"], path=path + [id], structures=structures
            )

    return structures


def parse_structures(structures_file: Path):
    """Parse the structures XML file to extract atlas region metadata.

    Parameters
    ----------
    structures_file : pathlib.Path
        The path to the .ilf XML file containing structure definitions.

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its name, acronym, ID, path, and RGB triplet.
    """
    parsed_xml = xmltodict.parse(structures_file.read_text())
    root = parsed_xml["milf"]["structure"]
    root["@abbreviation"] = "root"
    root["@color"] = "#ffffff"
    root["@id"] = "10000"
    root["@name"] = "Root"

    structures = parse_structures_xml(root)
    return structures


def download_resources():
    """
    Download the necessary resources for the atlas (with Pooch).
    """
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    template_path = DOWNLOAD_DIR_PATH / TEMPLATE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME

    needs_download = (not template_path.exists()) or (
        not annotation_path.exists()
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    if should_fetch(template_path):
        pooch.retrieve(
            TEMPLATE_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=TEMPLATE_FNAME,
            known_hash="c764a7a15aa1e4b54fb5277057a7d4af1169cb00864feceb2a0103dc9b74fa7e",
            progressbar=True,
        )
    if should_fetch(annotation_path):
        pooch.retrieve(
            ANNOTATION_URL,
            path=DOWNLOAD_DIR_PATH,
            fname=ANNOTATION_FNAME,
            known_hash="cb8087e4c108a63d1350411d5cbafb394314b43c55afb831e15810c9cfaf0726",
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference = None
    annotation = None
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 0 marks the left hemisphere
    and 1 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information(annotation):
    """
    Return a list of dictionaries with information about the atlas.

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    The expected format for each dictionary is:

    .. code-block:: python

        {
            "id": int,
            "name": str,
            "acronym": str,
            "structure_id_path": list[int],
            "rgb_triplet": list[int, int, int],
        }

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    # Download and extract zip archives from Waxholm/NITRC
    download_waxholm_atlas_files(DOWNLOAD_DIR_PATH, LABELS_URL, ATLAS_NAME)
    labels_files_dir = DOWNLOAD_DIR_PATH / "MBAT_WHS_SD_rat_atlas_v4_pack/Data"

    # Parse structure metadata
    structures = parse_structures(labels_files_dir / LABELS_FNAME)

    # Remove structures with missing annotations
    tree = get_structures_tree(structures)
    labels = set(np.unique(annotation).astype(np.int32))
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

    return structures


def retrieve_or_construct_meshes():
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    If the atlas is packaged with mesh files, download and use them. Otherwise,
    construct the meshes using available helper functions.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dict = {}
    return meshes_dict


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    This function should be edited only if the atlas includes additional
    reference images. The dictionary should map the name of each additional
    reference image to its corresponding image stack data.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    if RESOLUTION is None:
        raise ValueError("RESOLUTION must be set before running this script.")

    bg_root_dir = BG_ROOT_DIR
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(bg_root_dir.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {bg_root_dir}. "
        )
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information(annotated_volume)
    meshes_dict = retrieve_or_construct_meshes()

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
