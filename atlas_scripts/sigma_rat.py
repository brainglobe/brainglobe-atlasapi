"""Package the SIGMA v2 Wistar rat anatomical atlas for BrainGlobe.

SIGMA v2 provides MRI templates with Waxholm v4 anatomical annotations
registered into SIGMA space. The primary template and any additional
references remain to be selected after source-file inspection.
"""

from pathlib import Path

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

# BrainGlobe package revision; distinct from SIGMA source version 2.0.
__version__ = 0
ATLAS_NAME = "sigma_rat"
CITATION = (
    "Barrière et al. 2019, https://doi.org/10.1038/s41467-019-13575-7; "
    "Kleven et al. 2023, https://doi.org/10.1038/s41592-023-02034-3"
)
SPECIES = "Rattus norvegicus"
ATLAS_LINK = "https://zenodo.org/records/10635831"
ATLAS_PACKAGER = "Amirreza Bahramani"

# Determine the source array orientation from the NIfTI files.
ORIENTATION = None
# Confirm the matching Waxholm hierarchy and root ID from source metadata.
ROOT_ID = None
# Select the primary template: ex vivo 90 µm or in vivo 150 µm isotropic.
RESOLUTION = None

# Zenodo lists CC BY 4.0, but NITRC lists non-commercial terms and the
# linked GitHub repository contains GPL v3. Resolve scope before release.

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
SOURCE_DATA_DIR = BG_ROOT_DIR / "source_data"


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """
    pass


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
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information():
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
    return None


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

    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
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
    structures = retrieve_structure_information()
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
