from pathlib import Path

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

###Metadata
__version__ = 1
ATLAS_NAME = "catlas"
CITATION = "https://doi.org/10.1002/cne.24271"
SPECIES = "Felis catus"
ATLAS_LINK = "https://github.com/CerebralSystemsLab/CATLAS"
ORIENTATION = "lps"
ROOT_ID = 999  # Placeholder as no hierarchy is present
RESOLUTION = 500  # um
ATLAS_PACKAGER = "Henry Crosswell"


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """

    pass


def retrieve_template_and_reference():
    """
    Retrieve the desired template and reference as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the template volume,
        and the second array is the reference volume.
    """
    template = None
    reference = None
    return template, reference


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
    This function should return a pandas DataFrame with information about your atlas.

    The DataFrame should be in the following format:

    ╭─────────────┬───────────────────────────────────┬─────────┬───────────────────────┬───────────────────╮
    | id          | name                              | acronym | structure_id_path     | rgb_triplet       |
    |             |                                   |         |                       |                   |
    ├─────────────┼───────────────────────────────────┼─────────┼───────────────────────┼───────────────────┤
    | 997         | root                              | root    | []                    | [255, 255, 255]   |
    ├─────────────┼───────────────────────────────────┼─────────┼───────────────────────┼───────────────────┤
    | 8           | Basic cell groups and regions     | grey    | [997]                 | [191, 218, 227]   |
    ├─────────────┼───────────────────────────────────┼─────────┼───────────────────────┼───────────────────┤
    | 567         | Cerebrum                          | CH      | [997, 8]              | [176, 240, 255]   |
    ╰─────────────┴───────────────────────────────────┴─────────┴───────────────────────┴───────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    return None


def retrieve_or_construct_meshes():
    """
    This function should return a dictionary of ids and corresponding paths to mesh files.
    Some atlases are packaged with mesh files, in these cases we should use these files.
    Then this function should download those meshes. In other cases we need to construct
    the meshes ourselves. For this we have helper functions to achieve this.
    """
    meshes_dict = {}
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be edited below (unless variables need to be passed between the functions).
bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
bg_root_dir.mkdir(exist_ok=True)
download_resources()
template_volume, reference_volume = retrieve_template_and_reference()
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
    reference_stack=template_volume,
    annotation_stack=annotated_volume,
    structures_list=structures,
    meshes_dict=meshes_dict,
    working_dir=working_dir,
    hemispheres_stack=None,
    cleanup_files=False,
    compress=True,
    scale_meshes=True,
)
