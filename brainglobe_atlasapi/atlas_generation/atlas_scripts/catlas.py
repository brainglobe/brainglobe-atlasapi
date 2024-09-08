""" Atlas for a domestic cat """

from pathlib import Path

import pooch

from brainglobe_atlasapi.utils import check_internet_connection

###Metadata
__version__ = 1
ATLAS_NAME = "catlas"
CITATION = "Stolzberg, Daniel et al 2017.https://doi.org/10.1002/cne.24271"
SPECIES = "Felis catus"
ATLAS_LINK = (
    "https://github.com/CerebralSystemsLab/CATLAS/blob/main/SlicerFiles/"
)
ORIENTATION = "lps"
ROOT_ID = 999  # Placeholder as no hierarchy is present
RESOLUTION = 500  # microns
ATLAS_PACKAGER = "Henry Crosswell"

annotated_volume = None
working_dir = Path("F:/Users/Henry/Downloads/Setup/CATLAS-main/temp_pooch")


def download_resources(working_dir):
    """
    Download the necessary resources for the atlas.
    If possible, please use the Pooch library to retrieve any resources.
    """
    # Setup download folder
    download_dir_path = working_dir / "download_dir"
    download_dir_path.mkdir(parents=True, exist_ok=True)
    # Setup atlas folder within download_dir
    atlas_dir_path = download_dir_path / "atlas_dir"
    atlas_dir_path.mkdir(exist_ok=True)

    check_internet_connection()
    local_file_path_list = []
    file_hash_list = [
        ["meanBrain.nii", "md5:ffa42f5d703b192770d41cdf5493efc8"],
        ["CorticalAtlas.nii", "md5:9b43e80b4052b7a8e8103163f4f5ff7d"],
        ["CATLAS_COLORS.txt", "md5:d18c626858b492b139afc2094130b047"],
        ["CorticalAtlas-Split.nii", "md5:c31fcaa92d658a20c3bb8059089bed14"],
        ["CATLAS_COLORS-SPLIT.txt", "md5:bd7df732c51f23dae44daccbc58618bb"],
    ]

    for file, hash in file_hash_list:
        file_path = atlas_dir_path / file
        cached_file = pooch.retrieve(
            url=ATLAS_LINK + file, known_hash=hash, path=file_path
        )
        local_file_path_list.append(cached_file)

    return local_file_path_list


print(download_resources(working_dir))


def retrieve_template_and_reference():
    """
    Retrieve the desired template and reference as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays.
        The first array is the template volume,
        and the second array is the reference volume.
    """
    template = None
    reference = None
    return template, reference


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    This function should return a pandas DataFrame
    with information about your atlas.

    The DataFrame should be in the following format:

    ╭─────┬──────────────────┬─────────┬───────────────────┬─────────────────╮
    | id  | name             | acronym | structure_id_path | rgb_triplet     |
    |     |                  |         |                   |                 |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997 | root             | root    | []                | [255, 255, 255] |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8   | grps and regions | grey    | [997]             | [191, 218, 227] |
    ├─────┼──────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567 | Cerebrum         | CH      | [997, 8]          | [176, 240, 255] |
    ╰─────┴──────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    return None


def retrieve_or_construct_meshes():
    """
    This should return a dict of ids and corresponding paths to mesh files.
    Use packaged mesh files if possible.
    Download or construct mesh files  - use helper function for this
    """
    meshes_dict = {}
    return meshes_dict


# commenting out to unit test

### If the code above this line has been filled correctly, nothing needs to be
# edited below (unless variables need to be passed between the functions).
# bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
# bg_root_dir.mkdir(exist_ok=True)
# working_dir = bg_root_dir
# download_resources()
# template_volume, reference_volume = retrieve_template_and_reference()
# hemispheres_stack = retrieve_hemisphere_map()
# structures = retrieve_structure_information()
# meshes_dict = retrieve_or_construct_meshes()

# output_filename = wrapup_atlas_from_data(
#     atlas_name=ATLAS_NAME,
#     atlas_minor_version=__version__,
#     citation=CITATION,
#     atlas_link=ATLAS_LINK,
#     species=SPECIES,
#     resolution=(RESOLUTION,) * 3,
#     orientation=ORIENTATION,
#     root_id=ROOT_ID,
#     reference_stack=template_volume,
#     annotation_stack=annotated_volume,
#     structures_list=structures,
#     meshes_dict=meshes_dict,
#     working_dir=working_dir,
#     hemispheres_stack=None,
#     cleanup_files=False,
#     compress=True,
#     scale_meshes=True,
# )
