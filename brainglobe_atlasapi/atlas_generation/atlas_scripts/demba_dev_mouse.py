from pathlib import Path

import pooch

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

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
ATLAS_NAME = "demba_dev_mouse"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1101/2024.06.14.598876"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mus musculus"

# The URL for the data files
ATLAS_LINK = (
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_volumes/DeMBA_templates",
    "https://data.kg.ebrains.eu/zip?container=https://data-proxy.ebrains.eu/api/v1/buckets/d-8f1f65bb-44cb-4312-afd4-10f623f929b8?prefix=interpolated_segmentations/AllenCCFv3_segmentations/20um/2022",
)

download_hashes = (
    "0ad9cc9ee260096a6d88178a7ceda5c9fc99c9eca3df23fbf7cf5abfbb21cebf",
)
# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 20


def download_resources(download_dir_path, atlas_file_url, ATLAS_NAME):

    utils.check_internet_connection()

    download_name = ATLAS_NAME
    destination_path = download_dir_path / download_name
    for al in atlas_file_url:
        pooch.retrieve(
            url=al,
            known_hash=None,
            path=destination_path,
            progressbar=True,
            processor=pooch.Unzip(extract_dir="."),
        )
    return destination_path


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    reference = None
    annotation = None
    return reference, annotation


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
    This function should return a pandas DataFrame with information about your
    atlas.

    The DataFrame should be in the following format:

    ╭────┬───────────────────┬─────────┬───────────────────┬─────────────────╮
    | id | name              | acronym | structure_id_path | rgb_triplet     |
    |    |                   |         |                   |                 |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 997| root              | root    | [997]             | [255, 255, 255] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 8  | Basic cell groups | grey    | [997, 8]          | [191, 218, 227] |
    ├────┼───────────────────┼─────────┼───────────────────┼─────────────────┤
    | 567| Cerebrum          | CH      | [997, 8, 567]     | [176, 240, 255] |
    ╰────┴───────────────────┴─────────┴───────────────────┴─────────────────╯

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    return None


def retrieve_or_construct_meshes():
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    meshes_dict = {}
    return meshes_dict


def retrieve_additional_references():
    """This function only needs editing if the atlas has additional reference
    images. It should return a dictionary that maps the name of each
    additional reference image to an image stack containing its data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)
    download_resources(
        download_dir_path=bg_root_dir,
        ATLAS_NAME=ATLAS_NAME,
        atlas_file_url=ATLAS_LINK,
    )
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
