"""Template script for generating a BrainGlobe atlas.

Use this script as a starting point to package a new BrainGlobe atlas by
filling in the required functions and metadata.
"""

from pathlib import Path
import nibabel as nib

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
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
ATLAS_NAME = "hoops_dragon"

# DOI of the most relevant citable document
CITATION = "https://doi.org/10.1007/s00429-021-02282-z"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Ctenophorus decresii"

# The URL for the data files
ATLAS_LINK = "https://osf.io/ujenq"

ATLAS_PACKAGER = "Jung Woo Kim"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 999

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 50

REFERENCE_URL = "https://storage.googleapis.com/cos-osf-prod-files-ca-1/1fbc4d1be3b8a6da5513d1ac44abd11d05eebe67dd3c2e4662bbe81008c7c4e5?response-content-disposition=attachment%3B%20filename%3D%22lizard_model.mnc%22%3B%20filename%2A%3DUTF-8%27%27lizard_model.mnc&GoogleAccessId=files-ca-1%40cos-osf-prod.iam.gserviceaccount.com&Expires=1782168332&Signature=29tpfsJIK1exuG7ZpJufYgvXET8WiXncJPksVR1hgRtWzI7fLaGek6QsHB0hq4Nct5I9coFfQEHA9WY6VEYsFHAcXsT8Bhhy0fU9yGFJmgCC2V1dziBd5imfiZVYznGMwKKK8GSBPE1PxxxrUGAz05o5ZbA0PMHvqbLbY5wuKMp9X%2FPolBX5CPbsY4fBfwzlFzBvgFFGV2v3YLXf7KLdky2Uo%2BeouWGaJeTyOMgblDnHIz%2B3oODI1oNEmo%2Be%2Bn22ZzizZLW6ANTcJJj5BBiBm%2BSzSUlGUW%2F3D0ffpe1xVXqc1rFinivoeumuJXWrNJ0%2Fbz2DSQv1pa3AKyBbNpYXow%3D%3D"
ANNOTATION_URL= "https://storage.googleapis.com/cos-osf-prod-files-ca-1/847de51f6de3f0cca6740ac7e393501a43af2ac85414b302f99e7a904df69df1?response-content-disposition=attachment%3B%20filename%3D%22lizard_segmentation_bilateral.mnc%22%3B%20filename%2A%3DUTF-8%27%27lizard_segmentation_bilateral.mnc&GoogleAccessId=files-ca-1%40cos-osf-prod.iam.gserviceaccount.com&Expires=1782168092&Signature=lgnY5wiFGrDm2PH2mSOctOaD4FKs0mAlQ8t2CpIk1jrW2iXlQCSvw4LDkmm%2FwBhdS1PlxnbrUxMpgD10zK0PX7HfGiase0ZFc9fgs5%2BmQKjUQkqTsDyYJn1XAn2gQSniGAHGgeyQTYHyMdDktrZ5uz096GCwn8SRX6CeyPcoGbSyiqNhSBo7WY81kCqzRm%2BtJrqxnjr6X9d4cjLfQ5DwuwCk71g2e7iG8OeBygc2wQ5aNnCr2c2KlmaK9iRpQwjWX1aZjQvTtXiYubAMA72OTw%2BgBtrEyAaJnNocOkjs8ObJFDh3aQjkAT2trys5eowJmdZ4Pwuf3C6S8%2BW%2B3JhTiQ%3D%3D"
LABELS_URL = "https://storage.googleapis.com/cos-osf-prod-files-ca-1/bba5569843825a905eaac1c6432f53a6c6302ec30fee65f520fc1eb4bc3eb084?response-content-disposition=attachment%3B%20filename%3D%22BilateralRegionIDs.csv%22%3B%20filename%2A%3DUTF-8%27%27BilateralRegionIDs.csv&GoogleAccessId=files-ca-1%40cos-osf-prod.iam.gserviceaccount.com&Expires=1782169007&Signature=Y3jg9Bb1GGPZq2N6NM2b1wZ9vUk7iSrHSalBFtxoNsNb5aFOFeoRLuKjF7nS4pFR4bIQLRVW7HKMfCOi4lQ%2FZhjyvfxBjqIeEOindN3uhKwQ%2Bzr9ooNXCn6SCDkz4WjjxcUTDnvYChoq8qVedxrO%2F2LeIaAkuItbVPrIAhdrVbY1L1oNALY5KAnZyiwilFyDGFgD5aUdAl7XHY%2F0w3re6zG5BeNyTkviPeQI4ioD%2FbQaP3EU1HnC34EPLGx%2FWy8lH%2BLPpss09bnExZCZ68GjzucLkOtPIFoYvDzwzjUz5Yc3wmW3SZEw2VY8dKFfwJVarAPp0XM%2Fia%2FWCxryL%2BCb9w%3D%3D"

REFERENCE_FNAME = "lizard_model.mnc"
ANNOTATION_FNAME = "lizard_segmentation_bilateral.mnc"
LABELS_FNAME = "BilateralRegionIDs.csv"

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
        atlas_packager=ATLAS_PACKAGER,
    )
