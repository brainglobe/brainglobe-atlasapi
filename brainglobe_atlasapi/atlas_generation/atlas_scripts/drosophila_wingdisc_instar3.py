from pathlib import Path

import pandas as pd
from brainglobe_utils.image.scale import scale_and_convert_to_16_bits
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    read_itk_labels,
)
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "drosophila_wingdisc_instar3"

# DOI of the most relevant citable document
CITATION = "unpublished"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Drosophila melanogaster"

# The URL for the data files
ATLAS_LINK = None

__version__ = 0

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "las"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 2
# Need review
gin_url = "https://gin.g-node.org/BrainGlobe/blackcap_materials/raw/master/blackcap_materials.zip"


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """
    """
    resources_path = pooch.retrieve(
        gin_url, known_hash=None, processor=pooch.Unzip(), progressbar=True
    )
    """
    resources_path = Path("D:/UCL/Postgraduate_programme/templates/Version3")
    return resources_path


resources_path = download_resources()


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    print("loading reference and annotation volume")
    # Need review
    """
    annotation_volume_path = Path(resources_path[0])
    reference_volume_path = Path(resources_path[2])

    annotation = load_nii(annotation_volume_path, as_array=True)
    reference = load_nii(reference_volume_path, as_array=True)
    """
    annotation_volume_path = Path(
        resources_path / "pouch_peripodial_hinge_notum_refined_"
        "filtered_filtered_filtered.nii.gz"
    )
    assert (
        annotation_volume_path.exists()
    ), "Annotation volume path does not exist."

    reference_volume_path = Path(
        resources_path
        / "template_wingdisc-CSLM-brightness-corrected-trimean.nii.gz"
    )
    assert (
        reference_volume_path.exists()
    ), "Reference volume path does not exist."

    annotation = load_nii(annotation_volume_path, as_array=True)
    reference = load_nii(reference_volume_path, as_array=True)
    reference = scale_and_convert_to_16_bits(reference)
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
    label_path = resources_path / "wingdisc_annotation.txt"

    itk_labels = pd.DataFrame(read_itk_labels(label_path, 2))
    atlas_info = pd.DataFrame(columns=["id", "name", "acronym", "rgb_triplet"])
    for index, row in itk_labels.iterrows():
        atlas_info.loc[len(atlas_info)] = {
            "id": row["id"],
            "name": row["name"],
            "acronym": row["acronym"],
            "rgb_triplet": list(row["rgb_triplet"]),
        }
    root_info = pd.DataFrame(
        {
            "id": 997,
            "name": ["root"],
            "acronym": ["root"],
            "rgb_triplet": [[255, 255, 255]],
        }
    )
    atlas_info = pd.concat([root_info, atlas_info], ignore_index=True)
    atlas_info["structure_id_path"] = atlas_info["id"].apply(
        lambda row: [997, row] if row != 997 else [997]
    )
    atlas_info = atlas_info[
        ["id", "name", "acronym", "structure_id_path", "rgb_triplet"]
    ]
    atlas_info = atlas_info.to_dict("records")
    return atlas_info


def retrieve_or_construct_meshes(annotated_volume, ROOT_ID):
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    download_path = resources_path / "wingdisc_meshes"
    download_path.mkdir(exist_ok=True, parents=True)
    structures = retrieve_structure_information()
    meshes_dict = construct_meshes_from_annotation(
        download_path, annotated_volume, structures, ROOT_ID
    )
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
    bg_root_dir.mkdir(exist_ok=True, parents=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, ROOT_ID)

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
