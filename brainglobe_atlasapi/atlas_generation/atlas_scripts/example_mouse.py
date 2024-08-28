__version__ = 1  # The version of the atlas in the brainglobe_atlasapi, this is internal, if this is the first time this atlas has been added the value should be 1
ATLAS_NAME = "example_mouse"  # The expected format is FirstAuthor_SpeciesCommonName, i.e., kleven_rat
CITATION = "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"  # DOI of the most relevant citable document
SPECIES = "Mus musculus"  # The scientific name of the species, i.e., Rattus norvegicus
ATLAS_LINK = "http://www.brain-map.org"  # The URL for the data files
ORIENTATION = "asr"  # The orientation of the atlas
ROOT_ID = 997  # The id of the highest level of the atlas. This is commonly called root or brain.
RESOLUTION = 100  # The resolution of your volume in microns.

from pathlib import Path

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from requests import exceptions
from tqdm import tqdm

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data


def download_resources():
    """
    Download the necessary resources for the atlas.
    """
    pass


def retrieve_template_and_reference():
    """
    Retrieve the desired template and reference as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the reference volume, and the second array is the annotated volume.
    """
    # Create temporary download directory
    download_dir_path = Path.cwd() / "downloading_path"
    download_dir_path.mkdir(exist_ok=True)

    # Setup the reference space cache
    spacecache = ReferenceSpaceCache(
        manifest=download_dir_path / "manifest.json",
        resolution=RESOLUTION,
        reference_space_key="annotation/ccf_2017",
    )

    # Download annotated and template volumes
    reference_volume, _ = spacecache.get_template_volume()
    annotation_volume, _ = spacecache.get_annotation_volume()
    return reference_volume, annotated_volume


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map. This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map, or None if the atlas is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Retrieve the structures tree and meshes.

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    download_dir_path = Path.cwd() / "downloading_path"
    oapi = OntologiesApi()
    spacecache = ReferenceSpaceCache(
        manifest=download_dir_path / "manifest.json",
        resolution=RESOLUTION,
        reference_space_key="annotation/ccf_2017",
    )
    struct_tree = spacecache.get_structure_tree()  # Download structures tree

    select_set = (
        "Structures whose surfaces are represented by a precomputed mesh"
    )
    mesh_set_ids = [
        s["id"]
        for s in oapi.get_structure_sets()
        if s["description"] == select_set
    ]
    structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)[:3]

    # Loop over structures, remove entries not used
    for struct in structs_with_mesh:
        [
            struct.pop(k)
            for k in ["graph_id", "structure_set_ids", "graph_order"]
        ]
    return structs_with_mesh


def retrieve_or_construct_meshes():
    """
    This function should return a dictionary of ids and corresponding paths to mesh files.
    Some atlases are packaged with mesh files, in these cases we should use these files.
    Then this function should download those meshes. In other cases we need to construct
    the meshes ourselves. For this we have helper functions to achieve this.
    """
    oapi = OntologiesApi()
    space = ReferenceSpaceApi()
    meshes_dir = Path.cwd() / "mesh_temp_download"
    meshes_dir.mkdir(exist_ok=True)

    meshes_dict = dict()
    structs_with_mesh = retrieve_structure_information()

    for s in tqdm(structs_with_mesh):
        name = s["id"]
        filename = meshes_dir / f"{name}.obj"
        try:
            space.download_structure_mesh(
                structure_id=s["id"],
                ccf_version="annotation/ccf_2017",
                file_name=filename,
            )
            meshes_dict[name] = filename
        except (exceptions.HTTPError, ConnectionError):
            print(s)
    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be edited below.
bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
bg_root_dir.mkdir(exist_ok=True)
download_resources()
reference_volume, annotated_volume = retrieve_template_and_reference()
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
    hemispheres_stack=hemispheres_stack,
    cleanup_files=False,
    compress=True,
    scale_meshes=True,
)
