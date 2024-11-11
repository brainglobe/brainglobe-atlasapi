import pooch
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from requests import exceptions
from tqdm import tqdm

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR

# a working example atlas-packaging script, which makes a simplified version
# of the Allen Mouse Brain Atlas, at 100um resolution. See `template_script.py`
# for a starting point to package your own atlas.
__version__ = 0  # This will make the example mouse version 1.0
ATLAS_NAME = "example_mouse"
CITATION = "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"
SPECIES = "Mus musculus"  # The scientific name of the species,
ATLAS_LINK = "http://www.brain-map.org"  # The URL for the data files
ORIENTATION = "asr"  # The orientation of the atlas
ROOT_ID = 997  # The id of the highest level of the atlas.
RESOLUTION = 100  # The resolution of your volume in microns.

BG_ROOT_DIR = DEFAULT_WORKDIR / ATLAS_NAME


def download_resources():
    """
    Download the necessary resources for the atlas. Here we don't, because we
    can out-source this to the Allen SDK in later functions.
    """
    pass


def retrieve_reference_and_annotation():
    """
    Retrieve the Allen Mouse atlas reference and annotation as two numpy arrays
    using the allen_sdk.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotated volume.
    """
    # Create temporary download directory
    download_dir_path = BG_ROOT_DIR / "downloading_path"
    print(download_dir_path)

    download_dir_path.mkdir(exist_ok=True)

    # Setup the reference space cache
    spacecache = ReferenceSpaceCache(
        manifest=download_dir_path / "manifest.json",
        resolution=RESOLUTION,
        reference_space_key="annotation/ccf_2017",
    )

    reference_volume, _ = spacecache.get_template_volume()
    annotation_volume, _ = spacecache.get_annotation_volume()
    expected_reference_hash = (
        "6c24cae773a5cf256586b0384af0ac93ad68564d211c9bdcff4bee9acf07786a"
    )
    expected_annotation_hash = (
        "451e6a82f531d3db4b58056d024d3e2311703c2adc15cefa75e0268e7f0e69a4"
    )
    reference_hash = pooch.file_hash(
        download_dir_path / "average_template_100.nrrd"
    )
    annotation_hash = pooch.file_hash(
        download_dir_path / "annotation" / "ccf_2017" / "annotation_100.nrrd"
    )
    assert (
        reference_hash == expected_reference_hash
    ), "The hash of the reference volume does not match the expected hash."

    assert (
        annotation_hash == expected_annotation_hash
    ), "The hash of the annotation volume does not match the expected hash."
    # Download annotated and template volumes
    return reference_volume, annotation_volume


def retrieve_hemisphere_map():
    """
    The Allen atlas is symmetrical, so we can just return `None` in this
    function.

        Returns:
            numpy.array or None: A numpy array representing the hemisphere map,
            or None if the atlas is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Retrieve the structures tree and meshes for the Allen mouse brain atlas.

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    download_dir_path = BG_ROOT_DIR / "downloading_path"
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
    This function should return a dictionary of ids and corresponding paths to
    mesh files. This atlas comes packaged with mesh files, so we don't need to
    use our helper functions to create them ourselves in this case.
    """
    space = ReferenceSpaceApi()
    meshes_dir = BG_ROOT_DIR / "mesh_temp_download"
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


# Set up for the example mouse done: use default code to wrap up the atlas
BG_ROOT_DIR.mkdir(exist_ok=True)
download_resources()
reference_volume, annotated_volume = retrieve_reference_and_annotation()
hemispheres_stack = retrieve_hemisphere_map()
structures = retrieve_structure_information()
meshes_dict = retrieve_or_construct_meshes()
if __name__ == "__main__":

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
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
