__version__ = "2"

from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from requests import exceptions
from tqdm import tqdm

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR

ATLAS_NAME = "allen_mouse"
SPECIES = "Mus musculus"
ATLAS_LINK = "http://www.brain-map.org"
CITATION = "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"
ORIENTATION = "asr"
RES_UM = 25


def create_atlas(working_dir, resolution):

    # Temporary folder for nrrd files download:
    download_dir_path = working_dir / "downloading_path"
    download_dir_path.mkdir(exist_ok=True)

    # Download annotated and template volume:
    #########################################
    spacecache = ReferenceSpaceCache(
        manifest=download_dir_path / "manifest.json",
        # downloaded files are stored relative to here
        resolution=resolution,
        reference_space_key="annotation/ccf_2017",
        # use the latest version of the CCF
    )

    # Download
    annotated_volume, _ = spacecache.get_annotation_volume()
    template_volume, _ = spacecache.get_template_volume()
    print("Download completed...")

    # Download structures tree and meshes:
    ######################################
    oapi = OntologiesApi()  # ontologies
    struct_tree = spacecache.get_structure_tree()  # structures tree

    # Find id of set of regions with mesh:
    select_set = (
        "Structures whose surfaces are represented by a precomputed mesh"
    )

    mesh_set_ids = [
        s["id"]
        for s in oapi.get_structure_sets()
        if s["description"] == select_set
    ]

    structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)

    # Directory for mesh saving:
    meshes_dir = working_dir / descriptors.MESHES_DIRNAME

    space = ReferenceSpaceApi()
    meshes_dict = dict()
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

    # Loop over structures, remove entries not used:
    for struct in structs_with_mesh:
        [
            struct.pop(k)
            for k in ["graph_id", "structure_set_ids", "graph_order"]
        ]

    # Wrap up, compress, and remove file:0
    print("Finalising atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(resolution,) * 3,
        orientation=ORIENTATION,
        root_id=997,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structs_with_mesh,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
    )

    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)

    create_atlas(bg_root_dir, RES_UM)
