__version__ = "0"

import json
import time

import meshio as mio
import nrrd
import numpy as np
import pooch
from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache
from requests import exceptions
from rich.progress import track
from tqdm import tqdm

from brainglobe_atlasapi import descriptors, utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "allen_mouse_bluebrain_barrels"
SPECIES = "Mus musculus"
ATLAS_LINK = "http://www.brain-map.org"
CITATION = "Bolaños-Puchet S., Teska A., et al. (2024).  https://doi.org/10.1162/imag_a_00209"
ATLAS_PACKAGER = "Axel Bisi"
ORIENTATION = "asr"
ROOT_ID = 997
RES_UM = 25


def create_atlas(working_dir, resolution):
    # Parameters for mesh creation:
    CLOSING_N_ITERS = 2
    DECIMATE_FRACTION = 0.3
    SMOOTH = True

    # Temporary folder for nrrd files download:
    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)

    # Download original Allen template volume:
    #########################################
    spacecache = ReferenceSpaceCache(
        manifest=download_dir_path / "manifest.json",
        # downloaded files are stored relative to here
        resolution=resolution,
        reference_space_key="annotation/ccf_2017",
        # use the latest version of the CCF
    )

    template_volume, _ = spacecache.get_template_volume()

    # Download enhanced barrel-containing Allen annotation files by BlueBrain,
    # and hierarchy:
    #########################################
    annotation_dir_path = working_dir / "downloads/annotation_enhanced"
    annotation_dir_path.mkdir(exist_ok=True)

    if resolution == 10:
        gin_url = "https://gin.g-node.org/BrainGlobe/bluebrain_barrel_materials/raw/master/annotation_barrels_10.nrrd"
    elif resolution == 25:
        gin_url = "https://gin.g-node.org/BrainGlobe/bluebrain_barrel_materials/raw/master/annotation_barrels_25.nrrd"
    else:
        raise ValueError("Resolution {}um not supported.".format(resolution))

    utils.check_internet_connection()
    annotation_file_path = pooch.retrieve(
        gin_url,
        known_hash=None,
        path=annotation_dir_path,
        progressbar=True,
    )

    # Load enhanced annotation volume:
    annotated_volume = nrrd.read(annotation_file_path)[0]

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

    # Get structures with mesh for both versions
    structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)

    # Download hierarchy:
    gin_url = "https://gin.g-node.org/BrainGlobe/bluebrain_barrel_materials/raw/master/hierarchy.json"
    hierarchy_path = pooch.retrieve(
        gin_url,
        known_hash=None,
        path=annotation_dir_path,
        fname="hierarchy.json",
        progressbar=True,
    )
    structs_with_barrels = json.load(open(hierarchy_path))

    # Add barrels structures to Allen structures
    def find_dicts_with_key_containing_substring(d, key, substring):
        """
        Recursively find all dictionaries within a nested dictionary that
        contain a specific substring in the value associated with a given key.

        Args:
        d (dict): The input dictionary.
        key (str): The key to search for.
        substring (str): The substring to search for in the value associated
        with the key.

        Returns:
        list: A list of dictionaries that contain the key with a value
        containing the substring.
        """
        if not isinstance(d, dict):
            raise ValueError("Input should be a dictionary")

        matching_dicts = []

        def recurse(sub_d):
            contains_substring = False

            for k, v in sub_d.items():
                if isinstance(v, dict):
                    recurse(v)
                elif isinstance(v, list):
                    for item in v:
                        if isinstance(item, dict):
                            recurse(item)
                if k == key and substring in str(v):
                    contains_substring = True

            if contains_substring:
                matching_dicts.append(sub_d)

        recurse(d)

        return matching_dicts

    matching_dicts = find_dicts_with_key_containing_substring(
        structs_with_barrels, key="name", substring="barrel"
    )
    matching_dicts = [
        d
        for d in matching_dicts
        if d["graph_order"] in [52, 53, 54, 55, 56, 57]
    ]
    main_barrel_parent_struct = [
        s for s in structs_with_mesh if s["acronym"] == "SSp-bfd"
    ][0]
    structures_present = [
        "SSp-bfd1",
        "SSp-bfd2/3",
        "SSp-bfd4",
        "SSp-bfd5",
        "SSp-bfd6a",
        "SSp-bfd6b",
    ]  # keep laminar structures
    keys_to_keep = [
        "acronym",
        "graph_id",
        "graph_order",
        "id",
        "name",
        "rgb_triplet",
        "structure_set_ids",
        "structure_id_path",
    ]
    dict_to_add = []
    for d in matching_dicts:

        # Ignore parent-level SSp-bfd layers
        if d["acronym"] in structures_present:
            print("Skipping because already present:", d)
            continue
        # Ignore sub-structures layer 2 and 3 to keep layer 2/3 structure
        if d["graph_order"] == 53 and d["acronym"] in ["SSp-bfd2", "SSp-bfd3"]:
            print("Excluding", d, "to keep layer 2/3 structure only.")
            continue
        elif d["graph_order"] == 54 and (
            "layer 2" in d["name"] or "layer 3" in d["name"]
        ):
            print("Excluding", d, "to keep layer 2/3 structure only.")
            continue

        # Add desired barrel-related structures, with corresponding fields
        else:
            current_id = d["id"]
            # Find corresponding parent structure
            if d["graph_order"] == 52:  # barrel-level -> find SSp-bfd
                # Create new structure_id_path for barrel structure
                d["structure_id_path"] = main_barrel_parent_struct[
                    "structure_id_path"
                ] + [current_id]
            elif (
                d["graph_order"] == 53
            ):  # barrel layer-level -> find SSp-bfd-barrel also
                parent_struct_id = d["parent_structure_id"]
                parent_struct = [
                    s for s in matching_dicts if s["id"] == parent_struct_id
                ][0]
                parent_struct["structure_id_path"] = main_barrel_parent_struct[
                    "structure_id_path"
                ] + [parent_struct_id]
                # Create new structure_id_path for barrel-layer structure
                d["structure_id_path"] = (
                    main_barrel_parent_struct["structure_id_path"]
                    + [d["parent_structure_id"]]
                    + [current_id]
                )

            # Complete with other keys
            d["rgb_triplet"] = main_barrel_parent_struct["rgb_triplet"]
            d["graph_id"] = 1
            d["structure_set_ids"] = None
            dict_to_add.append({k: d[k] for k in keys_to_keep})

    # Add list of dicts to structs_with_mesh
    structs_with_mesh = structs_with_mesh + dict_to_add

    # Directory for mesh saving:
    meshes_dir = (
        working_dir
        / descriptors.MESHES_DIRNAME
        / "meshes_{}".format(resolution)
    )
    # If directory exists, then skip
    if not meshes_dir.exists():
        meshes_dir.mkdir(exist_ok=False, parents=True)

    # Download existing Allen meshes:
    space = ReferenceSpaceApi()
    meshes_dict = dict()
    for s in tqdm(structs_with_mesh):
        name = s["id"]
        filename = meshes_dir / f"{name}.obj"

        if filename.exists():
            meshes_dict[name] = filename
            continue

        try:
            space.download_structure_mesh(
                structure_id=s["id"],
                ccf_version="annotation/ccf_2017",
                file_name=filename,
            )
            meshes_dict[name] = filename
        except (exceptions.HTTPError, ConnectionError):
            print(f"Failed to download mesh for {s['name']} ({s['id']})")

    # Create missing meshes
    tree = get_structures_tree(structs_with_mesh)
    print(
        f"Number of brain regions: {tree.size()}, "
        f"max tree depth: {tree.depth()}"
    )

    # generate binary mask for mesh creation
    labels = np.unique(annotated_volume).astype(np.int_)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    start = time.time()

    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        description="Creating meshes",
    ):

        # Check if mesh already exists
        file_name = meshes_dir / f"{node.identifier}.obj"
        if file_name.exists():
            meshes_dict[node.identifier] = file_name
            continue

        else:
            create_region_mesh(
                (
                    meshes_dir,
                    node,
                    tree,
                    labels,
                    annotated_volume,
                    ROOT_ID,
                    CLOSING_N_ITERS,
                    DECIMATE_FRACTION,
                    SMOOTH,
                )
            )
            meshes_dict[node.identifier] = file_name

    print(
        "Finished mesh extraction in : ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Once mesh creation is over, rescale
    for mesh_id, meshfile in meshes_dict.items():
        # Check if mesh is barrel-related
        if mesh_id in [s["id"] for s in dict_to_add]:

            try:
                mesh = mio.read(meshfile)
                mesh.points *= resolution
                mio.write(meshfile, mesh)
            except mio._exceptions.ReadError:
                print(f"Mesh file {meshfile} not found.")

    # Loop over structures, remove entries not used:
    for struct in structs_with_mesh:
        [
            struct.pop(k)
            for k in ["graph_id", "structure_set_ids", "graph_order"]
        ]

    # Remove problematic 545 structure
    if 545 in meshes_dict.keys():
        meshes_dict.pop(545)

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
        additional_metadata={"atlas_packager": ATLAS_PACKAGER},
    )

    return output_filename


if __name__ == "__main__":
    # Generated atlas path:
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)

    create_atlas(bg_root_dir, RES_UM)
