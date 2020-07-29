from pathlib import Path
import warnings
import nrrd
import requests
import tarfile
import tifffile
import numpy as np

from allensdk.core.structure_tree import StructureTree
from atlas_gen.wrapup import wrapup_atlas_from_data

from bg_atlasapi.utils import retrieve_over_http

# Specify information about the atlas:
RES_UM = 1
VERSION = 3
ATLAS_NAME = "mpin_zfish"
SPECIES = "Danio rerio"
ATLAS_LINK = "http://fishatlas.neuro.mpg.de"
CITATION = "Kunst et al 2019, https://doi.org/10.1016/j.neuron.2019.04.034"
ORIENTATION = "lai"


BASE_URL = r"https://fishatlas.neuro.mpg.de"


# Generated atlas path:
bg_root_dir = Path.home() / "brainglobe_workingdir" / "fish"
bg_root_dir.mkdir(exist_ok=True)


def add_path_inplace(parent):
    """ Recursively traverse hierarchy of regions and append for each region
    the full path of substructures in brainglobe standard list.

    Parameters
    ----------
    parent : dict
        node parsed from fishatlas website containing a "sub_regions" key

    """
    for ch in parent["sub_regions"]:
        new_root = parent["structure_id_path"] + [
            ch["id"],
        ]

        ch["structure_id_path"] = new_root

        add_path_inplace(ch)


def collect_all_inplace(
    node, traversing_list, download_path, mesh_dict,
):
    """ Recursively traverse a region hierarchy, download meshes, and append
    regions to a list inplace.

    Parameters
    ----------
    node
    traversing_list
    download_path
    mesh_dict


    """

    # Append clean dictionary with brainglobe standard info:
    traversing_list.append(
        {
            "name": node["name"],
            "acronym": node["name"],
            "id": node["id"],
            "rgb_triplet": StructureTree.hex_to_rgb(node["color"]),
            "structure_id_path": node["structure_id_path"],
        }
    )

    # Url for the mesh:
    mesh_url = (
        BASE_URL + node["files"]["file_3D"][:-4].replace("\\", "/") + ".stl"
    )

    # Try download, if mesh does not exist region is removed:
    try:
        filename = download_path / "{}.stl".format(node["id"])
        retrieve_over_http(mesh_url, filename)

        mesh_dict[node["id"]] = filename
    except requests.exceptions.ConnectionError:
        # Pop region from list:
        message = "No mesh found for {}".format(traversing_list.pop()["name"])
        warnings.warn(message)

    for region in node["sub_regions"]:
        collect_all_inplace(region, traversing_list, download_path, mesh_dict)


# Download reference:
#####################
reference_url = f"{BASE_URL}/media/brain_browser/Brain/MovieViewBrain/standard_brain_fixed_SYP_T_GAD1b.nrrd"
out_file_path = bg_root_dir / "reference.nrrd"
retrieve_over_http(reference_url, out_file_path)

# Swap axes b/c of nrrd reading
reference_stack, h = nrrd.read(str(out_file_path))
reference_stack = np.swapaxes(reference_stack, 0, 2)

# Download annotation and hemispheres from GIN repo:
gin_url = "https://gin.g-node.org/brainglobe/mpin_zfish/raw/master/mpin_zfish_annotations.tar.gz"
compressed_zip_path = bg_root_dir / "annotations.tar"
retrieve_over_http(gin_url, compressed_zip_path)

tar = tarfile.open(compressed_zip_path)
tar.extractall(path=bg_root_dir)

extracted_dir = bg_root_dir / "mpin_zfish_annotations"

annotation_stack = tifffile.imread(
    str(extracted_dir / "mpin_zfish_annotation.tif")
)

hemispheres_stack = tifffile.imread(
    str(extracted_dir / "mpin_zfish_hemispheres.tif")
)

# meshes from the website and stacks do not have the same orientation.
# Therefore, flip axes of the stacks so that bg-space reorientation is used on
# the meshes:
annotation_stack = annotation_stack.swapaxes(0, 2)
hemispheres_stack = hemispheres_stack.swapaxes(0, 2)
reference_stack = reference_stack.swapaxes(0, 2)


# Download structures tree and meshes:
######################################
regions_url = f"{BASE_URL}/neurons/get_brain_regions"

meshes_dir_path = bg_root_dir / "meshes_temp_download"
meshes_dir_path.mkdir(exist_ok=True)

# Download structures hierarchy:
structures = requests.get(regions_url).json()["brain_regions"]

# Initiate dictionary with root info:
structures_dict = {
    "name": "root",
    "id": 0,
    "sub_regions": structures.copy(),
    "structure_id_path": [0],
    "acronym": "root",
    "files": {
        "file_3D": "/media/Neurons_database/Brain_and_regions/Brains/Outline/Outline_new.txt"
    },
    "color": "#ffffff",
}

# Go through the regions hierarchy and create the structure path entry:
add_path_inplace(structures_dict)

# Create empty list and collect all regions traversing the regions hierarchy:
structures_list = []
meshes_dict = {}
collect_all_inplace(
    structures_dict, structures_list, meshes_dir_path, meshes_dict
)

# Wrap up, compress, and remove file:0
print("Finalising atlas")
wrapup_atlas_from_data(
    atlas_name=ATLAS_NAME,
    atlas_minor_version=VERSION,
    citation=CITATION,
    atlas_link=ATLAS_LINK,
    species=SPECIES,
    resolution=(RES_UM,) * 3,
    orientation=ORIENTATION,
    root_id=0,
    reference_stack=reference_stack,
    annotation_stack=annotation_stack,
    structures_list=structures_list,
    meshes_dict=meshes_dict,
    working_dir=bg_root_dir,
    hemispheres_stack=hemispheres_stack,
    cleanup_files=False,
    compress=True,
)
