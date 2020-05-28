from brainatlas_api.utils import retrieve_over_http
import trimesh
from brainatlas_api.structures import StructureTree
import requests
import warnings

BASE_URL = r"https://fishatlas.neuro.mpg.de"


def download_convert_mesh(
    url,
    interm_file_path,
    obj_file_path,
    refstack_axes,
    refstack_flips,
    ref_shape,
    cleanup=True,
):
    """
    Parameters
    ----------
    url : str
        mesh url for download
    interm_file_path : Path obj
        path of the intermediate .stl mesh
    obj_file_path : Path obj
        path of the final .obj object
    cleanup : bool (default True)
        if True, intermediate file is unlinked

    Returns
    -------

    """
    retrieve_over_http(url, interm_file_path)

    mesh = trimesh.load(interm_file_path)
    mesh.vertices = mesh.vertices[:, refstack_axes]
    for i, (f, size) in enumerate(zip(refstack_flips, ref_shape)):
        if f:
            mesh.vertices[:, i] = size - mesh.vertices[:, i]

    mesh.export(obj_file_path)

    if cleanup:
        interm_file_path.unlink()


def add_path_inplace(parent):
    """ Recursively traverse hierarchy of regions and append for each region
    the full path of substructures in brainglobe standard list.
    Parameters
    ----------
    parent : dict
        node parsed from fishatlas website containing a "sub_regions" key;

    Returns
    -------

    """
    for ch in parent["sub_regions"]:
        new_root = parent["structure_id_path"] + [
            parent["id"],
        ]

        ch["structure_id_path"] = new_root

        add_path_inplace(ch)


def collect_all_inplace(
    node,
    traversing_list,
    download_path,
    refstack_axes,
    refstack_flips,
    ref_shape,
):
    """ Recursively traverse a region hierarchy, download meshes, and append
    regions to a list inplace.

    Parameters
    ----------
    node
    traversing_list
    download_path

    Returns
    -------

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
        download_convert_mesh(
            mesh_url,
            download_path / "mesh.stl",
            download_path / "{}.obj".format(node["id"]),
            refstack_axes,
            refstack_flips,
            ref_shape,
        )
    except requests.exceptions.ConnectionError:
        # Pop region from list:
        message = "No mesh found for {}".format(traversing_list.pop()["name"])
        warnings.warn(message)

    for region in node["sub_regions"]:
        collect_all_inplace(
            region,
            traversing_list,
            download_path,
            refstack_axes,
            refstack_flips,
            ref_shape,
        )
