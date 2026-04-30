"""Module containing property descriptors for brainglobe-atlasapi."""

import numpy as np

# Base url of the gin repository:
remote_url_base = "https://gin.g-node.org/brainglobe/atlases/raw/master/{}"
remote_url_s3 = "s3://brainglobe/atlas/{}"
remote_url_s3_http = "https://brainglobe.s3.us-west-2.amazonaws.com/atlas/{}"

# Major version of atlases used by current brainglobe-atlasapi release:
ATLAS_MAJOR_V = 0

# Supported resolutions:
RESOLUTION = ["nm", "um", "mm"]

# Entries and types from this template will be used to check atlas info
# consistency. Please keep updated both this and the function when changing
# the structure.
# If the atlas is unpublished, specify "unpublished" in the citation.
METADATA_TEMPLATE = {
    "name": "source_species_additional-info",
    "citation": "Someone et al 2020, https://doi.org/somedoi",
    "atlas_link": "http://www.example.com",
    "species": "Gen species",
    "symmetric": False,
    "resolution": [1.0, 1.0, 1.0],
    "orientation": "asr",
    "shape": [100, 50, 100],
    "version": "0.0",
    "additional_references": [],
}


# Template for a structure dictionary:
STRUCTURE_TEMPLATE = {
    "acronym": "root",
    "id": 997,
    "name": "root",
    "structure_id_path": [997],
    "rgb_triplet": [255, 255, 255],
}


# File and directory names for the atlas package:
METADATA_FILENAME = "metadata.json"
STRUCTURES_FILENAME = "structures.json"
REFERENCE_FILENAME = "reference.tiff"
ANNOTATION_FILENAME = "annotation.tiff"
HEMISPHERES_FILENAME = "hemispheres.tiff"
MESHES_DIRNAME = "meshes"

# V2 file names
V2_ATLAS_ROOTDIR = "atlases"
V2_ANNOTATION_ROOTDIR = "annotation-sets"
V2_COORDINATE_SPACE_ROOTDIR = "coordinate-spaces"
V2_TEMPLATE_ROOTDIR = "templates"
V2_TERMINOLOGY_ROOTDIR = "terminologies"
V2_TERMINOLOGY_NAME = "terminology.csv"
V2_MESHES_DIRECTORY = "annotation.precomputed"
V2_TEMPLATE_NAME = "anatomical_template.ome.zarr"
V2_ANNOTATION_NAME = "annotation.ome.zarr"
V2_HEMISPHERES_NAME = "hemispheres.ome.zarr"

# Types for the atlas stacks:
REFERENCE_DTYPE = np.uint16
ANNOTATION_DTYPE = np.uint32
HEMISPHERES_DTYPE = np.uint8

# Standard orientation origin: Anterior, Superior, Right
# (using brainglobe-space definition)
ATLAS_ORIENTATION = "asr"


def format_component_stub(
    component_name: str,
    component_version: str,
    component_root_dir: str,
    component_file_name: str,
) -> str:
    """
    Format the component stub for a given component name, version,
    root directory and file name.

    Parameters
    ----------
    component_name : str
        The name of the component (e.g., allen-mouse-adult-stpt-template).
    component_version : str
        The version of the component.
    component_root_dir : str
        The root directory of the component.
    component_file_name : str
        The name of the component file.

    Returns
    -------
    str
        The formatted component stub.
    """
    component_version = component_version.replace(".", "_")
    stub = (
        f"{component_root_dir}/{component_name}/"
        f"{component_version}/{component_file_name}"
    )
    return stub


def format_template_stub(template_name: str, version: str) -> str:
    """
    Format the template stub for a given template name and version.

    Parameters
    ----------
    template_name : str
        The name of the template (e.g., allen-adult-mouse-stpt-template).
    version : str
        The version of the template.

    Returns
    -------
    str
        The formatted template stub.
    """
    return format_component_stub(
        template_name, version, V2_TEMPLATE_ROOTDIR, V2_TEMPLATE_NAME
    )


def format_annotation_stub(annotation_name: str, version: str) -> str:
    """
    Format the annotation stub for a given annotation name and version.

    Parameters
    ----------
    annotation_name : str
        The name of the annotation (e.g., allen-adult-mouse-annotation).
    version : str
        The version of the annotation.

    Returns
    -------
    str
        The formatted annotation stub.
    """
    return format_component_stub(
        annotation_name, version, V2_ANNOTATION_ROOTDIR, V2_ANNOTATION_NAME
    )


def format_hemispheres_stub(annotation_name: str, version: str) -> str:
    """
    Format the hemispheres stub for a given hemispheres name and version.

    Parameters
    ----------
    annotation_name : str
        The name of the annotation (e.g., allen-adult-mouse-annotation).
    version : str
        The version of the annotation.

    Returns
    -------
    str
        The formatted hemispheres stub.
    """
    return format_component_stub(
        annotation_name, version, V2_ANNOTATION_ROOTDIR, V2_HEMISPHERES_NAME
    )


def format_terminology_stub(terminology_name: str, version: str) -> str:
    """
    Format the terminology stub for a given terminology name and version.

    Parameters
    ----------
    terminology_name : str
        The name of the terminology (e.g., allen-adult-mouse-terminology).
    version : str
        The version of the terminology.

    Returns
    -------
    str
        The formatted terminology stub.
    """
    return format_component_stub(
        terminology_name, version, V2_TERMINOLOGY_ROOTDIR, V2_TERMINOLOGY_NAME
    )


def format_meshes_stub(annotation_name: str, version: str) -> str:
    """
    Format the meshes stub for a given annotation name and version.

    Parameters
    ----------
    annotation_name : str
        The name of the annotation (e.g., allen-adult-mouse-annotation).
    version : str
        The version of the annotation.

    Returns
    -------
    str
        The formatted meshes stub.
    """
    return format_component_stub(
        annotation_name, version, V2_ANNOTATION_ROOTDIR, V2_MESHES_DIRECTORY
    )
