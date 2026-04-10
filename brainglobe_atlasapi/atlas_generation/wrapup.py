"""Tools to finalise the atlas creation process."""

import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import brainglobe_space as bgs
import meshio as mio
import numpy as np
import numpy.typing as npt
import pandas as pd
import tifffile
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import write_multiscale

import brainglobe_atlasapi.atlas_generation
from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    generate_metadata_dict,
)
from brainglobe_atlasapi.atlas_generation.stacks import (
    save_annotation,
    save_hemispheres,
    save_template,
)
from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
)
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    get_all_validation_functions,
)
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.structure_tree_util import get_structures_tree
from brainglobe_atlasapi.utils import atlas_name_from_repr

# This should be changed every time we make changes in the atlas
# structure:
ATLAS_VERSION = brainglobe_atlasapi.atlas_generation.__version__


def filter_structures_not_present_in_annotation(structures, annotation):
    """
    Filter out structures not present in the annotation volume.

    This function removes structures from the provided list that are
    not found in the annotation volume, or whose children are also
    not present. It also prints the names and IDs of the removed structures.

    Parameters
    ----------
    structures : list of dict
        A list of dictionaries, where each dictionary contains information
        about a brain structure (e.g., ID, name, parent information).
    annotation : np.ndarray
        The annotation volume (3D NumPy array) where each voxel contains
        a structure ID.

    Returns
    -------
    list of dict
        A new list containing only the structure dictionaries that are
        present in the annotation volume or have descendants present.
    """
    present_ids = set(np.unique(annotation))
    # Create a structure tree for easy parent-child relationship traversal
    tree = get_structures_tree(structures)

    # Function to check if a structure or any of its descendants are present
    def is_present(structure_id):
        if structure_id in present_ids:
            return True
        # Recursively check all descendants
        for child_node in tree.children(structure_id):
            if is_present(child_node.identifier):
                return True
        return False

    removed = [s for s in structures if not is_present(s["id"])]
    for r in removed:
        print("Removed structure:", r["name"], "(ID:", r["id"], ")")

    return [s for s in structures if is_present(s["id"])]


def standardize_resolution(
    resolution: Tuple[int | float] | List[Tuple[int | float]],
) -> List[Tuple[int | float]]:
    """
    Standardize resolution input to a list of tuples.

    This function takes a resolution input that can be either a single tuple
    (for a single scale) or a list of tuples (for multiple scales) and
    standardizes it to always return a list of tuples. If the input is a
    single tuple, it will be wrapped in a list.

    Parameters
    ----------
    resolution : Tuple[int | float] | List[Tuple[int | float]]
        The resolution input, either a single tuple or a list of tuples.

    Returns
    -------
    List[Tuple[int | float]]
        A standardized list of tuples with the resolution at each scale.
    """
    if isinstance(resolution, tuple):
        return [resolution]
    elif isinstance(resolution, list):
        return resolution
    else:
        raise ValueError(
            "Resolution must be either a tuple or a list of tuples."
        )


def write_multiscale_ome_zarr(
    images: List[npt.NDArray],
    output_path: Path,
    axes: List[dict],
    transformations: List[List[dict]],
):
    """
    Write a multiscale OME Zarr file with the given images, and metadata.

    Parameters
    ----------
    images : List[npt.NDArray]
        A list of NumPy arrays representing the image data at different scales.
    output_path : Path
        The file path where the OME Zarr file will be saved.
    axes : List[dict]
        A list of dictionaries describing the axes of the image data.
    transformations : List[List[dict]]
        A set of dictionaries describing the transformations per scale.
    """
    zarr_loc = parse_url(output_path, mode="w")
    assert zarr_loc is not None
    store = zarr_loc.store
    root = zarr.group(store=store)

    write_multiscale(
        pyramid=images,
        group=root,
        axes=axes,
        coordinate_transformations=transformations,
    )


def wrapup_atlas_from_data(
    atlas_name,
    atlas_minor_version,
    citation,
    atlas_link,
    species,
    resolution,
    orientation,
    root_id,
    reference_stack,
    annotation_stack,
    structures_list,
    meshes_dict,
    working_dir,
    atlas_packager=None,
    hemispheres_stack=None,
    template_info: Optional[Tuple[str, str]] = None,
    annotation_info: Optional[Tuple[str, str]] = None,
    terminology_info: Optional[Tuple[str, str]] = None,
    coordinate_space_info: Optional[Tuple[str, str]] = None,
    scale_meshes=True,
    resolution_mapping=None,
    additional_references=[],
    additional_metadata={},
    overwrite=False,
):
    """
    Finalise an atlas with truly consistent format from all the data.

    Parameters
    ----------
    atlas_name : str
        Atlas name in the form author_species.
    atlas_minor_version : int or str
        Minor version number for this particular atlas.
    citation : str
        Citation for the atlas, if unpublished specify "unpublished".
    atlas_link : str
        Valid URL for the atlas.
    species : str
        Species name formatted as "CommonName (Genus species)".
    resolution : tuple
        Three elements tuple, resolution on three axes
    orientation :
        Orientation of the original atlas
        (tuple describing origin for BGSpace).
    root_id :
        Id of the root element of the atlas.
    reference_stack : str or Path or numpy array
        Reference stack for the atlas.
        If str or Path, will be read with tifffile.
    annotation_stack : str or Path or numpy array
        Annotation stack for the atlas.
        If str or Path, will be read with tifffile.
    structures_list : list of dict
        List of valid dictionary for structures.
    meshes_dict : dict
        dict of meshio-compatible mesh file paths in the form
        {struct_id: meshpath}
    working_dir : str or Path obj
        Path where the atlas folder and compressed file will be generated.
    atlas_packager : str or None
        Credit for those responsible for converting the atlas
        into the BrainGlobe format.
    hemispheres_stack : str or Path or numpy array, optional
        Hemisphere stack for the atlas.
        If str or Path, will be read with tifffile.
        If none is provided, atlas is assumed to be symmetric.
    scale_meshes: bool, optional
        (Default values = False).
        If True the meshes points are scaled by the resolution
        to ensure that they are specified in microns,
        regardless of the atlas resolution.
    resolution_mapping: list, optional
        a list of three mapping the target space axes to the source axes
        only needed for mesh scaling of anisotropic atlases
    additional_references: List[Tuple[Dict, npt.NDArray]], optional
        (Default value = empty list).
        List of tuples containing metadata and arrays for secondary templates.
    additional_metadata: dict, optional
        (Default value = empty dict).
        Additional metadata to write to metadata.json
    overwrite : bool, optional
        (Default value = False).
        If True, will overwrite existing atlas directory.
        If False and atlas directory exists, raises FileExistsError.
    """
    working_dir = Path(working_dir) / "brainglobe-atlasapi"
    atlas_version = f"{ATLAS_VERSION}.{atlas_minor_version}"
    atlas_version_underscore = atlas_version.replace(".", "_")

    atlas_name_with_res = f"{atlas_name}_{resolution[0]}um"

    atlas_location = (
        f"/{descriptors.V2_ATLAS_ROOTDIR}/"
        f"{atlas_name_with_res}/{atlas_version_underscore}"
    )
    atlas_dir = Path(working_dir) / atlas_location.strip("/")

    if annotation_info is not None:
        annotation_name, annotation_version = annotation_info
    else:
        annotation_name = f"{atlas_name}-annotation"
        annotation_version = atlas_version

    if template_info is not None:
        template_name, template_version = template_info
    else:
        template_name = f"{atlas_name}-template"
        template_version = atlas_version

    if terminology_info is not None:
        terminology_name, terminology_version = terminology_info
    else:
        terminology_name = f"{atlas_name}-terminology"
        terminology_version = atlas_version

    if coordinate_space_info is not None:
        coordinate_space_name, coordinate_space_version = coordinate_space_info
    else:
        coordinate_space_name = f"{atlas_name}-coordinate-space"
        coordinate_space_version = atlas_version

    template_metadata = {
        "name": template_name,
        "version": template_version,
        "location": f"/{descriptors.V2_TEMPLATE_ROOTDIR}/"
        f"{template_name}/"
        f"{template_version.replace('.', '_')}",
    }

    terminology_metadata = {
        "name": terminology_name,
        "version": terminology_version,
        "location": f"/{descriptors.V2_TERMINOLOGY_ROOTDIR}/"
        f"{terminology_name}/"
        f"{terminology_version.replace('.', '_')}",
    }

    annotation_metadata = {
        "name": annotation_name,
        "version": annotation_version,
        "location": f"/{descriptors.V2_ANNOTATION_ROOTDIR}/"
        f"{annotation_name}/"
        f"{annotation_version.replace('.', '_')}",
        "template": template_metadata,
        "terminology": terminology_metadata,
    }

    coordinate_space_metadata = {
        "name": coordinate_space_name,
        "version": coordinate_space_version,
        "location": f"/{descriptors.V2_COORDINATE_SPACE_ROOTDIR}/"
        f"{coordinate_space_name}/"
        f"{coordinate_space_version.replace('.', '_')}",
        "template": template_metadata,
    }

    resolution_standard = standardize_resolution(resolution)

    transformations = [
        [{"type": "scale", "scale": [res / 1000 for res in res_tuple]}]
        for res_tuple in resolution_standard
    ]

    if atlas_dir.exists():
        if overwrite:
            print(f"Atlas directory already exists, overwriting: {atlas_dir}")
            shutil.rmtree(atlas_dir)
        else:
            raise FileExistsError(
                f"Atlas output already exists at {atlas_dir}. "
                "Try setting overwrite=True"
            )

    # exist_ok would be more permissive but error-prone here as there might
    # be old files
    atlas_dir.mkdir(parents=True)

    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = hemispheres_stack is None

    if isinstance(annotation_stack, str) or isinstance(annotation_stack, Path):
        annotation_stack = tifffile.imread(annotation_stack)

    if isinstance(reference_stack, str) or isinstance(reference_stack, Path):
        reference_stack = tifffile.imread(reference_stack)

    structures_list = filter_structures_not_present_in_annotation(
        structures_list, annotation_stack
    )

    # Instantiate BGSpace obj, using original stack size in um as meshes
    # are un um:
    original_shape = reference_stack.shape
    volume_shape = tuple(res * s for res, s in zip(resolution, original_shape))
    space_convention = bgs.AnatomicalSpace(orientation, shape=volume_shape)

    # Check consistency of structures:
    check_struct_consistency(structures_list)

    stack_list = [reference_stack, annotation_stack]
    saving_fun_list = [save_template, save_annotation]
    stack_metadata_list = [template_metadata, annotation_metadata]

    # write OME Zarr stacks:
    for stack, saving_function, metadata in zip(
        stack_list, saving_fun_list, stack_metadata_list
    ):
        # Reorient stacks if required:
        stack = space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        shape = stack.shape

        dest_dir = working_dir / metadata["location"].lstrip("/")
        if dest_dir.exists():
            print(
                f"{metadata['name']} directory already exists, "
                f"skipping stack saving: {dest_dir}"
            )
        else:
            saving_function(stack, dest_dir, transformations)

    if hemispheres_stack is None:
        # initialize empty stack:
        hemispheres_stack = np.full(shape, 2, dtype=np.uint8)

        # Fill out with 2s the right hemisphere:
        slices = [slice(None) for _ in range(3)]
        slices[2] = slice(round(hemispheres_stack.shape[2] / 2), None)
        hemispheres_stack[tuple(slices)] = 1

        dest_dir = (
            working_dir
            / annotation_metadata["location"].lstrip("/")
            / descriptors.V2_HEMISPHERES_NAME
        )

        if dest_dir.exists():
            print(
                f"Hemispheres directory already exists, "
                f"skipping hemispheres stack saving: {dest_dir}"
            )
        else:
            save_hemispheres(
                hemispheres_stack, dest_dir.parent, transformations
            )

    for ref_tuple in additional_references:
        ref_metadata, stack = ref_tuple
        stack = space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        dest_dir = working_dir / ref_metadata["location"].lstrip("/")

        if dest_dir.exists():
            print(
                f"{ref_metadata['name']} directory already exists, "
                f"skipping stack saving: {dest_dir}"
            )
        else:
            save_template(stack, dest_dir, transformations)

    # Reorient vertices of the mesh.
    mesh_dest_dir = (
        working_dir
        / annotation_metadata["location"].lstrip("/")
        / descriptors.V2_MESHES_DIRECTORY
    )
    if mesh_dest_dir.exists():
        print(
            f"Mesh directory already exists, "
            f"skipping mesh saving: {mesh_dest_dir}"
        )
    else:
        mesh_dest_dir.mkdir(parents=True)

        for mesh_id, meshfile in meshes_dict.items():
            mesh = mio.read(meshfile)

            if scale_meshes:
                # Scale the mesh to the desired resolution, BEFORE transforming
                # Note that this transformation happens in original space,
                # but the resolution is passed in target space (typically ASR)
                if not resolution_mapping:
                    # isotropic case, so don't need to re-map resolution
                    mesh.points *= resolution
                else:
                    # resolution needs to be transformed back
                    # to original space in anisotropic case
                    original_resolution = (
                        resolution[resolution_mapping[0]],
                        resolution[resolution_mapping[1]],
                        resolution[resolution_mapping[2]],
                    )
                    mesh.points *= original_resolution

            # Reorient points:
            mesh.points = space_convention.map_points_to(
                descriptors.ATLAS_ORIENTATION, mesh.points
            )

            # Save in meshes dir:
            # TODO: parallelise and copy if not scaling or reorienting
            mio.write(
                mesh_dest_dir / f"{mesh_id}", mesh, file_format="neuroglancer"
            )

    # save regions list json:
    with open(dest_dir / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures_list, f)

    # Terminology
    terminology_path = working_dir / terminology_metadata["location"].strip(
        "/"
    )
    if terminology_path.exists():
        print(
            f"Terminology directory already exists, "
            f"skipping terminology saving: {terminology_path}"
        )
    else:
        terminology_path.mkdir(parents=True)
        terminology_path = terminology_path / "terminology.csv"

        structures_df = pd.DataFrame(structures_list)
        terminology_df = pd.DataFrame()

        terminology_df["identifier"] = structures_df["id"].astype(np.uint32)
        terminology_df["parent_identifier"] = (
            structures_df["structure_id_path"]
            .apply(lambda x: x[-2] if len(x) > 1 else None)
            .astype(pd.UInt32Dtype())
        )
        terminology_df["annotation_value"] = structures_df["id"].astype(
            np.uint32
        )
        terminology_df["name"] = structures_df["name"].astype(pd.StringDtype())
        terminology_df["abbreviation"] = structures_df["acronym"].astype(
            pd.StringDtype()
        )
        terminology_df["color_hex_triplet"] = structures_df[
            "rgb_triplet"
        ].apply(lambda x: "".join(f"{c:02X}" for c in x))
        terminology_df["color_hex_triplet"] = "#" + terminology_df[
            "color_hex_triplet"
        ].astype(pd.StringDtype())
        terminology_df["root_identifier_path"] = structures_df[
            "structure_id_path"
        ]

        terminology_df.to_csv(terminology_path, index=False)

    # Write coordinate space metadata
    coordinate_space_path = working_dir / coordinate_space_metadata[
        "location"
    ].strip("/")

    if coordinate_space_path.exists():
        print(
            f"Coordinate space directory already exists, skipping "
            f"coordinate space metadata saving: {coordinate_space_path}"
        )
    else:
        coordinate_space_path.mkdir(parents=True)
        coordinate_space_metadata_path = (
            coordinate_space_path / "manifest.json"
        )

        with open(coordinate_space_metadata_path, "w") as f:
            json.dump(coordinate_space_metadata, f, indent=4)

    # Finalize metadata dictionary:
    metadata_dict = generate_metadata_dict(
        name=atlas_name,
        location=atlas_location,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,  # We expect input to be asr
        orientation=descriptors.ATLAS_ORIENTATION,  # Pass orientation "asr"
        version=f"{ATLAS_VERSION}.{atlas_minor_version}",
        shape=shape,
        additional_references=[
            add_ref[0] for add_ref in additional_references
        ],
        atlas_packager=atlas_packager,
        coordinate_space_metadata=coordinate_space_metadata,
        terminology_metadata=terminology_metadata,
        annotation_set_metadata=annotation_metadata,
        template_metadata=template_metadata,
    )

    with open(atlas_dir / "manifest.json", "w") as f:
        json.dump(metadata_dict, f, indent=4)

    atlas_name_for_validation = atlas_name_from_repr(atlas_name, resolution[0])

    # creating BrainGlobe object from local folder (working_dir)
    atlas_to_validate = BrainGlobeAtlas(
        atlas_name=atlas_name_for_validation,
        brainglobe_dir=working_dir.parent,
        check_latest=False,
    )

    # Run validation functions
    print(f"Running atlas validation on {atlas_location}")

    validation_results = {}

    for func in get_all_validation_functions():
        try:
            func(atlas_to_validate)
            validation_results[func.__name__] = "Pass"
        except AssertionError as e:
            validation_results[func.__name__] = f"Fail: {str(e)}"

    def _check_validations(validation_results):
        # Helper function to check if all validations passed
        all_passed = all(
            result == "Pass" for result in validation_results.values()
        )

        if all_passed:
            print("This atlas is valid")
        else:
            failed_functions = [
                func
                for func, result in validation_results.items()
                if result != "Pass"
            ]
            error_messages = [
                result.split(": ")[1]
                for result in validation_results.values()
                if result != "Pass"
            ]

            print("These validation functions have failed:")
            for func, error in zip(failed_functions, error_messages):
                print(f"- {func}: {error}")

    _check_validations(validation_results)

    return atlas_dir
