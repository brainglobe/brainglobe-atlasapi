"""Tools to finalise the atlas creation process."""

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import brainglobe_space as bgs
import meshio as mio
import ngff_zarr as nz
import numpy as np
import numpy.typing as npt
import pandas as pd
import s3fs
import tifffile

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


def check_requested_component(
    component_info: Tuple[str, str, bool],
    local_path: Path,
    component_root: str,
    component_file_name: str,
) -> Tuple[str, str, bool]:
    """
    Check if a requested component already exists remotely and fetch metadata.

    This function checks if a component (e.g., annotation, template) with the
    specified name and version already exists in the remote storage.
    If it exists, it fetches the OME-Zarr metadata files for that component
    and saves them locally. It returns the component name, version,
    and a boolean indicating whether to skip saving the component data.

    Parameters
    ----------
    component_info : Tuple[str, str, bool]
        A tuple containing the component name, version, and a boolean for
        whether the component is published.
    component_root : str
        The root directory in the remote storage where the component is stored.
    component_file_name : str
        The name of the component file (e.g., "anatomical_template.ome.zarr").
    local_path : Path
        The local directory where the component metadata should be saved.

    Returns
    -------
    Tuple[str, str, bool]
        A tuple containing the component name, version, and a boolean for
        whether to skip saving the component data.
    """
    fs = s3fs.S3FileSystem(anon=True)
    component_name, component_version, _ = component_info

    if component_info[2]:
        remote_path = (
            f"{component_root}/"
            f"{component_name}/"
            f"{component_version.replace('.', '_')}/"
            f"{component_file_name}"
        )

        if not fs.exists(descriptors.remote_url_s3.format(remote_path)):
            raise FileNotFoundError(
                f"{component_name} version {component_version} "
                f"not found at {remote_path}"
            )
        else:
            skip_saving = True
            # Add wildcard to fetch all OME-Zarr metadata files
            if component_file_name.endswith(".ome.zarr"):
                remote_path += "/**/*.json"

            local_component_path = (
                local_path
                / component_root
                / component_name
                / component_version.replace(".", "_")
                / component_file_name
            )

            local_component_path.parent.mkdir(parents=True, exist_ok=True)
            fs.get(
                descriptors.remote_url_s3.format(remote_path),
                local_component_path,
            )
    else:
        skip_saving = False

    return component_name, component_version, skip_saving


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


def _resolve_component_info(
    component_info: Optional[Tuple[str, str, bool]],
    working_dir: Path,
    root_dir: str,
    file_name: str,
    default_name: str,
    atlas_version: str,
) -> Tuple[str, str, bool]:
    if component_info is not None:
        component_info = check_requested_component(
            component_info, working_dir, root_dir, file_name
        )
        return component_info[0], component_info[1], component_info[2]
    return default_name, atlas_version, False


def _make_component_metadata(
    name: str,
    version: str,
    root_dir: str,
    extra: Optional[dict] = None,
) -> dict:
    metadata = {
        "name": name,
        "version": version,
        "location": f"/{root_dir}/{name}/{version.replace('.', '_')}",
    }
    if extra:
        metadata.update(extra)
    return metadata


def _save_if_not_exists(
    stacks: List[npt.NDArray],
    dest_dir: Path,
    label: str,
    transformations: List[List[dict]],
    save_fn: Callable[[npt.NDArray, Path, List[List[dict]]], None],
) -> None:
    if dest_dir.exists():
        print(f"{label} directory already exists, skipping: {dest_dir}")
    else:
        save_fn(stacks, dest_dir, transformations)


def _check_validations(validation_results: dict) -> None:
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


def wrapup_atlas_from_data(
    atlas_name: str,
    atlas_minor_version: Union[int, str],
    citation: str,
    atlas_link: str,
    species: str,
    resolution: Tuple[int | float] | List[Tuple[int | float]],
    orientation: str,
    root_id: int,
    reference_stack: str | Path | npt.NDArray | List[str | Path | npt.NDArray],
    annotation_stack: (
        str | Path | npt.NDArray | List[str | Path | npt.NDArray]
    ),
    structures_list: List[Dict],
    meshes_dict: Dict[int | str, str | Path],
    working_dir: str | Path,
    atlas_packager=None,
    hemispheres_stack=None,
    template_info: Optional[Tuple[str, str, bool]] = None,
    annotation_info: Optional[Tuple[str, str, bool]] = None,
    terminology_info: Optional[Tuple[str, str, bool]] = None,
    coordinate_space_info: Optional[Tuple[str, str, bool]] = None,
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
    reference_stack : str | Path | npt.NDArray | List[str | Path | npt.NDArray]
        Reference stack for the atlas.
        If str or Path, will be read with tifffile.
        If list, should be list of stacks for each scale, ordered from highest
        to lowest resolution.
    annotation_stack : str | Path | npt.NDArray | List[str | Path | npt.NDArray]
        Annotation stack for the atlas.
        If str or Path, will be read with tifffile.
        If list, should be list of stacks for each scale, ordered from highest
        to lowest resolution.
    structures_list : List[Dict]
        List of valid dictionary for structures.
    meshes_dict : Dict[int | str, str | Path]
        dict of meshio-compatible mesh file paths in the form
        {struct_id: meshpath}
    working_dir : str or Path obj
        Path where the atlas folder and compressed file will be generated.
    atlas_packager : str or None
        Credit for those responsible for converting the atlas
        into the BrainGlobe format.
    hemispheres_stack : str | Path | npt.NDArray | List[str | Path | npt.NDArray] | None, optional
        Hemisphere stack for the atlas.
        If str or Path, will be read with tifffile.
        If list, should be list of stacks for each scale, ordered from highest
        to lowest resolution.
        If none is provided, atlas is assumed to be symmetric.
    scale_meshes: bool, optional
        (Default values = False).
        If True the meshes points are scaled by the resolution
        to ensure that they are specified in microns,
        regardless of the atlas resolution.
    resolution_mapping: List[int], optional
        a list of three mapping the target space axes to the source axes
        only needed for mesh scaling of anisotropic atlases
    additional_references: List[Tuple[Dict, str | Path | npt.NDArray | List[str | Path | npt.NDArray]]], optional
        (Default value = empty list).
        List of tuples containing metadata and arrays for secondary templates.
    additional_metadata: dict, optional
        (Default value = empty dict).
        Additional metadata to write to metadata.json
    overwrite : bool, optional
        (Default value = False).
        If True, will overwrite existing atlas directory.
        If False and atlas directory exists, raises FileExistsError.
    """  # noqa: E501
    working_dir = Path(working_dir) / "brainglobe-atlasapi"
    atlas_version = f"{ATLAS_VERSION}.{atlas_minor_version}"
    atlas_version_underscore = atlas_version.replace(".", "_")

    annotation_name, annotation_version, skip_annotation_saving = (
        _resolve_component_info(
            annotation_info,
            working_dir,
            descriptors.V2_ANNOTATION_ROOTDIR,
            descriptors.V2_ANNOTATION_NAME,
            f"{atlas_name}-annotation",
            atlas_version,
        )
    )

    template_name, template_version, skip_template_saving = (
        _resolve_component_info(
            template_info,
            working_dir,
            descriptors.V2_TEMPLATE_ROOTDIR,
            descriptors.V2_TEMPLATE_NAME,
            f"{atlas_name}-template",
            atlas_version,
        )
    )

    terminology_name, terminology_version, skip_terminology_saving = (
        _resolve_component_info(
            terminology_info,
            working_dir,
            descriptors.V2_TERMINOLOGY_ROOTDIR,
            descriptors.V2_TERMINOLOGY_NAME,
            f"{atlas_name}-terminology",
            atlas_version,
        )
    )

    (
        coordinate_space_name,
        coordinate_space_version,
        skip_coordinate_space_saving,
    ) = _resolve_component_info(
        coordinate_space_info,
        working_dir,
        descriptors.V2_COORDINATE_SPACE_ROOTDIR,
        "manifest.json",
        f"{atlas_name}-coordinate-space",
        atlas_version,
    )

    template_metadata = _make_component_metadata(
        template_name, template_version, descriptors.V2_TEMPLATE_ROOTDIR
    )
    terminology_metadata = _make_component_metadata(
        terminology_name,
        terminology_version,
        descriptors.V2_TERMINOLOGY_ROOTDIR,
    )
    annotation_metadata = _make_component_metadata(
        annotation_name,
        annotation_version,
        descriptors.V2_ANNOTATION_ROOTDIR,
        extra={
            "template": template_metadata,
            "terminology": terminology_metadata,
        },
    )
    coordinate_space_metadata = _make_component_metadata(
        coordinate_space_name,
        coordinate_space_version,
        descriptors.V2_COORDINATE_SPACE_ROOTDIR,
        extra={"template": template_metadata},
    )

    resolution_standard = standardize_resolution(resolution)

    transformations = [
        [{"type": "scale", "scale": [res / 1000 for res in res_tuple]}]
        for res_tuple in resolution_standard
    ]

    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = hemispheres_stack is None

    if isinstance(annotation_stack, str) or isinstance(annotation_stack, Path):
        annotation_stack = [tifffile.imread(annotation_stack)]
    elif isinstance(annotation_stack, npt.NDArray):
        annotation_stack = [annotation_stack]

    if isinstance(reference_stack, str) or isinstance(reference_stack, Path):
        reference_stack = [tifffile.imread(reference_stack)]
    elif isinstance(reference_stack, npt.NDArray):
        reference_stack = [reference_stack]

    structures_list = filter_structures_not_present_in_annotation(
        structures_list, annotation_stack[0]
    )

    # Instantiate BGSpace obj, using original stack size in um as meshes
    # are un um:
    original_shape = reference_stack[0].shape
    volume_shape = tuple(
        res * s for res, s in zip(resolution_standard[0], original_shape)
    )
    space_convention = bgs.AnatomicalSpace(orientation, shape=volume_shape)

    # Check consistency of structures:
    check_struct_consistency(structures_list)

    # Write template:
    if not skip_template_saving:
        # Reorient stacks if required:
        reference_stack = [
            space_convention.map_stack_to(
                descriptors.ATLAS_ORIENTATION, stack, copy=False
            )
            for stack in reference_stack
        ]

        shapes = [stack.shape for stack in reference_stack]
        dest_dir = working_dir / template_metadata["location"].lstrip("/")

        _save_if_not_exists(
            reference_stack,
            dest_dir,
            template_metadata["name"],
            transformations,
            save_template,
        )
    else:
        multiscale = nz.from_ngff_zarr(
            working_dir
            / template_metadata["location"].lstrip("/")
            / descriptors.V2_TEMPLATE_NAME
        )

        shapes = [image.data.shape for image in multiscale.images]

    # Write annotation:
    if not skip_annotation_saving:
        # Reorient stacks if required:
        annotation_stack = [
            space_convention.map_stack_to(
                descriptors.ATLAS_ORIENTATION, stack, copy=False
            )
            for stack in annotation_stack
        ]

        shapes = [stack.shape for stack in annotation_stack]
        dest_dir = working_dir / annotation_metadata["location"].lstrip("/")

        _save_if_not_exists(
            annotation_stack,
            dest_dir,
            annotation_metadata["name"],
            transformations,
            save_annotation,
        )

        if hemispheres_stack is None:
            # initialize empty stack:
            hemispheres_stack = [
                np.full(shape, 2, dtype=np.uint8) for shape in shapes
            ]

            # Fill out with 2s the right hemisphere:
            slices = [slice(None) for _ in range(3)] * len(annotation_stack)
            for stack in hemispheres_stack:
                slices[2] = slice(round(stack.shape[2] / 2), None)
                stack[tuple(slices)] = 1

            dest_dir = (
                working_dir
                / annotation_metadata["location"].lstrip("/")
                / descriptors.V2_HEMISPHERES_NAME
            )

            _save_if_not_exists(
                hemispheres_stack,
                dest_dir,
                annotation_metadata["name"],
                transformations,
                save_hemispheres,
            )

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
                    # Scale the mesh to the desired resolution,
                    # BEFORE transforming. Note that this transformation
                    # happens in original space, but the resolution is passed
                    # in target space (typically ASR)
                    if not resolution_mapping:
                        # isotropic case, so don't need to re-map resolution
                        mesh.points *= resolution_standard[0]
                    else:
                        # resolution needs to be transformed back
                        # to original space in anisotropic case
                        original_resolution = (
                            resolution_standard[0][resolution_mapping[0]],
                            resolution_standard[0][resolution_mapping[1]],
                            resolution_standard[0][resolution_mapping[2]],
                        )
                        mesh.points *= original_resolution

                # Reorient points:
                mesh.points = space_convention.map_points_to(
                    descriptors.ATLAS_ORIENTATION, mesh.points
                )

                # Save in meshes dir:
                # TODO: parallelise and copy if not scaling or reorienting
                mio.write(
                    mesh_dest_dir / f"{mesh_id}",
                    mesh,
                    file_format="neuroglancer",
                )

    additional_references_metadata = []
    for ref_tuple in additional_references:
        ref_metadata, additional_stack = ref_tuple
        if isinstance(additional_stack, str) or isinstance(
            additional_stack, Path
        ):
            additional_stack = [tifffile.imread(additional_stack)]
        elif isinstance(additional_stack, npt.NDArray):
            additional_stack = [additional_stack]

        additional_stack = [
            space_convention.map_stack_to(
                descriptors.ATLAS_ORIENTATION, stack, copy=False
            )
            for stack in additional_stack
        ]
        if isinstance(ref_metadata, str):
            ref_name = f"{atlas_name}-{ref_metadata}-template"
            ref_metadata = _make_component_metadata(
                ref_name, atlas_version, descriptors.V2_TEMPLATE_ROOTDIR
            )

        additional_references_metadata.append(ref_metadata)

        dest_dir = working_dir / ref_metadata["location"].lstrip("/")

        _save_if_not_exists(
            additional_stack,
            dest_dir,
            ref_metadata["name"],
            transformations,
            save_template,
        )

    if not skip_terminology_saving:
        terminology_path = working_dir / terminology_metadata[
            "location"
        ].strip("/")
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

            terminology_df["identifier"] = structures_df["id"].astype(
                np.uint32
            )
            terminology_df["parent_identifier"] = (
                structures_df["structure_id_path"]
                .apply(lambda x: x[-2] if len(x) > 1 else None)
                .astype(pd.UInt32Dtype())
            )
            terminology_df["annotation_value"] = structures_df["id"].astype(
                np.uint32
            )
            terminology_df["name"] = structures_df["name"].astype(
                pd.StringDtype()
            )
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

    if not skip_coordinate_space_saving:
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

    for resolution, shape in zip(resolution_standard, shapes):
        # Finalize metadata dictionary:
        atlas_name_with_res = f"{atlas_name}_{resolution[0]}um"

        atlas_location = (
            f"/{descriptors.V2_ATLAS_ROOTDIR}/"
            f"{atlas_name_with_res}/{atlas_version_underscore}"
        )
        atlas_dir = Path(working_dir) / atlas_location.strip("/")

        if atlas_dir.exists():
            if overwrite:
                print(
                    f"Atlas directory already exists, overwriting: {atlas_dir}"
                )
                shutil.rmtree(atlas_dir)
            else:
                raise FileExistsError(
                    f"Atlas output already exists at {atlas_dir}. "
                    "Try setting overwrite=True"
                )

        # exist_ok would be more permissive but error-prone here as there might
        # be old files
        atlas_dir.mkdir(parents=True)

        metadata_dict = generate_metadata_dict(
            name=atlas_name_with_res,
            location=atlas_location,
            citation=citation,
            atlas_link=atlas_link,
            species=species,
            symmetric=symmetric,
            resolution=resolution,  # We expect input to be asr
            orientation=descriptors.ATLAS_ORIENTATION,
            version=f"{ATLAS_VERSION}.{atlas_minor_version}",
            shape=shape,
            additional_references=additional_references_metadata,
            atlas_packager=atlas_packager,
            coordinate_space_metadata=coordinate_space_metadata,
            terminology_metadata=terminology_metadata,
            annotation_set_metadata=annotation_metadata,
            template_metadata=template_metadata,
        )

        with open(atlas_dir / "manifest.json", "w") as f:
            json.dump(metadata_dict, f, indent=4)

        atlas_name_for_validation = atlas_name_from_repr(
            atlas_name, resolution[0]
        )

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

        _check_validations(validation_results)

    return atlas_dir
