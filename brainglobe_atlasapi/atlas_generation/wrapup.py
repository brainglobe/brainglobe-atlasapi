"""Tools to finalise the atlas creation process."""

import json
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import brainglobe_space as bgs
import meshio as mio
import ngff_zarr as nz
import numpy as np
import numpy.typing as npt
import pandas as pd

from brainglobe_atlasapi import atlas_generation, descriptors
from brainglobe_atlasapi.atlas_generation.atlas_packaging_data import (
    AnnotationInfo,
    AtlasPackagingData,
    CoordinateSpaceInfo,
    TemplateInfo,
    TerminologyInfo,
)
from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    generate_metadata_dict,
)
from brainglobe_atlasapi.atlas_generation.stacks import (
    save_annotation,
    save_hemispheres,
    save_template,
    write_multiscale_ome_zarr,
)
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    get_all_validation_functions,
    report_validation_results,
)
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.utils import atlas_name_from_repr

# This should be changed every time we make changes in the atlas
# structure:
ATLAS_VERSION = atlas_generation.__version__


def _save_if_not_exists(
    stacks: List[npt.NDArray],
    dest_dir: Path,
    label: str,
    transformations: List[List[dict]],
    save_fn: Callable[[npt.NDArray, Path, List[List[dict]]], None],
) -> None:
    if dest_dir.exists():
        print(f"{label} directory already exists, skipping: {dest_dir}")
        return

    save_fn(stacks, dest_dir, transformations)


def _merge_resolutions_list(
    existing_resolutions: List[Tuple[int | float]],
    new_resolutions: List[Tuple[int | float]],
) -> List[Tuple[int | float]]:
    merged_resolutions = sorted(set(existing_resolutions + new_resolutions))

    return merged_resolutions


def _insert_into_multiscale(
    multiscale: nz.Multiscales,
    transformations: List[List[dict]],
    new_data: List[npt.NDArray],
    working_dir: Path,
) -> None:
    requested_resolutions = [
        tuple(transform[0]["scale"]) for transform in transformations
    ]
    # Merge existing multiscale transformations with new ones
    merged_resolutions = _merge_resolutions_list(
        [tuple(im.scale.values()) for im in multiscale.images],
        requested_resolutions,
    )

    # Create a mapping from resolution to new_data
    resolution_to_data = {
        res: data for res, data in zip(requested_resolutions, new_data)
    }

    # Extract existing data into the map
    for image in multiscale.images:
        res_tuple = tuple(image.scale.values())
        if res_tuple not in resolution_to_data:
            resolution_to_data[res_tuple] = image.data.compute()

    dtype = multiscale.images[0].data.dtype

    # Create new images list with merged resolutions
    stack_list = [
        resolution_to_data[res].astype(dtype) for res in merged_resolutions
    ]
    new_transformations = [
        [{"type": "scale", "scale": [res for res in res_tuple]}]
        for res_tuple in merged_resolutions
    ]

    write_multiscale_ome_zarr(
        images=stack_list,
        output_path=working_dir,
        transformations=new_transformations,
    )


def _build_transformations(
    resolution_standard: List[Tuple[int | float]],
) -> List[List[dict]]:
    return [
        [{"type": "scale", "scale": [res / 1000 for res in res_tuple]}]
        for res_tuple in resolution_standard
    ]


def _save_terminology_csv(
    structures_list: List[Dict],
    terminology_path: Path,
) -> None:
    structures_df = pd.DataFrame(structures_list)
    terminology_df = pd.DataFrame()

    terminology_df["identifier"] = structures_df["id"].astype(np.uint32)
    terminology_df["parent_identifier"] = (
        structures_df["structure_id_path"]
        .apply(lambda x: x[-2] if len(x) > 1 else None)
        .astype(pd.UInt32Dtype())
    )
    terminology_df["annotation_value"] = structures_df["id"].astype(np.uint32)
    terminology_df["name"] = structures_df["name"].astype(pd.StringDtype())
    terminology_df["abbreviation"] = structures_df["acronym"].astype(
        pd.StringDtype()
    )
    terminology_df["color_hex_triplet"] = structures_df["rgb_triplet"].apply(
        lambda x: "".join(f"{c:02X}" for c in x)
    )
    terminology_df["color_hex_triplet"] = "#" + terminology_df[
        "color_hex_triplet"
    ].astype(pd.StringDtype())
    terminology_df["root_identifier_path"] = structures_df["structure_id_path"]

    terminology_df.to_csv(terminology_path, index=False)


def _save_coordinate_space_manifest(
    coordinate_space_metadata: dict,
    coordinate_space_path: Path,
) -> None:
    with open(coordinate_space_path, "w") as f:
        json.dump(coordinate_space_metadata, f, indent=4)


def _save_meshes(
    meshes_dict: Dict[int | str, str | Path],
    mesh_dest_dir: Path,
    space_convention: bgs.AnatomicalSpace,
    scale_meshes: bool,
    resolution_standard: List[Tuple[int | float]],
    resolution_mapping: Optional[List[int]],
) -> None:
    if mesh_dest_dir.exists():
        print(f"Mesh directory already exists, skipping: {mesh_dest_dir}")
        return

    mesh_dest_dir.mkdir(parents=True)

    for mesh_id, meshfile in meshes_dict.items():
        mesh = mio.read(meshfile)

        if scale_meshes:
            if not resolution_mapping:
                mesh.points *= resolution_standard[0]
            else:
                original_resolution = (
                    resolution_standard[0][resolution_mapping[0]],
                    resolution_standard[0][resolution_mapping[1]],
                    resolution_standard[0][resolution_mapping[2]],
                )
                mesh.points *= original_resolution

        mesh.points = space_convention.map_points_to(
            descriptors.ATLAS_ORIENTATION, mesh.points
        )

        # TODO: parallelise and copy if not scaling or reorienting
        mio.write(
            mesh_dest_dir / f"{mesh_id}",
            mesh,
            file_format="neuroglancer",
        )


def _save_template_data(
    packaging_data: AtlasPackagingData,
    transformations: List[List[dict]],
) -> List[tuple]:
    template_info = packaging_data.template_info
    if not template_info.skip_saving and not template_info.update_existing:
        shapes = [stack.shape for stack in packaging_data.reference_stack]
        dest_dir = packaging_data.working_dir / template_info.metadata[
            "location"
        ].lstrip("/")
        _save_if_not_exists(
            packaging_data.reference_stack,
            dest_dir,
            template_info.metadata["name"],
            transformations,
            save_template,
        )
    elif template_info.update_existing:
        local_existing_path = (
            packaging_data.working_dir / template_info.existing_stub
        )
        multiscale = nz.from_ngff_zarr(local_existing_path)
        local_target_path = packaging_data.working_dir / template_info.stub
        _insert_into_multiscale(
            multiscale,
            transformations=transformations,
            new_data=packaging_data.reference_stack,
            working_dir=local_target_path,
        )
        updated_multiscale = nz.from_ngff_zarr(local_target_path)
        shapes = [image.data.shape for image in updated_multiscale.images]
    else:
        multiscale = nz.from_ngff_zarr(
            packaging_data.working_dir / template_info.stub
        )
        shapes = [image.data.shape for image in multiscale.images]

    return shapes


def _save_annotation_data(
    packaging_data: AtlasPackagingData,
    transformations: List[List[dict]],
    scale_meshes: bool,
    resolution_mapping: Optional[List[int]],
) -> List[tuple]:
    annotation_info = packaging_data.annotation_info

    if not annotation_info.skip_saving and not annotation_info.update_existing:
        shapes = [stack.shape for stack in packaging_data.annotation_stack]
        dest_dir = packaging_data.working_dir / annotation_info.metadata[
            "location"
        ].lstrip("/")

        _save_if_not_exists(
            packaging_data.annotation_stack,
            dest_dir,
            annotation_info.metadata["name"],
            transformations,
            save_annotation,
        )

        hemispheres_stub = descriptors.format_hemispheres_stub(
            annotation_info.name, annotation_info.version
        )
        dest_dir_hemi = packaging_data.working_dir / hemispheres_stub

        _save_if_not_exists(
            packaging_data.hemispheres_stack,
            dest_dir_hemi,
            annotation_info.metadata["name"],
            transformations,
            save_hemispheres,
        )
    elif annotation_info.update_existing:
        local_existing_path = (
            packaging_data.working_dir / annotation_info.existing_stub
        )
        multiscale = nz.from_ngff_zarr(local_existing_path)
        local_target_path = packaging_data.working_dir / annotation_info.stub
        _insert_into_multiscale(
            multiscale,
            transformations=transformations,
            new_data=packaging_data.annotation_stack,
            working_dir=local_target_path,
        )

        existing_hemispheres_stub = descriptors.format_hemispheres_stub(
            annotation_info.name, annotation_info.existing_version
        )
        local_existing_hemispheres = (
            packaging_data.working_dir / existing_hemispheres_stub
        )
        multiscale_hemispheres = nz.from_ngff_zarr(local_existing_hemispheres)
        hemispheres_stub = descriptors.format_hemispheres_stub(
            annotation_info.name, annotation_info.version
        )
        local_target_hemispheres = (
            packaging_data.working_dir / hemispheres_stub
        )

        _insert_into_multiscale(
            multiscale_hemispheres,
            transformations=transformations,
            new_data=packaging_data.hemispheres_stack,
            working_dir=local_target_hemispheres,
        )

        updated_multiscale = nz.from_ngff_zarr(local_target_path)
        shapes = [image.data.shape for image in updated_multiscale.images]
    else:
        multiscale = nz.from_ngff_zarr(
            packaging_data.working_dir / annotation_info.stub
        )
        shapes = [image.data.shape for image in multiscale.images]

    if not annotation_info.skip_saving or annotation_info.update_existing:
        meshes_stub = descriptors.format_meshes_stub(
            annotation_info.name, annotation_info.version
        )
        mesh_dest_dir = packaging_data.working_dir / meshes_stub
        _save_meshes(
            packaging_data.meshes_dict,
            mesh_dest_dir,
            packaging_data.space_convention,
            scale_meshes,
            packaging_data.resolution,
            resolution_mapping,
        )

    return shapes


def _save_additional_references(
    packaging_data: AtlasPackagingData,
    transformations: List[List[dict]],
) -> List[dict]:
    for ref_tuple in packaging_data.additional_references:
        ref_info, additional_template = ref_tuple

        if not ref_info.skip_saving and not ref_info.update_existing:
            dest_dir = packaging_data.working_dir / ref_info.metadata[
                "location"
            ].lstrip("/")
            _save_if_not_exists(
                additional_template,
                dest_dir,
                ref_info.metadata["name"],
                transformations,
                save_template,
            )
        elif ref_info.update_existing:
            local_existing_path = (
                packaging_data.working_dir / ref_info.existing_stub
            )
            multiscale = nz.from_ngff_zarr(local_existing_path)
            local_target_path = packaging_data.working_dir / ref_info.stub
            _insert_into_multiscale(
                multiscale,
                transformations=transformations,
                new_data=additional_template,
                working_dir=local_target_path,
            )

    return


def _finalize_atlas_at_resolution(
    resolution: Tuple[int | float],
    shape: tuple,
    packaging_data: AtlasPackagingData,
    additional_references_metadata: List[dict],
    overwrite: bool,
) -> Path:
    atlas_version = packaging_data.atlas_version
    atlas_version_underscore = atlas_version.replace(".", "_")
    symmetric = packaging_data.symmetric
    atlas_name = packaging_data.atlas_name

    atlas_name_with_res = f"{atlas_name}_{resolution[0]}um"
    atlas_location = (
        f"/{descriptors.V2_ATLAS_ROOTDIR}/"
        f"{atlas_name_with_res}/{atlas_version_underscore}"
    )
    atlas_dir = packaging_data.working_dir / atlas_location.strip("/")

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

    metadata_dict = generate_metadata_dict(
        name=atlas_name_with_res,
        location=atlas_location,
        citation=packaging_data.citation,
        atlas_link=packaging_data.atlas_link,
        species=packaging_data.species,
        symmetric=symmetric,
        resolution=resolution,
        orientation=descriptors.ATLAS_ORIENTATION,
        version=atlas_version,
        shape=shape,
        additional_references=additional_references_metadata,
        atlas_packager=packaging_data.atlas_packager,
        coordinate_space_metadata=packaging_data.coordinate_space_info.metadata,
        terminology_metadata=packaging_data.terminology_info.metadata,
        annotation_set_metadata=packaging_data.annotation_info.metadata,
        template_metadata=packaging_data.template_info.metadata,
    )

    with open(atlas_dir / "manifest.json", "w") as f:
        json.dump(metadata_dict, f, indent=4)

    atlas_name_for_validation = atlas_name_from_repr(atlas_name, resolution[0])

    atlas_to_validate = BrainGlobeAtlas(
        atlas_name=atlas_name_for_validation,
        brainglobe_dir=packaging_data.working_dir.parent,
        check_latest=False,
    )

    print(f"Running atlas validation on {atlas_location}")

    validation_results = {}

    for func in get_all_validation_functions():
        try:
            func(atlas_to_validate)
            validation_results[func.__name__] = "Pass"
        except AssertionError as e:
            validation_results[func.__name__] = f"Fail: {str(e)}"

    report_validation_results(validation_results)

    return atlas_dir


def wrapup_atlas_from_data(
    atlas_name: str,
    atlas_minor_version: int | str,
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
    template_info: Dict[str, str | bool] | None = None,
    annotation_info: Dict[str, str | bool] | None = None,
    terminology_info: Dict[str, str | bool] | None = None,
    coordinate_space_info: Dict[str, str | bool] | None = None,
    scale_meshes=True,
    resolution_mapping=None,
    additional_references: (
        List[
            Tuple[
                Dict | str,
                str | Path | npt.NDArray | List[str | Path | npt.NDArray],
            ]
        ]
        | None
    ) = None,
    additional_metadata: dict | None = None,
    overwrite=False,
    cleanup_files=None,
    compress=None,
):
    """
    Finalise an atlas with truly consistent format from all the data.

    Parameters
    ----------
    atlas_name : str
        Atlas name in the form author_species.
    atlas_minor_version : int | str
        Minor version number for this particular atlas.
    citation : str
        Citation for the atlas, if unpublished specify "unpublished".
    atlas_link : str
        Valid URL for the atlas.
    species : str
        Species name formatted as "CommonName (Genus species)".
    resolution : Tuple[int | float] | List[Tuple[int | float]]
        Three elements tuple, resolution on three axes or a list of such tuples
        for each scale, ordered from highest to lowest resolution.
    orientation : str
        Orientation of the original atlas
        (tuple describing origin for BGSpace).
    root_id : int
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
        List of valid dictionaries for structures.
    meshes_dict : Dict[int | str, str | Path]
        dict of meshio-compatible mesh file paths in the form
        {struct_id: meshpath}
    working_dir : str | Path
        Path where the atlas will be generated.
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
    additional_references: List[Tuple[Dict | str, str | Path | npt.NDArray | List[str | Path | npt.NDArray]]] | None
        List of tuples containing metadata and arrays for secondary templates.
    additional_metadata: dict, optional
        (Default value = empty dict).
        Additional metadata to write to manifest.json
    overwrite : bool, optional
        (Default value = False).
        If True, will overwrite existing atlas directory.
        If False and atlas directory exists, raises FileExistsError.
    cleanup_files : deprecated, optional
        (Default value = None).
        Deprecated and has no effect.
    compress : deprecated, optional
        (Default value = None).
        Deprecated and has no effect.
    """  # noqa: E501
    if cleanup_files is not None:
        print(
            "Warning: `cleanup_files` argument is deprecated and has no effect"
        )

    if compress is not None:
        print("Warning: `compress` argument is deprecated and has no effect")

    working_dir = Path(working_dir) / "brainglobe-atlasapi"
    atlas_version = f"{ATLAS_VERSION}.{atlas_minor_version}"

    if template_info is None:
        template_info = {
            "name": f"{atlas_name}-template",
            "version": atlas_version,
        }

    if terminology_info is None:
        terminology_info = {
            "name": f"{atlas_name}-terminology",
            "version": atlas_version,
        }

    if annotation_info is None:
        annotation_info = {
            "name": f"{atlas_name}-annotation",
            "version": atlas_version,
        }

    if coordinate_space_info is None:
        coordinate_space_info = {
            "name": f"{atlas_name}-coordinate-space",
            "version": atlas_version,
        }

    additional_template_list = []
    if additional_references is not None:
        for ref_tuple in additional_references:
            ref_metadata, _ = ref_tuple
            if isinstance(ref_metadata, str):
                ref_dict = {
                    "name": f"{ref_metadata}-template",
                    "version": atlas_version,
                }

            component_info = TemplateInfo(**ref_dict)
            additional_template_list.append((component_info, ref_tuple[1]))

    template_info = TemplateInfo(**template_info)
    terminology_info = TerminologyInfo(**terminology_info)
    annotation_info = AnnotationInfo(
        template=template_info, terminology=terminology_info, **annotation_info
    )
    coordinate_space_info = CoordinateSpaceInfo(
        template=template_info, **coordinate_space_info
    )

    packaging_data = AtlasPackagingData(
        atlas_name=atlas_name,
        atlas_version=atlas_version,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        resolution=resolution,
        orientation=orientation,
        root_id=root_id,
        reference_stack=reference_stack,
        annotation_stack=annotation_stack,
        working_dir=working_dir,
        template_info=template_info,
        annotation_info=annotation_info,
        terminology_info=terminology_info,
        coordinate_space_info=coordinate_space_info,
        structures_list=structures_list,
        meshes_dict=meshes_dict,
        atlas_packager=atlas_packager,
        hemispheres_stack=hemispheres_stack,
        additional_references=additional_template_list,
        additional_metadata=additional_metadata,
    )

    transformations = _build_transformations(packaging_data.resolution)

    shapes = _save_template_data(
        packaging_data,
        transformations,
    )

    shapes = _save_annotation_data(
        packaging_data,
        transformations,
        scale_meshes,
        resolution_mapping,
    )

    _save_additional_references(
        packaging_data,
        transformations,
    )

    if not terminology_info.skip_saving:
        terminology_dir = working_dir / terminology_info.stub

        terminology_dir.parent.mkdir(parents=True, exist_ok=True)
        _save_terminology_csv(
            packaging_data.structures_list,
            terminology_dir,
        )

    if not coordinate_space_info.skip_saving:
        coordinate_space_path = working_dir / coordinate_space_info.stub

        coordinate_space_path.parent.mkdir(parents=True, exist_ok=True)
        _save_coordinate_space_manifest(
            coordinate_space_info.metadata, coordinate_space_path
        )

    for resolution, shape in zip(packaging_data.resolution, shapes):
        _finalize_atlas_at_resolution(
            resolution=resolution,
            shape=shape,
            packaging_data=packaging_data,
            additional_references_metadata=[],
            overwrite=overwrite,
        )

    return
