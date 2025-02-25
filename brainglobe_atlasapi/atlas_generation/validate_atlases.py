"""Script to validate atlases"""

import json
import os
from pathlib import Path

import numpy as np

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.descriptors import REFERENCE_DTYPE
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_atlases_lastversions,
)
from brainglobe_atlasapi.update_atlases import update_atlas


def validate_atlas_files(atlas: BrainGlobeAtlas):
    """
    Checks if basic files exist in the atlas folder

    custom_atlas_path is used when the function is called as part of
    the wrapup function in the atlas packaging script. The expected
    input is working_dir
    """

    atlas_path = atlas.root_dir

    assert atlas_path.is_dir(), f"Atlas path {atlas_path} not found"
    expected_files = [
        "annotation.tiff",
        "reference.tiff",
        "metadata.json",
        "structures.json",
    ]
    for expected_file_name in expected_files:
        expected_path = Path(atlas_path / expected_file_name)
        assert (
            expected_path.is_file()
        ), f"Expected file not found at {expected_path}"

    meshes_path = atlas_path / "meshes"
    assert meshes_path.is_dir(), f"Meshes path {meshes_path} not found"
    return True


def _assert_close(mesh_coord, annotation_coord, pixel_size, diff_tolerance=10):
    """
    Helper function to check if the mesh and the annotation coordinate
    are closer to each other than an arbitrary tolerance value
    times the pixel size.

    The default tolerance value is 10.
    """
    assert abs(mesh_coord - annotation_coord) <= diff_tolerance * pixel_size, (
        f"Mesh coordinate {mesh_coord} and "
        f"annotation coordinate {annotation_coord}",
        f"differ by more than {diff_tolerance} "
        f"times pixel size {pixel_size}",
    )
    return True


def validate_mesh_matches_image_extents(atlas: BrainGlobeAtlas):
    """Checks if the mesh and the image extents are similar"""

    root_mesh = atlas.mesh_from_structure("root")
    annotation_image = atlas.annotation
    resolution = atlas.resolution

    # minimum and maximum values of the annotation image (z, y, x)
    z_range, y_range, x_range = np.nonzero(annotation_image)
    z_min, z_max = np.min(z_range), np.max(z_range)
    y_min, y_max = np.min(y_range), np.max(y_range)
    x_min, x_max = np.min(x_range), np.max(x_range)

    # minimum and maximum values of the annotation image
    # scaled by the atlas resolution
    z_min_scaled, z_max_scaled = z_min * resolution[0], z_max * resolution[0]
    y_min_scaled, y_max_scaled = y_min * resolution[1], y_max * resolution[1]
    x_min_scaled, x_max_scaled = x_min * resolution[2], x_max * resolution[2]

    # z, y and x coordinates of the root mesh (extent of the whole object)
    mesh_points = root_mesh.points
    z_coords, y_coords, x_coords = (
        mesh_points[:, 0],
        mesh_points[:, 1],
        mesh_points[:, 2],
    )

    # minimum and maximum coordinates of the root mesh
    z_min_mesh, z_max_mesh = np.min(z_coords), np.max(z_coords)
    y_min_mesh, y_max_mesh = np.min(y_coords), np.max(y_coords)
    x_min_mesh, x_max_mesh = np.min(x_coords), np.max(x_coords)

    # checking if root mesh and image are on the same scale
    _assert_close(z_min_mesh, z_min_scaled, resolution[0])
    _assert_close(z_max_mesh, z_max_scaled, resolution[0])
    _assert_close(y_min_mesh, y_min_scaled, resolution[1])
    _assert_close(y_max_mesh, y_max_scaled, resolution[1])
    _assert_close(x_min_mesh, x_min_scaled, resolution[2])
    _assert_close(x_max_mesh, x_max_scaled, resolution[2])

    return True


def open_for_visual_check(atlas: BrainGlobeAtlas):
    # implement visual checks later
    return True


def validate_checksum(atlas: BrainGlobeAtlas):
    # implement later
    return True


def validate_image_dimensions(atlas: BrainGlobeAtlas):
    """
    Check that annotation and reference image have the same dimensions.
    """
    assert atlas.annotation.shape == atlas.reference.shape, (
        "Annotation and reference image have different dimensions. \n"
        f"Annotation image has dimension: {atlas.annotation.shape}, "
        f"while reference image has dimension {atlas.reference.shape}."
    )
    return True


def validate_additional_references(atlas: BrainGlobeAtlas):
    """
    Check that additional references are different, but have same dimensions.
    """
    for (
        additional_reference_name
    ) in atlas.additional_references.references_list:
        additional_reference = atlas.additional_references[
            additional_reference_name
        ]
        assert additional_reference.shape == atlas.reference.shape, (
            f"Additional reference {additional_reference} "
            "has unexpected dimension."
        )
        assert not np.all(
            additional_reference == atlas.reference
        ), "Additional reference is not different to main reference."
    return True


def catch_missing_mesh_files(atlas: BrainGlobeAtlas):
    """
    Checks if all the structures in the atlas have a corresponding mesh file

    custom_atlas_path is used when the function is called as part of
    the wrapup function in the atlas packaging script. The expected
    input is working_dir
    """

    ids_from_bg_atlas_api = list(atlas.structures.keys())

    atlas_path = atlas.root_dir

    obj_path = Path(atlas_path / "meshes")

    ids_from_mesh_files = [
        int(Path(file).stem)
        for file in os.listdir(obj_path)
        if file.endswith(".obj")
    ]

    in_bg_not_mesh = []
    for id in ids_from_bg_atlas_api:
        if id not in ids_from_mesh_files:
            in_bg_not_mesh.append(id)

    if len(in_bg_not_mesh) != 0:
        raise AssertionError(
            f"Structures with IDs {in_bg_not_mesh} are in the atlas, "
            "but don't have a corresponding mesh file."
        )
    return True


def catch_missing_structures(atlas: BrainGlobeAtlas):
    """
    Checks if all the mesh files in the atlas folder
    are listed as a structure in the atlas.
    """

    ids_from_bg_atlas_api = list(atlas.structures.keys())

    atlas_path = atlas.root_dir

    obj_path = Path(atlas_path / "meshes")

    ids_from_mesh_files = [
        int(Path(file).stem)
        for file in os.listdir(obj_path)
        if file.endswith(".obj")
    ]

    in_mesh_not_bg = []
    for id in ids_from_mesh_files:
        if id not in ids_from_bg_atlas_api:
            in_mesh_not_bg.append(id)

    if len(in_mesh_not_bg) != 0:
        raise AssertionError(
            f"Structures with IDs {in_mesh_not_bg} have a mesh file, "
            "but are not accessible through the atlas."
        )
    return True


def validate_reference_image_pixels(atlas: BrainGlobeAtlas):
    """Validates that reference image was correctly rescaled to
    target datatype. Often goes wrong when naively passing float64
    (MRI) data to wrapup function.
    """
    assert not np.all(
        atlas.reference < 128
    ), f"Reference image is likely wrongly rescaled to {REFERENCE_DTYPE}"
    return True


def validate_annotation_symmetry(atlas: BrainGlobeAtlas):
    """Validates that equivalent regions in L+R hemispheres have same
    annotation value. This is done by naively comparing two central pixels
    that are opposite each other along the mid-sagittal plane, and near
    the mid-sagittal plane."""
    annotation = atlas.annotation
    centre = np.array(annotation.shape) // 2
    central_leftright_axis_annotations = annotation[centre[0], centre[1], :]
    label_5_left_of_centre = central_leftright_axis_annotations[centre[2] + 5]
    label_5_right_of_centre = central_leftright_axis_annotations[centre[2] - 5]
    assert (
        label_5_left_of_centre == label_5_right_of_centre
    ), "Annotation labels are asymmetric."
    return True


def validate_atlas_name(atlas: BrainGlobeAtlas):
    """
    Ensures atlas names are all lowercase
    """
    assert (
        atlas.atlas_name == atlas.atlas_name.lower()
    ), f"Atlas name {atlas.atlas_name} cannot contain capitals."
    return True


def get_all_validation_functions():
    """Returns all individual validation functions as a list.
    All functions should expect 1 argument, a BrainGlobeAtlas."""
    return [
        validate_atlas_files,
        validate_mesh_matches_image_extents,
        open_for_visual_check,
        validate_checksum,
        validate_image_dimensions,
        validate_additional_references,
        catch_missing_mesh_files,
        catch_missing_structures,
        validate_reference_image_pixels,
        validate_annotation_symmetry,
        validate_atlas_name,
    ]


def validate_atlas(atlas_name, version, validation_functions):
    """Validates the latest version of a given atlas"""

    print(atlas_name, version)
    BrainGlobeAtlas(atlas_name)
    updated = get_atlases_lastversions()[atlas_name]["updated"]
    if not updated:
        update_atlas(atlas_name)

    validation_results = {atlas_name: []}

    for i, validation_function in enumerate(validation_functions):
        try:
            validation_function(BrainGlobeAtlas(atlas_name))
            validation_results[atlas_name].append(
                (validation_function.__name__, None, str("Pass"))
            )
        except AssertionError as error:
            validation_results[atlas_name].append(
                (validation_function.__name__, str(error), str("Fail"))
            )

    return validation_results


if __name__ == "__main__":
    # list to store the validation functions
    all_validation_functions = [
        validate_atlas_files,
        validate_mesh_matches_image_extents,
        open_for_visual_check,
        validate_checksum,
        validate_image_dimensions,
        validate_additional_references,
        catch_missing_mesh_files,
        catch_missing_structures,
        validate_reference_image_pixels,
        validate_annotation_symmetry,
        validate_atlas_name,
    ]

    valid_atlases = []
    invalid_atlases = []
    validation_results = {}

    for atlas_name, version in get_all_atlases_lastversions().items():
        temp_validation_results = validate_atlas(
            atlas_name, version, all_validation_functions
        )
        validation_results.update(temp_validation_results)

    print("Validation has been completed")
    print("Find validation_results.json in ~/.brainglobe/atlases/validation/")

    # Get the directory path
    output_dir_path = str(get_brainglobe_dir() / "atlases/validation")

    # Create the directory if it doesn't exist
    if not os.path.exists(output_dir_path):
        os.makedirs(output_dir_path)

    # Open a file for writing (will overwrite any files from previous runs!)
    with open(
        str(
            get_brainglobe_dir() / "atlases/validation/validation_results.json"
        ),
        "w",
    ) as file:
        json.dump(validation_results, file)
