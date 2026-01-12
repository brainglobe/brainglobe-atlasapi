"""Script to validate atlases."""

import json
import os
import re
from pathlib import Path

import numpy as np

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.config import get_brainglobe_dir
from brainglobe_atlasapi.descriptors import METADATA_TEMPLATE, REFERENCE_DTYPE
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_atlases_lastversions,
)
from brainglobe_atlasapi.update_atlases import update_atlas


def validate_atlas_files(atlas: BrainGlobeAtlas):
    """
    Check if essential files exist in the atlas folder.

    This function verifies the presence of core files such as
    'annotation.tiff', 'reference.tiff', 'metadata.json', 'structures.json',
    and the 'meshes' directory within the atlas's root directory.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if all expected files and directories are found.

    Raises
    ------
    AssertionError
        If any expected file or directory is missing.
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
    Check if mesh and annotation coordinates are sufficiently close.

    Compares a mesh coordinate to an annotation coordinate, considering
    the pixel size and an arbitrary tolerance.

    The default tolerance value is 10.

    Parameters
    ----------
    mesh_coord : float
        The coordinate from the mesh.
    annotation_coord : float
        The coordinate from the annotation image, scaled by pixel size.
    pixel_size : float
        The size of a pixel in the dimension being checked.
    diff_tolerance : int, optional
        The maximum allowed difference between coordinates as a multiple
        of pixel size. By default, 10.

    Returns
    -------
    bool
        True if the coordinates are within the specified tolerance.

    Raises
    ------
    AssertionError
        If the absolute difference between `mesh_coord` and
        `annotation_coord` exceeds `diff_tolerance * pixel_size`.
    """
    assert abs(mesh_coord - annotation_coord) <= diff_tolerance * pixel_size, (
        f"Mesh coordinate {mesh_coord} and "
        f"annotation coordinate {annotation_coord}",
        f"differ by more than {diff_tolerance} times pixel size {pixel_size}",
    )
    return True


def validate_mesh_matches_image_extents(atlas: BrainGlobeAtlas):
    """Check if the mesh and the image extents are similar.

    Validates that the spatial extents of the `root` mesh align with the
    extents of the non-zero voxels in the annotation image, considering
    the atlas resolution.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if the mesh and image extents are sufficiently similar.

    Raises
    ------
    AssertionError
        If the extents differ by more than the allowed tolerance
        (default 10 times pixel size).
    """
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
    """Open the atlas for visual inspection (not implemented).

    This function is a placeholder for future visual validation routines.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The atlas to be visually checked.

    Returns
    -------
    bool
        Always True, as visual checks are not yet implemented.
    """
    # implement visual checks later
    return True


def validate_checksum(atlas: BrainGlobeAtlas):
    """Validate the atlas checksum (not implemented).

    This function is a placeholder for future checksum validation routines.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The atlas to be validated.

    Returns
    -------
    bool
        Always True, as checksum validation is not yet implemented.
    """
    # implement later
    return True


def validate_image_dimensions(atlas: BrainGlobeAtlas):
    """Check that annotation and reference images have identical dimensions.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if the dimensions match.

    Raises
    ------
    AssertionError
        If the `annotation` and `reference` image arrays have different shapes.
    """
    assert atlas.annotation.shape == atlas.reference.shape, (
        "Annotation and reference image have different dimensions. \n"
        f"Annotation image has dimension: {atlas.annotation.shape}, "
        f"while reference image has dimension {atlas.reference.shape}."
    )
    return True


def validate_additional_references(atlas: BrainGlobeAtlas):
    """Check that additional references have expected properties.

    Verifies that all additional reference images:
    1. Have the same dimensions as the main reference image.
    2. Are not identical to the main reference image (i.e., contain different
    data).

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if all additional references pass the validation checks.

    Raises
    ------
    AssertionError
        If an additional reference has unexpected dimensions or is identical
        to the main reference.
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
    """Check if all structures in the atlas have a corresponding mesh file.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if all structures listed in the atlas have a
        corresponding mesh file.

    Raises
    ------
    AssertionError
        If any structure ID found in `atlas.structures` does not have a
        matching `.obj` file in the atlas's `meshes` directory.
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
    """Check if all mesh files in the atlas folder are listed as a structure.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if all mesh files have a corresponding entry in the atlas's
        structures.

    Raises
    ------
    AssertionError
        If any .obj file found in the atlas's 'meshes' directory does not
        have a corresponding structure ID in `atlas.structures`.
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
    """Validate that the reference image was correctly rescaled.

    This check aims to catch issues where a float64 reference image (e.g., from
    MRI) might have been incorrectly rescaled or cast to the target integer
    data type (e.g., `REFERENCE_DTYPE`), resulting in pixel values that are
    too low. It asserts that not all pixel values are below 128
    (assuming 8-bit range).

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if the reference image's pixel values appear to be
        correctly scaled.

    Raises
    ------
    AssertionError
        If all pixel values in the reference image are less than 128,
        suggesting incorrect scaling.
    """
    assert not np.all(
        atlas.reference < 128
    ), f"Reference image is likely wrongly rescaled to {REFERENCE_DTYPE}"
    return True


def validate_annotation_symmetry(atlas: BrainGlobeAtlas):
    """Validate that equivalent regions in left and right hemispheres have the
    same annotation value.

    This is done by comparing annotation values at two pixels equidistant
    from the mid-sagittal plane along the central horizontal axis, near the
    center of the image.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if the selected annotation labels across the mid-sagittal plane
        are symmetric.

    Raises
    ------
    AssertionError
        If the annotation labels at the chosen symmetric points are different.
    """
    annotation = atlas.annotation
    centre = np.array(annotation.shape) // 2
    central_leftright_axis_annotations = annotation[centre[0], centre[1], :]
    label_5_left_of_centre = central_leftright_axis_annotations[centre[2] + 5]
    label_5_right_of_centre = central_leftright_axis_annotations[centre[2] - 5]
    assert (
        label_5_left_of_centre == label_5_right_of_centre
    ), "Annotation labels are asymmetric."
    return True


def validate_unique_acronyms(atlas: BrainGlobeAtlas):
    """Validate that all structure acronyms in the atlas are unique.

    Duplicate acronyms are incompatible with the current implementation
    of brainglobe-atlasapi as the acronym is used as a primary key to
    fetch details for a region.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if all acronyms are unique.

    Raises
    ------
    AssertionError
        If any duplicate acronyms are found in the atlas structures.
    """
    seen = set()
    duplicates = []

    for structure in atlas.structures:
        acronym = atlas.structures[structure]["acronym"]
        if acronym in seen:
            name = atlas.structures[structure]["name"]
            duplicates.append((acronym, name))
        else:
            seen.add(acronym)

    assert (
        len(duplicates) == 0
    ), f"Duplicate acronyms found in atlas structures: {sorted(duplicates)}"
    return True


def validate_atlas_name(atlas: BrainGlobeAtlas):
    """Validate the naming convention of the atlas.

    Checks if the atlas name adheres to specific rules:
    - Must be entirely lowercase.
    - Can only contain lowercase letters, digits, underscores, hyphens,
    and periods.
    - Must end with a resolution string (e.g., "5um", "37.5um", "1mm").

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object to validate.

    Returns
    -------
    bool
        True if the atlas name follows the specified conventions.

    Raises
    ------
    AssertionError
        If the atlas name contains capital letters, invalid characters, or
        does not end with a valid resolution string.
    """
    name = atlas.atlas_name
    allowed_chars = r"^[a-z0-9_.-]+$"
    res = name.split("_").pop()

    assert name == name.lower(), f"Atlas name {name} cannot contain capitals."

    assert re.match(
        allowed_chars, name
    ), f"Atlas name {name} contains invalid characters."

    resolution_pattern = r"\d+(\.\d+)?(nm|um|mm)$"
    assert re.search(resolution_pattern, res), (
        f"Atlas name {name} should end with a valid resolution "
        "(e.g., 5um, 1.5mm)."
    )

    return True


def validate_metadata(atlas: BrainGlobeAtlas):
    """Validate the atlas metadata.

    Checks that the metadata of the given atlas has the correct format.
    Specifically, it ensures that all required keys from `METADATA_TEMPLATE`
    are present and that the types of the values match the types specified
    in `METADATA_TEMPLATE`.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobeAtlas object whose metadata is to be validated.

    Returns
    -------
    bool
        True if the metadata adheres to the expected format and types.

    Raises
    ------
    AssertionError
        If a required key is missing from the metadata or if the type of
        a metadata value does not match the expected type.
    """
    for key, value in METADATA_TEMPLATE.items():
        assert key in atlas.metadata, f"Missing key: {key}"
        assert isinstance(atlas.metadata[key], type(value)), (
            f"{key} should be of type {type(value).__name__}, "
            f"but got {type(atlas.metadata[key]).__name__}."
        )
    return True


def get_all_validation_functions():
    """Return all individual validation functions as a list.

    All functions returned by this method are expected to accept
    a single argument: a `BrainGlobeAtlas` instance.

    Returns
    -------
    list of callable
        A list of functions that can be used to validate a BrainGlobe atlas.
    """
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
        validate_unique_acronyms,
        validate_atlas_name,
    ]


def validate_atlas(atlas_name, version, validation_functions):
    """Validate the latest version of a given atlas.

    This function attempts to load the specified atlas, updates it if
    necessary, and then runs a suite of provided validation functions
    against it. It collects and reports the results of each validation check.

    Parameters
    ----------
    atlas_name : str
        The name of the atlas to validate.
    version : str
        The version of the atlas to validate. (Currently not directly used
        for loading, but passed from `get_all_atlases_lastversions`).
    validation_functions : list of callable
        A list of functions, each expecting a `BrainGlobeAtlas` object
        as input and designed to perform a specific validation check.

    Returns
    -------
    dict
        A dictionary containing the validation results for the specified atlas.
        The format is `{atlas_name: [(function_name, error_message, status)]}`.
        `error_message` is None if the check passes.
    """
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
    """Main execution block for running atlas validations."""
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
        validate_unique_acronyms,
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
