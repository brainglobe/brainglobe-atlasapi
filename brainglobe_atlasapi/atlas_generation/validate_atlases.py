"""Script to validate atlases."""

import collections  # <--- NEW IMPORT
from typing import (  # Ensure these are imported for type hinting in get_all_validation_functions
    Callable,
    List,
)

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.descriptors import METADATA_TEMPLATE


def validate_atlas_files(atlas: BrainGlobeAtlas):
    # ... (rest of validate_atlas_files function remains unchanged)
    # ...
    meshes_path = atlas_path / "meshes"
    assert meshes_path.is_dir(), f"Meshes path {meshes_path} not found"
    return True


# ... (all other existing validation functions remain unchanged)
# ...


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


def check_unique_acronyms(atlas: BrainGlobeAtlas):
    """
    Check if all acronyms for brain regions in the atlas structures are unique.

    The validation ensures that the number of structures equals the number of
    unique acronyms found, preventing ambiguity during region lookup.

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
        If duplicate acronyms are found in the atlas structures.
    """
    # Get all acronyms from the structures dictionary values
    acronyms = [
        structure["acronym"] for structure in atlas.structures.values()
    ]

    # Assert that the length of the list equals the length of the set (unique values)
    assert len(acronyms) == len(set(acronyms)), (
        f"Duplicate acronyms found in atlas structures. "
        f"Duplicates: {[item for item, count in collections.Counter(acronyms).items() if count > 1]}"
    )
    return True


def get_all_validation_functions() -> List[Callable]:
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
        validate_atlas_name,
        validate_metadata,
        check_unique_acronyms,  # <--- NEW FUNCTION INTEGRATED
    ]


def validate_atlas(atlas_name, version, validation_functions):
    # ... (rest of validate_atlas function remains unchanged)
    # ...
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
        validate_atlas_name,
        validate_metadata,  # <--- NOW CORRECTLY INCLUDED
        check_unique_acronyms,  # <--- NEW FUNCTION ADDED
    ]

# ... (rest of __main__ block remains unchanged)
