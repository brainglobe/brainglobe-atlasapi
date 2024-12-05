import shutil
from pathlib import Path

import napari
from brainrender_napari.napari_atlas_representation import (
    NapariAtlasRepresentation,
)
from napari.viewer import Viewer

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_generation.validate_atlases import (
    catch_missing_mesh_files,
    catch_missing_structures,
    open_for_visual_check,
    validate_additional_references,
    validate_atlas_files,
    validate_checksum,
    validate_image_dimensions,
    validate_mesh_matches_image_extents,
)

all_validation_functions = [
    validate_atlas_files,
    #validate_mesh_matches_image_extents,
    open_for_visual_check,
    validate_checksum,
    validate_image_dimensions,
    validate_additional_references,
    catch_missing_mesh_files,
    catch_missing_structures,
]


# adapt this code block for newly packaged atlases
brainglobe_dir = Path.home() / ".brainglobe/"
working_dir = Path.home() / "brainglobe_workingdir/"
atlas_name = "columbia_cuttlefish"
resolution = 2
minor_version = 0

# nothing below this needs changing

# make sure we have the latest packaged version in .brainglobe
# by replacing it with the working_dir version if needed
#atlas_name_with_version = f"{atlas_name}_{resolution}um_v1.{minor_version}"
#source_dir = working_dir / atlas_name / atlas_name_with_version
#destination_dir = brainglobe_dir / atlas_name_with_version
#destination_dir = working_dir / atlas_name_with_version
#if destination_dir.exists() and destination_dir.is_dir():
#    shutil.rmtree(destination_dir)
#assert source_dir.exists()
#if source_dir.exists():
#    shutil.copytree(source_dir, destination_dir)
#assert destination_dir.exists()

# run validation functions on the new atlas
atlas = BrainGlobeAtlas(f"{atlas_name}_{resolution}um")
validation_results = {atlas_name: []}

for i, validation_function in enumerate(all_validation_functions):
    try:
        validation_function(atlas)
        validation_results[atlas_name].append(
            (validation_function.__name__, None, str("Pass"))
        )
    except AssertionError as error:
        validation_results[atlas_name].append(
            (validation_function.__name__, str(error), str("Fail"))
        )

# print validation results and open napari for a visual check
# in napari, we should see three layers:
# - the annotation
# - the reference image (visibility turned off by default)
# - the root mesh

failed_validations = [
    (result[0], result[1])
    for result in validation_results[atlas_name]
    if result[2] == "Fail"
]
if failed_validations:
    print("Failed validations:")
    for failed in failed_validations:
        print(failed)
else:
    print(f"{atlas_name} is a valid atlas")

viewer = Viewer()
viewer.dims.ndisplay = 3
napari_atlas = NapariAtlasRepresentation(
    atlas, viewer
)
napari_atlas.add_structure_to_viewer("AAB")
napari_atlas.add_structure_to_viewer("V")
napari_atlas.add_structure_to_viewer("Or")
napari_atlas.add_structure_to_viewer("BPCl")
napari_atlas.add_to_viewer()

napari.run()