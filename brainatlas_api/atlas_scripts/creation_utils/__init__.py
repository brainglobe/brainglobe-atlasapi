import json
from brainatlas_api.utils import read_tiff, read_json
from .metadata import generate_metadata_dict
from .structures import check_struct_consistency
from . import descriptors
import numpy as np
import tarfile


def wrapup_atlas_from_dir(dir_path, citation, atlas_link, species, resolution):
    # Check that all core files are contained:
    for element in [
        descriptors.STRUCTURES_FILENAME,
        descriptors.REFERENCE_FILENAME,
        descriptors.ANNOTATION_FILENAME,
    ]:
        assert (dir_path / element).exists()

    # Get name and version from dir name - in this way multiple
    # specifications are avoided:
    parsename = dir_path.name.split("_")

    atlas_name = "_".join(parsename[:-1])
    version = parsename[-1][1:]  # version: v0.0 format

    # Read stack shape:
    ref_stack = read_tiff(dir_path / descriptors.REFERENCE_FILENAME)
    shape = ref_stack.shape

    # If no hemisphere file is given, ensure the atlas is symmetric:
    if not (dir_path / descriptors.HEMISPHERES_FILENAME).exists():
        # assert np.allclose(ref_stack[:, :, :shape[2] // 2],
        #                   np.flip(ref_stack[:, :, -shape[2] // 2:], 2))
        symmetric = True
    else:
        symmetric = False

    # Check consistency of structures .json file:
    structures = read_json(dir_path / descriptors.STRUCTURES_FILENAME)
    check_struct_consistency(structures)

    # Finalize metadata dictionary
    print(version)
    metadata_dict = generate_metadata_dict(
        name=atlas_name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,
        version=version,
        shape=shape,
    )

    with open(dir_path / descriptors.METADATA_FILENAME, "w") as f:
        json.dump(metadata_dict, f)
