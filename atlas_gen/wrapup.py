import json
from brainatlas_api.atlas_gen.metadata_utils import create_metadata_files
from brainatlas_api.utils import read_tiff, read_json
from .metadata import generate_metadata_dict
from .structures import check_struct_consistency
from brainatlas_api import descriptors
import tarfile
import shutil


def wrapup_atlas_from_dir(
    dir_path,
    citation,
    atlas_link,
    species,
    resolution,
    cleanup_files=False,
    compress=True,
    root=997,
):
    """
    Check compliance of a folder with atlas standards, write metadata, and if required compress and cleanup.
    This function should be used to finalize all atlases as it runs the required
    controls.

    Parameters
    ----------
    dir_path : str or Path object
        directory with the atlases and regions description
    citation : str
        citation for the atlas, if unpublished specify "unpublished"
    atlas_link : str
        valid URL for the atlas
    species : str
        species name formatted as "CommonName (Genus species)"
    resolution : tuple
        tree elements, resolution on three axes
    cleanup_files : bool
         (Default value = False)
    compress : bool
         (Default value = True)


    """

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

    # Finalize metadata dictionary:
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

    # write metadata dict:
    with open(dir_path / descriptors.METADATA_FILENAME, "w") as f:
        json.dump(metadata_dict, f)

    # Create human readable .csv and .txt files
    create_metadata_files(dir_path, metadata_dict, structures, root)

    # Compress if required:
    if compress:
        output_filename = dir_path.parent / f"{dir_path.name}.tar.gz"
        print(f"Saving compressed atlas data at: {output_filename}")
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(dir_path, arcname=dir_path.name)

    # Cleanup if required:
    if cleanup_files:
        # Clean temporary directory and remove it:
        shutil.rmtree(dir_path)
