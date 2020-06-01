import json
from atlas_gen.metadata_utils import (
    create_metadata_files,
    generate_metadata_dict,
)
from atlas_gen.stacks import save_reference, save_annotation

# from brainatlas_api.utils import read_tiff, read_json
from atlas_gen.structures import check_struct_consistency
from brainatlas_api import descriptors
import tarfile
import shutil

# import bgspace as bgs

# This should be changed every time we make changes in the atlas
# structure:
ATLAS_VERSION = 0


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
    hemispheres_stack=None,
    cleanup_files=False,
    compress=True,
):
    """
    Finalise an atlas with truly consistent format from all the data.

    Parameters
    ----------
    dest_dir : str or Path object
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

    version = f"{ATLAS_VERSION}.{atlas_minor_version}"
    shape = reference_stack.shape

    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = hemispheres_stack is None

    # Check consistency of structures .json file:
    check_struct_consistency(structures_list)

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

    atlas_dir_name = atlas_name + "_v" + version
    dest_dir = working_dir / atlas_dir_name
    dest_dir.mkdir()  # exist_ok would be more permissive but error-prone here

    # save regions list json:
    with open(dest_dir / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures_list, f)

    # TODO use BGSpace and reorient stacks;
    # TODO use BGSpace and reorient mesh;
    # TODO find function to save meshes;
    # write tiff stacks:
    save_reference(reference_stack, dest_dir)
    save_annotation(annotation_stack, dest_dir)

    # Create human readable .csv and .txt files:
    create_metadata_files(dest_dir, metadata_dict, structures_list, root_id)

    # Compress if required:
    if compress:
        output_filename = dest_dir.parent / f"{dest_dir.name}.tar.gz"
        print(f"Saving compressed atlas data at: {output_filename}")
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(dest_dir, arcname=dest_dir.name)

    # Cleanup if required:
    if cleanup_files:
        # Clean temporary directory and remove it:
        shutil.rmtree(dest_dir)
