import json
import tarfile
import shutil
from pathlib import Path

import tifffile
import bgspace as bgs
import meshio as mio

from atlas_gen.metadata_utils import (
    create_metadata_files,
    generate_metadata_dict,
)
from atlas_gen.stacks import save_reference, save_annotation
from atlas_gen.structures import check_struct_consistency

from brainatlas_api import descriptors


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
    scale_meshes=False,
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
        Orientation of the original atlas (tuple describing origin for BGSpace).
    root_id :
        Id of the root element of the atlas.
    reference_stack : str or Path or numpy array
        Reference stack for the atlas. If str or Path, will be read with tifffile.
    annotation_stack : str or Path or numpy array
        Annotation stack for the atlas. If str or Path, will be read with tifffile.
    structures_list : list of dict
        List of valid dictionary for structures.
    meshes_dict : dict
        dict of meshio-compatible mesh file paths in the form {sruct_id: meshpath}
    working_dir : str or Path obj
        Path where the atlas folder and compressed file will be generated.
    hemispheres_stack : str or Path or numpy array, optional
        Hemisphere stack for the atlas. If str or Path, will be read with tifffile.
        If none is provided, atlas is assumed to be symmetric
    cleanup_files : bool, optional
         (Default value = False)
    compress : bool, optional
         (Default value = True)
    scale_meshes: bool, optional
        (Default values = False). If True the meshes points are scaled by the resolution
        to ensure that they are specified in microns, regardless of the atlas resolution.


    """

    version = f"{ATLAS_VERSION}.{atlas_minor_version}"

    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = hemispheres_stack is None

    # Instantiate BGSpace obj:
    space_convention = bgs.SpaceConvention(orientation)

    # Check consistency of structures .json file:
    check_struct_consistency(structures_list)

    atlas_dir_name = f"{atlas_name}_{resolution[0]}um_v{version}"
    dest_dir = Path(working_dir) / atlas_dir_name
    # exist_ok would be more permissive but error-prone here as there might
    # be old files
    dest_dir.mkdir()

    # write tiff stacks:
    for stack, saving_function in zip(
        [reference_stack, annotation_stack], [save_reference, save_annotation]
    ):

        if isinstance(stack, str) or isinstance(stack, Path):
            stack = tifffile.imread(stack)

        # Reorient stacks if required:
        original_shape = stack.shape
        stack = space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        shape = stack.shape

        saving_function(stack, dest_dir)

        del stack  # necessary?

    # Reorient vertices here as we need to know original stack size in um:
    volume_shape = tuple(res * s for res, s in zip(resolution, original_shape))

    mesh_dest_dir = dest_dir / descriptors.MESHES_DIRNAME
    mesh_dest_dir.mkdir()

    for mesh_id, meshfile in meshes_dict.items():
        mesh = mio.read(meshfile)

        # Reorient points:
        mesh.points = space_convention.map_points_to(
            descriptors.ATLAS_ORIENTATION, mesh.points, shape=volume_shape
        )

        # Scale the mesh to be in microns
        mesh.points *= resolution

        # Save in meshes dir:
        mio.write(mesh_dest_dir / f"{mesh_id}.obj", mesh)

    transformation_mat = space_convention.transformation_matrix_to(
        descriptors.ATLAS_ORIENTATION, shape=volume_shape
    )

    # save regions list json:
    with open(dest_dir / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(structures_list, f)

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
        transformation_mat=transformation_mat,
    )

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
