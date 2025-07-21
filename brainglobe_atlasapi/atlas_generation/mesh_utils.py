from scipy.ndimage import binary_closing, binary_fill_holes

try:
    from vedo import Mesh, Volume, write
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Mesh generation with these utils requires vedo\n"
        + '   please install with "pip install vedo -U"'
    )

try:
    import mcubes
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Mesh generation with these utils requires PyMCubes\n"
        + '   please install with "pip install PyMCubes -U"'
    )

from pathlib import Path

import numpy as np
from loguru import logger
from rich.progress import track

from brainglobe_atlasapi.atlas_generation.volume_utils import (
    create_masked_array,
)
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# ----------------- #
#   MESH CREATION   #
# ----------------- #


def extract_mesh_from_mask(
    volume,
    obj_filepath=None,
    threshold=0.5,
    smooth: bool = False,
    mcubes_smooth=False,
    closing_n_iters=8,
    decimate_fraction: float = 0.6,  # keep 60% of original fertices
    use_marching_cubes=False,
    extract_largest=False,
):
    """
    Returns a vedo mesh actor with just the outer surface of a
    binary mask volume. It's faster though less accurate than
    extract_mesh_from_mask


    Parameters
    ----------
    obj_filepath: str or Path object
        path to where the .obj mesh file will be saved
    volume: 3d np.ndarray
    threshold: float
        min value to threshold the volume for isosurface extraction
    smooth: bool
        if True the surface mesh is smoothed
    use_marching_cubes: bool:
        if true PyMCubes is used to extract the volume's surface
        it's slower and less accurate than vedo though.
    mcubes_smooth: bool,
        if True mcubes.smooth is used before applying marching cubes
    closing_n_iters: int
        number of iterations of closing morphological operation.
        set to None to avoid applying morphological operations
    decimate_fraction: float  in range [0, 1].
        What fraction of the original number of vertices is to be kept.
        EG .5 means that 50% of the vertices are kept,
        the others are removed.
    tol: float
        parameter for decimation, with larger values corresponding
        to more aggressive decimation.
        EG 0.02 -> points that are closer than 2% of the size of the mesh's
        bounding box are identified and removed (only one is kept).
    extract_largest: bool
        If True only the largest region are extracted. It can cause issues for
        bilateral regions as only one will remain

    """
    # check savepath argument
    if obj_filepath is not None:
        if isinstance(obj_filepath, str):
            obj_filepath = Path(obj_filepath)

        if not obj_filepath.parents[0].exists():
            raise FileExistsError(
                "The folder where the .obj file is to be saved doesn't exist"
                + f"\n      {str(obj_filepath)}"
            )

    # Check volume argument
    if not np.isin(volume, [0, 1]).all():
        raise ValueError(
            "Argument volume should be a binary mask with only "
            "0s and 1s when passing a np.ndarray"
        )

    # Apply morphological transformations
    if closing_n_iters is not None:
        volume = binary_fill_holes(volume).astype(int)
        volume = binary_closing(volume, iterations=closing_n_iters).astype(int)

    if not use_marching_cubes:
        # Use faster algorithm
        volume = Volume(volume)
        mesh = volume.clone().isosurface(value=threshold).cap()
    else:
        print(
            "The marching cubes algorithm might be rotated "
            "compared to your volume data"
        )
        # Apply marching cubes and save to .obj
        if mcubes_smooth:
            smooth_array = mcubes.smooth(volume)
            vertices, triangles = mcubes.marching_cubes(smooth_array, 0)
        else:
            vertices, triangles = mcubes.marching_cubes(volume, 0.5)

        #  create mesh
        mesh = Mesh((vertices, triangles))

    # Cleanup and save
    if extract_largest:
        mesh = mesh.extract_largest_region()

    # decimate
    mesh.decimate_pro(decimate_fraction)

    if smooth:
        mesh.smooth()

    if obj_filepath is not None:
        write(mesh, str(obj_filepath))

    return mesh


def create_region_mesh(args):
    """
    Automates the creation of a region's mesh. Given a volume of annotations
    and a structures tree, it takes the volume's region corresponding to the
    region of interest and all of it's children's labels and creates a mesh.
    It takes a tuple of arguments to facilitaed parallel processing with
    multiprocessing.pool.map

    Note, by default it avoids overwriting a structure's mesh if the
    .obj file exists already.

    Parameters
    ----------
    meshes_dir_path: pathlib Path object with folder where meshes are saved
    tree: treelib.Tree with hierarchical structures information
    node: tree's node corresponding to the region who's mesh is being created
    labels: list of unique label annotations in annotated volume,
    (list(np.unique(annotated_volume)))
    annotated_volume: 3d numpy array with annotaed volume
    ROOT_ID: int,
    id of root structure (mesh creation is a bit more refined for that)
    """
    # Split arguments
    logger.debug(f"Creating mesh for region {args[1].identifier}")
    meshes_dir_path = args[0]
    node = args[1]
    tree = args[2]
    labels = args[3]
    annotated_volume = args[4]
    ROOT_ID = args[5]
    closing_n_iters = args[6]
    decimate_fraction = args[7]
    smooth = args[8]

    # Avoid overwriting existing mesh
    savepath = meshes_dir_path / f"{node.identifier}.obj"
    # if savepath.exists():
    #     logger.debug(f"Mesh file save path exists already, skipping.")
    #     return

    # Get lables for region and it's children
    stree = tree.subtree(node.identifier)
    ids = list(stree.nodes.keys())

    # Keep only labels that are in the annotation volume
    matched_labels = [i for i in ids if i in labels]

    if (
        not matched_labels
    ):  # it fails if the region and all of it's children are not in annotation
        print(f"No labels found for {node.tag}")
        return
    else:
        # Create mask and extract mesh
        mask = create_masked_array(annotated_volume, ids)

        if not np.max(mask):
            print(f"Empty mask for {node.tag}")
        else:
            if node.identifier == ROOT_ID:
                extract_mesh_from_mask(
                    mask,
                    obj_filepath=savepath,
                    smooth=smooth,
                    decimate_fraction=decimate_fraction,
                )
            else:
                extract_mesh_from_mask(
                    mask,
                    obj_filepath=savepath,
                    smooth=smooth,
                    closing_n_iters=closing_n_iters,
                    decimate_fraction=decimate_fraction,
                )


def construct_meshes_from_annotation(
    download_path,
    volume,
    structures_list,
    root_id,
    closing_n_iters=2,
    decimate_fraction=0,
    smooth=False,
):
    """
    Retrieve or construct atlas region meshes for a given annotation volume.

    If an atlas is packaged with mesh files, reuse those. Otherwise, construct
    the meshes using the existing volume and structure tree. Returns a
    dictionary mapping structure IDs to their corresponding .obj mesh files.

    Parameters
    ----------
    download_path : Path
        Path to the directory where new mesh files will be saved.
    volume : np.ndarray
        3D annotation volume.
    structures_list : list
        List of structure dictionaries containing id information.
    root_id : int
        Identifier for the root structure.
    smooth: bool
        if True the surface mesh is smoothed
    closing_n_iters: int
        number of iterations of closing morphological operation.
        set to None to avoid applying morphological operations
    decimate_fraction: float  in range [0, 1].
        What fraction of the original number of vertices is to be kept.
        EG .5 means that 50% of the vertices are kept,
        the others are removed.
    Returns
    -------
    dict
        Dictionary of structure IDs and paths to their .obj mesh files.
    """
    meshes_dir_path = download_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures_list)
    labels = np.unique(volume).astype(np.int32)

    for key, node in tree.nodes.items():
        node.data = Region(key in labels)

    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        description="Creating meshes",
    ):
        create_region_mesh(
            (
                meshes_dir_path,
                node,
                tree,
                labels,
                volume,
                root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )
    meshes_dict = {}
    structures_with_mesh = []
    for s in structures_list:
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        if mesh_path.stat().st_size < 512:
            print(f"obj file for {s} is too small, ignoring it.")
            continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        (
            f"In the end, {len(structures_with_mesh)}",
            "structures with mesh are kept",
        )
    )
    return meshes_dict


class Region(object):
    """
    Class used to add metadata to treelib.Tree during atlas creation.
    Using this means that you can then filter tree nodes depending on
    whether or not they have a mesh/label
    """

    def __init__(self, has_label):
        self.has_label = has_label
