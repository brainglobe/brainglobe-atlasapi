import shutil

from scipy.ndimage import binary_closing, binary_fill_holes
from treelib import Tree

from brainglobe_atlasapi.structure_tree_util import preorder_dfs

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

import multiprocessing as mp
from pathlib import Path

import numpy as np
import zarr
from loguru import logger

from brainglobe_atlasapi.atlas_generation.volume_utils import (
    create_masked_array,
)

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
    if not (
        np.issubdtype(volume.dtype, np.integer) or volume.dtype == bool
    ) and not (np.max(volume) <= 1 and np.min(volume) >= 0):
        raise ValueError(
            "Argument volume should be a binary mask with only "
            "0s and 1s when passing a np.ndarray"
        )

    # Apply morphological transformations
    if closing_n_iters is not None:
        volume = binary_fill_holes(volume).astype(np.uint8)
        volume = binary_closing(volume, iterations=closing_n_iters).astype(
            np.uint8
        )

    if not use_marching_cubes:
        # Use faster algorithm
        volume = Volume(volume)
        mesh = volume.isosurface(value=threshold).cap()
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
    region of interest and all of its children's labels and creates a mesh.
    It takes a tuple of arguments to facilitate parallel processing with
    multiprocessing.pool.map

    Note, by default it avoids overwriting a structure's mesh if the
    .obj file exists already.

    Parameters
    ----------
    meshes_dir_path: pathlib Path object with folder where meshes are saved
    tree: treelib.Tree with hierarchical structures information
    node: tree's node corresponding to the region whose mesh is being created
    labels: list of unique label annotations in annotated volume,
    (list(np.unique(annotated_volume)))
    annotated_volume: 3d numpy array path to a zarr store with annotations
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

    if not isinstance(annotated_volume, np.ndarray):
        # If annotated_volume is a path to a zarr store, open it
        if isinstance(annotated_volume, (str, Path)):
            annotated_volume = zarr.open(annotated_volume, mode="r")
        else:
            raise ValueError(
                "Argument annotated_volume should be a np.ndarray"
                " or a path to a zarr store"
            )

    # Get labels for region and it's children
    stree = tree.subtree(node.identifier)
    ids = list(stree.nodes.keys())

    # Keep only labels that are in the annotation volume
    matched_labels = [i for i in ids if i in labels]

    if (
        not matched_labels
    ):  # it fails if the region and all of its children are not in annotation
        print(f"No labels found for {node.tag}")
        return
    else:
        # Create mask and extract mesh
        mask = create_masked_array(annotated_volume, ids)

        if np.sum(mask) == 0:
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


def create_meshes_from_annotated_volume(
    mesh_directory: Path,
    tree: Tree,
    annotated_volume: np.ndarray,
    closing_n_iters: int = 2,
    decimate_fraction: float = 0.2,
    smooth: bool = True,
    parallel: bool = True,
    num_threads: int = -1,
):
    """
    Creates meshes for all regions in the tree from the annotated volume.
    If parallel is uses multiprocessing to speed up the process.

    Parameters
    ----------
    mesh_directory: Path object
        Path to the working directory where meshes will be saved.
    tree: treelib.Tree
        Hierarchical structure of regions.
    annotated_volume: np.ndarray
        3d numpy array with annotated volume
    closing_n_iters: int
        Number of iterations of closing morphological operation.
    decimate_fraction: float  in range [0, 1].
        What fraction of the original number of vertices is to be kept.
        EG .5 means that 50% of the vertices are kept,
        the others are removed.
    smooth: bool
        if True the surface mesh is smoothed
    parallel: bool
        If True, uses multiprocessing to speed up mesh creation
    num_threads: int
        Number of threads to use for parallel processing.
        If -1, threads are set to the number of available cores minus 1.
        If > 0, uses that many threads.

    Returns
    -------
    None, saves meshes to working_dir
    """
    if num_threads == 0:
        raise ValueError("Number of threads cannot be 0")

    meshes_dir_path = mesh_directory
    meshes_dir_path.mkdir(parents=True, exist_ok=True)

    labels = list(np.unique(annotated_volume))
    # Only used for parallel processing
    ann_path = mesh_directory / "temp_annotations.zarr"

    if parallel:
        compressor = zarr.codecs.BloscCodec(
            cname="zstd", clevel=6, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        )

        if ann_path.exists():
            shutil.rmtree(ann_path)

        ann_store = zarr.storage.LocalStore(ann_path)
        zarr.create_array(
            ann_store,
            data=annotated_volume,
            compressors=compressor,
        )
        ann_store.close()
        annotated_volume = ann_path

    root_id = tree.root
    # Create a list of arguments for each region's mesh creation
    args_list = [
        (
            meshes_dir_path,
            node,
            tree,
            labels,
            annotated_volume,
            root_id,
            closing_n_iters,
            decimate_fraction,
            smooth,
        )
        for node in preorder_dfs(tree)
    ]

    if parallel:
        if num_threads == -1:
            num_threads = mp.cpu_count() - 1

        with mp.Pool(num_threads) as pool:
            pool.map(create_region_mesh, args_list, chunksize=1)

        shutil.rmtree(ann_path)  # Clean up temporary annotations zarr store
    else:
        for args in args_list:
            create_region_mesh(args)


class Region(object):
    """
    Class used to add metadata to treelib.Tree during atlas creation.
    Using this means that you can then filter tree nodes depending on
    whether they have a mesh/label
    """

    def __init__(self, has_label):
        self.has_label = has_label
