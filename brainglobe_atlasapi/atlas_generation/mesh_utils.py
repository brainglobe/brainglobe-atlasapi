"""Utility functions for working with meshes."""

import shutil

from brainglobe_utils.general.system import get_num_processes
from rich.progress import track
from scipy.ndimage import binary_closing, binary_fill_holes

from brainglobe_atlasapi.structure_tree_util import (
    get_structures_tree,
    preorder_depth_first_search,
)

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
    Return a vedo mesh actor with just the outer surface of a
    binary mask volume. It's faster though less accurate than
    extract_mesh_from_mask.

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
        (np.issubdtype(volume.dtype, np.integer) or volume.dtype == bool)
        and (np.max(volume) <= 1 and np.min(volume) >= 0)
    ):
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


def _create_region_mesh(
    meshes_dir_path: Path,
    node,
    tree,
    labels,
    annotated_volume,
    ROOT_ID: int,
    closing_n_iters: int,
    decimate_fraction: float,
    smooth: bool,
    verbosity: int = 0,
):
    """
    Create and save an `.obj` mesh for a region and its descendants.

    The mesh is generated from a binary mask built from `node.identifier` and
    the identifiers of all child nodes in `tree`. Only labels present in
    `annotated_volume` are considered. If no matching labels are found, or the
    resulting mask is empty, no mesh is written.

    `annotated_volume` may be provided as an in-memory NumPy array or as a
    path to a zarr store, which will be opened in read mode.

    For the root region (`node.identifier == ROOT_ID`), mesh extraction skips
    the `closing_n_iters` argument. For all other regions, that parameter is
    passed through to `extract_mesh_from_mask`.

    Parameters
    ----------
    meshes_dir_path : Path
        Directory where mesh `.obj` files are written.
    node
        Tree node corresponding to the region whose mesh should be created.
    tree
        Structure hierarchy containing `node` and its descendants.
    labels
        Unique annotation labels present in `annotated_volume`, typically
        `list(np.unique(annotated_volume))`.
    annotated_volume : numpy.ndarray or str or Path
        Annotation volume as a 3D array, or a path to a zarr store containing
        the annotations.
    ROOT_ID : int
        Identifier of the root structure.
    closing_n_iters : int
        Number of morphological closing iterations to apply for non-root
        regions during mesh extraction.
    decimate_fraction : float
        Fraction used to decimate the extracted mesh.
    smooth : bool
        Whether to smooth the extracted mesh.
    verbosity : int, optional
        Verbosity level used for debug output.

    Raises
    ------
    TypeError
        If `annotated_volume` is neither a NumPy array nor a path to a zarr
        store.

    Returns
    -------
    None
        Mesh data is written to disk when extraction succeeds.
    """
    if verbosity > 0:
        logger.debug(f"Creating mesh for region {node.identifier}")

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
            raise TypeError(
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
        if verbosity > 0:
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


def create_region_mesh(args):
    """
    Wrap _create_region_mesh which facilitates
    multiprocessing.
    """
    if not isinstance(args, (tuple, list)):
        raise TypeError("args must be a tuple or list")

    return _create_region_mesh(*args)


def construct_meshes_from_annotation(
    save_path: Path,
    volume: np.ndarray,
    structures_list,
    closing_n_iters=2,
    decimate_fraction=0,
    smooth=False,
    parallel: bool = True,
    num_threads: int = -1,
    verbosity: int = 0,
    skip_structure_ids=None,
):
    """
    Retrieve or construct atlas region meshes for a given annotation volume.

    If an atlas is packaged with mesh files, reuse those. Otherwise, construct
    the meshes using the existing volume and structure tree. Returns a
    dictionary mapping structure IDs to their corresponding .obj mesh files.

    Parameters
    ----------
    save_path : Path
        Path to the directory where new mesh files will be saved.
    volume : np.ndarray
        3D annotation volume.
    structures_list : list
        List of structure dictionaries containing id information.
    smooth: bool
        if True the surface mesh is smoothed
    closing_n_iters: int
        number of iterations of closing morphological operation.
        set to None to avoid applying morphological operations
    decimate_fraction: float  in range [0, 1].
        What fraction of the original number of vertices is to be kept.
        EG .5 means that 50% of the vertices are kept,
        the others are removed.
    parallel: bool
        If True, uses multiprocessing to speed up mesh creation
    num_threads: int
        Number of threads to use for parallel processing.
        If -1, threads are set to the maximum number based on
        available memory.
        If > 0, uses that many threads.
    verbosity: int
        Level of verbosity for logging. 0 for no output, 1 for basic info.
    skip_structure_ids: iterable of int or None
        If provided, mesh generation for these structure IDs is skipped.

    Returns
    -------
    dict
        Dictionary of structure IDs and paths to their .obj mesh files.
    """
    if num_threads == 0:
        raise ValueError("Number of threads cannot be 0")

    meshes_dir_path = save_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures_list)
    labels = np.unique(volume).astype(np.int32)

    # Only used for parallel processing
    ann_path = save_path / "temp_annotations.zarr"

    for key, node in tree.nodes.items():
        node.data = Region(key in labels)

    volume_size = volume.size
    if parallel:
        compressor = zarr.codecs.BloscCodec(
            cname="zstd", clevel=6, shuffle=zarr.codecs.BloscShuffle.bitshuffle
        )

        if ann_path.exists():
            shutil.rmtree(ann_path)

        ann_store = zarr.storage.LocalStore(ann_path)
        zarr.create_array(
            ann_store,
            data=volume,
            compressors=compressor,
        )
        ann_store.close()
        volume = ann_path

    root_id = tree.root

    # Normalise skip set so filtering is a simple membership check
    if skip_structure_ids is None:
        skip_structure_ids = set()
    elif not isinstance(skip_structure_ids, set):
        skip_structure_ids = set(skip_structure_ids)

    # Create a list of arguments for each region's mesh creation,
    # filtering out structures that should be skipped upstream to
    # avoid unnecessary inter-process communication.
    args_list = [
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
            verbosity,
        )
        for node in preorder_depth_first_search(tree)
        if node.identifier not in skip_structure_ids
    ]

    if parallel:
        if num_threads == -1:
            # Each thread uses ~ 7 times the number of voxels in the volume.
            mem_per_thread = 7 * volume_size
            num_threads = get_num_processes(
                ram_needed_per_process=mem_per_thread,
                n_max_processes=mp.cpu_count() - 1,
                fraction_free_ram=0.05,
            )
            logger.info(f"Using {num_threads} threads for mesh creation")

        with mp.Pool(num_threads) as pool:
            for _ in track(
                pool.imap(create_region_mesh, args_list),
                total=len(args_list),
                description="Creating meshes",
            ):
                pass

        shutil.rmtree(ann_path)  # Clean up temporary annotations zarr store
    else:
        for args in track(
            args_list, total=len(args_list), description="Creating meshes"
        ):
            _create_region_mesh(*args)

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
        f"In the end, {len(structures_with_mesh)}"
        " structures with mesh are kept",
    )
    return meshes_dict


class Region(object):
    """
    Class used to add metadata to treelib.Tree during atlas creation.
    Using this means that you can then filter tree nodes depending on
    whether they have a mesh/label.
    """

    def __init__(self, has_label):
        self.has_label = has_label
