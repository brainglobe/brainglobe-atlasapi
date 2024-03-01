try:
    from vedo import Mesh, Volume, load, show, write
    from vedo.applications import Browser, Slicer3DPlotter
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
import scipy
from loguru import logger

from brainglobe_atlasapi.atlas_generation.volume_utils import (
    create_masked_array,
)

# ----------------- #
#   MESH CREATION   #
# ----------------- #


def region_mask_from_annotation(
    structure_id,
    annotation,
    structures_list,
):
    """Generate mask for a structure from an annotation file
    and a list of structures.

    Parameters
    ----------
    structure_id : int
        id of the structure
    annotation : np.array
        annotation stack for the atlas
    structures_list : list
        list of structure dictionaries

    Returns
    -------

    """

    mask_stack = np.zeros(annotation.shape, np.uint8)

    for curr_structure in structures_list:
        if structure_id in curr_structure["structure_id_path"]:
            mask_stack[annotation == curr_structure["id"]] = 1

    return mask_stack


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
    if np.min(volume) > 0 or np.max(volume) < 1:
        raise ValueError(
            "Argument volume should be a binary mask with only "
            "0s and 1s when passing a np.ndarray"
        )

    # Apply morphological transformations
    if closing_n_iters is not None:
        volume = scipy.ndimage.morphology.binary_fill_holes(volume).astype(int)
        volume = scipy.ndimage.morphology.binary_closing(
            volume, iterations=closing_n_iters
        ).astype(int)

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
            smooth = mcubes.smooth(volume)
            vertices, triangles = mcubes.marching_cubes(smooth, 0)
        else:
            vertices, triangles = mcubes.marching_cubes(volume, 0.5)

        #  create mesh
        mesh = Mesh((vertices, triangles))

    # Cleanup and save
    if extract_largest:
        mesh = mesh.extractLargestRegion()

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


class Region(object):
    """
    Class used to add metadata to treelib.Tree during atlas creation.
    Using this means that you can then filter tree nodes depending on
    whether or not they have a mesh/label
    """

    def __init__(self, has_label):
        self.has_label = has_label


# ------------------- #
#   MESH INSPECTION   #
# ------------------- #
def compare_mesh_and_volume(mesh, volume):
    """
    Creates and interactive vedo
    visualisation to look at a reference volume
    and a mesh at the same time. Can be used to
    assess the quality of the mesh extraction.

    Parameters:
    -----------

    mesh: vedo Mesh
    volume: np.array or vtkvedoplotter Volume
    """
    if isinstance(volume, np.ndarray):
        volume = Volume(volume)

    vp = Slicer3DPlotter(volume, bg2="white", showHisto=False)
    vp.add(mesh.alpha(0.5))
    vp.show()


def inspect_meshes_folder(folder):
    """
    Used to create an interactive vedo visualisation
    to scroll through all .obj files saved in a folder

    Parameters
    ----------
    folder: str or Path object
        path to folder with .obj files
    """

    if isinstance(folder, str):
        folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError("The folder passed doesnt exist")

    mesh_files = folder.glob("*.obj")

    Browser([load(str(mf)).c("w").lw(0.25).lc("k") for mf in mesh_files])
    logger.debug("visualization ready")
    show()


if __name__ == "__main__":
    folder = (
        r"C:\Users\Federico\.brainglobe\temp\allen_human_500um_v0.1\meshes"
    )
    inspect_meshes_folder(folder)
