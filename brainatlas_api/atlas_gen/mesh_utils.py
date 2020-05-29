try:
    from vtkplotter import load, write
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Mesh generation with these utils requires vtkplotter\n"
        + '   please install with "pip install vtkplotter -U"'
    )

try:
    import mcubes
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "Mesh generation with these utils requires PyMCubes\n"
        + '   please install with "pip install PyMCubes -U"'
    )


import numpy as np
from pathlib import Path
import scipy


def extract_mesh_from_mask(
    volume,
    obj_filepath,
    smooth=True,
    closing_n_iters=10,
    decimate=True,
    scale=0.975,
):
    """
        Extracts a mesh from a volumetric binary mask 
        and saves it as .obj file.

        Parameters
        ----------
        volume: np.ndarray 
            (3d binary array with 1s where the object is and 0 elsewhere)
        obj_filepath: str or Path object
            path to where the .obj mesh file will be saved
        smooth: bool
            if true mcubes.smooth will be used to smooth the mesh (slow)
        closing_n_iters: int
            number of iterations of closing morphological operation
        decimate: bool
            if true the mesh created by mcubes will be decimated to reduce
            the number of vertices
        scale: float
            the resulting mesh will be scaled to this fraction of the original 
            size. 
    """

    # Check arguments
    if not isinstance(volume, np.ndarray):
        raise ValueError(
            f"Argument volume should be a numpy array not {type(volume)}"
        )
    if np.min(volume) > 0 or np.max(volume) < 1:
        raise ValueError(
            "Argument volume should be a binary mask with only 0s and 1s"
        )

    if isinstance(obj_filepath, str):
        obj_filepath = Path(obj_filepath)

    if not obj_filepath.parents[0].exists():
        raise FileExistsError(
            "The folder where the .obj file is to be saved doesn't exist"
            + f"\n      {str(obj_filepath)}"
        )

    # Apply morphological transformations
    volume = scipy.ndimage.morphology.binary_fill_holes(volume)
    volume = scipy.ndimage.morphology.binary_closing(
        volume, iterations=closing_n_iters
    )

    # Apply marching cubes and save to .obj
    if smooth:
        smooth = mcubes.smooth(volume)
        vertices, triangles = mcubes.marching_cubes(smooth, 0)
    else:
        vertices, triangles = mcubes.marching_cubes(volume, 0.5)
    mcubes.exporter.export_obj(vertices, triangles, obj_filepath)

    # Load .obj and cleanup + save again
    mesh = load(str(obj_filepath))
    if decimate:
        mesh.decimate()

    mesh = mesh.extractLargestRegion().scale(scale)
    write(mesh, str(obj_filepath))
