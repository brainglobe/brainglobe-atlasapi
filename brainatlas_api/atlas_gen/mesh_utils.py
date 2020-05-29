try:
    from vtkplotter import Mesh, write, load, show, Volume
    from vtkplotter.applications import Browser
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


def extract_mesh_from_mask_fast(
    volume, obj_filepath=None, threshold=0.5, smooth=False
):
    """ 
        Returns a vtkplotter mesh actor with just the outer surface of a 
        binary mask volume. It's faster though less accurate than 
        extract_mesh_from_mask

        
        Parameters
        ----------
        obj_filepath: str or Path object
            path to where the .obj mesh file will be saved
        volume: 3d np.ndarray or vtkplotter.Volume
        threshold: float
            min value to threshold the volume for isosurface extraction
        smooth: bool
            if True the surface mesh is smoothed

    """
    if isinstance(volume, np.ndarray):
        if np.min(volume) > 0 or np.max(volume) < 1:
            raise ValueError(
                "Argument volume should be a binary mask with only 0s and 1s when passing a np.ndarray"
            )
        volume = Volume(volume)

    if not isinstance(volume, Volume):
        raise TypeError(
            f"volume argument should be an instance of Volume or np.ndarray, not {type(volume)}"
        )

    if obj_filepath is not None:
        if isinstance(obj_filepath, str):
            obj_filepath = Path(obj_filepath)

        if not obj_filepath.parents[0].exists():
            raise FileExistsError(
                "The folder where the .obj file is to be saved doesn't exist"
                + f"\n      {str(obj_filepath)}"
            )

    mesh = volume.clone().isosurface(threshold=threshold).cap()

    if smooth:
        mesh.smoothLaplacian()

    if obj_filepath is not None:
        write(mesh, str(obj_filepath))
    return mesh


def extract_mesh_from_mask(
    volume,
    obj_filepath=None,
    smooth=True,
    smooth_mesh=False,
    closing_n_iters=10,
    decimate=True,
    scale=0.975,
):
    """
        Extracts a mesh from a volumetric binary mask 
        and saves it as .obj file.
        NOTE: this might not work well with regions that
        are made of 2 separate volumes.

        Parameters
        ----------
        volume: np.ndarray 
            (3d binary array with 1s where the object is and 0 elsewhere)
        obj_filepath: str or Path object
            path to where the .obj mesh file will be saved
        smooth: bool
            if true mcubes.smooth will be used to smooth the volume before marching cubes (slow)
        closing_n_iters: int
            number of iterations of closing morphological operation
        decimate: bool
            if true the mesh created by mcubes will be decimated to reduce
            the number of vertices
        smooth_mesh: bool
            if true the mesh is smoothed with the Laplacian method.
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

    if obj_filepath is not None:
        if isinstance(obj_filepath, str):
            obj_filepath = Path(obj_filepath)

        if not obj_filepath.parents[0].exists():
            raise FileExistsError(
                "The folder where the .obj file is to be saved doesn't exist"
                + f"\n      {str(obj_filepath)}"
            )

    # Apply morphological transformations
    volume = scipy.ndimage.morphology.binary_fill_holes(volume)
    if closing_n_iters:
        volume = scipy.ndimage.morphology.binary_closing(
            volume, iterations=closing_n_iters
        )

    # Apply marching cubes and save to .obj
    if smooth:
        smooth = mcubes.smooth(volume)
        vertices, triangles = mcubes.marching_cubes(smooth, 0)
    else:
        vertices, triangles = mcubes.marching_cubes(volume, 0.5)

    #  create mesh and cleanup + save
    mesh = Mesh((vertices, triangles))

    if decimate:
        mesh.clean()

    if smooth_mesh:
        mesh.smoothWSinc()

    mesh = mesh.extractLargestRegion().scale(scale)

    if obj_filepath is not None:
        write(mesh, str(obj_filepath))
    return mesh


def inspect_meshses_folder(folder):
    """
        Used to create an interactive vtkplotter visualisation
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

    Browser(load(str(folder)))
    show()


if __name__ == "__main__":
    folder = (
        r"C:\Users\Federico\.brainglobe\temp\allen_human_500um_v0.1\meshes"
    )
    inspect_meshses_folder(folder)
