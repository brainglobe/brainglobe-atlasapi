try:
    from vtkplotter import Mesh, write, load, show, Volume
    from vtkplotter.applications import Browser, Slicer
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


# ---------------------------------------------------------------------------- #
#                                 MESH CREATION                                #
# ---------------------------------------------------------------------------- #


def extract_mesh_from_mask(
    volume,
    obj_filepath=None,
    threshold=0.5,
    smooth=False,
    mcubes_smooth=False,
    closing_n_iters=8,
    decimate=True,
    use_marching_cubes=False,
):
    """ 
        Returns a vtkplotter mesh actor with just the outer surface of a 
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
            it's slower and less accurate than vtkplotter though.
        mcubes_smooth: bool,
            if True mcubes.smooth is used before applying marching cubes
        closing_n_iters: int
            number of iterations of closing morphological operation.
            set to None to avoid applying morphological operations

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
            "Argument volume should be a binary mask with only 0s and 1s when passing a np.ndarray"
        )

    # Apply morphological transformations
    if closing_n_iters is not None:
        volume = scipy.ndimage.morphology.binary_fill_holes(volume)
        volume = scipy.ndimage.morphology.binary_closing(
            volume, iterations=closing_n_iters
        )

    if not use_marching_cubes:
        # Use faster algorithm
        volume = Volume(volume)
        mesh = volume.clone().isosurface(threshold=threshold).cap()
    else:
        print(
            "The marching cubes algorithm might be rotated compared to your volume data"
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
    if smooth:
        mesh.smoothLaplacian()

    if decimate:
        mesh.clean()

    mesh = mesh.extractLargestRegion()

    if obj_filepath is not None:
        write(mesh, str(obj_filepath))

    return mesh


# ---------------------------------------------------------------------------- #
#                                MESH INSPECTION                               #
# ---------------------------------------------------------------------------- #
def compare_mesh_and_volume(mesh, volume):
    """
        Creates and interactive vtkplotter
        visualisation to look at a reference volume
        and a mesh at the same time. Can be used to 
        assess the quality of the mesh extraction. 

        Parameters:
        -----------

        mesh: vtkplotter Mesh
        volume: np.array or vtkplotter Volume
    """
    if isinstance(volume, np.ndarray):
        volume = Volume(volume)

    vp = Slicer(volume, bg2="white", showHisto=False)
    vp.add(mesh.alpha(0.5))
    vp.show()


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
