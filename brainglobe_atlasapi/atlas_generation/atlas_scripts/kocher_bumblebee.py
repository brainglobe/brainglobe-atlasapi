"""Atlas generation script for Kocher bumblebee brain atlas."""

from pathlib import Path

import numpy as np
import pandas as pd
import pooch
import tifffile
import vtk
from vedo import Mesh, write
from vtk.util import numpy_support

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "kocher_bumblebee"

# DOI of the most relevant citable document
CITATION = "doi:10.1016/j.cub.2022.04.066"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Bombus impatiens"

# The URL for the data files
ATLAS_LINK = "https://kocherlab.princeton.edu/"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 0

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
# (.64µm x .64µm x 2.542µm)
RESOLUTION = (
    0.686,
    0.686,
    2.542,
)
ATLAS_PACKAGER = "Scott Wolf"


ANNOTATION_URL = (
    "https://www.dropbox.com/s/wieog31o7qsi2rj/Supplement_Segment.vtk?dl=1"
)
TEMPLATE_URL = (
    "https://www.dropbox.com/s/sh8w84cwqx26lzj/Bombus_template.tif?dl=1"
)

# Bumblebee brain region mapping
BUMBLEBEE_REGION_MAP = [
    ["root", "root"],
    ["1", "Fan Shaped Body"],
    ["2", "Glomeruli"],
    ["3", "Calyx"],
    ["4", "Peduncle"],
    ["5", "Medulla"],
    ["6", "Lamina"],
    ["7", "Lobula"],
    ["8", "NA"],
    ["9", "Ellipsoid Body"],
]


def download_resources(download_dir: Path) -> tuple[Path, Path]:
    """Download the necessary resources for the atlas.

    Parameters
    ----------
    download_dir : Path
        Directory where files will be downloaded.

    Returns
    -------
    tuple[Path, Path]
        Tuple containing paths to annotations and template files.

    Notes
    -----
    If possible, please use the Pooch library to retrieve any resources.
    """
    # Ensure the download directory exists
    download_dir.mkdir(parents=True, exist_ok=True)

    annotations_path = download_dir / "Supplement_Segment.vtk"
    template_path = download_dir / "Bombus_template.tif"

    # Download the VTK file
    pooch.retrieve(
        url=ANNOTATION_URL,
        known_hash="cd5f939fbfb1d1c18713f54dddf4083065b8fa2c3d604a50c2e991071b112a95",
        path=download_dir,
        fname="Supplement_Segment.vtk",
        progressbar=True,
    )

    # Download the TIF file
    pooch.retrieve(
        url=TEMPLATE_URL,
        known_hash="ef9b264cf93f596041f78376ddcc944f26c6eaa438ee47c45a22751c095f1168",
        path=download_dir,
        fname="Bombus_template.tif",
        progressbar=True,
    )

    return annotations_path, template_path


def convert_vtk_to_annotation_tiff(
    vtk_path: Path, template_path: Path, output_path: Path
) -> Path:
    """Convert VTK mesh file to labeled annotation TIFF volume and
    pad to match template dimensions.

    Parameters
    ----------
    vtk_path : Path
        Path to the VTK file
    template_path : Path
        Path to the template file to match dimensions
    output_path : Path
        Path where the converted TIFF should be saved

    Returns
    -------
    Path
        Path to the created annotation TIFF file
    """
    # Load template to get the target dimensions
    template = tifffile.imread(str(template_path))
    template_shape = template.shape  # (Z, Y, X)
    print(f"Template shape: {template_shape}")

    # Use original VTK processing parameters
    spacing = np.array([1.0, 1.0, 1.0])
    origin = np.array([-1800.0, -1340.0, 0.0])
    dims = [1800, 1340, 180]

    print(
        f"Using original VTK spacing: {spacing}, "
        f"origin: {origin}, dims: {dims}"
    )

    # --- Load the mesh ---
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(vtk_path))
    reader.Update()
    polydata = reader.GetOutput()

    # Convert point scalars to cell scalars (if needed)
    point_to_cell = vtk.vtkPointDataToCellData()
    point_to_cell.SetInputData(polydata)
    point_to_cell.Update()
    polydata_with_cell_scalars = point_to_cell.GetOutput()

    # Grab scalar array from cell data
    cell_scalars = numpy_support.vtk_to_numpy(
        polydata_with_cell_scalars.GetCellData().GetScalars()
    )
    unique_labels = np.unique(cell_scalars)
    print(f"Unique labels in mesh: {unique_labels}")

    # Initialize final output volume with original VTK dimensions
    vtk_shape = (dims[2], dims[1], dims[0])  # Z, Y, X
    labeled_volume = np.zeros(vtk_shape, dtype=np.uint16)

    # === Loop through labels and rasterize each ===
    for label in unique_labels:
        if label == 0:
            continue  # Skip background if needed

        print(f"Processing label: {label}")

        # Threshold by current label
        threshold_filter = vtk.vtkThreshold()
        threshold_filter.SetInputData(polydata_with_cell_scalars)
        threshold_filter.SetInputArrayToProcess(
            0,
            0,
            0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            vtk.vtkDataSetAttributes.SCALARS,
        )
        threshold_filter.SetLowerThreshold(label)
        threshold_filter.SetUpperThreshold(label)
        threshold_filter.Update()

        # Convert to polydata
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputConnection(threshold_filter.GetOutputPort())
        geometry_filter.Update()
        filtered_polydata = geometry_filter.GetOutput()

        # Create blank white image
        white_image = vtk.vtkImageData()
        white_image.SetSpacing(spacing)
        white_image.SetOrigin(origin)
        white_image.SetDimensions(dims)
        white_image.SetExtent(0, dims[0] - 1, 0, dims[1] - 1, 0, dims[2] - 1)
        white_image.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
        white_image.GetPointData().GetScalars().Fill(255)

        # Convert to stencil
        pol2stenc = vtk.vtkPolyDataToImageStencil()
        pol2stenc.SetInputData(filtered_polydata)
        pol2stenc.SetOutputOrigin(origin)
        pol2stenc.SetOutputSpacing(spacing)
        pol2stenc.SetOutputWholeExtent(white_image.GetExtent())
        pol2stenc.Update()

        # Apply stencil
        imgstenc = vtk.vtkImageStencil()
        imgstenc.SetInputData(white_image)
        imgstenc.SetStencilConnection(pol2stenc.GetOutputPort())
        imgstenc.ReverseStencilOff()
        imgstenc.SetBackgroundValue(0)
        imgstenc.Update()

        # Extract array and write to final labeled volume
        arr = numpy_support.vtk_to_numpy(
            imgstenc.GetOutput().GetPointData().GetScalars()
        )
        arr = arr.reshape(vtk_shape)
        labeled_volume[arr > 0] = label  # Assign label where mask is set

    # Apply flips as in original code
    labeled_volume = np.flip(labeled_volume, axis=1)
    labeled_volume = np.flip(labeled_volume, axis=2)

    print(f"VTK annotation shape before padding: {labeled_volume.shape}")

    # Now pad/resize the annotation volume to match template dimensions
    if labeled_volume.shape != template_shape:
        # Create output volume with template dimensions
        padded_volume = np.zeros(template_shape, dtype=np.uint16)

        # Pad from origin (0,0,0) to preserve alignment - don't center
        # Calculate how much of the original annotation fits in the template
        copy_z = min(labeled_volume.shape[0], template_shape[0])
        copy_y = min(labeled_volume.shape[1], template_shape[1])
        copy_x = min(labeled_volume.shape[2], template_shape[2])

        # Copy the annotation data starting from (0,0,0) to preserve alignment
        padded_volume[:copy_z, :copy_y, :copy_x] = labeled_volume[
            :copy_z, :copy_y, :copy_x
        ]

        labeled_volume = padded_volume

    # Save the labeled TIFF
    tifffile.imwrite(str(output_path), labeled_volume.astype(np.uint16))
    print(
        f"Labeled volume saved to {output_path}",
        f"Shape: {labeled_volume.shape}",
    )
    print("Label stats:", np.unique(labeled_volume, return_counts=True))

    return output_path


def retrieve_reference_and_annotation(
    annotations_path: Path, template_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Retrieve the desired reference and annotation as two numpy arrays.

    Parameters
    ----------
    annotations_path : Path
        Path to the annotations file
    template_path : Path
        Path to the template file

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    # Load the reference template
    reference = tifffile.imread(str(template_path))

    # Convert VTK to annotation TIFF if it doesn't exist
    annotation_tiff_path = annotations_path.parent / "annotation_volume.tif"
    if not annotation_tiff_path.exists():
        print("Converting VTK mesh to annotation volume...")
        convert_vtk_to_annotation_tiff(
            annotations_path, template_path, annotation_tiff_path
        )
    else:
        # Check if existing annotation matches template dimensions
        existing_annotation = tifffile.imread(str(annotation_tiff_path))
        if existing_annotation.shape != reference.shape:
            print(
                "Existing annotation dimensions don't match template.",
                "Re-converting VTK to annotation volume.",
            )
            convert_vtk_to_annotation_tiff(
                annotations_path, template_path, annotation_tiff_path
            )

    # Load the annotation volume
    annotation = tifffile.imread(str(annotation_tiff_path))

    print(f"Reference shape: {reference.shape}, dtype: {reference.dtype}")
    print(f"Annotation shape: {annotation.shape}, dtype: {annotation.dtype}")

    return reference, annotation


def retrieve_hemisphere_map() -> np.ndarray | None:
    """Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None


def add_hierarchy(labels_df_row: pd.Series) -> list[int]:
    """Take the index at a given row and add the root_id to
    create a structure id path.

    Parameters
    ----------
    labels_df_row : pd.Series
        A pandas Series representing a row from the labels dataframe

    Returns
    -------
    list[int]
        List containing the structure id path with root id
    """
    structure_index = labels_df_row["id"]
    structure_id = [ROOT_ID, structure_index]
    return structure_id


def add_rgb_col_and_heirarchy(labels_df: pd.DataFrame) -> pd.DataFrame:
    """Re-formats df columns, from individual r,g,b,a into the desired [r,g,b].

    Parameters
    ----------
    labels_df : pd.DataFrame
        DataFrame containing label information with r, g, b, alpha columns

    Returns
    -------
    pd.DataFrame
        DataFrame with rgb_triplet and structure_id_path columns added
    """
    rgb_list = []
    structure_list = []
    for _, row in labels_df.iterrows():
        new_rgb_row = [row["r"], row["g"], row["b"]]
        rgb_list.append(new_rgb_row)

        structure_id = add_hierarchy(row)
        structure_list.append(structure_id)

    labels_df = labels_df.drop(columns=["r", "g", "b", "alpha"])
    labels_df["rgb_triplet"] = rgb_list
    labels_df["structure_id_path"] = structure_list
    return labels_df


def create_bumblebee_structure_info() -> list[dict]:
    """Create structure information for bumblebee brain regions based on VTK.

    Returns
    -------
    list[dict]
        A list containing dictionaries of the atlas structure information.
    """
    structures_list = []

    # Add root structure
    root_structure = {
        "id": ROOT_ID,
        "name": "root",
        "acronym": "root",
        "structure_id_path": [ROOT_ID],
        "rgb_triplet": [255, 255, 255],
    }
    structures_list.append(root_structure)

    # Add brain regions based on VTK labels [1, 2, 3, 4, 5, 6, 7, 8, 9]
    region_colors = [
        [255, 100, 100],  # Red-ish for region 1
        [100, 255, 100],  # Green-ish for region 2
        [100, 100, 255],  # Blue-ish for region 3
        [255, 255, 100],  # Yellow-ish for region 4
        [255, 100, 255],  # Magenta-ish for region 5
        [100, 255, 255],  # Cyan-ish for region 6
        [200, 150, 100],  # Brown-ish for region 7
        [150, 200, 100],  # Olive-ish for region 8
        [100, 150, 200],  # Steel blue-ish for region 9
    ]

    region_names = [
        ("1", "Fan Shaped Body"),
        ("2", "Glomeruli"),
        ("3", "Calyx"),
        ("4", "Peduncle"),
        ("5", "Medulla"),
        ("6", "Lamina"),
        ("7", "Lobula"),
        ("8", "NA"),  # Not applicable or not available
        ("9", "Ellipsoid Body"),
    ]

    # Create structures for each region
    for i, (acronym, name) in enumerate(region_names):
        region_id = int(acronym)
        structure = {
            "id": region_id,
            "name": name,
            "acronym": acronym,
            "structure_id_path": [ROOT_ID, region_id],
            "rgb_triplet": region_colors[i],
        }
        structures_list.append(structure)

    return structures_list


def retrieve_structure_information() -> list[dict]:
    """Retrieve atlas structure information.

    Returns
    -------
    list[dict]
        A list containing dictionaries of the atlas structure information.
    """
    return create_bumblebee_structure_info()


def extract_meshes_from_vtk(
    annotations_path: Path, working_dir: Path
) -> dict[str, str]:
    """Extract individual meshes from the VTK file for each labeled region.

    Parameters
    ----------
    annotations_path : Path
        Path to the VTK file
    working_dir : Path
        Working directory where meshes will be saved

    Returns
    -------
    dict[str, str]
        Dictionary mapping region IDs to mesh file paths
    """

    # annotations_path = annotations_path
    # working_dir = bg_root_dir
    mesh_dict = {}
    mesh_save_folder = working_dir / "meshes"
    mesh_save_folder.mkdir(parents=True, exist_ok=True)

    # Load the VTK file
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(annotations_path))
    reader.Update()
    polydata = reader.GetOutput()

    # Convert point scalars to cell scalars (if needed)
    point_to_cell = vtk.vtkPointDataToCellData()
    point_to_cell.SetInputData(polydata)
    point_to_cell.Update()
    polydata_with_cell_scalars = point_to_cell.GetOutput()

    # Get unique labels
    cell_scalars = numpy_support.vtk_to_numpy(
        polydata_with_cell_scalars.GetCellData().GetScalars()
    )
    unique_labels = np.unique(cell_scalars)
    print(f"Extracting meshes for labels: {unique_labels}")

    # Extract mesh for each label
    for label in unique_labels:
        if label == 0:
            continue  # Skip background

        # Convert label to integer to avoid float filenames like "8.0"
        label_int = int(label)
        print(f"Extracting mesh for label: {label_int}")

        # Threshold by current label
        threshold_filter = vtk.vtkThreshold()
        threshold_filter.SetInputData(polydata_with_cell_scalars)
        threshold_filter.SetInputArrayToProcess(
            0,
            0,
            0,
            vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS,
            vtk.vtkDataSetAttributes.SCALARS,
        )
        threshold_filter.SetLowerThreshold(label)
        threshold_filter.SetUpperThreshold(label)
        threshold_filter.Update()

        # Convert to polydata
        geometry_filter = vtk.vtkGeometryFilter()
        geometry_filter.SetInputConnection(threshold_filter.GetOutputPort())
        geometry_filter.Update()
        filtered_polydata = geometry_filter.GetOutput()

        # Save as temporary VTK file and then convert to vedo Mesh
        temp_vtk_path = mesh_save_folder / f"temp_{label_int}.vtk"
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileName(str(temp_vtk_path))
        writer.SetInputData(filtered_polydata)
        writer.Write()

        # Load with vedo and process
        mesh = Mesh(str(temp_vtk_path))

        # Transform mesh coordinates to match annotation volume coordinates
        # Use the same parameters as in convert_vtk_to_annotation_tiff
        origin = np.array([-1800.0, -1340.0, 0.0])
        spacing = np.array([1.0, 1.0, 1.0])
        # VTK dimensions [X, Y, Z]
        dims = [1800, 1340, 180]

        points = mesh.points

        voxel_coords = (points - origin) / spacing

        # Apply the same coordinate transformation as for root mesh
        transformed_points = np.zeros_like(voxel_coords)
        transformed_points[:, 2] = voxel_coords[:, 2]  # Z stays the same
        transformed_points[:, 1] = (dims[1] - 1) - voxel_coords[
            :, 1
        ]  # Y is flipped
        transformed_points[:, 0] = (dims[0] - 1) - voxel_coords[
            :, 0
        ]  # X is flipped

        # Make them Z, Y, X
        transformed_points = transformed_points[
            :, [2, 1, 0]
        ]  # Reorder to [Z, Y, X]

        # Update mesh with transformed points
        mesh.points = transformed_points

        # Apply mesh processing similar to cat atlas
        mesh.triangulate()
        mesh.decimate_pro(0.6)
        mesh.smooth()

        # Save as .obj file
        obj_file_path = mesh_save_folder / f"{label_int}.obj"
        mesh_dict[str(label_int)] = str(obj_file_path)

        if not obj_file_path.exists():
            write(mesh, str(obj_file_path))

        # Clean up temporary file
        temp_vtk_path.unlink(missing_ok=True)

    # Create root mesh from the entire brain
    print("Creating root mesh from entire brain...")
    root_mesh = Mesh(str(annotations_path))

    # Transform root mesh coordinates to match annotation volume coordinates
    origin = np.array([-1800.0, -1340.0, 0.0])
    spacing = np.array([1.0, 1.0, 1.0])
    dims = [1800, 1340, 180]  # VTK dimensions [X, Y, Z]

    # Get mesh points and transform them
    root_points = root_mesh.points

    # Transform from physical coordinates to voxel coordinates
    voxel_coords = (root_points - origin) / spacing

    # Apply the same coordinate transformation as for individual meshes
    transformed_root_points = np.zeros_like(voxel_coords)

    # Transform mesh points to [X, Y, Z]
    transformed_root_points[:, 2] = voxel_coords[:, 2]  # Z stays the same
    transformed_root_points[:, 1] = (dims[1] - 1) - voxel_coords[
        :, 1
    ]  # Y is flipped
    transformed_root_points[:, 0] = (dims[0] - 1) - voxel_coords[
        :, 0
    ]  # X is flipped

    # Reorder to [Z, Y, X] to match annotation volume coordinate system
    transformed_root_points = transformed_root_points[:, [2, 1, 0]]

    print(
        f"Transformed root mesh points shape: {transformed_root_points.shape}"
    )
    # Update root mesh with transformed points
    root_mesh.points = transformed_root_points

    root_mesh.triangulate()
    root_mesh.decimate_pro(0.6)
    root_mesh.smooth()
    # scale
    # root_mesh.scale(1/100.0)  # Scale down by factor of 100

    root_obj_path = mesh_save_folder / f"{ROOT_ID}.obj"
    mesh_dict[str(ROOT_ID)] = str(root_obj_path)

    if not root_obj_path.exists():
        write(root_mesh, str(root_obj_path))

    return mesh_dict


def retrieve_or_construct_meshes(
    annotations_path: Path, working_dir: Path
) -> dict[str, str]:
    """Return a dictionary of ids and corresponding paths to mesh files.

    Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.

    Parameters
    ----------
    annotations_path : Path
        Path to the annotations file
    working_dir : Path
        Working directory where meshes will be saved

    Returns
    -------
    dict[str, str]
        Dictionary mapping region IDs to mesh file paths
    """
    print("Extracting meshes from VTK file...")
    meshes_dict = extract_meshes_from_vtk(annotations_path, working_dir)
    return meshes_dict


def retrieve_additional_references() -> dict:
    """Retrieve additional reference images for the atlas.

    This function only needs editing if the atlas has additional reference
    images. It should return a dictionary that maps the name of each
    additional reference image to an image stack containing its data.

    Returns
    -------
    dict
        Dictionary mapping reference image names to image stacks.
    """
    additional_references = {}
    return additional_references


if __name__ == "__main__":
    # Main execution when script is run directly
    bg_root_dir = Path(".") / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(exist_ok=True)

    # Download resources and get paths
    annotations_path, template_path = download_resources(bg_root_dir)

    reference_volume, annotated_volume = retrieve_reference_and_annotation(
        annotations_path, template_path
    )

    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotations_path, bg_root_dir)

    print(f"Starting atlas wrap-up for {ATLAS_NAME}...")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=RESOLUTION,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=False,
        atlas_packager=ATLAS_PACKAGER,
        additional_references=additional_references,
    )
