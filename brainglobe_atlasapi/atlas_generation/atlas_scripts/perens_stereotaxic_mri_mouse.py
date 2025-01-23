import multiprocessing as mp
import os
import time
import urllib.request
from pathlib import Path

import numpy as np
import py7zr
from brainglobe_utils.IO.image import load_any
from rich.progress import Progress, track

from brainglobe_atlasapi import BrainGlobeAtlas, utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

# The Perens atlas re-uses information from the Allen atlas, so it's useful to
# have an instance of the Allen atlas around
allen_atlas = BrainGlobeAtlas("allen_mouse_25um")
PARALLEL = True  # disable parallel mesh extraction for easier debugging

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(Windows NT 10.0; "
        "Win64; x64; "
        "rv:129.0) "
        "Gecko/20100101 "
        "Firefox/129.0"
    ),
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "image/avif,"
        "image/webp,"
        "image/png,"
        "image/svg+xml,"
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-GB,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "DNT": "1",
    "Sec-GPC": "1",
    "Host": "www.neuropedia.dk",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "TE": "trailers",
    "Priority": "u=0, i",
}


# Copy-paste this script into a new file and fill in the functions to package
# your own atlas.

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "perens_stereotaxic_mri_mouse"

# DOI of the most relevant citable document
CITATION = "Perens et al. 2023, https://doi.org/10.1007/s12021-023-09623-9"

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mus musculus"

# The URL for the data files
ATLAS_LINK = "https://www.neuropedia.dk/resource/multimodal-3d-mouse-brain-atlas-framework-with-the-skull-derived-coordinate-system/"
ATLAS_FILE_URL = "https://www.neuropedia.dk/wp-content/uploads/Multimodal_mouse_brain_atlas_files_v2.7z"
# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "lai"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 25

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME


def download_resources():
    """
    Download the necessary resources for the atlas.

    If possible, please use the Pooch library to retrieve any resources.
    """
    download_dir_path = BG_ROOT_DIR / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_files_dir = download_dir_path / "atlas_files"

    ## Download atlas_file
    utils.check_internet_connection()

    destination_path = download_dir_path / "atlas_download.7z"
    if not os.path.isdir(
        atlas_files_dir / "Multimodal_mouse_brain_atlas_files"
    ):
        req = urllib.request.Request(ATLAS_FILE_URL, headers=HEADERS)
        with (
            urllib.request.urlopen(req) as response,
            open(destination_path, "wb") as out_file,
        ):
            total = int(response.headers.get("content-length", 0))
            with Progress() as progress:
                task = progress.add_task("[cyan]Downloading...", total=total)
                while not progress.finished:
                    chunk = response.read(1024)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    progress.update(task, advance=len(chunk))
        with py7zr.SevenZipFile(destination_path, mode="r") as z:
            z.extractall(path=atlas_files_dir)
        destination_path.unlink()


def retrieve_reference_and_annotation():
    """
    Retrieve the desired reference and annotation as two numpy arrays.

    Returns:
        tuple: A tuple containing two numpy arrays. The first array is the
        reference volume, and the second array is the annotation volume.
    """
    print("loading reference and annotation volume")
    download_dir_path = BG_ROOT_DIR / "downloads"
    atlas_files_dir = download_dir_path / "atlas_files"

    annotations_file = (
        atlas_files_dir
        / "Multimodal_mouse_brain_atlas_files"
        / "MRI_space_oriented"
        / "mri_ano.nii.gz"
    )
    reference_file = (
        atlas_files_dir
        / "Multimodal_mouse_brain_atlas_files"
        / "MRI_space_oriented"
        / "mri_temp.nii.gz"
    )

    annotated_volume = load_any(annotations_file)
    reference_volume = load_any(reference_file)

    return reference_volume, annotated_volume


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    If your atlas is asymmetrical, you may want to use a hemisphere map.
    This is an array in the same shape as your template,
    with 0's marking the left hemisphere, and 1's marking the right.

    If your atlas is symmetrical, ignore this function.

    Returns:
        numpy.array or None: A numpy array representing the hemisphere map,
        or None if the atlas is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Retrieve the structures tree and meshes for the Allen mouse brain atlas.

    Returns:
        pandas.DataFrame: A DataFrame containing the atlas information.
    """
    # Since this atlas inherits from the allen can we not simply get the data
    # from the bgapi?
    print("determining structures")
    allen_atlas = BrainGlobeAtlas("allen_mouse_25um")
    allen_structures = allen_atlas.structures_list
    allen_structures = [
        {
            "id": i["id"],
            "name": i["name"],
            "acronym": i["acronym"],
            "structure_id_path": i["structure_id_path"],
            "rgb_triplet": i["rgb_triplet"],
        }
        for i in allen_structures
    ]
    return allen_structures


def retrieve_or_construct_meshes():
    """
    This function should return a dictionary of ids and corresponding paths to
    mesh files. Some atlases are packaged with mesh files, in these cases we
    should use these files. Then this function should download those meshes.
    In other cases we need to construct the meshes ourselves. For this we have
    helper functions to achieve this.
    """
    print("constructing meshes")

    download_dir_path = BG_ROOT_DIR / "downloads"
    meshes_dir_path = download_dir_path / "meshes"
    meshes_dir_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures)

    labels = np.unique(annotated_volume).astype(np.int32)
    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False

        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2  # not used for this atlas
    decimate_fraction = 0.2  # not used for this atlas

    smooth = False
    start = time.time()
    if PARALLEL:
        pool = mp.Pool(mp.cpu_count() - 2)

        try:
            pool.map(
                create_region_mesh,
                [
                    (
                        meshes_dir_path,
                        node,
                        tree,
                        labels,
                        annotated_volume,
                        ROOT_ID,
                        closing_n_iters,
                        decimate_fraction,
                        smooth,
                    )
                    for node in tree.nodes.values()
                ],
            )
        except mp.pool.MaybeEncodingError:
            # error with returning results from pool.map but we don't care
            pass
    else:
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
                    annotated_volume,
                    ROOT_ID,
                    closing_n_iters,
                    decimate_fraction,
                    smooth,
                )
            )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )

    # Create meshes dict
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue

        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path

    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )

    return meshes_dict


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(exist_ok=True)
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes()
    print("wrapping up atlas")
    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION,) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=BG_ROOT_DIR,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
    )
