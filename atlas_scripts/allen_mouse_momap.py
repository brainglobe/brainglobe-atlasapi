"""Package the momap atlas: projection-defined modules of mouse motor cortex.

The source data is a single archive on Zenodo that already contains the
annotation, the reference, the structure list and one mesh per structure, so
this script only downloads, unzips and hands them to
``wrapup_atlas_from_data``.
"""

import json
import zipfile

import requests
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR

### Metadata ###

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
# remember to add {ATLAS_NAME}_{RESOLUTION}um to:
# brainglobe_atlasapi/atlas_name.py
ATLAS_NAME = "allen_mouse_momap"

# DOI of the most relevant citable document
CITATION = (
    "Falasconi and Kanodia et al., 2026: Projection-defined modules reveal "
    "mouse motor cortex architecture "
    "https://doi.org/10.1016/j.cell.2026.08.046"
)

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = "Mus musculus"

# The URL for the data files
ATLAS_LINK = "https://zenodo.org/records/21104472"

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 10

# Credit for those responsible for converting the atlas to BrainGlobe format
ATLAS_PACKAGER = "Antonio Falasconi and Harsh Kanodia"

# Zenodo record holding the single source archive
ZENODO_ID = 21104472
ZENODO_FILENAME = "files_momap.zip"
CHUNK_SIZE = 1 << 20  # 1 MiB

# BrainGlobe's own threshold below which a mesh file is considered unusable
MIN_MESH_FILE_SIZE = 100

BG_ROOT_DIR = DEFAULT_WORKDIR / ATLAS_NAME
DOWNLOAD_DIR = BG_ROOT_DIR / "downloads"
SOURCE_DIR = DOWNLOAD_DIR / "files_momap" / "momap_bgatlas"


def download_resources():
    """
    Download and unzip the momap source archive from Zenodo.

    The archive is saved in ``DOWNLOAD_DIR``, extracted there, then deleted, so
    only the extracted files remain. The archive has a single top-level
    ``files_momap/`` folder, which is extracted into ``DOWNLOAD_DIR`` rather
    than into a subfolder of the same name, so it is not nested twice.
    """
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(
        f"https://zenodo.org/api/records/{ZENODO_ID}", timeout=60
    )
    response.raise_for_status()
    files = {
        entry["key"]: entry["links"]["self"]
        for entry in response.json().get("files", [])
    }
    if ZENODO_FILENAME not in files:
        available = ", ".join(sorted(files)) or "(none)"
        raise RuntimeError(
            f"'{ZENODO_FILENAME}' is not on Zenodo record {ZENODO_ID}. "
            f"Available file(s): {available}"
        )

    archive = DOWNLOAD_DIR / ZENODO_FILENAME
    print(f"Downloading {archive.name} ...")
    with requests.get(files[ZENODO_FILENAME], stream=True, timeout=60) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with open(archive, "wb") as out:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                out.write(chunk)
                done += len(chunk)
                if total:
                    print(
                        f"\r  {100.0 * done / total:5.1f}% "
                        f" ({done}/{total} bytes)",
                        end="",
                        flush=True,
                    )
        if total:
            print()

    print(f"Unzipping {archive.name} ...")
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        tops = {name.split("/", 1)[0] for name in names if name.strip()}
        if tops == {archive.stem}:
            dest = DOWNLOAD_DIR
        else:
            dest = archive.with_suffix("")
        dest.mkdir(parents=True, exist_ok=True)
        zf.extractall(dest)

    archive.unlink()

    if not SOURCE_DIR.is_dir():
        extracted = ", ".join(sorted(tops)) or "(nothing)"
        raise FileNotFoundError(
            f"Expected the atlas inputs at {SOURCE_DIR} after unzipping "
            f"{ZENODO_FILENAME}, but the archive contained: {extracted}"
        )


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    Both are read from the extracted Zenodo archive; nothing is downloaded
    from elsewhere.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference = load_any(SOURCE_DIR / "reference.tiff")
    annotation = load_any(SOURCE_DIR / "annotation.tiff")
    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Returns
    -------
    None
        The atlas is symmetrical, so no hemisphere map is needed.
    """
    return None


def retrieve_structure_information():
    """
    Return a list of dictionaries with information about the atlas.

    The structure list ships with the Zenodo archive, already in the format
    expected by ``wrapup_atlas_from_data``: each entry holds ``id``, ``name``,
    ``acronym``, ``structure_id_path`` and ``rgb_triplet``.

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    structures_path = SOURCE_DIR / "structures_list.json"
    if not structures_path.is_file():
        raise FileNotFoundError(
            f"Structures file not found: {structures_path}. "
            "Run download_resources() first."
        )
    try:
        structures = json.loads(structures_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Could not read structures JSON at {structures_path}"
        ) from error
    if not isinstance(structures, list) or not structures:
        raise ValueError(
            f"Structures JSON must contain a non-empty list: {structures_path}"
        )
    return structures


def retrieve_or_construct_meshes(structures):
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    The archive ships one ``<structure_id>.obj`` per structure, so no mesh is
    constructed here. Missing or unusably small meshes are an error: an
    incomplete mesh dictionary would otherwise be packaged into a silently
    incomplete atlas.

    Parameters
    ----------
    structures : list[dict]
        The structure list returned by ``retrieve_structure_information``.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dir = SOURCE_DIR / "meshes"
    meshes_dict = {}
    missing = []
    for structure in structures:
        mesh_path = meshes_dir / f"{structure['id']}.obj"
        if (
            mesh_path.is_file()
            and mesh_path.stat().st_size >= MIN_MESH_FILE_SIZE
        ):
            meshes_dict[structure["id"]] = mesh_path
        else:
            missing.append(structure["id"])

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {len(structures)} structures have no usable "
            f"mesh in {meshes_dir}: {missing}"
        )

    return meshes_dict


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    Returns
    -------
    dict
        Empty: this atlas has a single reference image.
    """
    return {}


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(structures)

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
        atlas_packager=ATLAS_PACKAGER,
        hemispheres_stack=hemispheres_stack,
        scale_meshes=True,
        additional_references=additional_references,
        overwrite=True,
    )
    print(f"Atlas packaged: {output_filename}")
