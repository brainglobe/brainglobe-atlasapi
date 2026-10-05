"""Package the momap atlas: projection-defined modules of mouse motor cortex.

The source data is a single archive on Zenodo that already contains the
annotation, the reference, the structure list and one mesh per structure, so
this script only downloads, unzips and hands them to
``wrapup_atlas_from_data``.
"""

import json
import zipfile

import pooch
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

ZENODO_FILENAME = "files_momap.zip"
ZENODO_FILE_URL = f"{ATLAS_LINK}/files/{ZENODO_FILENAME}?download=1"
ZENODO_FILE_HASH = (
    "56ad8aeb5259bb78299dfd856b792e5d7874b798c7c93b27d455766ed3c8a8b3"
)

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

    archive = DOWNLOAD_DIR / ZENODO_FILENAME
    pooch.retrieve(
        url=ZENODO_FILE_URL,
        known_hash=ZENODO_FILE_HASH,
        fname=ZENODO_FILENAME,
        path=DOWNLOAD_DIR,
        progressbar=True,
    )

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
    return json.loads(
        (SOURCE_DIR / "structures_list.json").read_text(encoding="utf-8")
    )


def retrieve_or_construct_meshes(structures):
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    The archive ships one ``<structure_id>.obj`` per structure, so no mesh is
    constructed here.

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
    return {
        structure["id"]: meshes_dir / f"{structure['id']}.obj"
        for structure in structures
    }


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
