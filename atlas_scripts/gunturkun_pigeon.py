"""Template script for generating a BrainGlobe atlas.

Use this script as a starting point to package a new BrainGlobe atlas by
filling in the required functions and metadata.
"""

import os
from pathlib import Path

import nibabel as nib
import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

### Metadata ###
__version__ = 0

ATLAS_NAME = "gunturkun_pigeon"
CITATION = "https://doi.org/10.1007/s00429-012-0400-y"
SPECIES = "Columba livia"

ATLAS_LINK = "https://ruhr-uni-bochum.sciebo.de/s/el9oeWDkMtczWDx"
ATLAS_DOWNLOAD_URL = "https://ruhr-uni-bochum.sciebo.de/public.php/dav/files/el9oeWDkMtczWDx/Full_package/?accept=zip"
ATLAS_DOWNLOAD_FNAME = "Full_package.zip"

SOURCE_ORIENTATION = "ipl"
SOURCE_RESOLUTION = (100, 80, 80)  # in microns

ORIENTATION = "asr"
RESOLUTION = (80, 100, 80)

ROOT_ID = 999

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

NON_STRUCTURAL_DIRS = [
    "Brainsurface",
    "CT",
    "T2",
    "T2star",
    r"Descending systems\Wulst",
]

REGION_INDICES = {
    "auditory1.hdr": {
        1: "An",
        2: "La",
        3: "Mc",
        4: "MLD",
        5: "Ov",
        6: "Field L2",
    },
    "auditory2.hdr": {
        1: "OS",
        2: "LLv",
        3: "LLd",
    },
    "arcopallium.hdr": {
        1: "S",
        2: "GP",
        3: "TnA",
        4: "TnA",
    },
    "Olfactory.hdr": {1: "BO", 2: "CPP", 3: "CPi"},
    "GLd-and-rotundus.hdr": {1: "Rt", 2: "GLd", 3: "GLd"},
    "visual-Wulst_HA_HI_HD-until-A13.hdr": {
        1: "HA (v)",
        2: "HI - HD (v)",
    },
    "nBOR-Lentiformis-mesencephali.hdr": {
        1: "nBOR",
        2: "LM",
    },
    "SLu-Ipc-Imc-left.hdr": {
        1: "Imc",
        2: "Ipc",
        3: "SLu",
    },
    "PrV-and-Basalis.hdr": {
        1: "PrV",
        2: "Bas",
    },
    "Wulst_HA_HI_HD-frontal-from-A13.hdr": {
        1: "HA (s)",
        2: "HI - HD (s)",
    },
    "GC_DLP_DIVA.hdr": {
        1: "GC",
        2: "DLP",
        3: "DIVA",
    },
    "hippocampus.hdr": {
        1: "H",
    },
    "Nucleus-Taeniae.hdr": {
        1: "TnA"
    }
}

# For HA, HD, HI and IHA regions, additional hierarchical information is
# provided in parentheses due to repeated acronym usage in the original atlas.

ACRONYMS = {
    # Visual Systems
    "E": "entopallium",
    "GLd": "n. geniculatus lateralis pars dorsalis",
    # "HA (v)": "hyperpallium apicale (visual)",
    "HI - HD (v)": "hyperpallium intercalatum - dorsale (visual)",
    "Imc": "n. isthmi pars magnocellularis",
    "IHA (v)": "interstitial nucleus of the HA (visual)",
    "IO": "n. isthmo-opticus",
    "Ipc": "n. isthmi pars parvocellularis",
    "LM": "n. lentiformis mesencephali",
    "nBOR": "n. of the basal optic root",
    "PM": "n. pontis medialis",
    "Rt": "n. rotundus",
    "Slu": "n. semilunaris",
    "T": "n. triangularis",
    
    # Somatosensory Systems
    "Bas": "n. basorostralis palii",
    "DIVA": "n. dorsalis intermedius ventralis anterior",
    "DLP": "n. dorsolateralis posterior thalami",
    "Ex": "n. externus",
    "GC": "n. gracilis et cuneatus",
    # "HA (s)": "hyperpallium apicale (somatosensory)",
    "HI - HD (s)": "hyperpallium intercalatum - dorsale (somatosensory)",
    "IHA (s)": "interstitial nucleus of the HA (somatosensory)",
    "PrV": "n. sensorius principalis nervi trigemini",
    
    # Auditory System
    "An": "n. angularis",
    "Field L2": "Field L2",
    "La": "n. laminaris",
    "LLv": "n. of the lateral lemniscus (ventral)",
    "LLd": "n. of the lateral lemniscus (dorsal)",
    "Mc": "n. magnocellularis",
    "MLD": "n. mesencephalicus lateralis pars dorsalis",
    "OS": "oliva superior",
    "Ov": "n. ovoidalis",
    
    # Olfactory System
    "BO": "Bulbus olfactorius",
    "CPi": "cortex piriformis",
    "CPP": "cortex prepiriformis",
    "TnA": "n. taeniae amygdalae",
    
    # Hippocampus
    "H": "hippocampus",
    
    # Descending Systems
    "A": "Arcopallium/amygdala",
    "GP": "globus pallidus",
    "S": "striatum (S)",
    
    # Descending and Visual Systems
    "HA (v)": "hyperpallium apicale (visual)",
    
    # Descending and Somatosensory Systems
    "HA (s)": "hyperpallium apicale (somatosensory)",
}


def download_resources():
    """Download the necessary resources for the atlas with Pooch."""
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    atlas_download_path = DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        else:
            return False

    if should_fetch(atlas_download_path):
        pooch.retrieve(
            url=ATLAS_DOWNLOAD_URL,
            known_hash="0db28c1b3de1e354323740dfc933d9b172b8b0fac3b2b4bac163c26274035375",
            path=DOWNLOAD_DIR_PATH,
            fname=ATLAS_DOWNLOAD_FNAME,
            progressbar=True,
            processor=pooch.Unzip(extract_dir=""),
        )


def retrieve_reference():
    """
    Retrieve the reference volume.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    numpy.ndarray
        The reference volume.
    """
    reference = load_any(
        DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip") / "T2/T2.nii.gz"
    )

    # Remove the superior-most slice of reference, volume, as annotations are in (256, 308, 199)
    # but the reference is in (256, 308, 200).
    reference = np.delete(reference, 199, axis=2).squeeze()
    dmin = np.min(reference)
    dmax = np.max(reference)
    dscale = (2**16 - 1) / (dmax - dmin)
    reference = (reference - dmin) * dscale
    reference = reference.astype(np.uint16)
    return reference


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 1 marks the left hemisphere
    and 2 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    hemisphere_dir = (
        DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip") / "Brainsurface"
    )
    left = nib.load(hemisphere_dir / "brainsurface_left.hdr")
    left_hemisphere = left.get_fdata()

    hemispheres_stack = np.where(left_hemisphere == 0, 2, 1)
    return hemispheres_stack


def retrieve_structure_information(reference_volume: np.ndarray):
    """
    Return a list of dictionaries with information about the atlas, 
    as well as the annotated volume. 

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.
    Also returns a numpy array for the annotated volume. 

    The expected format for each dictionary is:

    .. code-block:: python

        {
            "id": int,
            "name": str,
            "acronym": str,
            "structure_id_path": list[int],
            "rgb_triplet": list[int, int, int],
        }

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
        
    np.ndarrary
        Annotation volume for the atlas, with the IDs for each atlas
        structure
    """
    structures_by_acronym = {
        "root": {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "rgb_triplet": [255, 255, 255],
        }
    }
    annotation_volume = np.zeros(reference_volume.shape)
    startpath = str(DOWNLOAD_DIR_PATH / ATLAS_DOWNLOAD_FNAME.strip(".zip"))
    for root, dirs, files in os.walk(startpath):
        current_dir = root.replace(startpath, "").strip(os.sep)
        if current_dir in NON_STRUCTURAL_DIRS:
            continue
        if not any('.hdr' in f for f in files):
            print(current_dir)
            continue
        level = root.replace(startpath, "").count(os.sep)
        structure_name_path = ["root"]
        for i in range(level):
            structure_name_path.append(root.split(os.sep)[-(level - i)])
        # print(structure_name_path)
        hdrs = list(filter(lambda x: ".hdr" in x, files))
        # print(root)
        for hdr in hdrs:
            print(hdr)
            region_file = nib.load(Path(root) / hdr)
            region_data = np.array(region_file.get_fdata()).astype(np.uint32)
            for i in np.unique(region_data):
                if np.count_nonzero(region_data == i) <= 1:
                    continue 
                if i == 0:
                    continue
                current_acronym = REGION_INDICES.get(hdr).get(i)
                print(i)
                region_data[region_data == i] = list(ACRONYMS).index(current_acronym)
            print(np.unique(region_data))
            
            # Check annotation_volume, 
        # print(hdrs)

    structures = list(structures_by_acronym.values())
    structures.sort(key=lambda s: (len(s["structure_id_path"]), s["id"]))
    return structures, annotation_volume


def retrieve_or_construct_meshes():
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    If the atlas is packaged with mesh files, download and use them. Otherwise,
    construct the meshes using available helper functions.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dict = {}
    return meshes_dict


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    This function should be edited only if the atlas includes additional
    reference images. The dictionary should map the name of each additional
    reference image to its corresponding image stack data.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}
    return additional_references


### If the code above this line has been filled correctly, nothing needs to be
### edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
        )
    download_resources()
    reference_volume = retrieve_reference()
    additional_references = retrieve_additional_references()
    hemispheres_stack = retrieve_hemisphere_map()
    structures, annotated_volume = retrieve_structure_information(reference_volume)
    meshes_dict = retrieve_or_construct_meshes()

    quit()
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
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        additional_references=additional_references,
    )
