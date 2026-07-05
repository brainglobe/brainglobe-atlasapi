"""Package the Danionella cerebrum mixed reference atlas.

This script packages the dc_mixed_hhg6@1.0 reference, MECE segmentation,
and same-space multimodal additional references.
"""

import json
import re
from pathlib import Path

import numpy as np
import pooch
from brainglobe_utils.IO.image import load_any

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata ###

__version__ = 0
ATLAS_NAME = "danionella_cerebrum_mixed"
CITATION = (
    "Kadobianskyi et al. 2026, bioRxiv, "
    "https://doi.org/10.64898/2026.03.09.710483"
)
SPECIES = "Danionella cerebrum"
ATLAS_LINK = "https://gin.g-node.org/danionella/dc_atlas"
ORIENTATION = "lps"
ROOT_ID = 9999
RESOLUTION = 2.5
ATLAS_PACKAGER = "Amirreza Bahramani"

SKIP_DOWNLOADS_IF_PRESENT = False

GIN_RAW_BASE_URL = f"{ATLAS_LINK}/raw/master"

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR_PATH = BG_ROOT_DIR / "downloads"

REFERENCE_FNAME = "reference.nii.gz"
ANNOTATION_FNAME = "segmentation.nii.gz"
HIERARCHY_FNAME = "dc_labels.json"
LABELS_FNAME = "dc_label_descriptions.txt"
CATALOG_FNAME = "dataset_catalog.json"

SOURCE_FILES = {
    REFERENCE_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/template/reference.nii.gz"
    ),
    ANNOTATION_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/segmentation/MECE/1.0/"
        "segmentation.nii.gz"
    ),
    HIERARCHY_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/segmentation/MECE/1.0/"
        "dc_labels.json"
    ),
    LABELS_FNAME: (
        "atlases/dc_mixed_hhg6/1.0/segmentation/MECE/1.0/"
        "dc_label_descriptions.txt"
    ),
    CATALOG_FNAME: "catalog/dataset_catalog.json",
}

# Same-space 33 additional references from the Danionella GIN catalog:
# registered.space == "dc_mixed_hhg6@1.0", checked 2026-07-05.
# Source: https://gin.g-node.org/danionella/dc_atlas/raw/master/catalog/dataset_catalog.json
ADDITIONAL_REFERENCE_FILES = (
    (
        "structural_confocal_reflectance",
        "datasets/structural/fiber_tracts/confocalreflectance_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/confocalreflectance_jlab_01.nii.gz",
    ),
    (
        "functional_auditory_density",
        "datasets/functional/lightsheet/auditory_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/auditory_jlab_01.nii.gz",
    ),
    (
        "functional_visual_density",
        "datasets/functional/lightsheet/visual_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/visual_jlab_01.nii.gz",
    ),
    (
        "hcr_adcyap1",
        "datasets/molecular/hcr/adcyap1_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/adcyap1_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_ar",
        "datasets/molecular/hcr/ar_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/ar_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_avp",
        "datasets/molecular/hcr/avp_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/avp_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_cart2",
        "datasets/molecular/hcr/cart2_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/cart2_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_chata",
        "datasets/molecular/hcr/chata_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/chata_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_dbh",
        "datasets/molecular/hcr/dbh_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/dbh_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_ddc",
        "datasets/molecular/hcr/ddc_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/ddc_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_emx1",
        "datasets/molecular/hcr/emx1_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/emx1_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_emx3",
        "datasets/molecular/hcr/emx3_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/emx3_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_eomesa",
        "datasets/molecular/hcr/eomesa_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/eomesa_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_gad1b",
        "datasets/molecular/hcr/gad1b_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/gad1b_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_galn",
        "datasets/molecular/hcr/galn_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/galn_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_gbx2",
        "datasets/molecular/hcr/gbx2_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/gbx2_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_isl1a",
        "datasets/molecular/hcr/isl1a_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/isl1a_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_nmbb",
        "datasets/molecular/hcr/nmbb_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/nmbb_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_npy",
        "datasets/molecular/hcr/npy_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/npy_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_nts",
        "datasets/molecular/hcr/nts_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/nts_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_oxt",
        "datasets/molecular/hcr/oxt_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/oxt_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_pdyn",
        "datasets/molecular/hcr/pdyn_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/pdyn_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_penka",
        "datasets/molecular/hcr/penka_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/penka_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_pomca",
        "datasets/molecular/hcr/pomca_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/pomca_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_prl",
        "datasets/molecular/hcr/prl_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/prl_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_pth2",
        "datasets/molecular/hcr/pth2_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/pth2_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_pyaa",
        "datasets/molecular/hcr/pyaa_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/pyaa_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_rln3a",
        "datasets/molecular/hcr/rln3a_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/rln3a_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_serta",
        "datasets/molecular/hcr/serta_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/serta_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_slc6a3",
        "datasets/molecular/hcr/slc6a3_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/slc6a3_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_tac1",
        "datasets/molecular/hcr/tac1_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/tac1_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_vglut2a",
        "datasets/molecular/hcr/vglut2a_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/vglut2a_hcr_jlab_01.nii.gz",
    ),
    (
        "hcr_vipa",
        "datasets/molecular/hcr/vipa_hcr_jlab_01/"
        "registered/dc_mixed_hhg6/1.0/vipa_hcr_jlab_01.nii.gz",
    ),
)


def _clean_name_text(text: str) -> str:
    """Normalize source name spacing without changing the label meaning."""
    text = re.sub(r"\s*/\s*", " / ", text.strip())
    return re.sub(r"\s+", " ", text)


def _clean_acronym_text(text: str) -> str:
    """Normalize source acronym spacing."""
    return re.sub(r"\s*/\s*", "/", text.strip())


def _parent_acronym_from_name(name: str) -> str:
    """Create a stable acronym key for non-annotated hierarchy parents."""
    acronym = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    if not acronym:
        raise ValueError("Cannot create an acronym from an empty name.")
    return acronym


def _additional_reference_diagnostics(name, stack, reference_shape):
    """Validate and summarize an additional reference volume."""
    if stack.shape != reference_shape:
        raise ValueError(
            f"{name} shape does not match reference: "
            f"{stack.shape} != {reference_shape}"
        )

    if not np.all(np.isfinite(stack)):
        raise ValueError(f"{name} contains non-finite values.")

    nonzero_voxels = np.count_nonzero(stack)
    negative_voxels = np.count_nonzero(stack < 0)
    positive_subunit_voxels = np.count_nonzero((stack > 0) & (stack < 1))
    positive_subunit_fraction = (
        positive_subunit_voxels / nonzero_voxels
        if nonzero_voxels
        else 0
    )

    return {
        "name": name,
        "shape": stack.shape,
        "dtype": stack.dtype,
        "min": float(np.min(stack)),
        "max": float(np.max(stack)),
        "nonzero_fraction": nonzero_voxels / stack.size,
        "negative_voxels": negative_voxels,
        "positive_subunit_fraction": positive_subunit_fraction,
    }


def _normalize_additional_reference_to_uint16(stack):
    """Clip negatives and use each reference's full uint16 display range."""
    # BrainGlobe saves additional references through the template stack path,
    # which uses uint16. Convert explicitly so raw negative/subunit floats are
    # not silently cast into a misleading display image during wrapup.
    np.maximum(stack, 0, out=stack)
    clipped_max = float(np.max(stack))

    if clipped_max == 0:
        return np.zeros(stack.shape, dtype=np.uint16), clipped_max

    stack *= np.iinfo(np.uint16).max / clipped_max
    np.rint(stack, out=stack)

    return stack.astype(np.uint16), clipped_max


def download_resources():
    """Download the mixed reference, annotation, and label metadata."""
    BG_ROOT_DIR.mkdir(exist_ok=True, parents=True)
    DOWNLOAD_DIR_PATH.mkdir(exist_ok=True)

    source_paths = {
        fname: DOWNLOAD_DIR_PATH / fname for fname in SOURCE_FILES
    }
    needs_download = any(not path.exists() for path in source_paths.values())

    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    for fname, source_path in SOURCE_FILES.items():
        destination_path = source_paths[fname]
        if not should_fetch(destination_path):
            continue

        pooch.retrieve(
            url=f"{GIN_RAW_BASE_URL}/{source_path}",
            known_hash=None,
            path=DOWNLOAD_DIR_PATH,
            fname=fname,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    If possible, use brainglobe_utils.IO.image.load_any for opening images.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    reference_path = DOWNLOAD_DIR_PATH / REFERENCE_FNAME
    annotation_path = DOWNLOAD_DIR_PATH / ANNOTATION_FNAME

    reference = load_any(reference_path, as_numpy=True)
    annotation = load_any(annotation_path, as_numpy=True)

    if reference.shape != annotation.shape:
        raise ValueError(
            "Reference and annotation shapes do not match: "
            f"{reference.shape} != {annotation.shape}"
        )

    reference = np.clip(
        reference, 0, np.iinfo(np.uint16).max
    ).astype(np.uint16)

    if annotation.min() < 0:
        raise ValueError("Annotation contains negative label IDs.")

    annotation_uint = annotation.astype(np.uint32)
    if not np.array_equal(annotation, annotation_uint):
        raise ValueError("Annotation contains non-integer label values.")

    annotation = annotation_uint

    return reference, annotation


def retrieve_hemisphere_map():
    """
    Return no source-provided hemisphere map for this atlas.

    The mixed Danionella hierarchy and ITK-SNAP label file use bilateral
    structure labels. They do not provide left/right-specific IDs or a
    hemisphere stack, so BrainGlobe should use symmetric hemisphere handling.

    Returns
    -------
    None
        No hemisphere map is provided by the source atlas.
    """
    return None


def retrieve_structure_information():
    """
    Return a list of dictionaries with information about the atlas.

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

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
    """
    hierarchy_path = DOWNLOAD_DIR_PATH / HIERARCHY_FNAME

    with open(hierarchy_path) as hierarchy_file:
        source_hierarchy = json.load(hierarchy_file)

    if source_hierarchy["id"] != ROOT_ID:
        raise ValueError(
            f"Expected root ID {ROOT_ID}, got {source_hierarchy['id']}."
        )

    structures = []
    structure_ids = set()
    acronyms = set()

    def add_structure(node, structure_id_path):
        structure_id = int(node["id"])
        if structure_id in structure_ids:
            raise ValueError(f"Duplicate structure ID: {structure_id}.")
        structure_ids.add(structure_id)

        name = _clean_name_text(node["name"])
        if structure_id == ROOT_ID:
            name = "root"
            acronym = "root"
            rgb_triplet = [255, 255, 255]
        elif node.get("is_segmentation"):
            acronym = _clean_acronym_text(node["abbreviation"])
            rgb_triplet = node["rgb"]
        else:
            # Source parent nodes do not have abbreviations. Keep source
            # abbreviations for annotated leaves and generate deterministic
            # slug-like acronyms only for hierarchy/grouping parents.
            acronym = _parent_acronym_from_name(name)
            rgb_triplet = node["rgb"]

        if acronym in acronyms:
            raise ValueError(f"Duplicate structure acronym: {acronym}.")
        acronyms.add(acronym)

        current_path = [*structure_id_path, structure_id]
        structures.append(
            {
                "id": structure_id,
                "name": name,
                "acronym": acronym,
                "structure_id_path": current_path,
                "rgb_triplet": rgb_triplet,
            }
        )

        for child in node.get("children", []):
            add_structure(child, current_path)

    add_structure(source_hierarchy, [])

    return structures


def retrieve_or_construct_meshes(annotated_volume, structures):
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
    meshes_dict = construct_meshes_from_annotation(
        save_path=DOWNLOAD_DIR_PATH,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=2,
        decimate_fraction=0.2,
        smooth=True,
        parallel=True,
        num_threads=6,
        verbosity=0,
    )

    return meshes_dict


def retrieve_additional_references(
    reference_shape=None, print_diagnostics=False
):
    """
    Return a dictionary of additional reference images.

    All additional references included here are source-registered to the
    mixed reference space, dc_mixed_hhg6@1.0.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    if reference_shape is None:
        reference_shape = load_any(
            DOWNLOAD_DIR_PATH / REFERENCE_FNAME, as_numpy=True
        ).shape

    needs_download = any(
        not (DOWNLOAD_DIR_PATH / f"{name}.nii.gz").exists()
        for name, _ in ADDITIONAL_REFERENCE_FILES
    )
    if needs_download:
        utils.check_internet_connection()

    def should_fetch(path: Path) -> bool:
        if not path.exists():
            return True
        return not SKIP_DOWNLOADS_IF_PRESENT

    additional_references = {}

    for name, source_path in ADDITIONAL_REFERENCE_FILES:
        fname = f"{name}.nii.gz"
        destination_path = DOWNLOAD_DIR_PATH / fname

        if should_fetch(destination_path):
            pooch.retrieve(
                url=f"{GIN_RAW_BASE_URL}/{source_path}",
                known_hash=None,
                path=DOWNLOAD_DIR_PATH,
                fname=fname,
                progressbar=True,
            )

        reference = load_any(destination_path, as_numpy=True)
        diagnostics = _additional_reference_diagnostics(
            name, reference, reference_shape
        )
        reference, clipped_max = _normalize_additional_reference_to_uint16(
            reference
        )

        if print_diagnostics:
            print(
                f"{diagnostics['name']}: "
                f"{diagnostics['shape']} {diagnostics['dtype']} "
                f"min={diagnostics['min']:.6g} "
                f"max={diagnostics['max']:.6g} "
                "nonzero="
                f"{diagnostics['nonzero_fraction']:.2%} "
                f"negative_voxels={diagnostics['negative_voxels']} "
                "positive_subunit_nonzero="
                f"{diagnostics['positive_subunit_fraction']:.2%} "
                f"converted={reference.dtype} "
                f"converted_min={reference.min()} "
                f"converted_max={reference.max()} "
                f"scale_max={clipped_max:.6g}"
            )
        additional_references[name] = reference

    return additional_references


if __name__ == "__main__":
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)
    additional_references = retrieve_additional_references(
        reference_shape=reference_volume.shape,
        print_diagnostics=False,
    )

    print(
        "additional references: "
        f"{len(additional_references)} loaded, shape-checked, and converted"
    )

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
    )

    print(f"Atlas packaged at: {output_filename}")
