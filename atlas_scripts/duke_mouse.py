"""Package the Duke Mouse Brain Atlas for BrainGlobe."""

import csv
import zipfile
from pathlib import Path

import numpy as np
import pooch
import SimpleITK as sitk

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    construct_meshes_from_annotation,
)
from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import atlas_name_from_repr

__version__ = 0
ATLAS_NAME = "duke_mouse"
CITATION = "Mansour et al. 2025, " "https://doi.org/10.1126/sciadv.adq8089"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://civmimagespace.civm.duhs.duke.edu/"
    "tp_item_detail.php/view/item_number=DMBA/set_id=315"
)

ORIENTATION = "ipr"

ROOT_ID = 997
RESOLUTION = 15

ATLAS_PACKAGER = "Amirreza Bahramani"

SOURCE_DATA_DIR = (
    Path.home() / "brainglobe_workingdir" / ATLAS_NAME / "source_data"
)
REFERENCE_PATH = SOURCE_DATA_DIR / "DMBA_md" / "DMBA_N06_md_M4D.nhdr"
ADDITIONAL_REFERENCE_PATHS = {
    "m2": SOURCE_DATA_DIR / "DMBA_m2" / "DMBA_N13_m2_M4D.nhdr",
}
ANNOTATION_PATH = SOURCE_DATA_DIR / "DMBA_RCCF_labels_M4D.nhdr"
STRUCTURES_ZIP_PATH = SOURCE_DATA_DIR / "DMBA_RCCF_labels_M4D.zip"
LABELS_TXT = "DMBA_RCCF_labels.txt"
LOOKUP_TXT = "DMBA_RCCF_labels_lookup.txt"

DMBA_DOWNLOADS = (
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_RCCF_labels_M4D.zip",
        STRUCTURES_ZIP_PATH,
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_RCCF_labels_M4D.raw",
        ANNOTATION_PATH.with_suffix(".raw"),
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_RCCF_labels_M4D.nhdr",
        ANNOTATION_PATH,
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_N06_md_M4D.raw",
        REFERENCE_PATH.with_suffix(".raw"),
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_N06_md_M4D.nhdr",
        REFERENCE_PATH,
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_N13_m2_M4D.raw",
        ADDITIONAL_REFERENCE_PATHS["m2"].with_suffix(".raw"),
    ),
    (
        "https://d3mof5o.s3.amazonaws.com/DMBA_N13_m2_M4D.nhdr",
        ADDITIONAL_REFERENCE_PATHS["m2"],
    ),
)

BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
NEGATIVE_STRUCTURE_ID_OFFSET = 1_000_000
MESH_NUM_THREADS = 6


def _load_nrrd(path):
    """Load a detached NRRD image and return its NumPy stack."""
    if not path.exists():
        raise FileNotFoundError(
            f"Expected downloaded DMBA file was not found: {path}"
        )
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def _scale_reference_to_uint16(reference):
    """Scale the float MD reference into BrainGlobe's uint16 template range."""
    if not np.all(np.isfinite(reference)):
        raise ValueError("Reference contains NaN or infinite values.")

    reference = reference.astype(np.float32, copy=False)
    dmin = float(reference.min())
    dmax = float(reference.max())
    drange = dmax - dmin
    if drange == 0:
        raise ValueError("Reference stack has zero range.")

    reference -= dmin
    reference *= np.iinfo(np.uint16).max / drange
    return np.rint(reference).astype(np.uint16)


def _read_actual_label_ids(labels_text):
    """Read voxel label IDs from the DSI Studio labels file."""
    label_ids = []
    for line in labels_text.splitlines():
        if line.strip():
            label_id, _ = line.split(maxsplit=1)
            label_ids.append(int(label_id))
    return label_ids


def _read_lookup_rows(lookup_text):
    """Read the 3D Slicer lookup table bundled with the RCCF labels."""
    lines = lookup_text.splitlines()
    header_index = next(
        i for i, line in enumerate(lines) if line.startswith("# ROI\t")
    )
    header = lines[header_index].lstrip("# ").split("\t")
    data_lines = [
        line
        for line in lines[header_index + 3 :]
        if line.strip() and not line.startswith("#")
    ]
    return csv.DictReader(data_lines, fieldnames=header, delimiter="\t")


def _is_number(value):
    value = (value or "").strip()
    if not value or value == "NaN":
        return False

    try:
        float(value)
    except ValueError:
        return False
    return True


def _read_int(value):
    return int(float(value.strip()))


def _is_lateralized_row(row):
    structure = row["Structure"].strip()
    return structure.endswith("_left") or structure.endswith("_right")


def _has_voxel_roi(row):
    return _is_number(row["ROI"])


def _canonical_row(rows):
    """Prefer the bilateral/non-voxel ontology row for duplicated IDs."""
    return sorted(
        rows,
        key=lambda row: (
            _is_lateralized_row(row),
            _has_voxel_roi(row),
            row["Structure"].strip(),
        ),
    )[0]


def _read_rccf_metadata():
    """Return voxel labels and all rows from the RCCF lookup table."""
    if not STRUCTURES_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Expected downloaded RCCF metadata was not found: "
            f"{STRUCTURES_ZIP_PATH}"
        )

    with zipfile.ZipFile(STRUCTURES_ZIP_PATH) as label_zip:
        label_ids = {
            label_id
            for label_id in _read_actual_label_ids(
                label_zip.read(LABELS_TXT).decode()
            )
            if label_id != 0
        }
        lookup_rows = list(
            _read_lookup_rows(
                label_zip.read(LOOKUP_TXT).decode("utf-8", errors="replace")
            )
        )

    return label_ids, lookup_rows


def _atlas_id_for_ontology_id(ontology_id):
    """Return a non-negative BrainGlobe ID for an RCCF ontology ID."""
    if ontology_id == ROOT_ID:
        return ROOT_ID
    if ontology_id > 0:
        return ontology_id
    return NEGATIVE_STRUCTURE_ID_OFFSET + abs(ontology_id)


def _rccf_ontology_tables():
    """Build RCCF ontology lookup tables for structures and annotation."""
    label_ids, lookup_rows = _read_rccf_metadata()

    rows_by_ontology_id = {}
    for row in lookup_rows:
        if _is_number(row["structure_id"]):
            rows_by_ontology_id.setdefault(
                _read_int(row["structure_id"]), []
            ).append(row)

    canonical_rows = {
        ontology_id: _canonical_row(rows)
        for ontology_id, rows in rows_by_ontology_id.items()
    }

    label_to_ontology_id = {}
    for row in lookup_rows:
        if not _is_number(row["ROI"]):
            continue

        roi_id = _read_int(row["ROI"])
        if roi_id in label_ids:
            label_to_ontology_id[roi_id] = _read_int(row["structure_id"])

    missing = sorted(label_ids - set(label_to_ontology_id))
    if missing:
        raise ValueError(f"Missing RCCF lookup rows for labels: {missing}")

    ontology_ids = {ROOT_ID, *label_to_ontology_id.values()}
    changed = True
    while changed:
        changed = False
        for ontology_id in list(ontology_ids):
            row = canonical_rows.get(ontology_id)
            if row is None:
                continue

            ancestor_ids = []
            if _is_number(row["parent_structure_id"]):
                ancestor_ids.append(_read_int(row["parent_structure_id"]))
            if row["id_path"].strip():
                ancestor_ids.extend(
                    _read_int(path_id)
                    for path_id in row["id_path"].strip("/").split("/")
                    if _is_number(path_id)
                )

            for ancestor_id in ancestor_ids:
                if ancestor_id not in ontology_ids:
                    ontology_ids.add(ancestor_id)
                    changed = True

    missing_ancestors = sorted(
        ontology_id
        for ontology_id in ontology_ids
        if ontology_id not in canonical_rows and ontology_id != ROOT_ID
    )
    if missing_ancestors:
        raise ValueError(
            "Missing RCCF ontology rows for ancestors: " f"{missing_ancestors}"
        )

    ontology_to_atlas_id = {
        ontology_id: _atlas_id_for_ontology_id(ontology_id)
        for ontology_id in ontology_ids
    }

    atlas_id_to_ontology_ids = {}
    for ontology_id, atlas_id in ontology_to_atlas_id.items():
        atlas_id_to_ontology_ids.setdefault(atlas_id, []).append(ontology_id)

    duplicate_atlas_ids = {
        atlas_id: ontology_ids
        for atlas_id, ontology_ids in atlas_id_to_ontology_ids.items()
        if len(ontology_ids) > 1
    }
    if duplicate_atlas_ids:
        raise ValueError(f"Duplicate RCCF atlas IDs: {duplicate_atlas_ids}")

    label_to_atlas_id = {
        label_id: ontology_to_atlas_id[ontology_id]
        for label_id, ontology_id in label_to_ontology_id.items()
    }

    return (
        canonical_rows,
        ontology_ids,
        ontology_to_atlas_id,
        label_to_atlas_id,
    )


def _structure_id_path(
    ontology_id,
    canonical_rows,
    ontology_to_atlas_id,
    path_cache,
):
    """Return a BrainGlobe structure path from RCCF parent links."""
    if ontology_id == ROOT_ID:
        return [ROOT_ID]
    if ontology_id in path_cache:
        return path_cache[ontology_id]

    row = canonical_rows[ontology_id]
    if _is_number(row["parent_structure_id"]):
        parent_id = _read_int(row["parent_structure_id"])
    else:
        parent_id = ROOT_ID

    parent_path = _structure_id_path(
        parent_id, canonical_rows, ontology_to_atlas_id, path_cache
    )
    path = [*parent_path, ontology_to_atlas_id[ontology_id]]
    path_cache[ontology_id] = path
    return path


def _remap_annotation_to_atlas_ids(annotation):
    """Map RCCF ROI labels onto bilateral ontology-backed atlas IDs."""
    _, _, _, label_to_atlas_id = _rccf_ontology_tables()

    annotation_labels = set(np.unique(annotation).astype(int))
    missing_labels = sorted(annotation_labels - {0} - set(label_to_atlas_id))
    if missing_labels:
        raise ValueError(
            f"Missing RCCF ontology mapping for labels: {missing_labels}"
        )

    label_lut = np.zeros(int(annotation.max()) + 1, dtype=np.uint32)
    for label_id, atlas_id in label_to_atlas_id.items():
        label_lut[label_id] = atlas_id

    return label_lut[annotation]


def _strip_laterality(text):
    """Remove source left/right suffixes from names and acronyms."""
    text = text.strip()
    for suffix in (
        " (left)",
        " (right)",
        "_left",
        "_right",
        " left",
        " right",
        "-L",
        "-R",
    ):
        if text.endswith(suffix):
            return text[: -len(suffix)].strip()
    return text


def _source_structure_name(row):
    """Build a readable source name from the RCCF Structure field."""
    structure = row["Structure"].strip()
    if "__" in structure:
        _, structure = structure.split("__", maxsplit=1)
    return _strip_laterality(structure.replace("_", " "))


def _structure_name(row):
    """Choose the most readable source-backed name for a structure row."""
    source_name = _source_structure_name(row)
    if "uncharted" in source_name.lower():
        return source_name

    if row["ARA_name"].strip():
        return row["ARA_name"].strip()
    if row["GN_Description"].strip():
        return _strip_laterality(row["GN_Description"])
    return source_name


def _structure_acronym(row):
    """Choose a concise source-backed acronym for a structure row."""
    if row["ARA_abbrev"].strip():
        return row["ARA_abbrev"].strip()
    if row["GN_Symbol"].strip():
        return _strip_laterality(row["GN_Symbol"])

    acronym = row["Structure"].split("__", maxsplit=1)[0].strip()
    return _strip_laterality(acronym)


def _deduplicate_acronyms(structures):
    """Make acronyms unique while preserving the first/source-standard form."""
    seen = set()
    for structure in structures:
        acronym = structure["acronym"]
        if acronym not in seen:
            seen.add(acronym)
            continue

        suffix = "uncharted" if "uncharted" in structure["name"] else "rccf"
        deduplicated = f"{acronym}-{suffix}"
        if deduplicated in seen:
            deduplicated = f"{acronym}-{structure['id']}"

        structure["acronym"] = deduplicated
        seen.add(deduplicated)

    return structures


def download_resources():
    """Download the necessary DMBA source files with Pooch."""
    missing_paths = [
        destination
        for _, destination in DMBA_DOWNLOADS
        if not destination.exists()
    ]
    if not missing_paths:
        print("All DMBA source files are already present.")
        return

    utils.check_internet_connection()
    for url, destination in DMBA_DOWNLOADS:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            print(f"Already present: {destination}")
            continue

        pooch.retrieve(
            url=url,
            known_hash=None,
            path=destination.parent,
            fname=destination.name,
            progressbar=True,
        )


def retrieve_reference_and_annotation():
    """Return the MD reference and ontology-backed RCCF annotation volumes."""
    reference = _load_nrrd(REFERENCE_PATH)
    annotation = _load_nrrd(ANNOTATION_PATH)

    if reference.shape != annotation.shape:
        raise ValueError(
            "Reference and annotation shapes do not match: "
            f"{reference.shape} != {annotation.shape}"
        )

    reference = _scale_reference_to_uint16(reference)
    annotation = annotation.astype(np.uint32, copy=False)
    annotation = _remap_annotation_to_atlas_ids(annotation)

    return reference, annotation


def retrieve_hemisphere_map():
    """Return no hemisphere map because labels are packaged bilaterally."""
    return None


def retrieve_structure_information():
    """Return bilateral RCCF structures with source hierarchy."""
    (
        canonical_rows,
        ontology_ids,
        ontology_to_atlas_id,
        _,
    ) = _rccf_ontology_tables()

    structures = [
        {
            "id": ROOT_ID,
            "name": "root",
            "acronym": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    ]

    path_cache = {ROOT_ID: [ROOT_ID]}
    for ontology_id in sorted(
        ontology_ids,
        key=lambda oid: _structure_id_path(
            oid, canonical_rows, ontology_to_atlas_id, path_cache
        ),
    ):
        if ontology_id == ROOT_ID:
            continue

        row = canonical_rows[ontology_id]
        structures.append(
            {
                "id": ontology_to_atlas_id[ontology_id],
                "name": _structure_name(row),
                "acronym": _structure_acronym(row),
                "structure_id_path": _structure_id_path(
                    ontology_id,
                    canonical_rows,
                    ontology_to_atlas_id,
                    path_cache,
                ),
                "rgb_triplet": [
                    int(row["c_r"]),
                    int(row["c_g"]),
                    int(row["c_b"]),
                ],
            }
        )

    structures = _deduplicate_acronyms(structures)
    check_struct_consistency(structures)
    return structures


def retrieve_or_construct_meshes(annotated_volume=None, structures=None):
    """Construct all structure meshes and return their file paths."""
    if annotated_volume is None:
        annotated_volume = _load_nrrd(ANNOTATION_PATH).astype(
            np.uint32, copy=False
        )
        annotated_volume = _remap_annotation_to_atlas_ids(annotated_volume)
    if structures is None:
        structures = retrieve_structure_information()

    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    meshes_dict = construct_meshes_from_annotation(
        save_path=BG_ROOT_DIR,
        volume=annotated_volume,
        structures_list=structures,
        closing_n_iters=1,
        decimate_fraction=0.2,
        smooth=False,
        parallel=True,
        num_threads=MESH_NUM_THREADS,
        verbosity=0,
    )

    mesh_ids = [structure["id"] for structure in structures]
    missing_mesh_ids = sorted(set(mesh_ids) - set(meshes_dict))
    if missing_mesh_ids:
        raise ValueError(f"Could not create meshes for: {missing_mesh_ids}")

    return meshes_dict


def retrieve_additional_references(reference_shape):
    """Return additional DMBA template volumes."""
    additional_references = {}
    for name, path in ADDITIONAL_REFERENCE_PATHS.items():
        reference = _load_nrrd(path)
        if reference.shape != reference_shape:
            raise ValueError(
                f"Additional reference {name!r} shape does not match "
                f"the main reference: {reference.shape} != {reference_shape}"
            )
        additional_references[name] = _scale_reference_to_uint16(reference)

    return additional_references


if __name__ == "__main__":
    if RESOLUTION is None:
        raise ValueError("RESOLUTION must be set before running this script.")

    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
        )
    download_resources()
    reference_volume, annotated_volume = retrieve_reference_and_annotation()
    additional_references = retrieve_additional_references(
        reference_volume.shape
    )
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information()
    meshes_dict = retrieve_or_construct_meshes(annotated_volume, structures)

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
    print(f"Atlas saved to {output_filename}")
