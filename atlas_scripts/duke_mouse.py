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
CITATION = "Mansour et al. 2025, https://doi.org/10.1126/sciadv.adq8089"
SPECIES = "Mus musculus"
ATLAS_LINK = (
    "https://civmimagespace.civm.duhs.duke.edu/"
    "tp_item_detail.php/view/item_number=DMBA/set_id=315"
)
ATLAS_PACKAGER = "Amirreza Bahramani"

ORIENTATION = "ipr"
ROOT_ID = 997
RESOLUTION = 15

SOURCE_DATA_DIR = (
    Path.home() / "brainglobe_workingdir" / ATLAS_NAME / "source_data"
)
BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_BASE_URL = "https://d3mof5o.s3.amazonaws.com/"

REFERENCE_FILES = {
    "md": ("DMBA_md", "DMBA_N06_md_M4D"),
    "ad": ("DMBA_ad", "DMBA_N01_ad_M4D"),
    "dwi": ("DMBA_dwi", "DMBA_N02_dwi_M4D"),
    "nqa": ("DMBA_nqa", "DMBA_N03_nqa_M4D"),
    "rd": ("DMBA_rd", "DMBA_N05_rd_M4D"),
    "fa": ("DMBA_fa", "DMBA_N09_fa_M4D"),
    "m0": ("DMBA_m0", "DMBA_N11_m0_M4D"),
    "m1": ("DMBA_m1", "DMBA_N12_m1_M4D"),
    "m2": ("DMBA_m2", "DMBA_N13_m2_M4D"),
    "m3": ("DMBA_m3", "DMBA_N14_m3_M4D"),
    "iso": ("DMBA_iso", "DMBA_N17_iso_M4D"),
    "mgre-unmasked": (
        "DMBA_mGRE-unmasked",
        "DMBA_N18_mGRE-unmasked_M4D",
    ),
}

ANNOTATION_STEM = "DMBA_RCCF_labels_M4D"
ANNOTATION_PATH = SOURCE_DATA_DIR / f"{ANNOTATION_STEM}.nhdr"
STRUCTURES_ZIP_PATH = SOURCE_DATA_DIR / f"{ANNOTATION_STEM}.zip"
LABELS_TXT = "DMBA_RCCF_labels.txt"
LOOKUP_TXT = "DMBA_RCCF_labels_lookup.txt"

NEGATIVE_STRUCTURE_ID_OFFSET = 1_000_000
MESH_NUM_THREADS = 6


def _is_number(value):
    """Return whether a lookup-table value contains a number."""
    value = (value or "").strip()
    if not value or value == "NaN":
        return False

    try:
        float(value)
    except ValueError:
        return False
    return True


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
        parent_id = int(float(row["parent_structure_id"]))
    else:
        parent_id = ROOT_ID

    parent_path = _structure_id_path(
        parent_id, canonical_rows, ontology_to_atlas_id, path_cache
    )
    path = [*parent_path, ontology_to_atlas_id[ontology_id]]
    path_cache[ontology_id] = path
    return path


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


def download_resources():
    """Download the necessary DMBA source files with Pooch."""
    downloads = [
        (
            f"{DOWNLOAD_BASE_URL}{ANNOTATION_STEM}.zip",
            STRUCTURES_ZIP_PATH,
        )
    ]

    image_files = [("", ANNOTATION_STEM), *REFERENCE_FILES.values()]
    for directory, filename in image_files:
        for suffix in (".nhdr", ".raw"):
            downloads.append(
                (
                    f"{DOWNLOAD_BASE_URL}{filename}{suffix}",
                    SOURCE_DATA_DIR / directory / f"{filename}{suffix}",
                )
            )

    missing_paths = [
        destination for _, destination in downloads if not destination.exists()
    ]
    if not missing_paths:
        print("All DMBA source files are already present.")
        return

    utils.check_internet_connection()
    for url, destination in downloads:
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


def retrieve_structure_information():
    """Return bilateral RCCF structures and the voxel-label ID mapping."""
    if not STRUCTURES_ZIP_PATH.exists():
        raise FileNotFoundError(
            f"Expected downloaded RCCF metadata was not found: "
            f"{STRUCTURES_ZIP_PATH}"
        )

    with zipfile.ZipFile(STRUCTURES_ZIP_PATH) as label_zip:
        labels_text = label_zip.read(LABELS_TXT).decode()
        label_ids = {
            int(line.split(maxsplit=1)[0])
            for line in labels_text.splitlines()
            if line.strip()
        }
        label_ids.discard(0)

        lookup_text = label_zip.read(LOOKUP_TXT).decode(
            "utf-8", errors="replace"
        )

    lines = lookup_text.splitlines()
    header_index = next(
        index for index, line in enumerate(lines) if line.startswith("# ROI\t")
    )
    header = lines[header_index].lstrip("# ").split("\t")
    data_lines = [
        line
        for line in lines[header_index + 3 :]
        if line.strip() and not line.startswith("#")
    ]
    lookup_rows = list(
        csv.DictReader(data_lines, fieldnames=header, delimiter="\t")
    )

    rows_by_ontology_id = {}
    for row in lookup_rows:
        if _is_number(row["structure_id"]):
            ontology_id = int(float(row["structure_id"]))
            rows_by_ontology_id.setdefault(ontology_id, []).append(row)

    canonical_rows = {
        ontology_id: min(
            rows,
            key=lambda row: (
                row["Structure"].strip().endswith(("_left", "_right")),
                _is_number(row["ROI"]),
                row["Structure"].strip(),
            ),
        )
        for ontology_id, rows in rows_by_ontology_id.items()
    }

    label_to_ontology_id = {}
    for row in lookup_rows:
        if not _is_number(row["ROI"]):
            continue

        label_id = int(float(row["ROI"]))
        if label_id in label_ids:
            label_to_ontology_id[label_id] = int(float(row["structure_id"]))

    missing_labels = sorted(label_ids - set(label_to_ontology_id))
    if missing_labels:
        raise ValueError(
            f"Missing RCCF lookup rows for labels: {missing_labels}"
        )

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
                ancestor_ids.append(int(float(row["parent_structure_id"])))
            if row["id_path"].strip():
                ancestor_ids.extend(
                    int(float(path_id))
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
            f"Missing RCCF ontology rows for ancestors: {missing_ancestors}"
        )

    ontology_to_atlas_id = {
        ontology_id: (
            ontology_id
            if ontology_id > 0
            else NEGATIVE_STRUCTURE_ID_OFFSET + abs(ontology_id)
        )
        for ontology_id in ontology_ids
    }

    atlas_id_to_ontology_ids = {}
    for ontology_id, atlas_id in ontology_to_atlas_id.items():
        atlas_id_to_ontology_ids.setdefault(atlas_id, []).append(ontology_id)

    duplicate_atlas_ids = {
        atlas_id: source_ids
        for atlas_id, source_ids in atlas_id_to_ontology_ids.items()
        if len(source_ids) > 1
    }
    if duplicate_atlas_ids:
        raise ValueError(f"Duplicate RCCF atlas IDs: {duplicate_atlas_ids}")

    label_to_atlas_id = {
        label_id: ontology_to_atlas_id[ontology_id]
        for label_id, ontology_id in label_to_ontology_id.items()
    }

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
    seen_acronyms = {"root"}
    for ontology_id in sorted(
        ontology_ids,
        key=lambda source_id: _structure_id_path(
            source_id, canonical_rows, ontology_to_atlas_id, path_cache
        ),
    ):
        if ontology_id == ROOT_ID:
            continue

        row = canonical_rows[ontology_id]
        source_name = row["Structure"].strip()
        if "__" in source_name:
            _, source_name = source_name.split("__", maxsplit=1)
        source_name = _strip_laterality(source_name.replace("_", " "))

        if "uncharted" in source_name.lower():
            name = source_name
        elif row["ARA_name"].strip():
            name = row["ARA_name"].strip()
        elif row["GN_Description"].strip():
            name = _strip_laterality(row["GN_Description"])
        else:
            name = source_name

        if row["ARA_abbrev"].strip():
            acronym = row["ARA_abbrev"].strip()
        elif row["GN_Symbol"].strip():
            acronym = _strip_laterality(row["GN_Symbol"])
        else:
            acronym = row["Structure"].split("__", maxsplit=1)[0]
            acronym = _strip_laterality(acronym)

        if acronym in seen_acronyms:
            suffix = "uncharted" if "uncharted" in name else "rccf"
            deduplicated = f"{acronym}-{suffix}"
            if deduplicated in seen_acronyms:
                deduplicated = (
                    f"{acronym}-{ontology_to_atlas_id[ontology_id]}"
                )
            acronym = deduplicated
        seen_acronyms.add(acronym)

        structures.append(
            {
                "id": ontology_to_atlas_id[ontology_id],
                "name": name,
                "acronym": acronym,
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

    check_struct_consistency(structures)
    return structures, label_to_atlas_id


def retrieve_reference_and_annotation(label_to_atlas_id):
    """Return the main template, annotation, and additional MR contrasts."""
    references = {}
    reference_shape = None
    for name, (directory, filename) in REFERENCE_FILES.items():
        path = SOURCE_DATA_DIR / directory / f"{filename}.nhdr"
        if not path.exists():
            raise FileNotFoundError(
                f"Expected downloaded DMBA file was not found: {path}"
            )

        reference = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
        if not np.all(np.isfinite(reference)):
            raise ValueError(f"Reference {name!r} contains non-finite values.")

        if reference_shape is None:
            reference_shape = reference.shape
        elif reference.shape != reference_shape:
            raise ValueError(
                f"Additional reference {name!r} shape does not match "
                f"the main reference: {reference.shape} != {reference_shape}"
            )

        reference = reference.astype(np.float32, copy=False)
        minimum = float(reference.min())
        value_range = float(reference.max()) - minimum
        if value_range == 0:
            raise ValueError(f"Reference {name!r} has zero intensity range.")

        reference -= minimum
        reference *= np.iinfo(np.uint16).max / value_range
        references[name] = np.rint(reference).astype(np.uint16)

    # Release the final float stack before loading and remapping the
    # annotation.
    del reference

    if not ANNOTATION_PATH.exists():
        raise FileNotFoundError(
            f"Expected downloaded DMBA file was not found: {ANNOTATION_PATH}"
        )
    annotation = sitk.GetArrayFromImage(sitk.ReadImage(str(ANNOTATION_PATH)))
    if annotation.shape != reference_shape:
        raise ValueError(
            "Reference and annotation shapes do not match: "
            f"{reference_shape} != {annotation.shape}"
        )

    annotation = annotation.astype(np.uint32, copy=False)
    annotation_labels = set(np.unique(annotation).astype(int))
    missing_labels = sorted(
        annotation_labels - {0} - set(label_to_atlas_id)
    )
    if missing_labels:
        raise ValueError(
            f"Missing RCCF ontology mapping for labels: {missing_labels}"
        )

    label_lut = np.zeros(int(annotation.max()) + 1, dtype=np.uint32)
    for label_id, atlas_id in label_to_atlas_id.items():
        label_lut[label_id] = atlas_id
    annotation = label_lut[annotation]

    reference = references.pop("md")
    return reference, annotation, references


def retrieve_or_construct_meshes(annotated_volume, structures):
    """Construct all structure meshes and return their file paths."""
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


if __name__ == "__main__":
    BG_ROOT_DIR.mkdir(parents=True, exist_ok=True)

    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(BG_ROOT_DIR.glob(f"{atlas_prefix}_v*"))
    if existing:
        raise FileExistsError(
            f"Atlas output already exists in {BG_ROOT_DIR}. "
        )

    download_resources()
    structures, label_to_atlas_id = retrieve_structure_information()
    reference_volume, annotated_volume, additional_references = (
        retrieve_reference_and_annotation(label_to_atlas_id)
    )
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
        hemispheres_stack=None,
        scale_meshes=True,
        additional_references=additional_references,
    )
    print(f"Atlas saved to {output_filename}")
