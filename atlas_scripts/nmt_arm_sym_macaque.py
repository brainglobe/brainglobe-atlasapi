"""Atlas generation script for the ARM macaque atlas."""

import colorsys
import re
from pathlib import Path
from tkinter import TRUE

import meshio as mio
import nibabel as nib
import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    extract_mesh_from_mask,
)
from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.utils import (
    atlas_name_from_repr,
    check_internet_connection,
)

### Metadata
__version__ = 0
ATLAS_NAME = "nmt_arm_sym_macaque"
CITATION = (
    "Jung et al., 2021. A comprehensive macaque fMRI pipeline and "
    "hierarchical atlas. NeuroImage. "
    "https://doi.org/10.1016/j.neuroimage.2021.117997; "
    "Hartig et al., 2021. The Subcortical Atlas of the Rhesus Macaque "
    "(SARM) for neuroimaging. NeuroImage. "
    "https://doi.org/10.1016/j.neuroimage.2021.117996"
)
SPECIES = "Macaca mulatta"
ATLAS_LINK = (
    "https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/"
    "macaque_tempatl/template_nmtv2.html"
)
ATLAS_FILE_URL = (
    "https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/"
    "macaque_tempatl/template_nmtv2.html#nh-macaque-template-nmtv2-sym-dl"
)
ORIENTATION = "lpi"
ROOT_ID = 9999
CORTEX_ID = 9001
SUBCORTEX_ID = 9002
RESOLUTION = 250  # microns
ATLAS_PACKAGER = "Amirreza Bahramani"

NMT_SYM_URL = (
    "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/"
    "NMT_v2.1_sym.tgz"
)
NMT_SYM_HASH = (
    "sha256:35c8770f050403a8e77416521116131fddeead82c732f929fd70e1b0f0ddb51c"
)

# Use the full-head NMT v2.1 volume so inferior structures are retained.
NMT_REFERENCE_FILENAME = "NMT_v2.1_sym_fh.nii.gz"
NMT_BRAINMASK_FILENAME = "NMT_v2.1_sym_fh_brainmask.nii.gz"
ARM_ANNOTATION_FILENAME = "ARM_6_in_NMT_v2.1_sym_fh.nii.gz"
ARM_MESH_RE = re.compile(r"^(CHARM|SARM)_(\d+)\.(.+)\.k(\d+)\.gii$")
HEMISPHERE_PREFIX_RE = re.compile(r"^(C[LR]|S[LR])_")


def download_resources(working_dir: Path) -> Path:
    """
    Download and extract the symmetric NMT v2.1 dataset.

    Parameters
    ----------
    working_dir : Path
        Directory where downloaded source files should be cached.

    Returns
    -------
    Path
        Path to the extracted NMT directory.
    """
    check_internet_connection()

    download_dir = Path(working_dir) / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    pooch.retrieve(
        url=NMT_SYM_URL,
        known_hash=NMT_SYM_HASH,
        path=download_dir,
        fname="NMT_v2.1_sym.tgz",
        processor=pooch.Untar(extract_dir="NMT_v2.1_sym"),
        progressbar=True,
    )

    return download_dir / "NMT_v2.1_sym"


def resolve_standard_nmt_dir(nmt_dir: Path) -> Path:
    """
    Find the directory containing the 250 um full-head symmetric NMT files.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.

    Returns
    -------
    Path
        Directory containing ``NMT_v2.1_sym_fh.nii.gz``.
    """
    nmt_dir = Path(nmt_dir)
    candidates = sorted(
        [
            path
            for path in [nmt_dir, *nmt_dir.rglob("*")]
            if path.is_dir() and (path / NMT_REFERENCE_FILENAME).exists()
        ]
    )

    if not candidates:
        raise FileNotFoundError(
            f"Could not find a directory containing {NMT_REFERENCE_FILENAME} "
            f"inside {nmt_dir}"
        )

    return candidates[0]


def strip_hemisphere_prefix(value: str) -> str:
    return HEMISPHERE_PREFIX_RE.sub("", str(value).strip())


def get_hemisphere_prefix(value: str) -> str:
    match = HEMISPHERE_PREFIX_RE.match(str(value).strip())

    if match is None:
        raise ValueError(f"Could not find ARM hemisphere prefix in {value!r}")

    return match.group(1)


def normalize_arm_domain(value: str) -> str:
    normalized = str(value).strip().lower()

    if normalized not in {"cortex", "subcortex"}:
        raise ValueError(f"Unexpected ARM Level_0 value: {value!r}")

    return normalized


def parse_arm_structure(row: pd.Series, level: int) -> dict:
    return {
        "id": int(row[f"Level_{level}_index"]),
        "name": strip_hemisphere_prefix(row[f"Level_{level}"]),
        "acronym": strip_hemisphere_prefix(row[f"Level_{level}_abbr"]),
    }


def build_arm_id_mappings(
    arm_table: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, dict]]:
    """Build source-to-canonical ID mappings for merged ARM structures."""
    source_to_canonical = {}
    canonical_info_by_id = {}

    for level in range(1, 7):
        name_column = f"Level_{level}"
        acronym_column = f"Level_{level}_abbr"
        id_column = f"Level_{level}_index"
        unique_structures = arm_table[
            ["Level_0", name_column, acronym_column, id_column]
        ].drop_duplicates()

        unique_structures = unique_structures.assign(
            domain=unique_structures["Level_0"].map(normalize_arm_domain),
            canonical_name=unique_structures[name_column].map(
                strip_hemisphere_prefix
            ),
            canonical_acronym=unique_structures[acronym_column].map(
                strip_hemisphere_prefix
            ),
            name_prefix=unique_structures[name_column].map(
                get_hemisphere_prefix
            ),
            acronym_prefix=unique_structures[acronym_column].map(
                get_hemisphere_prefix
            ),
        )

        grouped = unique_structures.groupby(
            ["domain", "canonical_name", "canonical_acronym"],
            sort=False,
            dropna=False,
        )

        for (
            domain,
            canonical_name,
            canonical_acronym,
        ), group in grouped:
            source_ids = sorted(int(value) for value in group[id_column])
            name_prefixes = set(group["name_prefix"])
            acronym_prefixes = set(group["acronym_prefix"])
            expected_prefixes = (
                {"CL", "CR"} if domain == "cortex" else {"SL", "SR"}
            )

            if len(source_ids) != 2:
                raise ValueError(
                    "Expected exactly one left/lower and one right/higher "
                    "ARM ID for merged structure "
                    f"{canonical_name} ({canonical_acronym}) at level "
                    f"{level}, found {source_ids}"
                )

            if (
                name_prefixes != expected_prefixes
                or acronym_prefixes != expected_prefixes
            ):
                raise ValueError(
                    "Inconsistent ARM hemisphere prefixes for merged "
                    f"structure {canonical_name} ({canonical_acronym}) at "
                    f"level {level}. Name prefixes: {name_prefixes}; "
                    f"acronym prefixes: {acronym_prefixes}"
                )

            canonical_id = min(source_ids)
            candidate_info = {
                "domain": domain,
                "name": canonical_name,
                "acronym": canonical_acronym,
                "source_ids": source_ids,
            }

            existing_info = canonical_info_by_id.get(canonical_id)
            if existing_info is None:
                canonical_info_by_id[canonical_id] = candidate_info
            elif existing_info != candidate_info:
                raise ValueError(
                    f"Conflicting ARM canonical metadata for ID "
                    f"{canonical_id}.\nExisting: {existing_info}\n"
                    f"New: {candidate_info}"
                )

            for source_id in source_ids:
                existing_mapping = source_to_canonical.get(source_id)

                if existing_mapping is None:
                    source_to_canonical[source_id] = canonical_id
                elif existing_mapping != canonical_id:
                    raise ValueError(
                        f"Conflicting ARM source-to-canonical mapping for "
                        f"ID {source_id}: {existing_mapping} vs "
                        f"{canonical_id}"
                    )

    return source_to_canonical, canonical_info_by_id


def remap_annotation_labels(
    annotation: np.ndarray,
    source_to_canonical: dict[int, int],
) -> np.ndarray:
    """Merge left/right ARM annotation labels into canonical labels."""
    annotation_ids = {int(value) for value in np.unique(annotation)}
    annotation_ids.discard(0)
    missing_ids = sorted(annotation_ids - set(source_to_canonical))

    if missing_ids:
        raise ValueError(
            "Some ARM annotation labels are not present in the ARM hierarchy "
            f"table: {missing_ids}"
        )

    max_id = max(max(annotation_ids), max(source_to_canonical))
    lookup = np.arange(max_id + 1, dtype=np.uint32)

    for source_id, canonical_id in source_to_canonical.items():
        lookup[source_id] = canonical_id

    return lookup[annotation.astype(np.uint32)]


def retrieve_reference_and_annotation(
    nmt_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the NMT reference volume and merged finest ARM annotation.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The normalized reference volume and merged level-6 ARM annotation
        volume.
    """
    standard_dir = resolve_standard_nmt_dir(nmt_dir)
    nmt_root_dir = standard_dir.parent
    reference_path = standard_dir / NMT_REFERENCE_FILENAME
    annotation_path = (
        standard_dir / "supplemental_ARM" / ARM_ANNOTATION_FILENAME
    )

    reference = load_nii(reference_path, as_array=True).astype(np.float32)
    annotation = load_nii(annotation_path, as_array=True).astype(np.uint32)

    reference -= reference.min()
    if reference.max() > 0:
        reference /= reference.max()
    reference = (reference * np.iinfo(np.uint16).max).astype(np.uint16)

    if reference.shape != annotation.shape:
        raise ValueError(
            f"Reference shape {reference.shape} does not match annotation "
            f"shape {annotation.shape}"
        )

    arm_table = pd.read_csv(nmt_root_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, _ = build_arm_id_mappings(arm_table)
    annotation = remap_annotation_labels(annotation, source_to_canonical)

    return reference, annotation


def retrieve_hemisphere_map() -> np.ndarray | None:
    """
    Retrieve a hemisphere map for the atlas.

    Returns
    -------
    None
        The symmetric NMT v2.1 / ARM atlas does not need a hemisphere map.
    """
    return None


def collapse_repeated_ids(path: list[int]) -> list[int]:
    """
    Collapse repeated adjacent IDs in an ARM structure path.

    ARM may repeat the same ID across adjacent levels to preserve a
    six-column CHARM/SARM hierarchy table. BrainGlobe only needs the
    structure once in the path.

    Parameters
    ----------
    path : list[int]
        Structure path that may contain repeated adjacent IDs.

    Returns
    -------
    list[int]
        Structure path with adjacent duplicate IDs removed.
    """
    collapsed = []

    for structure_id in path:
        if not collapsed or collapsed[-1] != structure_id:
            collapsed.append(structure_id)

    return collapsed


def hex_to_rgb(hex_color: str) -> list[int]:
    """
    Convert a hexadecimal color to an RGB triplet.

    Parameters
    ----------
    hex_color : str
        Hexadecimal color, for example ``#FFA6BC``.

    Returns
    -------
    list[int]
        RGB triplet, for example ``[255, 166, 188]``.
    """
    hex_color = hex_color.strip().lstrip("#")

    return [
        int(hex_color[0:2], 16),
        int(hex_color[2:4], 16),
        int(hex_color[4:6], 16),
    ]


def load_charm_rgb_triplets(nmt_root_dir: Path) -> dict[int, list[int]]:
    """
    Load CHARM RGB triplets from the distributed AFNI palette file.

    Parameters
    ----------
    nmt_root_dir : Path
        Extracted NMT root directory containing ``tables_CHARM``.

    Returns
    -------
    dict[int, list[int]]
        Mapping from CHARM structure ID to RGB triplet.
    """
    palette_path = nmt_root_dir / "tables_CHARM" / "hue_CHARM_cmap.pal"
    palette_lines = [
        line.strip()
        for line in palette_path.read_text().splitlines()
        if line.strip()
    ]
    color_lines = palette_lines[1:]

    return {
        structure_id: hex_to_rgb(hex_color)
        for structure_id, hex_color in enumerate(color_lines, start=1)
    }


def arm_sarm_cmap_gen(
    arm_table: pd.DataFrame,
    source_to_canonical: dict[int, int],
    seed: int = 77,
) -> dict[int, list[int]]:
    """Generate deterministic subcortical ARM RGB triplets."""
    base_color_hex = {
        "LVPal": "#EC9830",
        "MPal": "#7ED04B",
        "Amy": "#9DE79C",
        "BG": "#98D6F9",
        "DSP": "#96A7D3",
        "POC": "#FF5547",
        "Hy": "#E64438",
        "PreThal": "#F2483B",
        "Thal": "#FF7080",
        "EpiThal": "#FF909F",
        "PrT": "#FF90FF",
        "Mid": "#FF64FF",
        "Pons": "#FF9B88",
        "Cb": "#F0F080",
        "Med": "#FF9BCD",
        "HF": "#7ED04B",
        "Str": "#98D6F9",
        "Pd": "#8599CC",
    }
    base_colors = {
        acronym: hex_to_rgb(hex_color)
        for acronym, hex_color in base_color_hex.items()
    }
    rgb_triplets = {}
    level_1_children = {}

    subcortex_table = arm_table[
        arm_table["Level_0"].map(normalize_arm_domain) == "subcortex"
    ]

    for _, row in subcortex_table.iterrows():
        parsed_path = []

        for level in range(1, 7):
            structure = parse_arm_structure(row, level)
            structure["id"] = source_to_canonical[structure["id"]]

            if not parsed_path or structure["id"] != parsed_path[-1]["id"]:
                parsed_path.append(structure)

        if len(parsed_path) > 1:
            level_1_children.setdefault(parsed_path[0]["id"], set()).add(
                parsed_path[1]["id"]
            )

        anchor_rgb = None
        for structure in parsed_path:
            if structure["acronym"] in base_colors:
                anchor_rgb = base_colors[structure["acronym"]]

            if structure["id"] in rgb_triplets:
                continue

            if anchor_rgb is None:
                rgb_triplets[structure["id"]] = [255, 255, 255]
                continue

            if structure["acronym"] in base_colors:
                rgb_triplets[structure["id"]] = anchor_rgb
                continue

            rng = np.random.default_rng(seed + structure["id"])
            hue, lightness, saturation = colorsys.rgb_to_hls(
                *(channel / 255 for channel in anchor_rgb)
            )
            lightness = np.clip(lightness + rng.uniform(-0.10, 0.10), 0, 1)
            saturation = np.clip(saturation * rng.uniform(0.90, 1.10), 0, 1)
            rgb_triplets[structure["id"]] = [
                int(round(channel * 255))
                for channel in colorsys.hls_to_rgb(hue, lightness, saturation)
            ]

    for structure_id, children in level_1_children.items():
        child_colours = [
            rgb_triplets[child_id]
            for child_id in children
            if child_id in rgb_triplets
        ]
        rgb_triplets[structure_id] = [
            int(round(channel)) for channel in np.mean(child_colours, axis=0)
        ]

    return rgb_triplets


def check_or_add_structure(
    structures_by_id: dict[int, dict],
    candidate: dict,
) -> None:
    """
    Add a structure, checking repeated ARM entries are consistent.

    Parameters
    ----------
    structures_by_id : dict[int, dict]
        Structures collected so far, keyed by structure ID.
    candidate : dict
        New candidate structure dictionary.
    """
    structure_id = candidate["id"]

    if structure_id not in structures_by_id:
        structures_by_id[structure_id] = candidate
        return

    existing = structures_by_id[structure_id]

    for field in ["name", "acronym", "structure_id_path"]:
        if existing[field] != candidate[field]:
            raise ValueError(
                f"Conflicting ARM hierarchy entry for ID {structure_id}.\n"
                f"Existing {field}: {existing[field]}\n"
                f"New {field}: {candidate[field]}"
            )


def retrieve_structure_information(nmt_dir: Path) -> list[dict]:
    """
    Retrieve BrainGlobe-compatible structure information for ARM.

    ARM is a fusion of CHARM and SARM. The distributed hierarchy table adds
    ``Level_0`` for cortex/subcortex above the six CHARM/SARM levels. The
    level-6 annotation is used as the annotation volume.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.

    Returns
    -------
    list[dict]
        BrainGlobe-compatible atlas structure dictionaries.
    """
    standard_dir = resolve_standard_nmt_dir(nmt_dir)
    nmt_root_dir = standard_dir.parent
    arm_table = pd.read_csv(nmt_root_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, _ = build_arm_id_mappings(arm_table)
    charm_rgb_triplets = load_charm_rgb_triplets(nmt_root_dir)
    sarm_rgb_triplets = arm_sarm_cmap_gen(arm_table, source_to_canonical)

    structures_by_id = {
        ROOT_ID: {
            "acronym": "root",
            "id": ROOT_ID,
            "name": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        },
        CORTEX_ID: {
            "acronym": "CTX",
            "id": CORTEX_ID,
            "name": "cortex",
            "structure_id_path": [ROOT_ID, CORTEX_ID],
            "rgb_triplet": [255, 255, 255],
        },
        SUBCORTEX_ID: {
            "acronym": "SUB",
            "id": SUBCORTEX_ID,
            "name": "subcortex",
            "structure_id_path": [ROOT_ID, SUBCORTEX_ID],
            "rgb_triplet": [255, 255, 255],
        },
    }

    for _, row in arm_table.iterrows():
        domain = normalize_arm_domain(row["Level_0"])
        domain_id = CORTEX_ID if domain == "cortex" else SUBCORTEX_ID
        parsed_path = []

        for level in range(1, 7):
            structure = parse_arm_structure(row, level)
            structure["id"] = source_to_canonical[structure["id"]]
            parsed_path.append(structure)

        raw_id_path = [ROOT_ID, domain_id] + [
            structure["id"] for structure in parsed_path
        ]
        id_path = collapse_repeated_ids(raw_id_path)

        if len(id_path) != len(set(id_path)):
            raise ValueError(
                f"Non-adjacent repeated structure ID found in path: {id_path}"
            )

        parsed_by_id = {}
        for structure in parsed_path:
            structure_id = structure["id"]

            if (
                structure_id in parsed_by_id
                and parsed_by_id[structure_id] != structure
            ):
                raise ValueError(
                    f"Conflicting name/acronym for ID {structure_id} "
                    "within one ARM row: "
                    f"{parsed_by_id[structure_id]} vs {structure}"
                )

            parsed_by_id[structure_id] = structure

        for structure_id in id_path[2:]:
            structure = parsed_by_id[structure_id]
            path_to_structure = id_path[: id_path.index(structure_id) + 1]
            candidate = {
                "acronym": structure["acronym"],
                "id": structure_id,
                "name": structure["name"],
                "structure_id_path": path_to_structure,
                "rgb_triplet": (
                    charm_rgb_triplets[structure_id]
                    if domain == "cortex"
                    else sarm_rgb_triplets[structure_id]
                ),
            }

            check_or_add_structure(structures_by_id, candidate)

    return sorted(
        structures_by_id.values(),
        key=lambda structure: (
            len(structure["structure_id_path"]),
            structure["id"] != ROOT_ID,
            structure["id"],
        ),
    )


def parse_arm_mesh_filename(mesh_path: Path) -> tuple[str, int, str, int]:
    match = ARM_MESH_RE.match(mesh_path.name)

    if match is None:
        raise ValueError(f"Could not parse ARM mesh filename: {mesh_path.name}")

    source_atlas, level, region_name, region_id = match.groups()

    return source_atlas, int(level), region_name, int(region_id)


def load_gifti_mesh(mesh_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load vertices and triangular faces from a GIFTI surface file.

    Parameters
    ----------
    mesh_path : Path
        Path to a GIFTI surface file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mesh vertices and triangular faces.
    """
    gii = nib.load(str(mesh_path))
    vertices = None
    faces = None

    for darray in gii.darrays:
        data = np.asarray(darray.data)

        if darray.intent == 1008:  # NIFTI_INTENT_POINTSET
            vertices = data

        elif darray.intent == 1009:  # NIFTI_INTENT_TRIANGLE
            faces = data.astype(int)

    if vertices is None or faces is None:
        raise ValueError(f"Could not find vertices and faces in {mesh_path}")

    return vertices, faces


def load_gifti_mesh_in_voxel_space(
    mesh_path: Path,
    ras_mm_to_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GIFTI mesh and convert vertices to NMT voxel space.

    NMT GIFTI surfaces are stored in anatomical RAS millimetres. BrainGlobe
    wrapup can scale voxel-space mesh points into microns when
    ``scale_meshes=True`` is used.

    Parameters
    ----------
    mesh_path : Path
        Path to a GIFTI surface file.
    ras_mm_to_voxel : np.ndarray
        Inverse NIfTI affine mapping RAS millimetres to voxel coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Mesh vertices in voxel coordinates and triangular faces.
    """
    vertices_ras_mm, faces = load_gifti_mesh(mesh_path)
    vertices_voxel = nib.affines.apply_affine(ras_mm_to_voxel, vertices_ras_mm)

    return vertices_voxel, faces


def load_combined_gifti_mesh_in_voxel_space(
    mesh_paths: list[Path],
    ras_mm_to_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load, voxel-transform, and combine multiple GIFTI surface meshes.

    Parameters
    ----------
    mesh_paths : list[Path]
        GIFTI mesh paths to combine.
    ras_mm_to_voxel : np.ndarray
        Inverse NIfTI affine mapping RAS millimetres to voxel coordinates.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Combined vertices and triangular faces.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh_path in mesh_paths:
        vertices, faces = load_gifti_mesh_in_voxel_space(
            mesh_path=mesh_path,
            ras_mm_to_voxel=ras_mm_to_voxel,
        )

        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]

    return np.vstack(all_vertices), np.vstack(all_faces)


def collect_source_meshes(
    surfaces_dir: Path,
    source_to_canonical: dict[int, int],
    canonical_info_by_id: dict[int, dict],
    structure_ids: set[int],
) -> dict[int, dict]:
    """Collect CHARM/SARM GIFTI mesh paths by merged ARM structure ID."""
    mesh_sources = {}
    atlas_dirs = {
        "CHARM": surfaces_dir / "atlases" / "CHARM",
        "SARM": surfaces_dir / "atlases" / "SARM",
    }

    for source_atlas, atlas_dir in atlas_dirs.items():
        if not atlas_dir.exists():
            raise FileNotFoundError(
                f"Could not find ARM source mesh atlas directory: {atlas_dir}"
            )

        for level in range(1, 7):
            level_dir = atlas_dir / f"Level_{level}"

            if not level_dir.exists():
                raise FileNotFoundError(
                    f"Could not find {source_atlas} mesh level: {level_dir}"
                )

            for mesh_path in sorted(
                level_dir.glob(f"{source_atlas}_{level}.*.k*.gii")
            ):
                file_source_atlas, file_level, region_name, region_id = (
                    parse_arm_mesh_filename(mesh_path)
                )

                if file_source_atlas != source_atlas or file_level != level:
                    raise ValueError(
                        f"Mesh filename does not match folder: {mesh_path}"
                    )

                canonical_id = source_to_canonical.get(region_id)

                if canonical_id not in structure_ids:
                    continue

                expected_domain = canonical_info_by_id[canonical_id]["domain"]
                expected_atlas = "CHARM" if expected_domain == "cortex" else "SARM"

                if source_atlas != expected_atlas:
                    raise ValueError(
                        f"Mesh {mesh_path} belongs to {source_atlas}, but "
                        f"canonical ARM ID {canonical_id} is "
                        f"{expected_domain}"
                    )

                canonical_name = strip_hemisphere_prefix(region_name)
                previous_source = mesh_sources.get(canonical_id)

                if previous_source is not None:
                    if previous_source["name"] != canonical_name:
                        raise ValueError(
                            f"Region ID {canonical_id} has conflicting mesh "
                            f"names: {previous_source['name']} and "
                            f"{canonical_name}"
                        )

                    # Repeated IDs can have meshes at multiple levels. Keep
                    # the deepest level, and combine left/right meshes within
                    # that level.
                    if level < previous_source["level"]:
                        continue

                    if level > previous_source["level"]:
                        previous_source["level"] = level
                        previous_source["paths"] = []

                    previous_source["paths"].append(mesh_path)
                    continue

                mesh_sources[canonical_id] = {
                    "level": level,
                    "name": canonical_name,
                    "paths": [mesh_path],
                }

    for canonical_id, mesh_source in mesh_sources.items():
        expected_source_ids = canonical_info_by_id[canonical_id]["source_ids"]

        if len(mesh_source["paths"]) != len(expected_source_ids):
            raise ValueError(
                f"Expected {len(expected_source_ids)} source meshes for "
                f"merged ARM ID {canonical_id}, found "
                f"{len(mesh_source['paths'])}: {mesh_source['paths']}"
            )

    return mesh_sources


def retrieve_or_construct_meshes(
    nmt_dir: Path,
    structures: list[dict],
    working_dir: Path,
    annotation_volume: np.ndarray | None = None,
    cortex_surface: str = "gray",
) -> dict[int, Path]:
    """
    Retrieve and construct ARM meshes.

    Supplied CHARM and SARM GIFTI meshes are converted to OBJ files and
    left/right source meshes are merged into one canonical mesh per ARM
    structure. A synthetic root mesh is created from the full-head NMT
    brainmask, a cortex mesh is created from the NMT cortical pial surfaces,
    and a subcortex mesh is created from the merged ARM annotation mask.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.
    structures : list[dict]
        BrainGlobe-compatible structure list.
    working_dir : Path
        Atlas working directory where converted meshes should be written.
    annotation_volume : np.ndarray or None, optional
        Merged annotation volume used to construct parent meshes and verify
        mesh coverage.

    Returns
    -------
    dict[int, Path]
        Mapping from structure ID to converted OBJ mesh path.
    """
    standard_dir = resolve_standard_nmt_dir(nmt_dir)
    reference_path = standard_dir / NMT_REFERENCE_FILENAME
    brainmask_path = standard_dir / NMT_BRAINMASK_FILENAME
    nmt_root_dir = standard_dir.parent
    surfaces_dir = nmt_root_dir / "NMT_v2.1_sym_surfaces"
    output_mesh_dir = Path(working_dir) / "meshes"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    arm_table = pd.read_csv(nmt_root_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, canonical_info_by_id = build_arm_id_mappings(
        arm_table
    )

    if annotation_volume is None:
        annotation_path = (
            standard_dir / "supplemental_ARM" / ARM_ANNOTATION_FILENAME
        )
        annotation_volume = load_nii(annotation_path, as_array=True).astype(
            np.uint32
        )
        annotation_volume = remap_annotation_labels(
            annotation_volume,
            source_to_canonical,
        )

    reference_img = nib.load(str(reference_path))
    ras_mm_to_voxel = np.linalg.inv(reference_img.affine)
    structure_ids = {int(structure["id"]) for structure in structures}
    meshes_dict = {}

    print("Creating ARM root, cortex, and subcortex parent meshes")

    root_mesh_path = output_mesh_dir / f"{ROOT_ID}.obj"
    root_mask = load_nii(brainmask_path, as_array=True).astype(np.uint8)
    extract_mesh_from_mask(
        root_mask,
        obj_filepath=root_mesh_path,
        smooth=True,
        closing_n_iters=8,
        decimate_fraction=0.6,
    )
    meshes_dict[ROOT_ID] = root_mesh_path

    subcortex_ids = [
        structure_id
        for structure_id, info in canonical_info_by_id.items()
        if info["domain"] == "subcortex"
    ]

    cortex_mesh_path = output_mesh_dir / f"{CORTEX_ID}.obj"
    cortex_surface_paths = [
        surfaces_dir / f"lh.{cortex_surface}_surface.rsl.gii",
        surfaces_dir / f"rh.{cortex_surface}_surface.rsl.gii",
    ]

    for path in cortex_surface_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find cortex surface mesh: {path}"
            )

    cortex_vertices, cortex_faces = load_combined_gifti_mesh_in_voxel_space(
        mesh_paths=cortex_surface_paths,
        ras_mm_to_voxel=ras_mm_to_voxel,
    )
    cortex_mesh = mio.Mesh(
        points=cortex_vertices,
        cells=[("triangle", cortex_faces)],
    )
    mio.write(cortex_mesh_path, cortex_mesh)
    meshes_dict[CORTEX_ID] = cortex_mesh_path

    subcortex_mesh_path = output_mesh_dir / f"{SUBCORTEX_ID}.obj"
    subcortex_mask = np.isin(annotation_volume, subcortex_ids).astype(np.uint8)
    extract_mesh_from_mask(
        subcortex_mask,
        obj_filepath=subcortex_mesh_path,
        smooth=True,
        closing_n_iters=8,
        decimate_fraction=0.6,
    )
    meshes_dict[SUBCORTEX_ID] = subcortex_mesh_path

    print("Converting CHARM/SARM GIFTI meshes to merged ARM OBJ files")

    mesh_sources = collect_source_meshes(
        surfaces_dir=surfaces_dir,
        source_to_canonical=source_to_canonical,
        canonical_info_by_id=canonical_info_by_id,
        structure_ids=structure_ids,
    )

    for canonical_id, mesh_source in sorted(mesh_sources.items()):
        vertices, faces = load_combined_gifti_mesh_in_voxel_space(
            mesh_paths=mesh_source["paths"],
            ras_mm_to_voxel=ras_mm_to_voxel,
        )
        output_mesh_path = output_mesh_dir / f"{canonical_id}.obj"
        region_mesh = mio.Mesh(
            points=vertices,
            cells=[("triangle", faces)],
        )
        mio.write(output_mesh_path, region_mesh)
        meshes_dict[canonical_id] = output_mesh_path

    structure_ids_without_mesh = sorted(structure_ids - set(meshes_dict))
    if structure_ids_without_mesh:
        raise ValueError(
            "Some ARM structures do not have supplied or generated meshes: "
            f"{structure_ids_without_mesh}"
        )

    annotation_ids = {int(value) for value in np.unique(annotation_volume)}
    annotation_ids.discard(0)
    annotation_ids_without_mesh = sorted(annotation_ids - set(meshes_dict))

    if annotation_ids_without_mesh:
        raise ValueError(
            "Some labels present in the annotation volume do not have meshes: "
            f"{annotation_ids_without_mesh}"
        )

    return meshes_dict


if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    # Fail early if any version of this atlas already exists.
    atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
    existing = list(bg_root_dir.glob(f"{atlas_prefix}_v*"))

    if existing:
        raise FileExistsError(f"Atlas output already exists in {bg_root_dir}.")

    nmt_dir = download_resources(bg_root_dir)
    reference_volume, annotation_volume = retrieve_reference_and_annotation(
        nmt_dir
    )
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information(nmt_dir)
    check_struct_consistency(structures)
    meshes_dict = retrieve_or_construct_meshes(
        nmt_dir=nmt_dir,
        structures=structures,
        working_dir=bg_root_dir,
        annotation_volume=annotation_volume,
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
        annotation_stack=annotation_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=bg_root_dir,
        hemispheres_stack=hemispheres_stack,
        cleanup_files=False,
        compress=True,
        scale_meshes=True,
        atlas_packager=ATLAS_PACKAGER,
    )

    print("Packaged atlas:", output_filename)
