"""Atlas generation script for the CHARM macaque atlas."""

import re
from pathlib import Path

import meshio as mio
import nibabel as nib
import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii

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
ATLAS_NAME = "charm_macaque"
CITATION = (
    "Jung et al., 2021. A comprehensive macaque fMRI pipeline and "
    "hierarchical atlas. NeuroImage. "
    "https://doi.org/10.1016/j.neuroimage.2021.117997"
)
SPECIES = "Macaca mulatta"
ATLAS_LINK = (
    "https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/"
    "macaque_tempatl/atlas_charm.html"
)
ATLAS_FILE_URL = (
    "https://afni.nimh.nih.gov/pub/dist/doc/htmldoc/nonhuman/"
    "macaque_tempatl/template_nmtv2.html#nh-macaque-template-nmtv2-sym-dl"
)
ORIENTATION = "lpi"
ROOT_ID = 9999
RESOLUTION = 250  # microns
ATLAS_PACKAGER = "Amirreza Bahramani"

NMT_SYM_URL = (
    "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/"
    "NMT_v2.0_sym.tgz"
)
NMT_SYM_HASH = (
    "sha256:9c455431ec1e8257fef4127c137e49f710aa43ef8a87f1bf73701b83d5ef7e6d"
)
STRUCTURE_RE = re.compile(r"^\s*(\d+):\s*(.*?)\s*\((.*?)\)\s*$")
CHARM_MESH_RE = re.compile(r"^CHARM_(\d+)\.(.+)\.k(\d+)\.gii$")
UNUSED_CHARM_STRUCTURE_IDS = set(range(140, 148))


def download_resources(working_dir: Path) -> Path:
    """
    Download and extract the symmetric NMT v2 dataset.

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
        fname="NMT_v2.0_sym.tgz",
        processor=pooch.Untar(extract_dir="NMT_v2.0_sym"),
        progressbar=True,
    )

    return download_dir / "NMT_v2.0_sym"


def resolve_standard_nmt_dir(nmt_dir: Path) -> Path:
    """
    Find the directory containing the standard 250 um symmetric NMT files.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.

    Returns
    -------
    Path
        Directory containing ``NMT_v2.0_sym.nii.gz``.
    """
    nmt_dir = Path(nmt_dir)
    candidates = sorted(
        [
            path
            for path in [nmt_dir, *nmt_dir.rglob("*")]
            if path.is_dir() and (path / "NMT_v2.0_sym.nii.gz").exists()
        ]
    )

    if not candidates:
        raise FileNotFoundError(
            "Could not find a directory containing NMT_v2.0_sym.nii.gz "
            f"inside {nmt_dir}"
        )

    return candidates[0]


def retrieve_reference_and_annotation(
    nmt_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve the NMT reference volume and finest CHARM annotation.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        The normalized reference volume and level-6 CHARM annotation volume.
    """
    standard_dir = resolve_standard_nmt_dir(nmt_dir)
    reference_path = standard_dir / "NMT_v2.0_sym.nii.gz"
    annotation_path = (
        standard_dir / "supplemental_CHARM" / "CHARM_6_in_NMT_v2.0_sym.nii.gz"
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

    return reference, annotation


def retrieve_hemisphere_map() -> np.ndarray | None:
    """
    Retrieve a hemisphere map for the atlas.

    Returns
    -------
    None
        The symmetric NMT v2 / CHARM atlas does not need a hemisphere map.
    """
    return None


def parse_charm_structure(value: str) -> dict:
    """
    Parse a CHARM hierarchy table entry.

    Parameters
    ----------
    value : str
        Entry formatted like ``82: caudal_dorsal_premotor_cortex (PMdc)``.

    Returns
    -------
    dict
        Parsed structure ID, name, and acronym.
    """
    match = STRUCTURE_RE.match(str(value))
    if match is None:
        raise ValueError(f"Could not parse CHARM structure entry: {value!r}")

    return {
        "id": int(match.group(1)),
        "name": match.group(2),
        "acronym": match.group(3),
    }


def collapse_repeated_ids(path: list[int]) -> list[int]:
    """
    Collapse repeated adjacent IDs in a CHARM structure path.

    CHARM may repeat the same ID across adjacent levels to preserve a
    six-column hierarchy table. BrainGlobe only needs the structure once in
    the path.

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

    if len(color_lines) < 246:
        raise ValueError(
            f"Expected at least 246 CHARM colors, found {len(color_lines)}"
        )

    return {
        structure_id: hex_to_rgb(color_lines[structure_id - 1])
        for structure_id in range(1, 247)
    }


def check_or_add_structure(
    structures_by_id: dict[int, dict],
    candidate: dict,
) -> None:
    """
    Add a structure, checking repeated CHARM entries are consistent.

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
                f"Conflicting CHARM hierarchy entry for ID {structure_id}.\n"
                f"Existing {field}: {existing[field]}\n"
                f"New {field}: {candidate[field]}"
            )


def remove_unused_charm_structures(structures: list[dict]) -> list[dict]:
    """
    Remove CHARM table entries not instantiated in the atlas data.

    The CHARM key table includes hippocampal IDs 140-147, but these IDs do
    not appear as voxel labels in any distributed CHARM annotation level, and
    the NMT package does not provide corresponding CHARM GIFTI meshes. Keeping
    them would add structures that cannot be selected from the annotation
    volume and cannot be represented by a mesh in BrainGlobe.

    Parameters
    ----------
    structures : list[dict]
        Full structure list parsed from the CHARM key table.

    Returns
    -------
    list[dict]
        Structure list with unused table-only IDs removed.
    """
    filtered_structures = [
        structure
        for structure in structures
        if structure["id"] not in UNUSED_CHARM_STRUCTURE_IDS
    ]

    invalid_paths = [
        structure
        for structure in filtered_structures
        if any(
            structure_id in UNUSED_CHARM_STRUCTURE_IDS
            for structure_id in structure["structure_id_path"]
        )
    ]

    if invalid_paths:
        raise ValueError(
            "Cannot remove unused CHARM structures because other structures "
            "still reference them in structure_id_path."
        )

    return filtered_structures


def retrieve_structure_information(nmt_dir: Path) -> list[dict]:
    """
    Retrieve BrainGlobe-compatible structure information for CHARM.

    CHARM is a six-level cortical hierarchy. The level-6 annotation is used
    as the annotation volume, while levels 1-5 define the structure paths for
    each region.

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
    charm_table_path = nmt_root_dir / "tables_CHARM" / "CHARM_key_table.csv"
    charm_table = pd.read_csv(charm_table_path)
    rgb_triplets = load_charm_rgb_triplets(nmt_root_dir)

    structures_by_id = {
        ROOT_ID: {
            "acronym": "root",
            "id": ROOT_ID,
            "name": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        }
    }

    for _, row in charm_table.iterrows():
        parsed_path = [
            parse_charm_structure(row[f"Level {level}"])
            for level in range(1, 7)
        ]
        raw_id_path = [ROOT_ID] + [
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
                    "within one CHARM row: "
                    f"{parsed_by_id[structure_id]} vs {structure}"
                )

            parsed_by_id[structure_id] = structure

        for structure_id in id_path[1:]:
            structure = parsed_by_id[structure_id]
            path_to_structure = id_path[: id_path.index(structure_id) + 1]
            candidate = {
                "acronym": structure["acronym"],
                "id": structure_id,
                "name": structure["name"],
                "structure_id_path": path_to_structure,
                "rgb_triplet": rgb_triplets[structure_id],
            }

            check_or_add_structure(structures_by_id, candidate)

    structures_list = list(structures_by_id.values())
    structures_list = remove_unused_charm_structures(structures_list)

    return sorted(
        structures_list,
        key=lambda structure: (structure["id"] != ROOT_ID, structure["id"]),
    )


def parse_charm_mesh_filename(mesh_path: Path) -> tuple[int, str, int]:
    """
    Extract CHARM level, region name, and region ID from a mesh filename.

    Parameters
    ----------
    mesh_path : Path
        Path to a CHARM GIFTI mesh file.

    Returns
    -------
    tuple[int, str, int]
        CHARM level, region name, and region ID.
    """
    match = CHARM_MESH_RE.match(mesh_path.name)

    if match is None:
        raise ValueError(
            f"Could not parse CHARM mesh filename: {mesh_path.name}"
        )

    level, region_name, region_id = match.groups()

    return int(level), region_name, int(region_id)


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


def retrieve_or_construct_meshes(
    nmt_dir: Path,
    structures: list[dict],
    working_dir: Path,
    annotation_volume: np.ndarray | None = None,
    root_surface: str = "gray",
) -> dict[int, Path]:
    """
    Retrieve CHARM meshes from the provided NMT GIFTI files.

    The CHARM download provides GIFTI meshes for regions across the six
    hierarchy levels. This function converts those meshes to OBJ files that
    can be read by BrainGlobe's wrapup step. A synthetic root mesh is created
    by combining the left and right NMT cortical surfaces.

    Parameters
    ----------
    nmt_dir : Path
        Root directory returned by ``download_resources``.
    structures : list[dict]
        BrainGlobe-compatible structure list.
    working_dir : Path
        Atlas working directory where converted meshes should be written.
    annotation_volume : np.ndarray or None, optional
        Annotation volume used to verify that all labels have meshes.
    root_surface : str, optional
        NMT surface type to use for the root mesh.

    Returns
    -------
    dict[int, Path]
        Mapping from structure ID to converted OBJ mesh path.
    """
    standard_dir = resolve_standard_nmt_dir(nmt_dir)
    reference_path = standard_dir / "NMT_v2.0_sym.nii.gz"
    surfaces_dir = standard_dir.parent / "NMT_v2.0_sym_surfaces"
    charm_surfaces_dir = surfaces_dir / "atlases" / "CHARM"
    output_mesh_dir = Path(working_dir) / "meshes"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    reference_img = nib.load(str(reference_path))
    ras_mm_to_voxel = np.linalg.inv(reference_img.affine)
    structure_ids = {int(structure["id"]) for structure in structures}
    meshes_dict = {}
    mesh_sources = {}

    root_surface_paths = [
        surfaces_dir / f"lh.{root_surface}_surface.rsl.gii",
        surfaces_dir / f"rh.{root_surface}_surface.rsl.gii",
    ]

    for path in root_surface_paths:
        if not path.exists():
            raise FileNotFoundError(
                f"Could not find root surface mesh: {path}"
            )

    root_vertices, root_faces = load_combined_gifti_mesh_in_voxel_space(
        mesh_paths=root_surface_paths,
        ras_mm_to_voxel=ras_mm_to_voxel,
    )
    root_mesh_path = output_mesh_dir / f"{ROOT_ID}.obj"
    root_mesh = mio.Mesh(
        points=root_vertices,
        cells=[("triangle", root_faces)],
    )
    mio.write(root_mesh_path, root_mesh)

    meshes_dict[ROOT_ID] = root_mesh_path
    mesh_sources[ROOT_ID] = {
        "level": 0,
        "name": "NMT cortical surface",
        "source": root_surface_paths,
    }

    print("Converting CHARM GIFTI meshes to OBJ files")

    for level in range(1, 7):
        level_dir = charm_surfaces_dir / f"Level_{level}"

        if not level_dir.exists():
            raise FileNotFoundError(
                f"Could not find CHARM mesh level: {level_dir}"
            )

        for mesh_path in sorted(level_dir.glob(f"CHARM_{level}.*.k*.gii")):
            file_level, region_name, region_id = parse_charm_mesh_filename(
                mesh_path
            )

            if file_level != level:
                raise ValueError(
                    f"Mesh filename level {file_level} does not match folder "
                    f"level {level}: {mesh_path}"
                )

            if region_id not in structure_ids:
                continue

            previous_source = mesh_sources.get(region_id)

            if previous_source is not None:
                if previous_source["name"] != region_name:
                    raise ValueError(
                        f"Region ID {region_id} has conflicting mesh names: "
                        f"{previous_source['name']} and {region_name}"
                    )

                # If a repeated region ID has meshes at multiple levels, keep
                # the deepest one because it best matches the final hierarchy.
                if level <= previous_source["level"]:
                    continue

            vertices, faces = load_gifti_mesh_in_voxel_space(
                mesh_path=mesh_path,
                ras_mm_to_voxel=ras_mm_to_voxel,
            )
            output_mesh_path = output_mesh_dir / f"{region_id}.obj"
            region_mesh = mio.Mesh(
                points=vertices,
                cells=[("triangle", faces)],
            )
            mio.write(output_mesh_path, region_mesh)

            meshes_dict[region_id] = output_mesh_path
            mesh_sources[region_id] = {
                "level": level,
                "name": region_name,
                "source": mesh_path,
            }

    structure_ids_without_mesh = sorted(structure_ids - set(meshes_dict))
    if structure_ids_without_mesh:
        raise ValueError(
            "Some CHARM structures do not have supplied meshes: "
            f"{structure_ids_without_mesh}"
        )

    if annotation_volume is not None:
        annotation_ids = {int(value) for value in np.unique(annotation_volume)}
        annotation_ids.discard(0)
        annotation_ids_without_mesh = sorted(annotation_ids - set(meshes_dict))

        if annotation_ids_without_mesh:
            raise ValueError(
                "Some labels present in the annotation volume do not have "
                f"meshes: {annotation_ids_without_mesh}"
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
