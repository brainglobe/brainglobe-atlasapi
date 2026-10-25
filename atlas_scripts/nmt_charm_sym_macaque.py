"""Atlas generation script for the symmetric NMT CHARM macaque atlas."""

import re
from pathlib import Path

import meshio as mio
import nibabel as nib
import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

### Metadata
__version__ = 0
ATLAS_NAME = "nmt_charm_sym_macaque"
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
ORIENTATION = "lpi"
ROOT_ID = 9999
RESOLUTION = 250  # microns
ATLAS_PACKAGER = "Amirreza Bahramani"

NMT_SYM_URL = (
    "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/"
    "NMT_v2.1_sym.tgz"
)
NMT_SYM_HASH = (
    "sha256:35c8770f050403a8e77416521116131fddeead82c732f929fd70e1b0f0ddb51c"
)

NMT_REFERENCE_FILENAME = "NMT_v2.1_sym_fh.nii.gz"
NMT_SKULL_STRIPPED_FILENAME = "NMT_v2.1_sym_fh_SS.nii.gz"
CHARM_ANNOTATION_FILENAME = "CHARM_6_in_NMT_v2.1_sym_fh.nii.gz"
CHARM_MESH_RE = re.compile(r"^CHARM_(\d+)\.(.+)\.k(\d+)\.gii$")
HEMISPHERE_PREFIX_RE = re.compile(r"^C[LR]_")

# CHARM and ARM use the same full-head NMT v2.1 template and voxel grid.
TEMPLATE_INFO = {
    "name": "nmt_arm_sym_macaque-template",
    "version": "3.0",
    "use_existing": True,
}
SKULL_STRIPPED_TEMPLATE_INFO = {
    "name": "nmt_arm_sym_macaque-skull_stripped-template",
    "version": "3.0",
    "use_existing": True,
}
COORDINATE_SPACE_INFO = {
    "name": "nmt_arm_sym_macaque-space",
    "version": "3.0",
    "use_existing": True,
}


def download_resources(working_dir: Path) -> Path:
    """Download and extract the symmetric NMT v2.1 dataset."""
    download_dir = working_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)

    pooch.retrieve(
        url=NMT_SYM_URL,
        known_hash=NMT_SYM_HASH,
        path=download_dir,
        fname="NMT_v2.1_sym.tgz",
        processor=pooch.Untar(extract_dir="NMT_v2.1_sym"),
        progressbar=True,
    )

    return download_dir / "NMT_v2.1_sym" / "NMT_v2.1_sym"


def build_charm_id_mappings(charm_table: pd.DataFrame) -> dict[int, int]:
    """Map left/right source IDs to merged CHARM structure IDs."""
    source_to_canonical = {}

    for level in range(1, 7):
        name_column = f"Level_{level}"
        acronym_column = f"Level_{level}_abbr"
        id_column = f"Level_{level}_index"
        unique_structures = charm_table[
            [name_column, acronym_column, id_column]
        ].drop_duplicates()

        unique_structures = unique_structures.assign(
            canonical_name=unique_structures[name_column]
            .str.strip()
            .str.replace(HEMISPHERE_PREFIX_RE, "", regex=True),
            canonical_acronym=unique_structures[acronym_column]
            .str.strip()
            .str.replace(HEMISPHERE_PREFIX_RE, "", regex=True),
        )

        grouped = unique_structures.groupby(
            ["canonical_name", "canonical_acronym"],
            sort=False,
            dropna=False,
        )

        for _, group in grouped:
            source_ids = sorted(int(value) for value in group[id_column])
            canonical_id = min(source_ids)
            source_to_canonical.update(dict.fromkeys(source_ids, canonical_id))

    return source_to_canonical


def normalize_to_uint16(image: np.ndarray) -> np.ndarray:
    """Scale image intensities to the full uint16 range."""
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max()
    return (image * np.iinfo(np.uint16).max).astype(np.uint16)


def retrieve_reference_and_annotation(
    nmt_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the full-head NMT reference and merged finest CHARM annotation."""
    full_head_dir = nmt_dir / "NMT_v2.1_sym_fh"
    reference = load_nii(
        full_head_dir / NMT_REFERENCE_FILENAME,
        as_array=True,
    )
    reference = normalize_to_uint16(reference)

    annotation = load_nii(
        full_head_dir / "supplemental_CHARM" / CHARM_ANNOTATION_FILENAME,
        as_array=True,
    ).astype(np.uint32)
    charm_table = pd.read_csv(nmt_dir / "tables_CHARM" / "CHARM_key_table.csv")
    source_to_canonical = build_charm_id_mappings(charm_table)
    lookup = np.arange(max(source_to_canonical) + 1, dtype=np.uint32)
    for source_id, canonical_id in source_to_canonical.items():
        lookup[source_id] = canonical_id
    annotation = lookup[annotation]

    return reference, annotation


def retrieve_additional_references(
    nmt_dir: Path,
) -> list[tuple[dict, np.ndarray]]:
    """Load the skull-stripped reference and reuse its published component."""
    skull_stripped = load_nii(
        nmt_dir / "NMT_v2.1_sym_fh" / NMT_SKULL_STRIPPED_FILENAME,
        as_array=True,
    )
    skull_stripped = normalize_to_uint16(skull_stripped)

    return [(SKULL_STRIPPED_TEMPLATE_INFO, skull_stripped)]


def retrieve_hemisphere_map() -> np.ndarray | None:
    """Return no hemisphere map because this atlas is symmetric."""
    return None


def retrieve_structure_information(nmt_dir: Path) -> list[dict]:
    """Convert the CHARM hierarchy table to BrainGlobe structures."""
    charm_table = pd.read_csv(nmt_dir / "tables_CHARM" / "CHARM_key_table.csv")
    source_to_canonical = build_charm_id_mappings(charm_table)
    palette_lines = [
        line.strip()
        for line in (nmt_dir / "tables_CHARM" / "hue_CHARM_cmap.pal")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    rgb_triplets = {
        structure_id: list(bytes.fromhex(hex_color.removeprefix("#")))
        for structure_id, hex_color in enumerate(palette_lines[1:], start=1)
    }

    structures_by_id = {
        ROOT_ID: {
            "acronym": "root",
            "id": ROOT_ID,
            "name": "root",
            "structure_id_path": [ROOT_ID],
            "rgb_triplet": [255, 255, 255],
        },
    }

    for _, row in charm_table.iterrows():
        parsed_path = []
        parsed_by_id = {}

        for level in range(1, 7):
            structure = {
                "id": source_to_canonical[int(row[f"Level_{level}_index"])],
                "name": HEMISPHERE_PREFIX_RE.sub(
                    "", str(row[f"Level_{level}"]).strip()
                ),
                "acronym": HEMISPHERE_PREFIX_RE.sub(
                    "", str(row[f"Level_{level}_abbr"]).strip()
                ),
            }
            parsed_path.append(structure)
            parsed_by_id[structure["id"]] = structure

        id_path = [ROOT_ID]
        for structure in parsed_path:
            if structure["id"] != id_path[-1]:
                id_path.append(structure["id"])

        for structure_id in id_path[1:]:
            structure = parsed_by_id[structure_id]
            path_to_structure = id_path[: id_path.index(structure_id) + 1]
            structures_by_id[structure_id] = {
                "acronym": structure["acronym"],
                "id": structure_id,
                "name": structure["name"],
                "structure_id_path": path_to_structure,
                "rgb_triplet": rgb_triplets[structure_id],
            }

    return sorted(
        structures_by_id.values(),
        key=lambda structure: (
            len(structure["structure_id_path"]),
            structure["id"] != ROOT_ID,
            structure["id"],
        ),
    )


def write_combined_gifti_mesh(
    mesh_paths: list[Path],
    ras_mm_to_voxel: np.ndarray,
    output_path: Path,
) -> Path:
    """Load, transform, combine, and write GIFTI meshes as OBJ."""
    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for mesh_path in mesh_paths:
        gii = nib.load(str(mesh_path))
        vertices_ras_mm = np.asarray(
            gii.get_arrays_from_intent("NIFTI_INTENT_POINTSET")[0].data
        )
        vertices = nib.affines.apply_affine(
            ras_mm_to_voxel,
            vertices_ras_mm,
        )
        faces = np.asarray(
            gii.get_arrays_from_intent("NIFTI_INTENT_TRIANGLE")[0].data
        ).astype(int)

        all_vertices.append(vertices)
        all_faces.append(faces + vertex_offset)
        vertex_offset += vertices.shape[0]

    mesh = mio.Mesh(
        points=np.vstack(all_vertices),
        cells=[("triangle", np.vstack(all_faces))],
    )
    mio.write(output_path, mesh)
    return output_path


def collect_source_meshes(
    surfaces_dir: Path,
    source_to_canonical: dict[int, int],
    structure_ids: set[int],
) -> dict[int, dict]:
    """Collect CHARM GIFTI mesh paths by merged structure ID."""
    mesh_sources = {}
    charm_dir = surfaces_dir / "atlases" / "CHARM"

    for level in range(1, 7):
        level_dir = charm_dir / f"Level_{level}"

        for mesh_path in sorted(level_dir.glob(f"CHARM_{level}.*.k*.gii")):
            _, _, region_id = CHARM_MESH_RE.match(mesh_path.name).groups()
            canonical_id = source_to_canonical[int(region_id)]

            if canonical_id not in structure_ids:
                continue

            previous_source = mesh_sources.get(canonical_id)

            # Repeated IDs can have meshes at multiple levels. Keep the
            # deepest level, combining its left/right meshes.
            if previous_source is None or level > previous_source["level"]:
                mesh_sources[canonical_id] = {
                    "level": level,
                    "paths": [mesh_path],
                }
            elif level == previous_source["level"]:
                previous_source["paths"].append(mesh_path)

    return mesh_sources


def retrieve_or_construct_meshes(
    nmt_dir: Path,
    structures: list[dict],
    working_dir: Path,
) -> dict[int, Path]:
    """Convert the supplied CHARM and cortical-root GIFTI meshes to OBJ."""
    reference_path = nmt_dir / "NMT_v2.1_sym_fh" / NMT_REFERENCE_FILENAME
    surfaces_dir = nmt_dir / "NMT_v2.1_sym_surfaces"
    output_mesh_dir = working_dir / "meshes"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    charm_table = pd.read_csv(nmt_dir / "tables_CHARM" / "CHARM_key_table.csv")
    source_to_canonical = build_charm_id_mappings(charm_table)

    reference_img = nib.load(str(reference_path))
    ras_mm_to_voxel = np.linalg.inv(reference_img.affine)
    structure_ids = {int(structure["id"]) for structure in structures}
    meshes_dict = {}

    print("Creating CHARM root mesh from the left/right cortical surfaces")
    cortex_surface_paths = [
        surfaces_dir / "lh.gray_surface.rsl.gii",
        surfaces_dir / "rh.gray_surface.rsl.gii",
    ]
    meshes_dict[ROOT_ID] = write_combined_gifti_mesh(
        mesh_paths=cortex_surface_paths,
        ras_mm_to_voxel=ras_mm_to_voxel,
        output_path=output_mesh_dir / f"{ROOT_ID}.obj",
    )

    print("Converting CHARM GIFTI meshes to merged OBJ files")
    mesh_sources = collect_source_meshes(
        surfaces_dir=surfaces_dir,
        source_to_canonical=source_to_canonical,
        structure_ids=structure_ids,
    )

    for canonical_id, mesh_source in sorted(mesh_sources.items()):
        meshes_dict[canonical_id] = write_combined_gifti_mesh(
            mesh_paths=mesh_source["paths"],
            ras_mm_to_voxel=ras_mm_to_voxel,
            output_path=output_mesh_dir / f"{canonical_id}.obj",
        )

    return meshes_dict


if __name__ == "__main__":
    bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
    bg_root_dir.mkdir(parents=True, exist_ok=True)

    nmt_dir = download_resources(bg_root_dir)
    reference_volume, annotation_volume = retrieve_reference_and_annotation(
        nmt_dir
    )
    additional_references = retrieve_additional_references(nmt_dir)
    hemispheres_stack = retrieve_hemisphere_map()
    structures = retrieve_structure_information(nmt_dir)
    meshes_dict = retrieve_or_construct_meshes(
        nmt_dir=nmt_dir,
        structures=structures,
        working_dir=bg_root_dir,
    )

    output_directory = wrapup_atlas_from_data(
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
        template_info=TEMPLATE_INFO,
        coordinate_space_info=COORDINATE_SPACE_INFO,
        scale_meshes=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_references=additional_references,
        overwrite=True,
    )

    print("Packaged atlas:", output_directory)
