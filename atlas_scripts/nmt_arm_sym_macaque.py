"""Atlas generation script for the ARM macaque atlas."""

import colorsys
import re
from pathlib import Path

import meshio as mio
import nibabel as nib
import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    extract_mesh_from_mask,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data

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

NMT_REFERENCE_FILENAME = "NMT_v2.1_sym_fh.nii.gz"
NMT_SKULL_STRIPPED_FILENAME = "NMT_v2.1_sym_fh_SS.nii.gz"
NMT_BRAINMASK_FILENAME = "NMT_v2.1_sym_fh_brainmask.nii.gz"
ARM_ANNOTATION_FILENAME = "ARM_6_in_NMT_v2.1_sym_fh.nii.gz"
ARM_MESH_RE = re.compile(r"^(CHARM|SARM)_(\d+)\.(.+)\.k(\d+)\.gii$")
HEMISPHERE_PREFIX_RE = re.compile(r"^(C[LR]|S[LR])_")

# Disambiguate the CHARM/SARM acronyms that collide in the merged ARM atlas.
ARM_ACRONYM_REPLACEMENTS = {
    226: "paI",  # parainsular cortex
    1282: "PML",  # paramedian lobule
    1133: "CeM",  # central medial thalamus
    1202: "RN",  # red nucleus
    1106: "RMN",  # retromammillary hypothalamus
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
            domain=unique_structures["Level_0"].str.strip().str.lower(),
            canonical_name=unique_structures[name_column]
            .str.strip()
            .str.replace(HEMISPHERE_PREFIX_RE, "", regex=True),
            canonical_acronym=unique_structures[acronym_column]
            .str.strip()
            .str.replace(HEMISPHERE_PREFIX_RE, "", regex=True),
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
            canonical_id = min(source_ids)
            canonical_info_by_id[canonical_id] = {
                "domain": domain,
                "name": canonical_name,
                "acronym": canonical_acronym,
                "source_ids": source_ids,
            }
            source_to_canonical.update(dict.fromkeys(source_ids, canonical_id))

    return source_to_canonical, canonical_info_by_id


def retrieve_reference_and_annotation(
    nmt_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the full-head NMT reference and merged finest ARM annotation."""
    full_head_dir = nmt_dir / "NMT_v2.1_sym_fh"
    reference = load_nii(
        full_head_dir / NMT_REFERENCE_FILENAME,
        as_array=True,
    ).astype(np.float32)
    reference -= reference.min()
    reference /= reference.max()
    reference = (reference * np.iinfo(np.uint16).max).astype(np.uint16)

    annotation = load_nii(
        full_head_dir / "supplemental_ARM" / ARM_ANNOTATION_FILENAME,
        as_array=True,
    ).astype(np.uint32)
    arm_table = pd.read_csv(nmt_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, _ = build_arm_id_mappings(arm_table)
    lookup = np.arange(max(source_to_canonical) + 1, dtype=np.uint32)
    for source_id, canonical_id in source_to_canonical.items():
        lookup[source_id] = canonical_id
    annotation = lookup[annotation]

    return reference, annotation


def retrieve_additional_references(nmt_dir: Path) -> dict[str, np.ndarray]:
    """Load the skull-stripped full-head NMT reference."""
    skull_stripped = load_nii(
        nmt_dir / "NMT_v2.1_sym_fh" / NMT_SKULL_STRIPPED_FILENAME,
        as_array=True,
    ).astype(np.float32)
    skull_stripped -= skull_stripped.min()
    skull_stripped /= skull_stripped.max()
    skull_stripped = (skull_stripped * np.iinfo(np.uint16).max).astype(
        np.uint16
    )

    return {"skull_stripped": skull_stripped}


def retrieve_hemisphere_map() -> np.ndarray | None:
    """Return no hemisphere map because this atlas is symmetric."""
    return None


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
        acronym: list(bytes.fromhex(hex_color.removeprefix("#")))
        for acronym, hex_color in base_color_hex.items()
    }
    rgb_triplets = {}
    level_1_children = {}

    subcortex_table = arm_table[
        arm_table["Level_0"].str.strip().str.lower() == "subcortex"
    ]

    for _, row in subcortex_table.iterrows():
        parsed_path = []

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


def retrieve_structure_information(nmt_dir: Path) -> list[dict]:
    """Convert the ARM hierarchy table to BrainGlobe structures."""
    arm_table = pd.read_csv(nmt_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, _ = build_arm_id_mappings(arm_table)
    palette_lines = [
        line.strip()
        for line in (nmt_dir / "tables_CHARM" / "hue_CHARM_cmap.pal")
        .read_text()
        .splitlines()
        if line.strip()
    ]
    charm_rgb_triplets = {
        structure_id: list(bytes.fromhex(hex_color.removeprefix("#")))
        for structure_id, hex_color in enumerate(palette_lines[1:], start=1)
    }
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
        domain = row["Level_0"].strip().lower()
        domain_id = CORTEX_ID if domain == "cortex" else SUBCORTEX_ID
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

        id_path = [ROOT_ID, domain_id]
        for structure in parsed_path:
            if structure["id"] != id_path[-1]:
                id_path.append(structure["id"])

        for structure_id in id_path[2:]:
            structure = parsed_by_id[structure_id]
            path_to_structure = id_path[: id_path.index(structure_id) + 1]
            structures_by_id[structure_id] = {
                "acronym": ARM_ACRONYM_REPLACEMENTS.get(
                    structure_id,
                    structure["acronym"],
                ),
                "id": structure_id,
                "name": structure["name"],
                "structure_id_path": path_to_structure,
                "rgb_triplet": (
                    charm_rgb_triplets[structure_id]
                    if domain == "cortex"
                    else sarm_rgb_triplets[structure_id]
                ),
            }

    return sorted(
        structures_by_id.values(),
        key=lambda structure: (
            len(structure["structure_id_path"]),
            structure["id"] != ROOT_ID,
            structure["id"],
        ),
    )


def load_combined_gifti_mesh_in_voxel_space(
    mesh_paths: list[Path],
    ras_mm_to_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, transform, and combine GIFTI meshes."""
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

    return np.vstack(all_vertices), np.vstack(all_faces)


def collect_source_meshes(
    surfaces_dir: Path,
    source_to_canonical: dict[int, int],
    structure_ids: set[int],
) -> dict[int, dict]:
    """Collect CHARM/SARM GIFTI mesh paths by merged ARM structure ID."""
    mesh_sources = {}
    atlas_dirs = {
        "CHARM": surfaces_dir / "atlases" / "CHARM",
        "SARM": surfaces_dir / "atlases" / "SARM",
    }

    for source_atlas, atlas_dir in atlas_dirs.items():
        for level in range(1, 7):
            level_dir = atlas_dir / f"Level_{level}"

            for mesh_path in sorted(
                level_dir.glob(f"{source_atlas}_{level}.*.k*.gii")
            ):
                _, _, _, region_id = ARM_MESH_RE.match(mesh_path.name).groups()
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
    annotation_volume: np.ndarray,
) -> dict[int, Path]:
    """Construct parent meshes and convert the supplied ARM GIFTI meshes."""
    full_head_dir = nmt_dir / "NMT_v2.1_sym_fh"
    reference_path = full_head_dir / NMT_REFERENCE_FILENAME
    brainmask_path = full_head_dir / NMT_BRAINMASK_FILENAME
    surfaces_dir = nmt_dir / "NMT_v2.1_sym_surfaces"
    output_mesh_dir = working_dir / "meshes"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    arm_table = pd.read_csv(nmt_dir / "tables_ARM" / "ARM_key_table.csv")
    source_to_canonical, canonical_info_by_id = build_arm_id_mappings(
        arm_table
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
        surfaces_dir / "lh.gray_surface.rsl.gii",
        surfaces_dir / "rh.gray_surface.rsl.gii",
    ]

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
        scale_meshes=True,
        atlas_packager=ATLAS_PACKAGER,
        additional_references=additional_references,
    )

    print("Packaged atlas:", output_filename)
