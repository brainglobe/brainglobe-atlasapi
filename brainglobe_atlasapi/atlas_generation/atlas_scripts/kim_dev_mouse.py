import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pooch
from brainglobe_utils.IO.image import load_nii
from rich.progress import track

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from brainglobe_atlasapi.config import DEFAULT_WORKDIR
from brainglobe_atlasapi.structure_tree_util import get_structures_tree

ATLAS_NAME = "kim_dev_mouse"
SPECIES = "Mus musculus"
ATLAS_LINK = "https://kimlab.io/brain-map/DevCCF/"
CITATION = "Kronman, F.N., Liwang, J.K., Betty, R. et al. 2024, https://doi.org/10.1038/s41467-024-53254-w"
ORIENTATION = ["left", "superior", "posterior"]
ROOT_ID = 15564
VERSION = 3
PACKAGER = "Carlo Castoldi <castoldi[at]ipmc.cnrs.fr>"
ATLAS_FILE_URL = "https://doi.org/10.6084/m9.figshare.26377171.v1"

TIMEPOINTS = ("E11.5", "E13.5", "E15.5", "E18.5", "P04", "P14", "P56")
MODALITIES = (
    "LSFM",  # Light Sheet Fluorescence Microscopy
    "MRI-adc",  # MRI Apparent Diffusion Coefficient
    "MRI-dwi",  # MRI Difusion Weighted Imaging
    "MRI-fa",  # MRI Fractional Anisotropy
    "MRI-MTR",  # MRI Magnetization Transfer Ratio
    "MRI-T2",  # MRI T2-weighted
)


def pooch_init(download_dir_path: Path, timepoints: list[str]) -> pooch.Pooch:
    utils.check_internet_connection()

    empty_registry = {
        a + ".zip": None
        for a in [*timepoints, "DevCCFv1_OntologyStructure.xlsx"]
    }
    p = pooch.create(
        path=download_dir_path,
        base_url="doi:10.6084/m9.figshare.26377171.v1/",
        registry=empty_registry,
    )
    p.load_registry(
        Path(__file__).parent.parent / "hashes" / (ATLAS_NAME + ".txt")
    )
    return p


def fetch_animal(pooch_: pooch.Pooch, age: str, modality: str):
    assert age in TIMEPOINTS, f"Unknown age timepoint: '{age}'"
    archive = age + ".zip"
    if modality == "LSFM":
        resolution_um = 20
    elif modality in MODALITIES:
        match age:
            case "E11.5":
                resolution_um = 31.5
            case "E13.5":
                resolution_um = 34
            case "E15.5":
                resolution_um = 37.5
            case "E18.5":
                resolution_um = 40
            case _:
                resolution_um = 50
    else:
        raise RuntimeError(f"Unknown reference image modality: {modality}")
    members = [
        f"{age}/{age.replace('.','-')}_DevCCF_Annotations_{resolution_um}um.nii.gz",
        f"{age}/{age.replace('.','-')}_{modality}_{resolution_um}um.nii.gz",
    ]
    fetched_paths = pooch_.fetch(
        archive,
        progressbar=True,
        processor=pooch.Unzip(extract_dir=".", members=members),
    )
    # the file paths returned by pooch.ExtractorProcessor (superclass of Unzip)
    # may not respect the order of the given members.
    # see: https://github.com/fatiando/pooch/issues/457
    annotations_path = next(p for p in fetched_paths if p.endswith(members[0]))
    reference_path = next(p for p in fetched_paths if p.endswith(members[1]))
    annotations = load_nii(annotations_path, as_array=True)
    reference = load_nii(reference_path, as_array=True)
    dmin = np.min(reference)
    dmax = np.max(reference)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference = (reference - dmin) * dscale
    reference = reference.astype(np.uint16)
    return annotations, reference, resolution_um


def fetch_ontology(pooch_: pooch.Pooch):
    devccfv1_path = pooch_.fetch(
        "DevCCFv1_OntologyStructure.xlsx", progressbar=True
    )
    xl = pd.ExcelFile(devccfv1_path)
    # xl.sheet_names # it has two excel sheets
    # 'DevCCFv1_Ontology', 'README'
    df = xl.parse("DevCCFv1_Ontology", header=1)
    df = df[["Acronym", "ID16", "Name", "Structure ID Path16", "R", "G", "B"]]
    df.rename(
        columns={
            "Acronym": "acronym",
            "ID16": "id",
            "Name": "name",
            "Structure ID Path16": "structure_id_path",
            "R": "r",
            "G": "g",
            "B": "b",
        },
        inplace=True,
    )
    structures = list(df.to_dict(orient="index").values())
    for structure in structures:
        if structure["acronym"] == "mouse":
            structure["acronym"] = "root"
        structure_path = structure["structure_id_path"]
        structure["structure_id_path"] = [
            int(id) for id in structure_path.strip("/").split("/")
        ]
        structure["rgb_triplet"] = [
            structure["r"],
            structure["g"],
            structure["b"],
        ]
        del structure["r"]
        del structure["g"]
        del structure["b"]
    return structures


def create_meshes(
    output_path: str | Path,
    structures,
    annotation_volume,
    root_id,
    decimate_fraction,
):
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    tree = get_structures_tree(structures)

    labels = np.unique(annotation_volume).astype(np.uint16)

    for key, node in tree.nodes.items():
        if key in labels:
            is_label = True
        else:
            is_label = False
        node.data = Region(is_label)

    # Mesh creation
    closing_n_iters = 2
    smooth = True  # smooth meshes after creation
    start = time.time()
    for node in track(
        tree.nodes.values(),
        total=tree.size(),
        # list(tree.nodes.values())[:5],
        # total=5,
        description="Creating meshes",
    ):
        output_file = output_path / f"{node.identifier}.obj"
        if output_file.exists():
            continue
        create_region_mesh(
            (
                output_path,
                node,
                tree,
                labels,
                annotation_volume,
                root_id,
                closing_n_iters,
                decimate_fraction,
                smooth,
            )
        )

    print(
        "Finished mesh extraction in: ",
        round((time.time() - start) / 60, 2),
        " minutes",
    )
    return output_path


def create_mesh_dict(structures, meshes_dir_path):
    meshes_dict = dict()
    structures_with_mesh = []
    for s in structures:
        # Check if a mesh was created
        mesh_path = meshes_dir_path / f'{s["id"]}.obj'
        if not mesh_path.exists():
            print(f"No mesh file exists for: {s}, ignoring it")
            continue
        else:
            # Check that the mesh actually exists (i.e. not empty)
            if mesh_path.stat().st_size < 512:
                print(f"obj file for {s} is too small, ignoring it.")
                continue
        structures_with_mesh.append(s)
        meshes_dict[s["id"]] = mesh_path
    print(
        f"In the end, {len(structures_with_mesh)} "
        "structures with mesh are kept"
    )
    return meshes_dict, structures_with_mesh


class HideChoicesRawTextHelpFormatter(argparse.RawTextHelpFormatter):
    def _get_help_string(self, action):
        help_text = action.help
        if action.choices:
            help_text = help_text.split("(choices")[0].strip()
        return help_text

    def _format_action_invocation(self, action):
        if action.option_strings:
            return ", ".join(action.option_strings)
        if action.metavar:
            return action.metavar
        return super()._format_action_invocation(action)


arg_parser = argparse.ArgumentParser(
    formatter_class=HideChoicesRawTextHelpFormatter
)
timepoints_help = """the age timepoint at which the atlas will be generated.
    Options are:
        - E11.5         ⅂
        - E13.5         |- Embryonic day
        - E15.5         |
        - E18.5         ⅃
        - P04           ⅂
        - P14           |- Postnatal day
        - P56           ⅃

By default, it generates an atlas for each timepoint.
"""
modalities_help = """the reference image acquisition modality.
    Options are:
        - LSFM,         Light Sheet Fluorescence Microscopy
        - MRI-adc       MRI Apparent Diffusion Coefficient
        - MRI-dwi       MRI Difusion Weighted Imaging
        - MRI-fa        MRI Fractional Anisotropy
        - MRI-MTR       MRI Magnetization Transfer Ratio
        - MRI-T2        MRI T2-weighted

By default, LSFM
"""
decimate_fraction_help = (
    "fraction of original meshes' vertices to be kept."
    " Must be a number > 0 and <= 1.\nBy default, 0.2"
)
cached_meshes_help = (
    "A pre-computed atlas whose meshes can be used for the "
    + """chosen modality.
    Options are:
        - LSFM,         Light Sheet Fluorescence Microscopy
        - MRI-adc       MRI Apparent Diffusion Coefficient
        - MRI-dwi       MRI Difusion Weighted Imaging
        - MRI-fa        MRI Fractional Anisotropy
        - MRI-MTR       MRI Magnetization Transfer Ratio
        - MRI-T2        MRI T2-weighted
"""
)


def decimate_fraction_type(arg):
    try:
        f = float(arg)
        if 0 < f <= 1:
            return f
    except ValueError:
        pass
    raise argparse.ArgumentTypeError(
        f"invalid value: '{arg}' (must be a number > 0 and <= 1)"
    )


arg_parser.add_argument(
    "-t",
    "--timepoints",
    default=TIMEPOINTS,
    type=str,
    nargs="+",
    choices=TIMEPOINTS,
    help=timepoints_help,
)
arg_parser.add_argument(
    "-m",
    "--modality",
    default="LSFM",
    type=str,
    choices=MODALITIES,
    help=modalities_help,
)
arg_parser.add_argument(
    "-d",
    "--decimate-fraction",
    default=0.2,
    type=decimate_fraction_type,
    help=decimate_fraction_help,
)
arg_parser.add_argument(
    "-c",
    "--cached-meshes",
    type=str,
    choices=MODALITIES,
    help=cached_meshes_help,
)

if __name__ == "__main__":
    import sys

    params = vars(arg_parser.parse_args())
    timepoints = params["timepoints"]
    modality = params["modality"]
    decimate_fraction = params["decimate_fraction"]
    cached_modality = params["cached_meshes"]
    if (
        cached_modality is not None
        and modality.split("-")[0] != cached_modality.split("-")[0]
    ):
        # one is LSFM and the other is MRI-based
        print(
            f"Incompatible cached meshes '{cached_modality}' "
            + f"with modality '{modality}'!",
            file=sys.stderr,
        )
        exit(-1)
    bg_root_dir = DEFAULT_WORKDIR / ATLAS_NAME
    download_dir_path = bg_root_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True, parents=True)
    pooch_ = pooch_init(download_dir_path, timepoints)
    structures = fetch_ontology(pooch_)

    for age in timepoints:
        atlas_name = f"{ATLAS_NAME}_{age.replace('.', '-')}_{modality}"
        atlas_name = atlas_name.lower()
        annotation_volume, reference_volume, resolution_um = fetch_animal(
            pooch_, age, modality
        )
        atlas_dir = bg_root_dir / atlas_name
        atlas_dir.mkdir(exist_ok=True)
        print(f"Saving atlas data at {atlas_dir}")
        # Create meshes:
        if (
            cached_modality is not None
            and (
                cache_dir := bg_root_dir
                / f"{ATLAS_NAME}_{age.replace('.', '-')}"
                f"_{cached_modality}".lower() / "meshes"
            ).exists()
        ):
            meshes_dir_path = cache_dir
        else:
            meshes_dir_path = atlas_dir / "meshes"
            create_meshes(
                meshes_dir_path,
                structures,
                annotation_volume,
                ROOT_ID,
                decimate_fraction,
            )
        meshes_dict, structures_with_mesh = create_mesh_dict(
            structures, meshes_dir_path
        )
        # Wrap up, compress, and remove file:
        print("Finalising atlas")
        output_filename = wrapup_atlas_from_data(
            atlas_name=atlas_name,
            atlas_minor_version=VERSION,
            citation=CITATION,
            atlas_link=ATLAS_LINK,
            species=SPECIES,
            resolution=(resolution_um,) * 3,
            orientation=ORIENTATION,
            root_id=ROOT_ID,
            reference_stack=reference_volume,
            annotation_stack=annotation_volume,
            structures_list=structures_with_mesh,
            meshes_dict=meshes_dict,
            working_dir=atlas_dir,
            atlas_packager=PACKAGER,
            hemispheres_stack=None,  # it is symmetric
            cleanup_files=False,
            compress=True,
            scale_meshes=True,
        )
        print("Done. Atlas generated at: ", output_filename)
