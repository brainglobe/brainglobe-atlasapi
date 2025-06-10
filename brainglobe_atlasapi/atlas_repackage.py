import shutil
from pathlib import Path

import numpy.typing as npt
import yaml
import zarr

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.descriptors import (
    ANNOTATION_DTYPE,
    HEMISPHERES_DTYPE,
    MESHES_DIRNAME,
    REFERENCE_DTYPE,
    STRUCTURES_FILENAME,
)


def _write_zip_store(path: Path, data: zarr.Array):
    """
    Write a Zarr array to a zip store at the specified path.

    Parameters
    ----------
    path : Path
        The path where the zip store will be created.
    data : zarr.Array
        The Zarr array to write to the zip store.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    store = zarr.storage.ZipStore(path, mode="w")
    zarr.save_array(store, data, zarr_format=3)
    store.close()


def write_template(
    data: npt.NDArray[REFERENCE_DTYPE],
    resolution: int | float,
    template_name: str,
    dest_path: Path,
):
    templates_path = (
        dest_path / "templates" / template_name / f"{resolution}um.zarr.zip"
    )
    _write_zip_store(templates_path, data)


def write_hemispheres(
    data: npt.NDArray[HEMISPHERES_DTYPE],
    resolution: int | float,
    template_name: str,
    dest_path: Path,
):
    hemispheres_path = (
        dest_path / "hemispheres" / template_name / f"{resolution}um.zarr.zip"
    )
    _write_zip_store(hemispheres_path, data)


def write_annotation(
    data: npt.NDArray[ANNOTATION_DTYPE],
    resolution: int | float,
    annotation_name: str,
    dest_path: Path,
):
    annotation_path = (
        dest_path
        / "annotations"
        / annotation_name
        / f"{resolution}um.zarr.zip"
    )
    _write_zip_store(annotation_path, data)


def copy_meshes(atlas: BrainGlobeAtlas, dest_path: Path, annotation_name: str):
    meshes_path = dest_path / "meshes" / annotation_name
    meshes_path.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        atlas.root_dir / MESHES_DIRNAME, meshes_path, dirs_exist_ok=True
    )


def copy_structures(
    atlas: BrainGlobeAtlas, dest_path: Path, annotation_name: str
):
    structures_path = (
        dest_path / "annotations" / annotation_name / STRUCTURES_FILENAME
    )
    shutil.copy(atlas.root_dir / STRUCTURES_FILENAME, structures_path)
    structures_csv_path = (atlas.root_dir / STRUCTURES_FILENAME).with_suffix(
        ".csv"
    )
    shutil.copy(
        structures_csv_path,
        dest_path / "annotations" / annotation_name / structures_csv_path.name,
    )


def generate_atlas_yaml(
    atlas: BrainGlobeAtlas,
    dest_path: Path,
    template_name: str,
    annotation_name: str,
):
    """
    Generate a YAML file for the atlas with the necessary metadata.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The atlas object containing metadata and data.
    dest_path : Path
        The destination path where the YAML file will be saved.
    template_name : str
        The name of the template.
    annotation_name : str
        The name of the annotation.
    """
    metadata = {
        "name": atlas.metadata["name"],
        "citation": atlas.metadata["citation"],
        "atlas_link": atlas.metadata["atlas_link"],
        "species": atlas.metadata["species"],
        "symmetric": atlas.metadata["symmetric"],
        "resolution": list(atlas.resolution),
        "orientation": atlas.orientation,
        "atlas_version": ".".join([str(v) for v in atlas.local_version]),
        "shape": list(atlas.shape),
        "template_names": [
            template_name,
        ],
        "annotation_names": [
            annotation_name,
        ],
        "meshes": [
            annotation_name,
        ],
        "additional_references": list(atlas.metadata["additional_references"]),
    }

    yaml_path = dest_path / f"{atlas.atlas_name}.yaml"
    with open(yaml_path, "w") as yaml_file:
        yaml.dump(
            metadata, yaml_file, default_flow_style=False, sort_keys=False
        )


def repackage_atlas(
    atlas_name: str, dest_path: Path, template_name: str, annotation_name: str
):
    atlas = BrainGlobeAtlas(atlas_name)
    if atlas.resolution[0] == int(atlas.resolution[0]):
        resolution = int(atlas.resolution[0])
    else:
        resolution = atlas.resolution[0]

    dest_path.mkdir(parents=True, exist_ok=True)

    write_annotation(atlas.annotation, resolution, annotation_name, dest_path)
    write_template(atlas.reference, resolution, template_name, dest_path)

    copy_meshes(atlas, dest_path, annotation_name)
    copy_structures(atlas, dest_path, annotation_name)

    if not atlas.metadata["symmetric"]:
        write_hemispheres(
            atlas.hemispheres, resolution, template_name, dest_path
        )


if __name__ == "__main__":
    name = "allen_mouse_25um"
    dest = Path.home() / ".brainglobe" / "atlases"
    dest.mkdir(parents=True, exist_ok=True)

    repackage_atlas(name, dest, "allen_mouse_v3_2017", "allen_mouse_v1")
