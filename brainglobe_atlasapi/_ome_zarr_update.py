"""Update OME-Zarr group metadata from atlas-assets.

Every ``.ome.zarr`` group in the atlas-assets bucket
(``s3://aind-scratch-data/david.feng/allen-atlas-assets-rc12``) ships a
``zarr.json`` that ``ngff-zarr`` refuses to open. There are three
deltas from valid v0.6, each independently fatal:

1. ``ome.version`` is ``"0.6.dev3"``; only release versions parse.
2. ``sequence`` transformations name their ``input`` and ``output``
   with bare strings; v0.6 wants ``{"path": ...}`` for an array and
   ``{"name": ...}`` for a coordinate system.
3. ``coordinateSystems`` sits beside ``multiscales`` instead of inside
   the multiscale that refers to it.

This module is a temporary shim, applied to each group as it is
downloaded.
"""

import json
import re
from pathlib import Path
from typing import Set, Union

# "0.6.dev3" -> "0.6". Release versions are left alone.
_DEV_VERSION = re.compile(r"^(\d+\.\d+)\.dev\d+$")


def _wrap_references(transforms, dataset_paths: Set[str]) -> bool:
    """Wrap bare-string transform references in place.

    Applied to every transformation reachable from a multiscale, not
    just the dataset-level ones, so any sequence the writer emitted with
    string endpoints is normalised.

    Parameters
    ----------
    transforms : sequence
        A ``coordinateTransformations`` list.
    dataset_paths : set of str
        Array paths declared by the enclosing multiscale. A reference
        matching one of these is an array reference (``path``);
        anything else names a coordinate system (``name``).

    Returns
    -------
    bool
        True if any reference was rewritten.
    """
    changed = False

    for transform in transforms:
        if not isinstance(transform, dict):
            continue

        for field in ("input", "output"):
            value = transform.get(field)
            if isinstance(value, str):
                transform[field] = (
                    {"path": value}
                    if value in dataset_paths
                    else {"name": value}
                )
                changed = True

        nested = transform.get("transformations")
        if nested:
            changed |= _wrap_references(nested, dataset_paths)

    return changed


def update_ome_zarr_attributes(group_path: Union[str, Path]) -> bool:
    """Rewrite an old ``zarr.json`` in place.

    A no-op on valid metadata and on paths that hold no OME attributes,
    so it is safe to call on any downloaded group.

    Parameters
    ----------
    group_path : str or Path
        Directory of an OME-Zarr group, e.g. ``template.ome.zarr``.

    Returns
    -------
    bool
        True if the file was rewritten.
    """
    zarr_json = Path(group_path) / "zarr.json"
    if not zarr_json.is_file():
        return False

    metadata = json.loads(zarr_json.read_text())
    ome = metadata.get("attributes", {}).get("ome")
    if not isinstance(ome, dict):
        return False

    changed = False

    dev_version = _DEV_VERSION.match(str(ome.get("version", "")))
    if dev_version:
        ome["version"] = dev_version.group(1)
        changed = True

    multiscales = ome.get("multiscales", [])

    # Only relocate the coordinate systems once there is somewhere to
    # put them, so a group without multiscales never loses the block.
    systems = ome.get("coordinateSystems")
    if systems is not None and multiscales:
        for multiscale in multiscales:
            multiscale.setdefault("coordinateSystems", systems)
        del ome["coordinateSystems"]
        changed = True

    for multiscale in multiscales:
        datasets = multiscale.get("datasets", [])
        dataset_paths = {
            dataset["path"] for dataset in datasets if "path" in dataset
        }
        changed |= _wrap_references(
            multiscale.get("coordinateTransformations", []), dataset_paths
        )
        for dataset in datasets:
            changed |= _wrap_references(
                dataset.get("coordinateTransformations", []), dataset_paths
            )

    if changed:
        zarr_json.write_text(json.dumps(metadata, indent=2))

    return changed
