"""Convert atlas manifests into the metadata shape ``core.Atlas`` consumes.

BrainGlobe manifests already have that shape. atlas-assets manifests are
reshaped, with missing fields derived from zarr metadata and the asset name.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import zarr

from brainglobe_atlasapi.descriptors import V3_ANNOTATION_NAME

SPECIES_BY_NAME_SEGMENT = {
    "mouse": "Mus musculus",
    "human": "Homo sapiens",
    "marmoset": "Callithrix jacchus",
    "rhesusmacaque": "Macaca mulatta",
    "cynomolgusmacaque": "Macaca fascicularis",
}

# brainglobe-space origin letters, keyed by OME anatomical axis direction.
_ORIGIN_BY_AXIS_DIRECTION = {
    "anterior-to-posterior": "a",
    "posterior-to-anterior": "p",
    "superior-to-inferior": "s",
    "inferior-to-superior": "i",
    "dorsal-to-ventral": "s",
    "ventral-to-dorsal": "i",
    "left-to-right": "l",
    "right-to-left": "r",
    "caudal-to-rostral": "p",
    "rostral-to-caudal": "a",
}


def _species_from_name(atlas_name: str) -> Optional[str]:
    """Derive a binomial species name from an asset name.

    The specification's name grammar is explicitly non-normative and is
    not unambiguously parseable by position, so every segment is matched
    against a fixed vocabulary instead.

    Parameters
    ----------
    atlas_name : str
        Asset name, e.g. ``"allen-adult-mouse-ccf-atlas"``.

    Returns
    -------
    str or None
        Binomial name, or None when no segment matches the vocabulary.
    """
    for segment in atlas_name.split("-"):
        species = SPECIES_BY_NAME_SEGMENT.get(segment.lower())
        if species is not None:
            return species
    return None


def _intrinsic_axes(multiscale: dict) -> list:
    """Return the axes of the space the array indices live in.

    OME-Zarr 0.5 (the BrainGlobe bucket) declares one set of axes on the
    multiscale. 0.6 (atlas-assets) declares several coordinate systems
    and names, on each dataset, the one its indices map into; that
    intrinsic system is the one describing storage order, as opposed to
    any anatomical space derived from it.

    Parameters
    ----------
    multiscale : dict
        One entry of the ``multiscales`` list.

    Returns
    -------
    list
        The axis dictionaries, in storage order.

    Raises
    ------
    KeyError
        If neither shape declares axes.
    """
    axes = multiscale.get("axes")
    if axes is not None:
        return axes

    outputs = {
        transform["output"]["name"]
        for transform in multiscale["datasets"][0]["coordinateTransformations"]
        if "name" in transform.get("output", {})
    }
    for system in multiscale.get("coordinateSystems", []):
        if system["name"] in outputs:
            return system["axes"]

    raise KeyError(
        "Multiscale declares no axes and no coordinate system matching "
        f"the output(s) {sorted(outputs)} of its datasets."
    )


def _orientation_from_axes(ome_attrs: dict) -> str:
    """Derive a brainglobe-space origin from OME axis metadata.

    Parameters
    ----------
    ome_attrs : dict
        The ``attributes.ome`` block of a zarr group.

    Returns
    -------
    str
        Three-letter origin, e.g. ``"asl"``.

    Raises
    ------
    KeyError
        If an axis carries no recognised anatomical orientation.
    """
    axes = _intrinsic_axes(ome_attrs["multiscales"][0])[-3:]
    letters = []
    for axis in axes:
        direction = axis.get("orientation", {}).get("value")
        try:
            letters.append(_ORIGIN_BY_AXIS_DIRECTION[direction])
        except KeyError as e:
            raise KeyError(
                f"Axis {axis.get('name')!r} has unrecognised anatomical "
                f"orientation {direction!r}."
            ) from e
    return "".join(letters)


def _pyramid_level_from_attrs(
    ome_attrs: dict, resolution: float
) -> Tuple[int, str]:
    """Find the pyramid level matching a resolution in micrometres.

    Parameters
    ----------
    ome_attrs : dict
        The ``attributes.ome`` block of a zarr group.
    resolution : float
        Requested isotropic resolution, in micrometres.

    Returns
    -------
    tuple
        The ``(index, dataset_path)`` of the matching level.

    Raises
    ------
    ValueError
        If no level matches the requested resolution.
    """
    datasets = ome_attrs["multiscales"][0]["datasets"]
    for index, dataset in enumerate(datasets):
        scales = _scale_from_transforms(dataset["coordinateTransformations"])[
            -3:
        ]
        if all(np.isclose(resolution / 1000, scale) for scale in scales):
            return index, dataset["path"]

    raise ValueError(
        f"Requested resolution {resolution} um is invalid for this atlas."
    )


def _scale_from_transforms(transforms):
    """Return the scale factors from a coordinate transformation list.

    OME-Zarr 0.5 stores a flat ``[{scale}, {translation}]`` list, while 0.6
    wraps them as ``[{"type": "sequence", "transformations": [...]}]``. Both
    shapes are accepted, as raw dictionaries read from ``zarr.json``.

    Parameters
    ----------
    transforms : sequence
        Coordinate transformations for one multiscale dataset.

    Returns
    -------
    sequence of float
        The scale factors, one per axis.

    Raises
    ------
    ValueError
        If no scale transformation is present.
    """
    for transform in transforms:
        scale = transform.get("scale")
        if scale is not None:
            return scale
        for inner in transform.get("transformations", ()):
            scale = inner.get("scale")
            if scale is not None:
                return scale

    raise ValueError(
        "No scale transformation found in coordinateTransformations."
    )


def normalize_manifest(raw: dict, resolution, root_dir: Path) -> dict:
    """Return manifest metadata in the shape ``core.Atlas`` consumes.

    BrainGlobe manifests are returned unchanged. atlas-assets manifests
    are reshaped by ``_normalize_atlas_assets``.

    Parameters
    ----------
    raw : dict
        Parsed atlas ``manifest.json``.
    resolution : float or None
        Requested isotropic resolution, in micrometres. Defaults to the
        finest declared scale, with a printed note when there is more
        than one to choose from; if the manifest has a single fixed
        resolution instead (a BrainGlobe manifest), a mismatching value
        raises.
    root_dir : Path
        Local cache namespace directory holding the downloaded assets.

    Returns
    -------
    dict
        Metadata in BrainGlobe's shape.

    Raises
    ------
    ValueError
        If ``resolution`` does not match one of the declared scales.
    """
    if isinstance(raw.get("templates"), list):
        return _normalize_atlas_assets(raw, resolution, root_dir)

    if resolution is not None and not np.isclose(
        resolution, raw["resolution"][0]
    ):
        raise ValueError(
            f"Atlas {raw['name']} has a fixed resolution of "
            f"{raw['resolution'][0]} um; {resolution} um was "
            f"requested."
        )
    return raw


def _normalize_atlas_assets(raw: dict, resolution, root_dir: Path) -> dict:
    """Reshape an atlas-assets manifest into BrainGlobe's metadata shape.

    The first template and annotation set are promoted to singular keys,
    terminology is hoisted to the top level, and the fields the
    specification does not carry are derived from the zarr metadata and
    the asset name, or set to None. The result is tagged
    ``"format": "atlas-assets"`` so its meshes are dequantized.

    Parameters
    ----------
    raw : dict
        Parsed atlas ``manifest.json``.
    resolution : float or None
        Requested isotropic resolution, in micrometres, or None for the
        finest declared scale.
    root_dir : Path
        Local cache namespace directory holding the downloaded assets.

    Returns
    -------
    dict
        Metadata in BrainGlobe's shape.

    Raises
    ------
    ValueError
        If ``resolution`` does not match one of the declared scales.
    """
    annotation_set = raw["annotation_sets"][0]
    scales = annotation_set.get("scales", [])

    if resolution is None:
        resolution = min(scales)
        if len(scales) > 1:
            print(
                f"Atlas {raw['name']} provides resolutions {scales} um and "
                f"none was requested, so using the finest, {resolution} um. "
                "Pass resolution= to choose another."
            )
    if not any(np.isclose(resolution, scale) for scale in scales):
        raise ValueError(
            f"Resolution {resolution} um is not available for "
            f"{raw['name']}. Available scales: {scales}."
        )

    annotation_location = annotation_set["location"][1:]
    annotation_path = root_dir / annotation_location / V3_ANNOTATION_NAME
    annotation_group = zarr.open_group(str(annotation_path), mode="r")
    ome_attrs = annotation_group.attrs["ome"]
    _, dataset_path = _pyramid_level_from_attrs(ome_attrs, resolution)
    shape = annotation_group[dataset_path].shape

    return {
        "format": "atlas-assets",
        "name": raw["name"],
        "version": raw["version"],
        "location": raw["location"],
        "citation": None,
        "atlas_link": None,
        "species": _species_from_name(raw["name"]),
        "symmetric": True,
        "resolution": [float(resolution)] * 3,
        "orientation": _orientation_from_axes(ome_attrs),
        "shape": list(shape[-3:]),
        "additional_references": [],
        "coordinate_space": raw["coordinate_space"],
        "terminology": annotation_set["terminology"],
        "annotation_set": annotation_set,
        "template": raw["templates"][0],
    }


def get_annotation_set(manifest: dict) -> dict:
    """Return the annotation set entry from either manifest shape.

    Both shapes nest the annotation set's ``terminology`` and ``template``
    inside it, which is where ``core.Atlas`` reads them from.

    Parameters
    ----------
    manifest : dict
        Parsed atlas manifest, in either shape.

    Returns
    -------
    dict
        The annotation set entry.
    """
    annotation_sets = manifest.get("annotation_sets")
    if isinstance(annotation_sets, list):
        return annotation_sets[0]
    return manifest["annotation_set"]
