"""Defines the BrainGlobe Atlas API V3 classes and functions."""

from __future__ import annotations

import re
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import (
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
    overload,
)

import dask.array as da
import numpy as np
import numpy.typing as npt
import s3fs
import zarr
from fsspec.callbacks import TqdmCallback
from rich import print as rprint
from rich.console import Console

from brainglobe_atlasapi import config, core, descriptors
from brainglobe_atlasapi.atlas_name import AtlasName
from brainglobe_atlasapi.descriptors import (
    V3_ANNOTATION_MAP_NAME,
    V3_ANNOTATION_MASKS_NAME,
    V3_ANNOTATION_NAME,
    V3_ATLAS_ROOTDIR,
    V3_HEMISPHERES_NAME,
    V3_MESHES_DIRECTORY,
    V3_TEMPLATE_NAME,
)
from brainglobe_atlasapi.list_atlases import (
    folder_version_to_dotted,
    get_all_atlases_lastversions,
)
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_internet_connection,
    check_s3_status,
    get_latest_version,
    read_json,
    version_tuple_from_str,
)

# Version folder names: BrainGlobe's "3_0"/"2017", atlas-assets' "2024-05".
_VERSION_FOLDER_PATTERN = r"\d+(?:[_-]\d+)?"


def _cache_key(remote_root: str) -> str:
    """Return the local cache directory name for a remote root.

    The BrainGlobe root keeps its historical directory; every other root
    is cached under the last segment of its URL, e.g.
    ``s3://bucket/prefix/allen-atlas-assets-rc11`` under
    ``allen-atlas-assets-rc11``.

    Parameters
    ----------
    remote_root : str
        Remote root, without a trailing slash.

    Returns
    -------
    str
        Directory name, relative to the brainglobe directory.

    Raises
    ------
    ValueError
        If a non-default root would take the default root's directory.
    """
    if remote_root == descriptors.DEFAULT_REMOTE_ROOT:
        return descriptors.DEFAULT_ROOT_KEY
    key = remote_root.removeprefix("s3://").rsplit("/", 1)[-1]
    if key == descriptors.DEFAULT_ROOT_KEY:
        raise ValueError(
            f"{remote_root} would share the cache directory reserved "
            f"for {descriptors.DEFAULT_REMOTE_ROOT}."
        )
    return key


def _check_remote_root(cache_dir: Path, remote_root: str) -> None:
    """Check a cache directory does not mirror a different remote root.

    Two roots sharing a last URL segment map to the same directory; the
    record written by ``_record_remote_root`` keeps them from mixing.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote_root``.
    remote_root : str
        Remote root, without a trailing slash.

    Raises
    ------
    ValueError
        If the directory already records a different remote root.
    """
    record = cache_dir / descriptors.REMOTE_ROOT_FILENAME
    if record.exists():
        recorded = record.read_text().strip()
        if recorded != remote_root:
            raise ValueError(
                f"{cache_dir} holds atlases from {recorded}, not "
                f"{remote_root}. Remote roots sharing a last URL segment "
                "cannot share a cache directory."
            )


def _record_remote_root(cache_dir: Path, remote_root: str) -> None:
    """Record which remote root a cache directory mirrors, if unrecorded.

    Called only once the directory holds an atlas, so a failed lookup
    (a typo'd root or atlas name) never claims the directory.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote_root``.
    remote_root : str
        Remote root, without a trailing slash.
    """
    record = cache_dir / descriptors.REMOTE_ROOT_FILENAME
    if record.exists():
        return
    try:
        record.write_text(f"{remote_root}\n")
    except OSError:
        # A read-only shared cache still loads; it just goes unrecorded.
        pass


def _resolve_remote_root(
    fs, atlas_name: str, roots, is_reachable: Callable[[str], bool]
) -> Optional[str]:
    """Find which configured root holds an atlas.

    Roots are searched in declaration order, so the first-declared root
    wins when a name exists in more than one bucket. Each root is probed
    for reachability only when its turn comes, and unreachable roots are
    skipped, so finding an atlas on the first root costs two requests.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem used to list the atlas directory.
    atlas_name : str
        Name of the atlas to locate.
    roots : dict
        Mapping of root key to remote root, in resolution order.
    is_reachable : callable
        Returns whether a remote root answers.

    Returns
    -------
    str or None
        The winning remote root, or None if no root is reachable.

    Raises
    ------
    ValueError
        If the atlas name is present in none of the reachable roots.
    """
    searched = []
    for remote_root in roots.values():
        if not is_reachable(remote_root):
            continue
        searched.append(remote_root)
        try:
            if fs.ls(f"{remote_root}/{V3_ATLAS_ROOTDIR}/{atlas_name}"):
                return remote_root
        except FileNotFoundError:
            pass

    if not searched:
        return None

    raise ValueError(
        f"{atlas_name} is not a valid atlas name! "
        f"Searched remote roots: {', '.join(searched)}"
    )


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
        scales = core._scale_from_transforms(
            dataset["coordinateTransformations"]
        )[-3:]
        if all(np.isclose(resolution / 1000, scale) for scale in scales):
            return index, dataset["path"]

    raise ValueError(
        f"Requested resolution {resolution} um is invalid for this atlas."
    )


def _normalize_manifest(raw: dict, resolution, root_dir: Path) -> dict:
    """Return manifest metadata in the shape ``core.Atlas`` consumes.

    BrainGlobe manifests are returned unchanged. atlas-assets manifests
    are reshaped: the first template and annotation set are promoted to
    singular keys, terminology is hoisted to the top level, and the
    fields the specification does not carry are derived from the zarr
    metadata and the asset name, or set to None. They are also tagged
    ``"format": "atlas-assets"`` so their meshes are dequantized.

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
    if not isinstance(raw.get("templates"), list):
        if resolution is not None and not np.isclose(
            resolution, raw["resolution"][0]
        ):
            raise ValueError(
                f"Atlas {raw['name']} has a fixed resolution of "
                f"{raw['resolution'][0]} um; {resolution} um was "
                f"requested."
            )
        return raw

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


def _annotation_set(manifest: dict) -> dict:
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


class BrainGlobeAtlas(
    core.Atlas[core.AtlasArray],
    Generic[core.AtlasArray],
):
    """Add remote atlas fetching and version comparison functionalities
    to the core Atlas class.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to be used.
    version : str (optional)
        Desired version of the atlas. If None, the latest version will be used.
    brainglobe_dir : str or Path object
        Default folder for brainglobe downloads.
    check_latest : bool (optional)
        If true, check if we have the most recent atlas (default=True). Set
        this to False to avoid waiting for remote server response on atlas
        instantiation and to suppress warnings.
    fn_update : Callable
        Handler function to update during download. Takes completed and total
        bytes.
    dask : bool
        If True, atlas array properties return dask arrays instead of loading
        them into memory as numpy arrays.
    resolution : float (optional)
        Requested isotropic resolution in micrometres. Defaults to the
        finest scale the atlas declares, printing a note when it declares
        more than one. For atlases with a single fixed resolution, a value
        that disagrees with it raises.
    remote_root : str (optional)
        An ``s3://bucket/prefix`` to fetch the atlas from. When given, it
        takes full priority: it is the only root searched, with no
        fallback to the configured BrainGlobe/Allen defaults if the atlas
        isn't found there. Its local cache subdirectory is named after
        the last segment of the URL (``brainglobe-atlasapi`` for the
        BrainGlobe root). If None (default), the historical BrainGlobe root is
        assumed, falling back to the configured remote roots in
        declaration order.
    """

    @overload
    def __init__(
        self: BrainGlobeAtlas[npt.NDArray[np.integer]],
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
        dask: Literal[False] = False,
        resolution: Optional[float] = None,
        remote_root: Optional[str] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: BrainGlobeAtlas[da.Array],
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
        dask: Literal[True] = True,
        resolution: Optional[float] = None,
        remote_root: Optional[str] = None,
    ) -> None: ...

    @overload
    def __init__(
        self: BrainGlobeAtlas[core.AtlasArray],
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
        dask: bool = False,
        resolution: Optional[float] = None,
        remote_root: Optional[str] = None,
    ) -> None: ...

    def __init__(
        self,
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
        dask: bool = False,
        resolution: Optional[float] = None,
        remote_root: Optional[str] = None,
    ):
        self._resolution = resolution
        self._remote_version = None
        self._remote_version_str = None
        self._local_version_str = None
        self._local_full_name = None
        self._reachable_roots: set = set()
        self._requested_version = (
            version.replace(".", "_") if version else None
        )
        self._local_version = (
            version_tuple_from_str(version.replace("_", "."))
            if version
            else None
        )
        self.fs = s3fs.S3FileSystem(anon=True)

        self.atlas_name = atlas_name
        self.fn_update = fn_update

        # Read BrainGlobe configuration file:
        conf = config.read_config(config_dir)

        if remote_root is not None:
            remote_root = remote_root.rstrip("/")
        self._remote_root = remote_root or descriptors.DEFAULT_REMOTE_ROOT

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        if brainglobe_dir is None:
            base_dir = Path(conf["default_dirs"]["brainglobe_dir"])
        else:
            base_dir = Path(brainglobe_dir)

        self.brainglobe_dir = base_dir / _cache_key(self._remote_root)

        self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

        if remote_root is None and self.local_full_name is None:
            self._remote_root = self._locate_remote_root(
                base_dir, config.get_remote_roots(config_dir)
            )
            self.brainglobe_dir = base_dir / _cache_key(self._remote_root)
            self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

        _check_remote_root(self.brainglobe_dir, self._remote_root)

        # Look for this atlas in local brainglobe folder:
        if self.local_full_name is None:
            if self.remote_version is None:
                check_internet_connection(raise_error=True)

                # If internet is up, then the atlas name was invalid
                raise ValueError(f"{atlas_name} is not a valid atlas name!")
            else:
                self.download()
                assert self.local_full_name is not None, (
                    "Download failed: local atlas manifest not found after "
                    "download."
                )

        _record_remote_root(self.brainglobe_dir, self._remote_root)

        manifest_path = self.brainglobe_dir / self.local_full_name
        metadata = _normalize_manifest(
            read_json(manifest_path), self._resolution, self.brainglobe_dir
        )
        super().__init__(
            manifest_path,
            metadata=metadata,
            remote_root=self._remote_root,
            dask=dask,
        )

        if check_latest:
            self.check_latest_version()

    def _locate_remote_root(self, base_dir: Path, roots: dict) -> str:
        """Find the remote root for an atlas not cached under the default.

        The local cache of each configured root is searched first, in
        config order, then the cache of every other root recorded under
        ``base_dir``, so a cached copy loads offline and without S3 calls
        even if its root is not configured. Then each reachable configured
        root is searched on S3, so the download lands in (and is later
        found in) the matching cache.

        Parameters
        ----------
        base_dir : Path
            Directory holding one cache directory per remote root.
        roots : dict
            Mapping of root key to remote root, in resolution order.

        Returns
        -------
        str
            The remote root holding the atlas. When no root is reachable,
            the default root, so the name check in ``__init__`` fails
            loudly.

        Raises
        ------
        FileNotFoundError
            If no reachable root holds the atlas.
        """
        records = sorted(
            base_dir.glob(f"*/{descriptors.REMOTE_ROOT_FILENAME}")
        )
        recorded_roots = [record.read_text().strip() for record in records]
        # dict.fromkeys drops duplicates while keeping config order first.
        for remote_root in dict.fromkeys([*roots.values(), *recorded_roots]):
            self.brainglobe_dir = base_dir / _cache_key(remote_root)
            atlas_dir = (
                self.brainglobe_dir / V3_ATLAS_ROOTDIR / self.atlas_name
            )
            # local_full_name caches its result, so clear it per directory.
            self._local_full_name = None
            if atlas_dir.is_dir() and self.local_full_name is not None:
                return remote_root

        try:
            resolved = _resolve_remote_root(
                self.fs, self.atlas_name, roots, self._is_reachable
            )
        except ValueError as error:
            # Preserve the historical exception type for an invalid atlas
            # name: callers such as update_atlas depend on it.
            raise FileNotFoundError(str(error)) from error

        return resolved or descriptors.DEFAULT_REMOTE_ROOT

    def _is_reachable(self, remote_root: str) -> bool:
        """Check a remote root answers, probing each root at most once.

        Only success is remembered, so a later check can still see a
        connection that has come back.

        Parameters
        ----------
        remote_root : str
            Remote root to check.

        Returns
        -------
        bool
            Whether the remote root is reachable.
        """
        if remote_root not in self._reachable_roots and check_s3_status(
            raise_error=False, remote_root=remote_root
        ):
            self._reachable_roots.add(remote_root)
        return remote_root in self._reachable_roots

    @property
    def local_full_name(self):
        """
        Returns the local full path to the manifest.json file of the atlas.

        This will return either the path to the requested version if it is
        found locally, or the latest version found locally.

        If not found, returns None.
        """
        if self._local_full_name is not None:
            return self._local_full_name

        (self.brainglobe_dir / V3_ATLAS_ROOTDIR).mkdir(
            parents=True, exist_ok=True
        )

        if self._requested_version is not None:
            pattern = (
                f"{V3_ATLAS_ROOTDIR}/{self.atlas_name}/"
                f"{self._requested_version}/manifest.json"
            )
        else:
            pattern = (
                rf"{V3_ATLAS_ROOTDIR}/{self.atlas_name}/"
                rf"{_VERSION_FOLDER_PATTERN}/manifest.json"
            )

        glob_pattern = f"{V3_ATLAS_ROOTDIR}/{self.atlas_name}/*/manifest.json"

        available_versions: List[str] = [
            p.parent.name
            for p in self.brainglobe_dir.glob(glob_pattern)
            if re.search(pattern, p.as_posix())
        ]

        if len(available_versions) == 0:
            return None

        latest_version = get_latest_version(available_versions)
        self._local_version_str = latest_version

        self._local_full_name = (
            f"{V3_ATLAS_ROOTDIR}/"
            f"{self.atlas_name}/"
            f"{latest_version}/"
            f"manifest.json"
        )

        return self._local_full_name

    @property
    def local_version(self) -> Optional[Tuple[int, ...]]:
        """If atlas is local, return actual version of the downloaded files."""
        if self._local_version is not None:
            return self._local_version

        self._local_version = version_tuple_from_str(self.metadata["version"])

        return self._local_version

    @property
    def remote_version(self) -> Optional[tuple[int, ...]]:
        """Reads remote version from s3 bucket.

        Unless a version was requested, the latest version is the one listed
        in the remote last_versions.conf.
        If we are offline or using a custom atlas, return None.
        """
        if self._remote_version is not None:
            return self._remote_version

        if not self._is_reachable(self._remote_root):
            return None

        bucket_path = (
            f"{self._remote_root}/{V3_ATLAS_ROOTDIR}/{self.atlas_name}"
        )

        try:
            # One request when the atlas is there, where exists() costs
            # two, and served from the s3fs cache when root resolution has
            # just listed this path.
            versions_path = self.fs.ls(bucket_path)
        except FileNotFoundError:
            if self.local_full_name is not None:
                return None
            raise FileNotFoundError(
                f"{self.atlas_name} is not a valid atlas name!"
            ) from None

        if self._requested_version is None:
            # last_versions.conf only lists atlases on the BrainGlobe bucket;
            # other roots take their newest version folder.
            if self._remote_root == descriptors.DEFAULT_REMOTE_ROOT:
                last_versions = get_all_atlases_lastversions()
                if self.atlas_name not in last_versions:
                    if self.local_full_name is not None:
                        return None
                    raise FileNotFoundError(
                        f"{self.atlas_name} has no released version "
                        "available. Specify a version explicitly to "
                        "download it."
                    )
                self._remote_version_str = str(
                    last_versions[self.atlas_name]
                ).replace(".", "_")
                self._remote_version = version_tuple_from_str(
                    self._remote_version_str
                )
            else:
                # Skip anything that isn't a version folder, e.g. a README.
                available_versions: List[str] = [
                    name
                    for name in (p.split("/")[-1] for p in versions_path)
                    if re.fullmatch(_VERSION_FOLDER_PATTERN, name)
                ]
                if not available_versions:
                    # Without this, get_latest_version would fail with a
                    # bare IndexError.
                    raise FileNotFoundError(
                        f"{self.atlas_name} is not a valid atlas name!"
                    )
                latest_version = get_latest_version(available_versions)
                self._remote_version_str = latest_version
                self._remote_version = version_tuple_from_str(latest_version)
        else:
            requested_path = f"{bucket_path}/{self._requested_version}"
            if not self.fs.exists(requested_path):
                raise FileNotFoundError(
                    "Requested version "
                    f"{folder_version_to_dotted(self._requested_version)} "
                    f"for atlas {self.atlas_name} not found in remote."
                )

            self._remote_version_str = self._requested_version
            self._remote_version = version_tuple_from_str(
                self._requested_version
            )

        return self._remote_version

    def download(self):
        """Download and extract the atlas files from remote storage.

        The manifest file is removed if any error occurs during the
        download to ensure that incomplete downloads are retried.
        """
        if not self._is_reachable(self._remote_root):
            check_s3_status(remote_root=self._remote_root)

        # Ensure remote_version has been resolved so the folder name is set.
        _ = self.remote_version
        remote_version_str = self._remote_version_str
        key_name = (
            f"{V3_ATLAS_ROOTDIR}/{self.atlas_name}/"
            f"{remote_version_str}/manifest.json"
        )

        local_path = self.brainglobe_dir / key_name
        remote_path = f"{self._remote_root}/{key_name}"

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {self.atlas_name} atlas "
            f"v{folder_version_to_dotted(remote_version_str)} manifest:"
        )
        self.fs.get(remote_path, local_path, callback=TqdmCallback())
        self.metadata = read_json(local_path)

        try:
            annotation_set = _annotation_set(self.metadata)

            # Download terminology file
            terminology = annotation_set["terminology"]
            terminology_location = terminology["location"][1:]
            local_terminology_path = self.brainglobe_dir / terminology_location
            if not local_terminology_path.exists():
                remote_terminology_path = (
                    f"{self._remote_root}/{terminology_location}"
                )
                print(
                    f"Downloading terminology metadata "
                    f"for {terminology['name']}:"
                )
                self.fs.get(
                    remote_terminology_path,
                    local_terminology_path,
                    recursive=True,
                    callback=TqdmCallback(),
                )

            # Download coordinate space files
            coordspace_location = self.metadata["coordinate_space"][
                "location"
            ][1:]
            local_coordspace_path = self.brainglobe_dir / coordspace_location
            if not local_coordspace_path.exists():
                remote_coordspace_path = (
                    f"{self._remote_root}/{coordspace_location}"
                )
                print(
                    f"Downloading coordinate space metadata "
                    f"for {self.metadata['coordinate_space']['name']}:"
                )
                self.fs.get(
                    remote_coordspace_path,
                    local_coordspace_path,
                    recursive=True,
                    callback=TqdmCallback(),
                )

            # Download annotation metadata files
            annotation_location = annotation_set["location"][1:]
            local_annotation_path = self.brainglobe_dir / annotation_location
            if not local_annotation_path.exists():
                root_metadata_path = (
                    annotation_location + f"/{V3_ANNOTATION_NAME}/**/*.json"
                )
                remote_root_metadata_path = (
                    f"{self._remote_root}/{root_metadata_path}"
                )
                print(
                    f"Downloading annotation metadata "
                    f"for {annotation_set['name']}:"
                )
                self.fs.get(
                    remote_root_metadata_path,
                    local_annotation_path / V3_ANNOTATION_NAME,
                    callback=TqdmCallback(),
                )
                mesh_path = local_annotation_path / V3_MESHES_DIRECTORY
                mesh_path.mkdir(parents=True, exist_ok=True)

                # Download 4D masks metadata (JSON only; chunk data is lazy)
                try:
                    masks_metadata_glob = (
                        annotation_location
                        + f"/{V3_ANNOTATION_MASKS_NAME}/**/*.json"
                    )
                    remote_masks_metadata = (
                        f"{self._remote_root}/{masks_metadata_glob}"
                    )
                    self.fs.get(
                        remote_masks_metadata,
                        str(local_annotation_path / V3_ANNOTATION_MASKS_NAME),
                        callback=TqdmCallback(),
                    )
                    masks_annotation_values_path = (
                        annotation_location + f"/{V3_ANNOTATION_MASKS_NAME}"
                        f"/{V3_ANNOTATION_MAP_NAME}"
                    )
                    remote_masks_annotation_values_path = (
                        f"{self._remote_root}/{masks_annotation_values_path}"
                    )
                    self.fs.get(
                        remote_masks_annotation_values_path,
                        str(local_annotation_path / V3_ANNOTATION_MASKS_NAME),
                        callback=TqdmCallback(),
                        recursive=True,
                    )
                except FileNotFoundError as e:
                    raise FileNotFoundError(
                        f"Annotation masks metadata not found for atlas "
                        f"{self.atlas_name} "
                        f"v{folder_version_to_dotted(remote_version_str)}."
                    ) from e

                hemispheres_remote = (
                    f"{self._remote_root}/{annotation_location}"
                    f"/{V3_HEMISPHERES_NAME}"
                )
                if not self.metadata.get(
                    "symmetric", False
                ) and self.fs.exists(hemispheres_remote):
                    root_hemisphere_path = (
                        annotation_location
                        + f"/{V3_HEMISPHERES_NAME}/**/*.json"
                    )
                    remote_root_hemisphere_path = (
                        f"{self._remote_root}/{root_hemisphere_path}"
                    )
                    self.fs.get(
                        remote_root_hemisphere_path,
                        local_annotation_path / V3_HEMISPHERES_NAME,
                        callback=TqdmCallback(),
                    )

            # Download template metadata files
            template = annotation_set["template"]
            template_location = template["location"][1:]
            local_template_path = self.brainglobe_dir / template_location
            if not local_template_path.exists():
                root_metadata_path = (
                    template_location + f"/{V3_TEMPLATE_NAME}/**/*.json"
                )
                remote_root_metadata_path = (
                    f"{self._remote_root}/{root_metadata_path}"
                )

                print(f"Downloading template metadata for {template['name']}:")
                self.fs.get(
                    remote_root_metadata_path,
                    local_template_path / V3_TEMPLATE_NAME,
                    callback=TqdmCallback(),
                )

            additional_reference_names = self.metadata.get(
                "additional_references", []
            )

            for ref in additional_reference_names:
                template_location = ref["location"][1:]
                local_template_path = self.brainglobe_dir / template_location

                if not local_template_path.exists():
                    root_metadata_path = (
                        template_location + f"/{V3_TEMPLATE_NAME}/**/*.json"
                    )
                    remote_root_metadata_path = (
                        f"{self._remote_root}/{root_metadata_path}"
                    )
                    print(f"Downloading template metadata for {ref['name']}:")
                    self.fs.get(
                        remote_root_metadata_path,
                        local_template_path / V3_TEMPLATE_NAME,
                        callback=TqdmCallback(),
                    )
            # Reset local_full_name to ensure it is updated with new location
            self._local_full_name = None

        except Exception:
            # Remove the manifest so the next run detects the incomplete
            # download and retries rather than finding partial files.
            local_path.unlink(missing_ok=True)
            raise

    def check_latest_version(
        self, print_warning: bool = True
    ) -> Optional[bool]:
        """
        Check if the local version is the latest available
        and prompts the user to update if not.

        Parameters
        ----------
        print_warning : bool, optional
            If True, prints a message if the local version is not the latest,
            by default True. Useful to turn off, e.g. when the user is updating
            the atlas

        Returns
        -------
        Optional[bool]
            Returns False if the local version is not the latest,
            True if it is, and None if we are offline.
        """
        # Cache remote version to avoid multiple requests
        remote_version = self.remote_version
        # If we are offline, return None
        if remote_version is None:
            return None

        # Compare version folder names exactly, so "2024-05" != "2024_05".
        # local_full_name and remote_version have set both by now.
        local = self._local_version_str
        online = self._remote_version_str

        if local != online:
            if print_warning:
                rprint(
                    "[b][magenta2]brainglobe_atlasapi[/b]: "
                    f"[b]{self.atlas_name}[/b] version "
                    f"[b]{folder_version_to_dotted(local)}[/b] "
                    f"is not the latest available "
                    f"([b]{folder_version_to_dotted(online)}[/b]). "
                    "To update the atlas run in the terminal:[/magenta2]\n"
                    f" [gold1]brainglobe update -a {self.atlas_name}[/gold1]"
                )
            return False
        return True

    def __repr__(self) -> str:
        """Fancy print providing atlas information."""
        if "_" in self.atlas_name:
            name_split = self.atlas_name.split("_")
            res = f" (res. {name_split.pop()})"
            return f"{' '.join(name_split)} atlas{res}"

        resolution = self.metadata["resolution"][0]
        return f"{self.atlas_name} (res. {resolution}um)"

    def __str__(self) -> str:
        """
        If the atlas metadata are to be printed
        with the built-in print function instead of rich's, then
        print the rich panel as a string.

        It will miss the colors.

        """
        buf = StringIO()
        _console = Console(file=buf, force_jupyter=False)
        _console.print(self)

        return buf.getvalue()

    def __rich_console__(self, *args):
        """
        Use rich API's console protocol.
        Prints the atlas metadata as a table nested in a panel.
        """
        panel = _rich_atlas_metadata(self.atlas_name, self.metadata)
        yield panel
