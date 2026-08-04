"""Defines the BrainGlobe Atlas API V3 classes and functions."""

import re
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
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
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_internet_connection,
    check_s3_status,
    get_latest_version,
    read_json,
)


def _version_tuple_from_str(version_str: str) -> Tuple[int, ...]:
    """Parse a version string into a numeric tuple for comparison.

    Accepts ``.``, ``_`` and ``-`` as separators, so both BrainGlobe
    (``3_0``) and atlas-assets (``2024-05``) version folder names parse.

    Parameters
    ----------
    version_str : str
        Version string, e.g. ``"3.0"``, ``"3_0"`` or ``"2024-05"``.

    Returns
    -------
    tuple of int
        Numeric components, in order.
    """
    return tuple(int(n) for n in re.split(r"[._-]", version_str))


def _version_str_from_tuple(version_tuple: Tuple[int, ...]) -> str:
    return "_".join(str(num) for num in version_tuple)


def _resolve_remote_root(fs, atlas_name: str, roots) -> Tuple[str, str]:
    """Find which configured root holds an atlas.

    Roots are searched in declaration order, so the first-declared root
    wins when a name exists in more than one bucket.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem used to test for the atlas directory.
    atlas_name : str
        Name of the atlas to locate.
    roots : dict
        Mapping of root key to remote root, in resolution order.

    Returns
    -------
    tuple of str
        The winning ``(root_key, remote_root)``.

    Raises
    ------
    ValueError
        If the atlas name is present in none of the roots.
    """
    for root_key, remote_root in roots.items():
        if fs.exists(f"{remote_root}/{V3_ATLAS_ROOTDIR}/{atlas_name}"):
            return root_key, remote_root

    searched = ", ".join(roots.values())
    raise ValueError(
        f"{atlas_name} is not a valid atlas name! "
        f"Searched remote roots: {searched}"
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
    axes = ome_attrs["multiscales"][0]["axes"]
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
    metadata and the asset name, or set to None.

    Parameters
    ----------
    raw : dict
        Parsed atlas ``manifest.json``.
    resolution : float or None
        Requested isotropic resolution, in micrometres. Required when
        the manifest declares ``scales``.
    root_dir : Path
        Local cache namespace directory holding the downloaded assets.

    Returns
    -------
    dict
        Metadata in BrainGlobe's shape.

    Raises
    ------
    ValueError
        If ``resolution`` is missing, or not among the declared scales.
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
        raise ValueError(
            f"Atlas {raw['name']} provides multiple resolutions. Pass "
            f"resolution= to choose one of: {scales}."
        )
    if not any(np.isclose(resolution, scale) for scale in scales):
        raise ValueError(
            f"Resolution {resolution} um is not available for "
            f"{raw['name']}. Available scales: {scales}."
        )

    annotation_location = annotation_set["location"][1:]
    annotation_path = root_dir / annotation_location / V3_ANNOTATION_NAME
    ome_attrs = zarr.open_group(str(annotation_path), mode="r").attrs["ome"]
    level, dataset_path = _pyramid_level_from_attrs(ome_attrs, resolution)
    shape = zarr.open_group(str(annotation_path), mode="r")[dataset_path].shape

    return {
        "name": raw["name"],
        "version": raw["version"],
        "location": raw["location"],
        "citation": None,
        "atlas_link": None,
        "species": _species_from_name(raw["name"]),
        "symmetric": False,
        "resolution": [float(resolution)] * 3,
        "orientation": _orientation_from_axes(ome_attrs),
        "shape": list(shape[-3:]),
        "additional_references": [],
        "coordinate_space": raw["coordinate_space"],
        "terminology": annotation_set["terminology"],
        "annotation_set": annotation_set,
        "template": raw["templates"][0],
    }


def _component(manifest: dict, key: str) -> dict:
    """Return a component entry from either manifest shape.

    Parameters
    ----------
    manifest : dict
        Parsed atlas manifest, in either shape.
    key : str
        Either ``"annotation_set"`` or ``"template"``.

    Returns
    -------
    dict
        The component entry.
    """
    plural = {"annotation_set": "annotation_sets", "template": "templates"}
    if isinstance(manifest.get(plural[key]), list):
        return manifest[plural[key]][0]
    return manifest[key]


class BrainGlobeAtlas(core.Atlas):
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
    resolution : float (optional)
        Requested isotropic resolution in micrometres. Required for atlases
        that declare multiple scales, ignored otherwise.
    """

    # Class-level fallback so a partially constructed instance (built via
    # ``object.__new__``, as some existing tests do) still has a usable
    # remote root before ``__init__`` sets the instance attribute.
    _remote_root = descriptors.DEFAULT_REMOTE_ROOT
    # Same rationale: resolved folder-name strings, set as a side effect
    # of resolving ``remote_version``/``local_full_name``.
    _remote_version_str: Optional[str] = None
    _local_version_str: Optional[str] = None

    def __init__(
        self,
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
        resolution: Optional[float] = None,
    ):
        self._resolution = resolution
        self._remote_version = None
        self._remote_version_str = None
        self._local_version_str = None
        self._local_full_name = None
        self._requested_version = (
            version.replace(".", "_") if version else None
        )
        self._local_version = (
            _version_tuple_from_str(version.replace("_", "."))
            if version
            else None
        )
        self.fs = s3fs.S3FileSystem(anon=True)

        self.atlas_name = atlas_name
        self.fn_update = fn_update

        # Read BrainGlobe configuration file:
        conf = config.read_config(config_dir)

        # Assume the historical default root so an atlas that is already
        # cached locally (or was written directly to disk, as atlas
        # generation/validation does) is found without any network call.
        self._root_key = descriptors.DEFAULT_ROOT_KEY
        self._remote_root = descriptors.DEFAULT_REMOTE_ROOT

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        if brainglobe_dir is None:
            self.brainglobe_dir = Path(conf["default_dirs"]["brainglobe_dir"])
        else:
            self.brainglobe_dir = Path(brainglobe_dir)

        self.brainglobe_dir = self.brainglobe_dir / self._root_key

        self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

        if self.local_full_name is None:
            # Not cached under the default root: find out which configured
            # remote root actually holds this atlas before downloading, so
            # it lands in (and is later found in) the matching per-root
            # cache directory. Offline, keep the default root so an
            # already-cached atlas under it still loads.
            roots = config.get_remote_roots(config_dir)
            if check_s3_status(raise_error=False):
                try:
                    root_key, remote_root = _resolve_remote_root(
                        self.fs, atlas_name, roots
                    )
                except ValueError as error:
                    # Preserve the historical exception type for an
                    # invalid atlas name: BrainGlobeAtlas has always
                    # raised FileNotFoundError here, and callers (e.g.
                    # update_atlas) depend on that.
                    raise FileNotFoundError(str(error)) from error

                if root_key != self._root_key:
                    self._root_key, self._remote_root = (
                        root_key,
                        remote_root,
                    )
                    self.brainglobe_dir = (
                        self.brainglobe_dir.parent / self._root_key
                    )
                    self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

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

        manifest_path = self.brainglobe_dir / self.local_full_name
        metadata = _normalize_manifest(
            read_json(manifest_path), self._resolution, self.brainglobe_dir
        )
        super().__init__(
            manifest_path,
            metadata=metadata,
            remote_root=self._remote_root,
        )

        if check_latest:
            self.check_latest_version()

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
                rf"\d+(?:[_-]\d+)?/manifest.json"
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

        version_str = self.metadata["version"]
        if self._local_version_str is None:
            self._local_version_str = version_str
        self._local_version = _version_tuple_from_str(version_str)

        return self._local_version

    @property
    def remote_version(self) -> Optional[tuple[int, ...]]:
        """Reads remote version from s3 bucket.

        Largest numerical version assumed to be latest.
        If we are offline, return None.
        """
        if self._remote_version is not None:
            return self._remote_version

        if not check_s3_status(
            raise_error=False, remote_root=self._remote_root
        ):
            return None

        bucket_path = (
            f"{self._remote_root}/{V3_ATLAS_ROOTDIR}/{self.atlas_name}"
        )

        if self._requested_version is None:
            versions_path = self.fs.ls(bucket_path)
            if not versions_path:
                # Listing a nonexistent S3 prefix returns [] rather than
                # raising. This happens when resolution was skipped (atlas
                # found in the local cache) and the assumed root turns out
                # not to host this atlas name.
                raise FileNotFoundError(
                    f"{self.atlas_name} is not a valid atlas name!"
                )
            available_versions: List[str] = [
                path_str.split("/")[-1] for path_str in versions_path
            ]
            latest_version = get_latest_version(available_versions)
            self._remote_version_str = latest_version
            self._remote_version = _version_tuple_from_str(latest_version)
        else:
            requested_path = f"{bucket_path}/{self._requested_version}"
            if not self.fs.exists(requested_path):
                raise FileNotFoundError(
                    f"Requested version {self._requested_version} for atlas "
                    f"{self.atlas_name} not found in remote."
                )

            self._remote_version_str = self._requested_version
            self._remote_version = _version_tuple_from_str(
                self._requested_version
            )

        return self._remote_version

    def download(self):
        """Download and extract the atlas files from remote storage.

        The manifest file is removed if any error occurs during the
        download to ensure that incomplete downloads are retried.
        """
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
            f"v{remote_version_str} manifest:"
        )
        self.fs.get(remote_path, local_path, callback=TqdmCallback())
        self.metadata = read_json(local_path)

        try:
            # Download terminology file
            terminology = _component(self.metadata, "annotation_set")[
                "terminology"
            ]
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
            annotation_set = _component(self.metadata, "annotation_set")
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
                        f"v{remote_version_str}."
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
            template = _component(self.metadata, "template")
            template_location = template["location"][1:]
            local_template_path = self.brainglobe_dir / template_location
            if not local_template_path.exists():
                root_metadata_path = (
                    template_location + f"/{V3_TEMPLATE_NAME}/**/*.json"
                )
                remote_root_metadata_path = (
                    f"{self._remote_root}/{root_metadata_path}"
                )

                print(
                    f"Downloading template metadata "
                    f"for {template['name']}:"
                )
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
                    print(
                        f"Downloading template metadata " f"for {ref['name']}:"
                    )
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

        # Resolve the local folder name before comparing.
        local_version = self.local_version
        local = self._local_version_str
        if local is None:
            local = _version_str_from_tuple(local_version)
        online = self._remote_version_str
        if online is None:
            online = _version_str_from_tuple(remote_version)

        if local != online:
            if print_warning:
                rprint(
                    "[b][magenta2]brainglobe_atlasapi[/b]: "
                    f"[b]{self.atlas_name}[/b] version "
                    f"[b]{local}[/b] "
                    f"is not the latest available "
                    f"([b]{online}[/b]). "
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
