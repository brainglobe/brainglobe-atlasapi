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
from brainglobe_atlasapi.manifest import get_annotation_set
from brainglobe_atlasapi.remotes import (
    cache_key,
    check_remote,
    known_remotes,
    record_remote,
    resolve_remote,
)
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_s3_status,
    get_latest_version,
    read_json,
    version_tuple_from_str,
)

# Version folder names: BrainGlobe's "3_0"/"2017", atlas-assets' "2024-05".
_VERSION_FOLDER_PATTERN = r"\d+(?:[_-]\d+)?"


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
        Requested resolution in micrometres: the first axis of a
        non-isotropic atlas, e.g. 16 for 16 x 16 x 20 um. Defaults to the
        finest available, printing a note when there is more than one.
        For atlases with a single fixed resolution, a value that disagrees
        with it raises.
    remote : str (optional)
        An ``s3://bucket/prefix`` to fetch the atlas from. When given, it
        takes full priority: it is the only remote searched, with no
        fallback to the configured BrainGlobe/Allen defaults if the atlas
        isn't found there. Its local cache subdirectory is named after
        the last segment of the URL (``brainglobe-atlasapi`` for the
        BrainGlobe remote). If None (default), the historical BrainGlobe
        remote is assumed, falling back to the configured remotes in
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
        remote: Optional[str] = None,
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
        remote: Optional[str] = None,
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
        remote: Optional[str] = None,
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
        remote: Optional[str] = None,
    ):
        self._resolution = resolution
        self._remote_version_str = None
        self._local_version_str = None
        self._local_full_name = None
        self._reachable_remotes: set = set()
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

        if remote is not None:
            remote = remote.rstrip("/")
        self._remote = remote or descriptors.DEFAULT_REMOTE

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        if brainglobe_dir is None:
            base_dir = Path(conf["default_dirs"]["brainglobe_dir"])
        else:
            base_dir = Path(brainglobe_dir)

        self.brainglobe_dir = base_dir / cache_key(self._remote)

        self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

        if remote is None and self.local_full_name is None:
            self._remote = self._locate_remote(
                base_dir, config.get_remotes(config_dir)
            )
            self.brainglobe_dir = base_dir / cache_key(self._remote)
            self.brainglobe_dir.mkdir(parents=True, exist_ok=True)

        check_remote(self.brainglobe_dir, self._remote)

        # Look for this atlas in local brainglobe folder:
        if self.local_full_name is None:
            self.download()
            assert self.local_full_name is not None, (
                "Download failed: local atlas manifest not found after "
                "download."
            )

        record_remote(self.brainglobe_dir, self._remote)

        manifest_path = self.brainglobe_dir / self.local_full_name
        super().__init__(
            manifest_path,
            resolution=self._resolution,
            remote=self._remote,
            dask=dask,
        )

        if check_latest:
            self.check_latest_version()

    def _locate_remote(self, base_dir: Path, remotes: dict) -> str:
        """Find the remote for an atlas not cached under the default.

        The local cache of each configured remote is searched first, in
        config order, then the cache of every other remote recorded under
        ``base_dir``, so a cached copy loads offline and without S3 calls
        even if its remote is not configured. Then each reachable configured
        remote is searched on S3, so the download lands in (and is later
        found in) the matching cache.

        Parameters
        ----------
        base_dir : Path
            Directory holding one cache directory per remote.
        remotes : dict
            Mapping of remote key to remote, in resolution order.

        Returns
        -------
        str
            The remote holding the atlas. When no remote is reachable,
            the default remote, so the name check in ``__init__`` fails
            loudly.

        Raises
        ------
        FileNotFoundError
            If no reachable remote holds the atlas.
        """
        for remote in known_remotes(base_dir, remotes):
            self.brainglobe_dir = base_dir / cache_key(remote)
            atlas_dir = (
                self.brainglobe_dir / V3_ATLAS_ROOTDIR / self.atlas_name
            )
            # local_full_name caches its result, so clear it per directory.
            self._local_full_name = None
            if atlas_dir.is_dir() and self.local_full_name is not None:
                return remote

        try:
            resolved = resolve_remote(
                self.fs, self.atlas_name, remotes, self._is_reachable
            )
        except ValueError as error:
            # Preserve the historical exception type for an invalid atlas
            # name: callers such as update_atlas depend on it.
            raise FileNotFoundError(str(error)) from error

        return resolved or descriptors.DEFAULT_REMOTE

    def _is_reachable(self, remote: str) -> bool:
        """Check a remote answers, probing each remote at most once.

        Only success is remembered, so a later check can still see a
        connection that has come back.

        Parameters
        ----------
        remote : str
            Remote to check.

        Returns
        -------
        bool
            Whether the remote is reachable.
        """
        if remote not in self._reachable_remotes and check_s3_status(
            raise_error=False, remote=remote
        ):
            self._reachable_remotes.add(remote)
        return remote in self._reachable_remotes

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
        """Remote version of the atlas, or None if offline or not on remote.

        Unless a version was requested, the latest version is the one listed
        in the remote last_versions.conf on the BrainGlobe bucket, or the
        newest version folder on any other remote.
        """
        if self._remote_version_str is None:
            if not self._is_reachable(self._remote):
                return None
            try:
                self._resolve_remote_version()
            except FileNotFoundError:
                return None
        return version_tuple_from_str(self._remote_version_str)

    def _resolve_remote_version(self) -> str:
        """Find and cache the remote version folder name.

        Raises
        ------
        FileNotFoundError
            If the remote has no matching version of this atlas.
        """
        if self._remote_version_str is None:
            self._remote_version_str = self._pick_remote_version(
                self._remote_folders()
            )
        return self._remote_version_str

    def _remote_folders(self) -> List[str]:
        """List the names of the atlas's folders on the remote."""
        path = f"{self._remote}/{V3_ATLAS_ROOTDIR}/{self.atlas_name}"
        try:
            # One request, served from the s3fs cache when remote resolution
            # has just listed this path.
            return [p.rsplit("/", 1)[-1] for p in self.fs.ls(path)]
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{self.atlas_name} is not a valid atlas name!"
            ) from None

    def _pick_remote_version(self, folders: List[str]) -> str:
        """Pick the requested, released or newest version folder."""
        if self._requested_version is not None:
            if self._requested_version not in folders:
                raise FileNotFoundError(
                    "Requested version "
                    f"{folder_version_to_dotted(self._requested_version)} "
                    f"for atlas {self.atlas_name} not found in remote."
                )
            return self._requested_version

        if self._remote == descriptors.DEFAULT_REMOTE:
            # last_versions.conf only lists atlases on the BrainGlobe bucket.
            released = get_all_atlases_lastversions().get(self.atlas_name)
            if released is None:
                raise FileNotFoundError(
                    f"{self.atlas_name} has no released version available. "
                    "Specify a version explicitly to download it."
                )
            return str(released).replace(".", "_")

        # Skip anything that isn't a version folder, e.g. a README.
        versions = [
            f for f in folders if re.fullmatch(_VERSION_FOLDER_PATTERN, f)
        ]
        if not versions:
            raise FileNotFoundError(
                f"{self.atlas_name} is not a valid atlas name!"
            )
        return get_latest_version(versions)

    def download(self):
        """Download and extract the atlas files from remote storage.

        The manifest file is removed if any error occurs during the
        download to ensure that incomplete downloads are retried.
        """
        if not self._is_reachable(self._remote):
            check_s3_status(remote=self._remote)

        remote_version_str = self._resolve_remote_version()
        key_name = (
            f"{V3_ATLAS_ROOTDIR}/{self.atlas_name}/"
            f"{remote_version_str}/manifest.json"
        )

        local_path = self.brainglobe_dir / key_name
        remote_path = f"{self._remote}/{key_name}"

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {self.atlas_name} atlas "
            f"v{folder_version_to_dotted(remote_version_str)} manifest:"
        )
        self.fs.get(remote_path, local_path, callback=TqdmCallback())
        self.metadata = read_json(local_path)

        try:
            annotation_set = get_annotation_set(self.metadata)

            # Download terminology file
            terminology = annotation_set["terminology"]
            terminology_location = terminology["location"][1:]
            local_terminology_path = self.brainglobe_dir / terminology_location
            if not local_terminology_path.exists():
                remote_terminology_path = (
                    f"{self._remote}/{terminology_location}"
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
                    f"{self._remote}/{coordspace_location}"
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
                    f"{self._remote}/{root_metadata_path}"
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
                        f"{self._remote}/{masks_metadata_glob}"
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
                        f"{self._remote}/{masks_annotation_values_path}"
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
                    f"{self._remote}/{annotation_location}"
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
                        f"{self._remote}/{root_hemisphere_path}"
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
                    f"{self._remote}/{root_metadata_path}"
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
                        f"{self._remote}/{root_metadata_path}"
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
            True if it is, and None if offline or not on the remote.
        """
        # Cache remote version to avoid multiple requests
        remote_version = self.remote_version
        # Offline, or the remote does not have this atlas
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
