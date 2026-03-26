"""Defines the BrainGlobe Atlas API v2 classes and functions."""

import re
from collections.abc import Callable
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple, Union

import s3fs
from fsspec.callbacks import TqdmCallback
from rich import print as rprint
from rich.console import Console

from brainglobe_atlasapi import config, core
from brainglobe_atlasapi.atlas_name import AtlasName
from brainglobe_atlasapi.descriptors import (
    V2_ANNOTATION_NAME,
    V2_ATLAS_ROOTDIR,
    V2_HEMISPHERES_NAME,
    V2_MESHES_DIRECTORY,
    V2_TEMPLATE_NAME,
    remote_url_s3,
)
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_internet_connection,
    check_s3_status,
    get_latest_version,
    read_json,
)


def _version_tuple_from_str(version_str):
    return tuple([int(n) for n in version_str.split(".")])


def _version_str_from_tuple(version_tuple: Tuple[int, ...]) -> str:
    return "_".join(str(num) for num in version_tuple)


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
    """

    def __init__(
        self,
        atlas_name: AtlasName,
        version: Optional[str] = None,
        brainglobe_dir: Optional[Union[str, Path]] = None,
        check_latest: bool = True,
        config_dir: Optional[Union[str, Path]] = None,
        fn_update: Optional[Callable] = None,
    ):
        self._remote_version = None
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

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        if brainglobe_dir is None:
            self.brainglobe_dir = Path(conf["default_dirs"]["brainglobe_dir"])
        else:
            self.brainglobe_dir = Path(brainglobe_dir)

        self.brainglobe_dir = self.brainglobe_dir / "brainglobe-atlasapi"

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

        super().__init__(self.brainglobe_dir / self.local_full_name)

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

        (self.brainglobe_dir / V2_ATLAS_ROOTDIR).mkdir(
            parents=True, exist_ok=True
        )

        if self._requested_version is not None:
            pattern = (
                f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
                f"{self._requested_version}/manifest.json"
            )
        else:
            pattern = (
                rf"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
                rf"\d+(?:_\d+)?/manifest.json"
            )

        glob_pattern = f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/*/manifest.json"

        available_versions: List[str] = [
            p.parent.name
            for p in self.brainglobe_dir.glob(glob_pattern)
            if re.search(pattern, p.as_posix())
        ]

        if len(available_versions) == 0:
            return None

        latest_version = get_latest_version(available_versions)

        self._local_full_name = (
            f"{V2_ATLAS_ROOTDIR}/"
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
        self._local_version = _version_tuple_from_str(
            version_str.replace("_", ".")
        )

        return self._local_version

    @property
    def remote_version(self) -> Optional[tuple[int, ...]]:
        """Reads remote version from s3 bucket.

        Largest numerical version assumed to be latest.
        If we are offline, return None.
        """
        if self._remote_version is not None:
            return self._remote_version

        if not check_s3_status(raise_error=False):
            return None

        bucket_path = remote_url_s3.format(f"atlases/{self.atlas_name}")

        if self.fs.exists(bucket_path) is False:
            raise FileNotFoundError(
                f"{self.atlas_name} is not a valid atlas name!"
            )

        if self._requested_version is None:
            versions_path = self.fs.ls(bucket_path)
            available_versions: List[str] = [
                path_str.split("/")[-1] for path_str in versions_path
            ]
            latest_version = get_latest_version(available_versions)
            self._remote_version = _version_tuple_from_str(
                latest_version.replace("_", ".")
            )
        else:
            requested_path = f"{bucket_path}/{self._requested_version}"
            if not self.fs.exists(requested_path):
                raise FileNotFoundError(
                    f"Requested version {self._requested_version} for atlas "
                    f"{self.atlas_name} not found in remote."
                )

            self._remote_version = _version_tuple_from_str(
                self._requested_version.replace("_", ".")
            )

        return self._remote_version

    def download(self):
        """Download and extract the atlas files from remote storage.

        The manifest file is removed if any error occurs during the
        download to ensure that incomplete downloads are retried.
        """
        check_s3_status()

        remote_version_str = _version_str_from_tuple(self.remote_version)
        key_name = (
            f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
            f"{remote_version_str}/manifest.json"
        )

        local_path = self.brainglobe_dir / key_name
        remote_path = remote_url_s3.format(key_name)

        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading {self.atlas_name} atlas "
            f"v{remote_version_str.replace('_', '.')} manifest:"
        )
        self.fs.get(remote_path, local_path, callback=TqdmCallback())
        self.metadata = read_json(local_path)

        try:
            # Download terminology file
            terminology_location = self.metadata["terminology"]["location"][1:]
            local_terminology_path = self.brainglobe_dir / terminology_location
            if not local_terminology_path.exists():
                remote_terminology_path = remote_url_s3.format(
                    terminology_location
                )
                print(
                    f"Downloading terminology metadata "
                    f"for {self.metadata['terminology']['name']}:"
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
                remote_coordspace_path = remote_url_s3.format(
                    coordspace_location
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
            annotation_location = self.metadata["annotation_set"]["location"][
                1:
            ]
            local_annotation_path = self.brainglobe_dir / annotation_location
            if not local_annotation_path.exists():
                root_metadata_path = (
                    annotation_location + f"/{V2_ANNOTATION_NAME}/**/*.json"
                )
                remote_root_metadata_path = remote_url_s3.format(
                    root_metadata_path
                )
                print(
                    f"Downloading annotation metadata "
                    f"for {self.metadata['annotation_set']['name']}:"
                )
                self.fs.get(
                    remote_root_metadata_path,
                    local_annotation_path / V2_ANNOTATION_NAME,
                    callback=TqdmCallback(),
                )
                mesh_path = local_annotation_path / V2_MESHES_DIRECTORY
                mesh_path.mkdir(exist_ok=True)

                if not self.metadata["symmetric"]:
                    root_hemisphere_path = (
                        annotation_location
                        + f"/{V2_HEMISPHERES_NAME}/**/*.json"
                    )
                    remote_root_hemisphere_path = remote_url_s3.format(
                        root_hemisphere_path
                    )
                    self.fs.get(
                        remote_root_hemisphere_path,
                        local_annotation_path / V2_HEMISPHERES_NAME,
                    )

            # Download template metadata files
            template_location = self.metadata["annotation_set"]["template"][
                "location"
            ][1:]
            local_template_path = self.brainglobe_dir / template_location
            if not local_template_path.exists():
                root_metadata_path = (
                    template_location + f"/{V2_TEMPLATE_NAME}/**/*.json"
                )
                remote_root_metadata_path = remote_url_s3.format(
                    root_metadata_path
                )

                print(
                    f"Downloading template metadata "
                    f"for {self.metadata['template']['name']}:"
                )
                self.fs.get(
                    remote_root_metadata_path,
                    local_template_path / V2_TEMPLATE_NAME,
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
                        template_location + f"/{V2_TEMPLATE_NAME}/**/*.json"
                    )
                    remote_root_metadata_path = remote_url_s3.format(
                        root_metadata_path
                    )
                    print(
                        f"Downloading template metadata "
                        f"for {self.metadata['template']['name']}:"
                    )
                    self.fs.get(
                        remote_root_metadata_path,
                        local_template_path / V2_TEMPLATE_NAME,
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

        local = _version_str_from_tuple(self.local_version)
        online = _version_str_from_tuple(remote_version)

        if local != online:
            if print_warning:
                rprint(
                    "[b][magenta2]brainglobe_atlasapi[/b]: "
                    f"[b]{self.atlas_name}[/b] version "
                    f"[b]{local.replace('_', '.')}[/b] "
                    f"is not the latest available "
                    f"([b]{online.replace('_', '.')}[/b]). "
                    "To update the atlas run in the terminal:[/magenta2]\n"
                    f" [gold1]brainglobe update -a {self.atlas_name}[/gold1]"
                )
            return False
        return True

    def __repr__(self) -> str:
        """Fancy print providing atlas information."""
        name_split = self.atlas_name.split("_")
        res = f" (res. {name_split.pop()})"
        pretty_name = f"{' '.join(name_split)} atlas{res}"
        return pretty_name

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
