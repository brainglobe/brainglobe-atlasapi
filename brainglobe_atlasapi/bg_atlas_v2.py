"""Defines the BrainGlobe Atlas API v2 classes and functions."""

import re
from io import StringIO
from pathlib import Path
from typing import List, Optional, Union

import ngff_zarr
import ngff_zarr as nz
import numpy.typing as npt
import s3fs
from rich import print as rprint
from rich.console import Console
from typing_extensions import deprecated

from brainglobe_atlasapi import core
from brainglobe_atlasapi.bg_atlas import (
    _version_tuple_from_str,
)
from brainglobe_atlasapi.descriptors import (
    ANNOTATION_DTYPE,
    REFERENCE_DTYPE,
    V2_ANNOTATION_NAME,
    V2_ATLAS_ROOTDIR,
    V2_MESHES_DIRECTORY,
    V2_TEMPLATE_NAME,
    remote_url_s3,
)
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_internet_connection,
    read_json,
)


def _version_str_from_tuple(version_tuple):
    return "_".join(str(num) for num in version_tuple)


class BrainGlobeAtlas(core.Atlas):
    """Add remote atlas fetching and version comparison functionalities
    to the core Atlas class.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to be used.
    resolution : int or float
        Desired isotropic resolution in microns.
    version : str (optional)
        Desired version of the atlas. If None, the latest version will be used.
    brainglobe_dir : str or Path object
        Default folder for brainglobe downloads.
    interm_download_dir : str or Path object
        Folder to download the compressed file for extraction.
    check_latest : bool (optional)
        If true, check if we have the most recent atlas (default=True). Set
        this to False to avoid waiting for remote server response on atlas
        instantiation and to suppress warnings.
    print_authors : bool (optional)
        If true, disable default listing of the atlas reference.
    fn_update : Callable
        Handler function to update during download. Takes completed and total
        bytes.
    """

    atlas_name = None

    def __init__(
        self,
        atlas_name: str,
        version: Optional[str] = None,
        **kwargs,
    ):
        self._remote_version = None
        self._local_full_name = None
        self._template = None
        self._requested_version = version
        self._local_version = (
            _version_tuple_from_str(version) if version else None
        )
        self._shape = None
        self.fs = s3fs.S3FileSystem(anon=True)

        super().__init__(atlas_name, **kwargs)

        template_location = self.metadata["annotation_set"]["template"][
            "location"
        ][1:]
        template_path = self.root_dir / template_location / V2_TEMPLATE_NAME

        multiscale = nz.from_ngff_zarr(template_path)
        self._determine_pyramid_level(multiscale)
        self.additional_references.pyramid_level = self._pyramid_level

    @property
    def local_full_name(self):
        """
        Returns the local full path to the manifest.json file of the atlas.
        If not found, returns None.
        """
        if self._local_full_name is not None:
            return self._local_full_name

        (self.brainglobe_dir / V2_ATLAS_ROOTDIR).mkdir(
            parents=True, exist_ok=True
        )

        if self._requested_version is not None:
            self._requested_version = self._requested_version.replace(".", "_")

            pattern = (
                f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
                f"v{self._requested_version}/manifest.json"
            )
        else:
            pattern = (
                rf"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
                rf"\d+(?:_\d+)?/manifest.json"
            )

        glob_pattern = f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/*/manifest.json"

        available_versions: List[tuple[int, ...]] = [
            p.parent.name
            for p in self.brainglobe_dir.glob(glob_pattern)
            if re.search(pattern, str(p))
        ]

        if len(available_versions) == 0:
            return None

        available_versions.sort(reverse=True)

        self._local_full_name = (
            f"{V2_ATLAS_ROOTDIR}/"
            f"{self.atlas_name}/"
            f"{available_versions[0]}/"
            f"manifest.json"
        )

        return self._local_full_name

    @property
    def local_version(self):
        """
        If atlas is local, return actual version of the downloaded files.
        Else, return none.
        """
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

        if not check_internet_connection(raise_error=False):
            return None

        bucket_path = remote_url_s3.format(f"atlases/{self.atlas_name}")

        if self._requested_version is None:
            versions_path = self.fs.ls(bucket_path)
            available_versions: List[str] = [
                path_str.split("/")[-1] for path_str in versions_path
            ]
            available_versions.sort(reverse=True)

            self._remote_version = _version_tuple_from_str(
                available_versions[0].replace("_", ".")
            )
        else:
            requested_path = f"{bucket_path}/v{self._requested_version}"
            if not self.fs.exists(requested_path):
                raise ValueError(
                    f"Requested version {self._requested_version} for atlas "
                    f"{self.atlas_name} not found in remote."
                )

            self._remote_version = _version_tuple_from_str(
                self._requested_version.replace("_", ".")
            )

        return self._remote_version

    @property
    def template(self) -> npt.NDArray[REFERENCE_DTYPE]:
        """Return the template image data. Loads it if not already loaded."""
        if self._template is not None:
            return self._template

        template_location = self.metadata["annotation_set"]["template"][
            "location"
        ][1:]

        template_path = self.root_dir / template_location / V2_TEMPLATE_NAME

        multiscale = nz.from_ngff_zarr(template_path)
        resolution_path = template_path / str(self._pyramid_level)

        if not (resolution_path / "c").exists():
            print("Downloading template...")
            remote_path = remote_url_s3.format(
                f"{template_location}/{V2_TEMPLATE_NAME}/{self._pyramid_level}/"
            )
            self.fs.get(remote_path, resolution_path, recursive=True)

        self._template = multiscale.images[self._pyramid_level].data.compute()

        return self._template

    @property
    @deprecated("Use the 'template' property instead.")
    def reference(self):
        """Deprecated: use 'template' property instead."""
        return self.template

    @property
    def annotation(self) -> npt.NDArray[ANNOTATION_DTYPE]:
        """Return the annotation image data. Loads it if not already loaded."""
        if self._annotation is not None:
            return self._annotation

        annotation_location = self.metadata["annotation_set"]["location"][1:]
        annotation_path = (
            self.root_dir / annotation_location / V2_ANNOTATION_NAME
        )

        multiscale = nz.from_ngff_zarr(annotation_path)
        resolution_path = annotation_path / str(self._pyramid_level)

        if not (resolution_path / "c").exists():
            print("Downloading annotations...")
            remote_path = remote_url_s3.format(
                f"{annotation_location}/{V2_ANNOTATION_NAME}/{self._pyramid_level}/"
            )
            self.fs.get(remote_path, resolution_path, recursive=True)

        self._annotation = multiscale.images[
            self._pyramid_level
        ].data.compute()

        return self._annotation

    def _determine_pyramid_level(self, multiscale: ngff_zarr.Multiscales):
        for metadata in multiscale.metadata.datasets:
            scales = metadata.coordinateTransformations[0].scale
            if all(
                (res / 1000) == scale
                for res, scale in zip(self.resolution, scales)
            ):
                self._pyramid_level = int(metadata.path)
                break

        if self._pyramid_level is None:
            raise ValueError(
                f"Requested resolution {self.resolution} um is invalid."
            )

    def download_extract_file(self):
        """Download and extract the atlas files from remote storage."""
        check_internet_connection()

        remote_version_str = _version_str_from_tuple(self.remote_version)
        key_name = (
            f"{V2_ATLAS_ROOTDIR}/{self.atlas_name}/"
            f"{remote_version_str}/manifest.json"
        )

        local_path = self.brainglobe_dir / key_name
        remote_path = remote_url_s3.format(key_name)

        self.fs.get(remote_path, local_path)
        self.metadata = read_json(local_path)

        # Download terminology file
        terminology_location = self.metadata["terminology"]["location"][1:]
        local_terminology_path = self.brainglobe_dir / terminology_location
        if not local_terminology_path.exists():
            remote_terminology_path = remote_url_s3.format(
                terminology_location
            )
            self.fs.get(
                remote_terminology_path, local_terminology_path, recursive=True
            )

        # Download coordinate space files
        coordspace_location = self.metadata["coordinate_space"]["location"][1:]
        local_coordspace_path = self.brainglobe_dir / coordspace_location
        if not local_coordspace_path.exists():
            remote_coordspace_path = remote_url_s3.format(coordspace_location)
            self.fs.get(
                remote_coordspace_path, local_coordspace_path, recursive=True
            )

        # Download annotation metadata files
        annotation_location = self.metadata["annotation_set"]["location"][1:]
        local_annotation_path = self.brainglobe_dir / annotation_location
        if not local_annotation_path.exists():
            root_metadata_path = (
                annotation_location + f"/{V2_ANNOTATION_NAME}/**/*.json"
            )
            remote_root_metadata_path = remote_url_s3.format(
                root_metadata_path
            )

            self.fs.get(
                remote_root_metadata_path,
                local_annotation_path / V2_ANNOTATION_NAME,
            )
            mesh_path = local_annotation_path / V2_MESHES_DIRECTORY
            mesh_path.mkdir(exist_ok=True)

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

            self.fs.get(
                remote_root_metadata_path,
                local_template_path / V2_TEMPLATE_NAME,
            )

    def _get_from_structure(self, structure, key):
        if key == "mesh":
            if isinstance(structure, list) or isinstance(structure, tuple):
                for s in structure:
                    self._check_mesh_cached(s)
            else:
                self._check_mesh_cached(structure)

        return super()._get_from_structure(structure, key)

    def _check_mesh_cached(self, structure: Union[str, int]):
        """Check if the mesh is cached in the local directory.
        Download from the remote if not cached.

        Parameters
        ----------
        structure : str or int
            Name of the mesh file.

        Returns
        -------
        bool
            True if the mesh is cached, False otherwise.
        """
        mesh_path = Path(self._get_from_structure(structure, "mesh_filename"))

        try:
            mesh_id = int(structure)
        except ValueError:
            mesh_id = int(self.structures.acronym_to_id_map[structure])

        # Check if the mesh is cached
        if not mesh_path.exists():
            # If not cached, download it
            structure_name = self.structures[mesh_id]["acronym"]
            print(f"Downloading mesh for {structure_name}...")
            meshes_root = (
                self.metadata["annotation_set"]["location"][1:]
                + f"/{V2_MESHES_DIRECTORY}"
            )
            remote_mesh_path = remote_url_s3.format(f"{meshes_root}/{mesh_id}")
            local_mesh_path = self.brainglobe_dir / meshes_root / f"{mesh_id}"
            self.fs.get(remote_mesh_path, local_mesh_path)

        return

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
            return

        local = _version_str_from_tuple(self.local_version)
        online = _version_str_from_tuple(remote_version)

        if local != online:
            if print_warning:
                rprint(
                    "[b][magenta2]brainglobe_atlasapi[/b]: "
                    f"[b]{self.atlas_name}[/b] version [b]{local}[/b] "
                    f"is not the latest available ([b]{online}[/b]). "
                    "To update the atlas run in the terminal:[/magenta2]\n"
                    f" [gold1]brainglobe update -a {self.atlas_name}[/gold1]"
                )
            return False
        return True

    def __repr__(self):
        """Fancy print providing atlas information."""
        name_split = self.atlas_name.split("_")
        res = f" (res. {name_split.pop()})"
        pretty_name = f"{' '.join(name_split)} atlas{res}"
        return pretty_name

    def __str__(self):
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
