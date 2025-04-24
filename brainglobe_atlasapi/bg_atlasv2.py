import tarfile
from pathlib import Path
from typing import Union

import zarr
from typing_extensions import deprecated

from brainglobe_atlasapi import BrainGlobeAtlas, descriptors, utils
from brainglobe_atlasapi.descriptors import STRUCTURES_FILENAME
from brainglobe_atlasapi.utils import (
    check_gin_status,
    check_internet_connection,
    read_json,
)


class BrainGlobeAtlasV2(BrainGlobeAtlas):
    atlas_name = None
    _remote_url_base = descriptors.remote_url_base_v2

    def __init__(
        self,
        atlas_name,
        **kwargs,
    ):
        self._local_full_name = None
        self._template = None
        super().__init__(atlas_name, **kwargs)

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """
        if self._local_full_name is None:
            # Create the v2 atlases directory if it doesn't exist
            if not (self.brainglobe_dir / "atlases").exists():
                (self.brainglobe_dir / "atlases").mkdir(
                    parents=True, exist_ok=True
                )

            pattern = f"atlases/{self.atlas_name}_v*.json"
            self._local_full_name = self._get_local_full_name(pattern)
            # Prepend the atlases directory to the local full name
            if self._local_full_name is not None:
                self._local_full_name = "atlases/" + self._local_full_name

        return self._local_full_name

    @property
    def local_version(self):
        """If atlas is local, return actual version of the downloaded files;
        Else, return none.
        """
        version_str = self.metadata["version"]

        if version_str is None:
            return None

        return tuple(int(version) for version in version_str.split("."))

    @property
    def template(self):
        if self._template is None:
            template_path = (
                self.root_dir
                / "templates"
                / self.metadata["reference_images"][0]
                / f"{int(self.resolution[0])}um"
            )
            if not template_path.exists():
                remote_url = self._remote_url_base.format(
                    f"{self.metadata['reference_images'][0]}/{int(self.resolution[0])}um.tar.gz"
                )
                self.download_tar_file(remote_url, template_path)

            template_array = zarr.open_array(
                template_path, mode="r", zarr_format=3
            )
            self._template = template_array[:]

        return self._template

    @property
    @deprecated("Use the 'template' property instead.")
    def reference(self):
        return self.template

    @property
    def annotation(self):
        if self._annotation is None:
            annotation_path = (
                self.root_dir
                / "annotations"
                / self.metadata["annotation_images"][0]
                / f"{int(self.resolution[0])}um"
            )
            if not annotation_path.exists():
                remote_url = self._remote_url_base.format(
                    f"{self.metadata['annotation_images'][0]}/{int(self.resolution[0])}um.tar.gz"
                )
                self.download_tar_file(remote_url, annotation_path)

            annotation_array = zarr.open_array(
                annotation_path, mode="r", zarr_format=3
            )
            self._annotation = annotation_array[:]

        return self._annotation

    def download_tar_file(self, url: str, local_path: Path):
        """Download and extract a file from a URL."""
        # Implement the download and extraction logic here
        destination_path = (
            self.interm_download_dir / local_path.with_suffix(".tar.gz").name
        )

        utils.retrieve_over_http(url, destination_path, self.fn_update)
        # Create the directory if it doesn't exist
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Uncompress the downloaded file
        tar = tarfile.open(destination_path)
        tar.extractall(path=local_path.parent)
        tar.close()

        destination_path.unlink()

    @property
    def remote_url(self):
        if self.remote_version is not None:
            name = (
                f"{self.atlas_name}_v{self.remote_version[0]}."
                f"{self.remote_version[1]}.json"
            )

            return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url."""
        check_internet_connection()
        check_gin_status()
        # Cache to avoid multiple requests
        remote_version = self.remote_version

        name = (
            f"{self.atlas_name}_v{remote_version[0]}."
            f"{remote_version[1]}.json"
        )

        # Get path to folder where data will be saved
        destination_path = self.brainglobe_dir / "atlases" / name

        # Try to download atlas data
        utils.retrieve_over_http(
            self.remote_url, destination_path, self.fn_update
        )

        # Check if component directories exist, if not create them
        self.metadata = read_json(destination_path)
        annotations_dir = (
            destination_path.parent
            / "annotations"
            / self.metadata["annotation_images"][0]
        )
        template_dir = (
            destination_path.parent
            / "templates"
            / self.metadata["reference_images"][0]
        )
        meshes_dir = (
            destination_path.parent / "meshes" / self.metadata["meshes"][0]
        )

        if not annotations_dir.exists():
            annotations_dir.mkdir(parents=True, exist_ok=True)
        if not template_dir.exists():
            template_dir.mkdir(parents=True, exist_ok=True)
        if not meshes_dir.exists():
            meshes_dir.mkdir(parents=True, exist_ok=True)

        structures_path = (
            destination_path.parent
            / "annotations"
            / annotations_dir
            / STRUCTURES_FILENAME
        )
        structures_csv = STRUCTURES_FILENAME.split(".")[0] + ".csv"
        if not structures_path.exists():
            remote_url = self._remote_url_base.format(
                f"{annotations_dir.name}/{STRUCTURES_FILENAME}"
            )
            utils.retrieve_over_http(remote_url, structures_path)

        if not structures_path.with_suffix(".csv").exists():
            remote_url = self._remote_url_base.format(
                f"{annotations_dir.name}/{structures_csv}"
            )
            utils.retrieve_over_http(
                remote_url, structures_path.with_suffix(".csv")
            )

    def _get_from_structure(self, structure, key):
        """
        Internal interface to the structure dict. It supports querying with a
        single structure id or a list of ids.

        Parameters
        ----------
        structure : int or str or list
            Valid id or acronym, or list of ids or acronyms.
        key : str
            Key for the Structure dictionary (eg "name" or "rgb_triplet").

        Returns
        -------
        value or list of values
            If structure is a list, returns list.
        """
        # Check if mesh is cached
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
            remote_path = self._remote_url_base.format(
                f"{self.metadata['meshes'][0]}/{mesh_id}.obj"
            )
            utils.retrieve_over_http(remote_path, mesh_path)

        return
