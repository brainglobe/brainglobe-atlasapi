import tarfile
from pathlib import Path

import zarr

from brainglobe_atlasapi import BrainGlobeAtlas, descriptors, utils
from brainglobe_atlasapi.utils import (
    check_gin_status,
    check_internet_connection,
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
        super().__init__(atlas_name, **kwargs)

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """
        if self._local_full_name is None:
            pattern = f"{self.atlas_name}_v*.json"
            self._local_full_name = self._get_local_full_name(pattern)

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
    def reference(self):
        if self._reference is None:
            reference_path = (
                self.root_dir
                / "templates"
                / self.metadata["reference_images"][0]
                / f"{int(self.resolution[0])}um"
            )
            if not reference_path.exists():
                remote_url = self._remote_url_base.format(
                    f"{self.metadata['reference_images'][0]}/{int(self.resolution[0])}um.tar.gz"
                )
                self.download_tar_file(remote_url, reference_path)

            reference_array = zarr.open_array(
                reference_path, mode="r", zarr_format=3
            )
            self._reference = reference_array[:]

        return self._reference

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
        destination_path = self.interm_download_dir / local_path.with_suffix(
            ".tar.gz"
        )
        utils.retrieve_over_http(url, destination_path, self.fn_update)

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

        name = (
            f"{self.atlas_name}_v{self.remote_version[0]}."
            f"{self.remote_version[1]}.json"
        )

        # Get path to folder where data will be saved
        destination_path = self.brainglobe_dir / name

        # Try to download atlas data
        utils.retrieve_over_http(
            self.remote_url, destination_path, self.fn_update
        )
