import zarr

from brainglobe_atlasapi import BrainGlobeAtlas, descriptors


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
            zarr_array = zarr.open_array(
                reference_path, mode="r", zarr_format=3
            )
            self._reference = zarr_array[:]

        return self._reference
