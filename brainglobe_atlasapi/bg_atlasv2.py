from pathlib import Path

import zarr

from brainglobe_atlasapi import config, core, descriptors


class BrainGlobeAtlasV2(core.Atlas):
    atlas_name = None
    _remote_url_base = descriptors.remote_url_base_v2[0]

    def __init__(
        self,
        atlas_name,
        brainglobe_dir="/home/igor/.brainglobe-tests/.brainglobe_v2",
        interm_download_dir="/home/igor/.brainglobe-tests/.brainglobe_v2",
        check_latest=True,
        config_dir=None,
        fn_update=None,
    ):
        self.atlas_name = atlas_name
        self.fn_update = fn_update

        # Read BrainGlobe configuration file:
        conf = config.read_config(config_dir)

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        for dir, dirname in zip(
            [brainglobe_dir, interm_download_dir],
            ["brainglobe_dir", "interm_download_dir"],
        ):
            if dir is None:
                dir = conf["default_dirs"][dirname]

            # If the default folder does not exist yet, make it:
            dir_path = Path(dir)
            dir_path.mkdir(parents=True, exist_ok=True)
            setattr(self, dirname, dir_path)

        super().__init__(self.brainglobe_dir, self.local_full_name)

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """
        pattern = f"{self.atlas_name}_v*"
        candidate_dirs = list(self.brainglobe_dir.glob(pattern))

        # If multiple folders exist, raise error:
        if len(candidate_dirs) > 1:
            raise FileExistsError(
                f"Multiple versions of atlas {self.atlas_name} in "
                f"{self.brainglobe_dir}"
            )
        # If no one exist, return None:
        elif len(candidate_dirs) == 0:
            return
        # Else, return actual name:
        else:
            return candidate_dirs[0].name

    @property
    def reference(self):
        reference_path = (
            self.root_dir
            / self.metadata["reference_images"][0]
            / f"{int(self.resolution[0])}um"
        )
        zarr_array = zarr.open_array(reference_path, mode="r", zarr_version=3)

        return zarr_array[:]
