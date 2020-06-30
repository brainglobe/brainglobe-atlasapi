from pathlib import Path
import tarfile

from brainatlas_api import utils
from brainatlas_api import config
from brainatlas_api import core


COMPRESSED_FILENAME = "atlas.tar.gz"

__all__ = [
    "FishAtlas",
    "RatAtlas",
    "AllenBrain25Um",
    "AllenHumanBrain500Um",
    "KimUnified25Um",
    "KimUnified50Um",
]


def _version_tuple_from_str(version_str):
    return tuple([int(n) for n in version_str.split(".")])


class BrainGlobeAtlas(core.Atlas):
    """Add download functionalities to Atlas class.

        Parameters
        ----------
        brainglobe_dir : str or Path object
            default folder for brainglobe downloads

        interm_download_dir : str or Path object
            folder to download the compressed file for extraction
        """

    atlas_name = None
    _remote_url_base = (
        "https://gin.g-node.org/brainglobe/atlases/raw/master/{}"
    )

    def __init__(self, brainglobe_dir=None, interm_download_dir=None):
        # Read BrainGlobe configuration file:
        conf = config.read_config()

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
            dir_path.mkdir(exist_ok=True)
            setattr(self, dirname, dir_path)

        # Look for this atlas in local brainglobe folder:
        if self.local_full_name is None:
            print(0, f"{self.atlas_name} not found locally. Downloading...")
            self.download_extract_file()

        # Instantiate after eventual download:
        super().__init__(self.brainglobe_dir / self.local_full_name)

    @property
    def local_version(self):
        """If atlas is local, return actual version of the downloaded files;
        Else, return none.
        """
        full_name = self.local_full_name

        if full_name is None:
            return None

        return _version_tuple_from_str(full_name.split("_v")[-1])

    @property
    def remote_version(self):
        """Remote version read from GIN conf file.
        """
        remote_url = self._remote_url_base.format("last_versions.conf")
        versions_conf = utils.conf_from_url(remote_url)

        return _version_tuple_from_str(
            versions_conf["atlases"][self.atlas_name]
        )

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None
        """
        pattern = f"{self.atlas_name}_v*"
        candidate_dirs = list(self.brainglobe_dir.glob(pattern))

        # If multiple folders exist, raise error:
        if len(candidate_dirs) > 1:
            raise FileExistsError(
                f"Multiple versions of atlas {self.atlas_name} in {self.brainglobe_dir}"
            )
        # If no one exist, return None:
        elif len(candidate_dirs) == 0:
            return None
        # Else, return actual name:
        else:
            return candidate_dirs[0].name

    @property
    def remote_url(self):
        """Format complete url for download.
        """
        maj, min = self.remote_version
        name = f"{self.atlas_name}_v{maj}.{min}.tar.gz"
        return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url.
        """
        utils.check_internet_connection()

        # Get path to folder where data will be saved
        destination_path = self.interm_download_dir / COMPRESSED_FILENAME

        # Try to download atlas data
        utils.retrieve_over_http(self.remote_url, destination_path)

        # Uncompress in brainglobe path:
        tar = tarfile.open(destination_path)
        tar.extractall(path=self.brainglobe_dir)
        tar.close()

        destination_path.unlink()


class ExampleAtlas(BrainGlobeAtlas):
    atlas_name = "example_mouse_100um"


class FishAtlas(BrainGlobeAtlas):
    atlas_name = "mpin_zfish_1um"


class RatAtlas(BrainGlobeAtlas):
    # TODO fix hierarchy and meshes
    atlas_name = "ratatlas"


class AllenBrain25Um(BrainGlobeAtlas):
    atlas_name = "allen_mouse_25um"


class KimUnified25Um(BrainGlobeAtlas):
    atlas_name = "kim_unified_25um"


class KimUnified50Um(BrainGlobeAtlas):
    atlas_name = "kim_unified_50um"


class AllenHumanBrain500Um(BrainGlobeAtlas):
    # TODO fix meshes
    atlas_name = "allen_human_500um"
