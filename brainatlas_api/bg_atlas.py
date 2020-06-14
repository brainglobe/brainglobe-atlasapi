from pathlib import Path
import tarfile

from brainatlas_api import utils
from brainatlas_api import config
from brainatlas_api import core


COMPRESSED_FILENAME = "atlas.tar.gz"


class BrainGlobeAtlas(core.Atlas):
    """
    Add download functionalities to Atlas class.

        Parameters
        ----------
        brainglobe_path : str or Path object
            default folder for brainglobe downloads

        interm_dir_path : str or Path object
            folder to download the compressed file for extraction
        """

    atlas_name = None
    version = None
    _remote_url_base = (
        "https://gin.g-node.org/brainglobe/atlases/raw/master/{}.tar.gz"
    )

    def __init__(self, brainglobe_path=None, interm_dir_path=None):
        conf = config.read_config()

        if brainglobe_path is None:
            brainglobe_path = conf["default_dirs"]["atlas_dir"]
        self.brainglobe_path = Path(brainglobe_path)

        if interm_dir_path is None:
            interm_dir_path = conf["default_dirs"]["atlas_dir"]
        self.interm_dir_path = Path(interm_dir_path)

        try:
            super().__init__(self.brainglobe_path / self.atlas_full_name)

        except FileNotFoundError:
            print(
                0, f"{self.atlas_full_name} not found locally. Downloading..."
            )
            self.download_extract_file()

            super().__init__(self.brainglobe_path / self.atlas_full_name)

    @property
    def atlas_full_name(self):
        return f"{self.atlas_name}_v{self.version}"

    @property
    def remote_url(self):
        """ Format complete url for download.
        """
        return self._remote_url_base.format(self.atlas_full_name)

    def download_extract_file(self):
        """ Download and extract atlas from remote url.
        """
        utils.check_internet_connection()

        # Get path to folder where data will be saved
        destination_path = self.interm_dir_path / COMPRESSED_FILENAME

        # Try to download atlas data
        utils.retrieve_over_http(self.remote_url, destination_path)

        # Uncompress in brainglobe path:
        tar = tarfile.open(destination_path)
        tar.extractall(path=self.brainglobe_path)
        tar.close()

        destination_path.unlink()


class ExampleAtlas(BrainGlobeAtlas):
    atlas_name = "example_mouse_100um"
    version = "0.2"


class FishAtlas(BrainGlobeAtlas):
    atlas_name = "mpin_zfish_1um"
    version = "0.2"


class RatAtlas(BrainGlobeAtlas):
    # TODO fix
    atlas_name = "ratatlas"
    version = "0.1"


class AllenBrain25Um(BrainGlobeAtlas):
    atlas_name = "allen_mouse_25um"
    version = "0.2"


class AllenHumanBrain500Um(BrainGlobeAtlas):
    atlas_name = "allen_human_500um"
    version = "0.1"
