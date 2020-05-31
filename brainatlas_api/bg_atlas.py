from pathlib import Path
import urllib
import shutil
import tarfile
import warnings

from brainatlas_api import DEFAULT_PATH
from brainatlas_api.core import Atlas
from brainatlas_api.utils import check_internet_connection


COMPRESSED_FILENAME = "atlas.tar.gz"


class BrainGlobeAtlas(Atlas):
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

        if brainglobe_path is None:
            brainglobe_path = DEFAULT_PATH

        self.brainglobe_path = Path(brainglobe_path)

        self.interm_dir_path = (
            Path(interm_dir_path)
            if interm_dir_path is not None
            else self.brainglobe_path
        )

        self.atlas_name = self.atlas_name + f"_v{self.version}"

        try:
            super().__init__(self.brainglobe_path / self.atlas_name)

        except FileNotFoundError:
            warnings.warn(f"{self.atlas_name} not found. Downloading...")
            self.download_extract_file()

            super().__init__(self.brainglobe_path / self.atlas_name)

    @property
    def remote_url(self):
        """ Format complete url for download.
        """
        return self._remote_url_base.format(self.atlas_name)

    def download_extract_file(self):
        """ Download and extract atlas from remote url.
        """
        check_internet_connection()

        # Get path to folder where data will be saved
        destination_path = self.interm_dir_path / COMPRESSED_FILENAME

        # Try to download atlas data
        try:
            with urllib.request.urlopen(self.remote_url) as response:
                with open(destination_path, "wb") as outfile:
                    shutil.copyfileobj(response, outfile)
        except urllib.error.HTTPError as e:
            raise FileNotFoundError(
                f"Could not download data from {self.remote_url}. error code: {e.code}"
            )
        except urllib.error.URLError as e:
            raise FileNotFoundError(
                f"Could not download data from {self.remote_url}: {e.args}"
            )

        # Uncompress
        tar = tarfile.open(destination_path)

        # Ensure brainglobe path exists and extract there:
        self.brainglobe_path.mkdir(exist_ok=True)
        tar.extractall(path=self.brainglobe_path)
        tar.close()

        destination_path.unlink()


class TestAtlas(BrainGlobeAtlas):
    atlas_name = "test_allen_100um"
    version = "0.1"


class FishAtlas(BrainGlobeAtlas):
    atlas_name = "fishatlas"
    version = "0.1"


class RatAtlas(BrainGlobeAtlas):
    atlas_name = "ratatlas"
    version = "0.1"


class AllenBrain25Um(BrainGlobeAtlas):
    atlas_name = "allenbrain25um"
    version = "0.1"


class AllenHumanBrain500Um(BrainGlobeAtlas):
    atlas_name = "allen_human_500um"
    version = "0.1"
