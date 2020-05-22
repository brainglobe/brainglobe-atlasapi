from pathlib import Path
from brainatlas_api.core import Atlas
import urllib
import shutil
import tarfile
import warnings

DEFAULT_PATH = Path.home() / ".brainglobe"
COMPRESSED_FILENAME = "atlas.tar.gz"

class BrainGlobeAtlas(Atlas):
    """
    Add download functionalities to Atlas class.

        Parameters
        ----------
        brainglobe_path : str or Path object
        default folder for brainglobe downloads
        interm_dir_path : str or Path object
        """

    _atlas_name = None
    _remote_url_base = "https://gin.g-node.org/brainglobe/atlases/raw/master/{}.tar.gz"
    def __init__(self, brainglobe_path=None, interm_dir_path=None):

        if brainglobe_path is None:
            brainglobe_path = DEFAULT_PATH

        self.brainglobe_path = Path(brainglobe_path)

        self.interm_dir_path = Path(interm_dir_path) if interm_dir_path is not None \
                else self.brainglobe_path

        try:
            super().__init__(self.brainglobe_path / self._atlas_name)

        except FileNotFoundError:
            warnings.warn(f"{self._atlas_name} not found. Downloading...")
            self.download_extract_file()

            super().__init__(self.brainglobe_path / self._atlas_name)

    @property
    def remote_url(self):
        """ Format complete url for download.
        """
        return self._remote_url_base.format(self._atlas_name)

    def download_extract_file(self):
        """ Download and extract atlas from remote url.
        """
        destination_path = self.interm_dir_path / COMPRESSED_FILENAME

        with urllib.request.urlopen(self.remote_url) as response:
            with open(destination_path, "wb") as outfile:
                shutil.copyfileobj(response, outfile)

        tar = tarfile.open(destination_path)

        # Ensure brainglobe path exists and extract there:
        self.brainglobe_path.mkdir(exist_ok=True)
        tar.extractall(path=self.brainglobe_path)
        tar.close()

        destination_path.unlink()


class TestAtlas(BrainGlobeAtlas):
    _atlas_name = "test"