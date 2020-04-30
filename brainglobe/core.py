from brainio import brainio
import json
from pathlib import Path


# Either a dictionary subclass or a loop to set json keys as properties, as you guys prefer
class Atlas(dict):
    def __init__(self, path):
        self.root = Path(path)

        if path.is_dir():
            with open(self.root / "atlas_metadata.json", "r") as f:
                atlas_metadata = json.load(f)
        else:
            raise TypeError(f"{path} is not a valid folder")

        super().__init__(**atlas_metadata)

    def _get_element_path(self, element_name):
        """Get the path to an 'element' of the atlas (i.e. the average brain,
        the atlas, or the hemispheres atlas)

        :param str element_name: The name of the item to retrieve
        :return: The path to that atlas element on the filesystem
        :rtype: str
        """

        return self.base_folder / self["data_files"][element_name]

    def get_data_from_element(self, element_name):
        """ This can be easily changed to a different loading API if needed.
        """
        full_path = self.base_folder / self[element_name]
        return brainio.load(full_path)
