from .utils import open_json, read_tiff, make_hemispheres_stack
from pathlib import Path


class Atlas():
    def __init__(self, path):
        self.root = Path(path)
        self.metadata = open_json(self.root / "atlas_metadata.json")

        for attr in ["name", "shape", "resolution"]:
            self.__setattr__(attr, self.metadata[attr])

        self.structures = open_json(self.root / "structures.json")

        self._reference = None
        self._annotated = None
        self._hemispheres = None

    @property
    def reference(self):
        if self._reference is None:
            self._reference = read_tiff(self.root / "reference.tiff")
        return self._reference

    @property
    def annotated(self):
        if self._annotated is None:
            self._annotated = read_tiff(self.root / "annotated.tiff")
        return self._annotated

    @property
    def hemispheres(self):
        if self._hemispheres is None:
            # If reference is symmetric generate hemispheres block:
            if self.metadata["symmetric"]:
                self._hemispheres = make_hemispheres_stack(self.shape)
            else:
                self._hemispheres = read_tiff(self.root / "hemispheres.tiff")
        return self._hemispheres

    def get_point_hemisphere(self, point):
        pass

    def get_point_region(self, point):
        pass

    def get_region_mesh(self, region_id):
        pass

    def get_brain_mesh(self):
        pass