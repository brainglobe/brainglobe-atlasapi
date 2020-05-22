from brainatlas_api.utils import open_json, read_tiff, make_hemispheres_stack
from pathlib import Path
from structures.brainatlas_api.structure_tree import StructureTree
from brainatlas_api.obj_utils import read_obj


class MeshDictionary(dict):
    """ Class to cache the loading of .obj mesh files
        Parameters
        ----------
        mesh_path : str or Path object
            path to folder containing all meshes .obj files
        """

    def __init__(self, mesh_path):
        self.root = Path(mesh_path)

        # Create dictionary of loadable files:
        self.files_dict = {int(f.resolve().stem): f for f in
                           self.root.glob("*.obj")}

    def __getitem__(self, item):
        """ Load mesh if it has not been read before, and add it to self dict.
        Parameters
        ----------
        item : int
            id of structure to be loaded

        Returns
        -------
        tuple :
            Tuple with mesh description

        """
        if not item in self.keys():
            value = read_obj(self.files_dict[item])
            super().__setitem__(item, value)

        return super().__getitem__(item)


class Atlas():
    """ Base class to handle atlases in brainglobe.

        Parameters
        ----------
        path : str or Path object
            path to folder containing data info
        """

    def __init__(self, path):
        self.root = Path(path)
        self.metadata = open_json(self.root / "atlas_metadata.json")

        # Class for structures:
        structures_list = open_json(self.root / "structures.json")
        self.structures = StructureTree(structures_list)

        # Cached loading of meshes:
        self.region_meshes_dict = MeshDictionary(self.root / "meshes")

        for attr in ["name", "shape", "resolution"]:
            self.__setattr__(attr, self.metadata[attr])

        self._reference = None
        self._annotated = None
        self._hemispheres = None

        # Dictionaries to map acronyms to ids...:
        self.acronym_to_id_map = self.structures.get_id_acronym_map()
        # ...and viceversa:
        self.id_to_acronym_map = {v: k for k, v in
                                  self.acronym_to_id_map.items()}

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

    def get_hemisphere_from_coords(self, coords):
        return self.hemispheres[self.idx_from_coords(coords)]

    def get_region_id_from_coords(self, coords):
        return self.annotated[self.idx_from_coords(coords)]

    def get_region_name_from_coords(self, coords):
        region_id = self.get_region_id_from_coords(coords)

        return self.id_to_acronym_map[region_id]

    def get_mesh_from_id(self, region_id):
        return self.region_meshes_dict[region_id]

    def get_mesh_from_name(self, region_name):
        region_id = self.acronym_to_id_map[region_name]
        return self.get_mesh_from_id(region_id)

    def get_brain_mesh(self):
        return self.get_mesh_from_name("root")

    @staticmethod
    def idx_from_coords(coords):
        return tuple([int(c) for c in coords])