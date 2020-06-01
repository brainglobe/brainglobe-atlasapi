import meshio as mio
from collections import UserDict


class Structure(UserDict):
    def __getitem__(self, item):
        if item == "mesh" and self.data[item] is None:
            # TODO gracefully fail with warning if no mesh:
            self.data[item] = mio.read(self["mesh_filename"])

        return self.data[item]

    def get_center_of_mass(self):
        pass

    def ancestors(self):
        pass

    def descendants(self):
        pass


class StructuresDict(UserDict):
    """ Class to handle dual indexing by either acronym or id.

        Parameters
        ----------
        mesh_path : str or Path object
            path to folder containing all meshes .obj files
        """

    def __init__(self, structures_list, mesh_dir_path):
        super().__init__()

        # Acronym to id map:
        self.acronym_to_id_map = {
            r["acronym"]: r["id"] for r in structures_list
        }

        for struct_dict in structures_list:
            sid = struct_dict["id"]
            mesh_filename = mesh_dir_path / f"{sid}.obj"
            self.data[sid] = Structure(
                mesh_filename=mesh_filename, mesh=None, **struct_dict
            )

    def __getitem__(self, item):
        if isinstance(item, int):
            return self.data[item]
        elif isinstance(item, str):
            return self.data[self.acronym_to_id_map[item]]
        elif isinstance(item, list):
            return [self.__getitem__(i) for i in item]
