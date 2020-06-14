import meshio as mio
from collections import UserDict
import warnings


class Structure(UserDict):
    """Class implementing the lazy loading of a mesh if the dictionary is
    queried for it.
    """

    def __getitem__(self, item):
        if item == "mesh" and self.data[item] is None:
            if self.data["mesh_filename"] is None:
                warnings.warn(
                    "No mesh filename for region {}".format(
                        self.data["acronym"]
                    )
                )
                return None
            try:
                self.data[item] = mio.read(self.data["mesh_filename"])
            except (TypeError, mio.ReadError):
                raise mio.ReadError(
                    "No valid mesh for region: {}".format(self.data["acronym"])
                )

        return self.data[item]


class StructuresDict(UserDict):
    """Class to handle dual indexing by either acronym or id.

    Parameters
    ----------
    mesh_path : str or Path object
        path to folder containing all meshes .obj files
    """

    def __init__(self, structures_list):
        super().__init__()

        # Acronym to id map:
        self.acronym_to_id_map = {
            r["acronym"]: r["id"] for r in structures_list
        }

        for struct in structures_list:
            sid = struct["id"]
            self.data[sid] = Structure(**struct, mesh=None)

    def __getitem__(self, item):
        """Core implementation of the class support for different indexing.

        Parameters
        ----------
        item :

        Returns
        -------

        """
        try:
            item = int(item)
        except ValueError:
            item = self.acronym_to_id_map[item]

        return self.data[int(item)]

    def __repr__(self):
        """String representation of the class, print all regions names
        """
        # TODO consider changing this to printing the tree structure as it's a more concise visualisation
        return "".join(
            ["({}) \t- {}\n".format(k, v["acronym"]) for k, v in self.items()]
        )
