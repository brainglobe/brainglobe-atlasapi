"""
Provide a class for representing hierarchical structures,
such as brain regions in an atlas.
"""

import warnings
from collections import UserDict

import meshio as mio

from brainglobe_atlasapi.structure_tree_util import get_structures_tree


class Structure(UserDict):
    """Class implementing the lazy loading of a mesh if the dictionary is
    queried for it.
    """

    def __getitem__(self, item):
        """
        Retrieve an item from the structure's data.

        If the item is `mesh` and the mesh data is currently None, it attempts
        to load the mesh from the `mesh_filename` if available.

        Parameters
        ----------
        item : str
            The key of the item to retrieve.

        Returns
        -------
        meshio.Mesh or None or any
            - If `item` is "mesh" and the mesh data is successfully loaded,
              returns a `meshio.Mesh` object.
            - If `item` is "mesh" and `mesh_filename` is None, returns `None`.
            - For other keys, returns the value associated with the given item,
              which can be of any type depending on the stored data.

        Raises
        ------
        meshio.ReadError
            If `item` is "mesh" and the mesh cannot be read.
            The value associated with the given item.
        """
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

        self.tree = get_structures_tree(structures_list)

    def __getitem__(self, item):
        """Core implementation of the class support for different indexing.

        Parameters
        ----------
        item : str or int
            The acronym (str) or id (int) of the requested structure.

        Returns
        -------
        Structure
            The Structure requested.
        """
        try:
            item = int(item)
        except ValueError:
            item = self.acronym_to_id_map[item]

        return self.data[int(item)]

    def __repr__(self):
        """Return string representation of the class,
        showing all region names.
        """
        return self.tree.show(stdout=False)
