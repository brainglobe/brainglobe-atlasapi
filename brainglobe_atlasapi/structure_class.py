"""
Provide a class for representing hierarchical structures,
such as brain regions in an atlas.
"""

import os
import warnings
from collections import UserDict
from pathlib import Path

import DracoPy
import meshio as mio
import s3fs
from fsspec.callbacks import TqdmCallback

from brainglobe_atlasapi.descriptors import remote_url_s3
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
            file_name = self.data["mesh_filename"]
            if file_name is None:
                warnings.warn(
                    "No mesh filename for region {}".format(
                        self.data["acronym"]
                    )
                )
                return None
            try:
                if not file_name.exists():
                    self._download_mesh(file_name)

                self.data[item] = self._read_mesh(file_name)
            except (
                TypeError,
                mio.ReadError,
                FileNotFoundError,
                DracoPy.FileTypeException,
            ) as e:
                raise RuntimeError(
                    f"Failed to read mesh for region {self.data['acronym']} "
                    f"from file {file_name}: {e}"
                ) from e

        return self.data[item]

    def _download_mesh(self, file_name: Path) -> None:
        """Download the mesh from the remote S3 bucket if it is not cached."""
        root_path = "/".join(str(file_name).split(os.sep)[-6:])
        remote_mesh_path = remote_url_s3.format(root_path)
        fs = s3fs.S3FileSystem(anon=True)

        if not fs.exists(remote_mesh_path):
            raise FileNotFoundError(
                f"Mesh file {file_name} not found locally or remotely."
            )

        try:
            fs.get(remote_mesh_path, file_name, callback=TqdmCallback())
        except BaseException:
            file_name.unlink(missing_ok=True)  # Removes corrupt file
            raise

    @staticmethod
    def _read_mesh(mesh_path: Path) -> mio.Mesh:
        """
        Read one object back into (vertices, faces).

        Re-orient from XYZ to ZYX and scale from nm to um.

        Returns
        -------
        meshio.Mesh
            The mesh object reoriented and scaled.
        """
        with open(mesh_path, "rb") as f:
            mesh = DracoPy.decode(f.read())

        points = mesh.points / 1000.0  # scale from nm to um
        points = points[:, [2, 1, 0]]  # reorient from XYZ to ZYX
        faces = mesh.faces[:, [2, 1, 0]]  # reorient from XYZ to ZYX

        return mio.Mesh(
            points=points,
            cells=[("triangle", faces)],
        )


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
