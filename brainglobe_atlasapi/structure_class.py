"""
Provide a class for representing hierarchical structures,
such as brain regions in an atlas.
"""

import json
import os
import warnings
from collections import UserDict
from io import BytesIO
from pathlib import Path

import DracoPy
import meshio as mio
import numpy as np
import s3fs
from fsspec.callbacks import TqdmCallback

from brainglobe_atlasapi.descriptors import DEFAULT_REMOTE_ROOT
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def _read_legacy_mesh(fs, manifest_path: str) -> tuple:
    """Read a `neuroglancer_legacy_mesh` manifest and its fragments.

    The atlas-assets bucket stores each structure as a JSON manifest at
    `<id>:0` listing fragment object names, each of which is a raw legacy
    mesh blob. Fragment names are full object names, resolved against the
    directory holding the manifest.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem exposing a single-path ``cat(path) -> bytes``.
    manifest_path : str
        Full remote path of the `<id>:0` manifest.

    Returns
    -------
    tuple of numpy.ndarray
        `(points, faces)`; float32 (N, 3) vertices in the units and axis
        order of the source bucket, and uint32 (M, 3) triangle indices
        offset so they index into the concatenated vertex array.
    """
    remote_dir = manifest_path.rsplit("/", 1)[0]
    fragments = json.loads(fs.cat(manifest_path))["fragments"]

    all_points, all_faces, offset = [], [], 0
    for fragment in fragments:
        mesh = mio.read(
            BytesIO(fs.cat(f"{remote_dir}/{fragment}")),
            file_format="neuroglancer",
        )
        all_points.append(mesh.points)
        all_faces.append(mesh.cells[0].data + offset)
        offset += len(mesh.points)

    points = np.concatenate(all_points).astype(np.float32)
    faces = np.concatenate(all_faces).astype(np.uint32)
    return points, faces


def _encode_draco(points: np.ndarray, faces: np.ndarray) -> bytes:
    """Encode a mesh as a single Draco fragment.

    Mirrors `atlas_generation.mesh_utils.write_mesh`: 16-bit quantization
    against the bounding *cube*, because Draco quantizes every axis
    against one uniform range and a per-axis range would distort.

    Parameters
    ----------
    points : numpy.ndarray
        (N, 3) vertices.
    faces : numpy.ndarray
        (M, 3) triangle indices.

    Returns
    -------
    bytes
        The Draco-encoded fragment, byte-for-byte what a BrainGlobe atlas
        stores at `<id>`.
    """
    vertices = np.ascontiguousarray(points, dtype=np.float32)
    triangles = np.ascontiguousarray(faces, dtype=np.uint32)

    bbox_min = vertices.min(axis=0)
    qrange = float((vertices.max(axis=0) - bbox_min).max())
    if qrange <= 0.0:  # degenerate (single point / planar in all axes)
        qrange = 1.0

    return DracoPy.encode(
        vertices,
        triangles,
        quantization_bits=16,
        compression_level=0,
        quantization_range=qrange,
        quantization_origin=bbox_min.tolist(),
    )


class Structure(UserDict):
    """Class implementing the lazy loading of a mesh if the dictionary is
    queried for it.

    Attributes
    ----------
    remote_root : str
        Remote root the mesh is fetched from. Set per instance by
        `StructuresDict`; defaults to the BrainGlobe bucket.
    """

    remote_root = DEFAULT_REMOTE_ROOT

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
        """Download the mesh from the remote S3 bucket if it is not cached.

        BrainGlobe buckets store a Draco fragment at `<id>`, atlas-assets
        buckets a `neuroglancer_legacy_mesh` manifest at `<id>:0` plus its
        fragments. The legacy form is converted on the way in, so the local
        cache holds `<id>` either way.
        """
        root_path = "/".join(str(file_name).split(os.sep)[-6:])
        remote_mesh_path = f"{self.remote_root}/{root_path}"
        fs = s3fs.S3FileSystem(anon=True)

        legacy_manifest_path = f"{remote_mesh_path}:0"
        if fs.exists(remote_mesh_path):
            convert = False
        elif fs.exists(legacy_manifest_path):
            convert = True
        else:
            raise FileNotFoundError(
                f"Mesh file {file_name} not found locally or remotely."
            )

        try:
            if convert:
                points, faces = _read_legacy_mesh(fs, legacy_manifest_path)
                file_name.write_bytes(_encode_draco(points, faces))
            else:
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
    remote_root : str or None
        Remote root the meshes are fetched from; defaults to the
        BrainGlobe bucket.
    """

    def __init__(self, structures_list, remote_root=None):
        super().__init__()

        # Acronym to id map:
        self.acronym_to_id_map = {
            r["acronym"]: r["id"] for r in structures_list
        }

        for struct in structures_list:
            sid = struct["id"]
            structure = Structure(**struct, mesh=None)
            structure.remote_root = remote_root or DEFAULT_REMOTE_ROOT
            self.data[sid] = structure

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
