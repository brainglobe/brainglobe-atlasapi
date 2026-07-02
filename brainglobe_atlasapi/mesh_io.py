"""Mesh I/O functions for handling Draco mesh files."""

import json
from pathlib import Path
from typing import List

import DracoPy
import meshio
import numpy as np
from cloudvolume.datasource.precomputed.mesh.multilod import (
    MultiLevelPrecomputedMeshManifest,
)


def write_mesh_info(
    mesh_dir: Path,
    vertex_quantization_bits: int = 16,
    transform: List[int] = None,
    lod_scale_multiplier: float = 1.0,
) -> dict:
    """
    Write the mesh-directory metadata file.

    Parameters
    ----------
    mesh_dir : Path
        Path to the mesh directory where the `info` file will be written.
    vertex_quantization_bits : int, optional
        Number of bits used for vertex quantization (default is 16).
    transform : List[int], optional
        A 4x3 transformation matrix in row-major order (default is identity).
    lod_scale_multiplier : float, optional
        Scale multiplier for level-of-detail (LOD) (default is 1.0).

    Returns
    -------
    dict
        The metadata dictionary that was written to the `info` file.
    """
    if transform is None:
        transform = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]

    info = {
        "@type": "neuroglancer_multilod_draco",
        "vertex_quantization_bits": int(vertex_quantization_bits),
        "transform": [float(x) for x in transform],
        "lod_scale_multiplier": float(lod_scale_multiplier),
    }
    (mesh_dir / "info").write_text(json.dumps(info, indent=2))
    return info


def write_mesh(
    mesh: meshio.Mesh,
    mesh_dir: Path,
    segment_id: int,
    vertex_quantization_bits: int = 16,
    compression_level: int = 0,
):
    """Write one object as a single-LOD, single-fragment multi-res mesh.

    Parameters
    ----------
    mesh_dir : path to the mesh directory (gets `info`, `<id>`, `<id>.index`)
    segment_id : int label of the object
    vertices : (N, 3) float array of xyz coordinates ("points")
    faces : (M, 3) int array of triangle vertex indices
    vertex_quantization_bits : 10 or 16 (Neuroglancer only allows these two)
    compression_level : 0-10 (higher = more compression)
    """
    vertices = np.ascontiguousarray(mesh.points, dtype=np.float32)
    faces = np.ascontiguousarray(mesh.cells[0].data, dtype=np.uint32)

    # Define the single octree cell that holds the whole mesh.
    # Draco quantizes ALL axes against one uniform range (a bounding *cube*).
    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    qrange = float((bbox_max - bbox_min).max())
    if qrange <= 0.0:  # degenerate (single point / planar in all axes)
        qrange = 1.0

    grid_origin = bbox_min.astype(np.float32)

    # Encode the one fragment
    fragment = DracoPy.encode(
        vertices,
        faces,
        quantization_bits=vertex_quantization_bits,
        compression_level=compression_level,
        quantization_range=qrange,
        quantization_origin=grid_origin.tolist(),
    )

    manifest = MultiLevelPrecomputedMeshManifest(
        segment_id=int(segment_id),
        chunk_shape=np.array([qrange, qrange, qrange], dtype=np.float32),
        grid_origin=grid_origin,
        num_lods=1,
        lod_scales=np.array([qrange], dtype=np.float32),
        vertex_offsets=np.zeros((1, 3), dtype=np.float32),
        num_fragments_per_lod=np.array([1], dtype=np.uint32),
        fragment_positions=[np.zeros((1, 3), dtype=np.uint32)],
        fragment_offsets=[np.array([len(fragment)], dtype=np.uint32)],
    )

    (mesh_dir / str(segment_id)).write_bytes(fragment)
    (mesh_dir / f"{int(segment_id)}.index").write_bytes(manifest.to_binary())


def read_mesh(mesh_path: Path) -> meshio.Mesh:
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

    return meshio.Mesh(
        points=points,
        cells=[("triangle", faces)],
    )
