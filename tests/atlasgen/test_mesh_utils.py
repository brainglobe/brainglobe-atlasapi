import numpy as np

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    extract_mesh_from_mask,  # TODO: is this function being used?
)


def test_extract_mesh_from_mask():
    volume = np.zeros((3, 3, 3), dtype=int)
    volume[1, 1, 1] = 1
    kwargs = {
        "volume": volume,
        "obj_filepath": None,
        "threshold": 0.5,
        "smooth": False,
        "mcubes_smooth": False,
        "closing_n_iters": None,  # default is 8
        "decimate_fraction": 0.6,
        "use_marching_cubes": False,
        "extract_largest": False,
    }

    mesh = extract_mesh_from_mask(**kwargs)

    assert np.isclose(mesh.area(), 1.73205)  #  the surface area of the mesh.
    assert np.isclose(mesh.volume(), 1 / 6)
    assert np.isclose(mesh.average_size(), 0.5)
    assert mesh.contains([1, 1, 1]) is True
    assert mesh.contains([2, 2, 2]) is False
