from unittest.mock import patch

import numpy as np
import pytest

from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    extract_mesh_from_mask,
)


@pytest.fixture
def mesh_from_mask():
    """Fixture with volume and default parameters for `mesh_from_mask`.

    The `volume` is a 100 unit cube of zeros with a centered 50 unit cube of
    ones in the middle.
    """
    volume = np.zeros((100, 100, 100), dtype=int)
    volume[24:75, 24:75, 24:75] = 1

    return {
        "volume": volume,
        "obj_filepath": None,
        "threshold": 0.5,
        "smooth": False,
        "mcubes_smooth": False,
        "closing_n_iters": 8,  # default
        "decimate_fraction": 0.6,
        "use_marching_cubes": False,
        "extract_largest": False,
    }


@pytest.mark.parametrize("obj_filepath_is_str", [True, False])
def test_extract_mesh_from_mask_object_filepath_str(
    mesh_from_mask, tmp_path, obj_filepath_is_str
):
    """Test conversion to path when `obj_filepath` is a string."""
    obj_filepath = tmp_path / "mesh"
    if obj_filepath_is_str:
        obj_filepath = str(obj_filepath)
    mesh_from_mask.update({"obj_filepath": obj_filepath})

    with patch(
        "brainglobe_atlasapi.atlas_generation.mesh_utils.Path"
    ) as mock_path:
        extract_mesh_from_mask(**mesh_from_mask)
        if obj_filepath_is_str:
            mock_path.assert_called_once_with(obj_filepath)
        else:
            mock_path.assert_not_called()


def test_extract_mesh_from_mask_none_existing_parent(mesh_from_mask, tmp_path):
    """Test handling of obj filepath with missing parent"""
    obj_filepath = tmp_path / "non_existing_parent" / "mesh"
    mesh_from_mask.update({"obj_filepath": obj_filepath})
    match = "The folder where the .obj file is to be saved doesn't exist"
    with pytest.raises(FileExistsError, match=match):
        extract_mesh_from_mask(**mesh_from_mask)


def test_extract_mesh_from_mask_defaults(mesh_from_mask):
    """Test extract_mesh_from_mask with defaults"""
    mesh = extract_mesh_from_mask(**mesh_from_mask)
    assert mesh.contains([50, 50, 50]) is True
    assert mesh.contains([2, 2, 2]) is False


def test_extract_mesh_from_mask_ValueError(mesh_from_mask):
    """Test ValueError for non-binary volume in `extract_mesh_from_mask`."""
    volume = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    mesh_from_mask.update({"volume": volume})
    with pytest.raises(ValueError, match="volume should be a binary mask"):
        extract_mesh_from_mask(**mesh_from_mask)


# TODO: check what smooth is supposed to be, see point 9 in issue #541
@pytest.mark.xfail
@pytest.mark.parametrize("mcubes_smooth", [False, True])
def test_extract_mesh_from_mask_marching_cubes(
    mcubes_smooth, mesh_from_mask, capsys
):
    """Test mesh_from_mask using marching cubes with/without mcubes_smooth."""
    mesh_from_mask.update({"use_marching_cubes": True})
    mesh_from_mask.update({"mcubes_smooth": mcubes_smooth})
    extract_mesh_from_mask(**mesh_from_mask)
    captured = capsys.readouterr()
    assert captured.out.startswith(
        "The marching cubes algorithm might be rotated "
    )


# TODO: check what is expected when extract_largest=True
@pytest.mark.xfail
@pytest.mark.parametrize("extract_largest", [False, True])
def test_extract_largest_mesh_from_mask(extract_largest, mesh_from_mask):
    """Test `extract_largest` during mesh from mask extraction.

    Tests mesh volume when extract_largest parameter set to True / False.

    `expected_mesh_largest_true` is created using the volume mask from the
    `mesh_from_mask` fixture, which contains one large region.

    `expected_mesh_largest_false` is created using a volume mask containing
    the original large region and an additional smaller region (with default
    extract_largest = False)

    `mesh` is created using the volume mask with both regions, setting
    extract_largest to True / False.

    The assertions check that the mesh area matches the expected values based
    on the `extract_largest` parameter.
    """

    # using the origingal volume containing only the large region
    expected_mesh_largest_true = extract_mesh_from_mask(**mesh_from_mask)

    volume = np.zeros((100, 100, 100), dtype=int)
    volume[24:75, 24:75, 24:75] = 1  # Large region
    volume[4:10, 4:10, 4:10] = 1  # Small region

    mesh_from_mask.update({"volume": volume})
    expected_mesh_largest_false = extract_mesh_from_mask(**mesh_from_mask)

    # parametrize extract_largest to False / True
    mesh_from_mask.update({"extract_largest": extract_largest})
    mesh = extract_mesh_from_mask(**mesh_from_mask)

    if extract_largest is True:
        assert np.isclose(mesh.area(), expected_mesh_largest_true.area())
    else:
        assert np.isclose(
            mesh.area(),
            expected_mesh_largest_false.area(),
        )


@pytest.mark.parametrize("zeros_ones", [0, 1])
def test_mesh_from_mask_only_zeros_or_ones(zeros_ones, mesh_from_mask):
    """Test mesh extraction from masks containing only zeros or only ones."""
    volume = np.full((100, 100, 100), zeros_ones, dtype=int)
    mesh_from_mask.update({"volume": volume})
    mesh = extract_mesh_from_mask(**mesh_from_mask)
    if zeros_ones == 0:
        assert np.isclose(mesh.area(), 0)
    else:
        assert mesh.area() > 0
