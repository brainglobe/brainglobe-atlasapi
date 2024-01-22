from pathlib import Path

import numpy as np
import pytest
from bg_atlasapi import BrainGlobeAtlas
from bg_atlasapi.config import get_brainglobe_dir

from bg_atlasgen.validate_atlases import (
    _assert_close,
    validate_atlas_files,
    validate_mesh_matches_image_extents,
)


def test_validate_mesh_matches_image_extents():
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    assert validate_mesh_matches_image_extents(atlas)


def test_validate_mesh_matches_image_extents_negative(mocker):
    atlas = BrainGlobeAtlas("allen_mouse_100um")
    flipped_annotation_image = np.transpose(atlas.annotation)
    mocker.patch(
        "bg_atlasapi.BrainGlobeAtlas.annotation",
        new_callable=mocker.PropertyMock,
        return_value=flipped_annotation_image,
    )
    with pytest.raises(
        AssertionError, match="differ by more than 10 times pixel size"
    ):
        validate_mesh_matches_image_extents(atlas)


def test_valid_atlas_files():
    _ = BrainGlobeAtlas("allen_mouse_100um")
    atlas_path = Path(get_brainglobe_dir()) / "allen_mouse_100um_v1.2"
    assert validate_atlas_files(atlas_path)


def test_invalid_atlas_path():
    atlas_path = Path.home()
    with pytest.raises(AssertionError, match="Expected file not found"):
        validate_atlas_files(atlas_path)


def test_assert_close():
    assert _assert_close(99.5, 8, 10)


def test_assert_close_negative():
    with pytest.raises(
        AssertionError, match="differ by more than 10 times pixel size"
    ):
        _assert_close(99.5, 30, 2)
