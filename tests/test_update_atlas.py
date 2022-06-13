import pytest

from bg_atlasapi import update_atlases


def test_update():
    update_atlases.update_atlas(atlas_name="example_mouse_100um")

    update_atlases.update_atlas(atlas_name="example_mouse_100um", force=True)


def test_update_wrong_name():
    with pytest.raises(ValueError) as error:
        update_atlases.update_atlas("allen_madasadsdouse_25um")
    assert "is not a valid atlas name!" in str(error)
