import pytest

from brainglobe_atlasapi import update_atlases


def test_update():
    update_atlases.update_atlas(atlas_name="example_mouse_100um")

    update_atlases.update_atlas(atlas_name="example_mouse_100um", force=True)


def test_update_wrong_name():
    with pytest.raises(ValueError) as error:
        update_atlases.update_atlas("allen_madasadsdouse_25um")
    assert "is not a valid atlas name!" in str(error)


def test_update_atlas_value_error(mocker):
    """Test error on old atlas deletion failure."""
    mocker.patch("shutil.rmtree")
    expected_error = "Something went wrong while trying to delete the old"
    " version of the atlas, aborting."
    with pytest.raises(ValueError, match=expected_error):
        update_atlases.update_atlas(
            atlas_name="example_mouse_100um", force=True
        )
