"""Tests for the atlas update functionality."""

import pytest

from brainglobe_atlasapi import update_atlases


def test_update():
    """Test updating an atlas without and with force flag."""
    update_atlases.update_atlas(atlas_name="example_mouse_100um")

    update_atlases.update_atlas(atlas_name="example_mouse_100um", force=True)


def test_update_wrong_name():
    """Test updating with an invalid atlas name."""
    with pytest.raises(FileNotFoundError) as error:
        update_atlases.update_atlas("allen_madasadsdouse_25um")
    assert "is not a valid atlas name!" in str(error)


def test_update_atlas_value_error(mocker):
    """Test error on old atlas deletion failure.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    mocker.patch("shutil.rmtree")
    expected_error = "Something went wrong while trying to delete the old"
    " version of the atlas, aborting."
    with pytest.raises(ValueError, match=expected_error):
        update_atlases.update_atlas(
            atlas_name="example_mouse_100um", force=True
        )


@pytest.mark.parametrize(
    "input, input_type",
    [
        pytest.param(0, "int"),
        pytest.param(0.1, "float"),
        pytest.param([], "list"),
        pytest.param(None, "NoneType"),
    ],
)
def test_install_atlas(input, input_type):
    """Test correct raising of TypeError when installing an atlas.

    Parameters
    ----------
    input : any
        Invalid input for atlas name.
    input_type : str
        String representation of the input type.
    """
    expected_error = f"Atlas name should be a string, not a {input_type}"
    with pytest.raises(TypeError, match=expected_error):
        update_atlases.install_atlas(atlas_name=input)


def test_install_atlas_istantiate_download(mocker):
    """Test download instantiation when no local atlases are available.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    name = "example_mouse_100um"
    with mocker.patch(
        "brainglobe_atlasapi.update_atlases.get_downloaded_atlases",
        return_value=[],
    ):
        mock_BrainGlobeAtlas = mocker.patch(
            "brainglobe_atlasapi.update_atlases.BrainGlobeAtlas"
        )
        update_atlases.install_atlas(atlas_name=name)

    mock_BrainGlobeAtlas.assert_called_once_with(
        name, fn_update=None, remote_root=None
    )


@pytest.mark.parametrize("function", ["install_atlas", "update_atlas"])
def test_remote_root_reaches_the_atlas(function, mocker):
    """remote_root is forwarded to BrainGlobeAtlas.

    Parameters
    ----------
    function : str
        Name of the update_atlases function under test.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    mocker.patch(
        "brainglobe_atlasapi.update_atlases.get_downloaded_atlases",
        return_value=[],
    )
    mock_BrainGlobeAtlas = mocker.patch(
        "brainglobe_atlasapi.update_atlases.BrainGlobeAtlas"
    )
    # update_atlas stops early when the atlas is already the latest.
    mock_BrainGlobeAtlas.return_value.check_latest_version.return_value = True
    mock_BrainGlobeAtlas.return_value.local_version = (1, 0)

    getattr(update_atlases, function)(
        "some-atlas", remote_root="s3://bucket/prefix"
    )

    kwargs = mock_BrainGlobeAtlas.call_args.kwargs
    assert kwargs["remote_root"] == "s3://bucket/prefix"
