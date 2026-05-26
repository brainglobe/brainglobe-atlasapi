"""Tests for utility functions in brainglobe_atlasapi."""

import os
import sys
from typing import Callable
from unittest import mock

import pytest
import requests
import rich.panel

from brainglobe_atlasapi import descriptors, utils

METADATA = descriptors.METADATA_TEMPLATE

conf_url = descriptors.remote_url_s3_http.format("atlases/last_versions.conf")


def test_http_check():
    """Test internet connection check utility."""
    assert utils.check_internet_connection()

    with pytest.raises(ConnectionError) as error:
        utils.check_internet_connection(url="http://asd")

    assert "No internet connection, try again" in str(error)

    assert not utils.check_internet_connection(
        url="http://asd", raise_error=False
    )


def test_conf_from_file(temp_path):
    """Test reading configuration from a file.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    """
    conf_path = temp_path / "conf.conf"
    content = (
        "[atlases]\n"
        "example_mouse_100um = 1.2\n"
        "allen_mouse_10um = 1.2\n"
        "allen_mouse_25um = 1.2"
    )
    conf_path.write_text(content)
    # Test with a valid file
    conf = utils.conf_from_file(conf_path)

    assert dict(conf["atlases"]) == {
        "example_mouse_100um": "1.2",
        "allen_mouse_10um": "1.2",
        "allen_mouse_25um": "1.2",
    }


def test_conf_from_file_no_file(temp_path):
    """Test reading configuration when the file does not exist.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    """
    conf_path = temp_path / "conf.conf"

    # Test with a non-existing file
    with pytest.raises(FileNotFoundError) as e:
        utils.conf_from_file(conf_path)

    assert "Last versions cache file not found." == str(e.value)


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_conf_from_url_read_only(temp_path, mocker):
    """Test reading configuration from URL with read-only permissions on
    cache directory.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    # Test with a valid URL and a non-existing parent folder
    cache_path = temp_path / "last_versions.conf"
    mock_print = mocker.patch("builtins.print")
    # Save the current permissions
    curr_mode = oct(os.stat(temp_path).st_mode)[-3:]

    # Change the permissions to read-only
    temp_path.chmod(0o444)
    utils.conf_from_url(conf_url, cache_path)

    mock_print.assert_called_once_with(
        f"Could not update the latest atlas versions cache: [Errno 13] "
        f"Permission denied: '{temp_path / 'last_versions.conf'}'"
    )

    # Set the permissions back to the original
    temp_path.chmod(int(curr_mode, 8))


def test_conf_from_url_gin_200(temp_path, mocker):
    """Test reading configuration from URL when GIN returns 200.

    Connected to the internet, GIN is available and returns 200.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    cache_path = temp_path / "last_versions.conf"
    mock_response = mock.patch("requests.Response", autospec=True)
    mock_response.status_code = 200
    mock_response.text = (
        "[atlases]\n"
        "example_mouse_100um = 1.2\n"
        "allen_mouse_10um = 1.2\n"
        "allen_mouse_25um = 1.2"
    )
    with mock.patch(
        "requests.get", autospec=True, return_value=mock_response
    ) as mock_request:
        config = utils.conf_from_url(conf_url, cache_path)
        mock_request.assert_called_once_with(conf_url)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_wrong_status_with_cache(temp_path, mocker):
    """Test reading configuration from URL when GIN returns wrong status
    but local cache is available.

    Connected to the internet, but GIN is down (404),
    so revert to local cache of atlas versions.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    mock_response = mock.patch("requests.Response", autospec=True)
    mock_response.status_code = 404

    with open(temp_path / "last_versions.conf", "w") as f:
        f.write(
            "[atlases]\n"
            "example_mouse_100um = 1.2\n"
            "allen_mouse_10um = 1.2\n"
            "allen_mouse_25um = 1.2"
        )

    cache_path = temp_path / "last_versions.conf"

    with mock.patch(
        "requests.get", autospec=True, return_value=mock_response
    ) as mock_request:
        config = utils.conf_from_url(conf_url, cache_path)
        mock_request.assert_called_with(conf_url)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_no_connection_with_cache(temp_path, mocker):
    """Test reading configuration from URL when no connection but local
    cache is available.

    Not connected to the internet,
    but can revert to local cache of atlas versions.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    with open(temp_path / "last_versions.conf", "w") as f:
        f.write(
            "[atlases]\n"
            "example_mouse_100um = 1.2\n"
            "allen_mouse_10um = 1.2\n"
            "allen_mouse_25um = 1.2"
        )

    cache_path = temp_path / "last_versions.conf"
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()
        config = utils.conf_from_url(conf_url, cache_path)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_no_connection_no_cache(temp_path, mocker):
    """Test reading configuration from URL when no connection and no
    local cache.

    Not connected to the internet
    and have no local cache of atlas version.

    Parameters
    ----------
    temp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    cache_path = temp_path / "last_versions.conf"
    mocker.patch("brainglobe_atlasapi.utils.sleep")
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()
        with pytest.raises(FileNotFoundError) as e:
            utils.conf_from_url(conf_url, cache_path)
        assert "Last versions cache file not found." == str(e.value)


@pytest.mark.parametrize(
    ["name", "title"],
    [
        pytest.param(
            "brainglobe",
            "Brainglobe",
            id="capitalisation of first letter",
        ),
        pytest.param(
            "BrainGlobe",
            "Brainglobe",
            id="decapitalisation of everything but first charachter",
        ),
        pytest.param(
            "___Brain_globe___",
            "   brain globe   ",
            id="underscores become spaces",
        ),
        pytest.param(
            "",
            "",
            id="no name",
        ),
    ],
)
def test_rich_atlas_metadata_table_title(name, title):
    """Test atlas name conversion for rich panel title.

    Parameters
    ----------
    name : str
        The input atlas name.
    title : str
        The expected title for the rich panel.
    """
    panel = utils._rich_atlas_metadata(atlas_name=name, metadata=METADATA)
    assert panel.renderable.title == title


def test_rich_atlas_metadata_type():
    """Test correct data type is created for rich atlas metadata panel."""
    panel = utils._rich_atlas_metadata(
        atlas_name=METADATA["name"],
        metadata=METADATA,
    )
    assert isinstance(panel, rich.panel.Panel)


@pytest.fixture(
    params=[
        pytest.param(
            {
                "name": "kim_dev_mouse_e15-5_mri-adc_37.5um_v1.3",
                "repr": {
                    "name": "kim_dev_mouse_e15-5_mri-adc",
                    "major_vers": "1",
                    "minor_vers": "3",
                    "resolution": "37.5",
                    "unit": "um",
                },
            },
            id="kim_dev_mouse_e15-5_mri-adc_37.5um_v1.3",
        ),
        pytest.param(
            {
                "name": "axolotl_1um",
                "repr": {
                    "name": "axolotl",
                    "major_vers": None,
                    "minor_vers": None,
                    "resolution": "1",
                    "unit": "um",
                },
            },
            id="axolotl_1um",
        ),
        pytest.param(
            {
                "name": "axolotl_1mm_v5.2",
                "repr": {
                    "name": "axolotl",
                    "major_vers": "5",
                    "minor_vers": "2",
                    "resolution": "1",
                    "unit": "mm",
                },
            },
            id="axolotl_1mm_v5.2",
        ),
        pytest.param(
            {
                "name": "axolotl_1nm",
                "repr": {
                    "name": "axolotl",
                    "major_vers": None,
                    "minor_vers": None,
                    "resolution": "1",
                    "unit": "nm",
                },
            },
            id="axolotl_1nm",
        ),
    ]
)
def name_repr(request):
    """Provide atlas name and representation pairs.

    Returns
    -------
    dict
        A dictionary containing the atlas name and its expected representation.
    """
    return request.param


def test_atlas_repr_from_name(name_repr):
    """Test atlas name to representation conversion.

    Parameters
    ----------
    name_repr : dict
        Fixture with atlas name and representation pairs.
    """
    name = name_repr["name"]
    expected_repr = name_repr["repr"]
    assert utils.atlas_repr_from_name(name) == expected_repr


def test_atlas_name_from_repr(name_repr):
    """Test atlas representation to name conversion.

    Parameters
    ----------
    name_repr : dict
        Fixture with atlas name and representation pairs.
    """
    expected_name = name_repr["name"]
    repr = name_repr["repr"]
    assert utils.atlas_name_from_repr(**repr) == expected_name


def test_retrieve_over_http_ConnectionError(tmp_path):
    """Test ConnectionError handling in retrieve_over_http.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path for test files.
    """
    with mock.patch(
        "requests.get",
        side_effect=requests.exceptions.ConnectionError,
    ):
        with pytest.raises(
            requests.exceptions.ConnectionError,
            match="Could not download file from elephants",
        ):
            utils.retrieve_over_http(
                url="elephants",
                output_file_path=tmp_path / "elephant",
            )


@pytest.mark.parametrize(
    "content_length, expected_last_call",
    [pytest.param("6", (6, 6), id="6/6"), pytest.param(0, (6, 0), id="6/6")],
)
def test_retrieve_over_http_fn_update(
    tmp_path, content_length, expected_last_call
):
    """Test handling of fn_update in retrieve_over_http when not None.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path for test files.
    content_length : str or int
        The value for the 'content-length' header.
    expected_last_call : tuple
        The expected arguments for the last call to fn_update.
    """
    mock_fn_update: Callable[[int, int]] = mock.Mock()

    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.iter_content.return_value = [
            b"1",  # First chunk of data (1 byte)
            b"11",  # Second chunk of data (2 bytes)
            b"111",  # Third chunk of data (3 bytes)
        ]
        mock_response.headers = {"content-length": content_length}
        mock_request.return_value = mock_response

        with mock.patch(
            "brainglobe_atlasapi.utils.get_download_size",
            side_effect=Exception if content_length == 0 else None,
        ):
            utils.retrieve_over_http(
                url="elephants",
                output_file_path=tmp_path / "elephant",
                fn_update=mock_fn_update,
            )

    mock_fn_update.assert_called_with(*expected_last_call)
    assert mock_fn_update.call_count == 3


def test_conf_from_url_no_cache_path_parent(tmp_path, mocker):
    """Test creating a directory if it does not exist when getting conf
    from URL.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path for test files.
    mocker : pytest_mock.plugin.MockerFixture
        Mocker fixture for patching.
    """
    mock_cache_path = tmp_path / "parent" / "file" / "last_versions.conf"

    assert not mock_cache_path.exists()
    utils.conf_from_url(conf_url, cache_path=mock_cache_path)
    assert mock_cache_path.parent.exists()
