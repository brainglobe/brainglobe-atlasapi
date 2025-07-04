import os
import sys
from typing import Callable
from unittest import mock

import pytest
import requests
import rich.panel
from requests import HTTPError

from brainglobe_atlasapi import descriptors, utils

METADATA = descriptors.METADATA_TEMPLATE

test_url = "https://gin.g-node.org/BrainGlobe/atlases/raw/master/example_mouse_100um_v1.2.tar.gz"
conf_url = (
    "https://gin.g-node.org/BrainGlobe/atlases/raw/master/last_versions.conf"
)


def test_http_check():
    assert utils.check_internet_connection()

    with pytest.raises(ConnectionError) as error:
        utils.check_internet_connection(url="http://asd")

    assert "No internet connection, try again" in str(error)

    assert not utils.check_internet_connection(
        url="http://asd", raise_error=False
    )


def test_get_download_size_bad_url():
    with pytest.raises(IndexError):
        utils.get_download_size(url="http://asd")


def test_get_download_size_no_size_url():
    with pytest.raises(ValueError):
        utils.get_download_size(
            "https://gin.g-node.org/BrainGlobe/atlases/src/5ee75365555e3b4665c685b65a488bca3461ac94/last_versions.conf"
        )


@pytest.mark.parametrize(
    "url, real_size",
    [
        (
            "https://gin.g-node.org/BrainGlobe/atlases/raw/master/example_mouse_100um_v1.2.tar.gz",
            7.3,
        ),
        (
            "https://gin.g-node.org/BrainGlobe/atlases/raw/master/allen_mouse_100um_v1.2.tar.gz",
            61,
        ),
        (
            "https://gin.g-node.org/BrainGlobe/atlases/raw/master/admba_3d_p56_mouse_25um_v1.0.tar.gz",
            335,
        ),
        (
            "https://gin.g-node.org/BrainGlobe/atlases/raw/master/osten_mouse_10um_v1.1.tar.gz",
            3600,
        ),
    ],
)
def test_get_download_size(url, real_size):
    size = utils.get_download_size(url)

    real_size = real_size * 1e6

    assert size == real_size


def test_get_download_size_kb():
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.content = b"asd 24.7 KB 123sd"
        mock_request.return_value = mock_response

        size = utils.get_download_size(test_url)

        assert size == 24700


def test_get_download_size_HTTPError():
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = HTTPError()

        with pytest.raises(HTTPError):
            utils.get_download_size(test_url)


def test_check_gin_status():
    # Test with requests.get returning a valid response
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_response = mock.Mock(spec=requests.Response)
        mock_response.status_code = 200
        mock_request.return_value = mock_response

        assert utils.check_gin_status()


def test_check_gin_status_down():
    # Test with requests.get returning a 404 response
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()

        with pytest.raises(ConnectionError) as e:
            utils.check_gin_status()
            assert "GIN server is down" == e.value


def test_check_gin_status_down_no_error():
    # Test with requests.get returning a 404 response
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()

        assert not utils.check_gin_status(raise_error=False)


def test_conf_from_file(temp_path):
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
    conf_path = temp_path / "conf.conf"

    # Test with a non-existing file
    with pytest.raises(FileNotFoundError) as e:
        utils.conf_from_file(conf_path)

    assert "Last versions cache file not found." == str(e.value)


@pytest.mark.skipif(sys.platform == "win32", reason="Does not run on Windows")
def test_conf_from_url_read_only(temp_path, mocker):
    # Test with a valid URL and a non-existing parent folder
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir"
    ).return_value = temp_path
    mock_print = mocker.patch("builtins.print")
    # Save the current permissions
    curr_mode = oct(os.stat(temp_path).st_mode)[-3:]

    # Change the permissions to read-only
    temp_path.chmod(0o444)
    utils.conf_from_url(conf_url)

    mock_print.assert_called_once_with(
        f"Could not update the latest atlas versions cache: [Errno 13] "
        f"Permission denied: '{temp_path / 'last_versions.conf'}'"
    )

    # Set the permissions back to the original
    temp_path.chmod(int(curr_mode, 8))


def test_conf_from_url_gin_200(temp_path, mocker):
    """Connected to the internet, GIN is available and returns 200"""
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir",
        return_value=temp_path,
    )
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
        config = utils.conf_from_url(conf_url)
        mock_request.assert_called_once_with(conf_url)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_wrong_status_with_cache(temp_path, mocker):
    """Connected to the internet, but GIN is down (404),
    so we revert to local cache of atlas versions"""
    mock_response = mock.patch("requests.Response", autospec=True)
    mock_response.status_code = 404

    with open(temp_path / "last_versions.conf", "w") as f:
        f.write(
            "[atlases]\n"
            "example_mouse_100um = 1.2\n"
            "allen_mouse_10um = 1.2\n"
            "allen_mouse_25um = 1.2"
        )
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir",
        return_value=temp_path,
    )
    with mock.patch(
        "requests.get", autospec=True, return_value=mock_response
    ) as mock_request:
        config = utils.conf_from_url(conf_url)
        mock_request.assert_called_with(conf_url)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_no_connection_with_cache(temp_path, mocker):
    """Not connected to the internet,
    but we can revert to local cache of atlas versions"""
    with open(temp_path / "last_versions.conf", "w") as f:
        f.write(
            "[atlases]\n"
            "example_mouse_100um = 1.2\n"
            "allen_mouse_10um = 1.2\n"
            "allen_mouse_25um = 1.2"
        )
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir",
        return_value=temp_path,
    )
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()
        config = utils.conf_from_url(conf_url)
        assert dict(config["atlases"]) == {
            "example_mouse_100um": "1.2",
            "allen_mouse_10um": "1.2",
            "allen_mouse_25um": "1.2",
        }


def test_conf_from_url_no_connection_no_cache(temp_path, mocker):
    """Not connected to the internet
    and we have no local cache of atlas version"""
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir",
        return_value=temp_path,
    )
    mocker.patch("brainglobe_atlasapi.utils.sleep")
    with mock.patch("requests.get", autospec=True) as mock_request:
        mock_request.side_effect = requests.ConnectionError()
        with pytest.raises(FileNotFoundError) as e:
            utils.conf_from_url(conf_url)
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
    """Tests atlas name conversion for rich panel."""
    panel = utils._rich_atlas_metadata(atlas_name=name, metadata=METADATA)
    assert panel.renderable.title == title


def test_rich_atlas_metadata_type():
    """Tests right data type is created"""
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
                    "resolution": "1um",
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
                    "resolution": "1nm",
                },
            },
            id="axolotl_1nm",
        ),
    ]
)
def name_repr(request):
    """Fixture with atlas name and representation pairs."""
    return request.param


def test_atlas_repr_from_name(name_repr):
    """Test atlas name to repr conversion."""
    name = name_repr["name"]
    expected_repr = name_repr["repr"]
    assert utils.atlas_repr_from_name(name) == expected_repr


def test_atlas_name_from_repr(name_repr):
    """Test atlas repr to name conversion."""
    expected_name = name_repr["name"]
    repr = name_repr["repr"]
    assert utils.atlas_name_from_repr(**repr) == expected_name


def test_retrieve_over_http_ConnectionError(tmp_path):
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
    """Test handling of fn_update when not None."""
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
    """Test creating a directory if it does not exist."""
    mock_cache_path = tmp_path / "parent" / "file"
    mocker.patch(
        "brainglobe_atlasapi.utils.config.get_brainglobe_dir",
        return_value=mock_cache_path,
    )
    assert not mock_cache_path.exists()
    utils.conf_from_url(conf_url)
    assert mock_cache_path.parent.exists()
