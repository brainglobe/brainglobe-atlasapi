from unittest import mock

import pytest
import requests
from requests import HTTPError

from brainglobe_atlasapi import utils

test_url = "https://gin.g-node.org/BrainGlobe/atlases/raw/master/example_mouse_100um_v1.2.tar.gz"


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

        assert "Last versions cache file not found." == str(e)
