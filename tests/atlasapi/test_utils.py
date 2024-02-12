from unittest import mock

import pytest
import requests
from requests import HTTPError

from bg_atlasapi import utils

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
            "https://gin.g-node.org/BrainGlobe/atlases/src/master/last_versions.conf"
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
