import pytest

from bg_atlasapi import utils


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
