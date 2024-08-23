from unittest import mock

from brainglobe_atlasapi import config
from brainglobe_atlasapi.list_atlases import (
    get_all_atlases_lastversions,
    get_atlases_lastversions,
    get_downloaded_atlases,
    get_local_atlas_version,
    show_atlases,
)


def test_get_downloaded_atlases():
    available_atlases = get_downloaded_atlases()

    # Check that example is listed:
    assert "example_mouse_100um" in available_atlases


def test_get_local_atlas_version():
    v = get_local_atlas_version("example_mouse_100um")

    assert len(v.split(".")) == 2


def test_lastversions():
    last_versions = get_atlases_lastversions()
    example_atlas = last_versions["example_mouse_100um"]

    local_v = get_local_atlas_version("example_mouse_100um")

    assert example_atlas["version"] == local_v
    assert all(
        [
            int(last) <= int(r)
            for last, r in zip(
                example_atlas["latest_version"].split("."), local_v.split(".")
            )
        ]
    )


def test_show_atlases():
    # TODO add more valid testing than just look for errors when running:
    show_atlases(show_local_path=True)


def test_get_all_atlases_lastversions():
    last_versions = get_all_atlases_lastversions()

    assert "example_mouse_100um" in last_versions
    assert "osten_mouse_50um" in last_versions
    assert "allen_mouse_25um" in last_versions


def test_get_all_atlases_lastversions_offline():
    cleanup_cache = False
    cache_path = config.get_brainglobe_dir() / "last_versions.conf"

    if not cache_path.exists():
        cache_path.touch()
        cache_path.write_text(
            """
            [atlases]
            example_mouse_100um = 1.0
            osten_mouse_50um = 1.0
            allen_mouse_25um = 1.0
            """
        )
        cleanup_cache = True

    with mock.patch(
        "brainglobe_atlasapi.utils.check_internet_connection"
    ) as mock_check_internet_connection:
        mock_check_internet_connection.return_value = False
        last_versions = get_all_atlases_lastversions()

        assert "example_mouse_100um" in last_versions
        assert "osten_mouse_50um" in last_versions
        assert "allen_mouse_25um" in last_versions

    if cleanup_cache:
        cache_path.unlink()


def test_get_all_atlases_lastversions_gin_down():
    cleanup_cache = False
    cache_path = config.get_brainglobe_dir() / "last_versions.conf"

    if not cache_path.exists():
        cache_path.touch()
        cache_path.write_text(
            """
            [atlases]
            example_mouse_100um = 1.0
            osten_mouse_50um = 1.0
            allen_mouse_25um = 1.0
            """
        )
        cleanup_cache = True

    with mock.patch(
        "brainglobe_atlasapi.utils.check_gin_status"
    ) as mock_check_internet_connection:
        mock_check_internet_connection.return_value = False
        last_versions = get_all_atlases_lastversions()

        assert "example_mouse_100um" in last_versions
        assert "osten_mouse_50um" in last_versions
        assert "allen_mouse_25um" in last_versions

    if cleanup_cache:
        cache_path.unlink()
