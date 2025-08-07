"""Test functions for listing and managing BrainGlobe atlases."""

from unittest import mock

import pytest
from rich.console import Console
from rich.table import Table

from brainglobe_atlasapi import config
from brainglobe_atlasapi.list_atlases import (
    add_atlas_to_row,
    get_all_atlases_lastversions,
    get_atlases_lastversions,
    get_downloaded_atlases,
    get_local_atlas_version,
    show_atlases,
)


def test_get_downloaded_atlases():
    """Test retrieving a list of downloaded atlases."""
    available_atlases = get_downloaded_atlases()

    # Check that example is listed:
    assert "example_mouse_100um" in available_atlases


def test_get_local_atlas_version_real_atlas():
    """Test getting the version of a real, downloaded atlas."""
    v = get_local_atlas_version("example_mouse_100um")
    assert len(v.split(".")) == 2


def test_get_local_atlas_version_missing_atlas(capsys):
    """Test retrieving the version of a non-existent atlas.

    Parameters
    ----------
    capsys : pytest.CaptureFixture
        Fixture to capture stdout/stderr.
    """
    atlas_name = "unicorn_atlas"
    assert get_local_atlas_version(atlas_name) is None
    captured = capsys.readouterr()
    assert f"No atlas found with the name: {atlas_name}" in captured.out


def test_lastversions():
    """Test retrieving atlas versions from the online source."""
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
    """Test displaying a table of available atlases."""
    # TODO add more valid testing than just look for errors when running:
    show_atlases(show_local_path=True)


def test_get_all_atlases_lastversions():
    """Test retrieving the latest versions of all known atlases."""
    last_versions = get_all_atlases_lastversions()

    assert "example_mouse_100um" in last_versions
    assert "osten_mouse_50um" in last_versions
    assert "allen_mouse_25um" in last_versions


def test_get_all_atlases_custom_atlases(mocker):
    """Check inclusion of available custom atlases in the list of all atlases.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture.
    """
    custom_path = config.get_brainglobe_dir() / "custom_atlases.conf"
    mock_custom_atlas = {"atlases": {"mock_custom_atlas": "1.1"}}

    with mocker.patch(
        "brainglobe_atlasapi.utils.conf_from_file",
        side_effect=lambda file_path: {
            custom_path: mock_custom_atlas,
        }.get(file_path, FileNotFoundError),
    ):
        last_versions = get_all_atlases_lastversions()
        assert last_versions["mock_custom_atlas"] == "1.1"


def test_get_all_atlases_lastversions_offline():
    """Test retrieving atlas versions from cache when offline."""
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
    """Test retrieving atlas versions from cache when GIN is down."""
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


@pytest.mark.parametrize(
    ["version", "expected_print"],
    [
        pytest.param(
            {
                "version": "1",
                "latest_version": "2",
            },
            "│ awesome_name │ ✔ │ x │ 1 │ 2 │",
            id="version != latest_version",
        ),
        pytest.param(
            {
                "version": "1",
                "latest_version": "1",
            },
            "│ awesome_name │ ✔ │ ✔ │ 1 │ 1 │",
            id="version == latest_version",
        ),
    ],
)
def test_add_atlas_to_row(version, expected_print, capsys):
    """Test correct print formatting when atlas versions match or mismatch.

    Parameters
    ----------
    version : dict
        A dictionary containing "version" and "latest_version" strings.
    expected_print : str
        The expected string output in the console.
    capsys : pytest.CaptureFixture
        Fixture to capture stdout/stderr.
    """
    info = {
        "downloaded": True,
        "version": version["version"],
        "latest_version": version["latest_version"],
    }
    table = add_atlas_to_row(atlas="awesome_name", info=info, table=Table())
    Console().print(table)
    captured = capsys.readouterr()
    assert expected_print in captured.out
