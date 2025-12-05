"""Test functions for listing and managing BrainGlobe atlases."""

from unittest import mock
from pathlib import Path

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


@pytest.fixture
def mock_atlas_dir(tmp_path, mocker):
    """
    Creates a mock atlas directory and patches get_brainglobe_dir
    to point to the temporary location.
    """
    # 1. Create the mock brainglobe directory structure
    bg_dir = tmp_path / ".brainglobe"
    bg_dir.mkdir()
    
    # 2. Create the mock atlas directory with a version tag
    atlas_name_versioned = "example_mouse_100um_v1.0"
    atlas_dir = bg_dir / atlas_name_versioned
    atlas_dir.mkdir()
    
    # 3. Patch the function that returns the brainglobe directory
    mocker.patch(
        "brainglobe_atlasapi.config.get_brainglobe_dir",
        return_value=bg_dir
    )
    
    # Return the name of the atlas for use in tests
    return "example_mouse_100um"


def test_get_downloaded_atlases(mock_atlas_dir):
    """Test retrieving a list of downloaded atlases."""
    available_atlases = get_downloaded_atlases()

    # Check that example is listed:
    assert mock_atlas_dir in available_atlases


def test_get_local_atlas_version_real_atlas(mock_atlas_dir):
    """Test getting the version of a real, downloaded atlas."""
    v = get_local_atlas_version(mock_atlas_dir)
    assert v == "1.0"


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


def test_lastversions(mock_atlas_dir, mocker):
    """Test retrieving atlas versions from the online source."""
    
    # Mock the remote/cached official versions to include our mock atlas
    mock_official_atlases = {"atlases": {mock_atlas_dir: "1.0"}}
    
    # Patch utils.conf_from_url to return our mock data
    mocker.patch(
        "brainglobe_atlasapi.list_atlases.utils.conf_from_url",
        return_value=mock_official_atlases,
    )
    # Patch utils.conf_from_file for the cached official versions
    mocker.patch(
        "brainglobe_atlasapi.list_atlases.utils.conf_from_file",
        side_effect=lambda file_path: (
            mock_official_atlases
            if file_path.name == "last_versions.conf"
            # FIX APPLIED HERE: Return {} instead of FileNotFoundError 
            # to avoid the AttributeError when the test tries to access the result.
            else {} 
        ),
    )

    last_versions = get_atlases_lastversions()
    example_atlas = last_versions[mock_atlas_dir]
    local_v = get_local_atlas_version(mock_atlas_dir)

    assert example_atlas["version"] == local_v
    assert example_atlas["latest_version"] == local_v


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


def test_empty_custom_config_no_crash(tmp_path: Path, mocker):
    """Test that an empty custom_atlases.conf file does not cause a crash.

    Parameters
    ----------
    tmp_path : pathlib.Path
        A temporary directory created by pytest.
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture.
    """
    # Create a temporary brainglobe directory and an empty config file
    bg_dir = tmp_path / ".brainglobe"
    bg_dir.mkdir()
    empty_conf_file = bg_dir / "custom_atlases.conf"
    empty_conf_file.touch()

    # Create a mock for the official atlases which are needed for the function to run
    mock_official_atlases = {"atlases": {"official_atlas": "1.0"}}
    
    # Patch the function that reads the official atlases from URL/cache
    mocker.patch(
        "brainglobe_atlasapi.list_atlases.utils.conf_from_url",
        return_value=mock_official_atlases
    )
    mocker.patch(
        "brainglobe_atlasapi.list_atlases.utils.conf_from_file",
        side_effect=lambda file_path: (
            mock_official_atlases
            if file_path.name == "last_versions.conf"
            # Return an empty dict if the file path is the custom one (simulating empty content)
            else {} 
        ),
    )

    # Patch the config.get_brainglobe_dir to point to the temp directory
    with mocker.patch(
        "brainglobe_atlasapi.config.get_brainglobe_dir", 
        return_value=bg_dir
    ):
        # The test passes if this call does not raise a KeyError
        last_versions = get_all_atlases_lastversions()
        
        # Ensure the result is a dictionary and contains the official atlas
        assert isinstance(last_versions, dict)
        assert "official_atlas" in last_versions