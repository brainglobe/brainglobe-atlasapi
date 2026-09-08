"""Test functions for listing and managing BrainGlobe atlases."""

from typing import get_args
from unittest import mock

import pytest
from rich.console import Console
from rich.table import Table

from brainglobe_atlasapi import config, utils
from brainglobe_atlasapi.atlas_name import AtlasName
from brainglobe_atlasapi.list_atlases import (
    add_atlas_to_row,
    folder_version_to_dotted,
    get_all_atlases_lastversions,
    get_atlases_lastversions,
    get_downloaded_atlases,
    get_local_atlas_version,
    show_atlases,
)


@pytest.fixture
def mock_installed_atlases(mocker):
    """Pretend one official and one custom atlas are installed locally.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture.

    Returns
    -------
    Callable
        Call with the contents of ``custom_atlases.conf`` to apply the mocks.
    """

    def _apply(registered_custom):
        official = {"example_mouse_100um": "3.0"}
        mocker.patch(
            "brainglobe_atlasapi.list_atlases.get_downloaded_atlases",
            return_value=["example_mouse_100um", "my_custom_atlas_10um"],
        )
        mocker.patch(
            "brainglobe_atlasapi.list_atlases.get_all_atlases_lastversions",
            return_value={**official, **registered_custom},
        )
        mocker.patch(
            "brainglobe_atlasapi.list_atlases.get_custom_atlases",
            return_value=registered_custom,
        )
        mocker.patch(
            "brainglobe_atlasapi.list_atlases.get_local_atlas_version",
            side_effect=lambda name: (
                "3.0" if name == "example_mouse_100um" else "1.0"
            ),
        )

    return _apply


def test_folder_version_to_dotted():
    """Test conversion from folder-style version to dotted version."""
    assert folder_version_to_dotted("3_0") == "3.0"
    assert folder_version_to_dotted("5_2") == "5.2"
    assert folder_version_to_dotted(None) is None


def test_get_downloaded_atlases():
    """Test retrieving a list of downloaded atlases."""
    available_atlases = get_downloaded_atlases()

    # Check that example is listed:
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
                example_atlas["latest_version"].split("."),
                local_v.replace("_", ".").split("."),
            )
        ]
    )
    assert example_atlas["updated"] == (
        example_atlas["version"] == example_atlas["latest_version"]
    )


def test_show_atlases():
    """Test displaying a table of available atlases."""
    # TODO add more valid testing than just look for errors when running:
    show_atlases(show_local_path=True)


def test_lastversions_unregistered_custom_atlas(mock_installed_atlases):
    """A locally installed atlas absent from both conf files is listed.

    Parameters
    ----------
    mock_installed_atlases : Callable
        Fixture mocking the set of installed atlases.
    """
    mock_installed_atlases({})

    atlases = get_atlases_lastversions()

    assert "my_custom_atlas_10um" in atlases
    custom = atlases["my_custom_atlas_10um"]
    assert custom["custom"] is True
    assert custom["downloaded"] is True
    assert custom["version"] == "1.0"
    # There is no known remote version to compare the local files against.
    assert custom["latest_version"] == ""
    assert custom["updated"] is None

    official = atlases["example_mouse_100um"]
    assert official["custom"] is False
    assert official["latest_version"] == "3.0"
    assert official["updated"] is True


def test_lastversions_registered_custom_atlas(mock_installed_atlases):
    """An atlas in custom_atlases.conf is listed and flagged as custom.

    Parameters
    ----------
    mock_installed_atlases : Callable
        Fixture mocking the set of installed atlases.
    """
    mock_installed_atlases({"my_custom_atlas_10um": "1.0"})

    atlases = get_atlases_lastversions()

    custom = atlases["my_custom_atlas_10um"]
    assert custom["custom"] is True
    assert custom["latest_version"] == "1.0"
    assert custom["updated"] is True
    assert atlases["example_mouse_100um"]["custom"] is False


def test_show_atlases_lists_custom_atlas(mock_installed_atlases, capsys):
    """Custom atlases appear in the printed table.

    Parameters
    ----------
    mock_installed_atlases : Callable
        Fixture mocking the set of installed atlases.
    capsys : pytest.CaptureFixture
        Fixture to capture stdout/stderr.
    """
    mock_installed_atlases({})

    show_atlases()

    # The "Custom" column is dropped by rich on narrow terminals, so only
    # the name is checked here. The marker itself is covered by
    # `test_add_atlas_to_row`.
    captured = capsys.readouterr()
    assert "my_custom_atlas_10um" in captured.out


def test_get_all_atlases_lastversions():
    """Test retrieving the latest versions of all known atlases."""
    last_versions = get_all_atlases_lastversions()

    assert "example_mouse_100um" in last_versions
    assert "osten_mouse_50um" in last_versions
    assert "allen_mouse_25um" in last_versions


def test_atlas_name_matches_lastversions():
    """Ensure all atlases in last_versions.conf are valid AtlasName values."""
    atlas_name_values = list(get_args(AtlasName))
    cache_path = (
        config.get_brainglobe_dir()
        / "brainglobe-atlasapi"
        / "atlases"
        / "last_versions.conf"
    )
    # we read the file directly, using lastversions() includes custom atlases.
    last_versions = utils.conf_from_file(cache_path)["atlases"]
    last_version_names = list(last_versions.keys())

    assert len(atlas_name_values) == len(set(atlas_name_values))
    assert len(last_version_names) == len(set(last_version_names))
    assert set(atlas_name_values).issuperset(set(last_version_names))


def test_get_all_atlases_custom_atlases(mocker):
    """Check inclusion of available custom atlases in the list of all atlases.

    Parameters
    ----------
    mocker : pytest_mock.plugin.MockerFixture
        The mocker fixture.
    """
    custom_path = (
        config.get_brainglobe_dir()
        / "brainglobe-atlasapi"
        / "atlases"
        / "custom_atlases.conf"
    )
    mock_custom_atlas = {"atlases": {"mock_custom_atlas": "1.1"}}

    with mocker.patch(
        "brainglobe_atlasapi.utils.conf_from_file",
        side_effect=lambda file_path: {
            custom_path: mock_custom_atlas,
        }.get(file_path, FileNotFoundError),
    ):
        last_versions = get_all_atlases_lastversions()
        assert last_versions["mock_custom_atlas"] == "1.1"


def test_get_all_atlases_lastversions_empty_custom_atlases(tmp_path):
    """Test retrieving atlas versions when custom_atlases.conf is empty."""
    custom_path = tmp_path / "custom_atlases.conf"
    custom_path.touch()

    with mock.patch(
        "brainglobe_atlasapi.list_atlases.config.get_brainglobe_dir",
        return_value=tmp_path,
    ):
        last_versions = get_all_atlases_lastversions()

        assert "example_mouse_100um" in last_versions
        assert "osten_mouse_50um" in last_versions
        assert "allen_mouse_25um" in last_versions


def test_get_all_atlases_lastversions_offline():
    """Test retrieving atlas versions from cache when offline."""
    cleanup_cache = False
    cache_path = (
        config.get_brainglobe_dir()
        / "brainglobe-atlasapi"
        / "atlases"
        / "last_versions.conf"
    )

    if not cache_path.exists():
        cache_path.touch()
        cache_path.write_text("""
            [atlases]
            example_mouse_100um = 1.0
            osten_mouse_50um = 1.0
            allen_mouse_25um = 1.0
            """)
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
    cache_path = (
        config.get_brainglobe_dir()
        / "brainglobe-atlasapi"
        / "atlases"
        / "last_versions.conf"
    )

    if not cache_path.exists():
        cache_path.touch()
        cache_path.write_text("""
            [atlases]
            example_mouse_100um = 1.0
            osten_mouse_50um = 1.0
            allen_mouse_25um = 1.0
            """)
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
                "updated": False,
                "custom": False,
            },
            "│ awesome_name │ ✔ │  │ x │ 1 │ 2 │",
            id="version != latest_version",
        ),
        pytest.param(
            {
                "version": "1",
                "latest_version": "1",
                "updated": True,
                "custom": False,
            },
            "│ awesome_name │ ✔ │  │ ✔ │ 1 │ 1 │",
            id="version == latest_version",
        ),
        pytest.param(
            {
                "version": "1",
                "latest_version": "",
                "updated": None,
                "custom": True,
            },
            "│ awesome_name │ ✔ │ ✔ │  │ 1 │  │",
            id="custom atlas without a remote version",
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
        "updated": version["updated"],
        "custom": version["custom"],
    }
    table = add_atlas_to_row(atlas="awesome_name", info=info, table=Table())
    Console().print(table)
    captured = capsys.readouterr()
    assert expected_print in captured.out
