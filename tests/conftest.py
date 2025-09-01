"""Pytest fixtures for the brainglobe_atlasapi package."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas, config


@pytest.fixture(autouse=True)
def mock_brainglobe_user_folders(monkeypatch):
    """Mock BrainGlobe user folders.

    Ensure user config and data is mocked during all local testing to avoid
    interfering with actual user data.

    Mocking is achieved by turning user data folders used in tests into
    subfolders of a new ~/.brainglobe-tests folder instead of the user's home
    directory. It is not sufficient to mock the home path in the tests, as this
    will leave later imports in other modules unaffected.

    Note
    ----
    GitHub Actions workflow will test with default user folders.
    """
    if not os.getenv("GITHUB_ACTIONS"):
        home_path = Path.home()  # actual home path
        mock_home_path = home_path / ".brainglobe-tests"
        if not mock_home_path.exists():
            mock_home_path.mkdir()

        def mock_home():
            return mock_home_path

        monkeypatch.setattr(Path, "home", mock_home)

        # also mock global variables of config.py
        monkeypatch.setattr(
            config, "DEFAULT_PATH", mock_home_path / ".brainglobe"
        )
        monkeypatch.setattr(
            config, "CONFIG_DIR", mock_home_path / ".config" / "brainglobe"
        )
        monkeypatch.setattr(
            config, "CONFIG_PATH", config.CONFIG_DIR / config.CONFIG_FILENAME
        )
        mock_default_dirs = {
            "default_dirs": {
                "brainglobe_dir": mock_home_path / ".brainglobe",
                "interm_download_dir": mock_home_path / ".brainglobe",
            }
        }
        monkeypatch.setattr(config, "TEMPLATE_CONF_DICT", mock_default_dirs)


@pytest.fixture()
def atlas():
    """Provide a default BrainGlobeAtlas instance.

    Returns
    -------
    BrainGlobeAtlas
        An instance of BrainGlobeAtlas for 'example_mouse_100um'.
    """
    return BrainGlobeAtlas("example_mouse_100um")


@pytest.fixture()
def asymmetric_atlas():
    """Provide an asymmetric BrainGlobeAtlas instance.

    Returns
    -------
    BrainGlobeAtlas
        An instance of BrainGlobeAtlas for 'unam_axolotl_40um'.
    """
    return BrainGlobeAtlas("unam_axolotl_40um")


@pytest.fixture()
def temp_path():
    """Create a temporary directory for testing.

    The directory is automatically removed after the test.

    Yields
    ------
    pathlib.Path
        Path to the temporary directory.
    """
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def atlas_path():
    """Provide the root directory path of the default atlas.

    Returns
    -------
    pathlib.Path
        The root directory path of the 'example_mouse_100um' atlas.
    """
    return BrainGlobeAtlas("example_mouse_100um").root_dir
