import os
import shutil
import tempfile
from pathlib import Path

import pytest

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas, config


@pytest.fixture(autouse=True)
def mock_brainglobe_user_folders(monkeypatch):
    """Ensures user config and data is mocked during all local testing.

    User config and data need mocking to avoid interfering with user data.
    Mocking is achieved by turning user data folders used in tests into
    subfolders of a new ~/.brainglobe-tests folder instead of ~/.

    It is not sufficient to mock the home path in the tests, as this
    will leave later imports in other modules unaffected.

    GH actions workflow will test with default user folders.
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
    return BrainGlobeAtlas("example_mouse_100um")


@pytest.fixture()
def asymmetric_atlas():
    return BrainGlobeAtlas("unam_axolotl_40um")


@pytest.fixture()
def temp_path():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def atlas_path():
    return BrainGlobeAtlas("example_mouse_100um").root_dir
