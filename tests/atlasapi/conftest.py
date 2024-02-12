import shutil
import tempfile
from pathlib import Path

import pytest

from bg_atlasapi.bg_atlas import BrainGlobeAtlas


@pytest.fixture()
def atlas():
    return BrainGlobeAtlas("example_mouse_100um")


@pytest.fixture()
def temp_path():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path)


@pytest.fixture(scope="module")
def atlas_path():
    return BrainGlobeAtlas("example_mouse_100um").root_dir


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getvalue("runslow"):
        pytest.skip("need --runslow option to run")
