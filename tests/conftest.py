import shutil
import tempfile
from pathlib import Path

import pytest

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


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
