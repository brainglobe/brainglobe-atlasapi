import pytest
from brainatlas_api.bg_atlas import TestAtlas
import tempfile

@pytest.fixture(scope="module")
def atlas_path():
    return TestAtlas(brainglobe_path=tempfile.mkdtemp()).root