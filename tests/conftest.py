import pytest
from brainatlas_api.bg_atlas import ExampleAtlas

# import tempfile


@pytest.fixture(scope="module")
def atlas_path():
    # brainglobe_path=tempfile.mkdtemp()
    return ExampleAtlas().root_dir
