import pytest
from bg_atlasapi.bg_atlas import ExampleAtlas

# import tempfile


@pytest.fixture()
def atlas():
    return ExampleAtlas()


@pytest.fixture(scope="module")
def atlas_path():
    # brainglobe_path=tempfile.mkdtemp()
    return ExampleAtlas().root_dir


def pytest_addoption(parser):
    parser.addoption("--runslow", action="store_true", help="run slow tests")


def pytest_runtest_setup(item):
    if "slow" in item.keywords and not item.config.getvalue("runslow"):
        pytest.skip("need --runslow option to run")
