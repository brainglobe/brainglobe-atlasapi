"""Pytest fixtures for the brainglobe_atlasapi package."""

import functools
import os
import shutil
import tempfile
from pathlib import Path

import meshio
import numpy as np
import pytest

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas, config
from brainglobe_atlasapi.update_atlases import update_atlas


@pytest.fixture(autouse=True, scope="session")
def mock_brainglobe_user_folders():
    """Mock BrainGlobe user folders.

    Ensure user config and data is mocked during all local testing to avoid
    interfering with actual user data.

    Mocking is achieved by turning user data folders used in tests into
    subfolders of a new ~/.brainglobe-tests folder instead of the user's home
    directory. It is not sufficient to mock the home path in the tests, as this
    will leave later imports in other modules unaffected.

    Session-scoped so that it is set up before session-scoped fixtures that
    download atlases (e.g. `tests/atlasgen/conftest.py`); a function-scoped
    mock would run too late and those would use the real home directory.

    Note
    ----
    GitHub Actions workflow will test with default user folders.
    """
    if os.getenv("GITHUB_ACTIONS"):
        yield
        return

    with pytest.MonkeyPatch.context() as monkeypatch:
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
        yield


@functools.cache
def _update_once(atlas_name):
    """Check an atlas is at its latest version, once per test session.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to check.
    """
    update_atlas(atlas_name)


def _cached_atlas(atlas_name, **kwargs):
    """Build an atlas from the local cache, downloading only what is missing.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to build.
    **kwargs
        Passed on to `BrainGlobeAtlas`.

    Returns
    -------
    BrainGlobeAtlas
        The requested atlas.
    """
    _update_once(atlas_name)
    return BrainGlobeAtlas(atlas_name, check_latest=False, **kwargs)


@pytest.fixture()
def atlas():
    """Provide a default BrainGlobeAtlas instance.

    A new instance per test, as tests modify it, built from the local cache.

    Returns
    -------
    BrainGlobeAtlas
        An instance of BrainGlobeAtlas for 'example_mouse_100um'.
    """
    return _cached_atlas("example_mouse_100um")


@pytest.fixture(scope="session")
def atlas_assets_atlas():
    """Provide an atlas-assets atlas, shared by the whole test session.

    Tests must treat it as read-only.

    Returns
    -------
    BrainGlobeAtlas
        An instance of 'allen-adult-mouse-ccf-atlas' at 100 um.
    """
    return _cached_atlas("allen-adult-mouse-ccf-atlas", resolution=100)


@pytest.fixture(scope="session")
def anisotropic_atlas():
    """Provide a non-isotropic atlas, shared by the whole test session.

    Tests must treat it as read-only.

    Returns
    -------
    BrainGlobeAtlas
        An instance of 'allen-dev-mouse-e11pt5-nissl-atlas', whose voxels
        are 16 x 16 x 20 um.
    """
    return _cached_atlas("allen-dev-mouse-e11pt5-nissl-atlas")


@pytest.fixture()
def asymmetric_atlas():
    """Provide an asymmetric BrainGlobeAtlas instance.

    Returns
    -------
    BrainGlobeAtlas
        An instance of BrainGlobeAtlas for 'unam_axolotl_40um'.
    """
    return _cached_atlas("unam_axolotl_40um")


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


@pytest.fixture()
def atlas_path():
    """Provide the root directory path of the default atlas.

    Returns
    -------
    pathlib.Path
        The root directory path of the 'example_mouse_100um' atlas.
    """
    return _cached_atlas("example_mouse_100um").root_dir


@pytest.fixture(scope="session")
def mask_structures():
    """Provide a nested 3-structure list used by 4D-mask tests.

    The structure tree is::

        root (999)
        └── region_a (1)
            └── leaf_b (2)

    Post-order mapping (leaf first): leaf_b -> 0, region_a -> 1, root -> 2.

    Returns
    -------
    list
        Structure dictionaries with id, acronym, name, RGB triplet and
        structure ID path.
    """
    return [
        {
            "id": 999,
            "acronym": "root",
            "name": "root",
            "rgb_triplet": [255, 255, 255],
            "structure_id_path": [999],
        },
        {
            "id": 1,
            "acronym": "region_a",
            "name": "Region A",
            "rgb_triplet": [100, 150, 200],
            "structure_id_path": [999, 1],
        },
        {
            "id": 2,
            "acronym": "leaf_b",
            "name": "Leaf B",
            "rgb_triplet": [200, 100, 50],
            "structure_id_path": [999, 1, 2],
        },
    ]


@pytest.fixture(scope="session")
def tetrahedron_mesh_file(tmp_path_factory):
    """Write a minimal tetrahedron surface mesh as an .obj file.

    The mesh content is independent of any structure ID; callers map whatever
    IDs they need to the returned path when building a ``meshes_dict``.

    Returns
    -------
    pathlib.Path
        Path to the written .obj mesh.
    """
    mesh_path = tmp_path_factory.mktemp("meshes") / "tetrahedron.obj"
    points = np.array(
        [[0, 0, 0], [10, 0, 0], [0, 10, 0], [0, 0, 10]], dtype=float
    )
    cells = [
        ("triangle", np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]))
    ]
    meshio.write(str(mesh_path), meshio.Mesh(points=points, cells=cells))
    return mesh_path
