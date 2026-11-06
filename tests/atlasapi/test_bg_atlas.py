"""Test the BrainGlobeAtlas class."""

import shutil
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import brainglobe_atlasapi
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


def test_versions(atlas):
    """Assert local and remote versions are equal."""
    assert atlas.local_version == atlas.remote_version


def test_remote_version_connection_error():
    """Test handling a connection error when fetching the remote version."""
    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=False
    ):
        atlas = object.__new__(BrainGlobeAtlas)
        atlas._remote_version = None
        assert atlas.remote_version is None


def _atlas_for_remote_version(
    atlas_name, brainglobe_dir, local_full_name=None
):
    """Create a bare BrainGlobeAtlas with a mocked filesystem."""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    atlas.brainglobe_dir = brainglobe_dir
    atlas._remote_version = None
    atlas._requested_version = None
    atlas._local_full_name = local_full_name
    atlas.fs = MagicMock()
    atlas.fs.exists.return_value = True
    # An in-progress version is present on the remote
    atlas.fs.ls.return_value = [
        f"brainglobe/atlas/atlases/{atlas_name}/1_0",
        f"brainglobe/atlas/atlases/{atlas_name}/2_0",
    ]
    return atlas


def test_remote_version_uses_last_versions_conf(tmp_path):
    """Test the latest remote version comes from last_versions.conf,
    ignoring newer versions on the remote that are not yet released.
    """
    atlas = _atlas_for_remote_version("example_mouse_100um", tmp_path)
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={"example_mouse_100um": "1.0"},
        ),
    ):
        assert atlas.remote_version == (1, 0)


def test_remote_version_unreleased_atlas(tmp_path):
    """Test an atlas absent from last_versions.conf is not fetched."""
    atlas = _atlas_for_remote_version("example_mouse_100um", tmp_path)
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={},
        ),
    ):
        with pytest.raises(FileNotFoundError, match="no released version"):
            atlas.remote_version


def test_remote_version_unreleased_atlas_local(tmp_path):
    """Test an atlas absent from last_versions.conf but available locally
    returns no remote version.
    """
    atlas = _atlas_for_remote_version(
        "example_mouse_100um",
        tmp_path,
        local_full_name="atlases/example_mouse_100um/2_0/manifest.json",
    )
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={},
        ),
    ):
        assert atlas.remote_version is None


@pytest.mark.parametrize(
    "local_version, remote_version, expected",
    [
        pytest.param((1, 0), (2, 0), False, id="local < remote"),
        pytest.param((1, 0), (1, 0), True, id="local = remote"),
        pytest.param((1, 0), None, None, id="no remote version"),
    ],
)
def test_check_latest_version_local(local_version, remote_version, expected):
    """Test `check_latest_version` method.

    Parameters
    ----------
    local_version : tuple
        The mock local version.
    remote_version : tuple
        The mock remote version.
    expected : bool or None
        The expected result of `check_latest_version`.
    """
    with (
        patch.object(
            BrainGlobeAtlas, "local_version", new_callable=PropertyMock
        ) as mock_local_version,
        patch.object(
            BrainGlobeAtlas, "remote_version", new_callable=PropertyMock
        ) as mock_remote_version,
    ):
        mock_local_version.return_value = local_version
        mock_remote_version.return_value = remote_version
        atlas = object.__new__(BrainGlobeAtlas)
        assert atlas.check_latest_version(print_warning=False) == expected


@pytest.mark.parametrize(
    "atlas_name, expected_repr",
    [
        pytest.param(
            "nadkarni_mri_mouselemur_91um",
            "nadkarni mri mouselemur atlas (res. 91um)",
            id="nadkarni_mri_mouselemur_91um",
        ),
        pytest.param(
            "example_mouse_100um",
            "example mouse atlas (res. 100um)",
            id="example_mouse_100um",
        ),
        pytest.param(
            "axolotl_50um", "axolotl atlas (res. 50um)", id="axolotl_50um"
        ),
    ],
)
def test_repr(atlas_name, expected_repr):
    """Test `BrainGlobeAtlas` `__repr__` method.

    Parameters
    ----------
    atlas_name : str
        The name of the atlas.
    expected_repr : str
        The expected string representation.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    assert repr(atlas) == expected_repr


def test_str(atlas, capsys):
    """Test `BrainGlobeAtlas` `__str__` method."""
    print(atlas)
    captured = capsys.readouterr()
    expected_doi = "https://doi.org/10.1016/j.cell.2020.04.007"
    assert expected_doi in captured.out
    assert captured.err == ""


def test_fn_update_called_on_install(tmp_path):
    """`fn_update` is called while an atlas is downloaded.

    `BrainGlobeAtlas` accepts `fn_update` so that external progress bars can
    follow an install, but the download used to ignore it entirely.
    """
    calls = []

    BrainGlobeAtlas(
        "example_mouse_100um",
        brainglobe_dir=tmp_path,
        check_latest=False,
        fn_update=lambda completed, total: calls.append((completed, total)),
    )

    assert calls, "fn_update was not called during the atlas download"
    assert all(0 <= completed <= total for completed, total in calls)
    assert any(completed == total > 0 for completed, total in calls)


def test_local_search(tmpdir):
    """Test local atlas search and handling of multiple versions."""
    atlas_name = "example_mouse_100um"
    temp_brainglobe_dir = tmpdir.mkdir("brainglobe")

    atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=temp_brainglobe_dir,
    )

    assert atlas.atlas_name in atlas.local_full_name

    # Make a copy:
    new_version = "4_5"
    old_version = "_".join(str(v) for v in atlas.local_version)
    atlas_path = atlas.local_full_name.split("/")
    atlas_root = "/".join(atlas_path[:-2])
    new_path = atlas.root_dir / atlas_root / new_version
    old_path = atlas.root_dir / atlas_root / old_version
    shutil.copytree(old_path, new_path)

    # Should find the latest version:
    atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=temp_brainglobe_dir,
    )
    assert atlas.local_full_name == f"{atlas_root}/{new_version}/manifest.json"
