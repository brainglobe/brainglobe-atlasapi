import shutil
from unittest.mock import PropertyMock, patch

import pytest
import requests

from brainglobe_atlasapi import config, utils
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


def test_versions(atlas):
    assert atlas.local_version == atlas.remote_version


def test_local_full_name_none():
    """Test local_version when local_full_name is None."""
    with patch.object(
        BrainGlobeAtlas, "local_full_name", new_callable=PropertyMock
    ) as mock_local_full_name:
        mock_local_full_name.return_value = None
        atlas = object.__new__(BrainGlobeAtlas)
        assert atlas.local_version is None


def test_remote_version_connection_error():
    """Test handling a connection error when fetching the remote version."""
    with patch.object(
        utils, "conf_from_url", side_effect=requests.ConnectionError
    ):
        atlas = object.__new__(BrainGlobeAtlas)
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
    """Test check_latest_version"""
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
        assert atlas.check_latest_version() == expected


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
    """Test BrainGlobeAtlas repr method"""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    assert repr(atlas) == expected_repr


def test_str(atlas, capsys):
    """Test BrainGlobeAtlas str method"""
    print(atlas)
    captured = capsys.readouterr()
    expected_doi = "https://doi.org/10.1016/j.cell.2020.04.007"
    assert expected_doi in captured.out
    assert captured.err == ""


def test_local_search(tmpdir):
    atlas_file_name = "example_mouse_100um_v1.2"
    brainglobe_dir = config.get_brainglobe_dir()
    temp_brainglobe_dir = tmpdir.mkdir("brainglobe")
    shutil.copytree(
        brainglobe_dir / atlas_file_name,
        temp_brainglobe_dir / atlas_file_name,
        dirs_exist_ok=True,
    )
    interim_download_dir = tmpdir.mkdir("interim_download")

    atlas = BrainGlobeAtlas(
        "example_mouse_100um",
        brainglobe_dir=temp_brainglobe_dir,
        interm_download_dir=interim_download_dir,
    )

    assert atlas.atlas_name in atlas.local_full_name

    # Make a copy:
    copy_filename = atlas.root_dir.parent / (atlas.root_dir.name + "_2")
    shutil.copytree(atlas.root_dir, copy_filename)

    with pytest.raises(FileExistsError) as error:
        _ = BrainGlobeAtlas(
            "example_mouse_100um", brainglobe_dir=temp_brainglobe_dir
        )
    assert "Multiple versions of atlas" in str(error)
