import shutil
import tempfile
from unittest.mock import PropertyMock, patch

import pytest
import requests

from brainglobe_atlasapi import utils
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


def test_versions(atlas):
    assert atlas.local_version == atlas.remote_version


def test_local_full_name_none():
    with patch.object(
        BrainGlobeAtlas, "local_full_name", new_callable=PropertyMock
    ) as mock_local_full_name:
        mock_local_full_name.return_value = None
        atlas = object.__new__(BrainGlobeAtlas)  # Avoids calling __init__
        assert atlas.local_version is None


def test_remote_version_connection_error():
    with patch.object(
        utils, "conf_from_url", side_effect=requests.ConnectionError
    ):
        atlas = object.__new__(BrainGlobeAtlas)  # Avoids calling __init__
        assert atlas.remote_version is None


def test_check_latest_version_offline():
    with patch.object(
        BrainGlobeAtlas, "remote_version", new_callable=PropertyMock
    ) as mock_remote_version:
        mock_remote_version.return_value = None
        atlas = object.__new__(BrainGlobeAtlas)  # Avoids calling __init__
        assert atlas.check_latest_version() is None


def test_local_search():
    brainglobe_dir = tempfile.mkdtemp()
    interm_download_dir = tempfile.mkdtemp()

    atlas = BrainGlobeAtlas(
        "example_mouse_100um",
        brainglobe_dir=brainglobe_dir,
        interm_download_dir=interm_download_dir,
    )

    assert atlas.atlas_name in atlas.local_full_name

    # Make a copy:
    copy_filename = atlas.root_dir.parent / (atlas.root_dir.name + "_2")
    shutil.copytree(atlas.root_dir, copy_filename)

    with pytest.raises(FileExistsError) as error:
        _ = BrainGlobeAtlas(
            "example_mouse_100um", brainglobe_dir=brainglobe_dir
        )
    assert "Multiple versions of atlas" in str(error)

    shutil.rmtree(brainglobe_dir)
    shutil.rmtree(interm_download_dir)
