import shutil
import tempfile

import pytest

from bg_atlasapi.bg_atlas import BrainGlobeAtlas


def test_versions(atlas):
    assert atlas.local_version == atlas.remote_version


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
