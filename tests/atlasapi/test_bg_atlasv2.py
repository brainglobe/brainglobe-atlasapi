from pathlib import Path

import numpy as np
import pytest

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.bg_atlasv2 import BrainGlobeAtlasV2

MOCK_V2_DIRECTORY = Path.home() / ".brainglobe-tests" / ".brainglobe_v2"


@pytest.fixture(scope="module")
def v1_atlas():
    return BrainGlobeAtlas("allen_mouse_25um")


@pytest.fixture(scope="module")
def v2_atlas():
    return BrainGlobeAtlasV2(
        "allen_mouse_25um",
        brainglobe_dir=MOCK_V2_DIRECTORY,
        interm_download_dir=MOCK_V2_DIRECTORY,
        check_latest=False,
    )


def test_bg_atlasv2_init(v1_atlas, v2_atlas):
    assert isinstance(v2_atlas, BrainGlobeAtlasV2)
    assert v2_atlas.atlas_name == v1_atlas.atlas_name
    assert v2_atlas.brainglobe_dir == MOCK_V2_DIRECTORY
    assert v2_atlas.interm_download_dir == MOCK_V2_DIRECTORY


@pytest.mark.xfail
def test_local_version(v1_atlas, v2_atlas):
    assert v2_atlas.local_version == v1_atlas.local_version


@pytest.mark.xfail
def test_remote_version(v1_atlas, v2_atlas):
    assert v2_atlas.remote_version == v1_atlas.remote_version


def test_local_full_name(v1_atlas, v2_atlas):
    assert v2_atlas.local_full_name == "allen_mouse_25um_v1.2.json"


# @pytest.mark.xfail
# def test_remote_url(v1_atlas, v2_atlas):
#     assert v2_atlas.remote_url == v1_atlas.remote_url


# @pytest.mark.xfail
# def test_check_latest_version(v1_atlas, v2_atlas):
#     assert v2_atlas.check_latest_version() == v1_atlas.check_latest_version()


@pytest.mark.xfail
def test_repr(v1_atlas, v2_atlas):
    assert repr(v2_atlas) == repr(v1_atlas)


@pytest.mark.xfail
def test_str(v1_atlas, v2_atlas):
    assert str(v2_atlas) == str(v1_atlas)


def test_reference(v1_atlas, v2_atlas):
    assert np.array_equal(v2_atlas.reference, v1_atlas.reference)
