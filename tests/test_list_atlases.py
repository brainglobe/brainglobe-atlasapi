import bg_atlasapi
from bg_atlasapi import list_atlases
import pytest


def test_show_atlases():
    list_atlases.show_atlases(show_local_path=True)


@pytest.mark.parametrize(
    "key, is_none", [("allen_mouse_25um", False), ("xxx", True)]
)
def test_get_atlas_from_name(key, is_none):
    a = bg_atlasapi.get_atlas_class_from_name(key)
    if is_none:
        assert a is None
    else:
        assert a is not None
