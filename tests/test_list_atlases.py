import brainatlas_api
import pytest


def test_list_atlases():
    brainatlas_api.list_atlases()


@pytest.mark.parametrize(
    "key, is_none", [("allen_mouse_25um", False), ("xxx", True)]
)
def test_get_atlas_from_name(key, is_none):
    a = brainatlas_api.get_atlas_class_from_name(key)
    if is_none:
        assert a is None
    else:
        assert a is not None
