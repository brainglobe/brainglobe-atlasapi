import brainatlas_api


def test_list_atlases():
    brainatlas_api.list_atlases()


def test_get_atlas_from_name():
    a1 = brainatlas_api.get_atlas_class_from_name("allen_mouse_25um_v0.2")
    a2 = brainatlas_api.get_atlas_class_from_name("xxxx")

    if a1 is None:
        raise ValueError
    if a2 is not None:
        raise ValueError
