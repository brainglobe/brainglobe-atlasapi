from bg_atlasapi import update


def test_update():
    update.update_atlas(atlas_name="example_mouse_100um")

    update.update_atlas(atlas_name="example_mouse_100um", force=True)


def test_update_wrong_name():
    update.update_atlas("allen_madasadsdouse_25um")
