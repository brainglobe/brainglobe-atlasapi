from brainatlas_api import update


def test_update():
    update.update_atlas("allen_mouse_25um")

    update.update_atlas("allen_mouse_25um", force=True)


def test_update_wrong_name():
    update.update_atlas("allen_madasadsdouse_25um")
