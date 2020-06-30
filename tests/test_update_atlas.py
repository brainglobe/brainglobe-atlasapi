from brainatlas_api import update


def test_update():
    update.update_atlas("allen_mouse_25um")


def test_update_wrong_name():
    update.update_atlas("allen_madasadsdouse_25um")
