from brainatlas_api import update
from click.testing import CliRunner


def test_update():
    update.update_atlas(atlas_name="allen_mouse_25um")

    update.update_atlas(atlas_name="allen_mouse_25um", force=True)


def test_update_wrong_name():
    update.update_atlas("allen_madasadsdouse_25um")


def test_update_command():
    runner = CliRunner()

    # Test printing of config file:
    runner.invoke(update.cli_atlas_command, ["-a", "allen_mouse_25um"])
