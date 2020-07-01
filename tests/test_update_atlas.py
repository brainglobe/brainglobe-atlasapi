from bg_atlasapi import update
from click.testing import CliRunner


def test_update():
    update.update_atlas(atlas_name="example_mouse_100um")

    update.update_atlas(atlas_name="example_mouse_100um", force=True)


def test_update_wrong_name():
    update.update_atlas("allen_madasadsdouse_25um")


def test_update_command():
    runner = CliRunner()

    # Test printing of config file:
    result = runner.invoke(
        update.cli_update_atlas_command, ["-a", "example_mouse_100um"]
    )

    assert result.exit_code == 0
