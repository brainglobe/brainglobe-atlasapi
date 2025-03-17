import pytest
from click.testing import CliRunner

from brainglobe_atlasapi import cli, config


# This testing of the command line application does not really
# cange anything in the filesystem, so the repo config will remain unchanged:
def test_config_cli():
    runner = CliRunner()

    # Test printing of config file:
    result = runner.invoke(cli.bg_cli, ["config", "--show"])
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    # Correct edit (this does not really change the file):
    result = runner.invoke(
        cli.bg_cli, ["config", "-k brainglobe_dir -v valid_path"]
    )
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"


def test_update_command():
    runner = CliRunner()

    # Test printing of config file:
    runner.invoke(cli.bg_cli, ["config", "-a", "example_mouse_100um", "-f"])


def test_install_command():
    runner = CliRunner()

    # Test printing of config file:
    runner.invoke(cli.bg_cli, ["install", "-a", "example_mouse_100um"])


def test_install_command_value_error():
    """Test whether install command without atlas name raises ValueError."""
    with pytest.raises(
        ValueError,
        match='No atlas named passed with command "install". Use the "-a"\
                                argument to pass an atlas name',
    ):
        CliRunner().invoke(
            cli.bg_cli, ["install", "-a", None], catch_exceptions=False
        )
