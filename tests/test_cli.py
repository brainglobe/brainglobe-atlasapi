from click.testing import CliRunner

from bg_atlasapi import cli, config


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


def test_list_command():
    runner = CliRunner()

    # Test printing of config file:
    runner.invoke(cli.bg_cli, ["list", "s"])
