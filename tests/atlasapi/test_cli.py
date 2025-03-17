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
    """Test update command."""
    update_atlas = CliRunner().invoke(
        cli.bg_cli, ["update", "-a", "example_mouse_100um", "-f"]
    )
    expected_output = "updating example_mouse_100um\nDownloading..."
    assert expected_output in update_atlas.output


def test_install_command():
    runner = CliRunner()

    # Test printing of config file:
    runner.invoke(cli.bg_cli, ["install", "-a", "example_mouse_100um"])


def test_cli_list():
    """Test list command."""
    atlases_table = CliRunner().invoke(cli.bg_cli, ["list", "--show"])
    assert atlases_table.exit_code == 0
    assert "nadkarni_mri_mouselemur_91um" in atlases_table.output


@pytest.mark.parametrize(
    ["command"], [pytest.param("install"), pytest.param("update")]
)
def test_atlas_name_is_none_value_error(command):
    """Test whether command without atlas name raises ValueError."""
    with pytest.raises(
        ValueError,
        match=(
            f'No atlas named passed with command "{command}". Use the "-a"'
            r"\s+argument to pass an atlas name"
        ),
    ):
        CliRunner().invoke(
            cli.bg_cli, [command, "-a", None], catch_exceptions=False
        )


def test_cli_incorrect_command():
    """Test whether incorrect "flibble" command raises ValueError."""
    command = "flibble"
    with pytest.raises(
        ValueError,
        match=f'Invalid command {command}. use "brainglobe -h for more info."',
    ):
        CliRunner().invoke(cli.bg_cli, [command], catch_exceptions=False)
