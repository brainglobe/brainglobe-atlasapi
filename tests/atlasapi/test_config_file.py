"""Test configuration file handling.

Verify the creation, modification, and CLI interaction with the Brainglobe
atlas API configuration file.
"""

import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from brainglobe_atlasapi import bg_atlas, cli, config
from brainglobe_atlasapi.config import cli_modify_config


@pytest.fixture()
def conf_path():
    """Create a temporary configuration file for testing.

    Yields
    ------
    Path
        The path to the temporary configuration file.
    """
    temp_dir = Path(tempfile.mkdtemp())
    conf_path = temp_dir / config.CONFIG_FILENAME
    config.write_default_config(conf_path)

    yield conf_path

    shutil.rmtree(temp_dir)


def test_config_creation(conf_path):
    """Verify default configuration file creation.

    Checks that a newly created configuration file contains all default
    settings as expected.
    """
    conf = config.read_config(conf_path)
    for sectname, sectcont in conf.items():
        for k, val in sectcont.items():
            assert val == str(config.TEMPLATE_CONF_DICT[sectname][k])


def test_config_edit(tmp_path):
    """Test editing configuration values via CLI and direct access.

    Ensures that modifying a configuration value (e.g., 'brainglobe_dir')
    through the `config.write_config_value` function correctly updates the
    configuration and affects how `BrainGlobeAtlas` locates its root
    directory.
    """
    runner = CliRunner()
    result = runner.invoke(cli.bg_cli, ["config", "--show"])

    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    config_pre = config.read_config()
    original_bg_folder = config_pre["default_dirs"]["brainglobe_dir"]

    new_atlas_dir = tmp_path / "new_brainglobe_dir"
    new_atlas_dir.mkdir()

    config.write_config_value("brainglobe_dir", str(new_atlas_dir))
    config_post = config.read_config()

    assert config_post["default_dirs"]["brainglobe_dir"] == str(new_atlas_dir)

    atlas = bg_atlas.BrainGlobeAtlas(atlas_name="example_mouse_100um")
    assert atlas.root_dir.parent == new_atlas_dir

    config.write_config_value("brainglobe_dir", original_bg_folder)


@pytest.mark.parametrize(
    "key, show, value_factory, expected_output",
    [
        (
            "some_dir",
            False,
            lambda tmpdir: tmpdir.mkdir("valid").join("path"),
            "True",
        ),
        (
            "some_dir",
            False,
            lambda tmpdir: Path(str(tmpdir.join("invalid/path"))),
            "False",
        ),
        (
            "not_a_directory",
            False,
            lambda tmpdir: tmpdir.mkdir("valid").join("path"),
            "[DEFAULT]",
        ),
        (
            "not_a_directory",
            True,
            lambda tmpdir: tmpdir.mkdir("valid").join("path"),
            "[DEFAULT]",
        ),
    ],
)
def test_cli_modify_config(
    key, show, value_factory, expected_output, capsys, tmpdir
):
    """Test the `cli_modify_config` function used by the CLI "config" command.

    Checks that the `cli_modify_config` handles valid/invalid paths
    correctly and the behavior is as expected when `show` is set to True.
    Ensures that the start of the captured output (printed out by
    _print_config) is as expected.

    Parameters
    ----------
    key : str
        The configuration key to modify.
    show : bool
        Whether to show the configuration after modification.
    value_factory : callable
        A factory function to generate the value for the key, taking `tmpdir`
        as an argument.
    expected_output : str
        The expected start of the captured standard output.
    capsys : pytest.CaptureFixture
        Fixture to capture stdout/stderr.
    tmpdir : pytest.pathlib.Path
        A temporary directory fixture.
    """
    value = str(value_factory(tmpdir))
    cli_modify_config(key=key, value=value, show=show)
    captured = capsys.readouterr()
    assert captured.out.startswith(expected_output)
