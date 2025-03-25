import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from brainglobe_atlasapi import bg_atlas, cli, config
from brainglobe_atlasapi.config import cli_modify_config


@pytest.fixture()
def conf_path():
    temp_dir = Path(tempfile.mkdtemp())
    conf_path = temp_dir / config.CONFIG_FILENAME
    config.write_default_config(conf_path)

    yield conf_path

    shutil.rmtree(temp_dir)


def test_config_creation(conf_path):
    conf = config.read_config(conf_path)
    for sectname, sectcont in conf.items():
        for k, val in sectcont.items():
            assert val == str(config.TEMPLATE_CONF_DICT[sectname][k])


def test_config_edit(tmp_path):
    # Create a CliRunner instance
    runner = CliRunner()
    result = runner.invoke(cli.bg_cli, ["config", "--show"])

    # Assert the command ran successfully
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    # Read the current config
    config_pre = config.read_config()
    original_bg_folder = config_pre["default_dirs"]["brainglobe_dir"]

    # Create a temporary directory for the new atlas directory
    new_atlas_dir = tmp_path / "new_brainglobe_dir"
    new_atlas_dir.mkdir()

    # Replace the brainglobe_dir in the config with the temp directory
    config.write_config_value("brainglobe_dir", str(new_atlas_dir))
    config_post = config.read_config()

    # Assert that the config was updated correctly
    assert config_post["default_dirs"]["brainglobe_dir"] == str(new_atlas_dir)

    # Use new location to download the atlas
    atlas = bg_atlas.BrainGlobeAtlas(atlas_name="example_mouse_100um")
    assert atlas.root_dir.parent == new_atlas_dir

    # Restore the original config value
    config.write_config_value("brainglobe_dir", original_bg_folder)


@pytest.mark.parametrize(
    "key, value_factory, expected_output",
    [
        (
            "temporal_dir",
            lambda tmpdir: tmpdir.mkdir("valid").join("path"),
            "True",
        ),
        (
            "temporal_dir",
            lambda tmpdir: Path("/invalid/path"),
            "False\n/invalid/path is not a valid path. "
            "Path must be a valid path string, and its parent must exist!",
        ),
    ],
)
def test_cli_modify_config(
    monkeypatch, key, value_factory, expected_output, capsys, tmpdir
):
    def mock_write_config_value(k, v):
        pass  # Mock function to bypass actual writing

    monkeypatch.setattr(
        "brainglobe_atlasapi.config.write_config_value",
        mock_write_config_value,
    )
    value = str(value_factory(tmpdir))
    cli_modify_config(key=key, value=value, show=False)
    captured = capsys.readouterr()
    assert expected_output in captured.out
