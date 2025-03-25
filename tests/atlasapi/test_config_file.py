import os
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
    "key, value_factory, expected_output_factory",
    [
        (
            "some_dir",
            lambda tmpdir: tmpdir.mkdir("valid").join("path"),
            lambda tmpdir: "True",
        ),  # Creating a valid temp path
        (
            "some_dir",
            lambda tmpdir: Path(str(tmpdir.join("invalid/path"))),
            lambda tmpdir: f"False{os.linesep}{str(tmpdir.join('invalid/path'))} is not a valid path. Path must be a valid path string, and its parent must exist!",
        ),
    ],
)
def test_cli_modify_config(
    monkeypatch, key, value_factory, expected_output_factory, capsys, tmpdir
):
    def mock_write_config_value(k, v):
        pass

    monkeypatch.setattr(
        "brainglobe_atlasapi.config.write_config_value",
        mock_write_config_value,
    )
    value = str(value_factory(tmpdir))
    expected_output = expected_output_factory(tmpdir)
    cli_modify_config(key=key, value=value, show=False)
    captured = capsys.readouterr()
    captured_output = captured.out.replace("\\", "/").strip()
    expected_output = expected_output.replace("\\", "/").strip()
    assert expected_output in captured_output
