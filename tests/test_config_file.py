import shutil
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from bg_atlasapi import bg_atlas, cli, config


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


# Ugly test zone: here we use the terminal commands, which edit the config
# file in the bg_atlasapi repo from which the tests are being run.
# This is not the cleanest way, the alternative would be to run this test in
# a new env.
@pytest.mark.slow
def test_config_edit():
    runner = CliRunner()
    result = runner.invoke(cli.bg_cli, ["config", "--show"])
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    config_pre = config.read_config()
    original_bg_folder = config_pre["default_dirs"]["brainglobe_dir"]

    new_atlas_dir = Path(tempfile.mkdtemp())
    config.write_config_value("brainglobe_dir", new_atlas_dir)
    config_post = config.read_config()

    assert config_post["default_dirs"]["brainglobe_dir"] == str(new_atlas_dir)

    # Use new location to download:
    atlas = bg_atlas.BrainGlobeAtlas(atlas_name="example_mouse_100um")

    assert atlas.root_dir.parent == new_atlas_dir

    # Fix the mess:
    config.write_config_value("brainglobe_dir", original_bg_folder)

    # cleanup:
    shutil.rmtree(new_atlas_dir)
