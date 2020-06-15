import pytest
import tempfile
from pathlib import Path
from brainatlas_api import config
from click.testing import CliRunner
from brainatlas_api import bg_atlas
import shutil


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


# This testing of the command line application does not really
# cange anything in the filesystem, so the repo config will remain unchanged:
def test_config_cli():
    runner = CliRunner()

    # Test printing of config file:
    result = runner.invoke(config.cli_modify_config, ["--show"])
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    # Correct edit (this does not really change the file):
    result = runner.invoke(
        config.cli_modify_config, [f"-k brainglobe_dir -v valid_path"]
    )
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"


# Ugly test zone: here we use the terminal commands, which edit the config
# file in the brainatlas_api repo from which the tests are being run.
# This is not the cleanest way, the alternative would be to run this test in
# a new env.
@pytest.mark.slow
def test_config_edit():
    runner = CliRunner()
    result = runner.invoke(config.cli_modify_config, ["--show"])
    assert result.exit_code == 0
    assert result.output == config._print_config() + "\n"

    config_pre = config.read_config()
    original_bg_folder = config_pre["default_dirs"]["brainglobe_dir"]

    new_atlas_dir = Path(tempfile.mkdtemp())
    config.write_config_value("brainglobe_dir", new_atlas_dir)
    config_post = config.read_config()

    assert config_post["default_dirs"]["brainglobe_dir"] == str(new_atlas_dir)

    # Use new location to download:
    atlas = bg_atlas.ExampleAtlas()

    assert atlas.root_dir.parent == new_atlas_dir

    # Fix the mess:
    config.write_config_value("brainglobe_dir", original_bg_folder)

    # cleanup:
    shutil.rmtree(new_atlas_dir)
