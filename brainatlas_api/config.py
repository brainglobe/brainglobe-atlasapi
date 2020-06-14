import configparser
from pathlib import Path
import click

# Might be problematic if module is refactored in subdir, but:
CONFIG_PATH = Path(__file__).parent.parent / "bg_config.conf"

# 2 level dictionary for sections and values:
DEFAULT_PATH = Path.home() / ".brainglobe"
TEMPLATE_CONF_DICT = {
    "default_dirs": {
        "brainglobe_dir": DEFAULT_PATH,
        "interm_download_dir": DEFAULT_PATH,
    }
}


def write_default_config():
    """Write configuration file at first repo usage. In this way,
    we don't need to keep a confusing template config file in the repo.

    Parameters
    ----------
    file_path

    Returns
    -------

    """

    conf = configparser.ConfigParser()
    for k, val in TEMPLATE_CONF_DICT.items():
        conf[k] = val

    with open(CONFIG_PATH, "w") as f:
        conf.write(f)


def read_config():
    """
    Returns
    -------

    """

    # If no config file exists yet, write the default one:
    if not CONFIG_PATH.exists():
        write_default_config()

    conf = configparser.ConfigParser()
    conf.read(CONFIG_PATH)

    return conf


def write_config_value(key, val):
    """Write a new value in the config file. To make things simple, ignore
    sections and look directly for matching parameters names.

    Parameters
    ----------
    key : str
        Name of the parameter to configure.
    val :
        New value

    """
    conf = configparser.ConfigParser()
    conf.read(CONFIG_PATH)
    for sect_name, sect_dict in conf.items():
        if key in sect_dict.keys():
            conf[sect_name][key] = val

    with open(CONFIG_PATH, "w") as f:
        conf.write(f)


@click.command()
@click.option("-k", "--key")
@click.option("-v", "--value")
@click.option("-s", "--show", is_flag=True)
def cli_modify_config(key=0, value=0, show=False):
    # Ensure that we choose valid paths for default directory. The path does
    # not have to exist yet, but the parent must be valid:
    if not show:
        if key[-3:] == "dir":
            path = Path(value)
            click.echo(path.parent.exists())
            if not path.parent.exists():
                click.echo(
                    f"{value} is not a valid path. Path must be a valid path string, and its parent must exist!"
                )
                return
        write_config_value(key, value)

    click.echo(_print_config())


def _print_config():
    config = read_config()
    string = ""
    for sect_name, sect_content in config.items():
        string += f"[{sect_name}]\n"
        for k, val in sect_content.items():
            string += f"\t{k}: {val}\n"

    return string
