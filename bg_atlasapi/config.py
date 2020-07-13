import configparser
from pathlib import Path
from pkg_resources import resource_filename
import click

CONFIG_FILENAME = "bg_config.conf"
CONFIG_PATH = Path(resource_filename("bg_atlasapi", CONFIG_FILENAME))

# 2 level dictionary for sections and values:
DEFAULT_PATH = Path.home() / ".brainglobe"
TEMPLATE_CONF_DICT = {
    "default_dirs": {
        "brainglobe_dir": DEFAULT_PATH,
        "interm_download_dir": DEFAULT_PATH,
    }
}


def write_default_config(path=CONFIG_PATH, template=TEMPLATE_CONF_DICT):
    """Write configuration file at first repo usage. In this way,
    we don't need to keep a confusing template config file in the repo.

    Parameters
    ----------
    path : Path object
        Path of the config file (optional).
    template : dict
        Template of the config file to be written (optional).

    """

    conf = configparser.ConfigParser()
    for k, val in template.items():
        conf[k] = val

    with open(path, "w") as f:
        conf.write(f)


def read_config(path=CONFIG_PATH):
    """Read BrainGlobe config.

    Parameters
    ----------
    path : Path object
        Path of the config file (optional).

    Returns
    -------
    ConfigParser object
        brainglobe configuration
    """

    # If no config file exists yet, write the default one:
    if not path.exists():
        write_default_config()

    conf = configparser.ConfigParser()
    conf.read(path)

    return conf


def write_config_value(key, val, path=CONFIG_PATH):
    """Write a new value in the config file. To make things simple, ignore
    sections and look directly for matching parameters names.

    Parameters
    ----------
    key : str
        Name of the parameter to configure.
    val :
        New value.
    path : Path object
        Path of the config file (optional).

    """
    conf = configparser.ConfigParser()
    conf.read(path)
    for sect_name, sect_dict in conf.items():
        if key in sect_dict.keys():
            conf[sect_name][key] = str(val)

    with open(CONFIG_PATH, "w") as f:
        conf.write(f)


def get_brainglobe_dir():
    """Return brainglobe default directory.

    Returns
    -------
    Path object
        default BrainGlobe directory with atlases
    """
    conf = read_config()
    return Path(conf["default_dirs"]["brainglobe_dir"])


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
    """Print configuration.
    """
    config = read_config()
    string = ""
    for sect_name, sect_content in config.items():
        string += f"[{sect_name}]\n"
        for k, val in sect_content.items():
            string += f"\t{k}: {val}\n"

    return string
