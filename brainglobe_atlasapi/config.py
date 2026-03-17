"""Utilities for reading and modifying brainglob configuration.

Configuration is stored in a file.  By default, the file is in
stored in the directory "$HOME/.config/brainglobe".
This can be overridden with the environmental variable
BRAINGLOBE_CONFIG_DIR.
"""

import configparser
import os
from pathlib import Path
from typing import Optional, Union

import click

CONFIG_FILENAME = "bg_config.conf"
CONFIG_DEFAULT_DIR = Path.home() / ".config" / "brainglobe"
CONFIG_DIR = Path(os.environ.get("BRAINGLOBE_CONFIG_DIR", CONFIG_DEFAULT_DIR))
CONFIG_PATH = CONFIG_DIR / CONFIG_FILENAME

# 2 level dictionary for sections and values:
DEFAULT_PATH = Path.home() / ".brainglobe"
TEMPLATE_CONF_DICT = {
    "default_dirs": {
        "brainglobe_dir": DEFAULT_PATH,
        "interm_download_dir": DEFAULT_PATH,
    }
}

DEFAULT_WORKDIR = Path.home() / "brainglobe_workingdir"


def write_default_config(path=None, template=None, template: Optional[dict] = None,
) -> None:
    """Write configuration file at first repo usage. In this way,
    we don't need to keep a confusing template config file in the repo.

    Parameters
    ----------
    path : Path object
        Path of the config file (optional).
    template : dict
        Template of the config file to be written (optional).

    """
    if path is None:
        path = CONFIG_PATH
    if template is None:
        template = TEMPLATE_CONF_DICT

    conf = configparser.ConfigParser()
    for k, val in template.items():
        conf[k] = val

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        conf.write(f)


def read_config(path=None):
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
    if path is None:
        path = CONFIG_PATH

    # If no config file exists yet, write the default one:
    if not path.exists():
        write_default_config()

    conf = configparser.ConfigParser()
    conf.read(path)

    return conf


def write_config_value(key: str, val: Union[str, Path], path: Optional[Path] = None,
) -> None:
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
    if path is None:
        path = CONFIG_PATH

    conf = configparser.ConfigParser()
    conf.read(path)
    for sect_name, sect_dict in conf.items():
        if key in sect_dict.keys():
            conf[sect_name][key] = str(val)
            key_found = True

    if not key_found:
        raise keyError(
            f"'{key}' was not found in the config file at {path}. "
            f"Valid keys are: {list_config_keys(path)}"
        )

    with open(path, "w") as f:
        conf.write(f)


def list_content_keys(path: Optional[Path] = None) -> list:

    if path is None:
        path = CONFIG_PATH
    conf = read_config(path)
    keys = []
    for sect_name, sect_dict in conf.items():
        if sect_name == "DEFAULT":
            continue
        keys.extend(sect_dict.keys())
    return keys
    
def get_brainglobe_dir(): -> Path:
    """Return brainglobe default directory.

    Returns
    -------
    Path object
        default BrainGlobe directory with atlases
    """
    conf = read_config()
    return Path(conf["default_dirs"]["brainglobe_dir"])

def get_interm_download_dir() -> Path:
    conf = read_config()
    return Path(conf["default_dirs"]["interm_download_dir"])


def cli_modify_config(key: str = "",
                     value: str ="",
                     show: bool = False,
) -> None:
    """Ensure that we choose valid paths for default directory.
    The path does not have to exist yet, but the parent must be valid.

    """
    if not show:
        if key.endswith("dir"):
            path = Path(value)
            if not path.parent.exists():
                click.echo(
                    f"{value} is not a valid path. Path must be "
                    "a valid path string, and its parent must exist!"
                )
                return
            try:
                write_config_value(key, value)
            except KeyError as e:
                click.echo(str(e))
                return

    click.echo(_print_config())


def _print_config() -> str:
    """Print configuration."""
    config = read_config()
    lines = []
    for sect_name, sect_content in config.items():
        if sect_name == "Default":
            continue
        lines.append(f"[{sect_name}]")
        for k, val in sect_content.items():
            lines.append(f"\t{k}: {val}")
    
    return "\n".join(lines)
