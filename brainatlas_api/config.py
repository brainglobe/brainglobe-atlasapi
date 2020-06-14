import configparser
from pathlib import Path

# Might be problematic if module is refactored:
CONFIG_PATH = Path(__file__).parent.parent / "bg_config.conf"

# 2 level dict for sections and values:
DEFAULT_PATH = Path.home() / ".brainglobe"
TEMPLATE_CONF_DICT = {
    "default_dirs": {"atlas_dir": DEFAULT_PATH, "download_dir": DEFAULT_PATH}
}


def write_default_conf():
    """ Write configuration file at first repo usage. In this way,
    we don't need to keep a template config file.

    Parameters
    ----------
    file_path

    Returns
    -------

    """

    config = configparser.ConfigParser()
    for k, val in TEMPLATE_CONF_DICT.items():
        config[k] = val

    with open(CONFIG_PATH, "w") as f:
        config.write(f)


def read_config():
    """
    Returns
    -------

    """

    # If no config file exists yet, write the default one:
    if not CONFIG_PATH.exists():
        write_default_conf(CONFIG_PATH)

    config = configparser.ConfigParser()
    config.read(CONFIG_PATH)

    return config
