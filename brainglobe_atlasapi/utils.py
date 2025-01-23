import configparser
import json
import logging
import re
from pathlib import Path
from time import sleep
from typing import Callable, Optional

import requests
import tifffile
from rich.panel import Panel
from rich.pretty import Pretty
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
from rich.table import Table
from rich.text import Text

from brainglobe_atlasapi import config

logging.getLogger("urllib3").setLevel(logging.WARNING)


def _rich_atlas_metadata(atlas_name, metadata):
    orange = "#f59e42"
    dimorange = "#b56510"
    gray = "#A9A9A9"
    mocassin = "#FFE4B5"
    cit_name, cit_link = metadata["citation"].split(", ")

    # Create a rich table
    tb = Table(
        box=None,
        show_lines=False,
        title=atlas_name.replace("_", " ").capitalize(),
        title_style=f"bold {orange}",
    )

    # Add entries to table
    tb.add_column(
        style=f"bold {mocassin}",
        justify="right",
        min_width=8,
        max_width=40,
    )
    tb.add_column(min_width=20, max_width=48)

    tb.add_row(
        "name:",
        Text.from_markup(
            metadata["name"] + f' [{gray}](v{metadata["version"]})'
        ),
    )
    tb.add_row("species:", Text.from_markup(f'[i]{metadata["species"]}'))
    tb.add_row("citation:", Text.from_markup(f"{cit_name} [{gray}]{cit_link}"))
    tb.add_row("link:", Text.from_markup(metadata["atlas_link"]))

    tb.add_row("")
    tb.add_row(
        "orientation:",
        Text.from_markup(f"[bold]{metadata['orientation']}"),
    )
    tb.add_row("symmetric:", Pretty(metadata["symmetric"]))
    tb.add_row("resolution:", Pretty(metadata["resolution"]))
    tb.add_row("shape:", Pretty(metadata["shape"]))

    # Fit into panel and yield
    panel = Panel.fit(tb, border_style=dimorange)
    return panel


def atlas_repr_from_name(name):
    """Generate dictionary with atlas description given the name."""
    parts = name.split("_")

    # if atlas name with no version:
    version_str = parts.pop() if not parts[-1].endswith("um") else None
    resolution_str = parts.pop()

    atlas_name = "_".join(parts)

    # For unspecified version:
    if version_str:
        major_vers, minor_vers = version_str[2:].split(".")
    else:
        major_vers, minor_vers = None, None

    return dict(
        name=atlas_name,
        major_vers=major_vers,
        minor_vers=minor_vers,
        resolution=resolution_str[:-2],
    )


def atlas_name_from_repr(name, resolution, major_vers=None, minor_vers=None):
    """Generate atlas name given a description."""
    if major_vers is None and minor_vers is None:
        return f"{name}_{resolution}um"
    else:
        return f"{name}_{resolution}um_v{major_vers}.{minor_vers}"


### Web requests


def check_internet_connection(
    url="http://www.google.com/", timeout=5, raise_error=True
):
    """Check that there is an internet connection
    url : str
        url to use for testing (Default value = 'http://www.google.com/')
    timeout : int
        timeout to wait for [in seconds] (Default value = 5).
    raise_error : bool
        if false, warning but no error.
    """

    try:
        _ = requests.get(url, timeout=timeout)

        return True
    except requests.ConnectionError as e:
        if not raise_error:
            print("No internet connection available.")
        else:
            raise ConnectionError(
                "No internet connection, try again when you are "
                "connected to the internet."
            ) from e

    return False


def check_gin_status(timeout=5, raise_error=True):
    """Check that the GIN server is up.

    timeout : int
        timeout to wait for [in seconds] (Default value = 5).
    raise_error : bool
        if false, warning but no error.
    """
    url = "https://gin.g-node.org/"

    try:
        _ = requests.get(url, timeout=timeout)

        return True
    except requests.ConnectionError as e:
        error_message = "GIN server is down."
        if not raise_error:
            print(error_message)
        else:
            raise ConnectionError(error_message) from e

    return False


def retrieve_over_http(
    url,
    output_file_path,
    fn_update: Optional[Callable[[int, int], None]] = None,
):
    """Download file from remote location, with progress bar.

    Parameters
    ----------
    url : str
        Remote URL.
    output_file_path : str or Path
        Full file destination for download.
    fn_update : Callable
        Handler function to update during download. Takes completed and total
        bytes.

    """
    # Make Rich progress bar
    progress = Progress(
        TextColumn("[bold]Downloading...", justify="right"),
        BarColumn(bar_width=None),
        "{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "• speed:",
        TransferSpeedColumn(),
        "• ETA:",
        TimeRemainingColumn(),
    )

    CHUNK_SIZE = 4096

    try:
        response = requests.get(url, stream=True)
        with progress:
            tot = int(response.headers.get("content-length", 0))

            if tot == 0:
                try:
                    tot = get_download_size(url)
                except Exception:
                    tot = 0

            task_id = progress.add_task(
                "download",
                filename=output_file_path.name,
                start=True,
                total=tot,
            )

            with open(output_file_path, "wb") as fout:
                completed = 0
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    fout.write(chunk)
                    adv = len(chunk)
                    completed += adv
                    progress.update(task_id, completed=min(completed, tot))

                    if fn_update:
                        # update handler with completed and total bytes
                        fn_update(completed, tot)

    except requests.exceptions.ConnectionError:
        output_file_path.unlink()
        raise requests.exceptions.ConnectionError(
            f"Could not download file from {url}"
        )


def get_download_size(url: str) -> int:
    """Get file size based on the MB value on the "src" page of each atlas

    Parameters
    ----------
    url : str
        atlas file url (in a repo, make sure the "raw" url is passed)

    Returns
    -------
    int
        size of the file to download

    Raises
    ------
        requests.exceptions.HTTPError: If there's an issue with HTTP request.
        ValueError: If the file size cannot be extracted from the response.
        IndexError: If the url is not formatted as expected

    """
    try:
        # Replace the 'raw' in the url with 'src'
        url_split = url.split("/")
        url_split[5] = "src"
        url = "/".join(url_split)

        response = requests.get(url)
        response.raise_for_status()

        response_string = response.content.decode("utf-8")
        search_result = re.search(
            r"([0-9]+\.[0-9] [MGK]B)|([0-9]+ [MGK]B)", response_string
        )

        assert search_result is not None

        size_string = search_result.group()

        assert size_string is not None

        size = float(size_string[:-3])
        prefix = size_string[-2]

        if prefix == "G":
            size *= 1e9
        elif prefix == "M":
            size *= 1e6
        elif prefix == "K":
            size *= 1e3

        return int(size)

    except requests.exceptions.HTTPError as e:
        raise e
    except AssertionError:
        raise ValueError("File size information not found in the response.")
    except IndexError:
        raise IndexError("Improperly formatted URL")


def conf_from_url(url) -> configparser.ConfigParser:
    """Read conf file from a URL. And cache a copy in the brainglobe dir.
    Parameters
    ----------
    url : str
        conf file url (in a repo, make sure the "raw" url is passed)

    Returns
    -------
    conf object

    """
    cache_path: Path = config.get_brainglobe_dir() / "last_versions.conf"

    result = requests.get(url)
    max_tries = 5
    sleep_time = 0.25

    while max_tries > 0 and result.status_code != 200:
        result = requests.get(url)
        max_tries -= 1
        sleep(sleep_time)
        sleep_time *= 2

    if result.status_code != 200:
        print(
            f"Could not fetch the latest atlas versions: {result.status_code}"
        )
        print(f"Using the last cached version from {cache_path}")

        return conf_from_file(cache_path)

    text = result.text
    config_obj = configparser.ConfigParser()
    config_obj.read_string(text)

    try:
        if not cache_path.parent.exists():
            cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Cache the available atlases
        with open(cache_path, "w") as f_out:
            config_obj.write(f_out)
    except OSError as e:
        print(f"Could not update the latest atlas versions cache: {e}")

    return config_obj


def conf_from_file(file_path: Path) -> configparser.ConfigParser:
    """Read conf file from a local file path.
    Parameters
    ----------
    file_path : Path
        conf file path (obtained from config.get_brainglobe_dir())

    Returns
    -------
    conf object if file available

    """
    if not file_path.exists():
        raise FileNotFoundError("Last versions cache file not found.")

    with open(file_path, "r") as file:
        text = file.read()

    config = configparser.ConfigParser()
    config.read_string(text)

    return config


### File I/O


def read_json(path):
    """Read a json file.

    Parameters
    ----------
    path : str or Path object

    Returns
    -------
    dict
        Dictionary from the json

    """
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_tiff(path):
    """Read a tiff file.

    Parameters
    ----------
    path : str or Path object

    Returns
    -------
    np.array
        Numpy stack read from the tiff.

    """
    return tifffile.imread(str(path))
