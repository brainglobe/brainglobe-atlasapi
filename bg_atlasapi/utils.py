import json
import tifffile
import requests
import logging
import configparser
from tqdm.auto import tqdm

logging.getLogger("urllib3").setLevel(logging.WARNING)


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


# ------------------------------- Web requests ------------------------------- #


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
    except requests.ConnectionError:
        if not raise_error:
            print("No internet connection available.")
        else:
            raise ConnectionError(
                "No internet connection, try again when you are connected to the internet."
            )
    return False


def retrieve_over_http(url, output_file_path):
    """Download file from remote location, with progress bar.

    Parameters
    ----------
    url : str
        Remote URL.
    output_file_path : str or Path
        Full file destination for download.

    """
    CHUNK_SIZE = 4096
    response = requests.get(url, stream=True)

    try:
        with tqdm.wrapattr(
            open(output_file_path, "wb"),
            "write",
            miniters=1,
            total=int(response.headers.get("content-length", 0)),
            desc=output_file_path.name,
        ) as fout:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                fout.write(chunk)

    except requests.exceptions.ConnectionError:
        output_file_path.unlink()
        raise requests.exceptions.ConnectionError(
            f"Could not download file from {url}"
        )


def conf_from_url(url):
    """Read conf file from an URL.
    Parameters
    ----------
    url : str
        conf file url (in a repo, make sure the "raw" url is passed)

    Returns
    -------
    conf object

    """
    text = requests.get(url).text
    config = configparser.ConfigParser()
    config.read_string(text)

    return config


# --------------------------------- File I/O --------------------------------- #
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
