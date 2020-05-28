import json
import tifffile
import numpy as np
import requests
import os
from tqdm.auto import tqdm


# ------------------------------- Web requests ------------------------------- #


def check_internet_connection(
    url="http://www.google.com/", timeout=5, raise_error=True
):
    """
        Check that there is an internet connection

        :param url: url to use for testing (Default value = 'http://www.google.com/')
        :param timeout:  timeout to wait for [in seconds] (Default value = 5)
    """

    try:
        _ = requests.get(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        if not raise_error:
            print("No internet connection available.")
        else:
            raise ValueError(
                "No internet connection, try again when you are connected to the internet."
            )
    return False


def retrieve_over_http(url, output_file_path):
    response = requests.get(url, stream=True)

    try:
        with tqdm.wrapattr(
            open(output_file_path, "wb"),
            "write",
            miniters=1,
            total=int(response.headers.get("content-length", 0)),
            desc=output_file_path.name,
        ) as fout:
            for chunk in response.iter_content(chunk_size=16384):
                fout.write(chunk)

    except requests.exceptions.ConnectionError:
        output_file_path.unlink()


# --------------------------------- File I/O --------------------------------- #
def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read_tiff(path):
    return tifffile.imread(str(path))


# -------------------------------- Folders I/O ------------------------------- #
def get_subdirs(folderpath):
    """
        Returns the subfolders in a given folder
    """
    return [f.path for f in os.scandir(folderpath) if f.is_dir()]


# ------------------------------- Data handling ------------------------------ #
def make_hemispheres_stack(shape):
    """ Make stack with hemispheres id. Assumes CCFv3 orientation.
    0: left hemisphere, 1:right hemisphere.
    :param shape: shape of the stack
    :return:
    """
    stack = np.zeros(shape, dtype=np.uint8)
    stack[(shape[0] // 2) :, :, :] = 1

    return stack
