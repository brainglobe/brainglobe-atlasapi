import json
import tifffile
import numpy as np
import requests
from tqdm.auto import tqdm


def retrieve_over_http(url, output_file_path):
    response = requests.get(url, stream=True)

    try:
        with tqdm.wrapattr(open(output_file_path, "wb"), "write", miniters=1,
                           total=int(response.headers.get('content-length', 0)),
                           desc=output_file_path.name) as fout:
            for chunk in response.iter_content(chunk_size=16384):
                fout.write(chunk)

    except requests.exceptions.ConnectionError:
        output_file_path.unlink()

def open_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def read_tiff(path):
    return tifffile.imread(str(path))

def make_hemispheres_stack(shape):
    """ Make stack with hemispheres id. Assumes CCFv3 orientation.
    0: left hemisphere, 1:right hemisphere.
    :param shape: shape of the stack
    :return:
    """
    stack = np.zeros(shape, dtype=np.uint8)
    stack[(shape[0] // 2):, :, :] = 1

    return stack