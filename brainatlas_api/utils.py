import json
import tifffile
import numpy as np

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