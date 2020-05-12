import json
import tifffile


def read_tiff(path):
    return tifffile.imread(str(path))



# ------------------------------------ HDF ----------------------------------- #
def open_hdf(filepath):
    check_file_exists(filepath, raise_error=True)
    f =  h5py.File(filepath, 'r')

    # List all groups and subgroups
    keys = list(f.keys())
    subkeys = {} 
    for k in keys:
        try:
            subk = list(f[k].keys())
            subkeys[k] = subk
        except:
            pass

    all_keys = []
    f.visit(all_keys.append)

    return f, keys, subkeys, all_keys



# --------------------------------- CSV FILES -------------------------------- #
def load_excel_file(filepath):
    check_file_exists(filepath, raise_error=True)
    return pyexcel.get_records(file_name=filepath)


def create_csv_file(filepath, fieldnames):
    with open(filepath, "a", newline="") as f:
        logger = csv.DictWriter(f, fieldnames=fieldnames)
        logger.writeheader()


def append_csv_file(csv_file, row, fieldnames):
    check_file_exists(csv_file, raise_error=True)
    with open(csv_file, "a", newline="") as f:
        logger = csv.DictWriter(f, fieldnames=fieldnames)
        logger.writerow(row)


def load_csv_file(csv_file):
    check_file_exists(csv_file, raise_error=True)
    return pd.read_csv(csv_file)


# ----------------------------------- JSON ----------------------------------- #
def save_json(filepath, content, append=False):
    """
	Saves content to a JSON file

	:param filepath: path to a file (must include .json)
	:param content: dictionary of stuff to save

	"""
    if not "json" in filepath:
        raise ValueError("filepath is invalid")

    if not append:
        with open(filepath, "w") as json_file:
            json.dump(content, json_file, indent=4)
    else:
        with open(filepath, "w+") as json_file:
            json.dump(content, json_file, indent=4)


def load_json(filepath):
    """
	Load a JSON file

	:param filepath: path to a file

	"""
    if not os.path.isfile(filepath) or not ".json" in filepath.lower():
        raise ValueError("unrecognized file path: {}".format(filepath))
    with open(filepath) as f:
        data = json.load(f)
    return data


# ----------------------------------- YAML ----------------------------------- #
def save_yaml(filepath, content, append=False, topcomment=None):
    """
	Saves content to a yaml file

	:param filepath: path to a file (must include .yaml)
	:param content: dictionary of stuff to save

	"""
    if not "yaml" in filepath and not "yml" in filepath:
        raise ValueError("filepath is invalid")

    if not append:
        method = "w"
    else:
        method = "w+"

    with open(filepath, method) as yaml_file:
        if topcomment is not None:
            yaml_file.write(topcomment)
        yaml.dump(content, yaml_file, default_flow_style=False, indent=4)


def load_yaml(filepath):
    """
	Load a YAML file

	:param filepath: path to yaml file

	"""
    if filepath is None or not os.path.isfile(filepath):
        raise ValueError("unrecognized file path: {}".format(filepath))
    if not "yml" in filepath and not "yaml" in filepath:
        raise ValueError("unrecognized file path: {}".format(filepath))
    return yaml.load(open(filepath), Loader=yaml.FullLoader)



# ---------------------------------- PICKLE ---------------------------------- #
def save_pickle(filepath, data):
    with open(filepath, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(filepath):
    """
	Load a pickle file

	:param filepath: path to pickle file

	"""
    if filepath is None or not os.path.isfile(filepath):
        raise ValueError("unrecognized file path: {}".format(filepath))
    if not "pkl" in filepath and not "pickle" in filepath:
        raise ValueError("unrecognized file path: {}".format(filepath))

    with open(filepath, 'rb') as handle:
        data = pickle.load(handle)
    return data    