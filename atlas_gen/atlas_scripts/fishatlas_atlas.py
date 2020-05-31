from atlas_gen.atlas_scripts.fishatlas_utils import (
    add_path_inplace,
    collect_all_inplace,
)

from pathlib import Path
import tempfile
from brainatlas_api.utils import retrieve_over_http
import json
import tarfile

import nrrd
import numpy as np
import tifffile
import requests

ATLAS_NAME = "fishatlas"

base_url = r"https://fishatlas.neuro.mpg.de"


# Generated atlas path:
bg_root_dir = Path.home() / "brainglobe"
bg_root_dir.mkdir(exist_ok=True)

# Temporary folder for nrrd files download:
temp_path = Path(tempfile.mkdtemp())
download_dir_path = temp_path / "downloading_path"
download_dir_path.mkdir()

# Temporary folder for files before compressing:
uncompr_atlas_path = temp_path / ATLAS_NAME
uncompr_atlas_path.mkdir()

# Download reference:
#####################
reference_url = f"{base_url}/media/brain_browser/Brain/MovieViewBrain/standard_brain_fixed_SYP_T_GAD1b.nrrd"
out_file_path = download_dir_path / "reference.nrrd"

retrieve_over_http(reference_url, out_file_path)

# Cleanup to have in brainglobe order:
refstack_axes = (1, 2, 0)
refstack_flips = [False, True, False]

refstack, h = nrrd.read(str(out_file_path))

refstack = refstack.transpose(refstack_axes)
for i, flip in enumerate(refstack_flips):
    if flip:
        refstack = np.flip(refstack, i)


tifffile.imsave(str(uncompr_atlas_path / "reference.tiff"), refstack)

# Download structures tree and meshes:
######################################
regions_url = f"{base_url}/neurons/get_brain_regions"

meshes_dir_path = uncompr_atlas_path / "meshes"
meshes_dir_path.mkdir(exist_ok=True)

# Download structures hierarchy:
regions = requests.get(regions_url).json()["brain_regions"]

# Initiate dictionary with root info:
regions_dict = {
    "name": "root",
    "id": 0,
    "sub_regions": regions.copy(),
    "structure_id_path": [],
    "acronym": "root",
    "files": {
        "file_3D": "/media/Neurons_database/Brain_and_regions/Brains/Outline/Outline_new.txt"
    },
    "color": "#ffffff",
}

# Go through the regions hierarchy and create the structure path entry:
add_path_inplace(regions_dict)

# Create empty list and collect all regions traversing the regions hierarchy:
regions_list = []
collect_all_inplace(
    regions_dict,
    regions_list,
    meshes_dir_path,
    refstack_axes,
    refstack_flips,
    refstack.shape,
)

# save regions list json:
with open(uncompr_atlas_path / "structures.json", "w") as f:
    json.dump(regions_list, f)

# Write metadata:
#################
metadata_dict = {
    "name": ATLAS_NAME,
    "citation": "Kunst et al 2019, https://doi.org/10.1016/j.neuron.2019.04.034",
    "atlas_link": "https://fishatlas.neuro.mpg.de",
    "species": "Danio rerio",
    "symmetric": False,
    "resolution": (0.994, 1, 0.994),
    "shape": refstack.shape,
}

with open(uncompr_atlas_path / "atlas_metadata.json", "w") as f:
    json.dump(metadata_dict, f)

# Compress folder:
output_filename = bg_root_dir / f"{uncompr_atlas_path.name}.tar.gz"
with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(uncompr_atlas_path, arcname=uncompr_atlas_path.name)

# Clean temporary directory and remove it:
for f in download_dir_path.glob("*"):
    f.unlink()
download_dir_path.rmdir()
