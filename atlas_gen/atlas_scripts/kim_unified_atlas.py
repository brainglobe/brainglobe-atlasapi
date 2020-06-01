import tempfile
import json
import tarfile
import tifffile
import os


import pandas as pd
import numpy as np


from pathlib import Path
from tqdm import tqdm
from vtkplotter import write, Volume

from brainio.brainio import load_any
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from atlas_gen.volume_utils import (
    extract_volume_surface,
    load_labelled_volume,
)
from atlas_gen.metadata_utils import create_metadata_files


paxinos_allen_directory = Path(
    "/media/adam/Storage/cellfinder/data/paxinos_allen/"
)
annotations_image = paxinos_allen_directory / "annotations_coronal.tif"
structures_file = paxinos_allen_directory / "structures.csv"

# assume isotropic
ANNOTATIONS_RES_UM = 10

RES_UM = 50
ATLAS_NAME = f"kim_unified{RES_UM}um"

# Generated atlas path:
bg_root_dir = Path.home() / "brainglobe"
bg_root_dir.mkdir(exist_ok=True)

# Temporary folder for nrrd files download:
temp_path = Path(tempfile.mkdtemp())
downloading_path = temp_path / "downloading_path"
downloading_path.mkdir()

# Temporary folder for files before compressing:
uncompr_atlas_path = temp_path / ATLAS_NAME
uncompr_atlas_path.mkdir()


# Load (and possibly downsample) annotated volume:
#########################################
scaling_factor = ANNOTATIONS_RES_UM / RES_UM
print(
    f"Loading: {annotations_image.name} and downscaling by: {scaling_factor}"
)
annotations_array = load_any(
    annotations_image,
    x_scaling_factor=scaling_factor,
    y_scaling_factor=scaling_factor,
    z_scaling_factor=scaling_factor,
    anti_aliasing=False,
)


# Download template volume:
#########################################
spacecache = ReferenceSpaceCache(
    manifest=downloading_path / "manifest.json",
    # downloaded files are stored relative to here
    resolution=RES_UM,
    reference_space_key="annotation/ccf_2017"
    # use the latest version of the CCF
)

# Download
print("Downloading template file")
template_volume, _ = spacecache.get_template_volume()
print("Download completed...")


# Save tiff stacks:
tifffile.imsave(str(uncompr_atlas_path / "reference.tiff"), template_volume)
del template_volume
tifffile.imsave(
    str(uncompr_atlas_path / "annotations.tiff"), annotations_array
)

# Parse region names & hierarchy
# ######################################

df = pd.read_csv(structures_file)
df = df.drop(columns=["Unnamed: 0", "parent_id", "parent_acronym"])

# split by "/" and convert list of strings to list of ints
df["structure_id_path"] = (
    df["structure_id_path"]
    .str.split(pat="/")
    .map(lambda x: [int(i) for i in x])
)

structures = df.to_dict("records")

for structure in structures:
    structure.update({"rgb_triplet": [255, 255, 255]})
    # root doesn't have a parent
    if structure["id"] != 997:
        structure["structure_id_path"].append(structure["id"])

# save regions list json:
with open(uncompr_atlas_path / "structures.json", "w") as f:
    json.dump(structures, f)


# Create meshes
# ######################################
print(f"Saving atlas data at {uncompr_atlas_path}")
meshes_dir_path = uncompr_atlas_path / "meshes"
meshes_dir_path.mkdir(exist_ok=True)

volume = load_labelled_volume(annotations_array)

root = extract_volume_surface(volume)

write(root, str(meshes_dir_path / "997.obj"))
del volume
del root

non_minor_regions = []
# First create a mesh for every minor region
for region in tqdm(structures):

    savepath = str(
        meshes_dir_path
        / f'{region["id"]}.obj'.replace("/", "-").replace("\\", "-")
    )
    if os.path.isfile(savepath):
        continue

    vol = np.zeros_like(
        annotations_array
    )  # this is made easy by volume_utils.create_masked_array

    if not np.isin(np.float(region["id"]), annotations_array):
        print(
            f'{region["name"]} is not a terminal region, so will be extracted separately'
        )
        non_minor_regions.append(region)
        continue

    vol[annotations_array == np.float32(np.float(region["id"]))] = 1
    if np.max(vol) < 1:
        raise ValueError

    write(extract_volume_surface(Volume(vol)), savepath)

# Get other regions
for region in non_minor_regions:
    print(f'finding subregions for: {region["name"]}')

    savepath = str(
        meshes_dir_path
        / f'{region["id"]}.obj'.replace("/", "-").replace("\\", "-")
    )
    if os.path.isfile(savepath):
        continue

    vol = np.zeros_like(
        annotations_array
    )  # this is made easy by volume_utils.create_masked_array

    # I've moved this code to atlas_gen.structures.get_structure_children
    sub_region_ids = []
    for subregion in structures:
        if region["id"] in subregion["structure_id_path"]:
            sub_region_ids.append(subregion["id"])

    if sub_region_ids == []:
        print(f'{region["acronym"]} doesnt seem to contain any other regions')
        continue

    # this is made easy by volume_utils.create_masked_array
    vol[np.isin(annotations_array, sub_region_ids)] = 1
    if np.max(vol) < 1:
        continue

    write(extract_volume_surface(Volume(vol)), savepath)
print("Finished mesh extraction")

# Write metadata
# ######################################
metadata_dict = {
    "name": ATLAS_NAME,
    "citation": "Chon et al. 2019, https://doi.org/10.1038/s41467-019-13057-w",
    "atlas_link": "https://kimlab.io/brain-map/atlas/",
    "species": "Mus musculus",
    "symmetric": True,
    "resolution": (RES_UM, RES_UM, RES_UM),
    "shape": annotations_array.shape,
}

with open(uncompr_atlas_path / "atlas_metadata.json", "w") as f:
    json.dump(metadata_dict, f)

# Create human readable files
create_metadata_files(uncompr_atlas_path, metadata_dict, structures)

output_filename = bg_root_dir / f"{uncompr_atlas_path.name}.tar.gz"
with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(uncompr_atlas_path, arcname=uncompr_atlas_path.name)

# Clean temporary directory and remove it:
for f in downloading_path.glob("*"):
    f.unlink()
downloading_path.rmdir()
print("Finished atlas generation!")
