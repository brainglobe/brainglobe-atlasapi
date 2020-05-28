import tempfile
import json
import tarfile
import tifffile

import pandas as pd

from pathlib import Path
from brainio.brainio import load_any
from allensdk.core.reference_space_cache import ReferenceSpaceCache


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
loaded = load_any(
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
tifffile.imsave(str(uncompr_atlas_path / "annotations.tiff"), loaded)

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
    structure["structure_id_path"].append(structure["id"])

root = {
    "acronym": "root",
    "id": 997,
    "name": "root",
    "structure_id_path": [997],
    "rgb_triplet": [255, 255, 255],
}

structures.append(root)

# save regions list json:
with open(uncompr_atlas_path / "structures.json", "w") as f:
    json.dump(structures, f)


# Create meshes
# ######################################


# Write metadata
# ######################################
metadata_dict = {
    "name": ATLAS_NAME,
    "citation": "Chon et al. 2019, https://doi.org/10.1038/s41467-019-13057-w",
    "atlas_link": "https://kimlab.io/brain-map/atlas/",
    "species": "Mus musculus",
    "symmetric": True,
    "resolution": (RES_UM, RES_UM, RES_UM),
    "shape": template_volume.shape,
}

with open(uncompr_atlas_path / "atlas_metadata.json", "w") as f:
    json.dump(metadata_dict, f)

output_filename = bg_root_dir / f"{uncompr_atlas_path.name}.tar.gz"
with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(uncompr_atlas_path, arcname=uncompr_atlas_path.name)

# Clean temporary directory and remove it:
for f in downloading_path.glob("*"):
    f.unlink()
downloading_path.rmdir()
