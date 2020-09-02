from atlas_gen.volume_utils import (
    extract_volume_surface,
    load_labelled_volume,
)
from atlas_gen.metadata_utils import create_metadata_files
from brainio.brainio import load_any


from pathlib import Path
import pandas as pd
import json
import tarfile
import os

import numpy as np
import tifffile


from tqdm import tqdm

from vtkplotter import write, Volume


import sys

sys.path.append(os.getcwd())


ATLAS_NAME = "ratatlas"

base_url = ""

# Generated atlas path:
bg_root_dir = Path.home() / ".brainglobe"
bg_root_dir.mkdir(exist_ok=True)

# Temporary folder for nrrd files download:
# temp_path = Path(tempfile.mkdtemp())
temp_path = Path(
    r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy\Atlases\atlasesforbrainrender\goldcustomrat"
)


# Temporary folder for files before compressing:
uncompr_atlas_path = temp_path / ATLAS_NAME
uncompr_atlas_path.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------- #
#                               Load volume data                               #
# ---------------------------------------------------------------------------- #
# Load annotated and reference tiff stacks (already aligned to brainglobe by Adam)
# And save to folder with all atlas data
base_data_fld = Path(
    r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy\Atlases\atlasesforbrainrender\goldcustomrat"
)

for name in ["reference", "annotated"]:
    loaded = load_any(
        str(base_data_fld / f"{name}.tif")
    )  # shape (186, 160, 160)
    tifffile.imsave(str(uncompr_atlas_path / f"{name}.tiff"), loaded)


# ---------------------------------------------------------------------------- #
#                         LOAD/PARSE HIERARCHICAL DATA                         #
# ---------------------------------------------------------------------------- #

"""
    Hierarchy is organized:

    /major/submajor/minor

    hierarchy dataframe maps region names to voxel value in annotated.tiff (minor column)
    major and submajors map major/submajor values in hierarchy to the corresponding name
"""

hierarchy = pd.read_excel(
    str(base_data_fld / "SwansonAtlasCategories-Mar_2_2005.xls"),
    header=1,
    usecols=["Abbreviation", "Name of Area", "Major", "Sub_Major", "Minor"],
    nrows=1276,
)

majors = pd.read_excel(
    str(base_data_fld / "SwansonAtlasCategories-Mar_2_2005.xls"),
    header=3,
    usecols=[13, 14],
    nrows=20,
)

submajors = pd.read_excel(
    str(base_data_fld / "SwansonAtlasCategories-Mar_2_2005.xls"),
    header=3,
    usecols=[15, 16],
    nrows=89,
)


clean_hierarchy = dict(
    abbreviation=[],
    name=[],
    major=[],
    majornum=[],
    submajor=[],
    submajornum=[],
    minor=[],
)
for i, region in hierarchy.iterrows():
    clean_hierarchy["abbreviation"].append(region.Abbreviation)
    clean_hierarchy["name"].append(region["Name of Area"])
    clean_hierarchy["major"].append(
        majors.loc[majors.ANC == region.Major]["ANC Name"].values[0]
    )
    clean_hierarchy["majornum"].append(
        majors.loc[majors.ANC == region.Major]["ANC"].values[0]
    )
    clean_hierarchy["minor"].append(region["Minor"])
    try:
        clean_hierarchy["submajor"].append(
            submajors.loc[submajors.SubANC == region.Sub_Major][
                "SubANC Name"
            ].values[0]
        )
        clean_hierarchy["submajornum"].append(
            int(
                submajors.loc[submajors.SubANC == region.Sub_Major][
                    "SubANC"
                ].values[0]
            )
        )
    except Exception as e:
        print(e)
        clean_hierarchy["submajor"].append(None)
        clean_hierarchy["submajornum"].append(None)


clean_hierarchy = pd.DataFrame(clean_hierarchy)

# ------------------------ Organize hierarchy metadata ----------------------- #

idn = 0

"""
    Given that the way the matadata is organised, not every region has a unique 
    numerical ID value associated with it (e.g. a region might have a minor 1, but
    a submajor region's numerical value is also 1), here we reassign a numerical id 
    to each brain structure. number increase from root > majors > submajors > minors.
"""

structures = [
    {
        "acronym": "root",
        "id": idn,
        "name": "root",
        "structure_id_path": [0],
        "rgb_triplet": [255, 255, 255],
    }
]


for i, major in majors.iterrows():
    if not isinstance(major["ANC Name"], str):
        continue

    idn += 1
    structures.append(
        {
            "acronym": major["ANC Name"].replace(" ", "-"),
            "id": idn,
            "name": major["ANC Name"],
            "structure_id_path": [0, idn],
            "rgb_triplet": [255, 255, 255],
        }
    )


for i, submajor in submajors.iterrows():
    # Get an entry in clean hierarchy with this submajor
    try:
        entry = clean_hierarchy.loc[
            clean_hierarchy.submajornum == submajor["SubANC"]
        ].iloc[0]
    except Exception as e:
        print(e)
        pass

    # Get path
    idn += 1
    path = [0, int(entry.majornum), idn]

    # Append
    structures.append(
        {
            "acronym": submajor["SubANC Name"].replace(" ", "-"),
            "id": idn,
            "name": submajor["SubANC Name"],
            "structure_id_path": path,
            "rgb_triplet": [255, 255, 255],
        }
    )


for i, region in clean_hierarchy.iterrows():
    idn += 1
    if np.isnan(region.submajornum):
        path = [0, region.majornum, idn]

    else:
        path = [0, int(region.majornum), int(region.submajornum), idn]

    structures.append(
        {
            "acronym": region.abbreviation,
            "id": idn,
            "name": region.name,
            "structure_id_path": path,
            "rgb_triplet": [255, 255, 255],
        }
    )

# save regions list json:
with open(uncompr_atlas_path / "structures.json", "w") as f:
    json.dump(structures, f)


# ---------------------------------------------------------------------------- #
#                                 Create MESEHS                                #
# ---------------------------------------------------------------------------- #
print(f"Saving atlas data at {uncompr_atlas_path}")
meshes_dir_path = uncompr_atlas_path / "meshes"
meshes_dir_path.mkdir(exist_ok=True)

volume = load_labelled_volume(load_any(str(base_data_fld / "annotated.tif")))

root = extract_volume_surface(volume)

write(root, str(meshes_dir_path / "0.obj"))

# First create a mesh for every minor region
volume_data = load_any(str(base_data_fld / "annotated.tif"))
for i, region in tqdm(clean_hierarchy.iterrows()):
    structure = [
        s for s in structures if s["acronym"] == region["abbreviation"]
    ][0]
    savepath = str(
        meshes_dir_path
        / f'{structure["id"]}.obj'.replace("/", "-").replace("\\", "-")
    )
    if os.path.isfile(savepath):
        continue

    vol = np.zeros_like(volume_data)

    if not np.isin(np.float(region.minor), volume_data):
        # print(f'{region.abbreviation} doesnt seem to appear in annotated dataset')
        continue

    vol[volume_data == np.float32(region.minor)] = 1
    if np.max(vol) < 1:
        raise ValueError

    write(extract_volume_surface(Volume(vol)), savepath)


# Create a mesh for every submajor and major region
for i, submajor in tqdm(submajors.iterrows()):
    structure = [
        s
        for s in structures
        if s["acronym"] == submajor["SubANC Name"].replace(" ", "-")
    ][0]
    savepath = str(
        meshes_dir_path
        / f'{structure["id"]}.obj'.replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    if os.path.isfile(savepath):
        continue

    regions = list(
        clean_hierarchy.loc[
            clean_hierarchy.submajor == submajor["SubANC Name"]
        ].minor.values
    )
    if not regions:
        continue

    vol = np.zeros_like(volume_data)

    for region in regions:
        vol[volume_data == region] = 1

    if np.max(vol) < 1:
        continue

    write(extract_volume_surface(Volume(vol)), savepath)


for i, major in tqdm(majors.iterrows()):
    if not isinstance(major["ANC Name"], str):
        continue
    structure = [
        s
        for s in structures
        if s["acronym"] == major["ANC Name"].replace(" ", "-")
    ][0]
    savepath = str(
        meshes_dir_path
        / f'{structure["id"]}.obj'.replace(" ", "-")
        .replace("/", "-")
        .replace("\\", "-")
    )
    if os.path.isfile(savepath):
        continue

    regions = list(
        clean_hierarchy.loc[
            clean_hierarchy.major == major["ANC Name"]
        ].minor.values
    )
    if not regions:
        continue

    vol = np.zeros_like(volume_data)

    for region in regions:
        vol[volume_data == region] = 1

    if np.max(vol) < 1:
        continue

    write(extract_volume_surface(Volume(vol)), savepath)


# ---------------------------------------------------------------------------- #
#                            FINAL METADATA AND SAVE                           #
# ---------------------------------------------------------------------------- #

metadata_dict = {
    "name": ATLAS_NAME,
    "species": "Rattus Norvegicus",
    "citation": "Swanson 2018, https://pubmed.ncbi.nlm.nih.gov/29277900/",
    "atlas_link": "",
    "symmetric": False,
    "resolution": (1.25, 1.25, 1.25),
    "shape": loaded.shape,
}

with open(uncompr_atlas_path / "atlas_metadata.json", "w") as f:
    json.dump(metadata_dict, f)


# Create human readable files
create_metadata_files(uncompr_atlas_path, metadata_dict, structures)


# Compress folder:
output_filename = bg_root_dir / f"{uncompr_atlas_path.name}.tar.gz"
print(f"Saving compressed at {output_filename}")

with tarfile.open(output_filename, "w:gz") as tar:
    tar.add(uncompr_atlas_path, arcname=uncompr_atlas_path.name)
