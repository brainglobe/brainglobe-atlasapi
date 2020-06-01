from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from requests import exceptions
from pathlib import Path
import tempfile
import json

import tifffile
import pandas as pd

RES_UM = 25
ATLAS_NAME = f"allenbrain{RES_UM}um"

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

# Download annotated and template volume:
#########################################
spacecache = ReferenceSpaceCache(
    manifest=downloading_path / "manifest.json",
    # downloaded files are stored relative to here
    resolution=RES_UM,
    reference_space_key="annotation/ccf_2017"
    # use the latest version of the CCF
)

# Download
annotated_volume, _ = spacecache.get_annotation_volume()
template_volume, _ = spacecache.get_template_volume()
print("Download completed...")
# Save tiff stacks:
tifffile.imsave(str(uncompr_atlas_path / "reference.tiff"), template_volume)
tifffile.imsave(str(uncompr_atlas_path / "annotated.tiff"), annotated_volume)

# Download structures tree and meshes:
######################################
oapi = OntologiesApi()  # ontologies
struct_tree = spacecache.get_structure_tree()  # structures tree

# Find id of set of regions with mesh:
select_set = "Structures whose surfaces are represented by a precomputed mesh"

all_sets = pd.DataFrame(oapi.get_structure_sets())
mesh_set_id = all_sets[all_sets.description == select_set].id.values[0]

structs_with_mesh = struct_tree.get_structures_by_set_id([mesh_set_id])

meshes_dir = uncompr_atlas_path / "meshes"  # directory to save meshes into
space = ReferenceSpaceApi()
for s in structs_with_mesh:
    name = s["id"]
    try:
        space.download_structure_mesh(
            structure_id=s["id"],
            ccf_version="annotation/ccf_2017",
            file_name=meshes_dir / f"{name}.obj",
        )
    except (exceptions.HTTPError, ConnectionError):
        print(s)

# Loop over structures, remove entries not used in brainglobe:
for struct in structs_with_mesh:
    [struct.pop(k) for k in ["graph_id", "structure_set_ids", "graph_order"]]

with open(uncompr_atlas_path / "structures.json", "w") as f:
    json.dump(structs_with_mesh, f)

metadata_dict = {
    "name": ATLAS_NAME,
    "citation": "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007",
    "atlas_link": "www.brain-map.org.com",
    "species": "Mus musculus",
    "symmetric": True,
    "resolution": (RES_UM, RES_UM, RES_UM),
    "shape": template_volume.shape,
}

with open(uncompr_atlas_path / "atlas_metadata.json", "w") as f:
    json.dump(metadata_dict, f)
