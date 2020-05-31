from allensdk.api.queries.ontologies_api import OntologiesApi
from allensdk.api.queries.reference_space_api import ReferenceSpaceApi
from allensdk.core.reference_space_cache import ReferenceSpaceCache

from requests import exceptions
from pathlib import Path
import json
import shutil

from atlas_gen.stacks import save_anatomy, save_annotation
from atlas_gen.wrapup import wrapup_atlas_from_dir
from brainatlas_api import descriptors

# Specify information about the atlas:
RES_UM = 100
VERSION = "0.1"
ATLAS_NAME = f"test_allen_{RES_UM}um_v{VERSION}"
SPECIES = "mouse (Mus musculus)"
ATLAS_LINK = "http://www.brain-map.org.com"
CITATION = "Wang et al 2020, https://doi.org/10.1016/j.cell.2020.04.007"

# Working path on disk:
bg_root_dir = Path.home() / "brainglobe_workdir"
bg_root_dir.mkdir(exist_ok=True)

# Temporary folder for nrrd files download:
download_dir_path = bg_root_dir / "downloading_path"
download_dir_path.mkdir(exist_ok=True)

# Temporary folder for files before compressing:
uncompr_atlas_path = bg_root_dir / ATLAS_NAME
uncompr_atlas_path.mkdir(exist_ok=True)

# Download annotated and template volume:
#########################################
spacecache = ReferenceSpaceCache(
    manifest=download_dir_path / "manifest.json",
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
save_anatomy(template_volume, uncompr_atlas_path)
save_annotation(annotated_volume, uncompr_atlas_path)

# Download structures tree and meshes:
######################################
oapi = OntologiesApi()  # ontologies
struct_tree = spacecache.get_structure_tree()  # structures tree

# Find id of set of regions with mesh:
select_set = "Structures whose surfaces are represented by a precomputed mesh"

mesh_set_ids = [
    s["id"]
    for s in oapi.get_structure_sets()
    if s["description"] == select_set
]

structs_with_mesh = struct_tree.get_structures_by_set_id(mesh_set_ids)[:3]

# Directory for mesh saving:
meshes_dir = uncompr_atlas_path / descriptors.MESHES_DIRNAME

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

# Loop over structures, remove entries not used:
for struct in structs_with_mesh:
    [struct.pop(k) for k in ["graph_id", "structure_set_ids", "graph_order"]]

with open(uncompr_atlas_path / descriptors.STRUCTURES_FILENAME, "w") as f:
    json.dump(structs_with_mesh, f)

# Wrap up, compress, and remove file:
print(f"Saving compressed files at {uncompr_atlas_path.parents[0]}")
wrapup_atlas_from_dir(
    uncompr_atlas_path,
    CITATION,
    ATLAS_LINK,
    SPECIES,
    (RES_UM,) * 3,
    cleanup_files=True,
    compress=True,
)

shutil.rmtree(download_dir_path)
