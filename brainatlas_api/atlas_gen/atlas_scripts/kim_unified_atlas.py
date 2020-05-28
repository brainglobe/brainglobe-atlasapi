from allensdk.core.reference_space_cache import ReferenceSpaceCache

from pathlib import Path
import tempfile
import json
import tarfile

import tifffile

from brainio.brainio import load_any

annotations_image = Path(
    "/media/adam/Storage/cellfinder/data/paxinos_allen/annotations_coronal.tif"
)
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
#
# # Download structures tree and meshes:
# ######################################
# oapi = OntologiesApi()  # ontologies
# struct_tree = spacecache.get_structure_tree()  # structures tree
#
# # Find id of set of regions with mesh:
# select_set = "Structures whose surfaces are represented by a precomputed mesh"
#
# all_sets = pd.DataFrame(oapi.get_structure_sets())
# mesh_set_id = all_sets[all_sets.description == select_set].id.values[0]
#
# structs_with_mesh = struct_tree.get_structures_by_set_id([mesh_set_id])
#
# meshes_dir = uncompr_atlas_path / "meshes"  # directory to save meshes into
# space = ReferenceSpaceApi()
# for s in structs_with_mesh:
#     name = s["id"]
#     try:
#         space.download_structure_mesh(
#             structure_id=s["id"],
#             ccf_version="annotation/ccf_2017",
#             file_name=meshes_dir / f"{name}.obj",
#         )
#     except (exceptions.HTTPError, ConnectionError):
#         print(s)
#
# # Loop over structures, remove entries not used in brainglobe:
# for struct in structs_with_mesh:
#     [struct.pop(k) for k in ["graph_id", "structure_set_ids", "graph_order"]]
#
# with open(uncompr_atlas_path / "structures.json", "w") as f:
#     json.dump(structs_with_mesh, f)

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
