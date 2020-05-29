from pathlib import Path
import json
import pandas as pd

from brainio import brainio
import numpy as np
import urllib3
import multiprocessing as mp
import time

from brainatlas_api.atlas_gen import (
    save_anatomy,
    save_annotation,
    descriptors,
    wrapup_atlas_from_dir,
    volume_utils,
    mesh_utils,
)

from brainatlas_api.structures.structure_tree import StructureTree


def create_structure_mesh(args):
    structures, annotation_whole, meshes_dir, a = args

    lbl = structures.loc[a, "acronym"]
    volume = volume_utils.create_masked_array(
        annotation_whole, a, greater_than=False
    )

    if np.max(volume) < 1:
        print(f"No voxel data for region {lbl}")
        return None
    else:
        print(f"Creating mesh for {a}.obj")
        savepath = meshes_dir / f"{a}.obj"
        if not savepath.exists():
            # mesh_utils.extract_mesh_from_mask(volume, savepath, smooth=False, decimate=False, smooth_mesh=False, closing_n_iters=1)
            mesh = mesh_utils.extract_mesh_from_mask_fast(
                volume, obj_filepath=savepath
            )
        return mesh


if __name__ == "__main__":
    # Specify information about the atlas:
    RES_UM = 500
    VERSION = "0.1"
    ATLAS_NAME = f"allen_human_{RES_UM}um_v{VERSION}"
    SPECIES = "human (Homo sapiens)"
    ATLAS_LINK = "http://download.alleninstitute.org/informatics-archive/allen_human_reference_atlas_3d_2020/version_1/"
    CITATION = "Ding et al 2020, https://doi.org/10.1002/cne.24080"

    data_fld = Path(
        r"D:\Dropbox (UCL - SWC)\Rotation_vte\Anatomy\Atlases\atlasesforbrainrender\AllenHuman"
    )

    # Generated atlas path:
    bg_root_dir = Path.home() / ".brainglobe"
    bg_root_dir.mkdir(exist_ok=True)

    # Temporary folder for nrrd files download:
    temp_path = bg_root_dir / "temp"
    temp_path.mkdir(exist_ok=True)

    # Temporary folder for files before compressing:
    uncompr_atlas_path = temp_path / ATLAS_NAME
    uncompr_atlas_path.mkdir(exist_ok=True)

    # Open reference:
    #################
    # TODO check if re-orienting is necessary

    annotation = brainio.load_any(
        data_fld / "annotation.nii"
    )  # shape (394, 466, 378)

    anatomy = brainio.load_any(
        data_fld
        / "mni_icbm152_nlin_sym_09b"
        / "mni_icbm152_pd_tal_nlin_sym_09b_hires.nii"
    )  # shape (394, 466, 378)

    # Remove weird artefact
    annotation = annotation[:197, :, :]
    anatomy = anatomy[:197, :, :]

    # These data only have one hemisphere, so mirror them
    annotation_whole = np.zeros(
        (annotation.shape[0] * 2, annotation.shape[1], annotation.shape[2]),
        annotation.dtype,
    )
    annotation_whole[: annotation.shape[0], :, :] = annotation
    annotation_whole[annotation.shape[0] :, :, :] = annotation[::-1, :, :]

    anatomy_whole = np.zeros(
        (anatomy.shape[0] * 2, anatomy.shape[1], anatomy.shape[2]),
        anatomy.dtype,
    )
    anatomy_whole[: anatomy.shape[0], :, :] = anatomy
    anatomy_whole[anatomy.shape[0] :, :, :] = anatomy[::-1, :, :]

    # Save as .tiff
    save_annotation(annotation_whole, uncompr_atlas_path)
    save_anatomy(anatomy_whole, uncompr_atlas_path)
    del anatomy_whole, annotation, anatomy

    # Download structure tree
    #########################

    # RMA query to fetch structures for the structure graph
    query_url = "http://api.brain-map.org/api/v2/data/query.json?criteria=model::Structure"
    query_url += ",rma::criteria,[graph_id$eq%d]" % 16
    query_url += (
        ",rma::options[order$eq'structures.graph_order'][num_rows$eqall]"
    )

    http = urllib3.PoolManager()
    r = http.request("GET", query_url)
    data = json.loads(r.data.decode("utf-8"))["msg"]
    structures = pd.read_json(json.dumps(data))

    # Create empty list and collect all regions traversing the regions hierarchy:
    regions_list = []

    for i, region in structures.iterrows():
        if i == 0:
            acronym = "root"
        else:
            acronym = region["acronym"]

        regions_list.append(
            {
                "name": region["name"],
                "acronym": acronym,
                "id": region["id"],
                "rgb_triplet": StructureTree.hex_to_rgb(
                    region["color_hex_triplet"]
                ),
                "structure_id_path": StructureTree.path_to_list(
                    region["structure_id_path"]
                ),
            }
        )

    # save regions list json:
    with open(uncompr_atlas_path / descriptors.STRUCTURES_FILENAME, "w") as f:
        json.dump(regions_list, f)

    # Create meshes
    ###############
    meshes_dir = uncompr_atlas_path / descriptors.MESHES_DIRNAME
    meshes_dir.mkdir(exist_ok=True)

    unique_values, unique_counts = np.unique(
        annotation_whole, return_counts=True
    )
    voxel_counts = dict(zip(unique_values, unique_counts))
    if 0 in voxel_counts:
        del voxel_counts[0]
    structures.set_index("id", inplace=True)

    # Create root first
    root = [s for s in regions_list if s["acronym"] == "root"][0]
    root_volume = volume_utils.create_masked_array(
        annotation_whole, 0, greater_than=True
    )
    savepath = meshes_dir / f'{root["id"]}.obj'
    if not savepath.exists():
        mesh_utils.extract_mesh_from_mask(
            root_volume,
            savepath,
            smooth=False,
            decimate=True,
            smooth_mesh=True,
        )

    start = time.time()
    pool = mp.Pool(mp.cpu_count() - 2)
    try:
        pool.map(
            create_structure_mesh,
            [
                (structures, annotation_whole, meshes_dir, a)
                for a in voxel_counts
            ],
        )
    except mp.pool.MaybeEncodingError:
        pass  # error with returning results from pool.map but we don't care
    print(
        f"Creating meshes for {len(voxel_counts)} structures took: {round(time.time() - start, 3)}s"
    )

    # TODO extract meshes for non leaf regions
    # TODO try: scipy.ndimage.morphology.binary_fill_holes to fix root

    # Wrap up, compress, and remove file:
    #####################################
    wrapup_atlas_from_dir(
        uncompr_atlas_path,
        CITATION,
        ATLAS_LINK,
        SPECIES,
        (RES_UM,) * 3,
        cleanup_files=False,
        compress=True,
    )
