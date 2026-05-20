"""Atlas generation script for the VFB's JRC2018Unisex Neuropils Fly atlas."""

import gzip
import json
from pathlib import Path

import numpy as np
import pooch

### Metadata
__version__ = 0

ATLAS_NAME = "vfb_jrc2018u_neuropils_fly"

CITATION = (
    "Bogovic et al. 2020, " "https://doi.org/10.1371/journal.pone.0236495"
)

SPECIES = "Drosophila melanogaster"

ATLAS_LINK = (
    "https://www.virtualflybrain.org/blog/2022/01/01/"
    "jrc-2018-templates-rois-jrc2018/"
)

ATLAS_FILE_URL = (
    "https://www.virtualflybrain.org/data/VFB/i/0010/1567/"
    "VFB_00101567/volume.nrrd"
)

VFB_TEMPLATE_ID = "VFB_00101567"

VFB_SOLR_TERM_INFO_URL = (
    "http://solr.virtualflybrain.org/solr/vfb_json/select"
    f"?q=id:{VFB_TEMPLATE_ID}&fl=id,term_info&rows=1&wt=json"
)

ORIENTATION = "lps"

ROOT_ID = 999
RESOLUTION = (0.5189161, 0.5189161, 1.0)  # microns

ATLAS_PACKAGER = "Amirreza Bahramani"


def _secure_url(url):
    """Use HTTPS for VFB data files when available."""
    return url.replace(
        "http://www.virtualflybrain.org",
        "https://www.virtualflybrain.org",
    )


def _retrieve(url, download_dir, file_name):
    return Path(
        pooch.retrieve(
            url=url,
            known_hash=None,
            path=download_dir,
            fname=file_name,
            progressbar=True,
        )
    )


def _load_vfb_term_info(term_info_response_path):
    with open(term_info_response_path, encoding="utf-8") as f:
        response = json.load(f)

    docs = response["response"]["docs"]
    if not docs:
        raise RuntimeError(
            f"No VFB term-info record found for {VFB_TEMPLATE_ID}"
        )

    term_info = docs[0]["term_info"]
    if isinstance(term_info, list):
        term_info = term_info[0]

    return json.loads(term_info)


def _load_nrrd_array(nrrd_path):
    """Load a gzip-encoded NRRD file into a numpy array."""
    nrrd_path = Path(nrrd_path)
    file_bytes = nrrd_path.read_bytes()

    header_end = file_bytes.find(b"\n\n")
    separator_length = 2
    if header_end == -1:
        header_end = file_bytes.find(b"\r\n\r\n")
        separator_length = 4

    if header_end == -1:
        raise ValueError(f"Could not find NRRD header end in {nrrd_path}")

    header = file_bytes[:header_end].decode("ascii")
    data = file_bytes[header_end + separator_length :]

    header_fields = {}
    for line in header.splitlines():
        if not line or line.startswith("#") or ":" not in line:
            continue

        key, value = line.split(":", maxsplit=1)
        header_fields[key.strip()] = value.strip()

    sizes = tuple(int(size) for size in header_fields["sizes"].split())
    data_type = header_fields["type"]
    encoding = header_fields.get("encoding", "raw").lower()

    if encoding in {"gzip", "gz"}:
        data = gzip.decompress(data)
    elif encoding != "raw":
        raise ValueError(f"Unsupported NRRD encoding: {encoding}")

    dtype = {
        "uint8": np.uint8,
        "uchar": np.uint8,
        "unsigned char": np.uint8,
        "uint16": np.uint16,
        "ushort": np.uint16,
        "unsigned short": np.uint16,
    }[data_type]

    return np.frombuffer(data, dtype=dtype).reshape(sizes, order="F")


def download_resources():
    """
    Download the necessary resources for the atlas.

    Returns
    -------
    dict[str, pathlib.Path]
        Paths to the downloaded VFB template, ROI masks, and mesh files.
    """
    download_dir = (
        Path.home() / "brainglobe_workingdir" / ATLAS_NAME / "source_data"
    )
    roi_volumes_dir = download_dir / "roi_volumes"
    meshes_dir = download_dir / "meshes"

    download_dir.mkdir(parents=True, exist_ok=True)
    roi_volumes_dir.mkdir(exist_ok=True)
    meshes_dir.mkdir(exist_ok=True)

    term_info_response_path = _retrieve(
        VFB_SOLR_TERM_INFO_URL,
        download_dir,
        "jrc2018u_term_info_response.json",
    )
    term_info = _load_vfb_term_info(term_info_response_path)

    term_info_path = download_dir / "jrc2018u_term_info.json"
    with open(term_info_path, "w", encoding="utf-8") as f:
        json.dump(term_info, f, indent=2)

    template_channel = term_info["template_channel"]
    reference_url = _secure_url(
        template_channel.get("image_nrrd") or ATLAS_FILE_URL
    )
    reference_path = _retrieve(
        reference_url,
        download_dir,
        "jrc2018u_template.nrrd",
    )

    root_mesh_path = _retrieve(
        _secure_url(template_channel["image_obj"]),
        meshes_dir,
        f"{ROOT_ID}.obj",
    )

    roi_volumes = {}
    meshes = {ROOT_ID: root_mesh_path}
    domain_metadata = []

    domains = sorted(
        term_info["template_domains"],
        key=lambda domain: int(domain["index"][0]),
    )
    for domain in domains:
        vfb_index = int(domain["index"][0])
        if vfb_index == 0:
            continue

        structure_id = vfb_index
        vfb_id = domain["anatomical_individual"]["short_form"]

        roi_volumes[structure_id] = _retrieve(
            _secure_url(domain["image_nrrd"]),
            roi_volumes_dir,
            f"{structure_id}.nrrd",
        )
        meshes[structure_id] = _retrieve(
            _secure_url(domain["image_obj"]),
            meshes_dir,
            f"{structure_id}.obj",
        )

        domain_metadata.append(
            {
                "id": structure_id,
                "vfb_id": vfb_id,
                "label": domain["anatomical_individual"]["label"],
                "type_id": domain["anatomical_type"]["short_form"],
                "type_label": domain["anatomical_type"]["label"],
                "nrrd": str(roi_volumes[structure_id]),
                "obj": str(meshes[structure_id]),
            }
        )

    domain_metadata_path = download_dir / "jrc2018u_domain_metadata.json"
    with open(domain_metadata_path, "w", encoding="utf-8") as f:
        json.dump(domain_metadata, f, indent=2)

    return {
        "reference": reference_path,
        "term_info": term_info_path,
        "domain_metadata": domain_metadata_path,
        "roi_volumes": roi_volumes,
        "meshes": meshes,
    }


def retrieve_reference_and_annotation():
    """
    Retrieve the reference and annotation volumes.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        A tuple containing the reference volume and the annotation volume.
    """
    download_dir = (
        Path.home() / "brainglobe_workingdir" / ATLAS_NAME / "source_data"
    )
    reference_path = download_dir / "jrc2018u_template.nrrd"
    roi_volumes_dir = download_dir / "roi_volumes"

    reference = _load_nrrd_array(reference_path)
    annotation = np.zeros(reference.shape, dtype=np.uint16)

    roi_paths = sorted(
        roi_volumes_dir.glob("*.nrrd"),
        key=lambda path: int(path.stem),
    )
    for roi_path in roi_paths:
        structure_id = int(roi_path.stem)
        roi_mask = _load_nrrd_array(roi_path)
        if roi_mask.shape != reference.shape:
            raise ValueError(
                f"ROI {structure_id} has shape {roi_mask.shape}, "
                f"but reference has shape {reference.shape}"
            )

        mask_voxels = roi_mask > 0
        if np.any(annotation[mask_voxels] != 0):
            raise ValueError(f"ROI {structure_id} overlaps another ROI")

        annotation[mask_voxels] = structure_id

    return reference, annotation


def retrieve_hemisphere_map():
    """
    Retrieve a hemisphere map for the atlas.

    Use a hemisphere map if the atlas is asymmetrical. This map is an array
    with the same shape as the template, where 0 marks the left hemisphere
    and 1 marks the right.

    Returns
    -------
    np.ndarray or None
        A numpy array representing the hemisphere map, or None if the atlas
        is symmetrical.
    """
    return None


def retrieve_structure_information():
    """
    Return a list of dictionaries with information about the atlas.

    Returns a list of dictionaries, where each dictionary represents a
    structure and contains its ID, name, acronym, hierarchical path,
    and RGB triplet.

    The expected format for each dictionary is:

    .. code-block:: python

        {
            "id": int,
            "name": str,
            "acronym": str,
            "structure_id_path": list[int],
            "rgb_triplet": list[int, int, int],
        }

    Returns
    -------
    list[dict]
        A list of dictionaries, each containing information for a single
        atlas structure.
    """
    return None


def retrieve_or_construct_meshes():
    """
    Return a dictionary mapping structure IDs to paths of mesh files.

    If the atlas is packaged with mesh files, download and use them. Otherwise,
    construct the meshes using available helper functions.

    Returns
    -------
    dict
        A dictionary where keys are structure IDs and values are paths to the
        corresponding mesh files.
    """
    meshes_dict = {}
    return meshes_dict


def retrieve_additional_references():
    """
    Return a dictionary of additional reference images.

    This function should be edited only if the atlas includes additional
    reference images. The dictionary should map the name of each additional
    reference image to its corresponding image stack data.

    Returns
    -------
    dict
        A dictionary mapping reference image names to their image stack data.
    """
    additional_references = {}
    return additional_references


if __name__ == "__main__":
    download_resources()
    retrieve_reference_and_annotation()


# ### If the code above this line has been filled correctly, nothing needs to be
# ### edited below (unless variables need to be passed between the functions).
# if __name__ == "__main__":
#     if RESOLUTION is None:
#         raise ValueError("RESOLUTION must be set before running this script.")

#     bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
#     bg_root_dir.mkdir(parents=True, exist_ok=True)

#     # Fail early if any version of this atlas already exists
#     atlas_prefix = atlas_name_from_repr(ATLAS_NAME, RESOLUTION)
#     existing = list(bg_root_dir.glob(f"{atlas_prefix}_v*"))

#     if existing:
#         raise FileExistsError(
#             f"Atlas output already exists in {bg_root_dir}. "
#         )
#     download_resources()
#     reference_volume, annotated_volume = retrieve_reference_and_annotation()
#     additional_references = retrieve_additional_references()
#     hemispheres_stack = retrieve_hemisphere_map()
#     structures = retrieve_structure_information()
#     meshes_dict = retrieve_or_construct_meshes()

#     output_filename = wrapup_atlas_from_data(
#         atlas_name=ATLAS_NAME,
#         atlas_minor_version=__version__,
#         citation=CITATION,
#         atlas_link=ATLAS_LINK,
#         species=SPECIES,
#         resolution=(RESOLUTION,) * 3,
#         orientation=ORIENTATION,
#         root_id=ROOT_ID,
#         reference_stack=reference_volume,
#         annotation_stack=annotated_volume,
#         structures_list=structures,
#         meshes_dict=meshes_dict,
#         working_dir=bg_root_dir,
#         hemispheres_stack=None,
#         cleanup_files=False,
#         compress=True,
#         scale_meshes=True,
#         additional_references=additional_references,
#     )
