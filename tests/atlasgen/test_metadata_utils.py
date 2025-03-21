import brainglobe_space as bgs
import numpy as np
import pooch
from brainglobe_utils.IO.image import load_nii

import brainglobe_atlasapi.atlas_generation
from brainglobe_atlasapi import descriptors, utils
from brainglobe_atlasapi.atlas_generation.metadata_utils import (
    generate_metadata_dict,
)


def test_generate_metadata_dict(tmp_path):
    __version__ = "1"

    atlas_name = "unam_axolotl"
    species = "Ambystoma mexicanum"
    atlas_link = "https://zenodo.org/records/4595016"
    citation = (
        "Lazcano, I. et al. 2021, https://doi.org/10.1038/s41598-021-89357-3"
    )
    ORIENTATION = "lpi"
    atlas_packager = "Saima Abdus, David Perez-Suarez, Alessandro Felder"
    additional_references = {}
    resolution = 40, 40, 40  # Resolution tuple
    # If no hemisphere file is given, assume the atlas is symmetric:
    symmetric = False
    atlas_path = tmp_path / "atlas_files"
    atlas_path.mkdir(exist_ok=True)
    # download atlas files
    utils.check_internet_connection()
    nii_file = "axolotl_template_40micra.nii.gz"
    hash = "md5:66df0da5d7eed10ff59add32741d0bf2"
    list_files = {nii_file: hash}

    for filename, hash in list_files.items():
        pooch.retrieve(
            url=f"{atlas_link}/files/{filename}",
            known_hash=hash,
            path=atlas_path,
            progressbar=True,
            processor=pooch.Decompress(name=filename[:-3]),
        )

    reference_file = atlas_path / "axolotl_template_40micra.nii"

    reference_image = load_nii(reference_file, as_array=True)
    dmin = np.min(reference_image)
    dmax = np.max(reference_image)
    drange = dmax - dmin
    dscale = (2**16 - 1) / drange
    reference_image = (reference_image - dmin) * dscale
    reference_image = reference_image.astype(np.uint16)
    # Instantiate BGSpace obj, using original stack size in um as meshes
    # are un um:
    shape = reference_image.shape
    volume_shape = tuple(res * s for res, s in zip(resolution, shape))
    space_convention = bgs.AnatomicalSpace(ORIENTATION, shape=volume_shape)

    transformation_mat = space_convention.transformation_matrix_to(
        descriptors.ATLAS_ORIENTATION
    )
    ATLAS_VERSION = brainglobe_atlasapi.atlas_generation.__version__
    atlas_minor_version = __version__

    generate_metadata_dict(
        name=atlas_name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,
        orientation=ORIENTATION,
        version=f"{ATLAS_VERSION}.{atlas_minor_version}",
        shape=shape,
        transformation_mat=transformation_mat,
        additional_references=[k for k in additional_references.keys()],
        atlas_packager=atlas_packager,
    )

    pass
