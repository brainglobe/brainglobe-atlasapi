import numpy as np

# Entries and types from this template will be used to check atlas info
# consistency. Please keep updated both this and the function when changing the structure.

# If the atlas is unpublished, specify "unpublished" in the citation.

METADATA_TEMPLATE = {
    "name": "name/author/institute_species_[optionalspecs]",
    "citation": "Someone et al 2020, https://doi.org/somedoi",
    "atlas_link": "http://www.example.com",
    "species": "Gen species",
    "symmetric": False,
    "resolution": (1.0, 1.0, 1.0),
    "shape": (100, 50, 100),
    "version": "0.0",
    "supplementary_stacks": [],
}

STRUCTURE_TEMPLATE = {
    "acronym": "root",
    "id": 997,
    "name": "root",
    "structure_id_path": [997],
    "rgb_triplet": [255, 255, 255],
}


METADATA_FILENAME = "metadata.json"
STRUCTURES_FILENAME = "structures.json"
REFERENCE_FILENAME = "reference.tiff"
ANNOTATION_FILENAME = "annotation.tiff"
HEMISPHERES_FILENAME = "hemispheres.tiff"
MESHES_DIRNAME = "meshes"

REFERENCE_DTYPE = np.uint16
ANNOTATION_DTYPE = np.int32
