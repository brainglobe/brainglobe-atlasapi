"""
    Automatic creation of 
        . structures.csv
        . README.txt
"""
import re
import json
from datetime import datetime
from brainatlas_api import descriptors

import requests
from requests.exceptions import MissingSchema, InvalidURL, ConnectionError

from atlas_gen.structure_json_to_csv import convert_structure_json_to_csv
from brainatlas_api.structure_tree_util import get_structures_tree


def generate_metadata_dict(
    name,
    citation,
    atlas_link,
    species,
    symmetric,
    resolution,
    version,
    shape,
    transformation_mat,
):

    # Name should be author_species
    assert len(name.split("_")) >= 2

    # Control version formatting:
    assert re.match("[0-9]+\\.[0-9]+", version)

    # We ask for DOI and correct link only if atlas is published:
    if citation != "unpublished":
        assert "doi" in citation

        # Test url:
        try:
            _ = requests.get(atlas_link)
        except (MissingSchema, InvalidURL, ConnectionError):
            raise InvalidURL(
                "Ensure that the url is valid and formatted correctly!"
            )

    # Enforce correct format for symmetric, resolution and shape:
    assert type(symmetric) == bool
    assert len(resolution) == 3
    assert len(shape) == 3

    resolution = tuple([float(v) for v in resolution])
    shape = tuple(int(v) for v in shape)

    return dict(
        name=name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,
        version=version,
        shape=shape,
        trasform_to_bg=tuple([tuple(m) for m in transformation_mat]),
    )


def create_readme(uncompr_atlas_path, metadata_dict, structures):
    readmepath = str(uncompr_atlas_path / "README.txt")

    # First write the structure tree
    structuresTree = get_structures_tree(structures)
    structuresTree.save2file(readmepath)

    # The prepend the header and info
    with open(readmepath, "r") as original:
        tree = original.read()

    with open(readmepath, "w") as out:
        out.write("-- BRAINGLOBE ATLAS --\n")

        now = datetime.now()
        out.write("Generated on: " + now.strftime("%d/%m/%Y") + "\n\n")

        out.write("------------------------------\n\n\n")

        for key, value in metadata_dict.items():
            out.write(f"    {key}:   {value}\n")

        out.write("\n\n\n")
        out.write("------------------------------\n\n\n")
        out.write("\n\n\n")

        out.write("-- BRAIN STRUCTURES TREE --\n")

        out.write(tree)


def create_structures_csv(uncompr_atlas_path, root):
    """
    Converts an atlas structure json dictionary to csv. For cellfinder
    compatibility and ease of browsing.

    Parameters
    ----------
    uncompr_atlas_path : str or Path object
        path to uncompressed atlas folder
    """
    convert_structure_json_to_csv(
        uncompr_atlas_path / "structures.json", root=root
    )


def create_metadata_files(dest_dir, metadata_dict, structures, root_id):
    """
        Automatic creation of 
            . structures.csv
            . README.txt
        from an atlas files. All Files are saved in the uncompressed atlas folder
        awaiting compression and upload to GIN.

        :param uncompr_atlas_path: path to uncompressed atlas folder
        :param metadata_dict: dict with atlas metadata
        :param structures: list of dictionaries with structures hierarchical info
    """
    # write metadata dict:
    with open(dest_dir / descriptors.METADATA_FILENAME, "w") as f:
        json.dump(metadata_dict, f)

    create_structures_csv(dest_dir, root_id)
    create_readme(dest_dir, metadata_dict, structures)
