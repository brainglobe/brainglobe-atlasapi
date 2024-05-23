"""
    Automatic creation of
        . structures.csv
        . README.txt
"""

import json
import re
from datetime import datetime

from requests.exceptions import ConnectionError, InvalidURL, MissingSchema

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.structure_json_to_csv import (
    convert_structure_json_to_csv,
)
from brainglobe_atlasapi.structure_tree_util import get_structures_tree


def generate_metadata_dict(
    name,
    citation,
    atlas_link,
    species,
    symmetric,
    resolution,
    orientation,
    version,
    shape,
    transformation_mat,
    additional_references,
    atlas_packager,
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
            pass
            # _ = requests.get(atlas_link)
        except (MissingSchema, InvalidURL, ConnectionError):
            raise InvalidURL(
                "Ensure that the url is valid and formatted correctly!"
            )

    # Enforce correct format for symmetric, resolution and shape:
    assert isinstance(symmetric, bool)
    assert len(resolution) == 3
    assert len(shape) == 3

    resolution = tuple([float(v) for v in resolution])
    shape = tuple(int(v) for v in shape)

    assert isinstance(additional_references, list)

    return dict(
        name=name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,
        orientation=orientation,
        version=version,
        shape=shape,
        trasform_to_bg=tuple([tuple(m) for m in transformation_mat]),
        additional_references=additional_references,
        atlas_packager=atlas_packager,
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


def create_metadata_files(
    dest_dir, metadata_dict, structures, root_id, additional_metadata={}
):
    """
    Automatic creation of
        . structures.csv
        . README.txt
    from an atlas files. All Files are saved in the uncompressed atlas folder
    awaiting compression and upload to GIN.

    :param uncompr_atlas_path: path to uncompressed atlas folder
    :param metadata_dict: dict with atlas metadata
    :param structures: list of dictionaries with structures hierarchical info
    :param additional_metadata: Dict to add to atlas metadata
    """
    # write metadata dict:
    with open(dest_dir / descriptors.METADATA_FILENAME, "w") as f:
        # only save additional metadata to json, don't include in readme
        json.dump({**metadata_dict, **additional_metadata}, f)

    create_structures_csv(dest_dir, root_id)
    create_readme(dest_dir, metadata_dict, structures)
