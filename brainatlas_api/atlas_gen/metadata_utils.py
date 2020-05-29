"""
    Automatic creation of 
        . structures.csv
        . README.txt
"""
from datetime import datetime

from brainatlas_api.structures.structure_tree import StructureTree


def create_readme(uncompr_atlas_path, metadata_dict, structures):
    readmepath = str(uncompr_atlas_path / "README.txt")

    # First write the structure tree
    structuresTree = StructureTree(structures)
    structuresTree.print_structures_tree(
        to_file=True, save_filepath=readmepath
    )

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


def create_metadata_files(uncompr_atlas_path, metadata_dict, structures):
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

    create_readme(uncompr_atlas_path, metadata_dict, structures)
