"""
    Automatic creation of 
        . structures.csv
        . structures.txt
        . structures_tree.txt
        . README.csv
"""
from datetime import datetime

from brainatlas_api.structures.structure_tree import StructureTree



def create_structures_txt_files(uncompr_atlas_path, structures):
    structuresTree = StructureTree(structures)

    structures_filepath = str(uncompr_atlas_path / 'structures.txt')
    structuresTree.print_structures(to_file=True, save_filepath=structures_filepath)

    structures_tree_filepath = str(uncompr_atlas_path / 'structures_tree.txt')
    structuresTree.print_structures_tree(to_file=True, save_filepath=structures_tree_filepath)



def create_readme(uncompr_atlas_path, metadata_dict):
    readmepath = str(uncompr_atlas_path / 'README.txt')

    with open(readmepath, 'w') as out:
        out.write('-- BRAINGLOBE ATLAS --\n')


        now = datetime.now()
        out.write('Generated on: ' + now.strftime("%d/%m/%Y") + "\n\n")

        out.write('------------------------------\n\n\n')

        for key, value in metadata_dict.items():
            out.write(f"    {key}:   {value}\n")





def create_metadata_files(uncompr_atlas_path, metadata_dict, structures):
    """
        Automatic creation of 
            . structures.csv
            . structures.txt
            . structures_tree.txt
            . README.csv
        from an atlas files. All Files are saved in the uncompressed atlas folder
        awaiting compression and upload to GIN.

        :param uncompr_atlas_path: path to uncompressed atlas folder
        :param metadata_dict: dict with atlas metadata
        :param structures: list of dictionaries with structures hierarchical info
    """

    create_structures_txt_files(uncompr_atlas_path, structures)
    create_readme(uncompr_atlas_path, metadata_dict)
    