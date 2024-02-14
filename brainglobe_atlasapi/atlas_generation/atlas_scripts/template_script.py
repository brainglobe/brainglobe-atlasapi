"""Template script for the generation of an atlas. Note that the script
has to be renamed to match the name of the atlas (e.g. allen_mouse.py)
"""

__version__ = "0"  # will be used to set minor version of the atlas

from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data


def create_atlas(working_dir, resolution):
    """Function to generate source data for an atlas.

    Parameters
    ----------
    working_dir : Path object
        Path where atlas will be created.
    resolution :
        Resolution of the atlas, in um.

    Returns
    -------
    Path object
        Path to the final compressed atlas file.

    """

    ATLAS_NAME = ""
    SPECIES = ""
    ATLAS_LINK = ""
    CITATION = ""
    ORIENTATION = ""

    # do stuff to create the atlas
    template_volume = None  # volume with reference
    annotated_volume = None  # volume with structures annotations
    structures_list = None  # list of valid structure dictionaries
    meshes_dict = None  # dictionary of files with region meshes
    root_id = None  # id of the root structure

    # Put here additional reference stacks
    # (different genotypes, filtered volumes, etc)
    additional_references = dict()

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(resolution,) * 3,  # if isotropic - highly recommended
        orientation=ORIENTATION,
        root_id=root_id,
        reference_stack=template_volume,
        annotation_stack=annotated_volume,
        structures_list=structures_list,
        meshes_dict=meshes_dict,
        working_dir=working_dir,
        additional_references=additional_references,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
    )

    return output_filename


# To test stuff locally:
if __name__ == "__main__":
    resolution = 100  # some resolution, in microns

    # Generated atlas path:
    bg_root_dir = "/path/to/some/dir"
    bg_root_dir.mkdir(exist_ok=True)

    create_atlas(bg_root_dir, resolution)
