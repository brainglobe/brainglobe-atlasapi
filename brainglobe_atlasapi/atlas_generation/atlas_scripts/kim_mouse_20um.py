import sys
from pathlib import Path
import tifffile
import numpy as np
import pandas as pd
import pooch
import nibabel as nib
from brainglobe_atlasapi.atlas_generation.mesh_utils import construct_meshes_from_annotation
from brainglobe_atlasapi.atlas_generation.wrapup import wrapup_atlas_from_data
from typing import List,Tuple, Dict, Any

# The minor version of the atlas in the brainglobe_atlasapi, this is internal,
# if this is the first time this atlas has been added the value should be 0
# (minor version is the first number after the decimal point, ie the minor
# version of 1.2 is 2)
__version__ = 0

# The expected format is FirstAuthor_SpeciesCommonName, e.g. kleven_rat, or
# Institution_SpeciesCommonName, e.g. allen_mouse.
ATLAS_NAME = "kim_mouse"

# DOI of the most relevant citable document
CITATION = 'Chon et al., 2019, Nature Communications (PMID: 31699990), doi: 10.1038/s41467-019-13057-w'

# The scientific name of the species, ie; Rattus norvegicus
SPECIES = 'Mus musculus'

# The URL for the data files
ATLAS_LINK = 'https://figshare.com/ndownloader/articles/25750983/versions/1'

# The orientation of the **original** atlas data, in BrainGlobe convention:
# https://brainglobe.info/documentation/setting-up/image-definition.html#orientation
ORIENTATION = "asr"

# The id of the highest level of the atlas. This is commonly called root or
# brain. Include some information on what to do if your atlas is not
# hierarchical
ROOT_ID = 997

# The resolution of your volume in microns. Details on how to format this
# parameter for non isotropic datasets or datasets with multiple resolutions.
RESOLUTION = 20

ATLAS_PACKAGER = "Pavel Vychyk / pavel.vychyk@brain.mpg.de"

# Custom globals for the retrieved file names
ONTOLOGY_FILE = 'UnifiedAtlas_Label_ontology_v2.csv'
ANNOTATION_FILE = 'UnifiedAtlas_Label_v2_20um-isotropic.nii'
REFERENCE_FILE = 'UnifiedAtlas_template_coronal_20um-isotropic.nii'


class AtlasBuilder:
    def __init__(self):
        self.bg_root_dir = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
        self.bg_root_dir.mkdir(parents=True, exist_ok=True)

    def download_resources(self) -> None:
        """
        Download the necessary resources for the atlas.

        If possible, please use the Pooch library to retrieve any resource
        """

        self.download_folder = self.bg_root_dir / "download"
        self.download_folder.mkdir(exist_ok=True)
        initial_download_folder_content = {
            item.name for item in self.download_folder.iterdir()}
        if ONTOLOGY_FILE not in initial_download_folder_content \
                and REFERENCE_FILE not in initial_download_folder_content \
                and ANNOTATION_FILE not in initial_download_folder_content:
            pooch.retrieve(
                url=ATLAS_LINK,
                path=(self.download_folder),
                progressbar=True,
                known_hash=None,
                processor=pooch.Unzip(extract_dir=self.download_folder)
            )
        self.download_dir_content = {
            item.name for item in self.download_folder.iterdir()}

    def retrieve_reference_and_annotation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieve the desired reference and annotation as two numpy arrays.

        Returns:
            tuple: A tuple containing two numpy arrays. The first array is the
            reference volume, and the second array is the annotation volume.
        """

        if ANNOTATION_FILE not in self.download_dir_content:
            raise Exception(
                f"{ANNOTATION_FILE} not found in {self.download_folder}")
        if REFERENCE_FILE not in self.download_dir_content:
            raise Exception(
                f"{REFERENCE_FILE} not found in {self.download_folder}")

        self.reference = AtlasBuilder.load_as_asr(
            self.download_folder / REFERENCE_FILE)
        self.annotation = AtlasBuilder.load_as_asr(
            self.download_folder / ANNOTATION_FILE)
        return self.reference, self.annotation

    def retrieve_hemisphere_map(self) -> None:
        """
        Retrieve a hemisphere map for the atlas.

        If your atlas is asymmetrical, you may want to use a hemisphere map.
        This is an array in the same shape as your template,
        with 0's marking the left hemisphere, and 1's marking the right.

        If your atlas is symmetrical, ignore this function.

        Returns:
            numpy.array or None: A numpy array representing the hemisphere map,
            or None if the atlas is symmetrical.
        """
        return None

    def retrieve_structure_information(self) -> List[Dict[str, Any]]:
        """
        This function should a return list with dictionaries containing information about the
        atlas.
        Example of the ROOT dictionary entry:
            {
                "id": ROOT_ID,
                "name": "root",
                "acronym": "root",
                "structure_id_path": [ROOT_ID],
                "rgb_triplet": [255, 255, 255]
            }
        Returns:
            list of dictionaries containing the atlas information.
        """
        if ONTOLOGY_FILE not in self.download_dir_content:
            raise Exception(
                f"{ONTOLOGY_FILE} not found in {self.download_folder}")
        df = pd.read_csv(self.download_folder / ONTOLOGY_FILE)
        int_cols = ["id", "RGB_1", "RGB_2", "RGB_3", "structure_id_path"]
        df[int_cols] = df[int_cols].apply(pd.to_numeric, errors="coerce")
        df = df.dropna(subset=["id"])
        df[int_cols] = df[int_cols].astype("int32")
        structures = [{"id": ROOT_ID,
                      "name": "root",
                       "acronym": "root",
                       "structure_id_path": [ROOT_ID],
                       "rgb_triplet": [255, 255, 255]
                       }]
        id_to_row = df.set_index("id").to_dict("index")
        for _, row in df.iterrows():
            structures.append({
                "id": row["id"],
                "name": row["name"],
                "acronym": row["acronym"],
                "structure_id_path": AtlasBuilder.get_path_to_root_id(id_to_row, df, row["id"]),
                "rgb_triplet": [row["RGB_1"], row["RGB_2"], row["RGB_3"]]
            })
        self.structures = structures
        return self.structures

    def retrieve_or_construct_meshes(self) -> Dict[int, str]:
        """
        This function should return a dictionary of ids and corresponding paths to
        mesh files. Some atlases are packaged with mesh files, in these cases we
        should use these files. Then this function should download those meshes.
        In other cases we need to construct the meshes ourselves. For this we have
        helper functions to achieve this.
        """
        meshes_dir = self.download_folder
        meshes_dict = construct_meshes_from_annotation(meshes_dir,
                                                       self.annotation,
                                                       self.structures,
                                                       ROOT_ID
                                                       )
        return meshes_dict

    @staticmethod
    def get_path_to_root_id(
            id_to_row: Dict,
            df: pd.DataFrame,
            current_id: int) -> List:
        path_to_root = []
        while True:
            current_row = id_to_row.get(current_id)
            if current_row is None:
                break
            path_to_root.insert(0, current_id)
            parent_id = current_row["structure_id_path"]
            if parent_id == current_id:
                break
            current_id = parent_id

        path_to_root.insert(0, ROOT_ID)
        return path_to_root

    @staticmethod
    def load_as_asr(path: Path) -> np.ndarray:
        """
        Prepare imaging data for asr convention
        """
        img = nib.load(path)
        data = np.asarray(img.dataobj).squeeze().astype(np.int32)
        reordered = np.flip(data, axis=2)
        reordered = np.moveaxis(reordered, 2, 0)
        reordered = np.rot90(reordered, k=-1, axes=(1, 2))
        return reordered


# If the code above this line has been filled correctly, nothing needs to be
# edited below (unless variables need to be passed between the functions).
if __name__ == "__main__":
    atlas_builder = AtlasBuilder()
    atlas_builder.download_resources()
    reference_volume, annotated_volume = atlas_builder.retrieve_reference_and_annotation()
    hemispheres_stack = atlas_builder.retrieve_hemisphere_map()
    structures = atlas_builder.retrieve_structure_information()
    meshes_dict = atlas_builder.retrieve_or_construct_meshes()

    output_filename = wrapup_atlas_from_data(
        atlas_name=ATLAS_NAME,
        atlas_minor_version=__version__,
        citation=CITATION,
        atlas_link=ATLAS_LINK,
        species=SPECIES,
        resolution=(RESOLUTION, ) * 3,
        orientation=ORIENTATION,
        root_id=ROOT_ID,
        reference_stack=reference_volume,
        annotation_stack=annotated_volume,
        structures_list=structures,
        meshes_dict=meshes_dict,
        working_dir=atlas_builder.bg_root_dir,
        hemispheres_stack=None,
        cleanup_files=False,
        compress=True,
        scale_meshes=True
    )
