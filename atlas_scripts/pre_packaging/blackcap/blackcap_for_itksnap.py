"""One-off script to convert final blackcap atlas to ITK-snap version."""

from pathlib import Path

from brainglobe_utils.IO.image.save import save_as_asr_nii

from brainglobe_atlasapi import BrainGlobeAtlas
from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    write_itk_labels,
)


def brainglobe_atlas_to_itksnap(atlas: BrainGlobeAtlas, path: Path):
    """
    Convert a BrainGlobe atlas to ITKsnap format.

    Exports the atlas reference image, annotation image, and anatomical labels
    to files compatible with ITKsnap, a neuroimaging visualization tool.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        The BrainGlobe atlas object containing reference, annotation, and
        structures information.
    path : Path
        Directory path where the ITKsnap files will be saved. Creates:
        - template.nii.gz: Reference image
        - annotations.nii.gz: Annotation image
        - labels.txt: Anatomical structure labels

    Returns
    -------
    None

    Notes
    -----
    Voxel sizes are converted from atlas native micron units to millimeters.
    """
    save_as_asr_nii(
        atlas.reference,
        vox_sizes=[res * 1000 for res in atlas.resolution],
        dest_path=path / "template.nii.gz",
    )
    save_as_asr_nii(
        atlas.annotation,
        vox_sizes=[res * 1000 for res in atlas.resolution],
        dest_path=path / "annotations.nii.gz",
    )
    write_itk_labels(path=path / "labels.txt", labels=atlas.structures_list)


if __name__ == "__main__":
    working_dir = Path.home() / "brainglobe_workingdir/"
    atlas_name = "eurasian_blackcap"
    resolution = 25
    minor_version = 5

    atlas = BrainGlobeAtlas(
        f"{atlas_name}_{resolution}um",
        check_latest=False,
        brainglobe_dir=working_dir,
    )

    brainglobe_atlas_to_itksnap(atlas=atlas, path=Path.home())
