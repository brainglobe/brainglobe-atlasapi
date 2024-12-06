"""One-off script to reproduce how we halved the symmetric blackcap template
to simplify annotation.
"""

from pathlib import Path

from brainglobe_utils.IO.image import load_nii, save_any

# note that this messes with the orientation in the nifti header
# so we need to manually correct with ITK snap!

if __name__ == "__main__":
    # make hemi-template
    template_root = Path(
        "/media/ceph-neuroinformatics/neuroinformatics/atlas-forge/BlackCap/templates/template_sym_res-25um_n-18_average-trimean/for_atlas/"
    )
    template_path = template_root / "reference_res-25um_image.nii.gz"
    # rescale reference volume into int16 range
    reference_volume = load_nii(template_path, as_array=True)

    extent_LR = reference_volume.shape[2]
    half_image = extent_LR // 2

    right_hemi_template = reference_volume[:, :, 0:half_image]
    save_any(
        right_hemi_template,
        template_root / "reference_res-25_hemi-right_image.nii.gz",
    )
