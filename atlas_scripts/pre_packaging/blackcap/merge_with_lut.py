"""
Quick and dirty script by @sweiler to merge several annotations from ITK snap.
Used for blackcap template v1.3 and v1.4 preparation.

Included for completeness.
"""

import re
import sys

import nibabel as nib
import numpy as np


def read_itksnap_lut(lut_file):
    """
    Read an ITK-SNAP label description file into a dictionary.

    Should have entries: {original_label_id: (R,G,B,A,VIS,MSH,LabelName)}.
    """
    lut = {}
    with open(lut_file, "r") as f:
        for line in f:
            if line.strip().startswith("#") or not line.strip():
                continue
            parts = re.split(r"\s+", line.strip(), maxsplit=7)
            if len(parts) < 8:
                continue
            idx = int(parts[0])
            r, g, b = map(int, parts[1:4])
            a = float(parts[4])
            vis = int(parts[5])
            msh = int(parts[6])
            name = parts[7].strip().strip('"')
            lut[idx] = (r, g, b, a, vis, msh, name)
    return lut


def merge_segmentations(output_seg, output_lut, nii_files, lut_files):
    """
    Merge multiple NIfTI segmentation files and their LUTs
    into a single segmentation and a single ITK-SNAP label
    description file with unique IDs and colors.
    """
    # Reference image for shape and affine
    ref_img = nib.load(nii_files[0])
    merged_data = np.zeros(ref_img.shape, dtype=np.int32)

    current_max = 0
    new_lut_entries = []

    # Loop over NIfTI + LUT pairs
    for idx, (seg_file, lut_file) in enumerate(
        zip(nii_files, lut_files), start=1
    ):
        seg_img = nib.load(seg_file)
        seg_data = seg_img.get_fdata().astype(np.int32)

        lut = read_itksnap_lut(lut_file)

        labels = np.unique(seg_data)
        labels = labels[labels != 0]  # skip background

        for label in labels:
            mask = seg_data == label
            new_label = current_max + label
            merged_data[mask] = new_label

            if label in lut:
                r, g, b, a, vis, msh, name = lut[label]
            else:
                r, g, b, a, vis, msh, name = (
                    200,
                    200,
                    200,
                    1.0,
                    1,
                    1,
                    f"Unknown_{label}",
                )

            # Add suffix to track original file
            new_lut_entries.append(
                (new_label, r, g, b, a, vis, msh, f"{name}_from_{seg_file}")
            )

        current_max = merged_data.max()
        print(f"? {seg_file} merged, labels now up to {current_max}")

    # Save merged NIfTI
    merged_img = nib.Nifti1Image(merged_data, ref_img.affine, ref_img.header)
    nib.save(merged_img, output_seg)
    print(f"\n Saved merged segmentation: {output_seg}")

    # Save merged LUT
    with open(output_lut, "w") as f:
        f.write("################################################\n")
        f.write("# ITK-SnAP Label Description File\n")
        f.write("# IDX   -R-  -G-  -B-  -A--  VIS MSH  LABEL\n")
        f.write("################################################\n")
        f.write('    0     0    0    0        0  0  0    "Clear Label"\n')

        for idx, r, g, b, a, vis, msh, name in new_lut_entries:
            f.write(
                f'{idx:<5d} {r:<4d} {g:<4d} {b:<4d} {a:<8.2f} {vis:<2d} {msh:<2d} "{name}"\n'  # noqa
            )

    print(f"?? Saved merged LUT: {output_lut}")


if __name__ == "__main__":
    if len(sys.argv) < 6 or (len(sys.argv) - 3) % 2 != 0:
        print(
            "Usage: python merge_with_lut.py output_seg.nii.gz output_labels.txt "  # noqa
            "seg1.nii.gz lut1.txt seg2.nii.gz lut2.txt ..."
        )
        sys.exit(1)

    output_seg = sys.argv[1]
    output_lut = sys.argv[2]
    pairs = sys.argv[3:]

    nii_files = pairs[0::2]  # every 2nd argument starting at 0
    lut_files = pairs[1::2]  # every 2nd argument starting at 1

    merge_segmentations(output_seg, output_lut, nii_files, lut_files)
