"""One-off script to reproduce how we combined the first version of
the annotations provided by the bird anatomists"""

import csv
from pathlib import Path

import numpy as np
from brainglobe_utils.IO.image import load_nii, save_any

from brainglobe_atlasapi.atlas_generation.annotation_utils import (
    read_itk_labels,
    write_itk_labels,
)

if __name__ == "__main__":
    # setup paths
    annotations_root = Path(
        "/media/ceph-neuroinformatics/neuroinformatics/atlas-forge/BlackCap"
        / "templates/template_sym_res-25um_n-18_average-trimean/for_atlas"
        / "annotations-right-hemisphere/"
    )
    main_annotation_path = (
        annotations_root / "corrected_AnnotationsMainBrainAreas_SW.nii.gz"
    )
    small_annotation_path = annotations_root / "corrected_smallareas_SW.nii.gz"
    main_labels_path = (
        annotations_root / "corrected_LabelMainBrainAreas_SW.txt"
    )
    small_labels_path = annotations_root / "corrected_smallareas_SW.txt"
    small_to_main_csv = annotations_root / "hierarchy_annotat1_annotat2.csv"

    # combine annotation images
    main_annotation_image = load_nii(main_annotation_path, as_array=True)
    small_annotation_image = load_nii(small_annotation_path, as_array=True)
    small_annotation_image *= (
        10  # avoid label clashes, there are 6 main labels
    )
    small_annotation_image = small_annotation_image.astype(np.uint8)
    combined_annotation_image = main_annotation_image.copy()

    small_to_main_map = {}
    with open(small_to_main_csv, mode="r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            print(row)
            if row[3]:
                small_to_main_map[10 * int(row[3])] = int(row[1])
    for small, main in small_to_main_map.items():
        is_main_value = main_annotation_image == main
        is_small_value = small_annotation_image == small
        combined_indices = is_main_value * is_small_value
        combined_annotation_image[combined_indices] = small
    save_any(
        combined_annotation_image,
        annotations_root / "combined_annotations.nii.gz",
    )

    # combine label data
    main_annotation_labels = read_itk_labels(main_labels_path)
    small_annotation_labels = read_itk_labels(small_labels_path)
    for label in small_annotation_labels:
        label["id"] = int(10 * label["id"])
    combined_labels = main_annotation_labels + small_annotation_labels
    write_itk_labels(
        annotations_root / "combined_label_descriptions.txt", combined_labels
    )
