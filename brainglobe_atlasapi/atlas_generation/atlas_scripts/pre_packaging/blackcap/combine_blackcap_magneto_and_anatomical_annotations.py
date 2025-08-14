"""One-off script to reproduce how we combined
the second version of the annotations provided by the bird anatomists.
"""

from pathlib import Path

from brainglobe_utils.IO.image import load_nii, save_any
from scipy.ndimage import binary_erosion, generic_filter


def modal_filter_ignore_zeros(window):
    """
    Compute the mode of the window, ignoring zero values.

    Parameters
    ----------
    window : numpy.ndarray
        The input window of values.

    Returns
    -------
    int or float
        The most common non-zero value in the window, or 0 if all values
        are zero.
    """
    # Remove zeros from the window
    non_zero_values = window[window != 0]
    if len(non_zero_values) == 0:
        return 0  # If all values are zero, return 0
    # Compute the mode (most common value)
    values, counts = np.unique(non_zero_values, return_counts=True)
    return values[np.argmax(counts)]


def apply_modal_filter(image, filter_size=3):
    """Apply a modal filter to the image, ignoring zero neighbors.

    Parameters
    ----------
    image : numpy.ndarray
        Input image as a 2D NumPy array.
    filter_size : int
        Size of the filtering window (must be odd).

    Returns
    -------
    numpy.ndarray
        Filtered image.
    """
    # Apply the modal filter using a sliding window
    filtered_image = generic_filter(
        image, function=modal_filter_ignore_zeros, size=filter_size
    )
    return filtered_image


if __name__ == "__main__":
    atlas_path = Path(
        "/media/ceph/margrie/sweiler/AnalyzedData/"
        "Bird_brain_project/black_cap/final_annotations24/"
    )

    detailed_annotations_file = (
        atlas_path
        / "Detailed annotations_281124/DetailedAnnotations_281124.nii.gz"
    )

    magneto_annotations_file = (
        atlas_path
        / "Magnetic Brain Areas_281124/magnetic_annotation_281124.nii.gz"
    )

    # open magneto and detailed image
    detailed_annotations = load_nii(detailed_annotations_file, as_array=True)
    magneto_annotations = load_nii(magneto_annotations_file, as_array=True)

    # make a numpy copy of detailed image
    combined_annotations = detailed_annotations.copy()

    ## pons magnetic areas
    # make detailed 180 where detailed is 5 and magneto is 180
    # make detailed 190 where detailed is 5 and magneto is 190
    # make detailed 10 where detailed is 5 and magneto is 18
    # make detailed 17 where detailed is 5 and magneto is 17
    import numpy as np

    assert np.any(
        np.logical_and(detailed_annotations == 5, magneto_annotations == 18)
    )
    combined_annotations[
        np.logical_and(detailed_annotations == 5, magneto_annotations == 18)
    ] = 18
    combined_annotations[
        np.logical_and(detailed_annotations == 5, magneto_annotations == 19)
    ] = 19
    combined_annotations[
        np.logical_and(detailed_annotations == 5, magneto_annotations == 10)
    ] = 10
    combined_annotations[
        np.logical_and(detailed_annotations == 5, magneto_annotations == 17)
    ] = 17

    ## Cluster N
    # make detailed 305 where detailed is 30 and magneto is 5
    # make detailed 505 where detailed is 50 and magneto is 5
    # make detailed 605 where detailed is 60 and magneto is 5
    # make detailed 905 where detailed is 90 and magneto is 5
    combined_annotations[
        np.logical_and(detailed_annotations == 30, magneto_annotations == 5)
    ] = 305
    combined_annotations[
        np.logical_and(detailed_annotations == 50, magneto_annotations == 5)
    ] = 505
    combined_annotations[
        np.logical_and(detailed_annotations == 90, magneto_annotations == 5)
    ] = 905

    ## NFT
    # make detailed 402 where detailed is 40 and magneto is 20
    combined_annotations[
        np.logical_and(detailed_annotations == 40, magneto_annotations == 20)
    ] = 402

    ## smooth labels preserving background-foreground boundary
    filter_size = 3
    combined_annotations = apply_modal_filter(
        combined_annotations, filter_size=filter_size
    )

    # annotations are a bit too wide
    # erode by a few pixels
    # but not on mid-sagittal plane!
    erosions = 3
    has_label = combined_annotations > 0
    mirrored_has_label = np.flip(has_label, axis=2)
    has_label = np.concatenate((has_label, mirrored_has_label), axis=2)
    has_label = binary_erosion(has_label, iterations=erosions)
    has_label = has_label[:, :, : has_label.shape[2] // 2]
    combined_annotations *= has_label

    # save as nifti
    save_any(
        combined_annotations,
        atlas_path
        / "CombinedBrainAreas_291124"
        / f"combined_annotations_modal-{filter_size}_eroded-{erosions}.nii.gz",
    )
