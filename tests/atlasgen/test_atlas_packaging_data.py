"""
Tests for the atlas_packaging_data module.

Verifies the correctness of resolution standardisation, stack loading,
hemisphere auto-generation, and stack reorientation helpers.
"""

import brainglobe_space as bgs
import numpy as np
import pytest
import tifffile

from brainglobe_atlasapi import descriptors

# Imported for use in Tasks 2-5 tests (not yet used in Task 1)
from brainglobe_atlasapi.atlas_generation.atlas_packaging_data import (
    AnnotationInfo,  # noqa: F401
    AtlasPackagingData,  # noqa: F401
    ComponentInfo,  # noqa: F401
    CoordinateSpaceInfo,  # noqa: F401
    TemplateInfo,  # noqa: F401
    TerminologyInfo,  # noqa: F401
    _auto_generate_hemispheres,
    _load_stack,
    _reorient_stacks,
    _standardize_resolution,
    check_requested_component,  # noqa: F401
)

# --- _standardize_resolution ---


@pytest.mark.parametrize(
    "resolution, expected",
    [
        pytest.param(
            (10, 20, 30),
            [(10, 20, 30)],
            id="single tuple",
        ),
        pytest.param(
            [(10, 20, 30), (20, 40, 60)],
            [(10, 20, 30), (20, 40, 60)],
            id="list of tuples",
        ),
    ],
)
def test_standardize_resolution(resolution, expected):
    """Test `_standardize_resolution` with various inputs.

    Parameters
    ----------
    resolution : tuple or list of tuples
        The resolution input to standardize.
    expected : list of tuples
        The expected output after standardization.
    """
    assert _standardize_resolution(resolution) == expected


def test_standardize_resolution_invalid():
    """Test `_standardize_resolution` raises ValueError for invalid input."""
    with pytest.raises(ValueError, match="Resolution must be either"):
        _standardize_resolution("invalid")


# --- _load_stack ---


def test_load_stack_ndarray():
    """Test `_load_stack` wraps a numpy array in a list."""
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    result = _load_stack(arr)
    assert isinstance(result, list)
    assert len(result) == 1
    assert result[0] is arr


def test_load_stack_list_passthrough():
    """Test `_load_stack` passes through a list unchanged."""
    arr1 = np.zeros((4, 4, 4), dtype=np.uint16)
    arr2 = np.ones((4, 4, 4), dtype=np.uint16)
    result = _load_stack([arr1, arr2])
    assert result == [arr1, arr2]


def test_load_stack_path_returns_list(tmp_path):
    """Test `_load_stack` returns a single-element list when given a Path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(tiff_path)
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_stack_path_reads_data(tmp_path):
    """Test `_load_stack` reads correct data from a TIFF file at a Path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(tiff_path)
    assert np.array_equal(result[0], arr)


def test_load_stack_str_returns_list(tmp_path):
    """Test `_load_stack` returns a list when given a string path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(str(tiff_path))
    assert isinstance(result, list)
    assert len(result) == 1


def test_load_stack_str_reads_data(tmp_path):
    """Test `_load_stack` reads correct data from a string TIFF file path.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    arr = np.zeros((4, 4, 4), dtype=np.uint16)
    tiff_path = tmp_path / "test.tiff"
    tifffile.imwrite(tiff_path, arr)
    result = _load_stack(str(tiff_path))
    assert np.array_equal(result[0], arr)


# --- _auto_generate_hemispheres ---


def test_auto_generate_hemispheres_shape():
    """Test `_auto_generate_hemispheres` with matching shapes."""
    shapes = [(4, 4, 4)]
    annotation = [np.zeros((4, 4, 4), dtype=np.uint32)]
    result = _auto_generate_hemispheres(shapes, annotation)
    assert len(result) == 1
    assert result[0].shape == (4, 4, 4)


def test_auto_generate_hemispheres_values():
    """Test `_auto_generate_hemispheres` splits the volume at the midpoint.

    The left half (dim 2, indices < midpoint) should be 2.
    The right half (dim 2, indices >= midpoint) should be 1.
    """
    shapes = [(4, 4, 4)]
    annotation = [np.zeros((4, 4, 4), dtype=np.uint32)]
    result = _auto_generate_hemispheres(shapes, annotation)
    assert np.all(result[0][:, :, :2] == 2)
    assert np.all(result[0][:, :, 2:] == 1)


def test_auto_generate_hemispheres_multiple_scales():
    """Test `_auto_generate_hemispheres` handles multiple annotation scales."""
    shapes = [(4, 4, 4), (2, 2, 2)]
    annotations = [
        np.zeros((4, 4, 4), dtype=np.uint32),
        np.zeros((2, 2, 2), dtype=np.uint32),
    ]
    result = _auto_generate_hemispheres(shapes, annotations)
    assert len(result) == 2
    assert result[0].shape == (4, 4, 4)
    assert result[1].shape == (2, 2, 2)


# --- _reorient_stacks ---


def test_reorient_stacks_identity():
    """Test `_reorient_stacks` leaves stacks unchanged when already in asr."""
    arr = np.arange(64, dtype=np.uint16).reshape((4, 4, 4))
    space = bgs.AnatomicalSpace(descriptors.ATLAS_ORIENTATION, shape=arr.shape)
    result = _reorient_stacks([arr], space)
    assert len(result) == 1
    assert np.array_equal(result[0], arr)


def test_reorient_stacks_reorders_axes():
    """Test `_reorient_stacks` correctly reorders axes for non-asr input."""
    arr = np.arange(64, dtype=np.uint16).reshape((4, 4, 4))
    # "sar" orientation means the mapping to "asr" should permute axes
    space = bgs.AnatomicalSpace("sar", shape=arr.shape)
    result = _reorient_stacks([arr], space)
    assert len(result) == 1
    expected = space.map_stack_to(
        descriptors.ATLAS_ORIENTATION, arr, copy=True
    )
    assert np.array_equal(result[0], expected)


# --- ComponentInfo ---


@pytest.fixture
def component_info():
    """Provide a ComponentInfo instance for testing.

    Returns
    -------
    ComponentInfo
        A ComponentInfo instance with name, version, root_dir, and file_name.
    """
    return ComponentInfo(
        name="test-component",
        version="1.2.3",
        root_dir="templates",
        file_name="anatomical_template.ome.zarr",
    )


def test_component_info_version_underscore(component_info):
    """Test ComponentInfo converts version dots to underscores.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert component_info.version == "1_2_3"


def test_component_info_stub(component_info):
    """Test ComponentInfo generates the correct stub.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    expected = descriptors.format_component_stub(
        "test-component", "1_2_3", "templates", "anatomical_template.ome.zarr"
    )
    assert component_info.stub == expected


def test_component_info_existing_stub_when_update_existing():
    """Test ComponentInfo generates existing_stub when update_existing=True."""
    info = ComponentInfo(
        name="test-component",
        version="2.0.0",
        existing_version="1.0.0",
        update_existing=True,
        root_dir="templates",
        file_name="anatomical_template.ome.zarr",
    )
    expected = descriptors.format_component_stub(
        "test-component", "1_0_0", "templates", "anatomical_template.ome.zarr"
    )
    assert info.existing_stub == expected


def test_component_info_no_existing_stub_without_update_existing():
    """Test ComponentInfo does not generate existing_stub when not updating."""
    info = ComponentInfo(
        name="test-component",
        version="1.0.0",
        root_dir="templates",
        file_name="anatomical_template.ome.zarr",
    )
    assert info.existing_stub is None


def test_component_info_metadata_has_name_key(component_info):
    """Test ComponentInfo.metadata contains a 'name' key.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert "name" in component_info.metadata


def test_component_info_metadata_has_version_key(component_info):
    """Test ComponentInfo.metadata contains a 'version' key.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert "version" in component_info.metadata


def test_component_info_metadata_has_location_key(component_info):
    """Test ComponentInfo.metadata contains a 'location' key.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert "location" in component_info.metadata


def test_component_info_metadata_version_uses_dots(component_info):
    """Test ComponentInfo.metadata version restores dots from underscores.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert component_info.metadata["version"] == "1.2.3"


def test_component_info_metadata_location(component_info):
    """Test ComponentInfo.metadata location contains root_dir and version.

    Parameters
    ----------
    component_info : ComponentInfo
        A ComponentInfo instance for testing.
    """
    assert "templates" in component_info.metadata["location"]
    assert "1_2_3" in component_info.metadata["location"]


# --- TemplateInfo, TerminologyInfo, AnnotationInfo, CoordinateSpaceInfo ---


@pytest.fixture
def template_info():
    """Provide a TemplateInfo instance for testing.

    Returns
    -------
    TemplateInfo
        A TemplateInfo instance with name and version.
    """
    return TemplateInfo(name="test-template", version="1.0")


@pytest.fixture
def terminology_info():
    """Provide a TerminologyInfo instance for testing.

    Returns
    -------
    TerminologyInfo
        A TerminologyInfo instance with name and version.
    """
    return TerminologyInfo(name="test-terminology", version="1.0")


@pytest.fixture
def annotation_info(template_info, terminology_info):
    """Provide an AnnotationInfo instance for testing.

    Parameters
    ----------
    template_info : TemplateInfo
        A TemplateInfo instance.
    terminology_info : TerminologyInfo
        A TerminologyInfo instance.

    Returns
    -------
    AnnotationInfo
        An AnnotationInfo instance linked to template and terminology.
    """
    return AnnotationInfo(
        name="test-annotation",
        version="1.0",
        template=template_info,
        terminology=terminology_info,
    )


@pytest.fixture
def coordinate_space_info(template_info):
    """Provide a CoordinateSpaceInfo instance for testing.

    Parameters
    ----------
    template_info : TemplateInfo
        A TemplateInfo instance.

    Returns
    -------
    CoordinateSpaceInfo
        A CoordinateSpaceInfo instance linked to the template.
    """
    return CoordinateSpaceInfo(
        name="test-space",
        version="1.0",
        template=template_info,
    )


def test_template_info_default_root_dir(template_info):
    """Test TemplateInfo uses V2_TEMPLATE_ROOTDIR as default root_dir.

    Parameters
    ----------
    template_info : TemplateInfo
        A TemplateInfo instance for testing.
    """
    assert template_info.root_dir == descriptors.V2_TEMPLATE_ROOTDIR


def test_template_info_default_file_name(template_info):
    """Test TemplateInfo uses V2_TEMPLATE_NAME as default file_name.

    Parameters
    ----------
    template_info : TemplateInfo
        A TemplateInfo instance for testing.
    """
    assert template_info.file_name == descriptors.V2_TEMPLATE_NAME


def test_terminology_info_default_root_dir(terminology_info):
    """Test TerminologyInfo uses V2_TERMINOLOGY_ROOTDIR as default root_dir.

    Parameters
    ----------
    terminology_info : TerminologyInfo
        A TerminologyInfo instance for testing.
    """
    assert terminology_info.root_dir == descriptors.V2_TERMINOLOGY_ROOTDIR


def test_terminology_info_default_file_name(terminology_info):
    """Test TerminologyInfo uses V2_TERMINOLOGY_NAME as default file_name.

    Parameters
    ----------
    terminology_info : TerminologyInfo
        A TerminologyInfo instance for testing.
    """
    assert terminology_info.file_name == descriptors.V2_TERMINOLOGY_NAME


def test_annotation_info_default_root_dir(annotation_info):
    """Test AnnotationInfo uses V2_ANNOTATION_ROOTDIR as default root_dir.

    Parameters
    ----------
    annotation_info : AnnotationInfo
        An AnnotationInfo instance for testing.
    """
    assert annotation_info.root_dir == descriptors.V2_ANNOTATION_ROOTDIR


def test_annotation_info_default_file_name(annotation_info):
    """Test AnnotationInfo uses V2_ANNOTATION_NAME as default file_name.

    Parameters
    ----------
    annotation_info : AnnotationInfo
        An AnnotationInfo instance for testing.
    """
    assert annotation_info.file_name == descriptors.V2_ANNOTATION_NAME


def test_annotation_info_metadata_contains_template(annotation_info):
    """Test AnnotationInfo.metadata includes template metadata.

    Parameters
    ----------
    annotation_info : AnnotationInfo
        An AnnotationInfo instance for testing.
    """
    assert "template" in annotation_info.metadata
    assert annotation_info.metadata["template"]["name"] == "test-template"


def test_annotation_info_metadata_contains_terminology(annotation_info):
    """Test AnnotationInfo.metadata includes terminology metadata.

    Parameters
    ----------
    annotation_info : AnnotationInfo
        An AnnotationInfo instance for testing.
    """
    assert "terminology" in annotation_info.metadata
    assert (
        annotation_info.metadata["terminology"]["name"] == "test-terminology"
    )


def test_coordinate_space_info_default_root_dir(coordinate_space_info):
    """Test CoordinateSpaceInfo uses V2_COORDINATE_SPACE_ROOTDIR as root_dir.

    Parameters
    ----------
    coordinate_space_info : CoordinateSpaceInfo
        A CoordinateSpaceInfo instance for testing.
    """
    assert (
        coordinate_space_info.root_dir
        == descriptors.V2_COORDINATE_SPACE_ROOTDIR
    )


def test_coordinate_space_info_default_file_name(coordinate_space_info):
    """Test CoordinateSpaceInfo uses "manifest.json" as default file_name.

    Parameters
    ----------
    coordinate_space_info : CoordinateSpaceInfo
        A CoordinateSpaceInfo instance for testing.
    """
    assert coordinate_space_info.file_name == "manifest.json"


def test_coordinate_space_info_metadata_contains_template(
    coordinate_space_info,
):
    """Test CoordinateSpaceInfo.metadata includes template metadata.

    Parameters
    ----------
    coordinate_space_info : CoordinateSpaceInfo
        A CoordinateSpaceInfo instance for testing.
    """
    assert "template" in coordinate_space_info.metadata
    assert (
        coordinate_space_info.metadata["template"]["name"] == "test-template"
    )


# --- check_requested_component ---


def test_check_requested_component_noop(mocker, tmp_path, template_info):
    """Test `check_requested_component` returns without S3 calls when
    neither skip_saving nor update_existing is set.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    tmp_path : Path
        Temporary directory path provided by pytest.
    template_info : TemplateInfo
        A TemplateInfo instance for testing.
    """
    mock_s3 = mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.s3fs.S3FileSystem"
    )
    check_requested_component(template_info, tmp_path)
    mock_s3.assert_not_called()


def test_check_requested_component_skip_saving_fetches(mocker, tmp_path):
    """Test `check_requested_component` fetches JSON metadata when
    skip_saving=True.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    mock_fs = mocker.MagicMock()
    mock_fs.exists.return_value = True
    mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.s3fs.S3FileSystem",
        return_value=mock_fs,
    )
    component = TemplateInfo(
        name="test-template",
        version="1.0",
        skip_saving=True,
        file_name="test.ome.zarr",
    )
    check_requested_component(component, tmp_path)
    mock_fs.get.assert_called_once()
    remote_arg = mock_fs.get.call_args[0][0]
    assert remote_arg.endswith("/**/*.json")


def test_check_requested_component_skip_saving_raises_if_not_found(
    mocker, tmp_path
):
    """Test `check_requested_component` raises FileNotFoundError when
    skip_saving=True but the remote component does not exist.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    mock_fs = mocker.MagicMock()
    mock_fs.exists.return_value = False
    mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.s3fs.S3FileSystem",
        return_value=mock_fs,
    )
    component = TemplateInfo(
        name="test-template", version="1.0", skip_saving=True
    )
    with pytest.raises(FileNotFoundError, match="not found at"):
        check_requested_component(component, tmp_path)


def test_check_requested_component_update_existing_raises_without_version(
    tmp_path,
):
    """Test `check_requested_component` raises ValueError when
    update_existing=True but existing_version is not set.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    component = TemplateInfo(
        name="test-template",
        version="2.0",
        update_existing=True,
    )
    with pytest.raises(ValueError, match="existing_version"):
        check_requested_component(component, tmp_path)


def test_check_requested_component_update_existing_fetches_recursively(
    mocker, tmp_path
):
    """Test `check_requested_component` fetches the existing component
    recursively when update_existing=True and the remote component exists.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    mock_fs = mocker.MagicMock()
    mock_fs.exists.return_value = True
    mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.s3fs.S3FileSystem",
        return_value=mock_fs,
    )
    mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.TqdmCallback"
    )
    component = TemplateInfo(
        name="test-template",
        version="2.0",
        existing_version="1.0",
        update_existing=True,
    )
    check_requested_component(component, tmp_path)
    mock_fs.get.assert_called_once()
    _, kwargs = mock_fs.get.call_args
    assert kwargs.get("recursive") is True


def test_check_requested_component_update_existing_raises_if_not_found(
    mocker, tmp_path
):
    """Test `check_requested_component` raises FileNotFoundError when
    update_existing=True but the remote component does not exist.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Mocker fixture for patching.
    tmp_path : Path
        Temporary directory path provided by pytest.
    """
    mock_fs = mocker.MagicMock()
    mock_fs.exists.return_value = False
    mocker.patch(
        "brainglobe_atlasapi.atlas_generation.atlas_packaging_data.s3fs.S3FileSystem",
        return_value=mock_fs,
    )
    component = TemplateInfo(
        name="test-template",
        version="2.0",
        existing_version="1.0",
        update_existing=True,
    )
    with pytest.raises(FileNotFoundError, match="not found at"):
        check_requested_component(component, tmp_path)
