"""Unit tests for brainglobe_atlasapi/descriptors.py format functions."""

from brainglobe_atlasapi import descriptors


def test_format_component_stub_structure():
    """Test that format_component_stub builds the correct path structure."""
    result = descriptors.format_component_stub(
        "my-template", "1.0.0", "templates", "file.zarr"
    )
    assert result == "templates/my-template/1_0_0/file.zarr"


def test_format_component_stub_replaces_dots_in_version():
    """Test that format_component_stub replaces dots with underscores."""
    result = descriptors.format_component_stub(
        "comp", "2.3.4", "root", "f.ext"
    )
    version_segment = result.split("/")[2]
    assert version_segment == "2_3_4"


def test_format_template_stub():
    """Test that format_template_stub correctly formats template paths."""
    result = descriptors.format_template_stub("my-template", "1.0.0")
    expected = (
        f"{descriptors.V2_TEMPLATE_ROOTDIR}/my-template/1_0_0"
        f"/{descriptors.V2_TEMPLATE_NAME}"
    )
    assert result == expected


def test_format_annotation_stub():
    """Test that format_annotation_stub correctly formats annotation paths."""
    result = descriptors.format_annotation_stub("my-annotation", "2.1.0")
    expected = (
        f"{descriptors.V2_ANNOTATION_ROOTDIR}/my-annotation/2_1_0"
        f"/{descriptors.V2_ANNOTATION_NAME}"
    )
    assert result == expected


def test_format_hemispheres_stub():
    """Test that format_hemispheres_stub correctly formats paths."""
    result = descriptors.format_hemispheres_stub("my-annotation", "1.0.0")
    expected = (
        f"{descriptors.V2_ANNOTATION_ROOTDIR}/my-annotation/1_0_0"
        f"/{descriptors.V2_HEMISPHERES_NAME}"
    )
    assert result == expected


def test_format_terminology_stub():
    """Test that format_terminology_stub correctly formats paths."""
    result = descriptors.format_terminology_stub("my-terminology", "1.0.0")
    expected = (
        f"{descriptors.V2_TERMINOLOGY_ROOTDIR}/my-terminology/1_0_0"
        f"/{descriptors.V2_TERMINOLOGY_NAME}"
    )
    assert result == expected


def test_format_meshes_stub():
    """Test that format_meshes_stub correctly formats meshes paths."""
    result = descriptors.format_meshes_stub("my-annotation", "1.0.0")
    expected = (
        f"{descriptors.V2_ANNOTATION_ROOTDIR}/my-annotation/1_0_0"
        f"/{descriptors.V2_MESHES_DIRECTORY}"
    )
    assert result == expected
