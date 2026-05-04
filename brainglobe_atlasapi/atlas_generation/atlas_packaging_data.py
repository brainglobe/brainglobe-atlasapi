"""Dataclass for holding all atlas packaging data."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import brainglobe_space as bgs
import numpy as np
import numpy.typing as npt
import s3fs
import tifffile
from fsspec.callbacks import TqdmCallback

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.atlas_generation.structures import (
    check_struct_consistency,
    filter_structures_not_present_in_annotation,
)
from brainglobe_atlasapi.descriptors import (
    Resolution,
    ResolutionList,
    ValidComponentData,
)


def check_requested_component(
    component_info: "ComponentInfo",
    working_dir: Path,
):
    """
    Check if a requested component already exists remotely and fetch it.

    If the component is set to skip_saving, will check if it exists remotely
    and fetch all metadata files locally.
    If the component is set to update_existing, will check if the existing
    version exists remotely and fetch all data and metadata files locally.

    Parameters
    ----------
    component_info : ComponentInfo
        A dictionary containing the component name, version, and a booleans for
        whether the component is published and whether to update existing.
    local_path : Path
        The local directory where the component metadata should be saved.
    component_root : str
        The root directory in the remote storage where the component is stored.
    component_file_name : str
        The name of the component file (e.g., "anatomical_template.ome.zarr").

    Raises
    ------
    ValueError
        If update_existing is True but existing_version is not provided.
    FileNotFoundError
        If the requested component or existing version is not found remotely.
    """
    if component_info.update_existing:
        if not component_info.existing_version:
            raise ValueError(
                "To update an existing component, 'existing_version' "
                "must be specified in component_info."
            )
        stub = component_info.existing_stub
    elif component_info.skip_saving:
        stub = component_info.stub
    else:
        return

    fs = s3fs.S3FileSystem(anon=True)

    component_stub = "/".join(stub.split("/")[:-1])

    remote_path = descriptors.remote_url_s3.format(component_stub)
    local_path = working_dir / component_stub

    if not fs.exists(remote_path):
        raise FileNotFoundError(
            f"Requested component {component_info.name} "
            f"not found at {remote_path}"
        )

    local_path.parent.mkdir(parents=True, exist_ok=True)

    if component_info.skip_saving:
        # Add wildcard to fetch all OME-Zarr metadata files
        if component_info.file_name.endswith(".ome.zarr"):
            remote_path += "/**/*.json"

        fs.get(
            remote_path,
            local_path,
        )
    elif component_info.update_existing:
        fs.get(
            remote_path,
            local_path,
            recursive=True,
            callback=TqdmCallback(
                desc=f"Fetching existing component {component_info.name}"
            ),
        )


@dataclass
class ComponentInfo:
    """
    Container for information about a component of a BrainGlobe atlas
    (e.g., template, annotation).

    This dataclass holds information about a specific component of a
    BrainGlobe atlas, such as a template or annotation. It includes
    fields for the component's name, version, and metadata, as well as
    flags for whether to skip saving the component data or
    update an existing component.

    Attributes
    ----------
    name : str
        The name of the component (e.g., "allen-adult-mouse-stpt-template").
    version : str
        The version of the component (e.g., "0.1.0").
    skip_saving : bool, optional
        Whether to skip saving the component data (default is False).
    update_existing : bool, optional
        Whether to update an existing component with the same name and version
        (default is False).
    existing_version : str, optional
        The version of the existing component to update
        (required if update_existing is True).
    root_dir : str, optional
        The root directory for the component (e.g., "templates").
    file_name : str, optional
        The name of the component file (e.g., "anatomical_template.ome.zarr").
    existing_stub : str, optional
        The remote stub for the existing component
        (automatically generated if update_existing is True).
    stub : str, optional
        The remote stub for the component
        (automatically generated if not provided).
    metadata : dict, optional
        A dictionary to hold the component metadata
        (automatically generated in __post_init__).
    """

    name: str
    version: str
    skip_saving: bool = False
    update_existing: bool = False
    existing_version: Optional[str] = None
    root_dir: Optional[str] = None
    file_name: Optional[str] = None
    existing_stub: Optional[str] = None
    stub: Optional[str] = None
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization processing to generate stubs and metadata.

        This method is called automatically after the dataclass is initialized.
        It generates the stubs for the component based on the provided
        name, version, and root directory.
        """
        self.version = self.version.replace(".", "_")
        if self.existing_version:
            self.existing_version = self.existing_version.replace(".", "_")

        if self.update_existing and self.existing_version:
            self.existing_stub = descriptors.format_component_stub(
                self.name,
                self.existing_version,
                self.root_dir,
                self.file_name,
            )

        if self.stub is None:
            self.stub = descriptors.format_component_stub(
                self.name,
                self.version,
                self.root_dir,
                self.file_name,
            )

        self.metadata = self.generate_metadata_dict()

    def generate_metadata_dict(self) -> Dict[str, str]:
        """
        Generate a dictionary containing metadata for this component.

        The metadata dictionary includes the component's name, version, and
        location stub.

        Returns
        -------
        Dict[str, str]
            A dictionary containing the component metadata.
        """
        metadata = {
            "name": self.name,
            "version": self.version.replace("_", "."),
            "location": f"/{self.root_dir}/{self.name}/{self.version}",
        }

        return metadata


@dataclass(kw_only=True)
class TemplateInfo(ComponentInfo):
    """
    Container for the template component of a BrainGlobe atlas.

    This dataclass holds information about a template component of a
    BrainGlobe atlas. It inherits from ComponentInfo and specifies the root
    directory and file name for template components.

    Attributes
    ----------
    root_dir : str, optional
        The root directory for template components
        (default is descriptors.V2_TEMPLATE_ROOTDIR).
    file_name : str, optional
        The name of the template component file
        (default is descriptors.V2_TEMPLATE_NAME).
    """

    root_dir: str = descriptors.V2_TEMPLATE_ROOTDIR
    file_name: str = descriptors.V2_TEMPLATE_NAME


@dataclass(kw_only=True)
class TerminologyInfo(ComponentInfo):
    """
    Container for the terminology component of a BrainGlobe atlas.

    This dataclass holds information about a terminology component of a
    BrainGlobe atlas. It inherits from ComponentInfo and specifies the root
    directory and file name for terminology components.

    Attributes
    ----------
    root_dir : str, optional
        The root directory for terminology components
        (default is descriptors.V2_TERMINOLOGY_ROOTDIR).
    file_name : str, optional
        The name of the terminology component file
        (default is descriptors.V2_TERMINOLOGY_NAME).
    """

    root_dir: str = descriptors.V2_TERMINOLOGY_ROOTDIR
    file_name: str = descriptors.V2_TERMINOLOGY_NAME


@dataclass(kw_only=True)
class AnnotationInfo(ComponentInfo):
    """
    Container for the annotation component of a BrainGlobe atlas.

    This dataclass holds information about an annotation component of a
    BrainGlobe atlas. It inherits from ComponentInfo and specifies the root
    directory and file name for annotation components.

    Overrides the __post_init__ method to include template and terminology
    metadata in the annotation metadata.

    Attributes
    ----------
    template : TemplateInfo
        The TemplateInfo object associated with this annotation component.
    terminology : TerminologyInfo
        The TerminologyInfo object associated with this annotation component.
    root_dir : str, optional
        The root directory for annotation components
        (default is descriptors.V2_ANNOTATION_ROOTDIR).
    file_name : str, optional
        The name of the annotation component file
        (default is descriptors.V2_ANNOTATION_NAME).
    """

    template: TemplateInfo
    terminology: TerminologyInfo
    root_dir: str = descriptors.V2_ANNOTATION_ROOTDIR
    file_name: str = descriptors.V2_ANNOTATION_NAME

    def __post_init__(self):
        """
        Post-initialization processing to include template and terminology
        metadata in the annotation metadata.
        """
        super().__post_init__()
        self.metadata.update(
            {
                "template": self.template.metadata,
                "terminology": self.terminology.metadata,
            }
        )


@dataclass(kw_only=True)
class CoordinateSpaceInfo(ComponentInfo):
    """
    Container for information about a coordinate space component of a
    BrainGlobe atlas.

    This dataclass holds information about a coordinate space component
    of a BrainGlobe atlas. It inherits from ComponentInfo and specifies
    the root directory and file name for coordinate space components.

    Overrides the __post_init__ method to include template metadata in the
    coordinate space metadata.

    Attributes
    ----------
    template : TemplateInfo
        The TemplateInfo object associated with this coordinate space.
    root_dir : str, optional
        The root directory for coordinate space components
        (default is descriptors.V2_COORDINATE_SPACE_ROOTDIR).
    file_name : str, optional
        The name of the coordinate space component file
        (default is "manifest.json").
    """

    template: TemplateInfo
    root_dir: str = descriptors.V2_COORDINATE_SPACE_ROOTDIR
    file_name: str = "manifest.json"

    def __post_init__(self):
        """
        Post-initialization processing to include template metadata in the
        coordinate space metadata.
        """
        super().__post_init__()
        self.metadata.update(
            {
                "template": self.template.metadata,
            }
        )


@dataclass
class AtlasPackagingData:
    """Container for all data required to package a BrainGlobe atlas.

    This dataclass is an internal implementation detail of the atlas packaging
    pipeline. It collects all atlas data in one place to avoid threading
    individual arguments through many function signatures.

    Attributes
    ----------
    atlas_name : str
        Atlas name in the form author_species.
    atlas_minor_version : int or str
        Minor version number for this particular atlas.
    citation : str
        Citation for the atlas, if unpublished specify "unpublished".
    atlas_link : str
        Valid URL for the atlas.
    species : str
        Species name formatted as "CommonName (Genus species)".
    resolution : Resolution | ResolutionList
        Resolution on three axes, or a list of such tuples for multi-scale.
    orientation : str
        Orientation of the original atlas (tuple describing origin for
        BGSpace).
    root_id : int
        Id of the root element of the atlas.
    reference_stack : ValidComponentData
        Reference stack for the atlas. If str or Path, will be read with
        tifffile. If list, should be ordered from highest to lowest
        resolution.
    annotation_stack : ValidComponentData
        Annotation stack for the atlas. If str or Path, will be read with
        tifffile. If list, should be ordered from highest to lowest
        resolution.
    structures_list : List[Dict]
        List of valid dictionaries for structures.
    meshes_dict : Dict[int | str, str | Path]
        Dict of meshio-compatible mesh file paths in the form
        {struct_id: meshpath}.
    atlas_packager : str, optional
        Credit for those responsible for converting the atlas into the
        BrainGlobe format.
    hemispheres_stack : ValidComponentData, optional
        Hemisphere stack for the atlas. If None, atlas is assumed symmetric.
    additional_references : List[Tuple[TemplateInfo, ValidComponentData]], optional
        List of tuples containing metadata and arrays for secondary
        templates.
    additional_metadata : Dict, optional
        Additional metadata to write to metadata.json.
    symmetric : bool, optional
        Whether the atlas is symmetric across the midline.
    """  # noqa: E501

    atlas_name: str
    atlas_version: str
    citation: str
    atlas_link: str
    species: str
    resolution: Resolution | ResolutionList
    root_id: int
    working_dir: Path
    reference_stack: ValidComponentData
    annotation_stack: ValidComponentData
    structures_list: List[Dict]
    meshes_dict: Dict[int | str, str | Path]
    template_info: TemplateInfo
    annotation_info: AnnotationInfo
    terminology_info: TerminologyInfo
    coordinate_space_info: CoordinateSpaceInfo
    orientation: str = descriptors.ATLAS_ORIENTATION
    space_convention: bgs.AnatomicalSpace | None = None
    atlas_version_underscore: str = None
    atlas_packager: str | None = None
    hemispheres_stack: ValidComponentData = None
    additional_references: List[
        Tuple[
            TemplateInfo,
            ValidComponentData,
        ],
    ] = field(default_factory=list)
    additional_metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Post-initialization processing to standardize data formats and
        check consistency.
        """
        self.atlas_version_underscore = self.atlas_version.replace(".", "_")

        self.resolution = _standardize_resolution(self.resolution)
        self.reference_stack = _load_stack(self.reference_stack)
        self.annotation_stack = _load_stack(self.annotation_stack)

        shape = self.reference_stack[0].shape
        volume_shape = tuple(
            res * s for res, s in zip(self.resolution[0], shape)
        )
        self.space_convention = bgs.AnatomicalSpace(
            self.orientation, shape=volume_shape
        )

        self.reference_stack = _reorient_stacks(
            self.reference_stack, self.space_convention
        )
        self.annotation_stack = _reorient_stacks(
            self.annotation_stack, self.space_convention
        )

        for i, stack_tuple in enumerate(self.additional_references):
            ref_stack = _load_stack(stack_tuple[1])
            ref_stack = _reorient_stacks(ref_stack, self.space_convention)
            self.additional_references[i] = (stack_tuple[0], ref_stack)

        self.symmetric = self.hemispheres_stack is None

        if not self.symmetric:
            self.hemispheres_stack = _load_stack(self.hemispheres_stack)
        else:
            self.hemispheres_stack = _auto_generate_hemispheres(
                shapes=[stack.shape for stack in self.annotation_stack],
            )

        self.structures_list = filter_structures_not_present_in_annotation(
            self.structures_list, self.annotation_stack[0]
        )

        check_struct_consistency(self.structures_list)

        check_requested_component(self.template_info, self.working_dir)
        check_requested_component(self.annotation_info, self.working_dir)
        check_requested_component(self.terminology_info, self.working_dir)
        check_requested_component(self.coordinate_space_info, self.working_dir)

        for template_info, _ in self.additional_references:
            check_requested_component(template_info, self.working_dir)


def _standardize_resolution(
    resolution: Resolution | ResolutionList,
) -> ResolutionList:
    if isinstance(resolution, tuple):
        return [resolution]
    elif isinstance(resolution, list):
        return resolution
    else:
        raise ValueError(
            "Resolution must be either a tuple or a list of tuples."
        )


def _auto_generate_hemispheres(
    shapes: List[tuple],
) -> List[npt.NDArray]:
    hemispheres_stack = [np.full(shape, 2, dtype=np.uint8) for shape in shapes]
    slices = ([slice(None) for _ in range(3)],) * len(shapes)
    for stack, slice_set in zip(hemispheres_stack, slices):
        slice_set[2] = slice(round(stack.shape[2] / 2), None)
        stack[tuple(slice_set)] = 1
    return hemispheres_stack


def _load_stack(
    stack: ValidComponentData,
) -> List[npt.NDArray]:
    if isinstance(stack, (str, Path)):
        return [tifffile.imread(stack)]
    elif isinstance(stack, np.ndarray):
        return [stack]
    return stack


def _reorient_stacks(
    stacks: List[npt.NDArray],
    space_convention: bgs.AnatomicalSpace,
) -> List[npt.NDArray]:
    return [
        space_convention.map_stack_to(
            descriptors.ATLAS_ORIENTATION, stack, copy=False
        )
        for stack in stacks
    ]
