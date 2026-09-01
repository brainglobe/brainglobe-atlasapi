"""Module containing the core Atlas class."""

from __future__ import annotations

import warnings
from collections import UserDict, deque
from pathlib import Path
from typing import (
    Dict,
    Generic,
    List,
    Literal,
    Tuple,
    TypeVar,
    Union,
    cast,
    overload,
)

import dask.array as da
import ngff_zarr as nz
import numpy as np
import numpy.typing as npt
import pandas as pd
import s3fs
import zarr
from brainglobe_space import AnatomicalSpace
from fsspec.callbacks import TqdmCallback

from brainglobe_atlasapi.descriptors import (
    ATLAS_ORIENTATION,
    V3_ANNOTATION_MAP_NAME,
    V3_ANNOTATION_MASKS_NAME,
    V3_ANNOTATION_NAME,
    V3_HEMISPHERES_NAME,
    V3_MESHES_DIRECTORY,
    V3_TEMPLATE_NAME,
    V3_TERMINOLOGY_NAME,
    remote_url_s3,
)
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_atlasapi.utils import (
    load_structures_from_csv,
    read_json,
)

TemplateArray = TypeVar(
    "TemplateArray", bound=Union[npt.NDArray[np.uint16], da.Array]
)
AnnotationArray = TypeVar(
    "AnnotationArray", bound=Union[npt.NDArray[np.uint32], da.Array]
)
LabelArray = TypeVar(
    "LabelArray", bound=Union[npt.NDArray[np.uint8], da.Array]
)


def _determine_pyramid_level(
    multiscale: nz.Multiscales, resolution: Tuple[float, float, float]
) -> int:
    """Return the pyramid level matching ``resolution``.

    The scale of each level is read from ``NgffImage.scale``, which ngff-zarr
    populates from the coordinate transformations of the corresponding
    dataset. That mapping is keyed by dimension name and is the same whatever
    OME-Zarr version the store is written in, so it avoids interpreting the
    version-specific transformation metadata directly.

    ``images`` and ``metadata.datasets`` are built in the same pass over the
    stored datasets, so their indices refer to the same pyramid level.

    Parameters
    ----------
    multiscale : nz.Multiscales
        The multiscale image to search.
    resolution : tuple of float
        Requested resolution in microns, in ``(z, y, x)`` order.

    Returns
    -------
    int
        Index of the matching pyramid level.

    Raises
    ------
    ValueError
        If no pyramid level has the requested resolution.
    """
    for idx, image in enumerate(multiscale.images):
        # Only check spatial scale against resolution
        scales = [image.scale[dim] for dim in image.dims[-3:]]
        if all(
            np.isclose(res / 1000, scale)
            for res, scale in zip(resolution, scales)
        ):
            return idx

    raise ValueError(f"Requested resolution {resolution} um is invalid.")


class Atlas(Generic[TemplateArray, AnnotationArray, LabelArray]):
    """Base class to handle atlases in BrainGlobe.

    Parameters
    ----------
    path : str or Path object
        Path to folder containing data info.
    lazy : bool
        If True, atlas array properties return dask arrays instead of loading
        them into memory as numpy arrays.
    """

    left_hemisphere_value = 1
    right_hemisphere_value = 2

    @overload
    def __init__(
        self: Atlas[
            npt.NDArray[np.uint16],
            npt.NDArray[np.uint32],
            npt.NDArray[np.uint8],
        ],
        path: Union[str, Path],
        lazy: Literal[False] = False,
    ) -> None: ...

    @overload
    def __init__(
        self: Atlas[da.Array, da.Array, da.Array],
        path: Union[str, Path],
        lazy: Literal[True] = True,
    ) -> None: ...

    @overload
    def __init__(
        self: Atlas[
            Union[npt.NDArray[np.uint16], da.Array],
            Union[npt.NDArray[np.uint32], da.Array],
            Union[npt.NDArray[np.uint8], da.Array],
        ],
        path: Union[str, Path],
        lazy: bool = False,
    ) -> None: ...

    def __init__(self, path: Union[str, Path], lazy: bool = False):
        self.lazy = lazy
        self._template_pyramid_level = 0
        self._annotation_pyramid_level = 0
        self.fs = s3fs.S3FileSystem(anon=True)

        atlas_path = Path(path)
        self.root_dir = atlas_path.parents[3]
        self.metadata = read_json(atlas_path)
        structures_path = (
            self.root_dir
            / self.metadata["terminology"]["location"][1:]
            / V3_TERMINOLOGY_NAME
        )
        structures_list = load_structures_from_csv(structures_path)
        meshes_path = (
            self.root_dir
            / self.metadata["annotation_set"]["location"][1:]
            / V3_MESHES_DIRECTORY
        )

        template_location = self.metadata["annotation_set"]["template"][
            "location"
        ][1:]
        template_path = self.root_dir / template_location / V3_TEMPLATE_NAME

        multiscale = nz.from_ngff_zarr(template_path)
        self._template_pyramid_level = _determine_pyramid_level(
            multiscale, self.resolution
        )
        annotation_location = self.metadata["annotation_set"]["location"][1:]
        annotation_path = (
            self.root_dir / annotation_location / V3_ANNOTATION_NAME
        )
        multiscale = nz.from_ngff_zarr(annotation_path)
        self._annotation_pyramid_level = _determine_pyramid_level(
            multiscale, self.resolution
        )

        self._annotation_mapping = None
        self._annotation_masks_pyramid_level = None
        masks_path = self._annotation_masks_path
        if masks_path.exists():
            masks_multiscale = nz.from_ngff_zarr(masks_path)
            self._annotation_masks_pyramid_level = _determine_pyramid_level(
                masks_multiscale, self.resolution
            )
            root = zarr.open_group(str(masks_path), mode="r")
            raw_mapping = root[V3_ANNOTATION_MAP_NAME][:]

            self._annotation_mapping = {
                int(annotation_id): array_ind
                for array_ind, annotation_id in enumerate(raw_mapping)
            }

        # keep to generate tree and dataframe views when necessary
        self.structures_list = structures_list

        # Add entry for file paths:
        for struct in structures_list:
            struct["mesh_filename"] = meshes_path / f'{struct["id"]}'

        self.structures = StructuresDict(structures_list)

        # Instantiate SpaceConvention object describing the current atlas:
        self.space = AnatomicalSpace(
            origin=ATLAS_ORIENTATION,
            shape=self.shape,
            resolution=self.resolution,
        )

        try:
            additional_references = self.metadata.get(
                "additional_references", []
            )
            self.additional_references = AdditionalRefDict(
                references_list=additional_references,
                data_path=self.root_dir,
                resolution=self.resolution,
            )
        except KeyError:
            warnings.warn(
                "This atlas seems to be outdated as no "
                "additional_references list "
                "is found in metadata!"
            )

        self._annotation = None
        self._template = None
        self._hemispheres = None
        self._lookup = None

    @property
    def resolution(self):
        """Make resolution more accessible from class."""
        return tuple(self.metadata["resolution"])

    @property
    def _annotation_masks_path(self) -> Path:
        annotation_location = self.metadata["annotation_set"]["location"][1:]
        return self.root_dir / annotation_location / V3_ANNOTATION_MASKS_NAME

    @property
    def orientation(self):
        """Make orientation more accessible from class."""
        return ATLAS_ORIENTATION

    @property
    def shape(self):
        """Make shape more accessible from class."""
        return tuple(self.metadata["shape"])

    @property
    def shape_um(self):
        """Make shape more accessible from class."""
        return tuple([s * r for s, r in zip(self.shape, self.resolution)])

    @property
    def hierarchy(self):
        """Returns a Treelib.tree object with structures hierarchy."""
        return self.structures.tree

    @property
    def lookup_df(self):
        """Returns a dataframe with id, acronym and name for each structure."""
        if self._lookup is None:
            self._lookup = pd.DataFrame(
                dict(
                    acronym=[r["acronym"] for r in self.structures_list],
                    id=[r["id"] for r in self.structures_list],
                    name=[r["name"] for r in self.structures_list],
                )
            )
        return self._lookup

    @property
    def template(self) -> TemplateArray:
        """Return the template image data. Loads it if not already loaded."""
        if self._template is not None:
            return cast(TemplateArray, self._template)

        template_location = self.metadata["annotation_set"]["template"][
            "location"
        ][1:]

        template_path = self.root_dir / template_location / V3_TEMPLATE_NAME

        multiscale = nz.from_ngff_zarr(template_path)
        dataset_path = multiscale.metadata.datasets[
            self._template_pyramid_level
        ].path
        resolution_path = template_path / dataset_path

        if not (resolution_path / "c").exists():
            print("Downloading template...")
            remote_path = remote_url_s3.format(
                f"{template_location}/{V3_TEMPLATE_NAME}/{dataset_path}/"
            )
            self.fs.get(
                remote_path,
                resolution_path,
                recursive=True,
                callback=TqdmCallback(),
            )

        data = multiscale.images[self._template_pyramid_level].data
        self._template = data if self.lazy else data.compute()

        return cast(TemplateArray, self._template)

    @property
    def reference(self):
        """Return the template image data.

        Warning: this is a deprecated alias for template, and will be removed
        in future versions. Use atlas.template instead.
        """
        print(
            "Warning: atlas.reference is a deprecated alias for "
            "atlas.template, and will be removed in future versions."
        )
        return self.template

    @property
    def annotation(self) -> AnnotationArray:
        """Return the annotation image data. Loads it if not already loaded."""
        if self._annotation is not None:
            return cast(AnnotationArray, self._annotation)

        annotation_location = self.metadata["annotation_set"]["location"][1:]
        annotation_path = (
            self.root_dir / annotation_location / V3_ANNOTATION_NAME
        )

        multiscale = nz.from_ngff_zarr(annotation_path)
        dataset_path = multiscale.metadata.datasets[
            self._annotation_pyramid_level
        ].path
        resolution_path = annotation_path / dataset_path

        if not (resolution_path / "c").exists():
            print("Downloading annotations...")
            remote_path = remote_url_s3.format(
                f"{annotation_location}/{V3_ANNOTATION_NAME}/{dataset_path}/"
            )
            self.fs.get(
                remote_path,
                resolution_path,
                recursive=True,
                callback=TqdmCallback(),
            )

        data = multiscale.images[self._annotation_pyramid_level].data
        self._annotation = data if self.lazy else data.compute()

        return cast(AnnotationArray, self._annotation)

    @property
    def hemispheres(self) -> LabelArray:
        """
        Returns a stack with the hemisphere information. 1 - left, 2 - right.

        If a symmetric reference is used, the hemisphere information is
        generated by splitting the reference in half along the frontal axis.
        If the reference has an odd number of voxels along the frontal axis,
        the middle plane is assigned to the left hemisphere.
        """
        if self._hemispheres is not None:
            return cast(LabelArray, self._hemispheres)

        # If reference is symmetric generate hemispheres block:
        if self.metadata["symmetric"]:
            shape = tuple(self.metadata["shape"])
            front_ax_idx = self.space.axes_order.index("frontal")
            split = round(shape[front_ax_idx] / 2)

            if self.lazy:
                right_shape = list(shape)
                right_shape[front_ax_idx] = split
                left_shape = list(shape)
                left_shape[front_ax_idx] -= split
                self._hemispheres = da.concatenate(
                    [
                        da.full(
                            right_shape,
                            self.right_hemisphere_value,
                            dtype=np.uint8,
                        ),
                        da.full(
                            left_shape,
                            self.left_hemisphere_value,
                            dtype=np.uint8,
                        ),
                    ],
                    axis=front_ax_idx,
                )
            else:
                stack = np.full(
                    shape, self.right_hemisphere_value, dtype=np.uint8
                )
                slices = [slice(None) for _ in range(3)]
                slices[front_ax_idx] = slice(split, None)
                stack[tuple(slices)] = self.left_hemisphere_value

                self._hemispheres = stack
        else:
            annotation_location = self.metadata["annotation_set"]["location"][
                1:
            ]
            hemispheres_path = (
                self.root_dir / annotation_location / V3_HEMISPHERES_NAME
            )

            multiscale = nz.from_ngff_zarr(hemispheres_path)
            dataset_path = multiscale.metadata.datasets[
                self._annotation_pyramid_level
            ].path
            resolution_path = hemispheres_path / dataset_path

            if not (resolution_path / "c").exists():
                print("Downloading hemispheres...")
                remote_path = remote_url_s3.format(
                    f"{annotation_location}/{V3_HEMISPHERES_NAME}/{dataset_path}/"
                )
                self.fs.get(
                    remote_path,
                    resolution_path,
                    recursive=True,
                    callback=TqdmCallback(),
                )

            data = multiscale.images[self._annotation_pyramid_level].data
            self._hemispheres = data if self.lazy else data.compute()

        return cast(LabelArray, self._hemispheres)

    def hemisphere_from_coords(
        self,
        coords: Union[Tuple, List, np.ndarray],
        microns: bool = False,
        as_string: bool = False,
    ) -> Union[int, str]:
        """Get the hemisphere from a coordinate triplet.

        Parameters
        ----------
        coords : tuple or list or numpy array
            Triplet of coordinates. Default in voxels, can be microns if
            microns=True
        microns : bool
            If true, coordinates are interpreted in microns.
        as_string : bool
            If true, returns "left" or "right".


        Returns
        -------
        int or string
            Hemisphere label.

        """
        hem = self.hemispheres[self._idx_from_coords(coords, microns)]
        if self.lazy:
            hem = int(hem.compute())
        if as_string:
            hem = ["left", "right"][hem - 1]
        return hem

    def structure_from_coords(
        self,
        coords,
        microns=False,
        as_acronym=False,
        hierarchy_lev=None,
        key_error_string="Outside atlas",
    ):
        """Get the structure from a coordinate triplet.

        Parameters
        ----------
        coords : tuple or list or numpy array
            Triplet of coordinates.
        microns : bool
            If true, coordinates are interpreted in microns.
        as_acronym : bool
            If true, the region acronym is returned.
            If outside atlas (structure gives key error),
            return "Outside atlas"
        hierarchy_lev : int or None
            If specified, return parent node at thi hierarchy level.

        Returns
        -------
        int or string
            Structure containing the coordinates.
        """
        rid = self.annotation[self._idx_from_coords(coords, microns)]
        if self.lazy:
            rid = int(rid.compute())

        # If we want to cut the result at some high level of the hierarchy:
        if hierarchy_lev is not None:
            rid = self.structures[rid]["structure_id_path"][hierarchy_lev]

        if as_acronym:
            try:
                d = self.structures[rid]
                return d["acronym"]
            except KeyError:
                return key_error_string
        else:
            return rid

    # Meshes-related methods:
    def _get_from_structure(self, structure, key):
        """Provide internal interface to the structure dict. It supports
        querying with a single structure id or a list of ids.

        Parameters
        ----------
        structure : int or str or list
            Valid id or acronym, or list if ids or acronyms.
        key : str
            Key for the Structure dictionary (eg "name" or "rgb_triplet").

        Returns
        -------
        value or list of values
            If structure is a list, returns list.

        """
        if isinstance(structure, list) or isinstance(structure, tuple):
            return [self._get_from_structure(s, key) for s in structure]
        else:
            return self.structures[structure][key]

    def mesh_from_structure(self, structure):
        """
        Retrieve the mesh associated with a given structure.

        Parameters
        ----------
        structure : int or str or list of int/str
            The ID or acronym of the structure for which to retrieve the mesh.
            If a list of IDs/acronyms is passed, a list of meshes will
            be returned.

        Returns
        -------
        meshio.Mesh or list of meshio.Mesh
            The mesh data (e.g., a Mesh object) associated with the
            structure(s).
        """
        return self._get_from_structure(structure, "mesh")

    def meshfile_from_structure(self, structure):
        """
        Retrieve the path to the mesh file associated with a given structure.

        Parameters
        ----------
        structure : int or str
            The ID or acronym of the structure for which to retrieve the mesh
            file path. If a list of IDs/acronyms is passed, a list of paths
            will be returned.

        Returns
        -------
        Path or list of Path
            The path(s) to the mesh file(s) for the structure(s).
        """
        return self._get_from_structure(structure, "mesh_filename")

    def root_mesh(self):
        """
        Retrieve the mesh for the root structure.

        Returns
        -------
            The mesh data for the root structure.
        """
        return self.mesh_from_structure("root")

    def root_meshfile(self):
        """
        Retrieve the path to the mesh file for the root structure.

        Returns
        -------
            str: The path to the mesh file for the root structure.
        """
        return self.meshfile_from_structure("root")

    def _idx_from_coords(self, coords, microns):
        # If microns are passed, convert:
        if microns:
            coords = [c / res for c, res in zip(coords, self.resolution)]

        return tuple([int(c) for c in coords])

    def get_structure_ancestors(self, structure: Union[str, int]) -> List[str]:
        """Return a list of acronyms for all ancestors of a given structure.

        Parameters
        ----------
        structure : str or int
            Structure id or acronym

        Returns
        -------
        list
            List of descendants acronyms

        """
        ancestors_id = self._get_from_structure(
            structure, "structure_id_path"
        )[:-1]

        return self._get_from_structure(ancestors_id, "acronym")

    def get_structure_descendants(
        self, structure: Union[str, int]
    ) -> List[str]:
        """Return a list of acronyms for all descendants of a given structure.

        Parameters
        ----------
        structure : str or int
            Structure id or acronym

        Returns
        -------
        list
            List of descendants acronyms

        """
        structure = self._get_from_structure(structure, "acronym")

        # For all structures check if given structure is ancestor
        descendants = []
        for struc in self.structures.keys():
            if structure in self.get_structure_ancestors(struc):
                descendants.append(self._get_from_structure(struc, "acronym"))

        return descendants

    def get_structures_at_hierarchy_level(
        self, structure, hierarchy_level=None, as_acronym=False
    ):
        """
        Get structures at a specific hierarchy level within the subgraph
        of nodes connected to the given structure.

        For a given brain structure, this method finds all leaf nodes
        (terminal structures with no children) in its subtree, then extracts
        the structures at the specified hierarchy level from their paths.

        Parameters
        ----------
        structure : str or int
            Structure ID or acronym to query.
        hierarchy_level : int or None, optional
            The hierarchy level to extract (0-indexed, where 0 is root).
            If None, returns all structures in the paths to all leaves
            in anatomical order (breadth-first traversal).
        as_acronym : bool, optional
            If True, return acronyms instead of IDs. Default is False.

        Returns
        -------
        list
            List of structure IDs (if as_acronym=False) or acronyms
            (if as_acronym=True) at the specified hierarchy level.

        Raises
        ------
        ValueError
            If hierarchy_level is not an integer or None.
            If the structure has no descendants at the specified level.

        Examples
        --------
        >>> atlas = BrainGlobeAtlas("allen_mouse_25um")
        >>> # Get all level-3 structures under cortex
        >>> ids = atlas.get_structures_at_hierarchy_level("CTX", 3)
        >>> # Get as acronyms instead
        >>> acronyms = atlas.get_structures_at_hierarchy_level(
        ...     "CTX", 3, as_acronym=True
        ... )
        """
        # Type validation
        if not (
            hierarchy_level is None
            or (
                isinstance(hierarchy_level, int)
                and not isinstance(hierarchy_level, bool)
            )
        ):
            raise ValueError(
                f"hierarchy_level must be an int or None, "
                f"got {type(hierarchy_level).__name__}"
            )

        # Validate non-negative hierarchy level
        if hierarchy_level is not None and hierarchy_level < 0:
            raise ValueError("hierarchy_level must be non-negative")

        # Validate structure exists
        try:
            input_id = self.structures[structure]["id"]
        except KeyError:
            raise KeyError(f"Structure '{structure}' not found in atlas")

        if hierarchy_level is None:
            # Return all ancestors + descendants in anatomical (BFS) order

            # Start with ancestors (already ordered root → input_id)
            ancestor_path = self.structures[input_id]["structure_id_path"]
            result = list(ancestor_path)
            seen = set(ancestor_path)

            # BFS through descendants
            queue = deque([input_id])
            while queue:
                current_id = queue.popleft()
                children = self.structures.tree.children(current_id)
                for child in children:
                    child_id = child.identifier
                    if child_id not in seen:
                        result.append(child_id)
                        seen.add(child_id)
                        queue.append(child_id)
        else:
            # Get structures at specific hierarchy level
            # Get all leaf nodes (terminal descendants) of this structure
            input_id_leaves = self.structures.tree.leaves(input_id)

            # If input_id is itself a leaf, use it as its own leaf
            if not input_id_leaves:
                input_id_leaves = [self.structures.tree.get_node(input_id)]

            try:
                result = list(
                    set(
                        self.structures[leaf.identifier]["structure_id_path"][
                            hierarchy_level
                        ]
                        for leaf in input_id_leaves
                    )
                )
            except IndexError:
                raise ValueError(
                    f'Structure {self.structures[structure]["acronym"]} '
                    f"has no descendants at hierarchy level {hierarchy_level}"
                )

        if as_acronym:
            return [self.structures[sid]["acronym"] for sid in result]
        return result

    def get_structure_mask(self, structure) -> LabelArray:
        """Return binary uint8 mask for the given structure.

        Reads directly from the pre-built 4D annotation masks array.

        Parameters
        ----------
        structure : str or int
            Structure acronym or id.

        Returns
        -------
        np.ndarray
            Binary uint8 array; 1 where the structure (or a descendant)
            has a voxel, 0 elsewhere.

        Raises
        ------
        FileNotFoundError
            If this atlas does not have a 4D mask array on disk.
        KeyError
            If the structure is not present in the annotation mapping.
        """
        if self._annotation_mapping is None:
            raise FileNotFoundError(
                "This atlas does not have a 4D mask array. "
                "Re-download the atlas to get the latest version."
            )
        structure_id = self.structures[structure]["id"]
        if structure_id not in self._annotation_mapping:
            raise KeyError(
                f"Structure {structure} (id={structure_id}) not found in "
                "annotation mapping."
            )

        index = self._annotation_mapping[structure_id]
        masks_path = self._annotation_masks_path
        multiscale = nz.from_ngff_zarr(masks_path)
        dataset_path = multiscale.metadata.datasets[
            self._annotation_masks_pyramid_level
        ].path
        # Check if the mask is cached locally. The presence of the
        # directory for the specific index indicates that the mask is cached.
        local_path = masks_path / dataset_path / "c" / str(index)
        if not local_path.exists():
            annotation_location = self.metadata["annotation_set"]["location"][
                1:
            ]
            remote_path = remote_url_s3.format(
                f"{annotation_location}/{V3_ANNOTATION_MASKS_NAME}"
                f"/{dataset_path}/c/{index}/"
            )
            try:
                self.fs.get(
                    remote_path,
                    local_path,
                    recursive=True,
                    callback=TqdmCallback(),
                )
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"Mask for structure {structure} (id={structure_id}) "
                    f"not found at {remote_path}"
                ) from e

        data = multiscale.images[self._annotation_masks_pyramid_level].data[
            index
        ]
        return cast(LabelArray, data if self.lazy else data.compute())


class AdditionalRefDict(UserDict):
    """Class implementing the lazy loading of secondary references
    if the dictionary is queried for it.
    """

    def __init__(
        self,
        references_list: List[Dict[str, str]],
        data_path,
        resolution: Tuple[float, float, float],
        *args,
        **kwargs,
    ):
        self.data_path = data_path
        self.references_names = [ref["name"] for ref in references_list]
        self.references_dict = {ref["name"]: ref for ref in references_list}
        self.resolution = resolution

        super().__init__(*args, **kwargs)

        for ref_name in self.references_names:
            self.data[ref_name] = None

    def __getitem__(self, key):
        """Retrieve an item from the dictionary using the reference name
        as key.

        If the reference image data for `ref_name` has not been loaded yet,
        it will be read from the disk and cached. If `ref_name` is not
        one of the predefined additional references, a warning is issued
        and None is returned.

        Parameters
        ----------
        key : str
            The name of the reference image to retrieve (e.g., "aba").

        Returns
        -------
        np.ndarray or None
            The image data associated with the reference name, or None if the
            reference name is not found in the list of available references.

        Raises
        ------
            KeyError: If the ref_name is not found.
        """
        if key not in self.references_names:
            warnings.warn(
                f"No reference named {key} "
                f"(available: {self.references_names})"
            )
            return None

        if self.data[key] is None:
            additional_ref_data = self.references_dict.get(key, key)

            additional_ref_location = additional_ref_data["location"][1:]
            local_path: Path = (
                self.data_path / additional_ref_location / V3_TEMPLATE_NAME
            )

            multiscale = nz.from_ngff_zarr(local_path)
            pyramid_level = _determine_pyramid_level(
                multiscale, self.resolution
            )

            dataset_path = multiscale.metadata.datasets[pyramid_level].path
            resolution_path = local_path / dataset_path

            if not (resolution_path / "c").exists():
                print("Downloading template...")
                remote_path = remote_url_s3.format(
                    f"{additional_ref_location}/{V3_TEMPLATE_NAME}/{dataset_path}/"
                )
                fs = s3fs.S3FileSystem(anon=True)
                fs.get(
                    remote_path,
                    resolution_path,
                    recursive=True,
                    callback=TqdmCallback(),
                )
            self.data[key] = multiscale.images[pyramid_level].data.compute()

        return self.data[key]
