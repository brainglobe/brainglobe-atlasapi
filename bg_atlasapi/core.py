from pathlib import Path
import pandas as pd
import numpy as np
from bg_space import SpaceConvention

from bg_atlasapi.utils import read_json, read_tiff
from bg_atlasapi.structure_class import StructuresDict
from bg_atlasapi.structure_tree_util import get_structures_tree
from bg_atlasapi.descriptors import (
    METADATA_FILENAME,
    STRUCTURES_FILENAME,
    REFERENCE_FILENAME,
    ANNOTATION_FILENAME,
    HEMISPHERES_FILENAME,
    MESHES_DIRNAME,
)


class Atlas:
    """Base class to handle atlases in BrainGlobe.

    Parameters
    ----------
    path : str or Path object
        path to folder containing data info.
    """

    def __init__(self, path):
        self.root_dir = Path(path)
        self.metadata = read_json(self.root_dir / METADATA_FILENAME)

        # Load structures list:
        structures_list = read_json(self.root_dir / STRUCTURES_FILENAME)
        self.structures_list = structures_list  # keep to generate tree and dataframe views when necessary

        # Add entry for file paths:
        for struct in structures_list:
            struct["mesh_filename"] = (
                self.root_dir / MESHES_DIRNAME / "{}.obj".format(struct["id"])
            )

        self.structures = StructuresDict(structures_list)

        # Instantiate SpaceConvention object describing the current atlas:
        self._space = SpaceConvention(
            origin=self.metadata["orientation"],
            shape=self.metadata["shape"],
            resolution=self.metadata["resolution"],
        )

        self._reference = None
        self._annotation = None
        self._hemispheres = None
        self._hierarchy = None
        self._lookup = None

    @property
    def resolution(self):
        """Make resolution more accessible from class.
        """
        return self.metadata["resolution"]

    @property
    def hierarchy(self):
        """Returns a Treelib.tree object with structures hierarchy.
        """
        if self._hierarchy is None:
            self._hierarchy = get_structures_tree(self.structures_list)
        return self._hierarchy

    @property
    def lookup_df(self):
        """Returns a dataframe with id, acronym and name for each structure.
        """
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
    def reference(self):
        if self._reference is None:
            self._reference = read_tiff(self.root_dir / REFERENCE_FILENAME)
        return self._reference

    @property
    def annotation(self):
        if self._annotation is None:
            self._annotation = read_tiff(self.root_dir / ANNOTATION_FILENAME)
        return self._annotation

    @property
    def hemispheres(self):
        if self._hemispheres is None:
            # If reference is symmetric generate hemispheres block:
            if self.metadata["symmetric"]:
                # initialize empty stack:
                stack = np.ones(self.metadata["shape"], dtype=np.uint8)

                # Use bgspace description to fill out with hemisphere values:
                front_ax_idx = self._space.axes_order.index("frontal")

                # Fill out with 2s the right hemisphere:
                slices = [slice(None) for _ in range(3)]
                slices[front_ax_idx] = slice(
                    stack.shape[front_ax_idx] // 2 + 1, None
                )
                stack[tuple(slices)] = 2

                self._hemispheres = stack
            else:
                self._hemispheres = read_tiff(
                    self.root_dir / HEMISPHERES_FILENAME
                )
        return self._hemispheres

    def hemisphere_from_coords(self, coords, microns=False, as_string=False):
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
        if as_string:
            hem = ["left", "right"][hem - 1]
        return hem

    def structure_from_coords(
        self, coords, microns=False, as_acronym=False, hierarchy_lev=None
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
        hierarchy_lev : int or None
            If specified, return parent node at thi hierarchy level.

        Returns
        -------
        int or string
            Structure containing the coordinates.
        """

        rid = self.annotation[self._idx_from_coords(coords, microns)]

        # If we want to cut the result at some high level of the hierarchy:
        if hierarchy_lev is not None:
            rid = self.structures[rid]["structure_id_path"][hierarchy_lev]

        if as_acronym:
            d = self.structures[rid]
            return d["acronym"]
        else:
            return rid

    # Meshes-related methods:
    def _get_from_structure(self, structure, key):
        """Internal interface to the structure dict. It support querying with a
        single structure id or a list of ids.

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
        return self._get_from_structure(structure, "mesh")

    def meshfile_from_structure(self, structure):
        return self._get_from_structure(structure, "mesh_filename")

    def root_mesh(self):
        return self.mesh_from_structure("root")

    def root_meshfile(self):
        return self.meshfile_from_structure("root")

    def _idx_from_coords(self, coords, microns):
        # If microns are passed, convert:
        if microns:
            coords = [c / res for c, res in zip(coords, self.resolution)]

        return tuple([int(c) for c in coords])

    # ------- BrainRender methods, might be useful to implement here ------- #

    # def get_region_unilateral(self):
    #     pass

    # def mirror_point_across_hemispheres(self):
    #     pass

    # # functions to create oriented planes that can be used to slice actors etc
    # def get_plane_at_point(self, pos, norm, sx, sy,
    #                        color='lightgray', alpha=.25,
    #                        **kwargs):
    #     """
    #         Returns a plane going through a point at pos, oriented
    #         orthogonally to the vector norm and of width and height
    #         sx, sy.
    #
    #         :param pos: 3-tuple or list with x,y,z, coords of point the plane goes through
    #         :param sx, sy: int, width and height of the plane
    #         :param norm: 3-tuple or list with 3d vector the plane is orthogonal to
    #         :param color, alpha: plane color and transparency
    #     """
    #     plane = Plane(pos=pos, normal=norm,
    #                   sx=sx, sy=sy, c=color, alpha=alpha)
    #     return plane
    #
    # def get_sagittal_plane(self, pos=None, **kwargs):
    #     """
    #         Creates a Plane actor centered at the midpoint of root (or a user given locatin)
    #         and oriented along the sagittal axis
    #
    #         :param pos: if not None, passe a list of 3 xyz defining the position of the
    #                         point the plane goes through.
    #     """
    #     if pos is None:
    #         pos = self._root_midpoint
    #         if pos[0] is None:
    #             raise ValueError(
    #                 f"The atlases _root_midpoint attribute is not specified")
    #     elif not isinstance(pos, (list, tuple)) or not len(pos) == 3:
    #         raise ValueError(f"Invalid pos argument: {pos}")
    #
    #     norm = self._planes_norms['sagittal']
    #     sx = float(np.diff(self._root_bounds[0]))
    #     sy = float(np.diff(self._root_bounds[1]))
    #
    #     sx += sx / 5
    #     sy += sy / 5
    #     sag_plane = self.get_plane_at_point(pos, norm, sx, sy, **kwargs)
    #
    #     return sag_plane
    #
    # def get_horizontal_plane(self, pos=None, **kwargs):
    #     """
    #         Creates a Plane actor centered at the midpoint of root (or a user given locatin)
    #         and oriented along the horizontal axis
    #
    #         :param pos: if not None, passe a list of 3 xyz defining the position of the
    #                         point the plane goes through.
    #     """
    #     if pos is None:
    #         pos = self._root_midpoint
    #         if pos[0] is None:
    #             raise ValueError(
    #                 f"The atlases _root_midpoint attribute is not specified")
    #     elif not isinstance(pos, (list, tuple)) or not len(pos) == 3:
    #         raise ValueError(f"Invalid pos argument: {pos}")
    #
    #     norm = self._planes_norms['horizontal']
    #     sx = float(np.diff(self._root_bounds[2]))
    #     sy = float(np.diff(self._root_bounds[0]))
    #
    #     sx += sx / 5
    #     sy += sy / 5
    #     hor_plane = self.get_plane_at_point(pos, norm, sx, sy, **kwargs)
    #
    #     return hor_plane
    #
    # def get_coronal_plane(self, pos=None, **kwargs):
    #     """
    #         Creates a Plane actor centered at the midpoint of root (or a user given locatin)
    #         and oriented along the coronal axis
    #
    #         :param pos: if not None, passe a list of 3 xyz defining the position of the
    #                         point the plane goes through.
    #     """
    #     if pos is None:
    #         pos = self._root_midpoint
    #         if pos[0] is None:
    #             raise ValueError(
    #                 f"The atlases _root_midpoint attribute is not specified")
    #     elif not isinstance(pos, (list, tuple)) or not len(pos) == 3:
    #         raise ValueError(f"Invalid pos argument: {pos}")
    #
    #     norm = self._planes_norms['coronal']
    #     sx = float(np.diff(self._root_bounds[2]))
    #     sy = float(np.diff(self._root_bounds[1]))
    #
    #     sx += sx / 5
    #     sy += sy / 5
    #     cor_plane = self.get_plane_at_point(pos, norm, sx, sy, **kwargs)
    #
    #     return cor_plane
