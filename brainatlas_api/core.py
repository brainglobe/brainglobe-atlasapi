import numpy as np
from pathlib import Path

from brainatlas_api.utils import read_json, read_tiff, make_hemispheres_stack
from brainatlas_api.structures.structure_tree import StructureTree
from brainatlas_api.obj_utils import read_obj
from brainatlas_api.descriptors import (
    METADATA_FILENAME,
    STRUCTURES_FILENAME,
    REFERENCE_FILENAME,
    ANNOTATION_FILENAME,
    HEMISPHERES_FILENAME,
    MESHES_DIRNAME,
)


class MeshDictionary(dict):
    """ Class to cache the loading of .obj mesh files
        Parameters
        ----------
        mesh_path : str or Path object
            path to folder containing all meshes .obj files
        """

    def __init__(self, mesh_path):
        self.root = Path(mesh_path)

        # Create dictionary of loadable files:
        self.files_dict = {
            int(f.resolve().stem): f for f in self.root.glob("*.obj")
        }

    def __getitem__(self, item):
        """ Load mesh if it has not been read before, and add it to self dict.
        Parameters
        ----------
        item : int
            id of structure to be loaded

        Returns
        -------
        tuple :
            Tuple with mesh description

        """
        if item not in self.keys():
            value = read_obj(self.files_dict[item])
            super().__setitem__(item, value)

        return super().__getitem__(item)


class Atlas:
    """ Base class to handle atlases in brainglobe.

        Parameters
        ----------
        path : str or Path object
            path to folder containing data info
        """

    def __init__(self, path):
        self.root_dir = Path(path)
        self.metadata = read_json(self.root_dir / METADATA_FILENAME)

        # Class for structures:
        structures_list = read_json(self.root_dir / STRUCTURES_FILENAME)
        self.structures = StructureTree(structures_list)

        # Cached loading of meshes:
        self.region_meshes_dict = MeshDictionary(
            self.root_dir / MESHES_DIRNAME
        )

        for attr in ["name", "shape", "resolution", "symmetric"]:
            self.__setattr__(attr, self.metadata[attr])

        self._reference = None
        self._annotated = None
        self._hemispheres = None

        # Dictionaries to map acronyms to ids...:
        self.acronym_to_id_map = self.structures.get_id_acronym_map()
        # ...and viceversa:
        self.id_to_acronym_map = {
            v: k for k, v in self.acronym_to_id_map.items()
        }

        # Store a list of all acronyms and names
        self.structures_acronyms = [
            n["acronym"] for n in self.structures.nodes()
        ]
        self.structures_names = [n["name"] for n in self.structures.nodes()]
        self.structures_ids = [n["name"] for n in self.structures.nodes()]

    @property
    def reference(self):
        if self._reference is None:
            try:
                self._reference = read_tiff(self.root_dir / REFERENCE_FILENAME)
            except FileNotFoundError:  # avoid general excepts
                raise FileNotFoundError(
                    f"Failed to load reference.tiff from {self.root_dir / REFERENCE_FILENAME}"
                )
        return self._reference

    @property
    def annotated(self):
        if self._annotated is None:
            try:
                self._annotated = read_tiff(
                    self.root_dir / ANNOTATION_FILENAME
                )
            except FileNotFoundError:  # avoid general excepts
                raise FileNotFoundError(
                    f"Failed to load annotated.tiff from {self.root_dir / ANNOTATION_FILENAME}"
                )
        return self._annotated

    @property
    def hemispheres(self):
        if self._hemispheres is None:
            # If reference is symmetric generate hemispheres block:
            if self.metadata["symmetric"]:
                self._hemispheres = make_hemispheres_stack(self.shape)
            else:
                self._hemispheres = read_tiff(
                    self.root_dir / HEMISPHERES_FILENAME
                )
        return self._hemispheres

    def get_hemisphere_from_coords(self, coords):
        return self.hemispheres[self.idx_from_coords(coords)]

    def get_region_id_from_coords(self, coords):
        return self.annotated[self.idx_from_coords(coords)]

    def get_region_name_from_coords(self, coords):
        region_id = self.get_region_id_from_coords(coords)

        return self.id_to_acronym_map[region_id]

    def get_region_color_from_acronym(self, region_acronym):
        region_id = self.acronym_to_id_map[region_acronym]
        return self.structures.get_structures_by_id([region_id])[0][
            "rgb_triplet"
        ]

    # Meshes-related methods:
    def get_mesh_from_id(self, region_id):
        return self.region_meshes_dict[region_id]

    def get_mesh_from_name(self, region_name):
        region_id = self.acronym_to_id_map[region_name]
        return self.get_mesh_from_id(region_id)

    def get_brain_mesh(self):
        return self.get_mesh_from_name("root")

    def get_mesh_file_from_acronym(self, region_acronym):
        region_id = self.acronym_to_id_map[region_acronym]

        try:
            return self.region_meshes_dict.files_dict[region_id]
        except Exception as e:
            raise ValueError(
                f"Failed to retrieve mesh file for {region_acronym}: {e}"
            )

    def get_region_CenterOfMass(self):
        pass

    @staticmethod
    def idx_from_coords(coords):
        return tuple([int(c) for c in coords])

    # ------- BrainRender methods, might be useful to implement here ------- #
    def _check_point_in_region(self, point, region_actor):
        pass

    def get_region_unilateral(self):
        pass

    def mirror_point_across_hemispheres(self):
        pass

    def get_colors_from_coordinates(self, coords):
        region_id = self.get_region_id_from_coords(coords)
        region = self.structures.get_structures_by_id([region_id])[0]
        return region["rgb_triplet"]

    def get_structure_ancestors(
        self, regions, ancestors=True, descendants=False
    ):
        pass

    def get_structure_descendants(self, regions):
        pass

    def get_structure_parent(self, acronyms):
        pass

    def print_structures(self):
        """
        Prints the name of every structure in the structure tree to the console.
        """
        acronyms, names = self.structures_acronyms, self.structures_names
        sort_idx = np.argsort(acronyms)
        acronyms, names = (
            np.array(acronyms)[sort_idx],
            np.array(names)[sort_idx],
        )
        [print("({}) - {}".format(a, n)) for a, n in zip(acronyms, names)]

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
