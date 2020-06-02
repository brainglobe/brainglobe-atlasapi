import meshio as mio


class Structure(dict):
    def __getitem__(self, item):
        if item == "mesh" and super().__getitem__(item) is None:
            # TODO gracefully fail with warning if no mesh:
            try:
                super().__setitem__(item, mio.read(self["mesh_filename"]))
            except (TypeError, mio.ReadError):
                raise mio.ReadError(
                    "No valid mesh for region: {}".format(self.data["acronym"])
                )

        return super().__getitem__(item)

    def get_center_of_mass(self):
        pass

    def ancestors(self):
        pass

    def descendants(self):
        pass


class StructuresDict(dict):
    """ Class to handle dual indexing by either acronym or id.

        Parameters
        ----------
        mesh_path : str or Path object
            path to folder containing all meshes .obj files
        """

    def __init__(self, structures_list):
        super().__init__()

        # Acronym to id map:
        self.acronym_to_id_map = {
            r["acronym"]: r["id"] for r in structures_list
        }

        for struct in structures_list:
            sid = struct["id"]
            super().__setitem__(sid, Structure(**struct, mesh=None))

    def __getitem__(self, item):
        """ Core implementation of the class support for different indexing.

        Parameters
        ----------
        item :

        Returns
        -------

        """
        try:
            item = int(item)
        except ValueError:
            item = self.acronym_to_id_map[item]

        return super().__getitem__(int(item))
