from pathlib import Path
from typing import Union

import zarr
from brainglobe_utils.IO.yaml import open_yaml
from pooch import create
from typing_extensions import deprecated

from brainglobe_atlasapi import BrainGlobeAtlas, descriptors
from brainglobe_atlasapi.descriptors import STRUCTURES_FILENAME
from brainglobe_atlasapi.utils import (
    check_gin_status,
    check_internet_connection,
)


class BrainGlobeAtlasV2(BrainGlobeAtlas):
    atlas_name = None
    _remote_url_base = descriptors.remote_url_base_v2

    def __init__(
        self,
        atlas_name,
        **kwargs,
    ):
        self._local_full_name = None
        self._template = None
        self.pooch = None
        super().__init__(atlas_name, **kwargs)

        if self.pooch is None:
            self.pooch = create(
                path=self.root_dir,
                base_url=self._remote_url_base.format(""),
                retry_if_failed=5,
            )

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """
        if self._local_full_name is None:
            # Create the v2 atlases directory if it doesn't exist
            (self.brainglobe_dir / "atlases").mkdir(
                parents=True, exist_ok=True
            )

            pattern = f"atlases/{self.atlas_name}_v*.yaml"
            self._local_full_name = self._get_local_full_name(pattern)
            # Prepend the atlases directory to the local full name
            if self._local_full_name is not None:
                self._local_full_name = "atlases/" + self._local_full_name

        return self._local_full_name

    @property
    def local_version(self):
        """If atlas is local, return actual version of the downloaded files;
        Else, return none.
        """
        version_str = self.metadata["version"]

        if version_str is None:
            return None

        return tuple(int(version) for version in version_str.split("."))

    @property
    def template(self):
        if self._template is None:
            template_path = (
                self.root_dir
                / "templates"
                / self.metadata["template_names"][0]
                / f"{int(self.resolution[0])}um.zarr.zip"
            )
            if not template_path.exists():
                print("Downloading template image...")
                remote_name = (
                    f"templates/{self.metadata['template_names'][0]}/"
                    f"{int(self.resolution[0])}um.zarr.zip"
                )
                self.pooch.registry[remote_name] = None
                self.pooch.fetch(
                    remote_name,
                    progressbar=True,
                )

            store = zarr.storage.ZipStore(template_path, mode="r")
            template_array = zarr.open_array(store, mode="r", zarr_format=3)
            self._template = template_array[...]
            store.close()

        return self._template

    @property
    @deprecated("Use the 'template' property instead.")
    def reference(self):
        return self.template

    @property
    def annotation(self):
        if self._annotation is None:
            annotation_path = (
                self.root_dir
                / "annotations"
                / self.metadata["annotation_names"][0]
                / f"{int(self.resolution[0])}um.zarr.zip"
            )
            if not annotation_path.exists():
                print("Downloading annotation image...")
                remote_name = (
                    f"annotations/{self.metadata['annotation_names'][0]}/"
                    f"{int(self.resolution[0])}um.zarr.zip"
                )
                self.pooch.registry[remote_name] = None
                self.pooch.fetch(
                    remote_name,
                    progressbar=True,
                )

            store = zarr.storage.ZipStore(annotation_path, mode="r")
            annotation_array = zarr.open_array(store, mode="r", zarr_format=3)
            self._annotation = annotation_array[...]
            store.close()

        return self._annotation

    @property
    def remote_url(self):
        if self.remote_version is not None:
            name = (
                f"{self.atlas_name}_v{self.remote_version[0]}_"
                f"{self.remote_version[1]}.yaml"
            )

            return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url."""
        check_internet_connection()
        check_gin_status()
        # Cache to avoid multiple requests
        remote_version = self.remote_version

        name = (
            f"{self.atlas_name}_v{remote_version[0]}_"
            f"{remote_version[1]}.yaml"
        )

        # Get path to folder where data will be saved
        destination_path = self.brainglobe_dir / "atlases" / name

        # Instantiate the pooch object
        self.pooch = create(
            path=self.brainglobe_dir / "atlases",
            base_url=self._remote_url_base.format(""),
            retry_if_failed=5,
        )

        # Try to download atlas data
        print("Downloading atlas metadata...")
        self.pooch.registry[name] = None
        self.pooch.fetch(name, progressbar=True)

        # Check if component directories exist, if not create them
        self.metadata = open_yaml(destination_path)
        annotations_dir = (
            destination_path.parent
            / "annotations"
            / self.metadata["annotation_names"][0]
        )
        template_dir = (
            destination_path.parent
            / "templates"
            / self.metadata["template_names"][0]
        )
        meshes_dir = (
            destination_path.parent / "meshes" / self.metadata["meshes"][0]
        )

        # Create the directories if they don't exist
        annotations_dir.mkdir(parents=True, exist_ok=True)
        template_dir.mkdir(parents=True, exist_ok=True)
        meshes_dir.mkdir(parents=True, exist_ok=True)

        structures_path = (
            destination_path.parent
            / "annotations"
            / annotations_dir
            / STRUCTURES_FILENAME
        )
        structures_csv_name = STRUCTURES_FILENAME.split(".")[0] + ".csv"
        if not structures_path.exists():
            structures_json = (
                f"annotations/{annotations_dir.name}/{STRUCTURES_FILENAME}"
            )
            self.pooch.registry[structures_json] = None
            self.pooch.fetch(structures_json, progressbar=True)

        if not structures_path.with_suffix(".csv").exists():
            structures_csv = (
                f"annotations/{annotations_dir.name}/{structures_csv_name}"
            )
            self.pooch.registry[structures_csv] = None
            self.pooch.fetch(structures_csv, progressbar=True)

    def _get_from_structure(self, structure, key):
        """
        Internal interface to the structure dict. It supports querying with a
        single structure id or a list of ids.

        Parameters
        ----------
        structure : int or str or list
            Valid id or acronym, or list of ids or acronyms.
        key : str
            Key for the Structure dictionary (eg "name" or "rgb_triplet").

        Returns
        -------
        value or list of values
            If structure is a list, returns list.
        """
        # Check if mesh is cached
        if key == "mesh":
            if isinstance(structure, list) or isinstance(structure, tuple):
                for s in structure:
                    self._check_mesh_cached(s)
            else:
                self._check_mesh_cached(structure)

        return super()._get_from_structure(structure, key)

    def _check_mesh_cached(self, structure: Union[str, int]):
        """Check if the mesh is cached in the local directory.
        Download from the remote if not cached.

        Parameters
        ----------
        structure : str or int
            Name of the mesh file.

        Returns
        -------
        bool
            True if the mesh is cached, False otherwise.
        """
        mesh_path = Path(self._get_from_structure(structure, "mesh_filename"))

        try:
            mesh_id = int(structure)
        except ValueError:
            mesh_id = int(self.structures.acronym_to_id_map[structure])

        # Check if the mesh is cached
        if not mesh_path.exists():
            # If not cached, download it
            structure_name = self.structures[mesh_id]["acronym"]
            print(f"Downloading mesh for {structure_name}...")
            file_name = f"meshes/{self.metadata['meshes'][0]}/{mesh_id}.obj"
            # Hacky way to not need a registry and avoid printing file hashes
            self.pooch.registry[file_name] = None
            self.pooch.fetch(file_name, progressbar=True)

        return
