"""Defines BrainGlobeAtlasAllen class for instantiating atlases from Allen."""

import json
import re
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import s3fs
import zarr
from typing_extensions import deprecated

from brainglobe_atlasapi import BrainGlobeAtlas, descriptors
from brainglobe_atlasapi.bg_atlas import _version_tuple_from_str
from brainglobe_atlasapi.utils import (
    check_internet_connection,
    read_json,
)


def _version_str_from_tuple(version_tuple):
    return ".".join(str(num) for num in version_tuple)


class BrainGlobeAtlasAllen(BrainGlobeAtlas):
    """Add remote atlas fetching and version comparison functionalities
    to the core Atlas class.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to be used.
    resolution : int or float
        Desired isotropic resolution in microns.
    version : str (optional)
        Desired version of the atlas. If None, the latest version will be used.
    brainglobe_dir : str or Path object
        Default folder for brainglobe downloads.
    interm_download_dir : str or Path object
        Folder to download the compressed file for extraction.
    check_latest : bool (optional)
        If true, check if we have the most recent atlas (default=True). Set
        this to False to avoid waiting for remote server response on atlas
        instantiation and to suppress warnings.
    print_authors : bool (optional)
        If true, disable default listing of the atlas reference.
    fn_update : Callable
        Handler function to update during download. Takes completed and total
        bytes.

    """

    atlas_name = None
    _remote_url_base = descriptors.remote_url_base_allen

    def __init__(
        self,
        atlas_name,
        resolution: Union[int, float],
        version: Optional[str] = None,
        **kwargs,
    ):
        self._remote_version = None
        self._local_full_name = None
        self._template = None
        self._requested_version = version
        self._local_version = (
            _version_tuple_from_str(version) if version else None
        )
        # Hacky solution handling: we assume isotropic atlases for now
        self._resolution = [resolution, resolution, resolution]
        self._pyramid_level = None
        self._shape = None
        self.fs = s3fs.S3FileSystem(anon=True)

        super().__init__(atlas_name, **kwargs)

    @property
    def local_full_name(self):
        """If atlas is local, return actual version of the downloaded files;
        Else, return none.
        """
        if self._local_full_name is None:
            # Create the v2 atlases directory if it doesn't exist
            (self.brainglobe_dir / "atlases_allen/atlases").mkdir(
                parents=True, exist_ok=True
            )

            if self._requested_version is not None:
                pattern = (
                    f"atlases_allen/atlases/{self.atlas_name}/"
                    f"{self._requested_version}/manifest.json"
                )
            else:
                pattern = (
                    rf"atlases_allen/atlases/"
                    rf"{self.atlas_name}/\d+(?:_\d+)?/manifest.json"
                )

            # Search for local directories matching the pattern
            glob_pattern = (
                f"atlases_allen/atlases/{self.atlas_name}/*/manifest.json"
            )
            available_versions: List[tuple[int, ...]] = [
                _version_tuple_from_str(p.parent.name)
                for p in self.brainglobe_dir.glob(glob_pattern)
                if re.search(pattern, str(p))
            ]

            if len(available_versions) == 0:
                return None

            available_versions.sort(reverse=True)

            self._local_full_name = (
                f"atlases_allen/atlases/"
                f"{self.atlas_name}/"
                f"{_version_str_from_tuple(available_versions[0])}/"
                f"manifest.json"
            )

        return self._local_full_name

    @property
    def shape(self):
        """Make shape more accessible from class."""
        if self._shape is not None:
            return self._shape

        annotation_set_path = self.metadata["annotation_set"]["location"][1:]
        annotation_path = (
            self.brainglobe_dir / "atlases_allen" / annotation_set_path
        )
        annotation_zarr_path = (
            annotation_path / "annotations_compressed.ome.zarr"
        )
        if annotation_zarr_path.exists():
            group = zarr.open(annotation_zarr_path, mode="r", zarr_format=3)
            ome_metadata = group.attrs.get("ome", None)
            pyramid_transforms = ome_metadata["multiscales"][0]["datasets"]

            for pyramid_level in pyramid_transforms:
                transforms = pyramid_level["coordinateTransformations"]
                for transform in transforms:
                    if transform["type"] == "scale":
                        # Assume scales are mm and resolution in um
                        scale_factors = np.array(transform["scale"]) * 1e3
                        if np.allclose(scale_factors, self._resolution):
                            self._pyramid_level = pyramid_level["path"]
                            self.metadata["shape"] = group[
                                self._pyramid_level
                            ].shape
                            self._shape = self.metadata["shape"]
                            self.metadata["resolution"] = self._resolution
                            break

                if self._pyramid_level is not None:
                    break

            if self._pyramid_level is None:
                raise ValueError(
                    f"Requested resolution {self._resolution} not found in "
                    f"available pyramid levels."
                )
        return self._shape

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
    def remote_version(self):
        """Reads remote version from s3 bucket.

        Largest numerical version assumed to be latest.
        If we are offline, return None.
        """
        if self._remote_version is None:
            bucket_path = descriptors.remote_bucket_allen.format(
                f"atlases/{self.atlas_name}"
            )
            if self._requested_version is None:
                versions_paths = self.fs.ls(bucket_path)
                versions_available = [
                    path_str.split("/")[-1] for path_str in versions_paths
                ]

                versions_available.sort(reverse=True)

                # Take the first (largest) version
                self._remote_version = _version_tuple_from_str(
                    versions_available[0]
                )
            else:
                requested_path = f"{bucket_path}/{self._requested_version}"
                if self.fs.exists(requested_path):
                    self._remote_version = _version_tuple_from_str(
                        self._requested_version
                    )
                else:
                    self._remote_version = None

        return self._remote_version

    @property
    def template(self):
        """Return the template image data. Loads it if not already loaded."""
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
        """Return the reference image data. Loads it if not already loaded."""
        return self.template

    @property
    def annotation(self):
        """
        Return the annotation image data.
        Loads it if not already loaded.
        """
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
        """Format complete url for download."""
        if self.remote_version is not None:
            name = (
                f"{self.atlas_name}_v{self.remote_version[0]}_"
                f"{self.remote_version[1]}.yaml"
            )

            return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url."""
        check_internet_connection()
        # Cache to avoid multiple requests
        remote_version = self.remote_version
        remote_version_str = _version_str_from_tuple(remote_version)

        working_dir = self.brainglobe_dir / "atlases_allen"

        name = f"atlases/{self.atlas_name}/{remote_version_str}/manifest.json"

        # Get path to folder where data will be saved
        local_path = working_dir / name
        remote_path = descriptors.remote_bucket_allen.format(name)

        self.fs.get(remote_path, local_path)

        # Read metadata
        self.metadata = read_json(local_path)

        # Read terminology and convert to structures.json
        terminology = self.metadata["terminology"]
        # Remove leading slash
        terminology_path = terminology["location"][1:]
        local_terminology_path = working_dir / terminology_path

        if not local_terminology_path.parent.exists():
            remote_terminology_path = descriptors.remote_bucket_allen.format(
                terminology_path
            )
            self.fs.get(
                remote_terminology_path, local_terminology_path, recursive=True
            )
            self._convert_terminology_to_structures(
                local_terminology_path / "terminology.csv"
            )

        # Check coordinate space exists
        coord_space = self.metadata["coordinate_space"]
        coord_space_path = coord_space["location"][1:]
        local_coord_space_path = working_dir / coord_space_path

        if not local_coord_space_path.exists():
            remote_coord_space_path = descriptors.remote_bucket_allen.format(
                coord_space_path
            )
            self.fs.get(
                remote_coord_space_path, local_coord_space_path, recursive=True
            )

        # Check annotation set directory exists
        annotation_set = self.metadata["annotation_set"]
        annotation_set_path = annotation_set["location"][1:]
        local_annotation_set_path = working_dir / annotation_set_path

        if not local_annotation_set_path.exists():
            # Only download metadata
            root_metadata_path = annotation_set_path + "/*.json"
            remote_root_metadata_path = descriptors.remote_bucket_allen.format(
                root_metadata_path
            )
            root_metadata_csv_path = annotation_set_path + "/*.csv"
            remote_root_metadata_csv_path = (
                descriptors.remote_bucket_allen.format(root_metadata_csv_path)
            )
            self.fs.get(
                [remote_root_metadata_path, remote_root_metadata_csv_path],
                local_annotation_set_path,
            )

            annotation_zarr_path = (
                annotation_set_path + "/annotations_compressed.ome.zarr"
            )
            local_zarr_path = (
                local_annotation_set_path / "annotations_compressed.ome.zarr"
            )
            self._fetch_ome_zarr_metadata(
                annotation_zarr_path, local_zarr_path
            )

        # Check template set directory exists
        template_set = self.metadata["annotation_set"]["template"]
        template_set_path = template_set["location"][1:]
        local_template_set_path = working_dir / template_set_path

        if not local_template_set_path.exists():
            # Only download metadata
            root_metadata_path = template_set_path + "/*.json"
            remote_root_metadata_path = descriptors.remote_bucket_allen.format(
                root_metadata_path
            )
            root_metadata_csv_path = template_set_path + "/*.csv"
            remote_root_metadata_csv_path = (
                descriptors.remote_bucket_allen.format(root_metadata_csv_path)
            )
            self.fs.get(
                [remote_root_metadata_path, remote_root_metadata_csv_path],
                local_template_set_path,
            )

            template_zarr_path = template_set_path + "/template.ome.zarr"
            local_zarr_path = local_template_set_path / "template.ome.zarr"
            self._fetch_ome_zarr_metadata(template_zarr_path, local_zarr_path)

    def _fetch_ome_zarr_metadata(self, remote_path, local_path):
        """Fetch zarr metadata from remote path to local path.

        Parameters
        ----------
        remote_path : str
            Remote path to the zarr metadata.
        local_path : Path
            Local path to save the zarr metadata.
        """
        remote_zarr_path = descriptors.remote_bucket_allen.format(
            remote_path + "/zarr.json"
        )
        self.fs.get(remote_zarr_path, local_path / "zarr.json")

        zarr_metadata = read_json(local_path / "zarr.json")
        datasets = zarr_metadata["attributes"]["ome"]["multiscales"][0][
            "datasets"
        ]
        for dataset in datasets:
            zarr_path = remote_path + f"/{dataset['path']}"
            remote_zarr_path = descriptors.remote_bucket_allen.format(
                zarr_path + "/zarr.json"
            )
            local_zarr_path = local_path / dataset["path"]
            self.fs.get(remote_zarr_path, local_zarr_path / "zarr.json")

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

    def _convert_terminology_to_structures(self, terminology_path: Path):
        """Convert the Allen terminology files to a structures.json file.

        Parameters
        ----------
        terminology_path : Path
            Path to the directory containing the Allen terminology files.
        """
        terminology_df = pd.read_csv(terminology_path)
        terminology_df.rename(
            columns={"abbreviation": "acronym"}, inplace=True
        )
        terminology_df["id"] = terminology_df["identifier"].apply(
            lambda x: x.split(":")[1]
        )
        terminology_df["parent_structure_id"] = terminology_df[
            "parent_identifier"
        ].apply(lambda x: x.split(":")[1] if pd.notna(x) else None)
        terminology_df.set_index("id", inplace=True)
        terminology_df["rgb_triplet"] = ""

        for row in terminology_df.itertuples():
            parent_id = row.parent_structure_id
            parent_str = f"/{row.Index}/"

            while parent_id is not None:
                parent_str = f"/{parent_id}" + parent_str
                parent_id = terminology_df.loc[
                    parent_id, "parent_structure_id"
                ]

            terminology_df.loc[row.Index, "structure_id_path"] = parent_str

            hex_color = row.color_hex_triplet.lstrip("#")
            rgb_triplet = [int(hex_color[i : i + 2], 16) for i in (0, 2, 4)]
            terminology_df.at[row.Index, "rgb_triplet"] = rgb_triplet

        terminology_df = terminology_df.reset_index()

        structures_json_df = terminology_df[
            ["acronym", "id", "name", "structure_id_path", "rgb_triplet"]
        ]
        structures_json_df.loc[:, "id"] = structures_json_df["id"].astype(int)
        structures_json_df.loc[:, "structure_id_path"] = structures_json_df[
            "structure_id_path"
        ].apply(lambda x: [int(i) for i in x.strip("/").split("/")])

        structures_json_list = structures_json_df.to_dict(orient="records")

        with open(
            terminology_path.parent / "structures.json", "w", encoding="utf-8"
        ) as f:
            json.dump(structures_json_list, f)

        structures_csv_df = terminology_df[
            [
                "acronym",
                "id",
                "name",
                "structure_id_path",
                "parent_structure_id",
            ]
        ]

        structures_csv_df.to_csv(
            terminology_path.parent / "structures.csv", index=False
        )
