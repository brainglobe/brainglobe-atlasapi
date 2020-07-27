from pathlib import Path
import tarfile
import requests

from rich import print as rprint

from bg_atlasapi import utils, config, core, descriptors


COMPRESSED_FILENAME = "atlas.tar.gz"


def _version_tuple_from_str(version_str):
    return tuple([int(n) for n in version_str.split(".")])


def _version_str_from_tuple(version_tuple):
    return f"{version_tuple[0]}.{version_tuple[1]}"


class BrainGlobeAtlas(core.Atlas):
    """Add remote atlas fetching and version comparison functionalities
    to the core Atlas class.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to be used.
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

    """

    atlas_name = None
    _remote_url_base = descriptors.remote_url_base

    def __init__(
        self,
        atlas_name,
        brainglobe_dir=None,
        interm_download_dir=None,
        check_latest=True,
        print_authors=True,
    ):
        self.atlas_name = atlas_name

        # Read BrainGlobe configuration file:
        conf = config.read_config()

        # Use either input locations or locations from the config file,
        # and create directory if it does not exist:
        for dir, dirname in zip(
            [brainglobe_dir, interm_download_dir],
            ["brainglobe_dir", "interm_download_dir"],
        ):
            if dir is None:
                dir = conf["default_dirs"][dirname]

            # If the default folder does not exist yet, make it:
            dir_path = Path(dir)
            dir_path.mkdir(exist_ok=True)
            setattr(self, dirname, dir_path)

        # Look for this atlas in local brainglobe folder:
        if self.local_full_name is None:
            rprint(
                f"[magenta2]Bgatlas_api: {self.atlas_name} not found locally. Downloading...[magenta2]"
            )
            self.download_extract_file()

        # Instantiate after eventual download:
        super().__init__(self.brainglobe_dir / self.local_full_name)

        if check_latest:
            self.check_latest_version()
        if print_authors:
            print(self)

    @property
    def local_version(self):
        """If atlas is local, return actual version of the downloaded files;
        Else, return none.
        """
        full_name = self.local_full_name

        if full_name is None:
            return None

        return _version_tuple_from_str(full_name.split("_v")[-1])

    @property
    def remote_version(self):
        """Remote version read from GIN conf file. If we are offline, return
        None.
        """
        remote_url = self._remote_url_base.format("last_versions.conf")

        # Grasp remote version if a connection is available:
        try:
            versions_conf = utils.conf_from_url(remote_url)
        except requests.ConnectionError:
            return

        try:
            return _version_tuple_from_str(
                versions_conf["atlases"][self.atlas_name]
            )
        except KeyError:
            raise ValueError(f"{self.atlas_name} is not a valid atlas name!")

    @property
    def local_full_name(self):
        """As we can't know the local version a priori, search candidate dirs
        using name and not version number. If none is found, return None.
        """
        pattern = f"{self.atlas_name}_v*"
        candidate_dirs = list(self.brainglobe_dir.glob(pattern))

        # If multiple folders exist, raise error:
        if len(candidate_dirs) > 1:
            raise FileExistsError(
                f"Multiple versions of atlas {self.atlas_name} in {self.brainglobe_dir}"
            )
        # If no one exist, return None:
        elif len(candidate_dirs) == 0:
            return
        # Else, return actual name:
        else:
            return candidate_dirs[0].name

    @property
    def remote_url(self):
        """Format complete url for download.
        """
        maj, min = self.remote_version
        name = f"{self.atlas_name}_v{maj}.{min}.tar.gz"
        return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url.
        """
        utils.check_internet_connection()

        # Get path to folder where data will be saved
        destination_path = self.interm_download_dir / COMPRESSED_FILENAME

        # Try to download atlas data
        utils.retrieve_over_http(self.remote_url, destination_path)

        # Uncompress in brainglobe path:
        tar = tarfile.open(destination_path)
        tar.extractall(path=self.brainglobe_dir)
        tar.close()

        destination_path.unlink()

    def check_latest_version(self):
        """Checks if the local version is the latest available
        and prompts the user to update if not.
        """
        if self.remote_version is None:  # in this case, we are offline
            return
        local = _version_str_from_tuple(self.local_version)
        online = _version_str_from_tuple(self.remote_version)

        if local != online:
            rprint(
                f"[b][magenta2]Bg_atlasapi[/b]: [b]{self.atlas_name}[/b] version [b]{local}[/b] is not the latest available ([b]{online}[/b]). "
                + "To update the atlas run in the terminal:[/magenta2]\n"
                + f"    [gold1]brainglobe update -a {self.atlas_name}[/gold1]"
            )
            return False
        return True

    def __repr__(self):
        """Fancy print for the atlas providing authors information.
        """
        meta = self.metadata
        name_split = self.atlas_name.split("_")
        pretty_name = "{} {} atlas (res. {})".format(*name_split)
        pretty_string = (
            f"{pretty_name}\nFrom: {meta['atlas_link']} ({meta['citation']} )"
        )
        return pretty_string
