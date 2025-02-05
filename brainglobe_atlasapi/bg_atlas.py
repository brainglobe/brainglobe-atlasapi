import tarfile
from io import StringIO
from pathlib import Path
from typing import Optional

import requests
from rich import print as rprint
from rich.console import Console

from brainglobe_atlasapi import config, core, descriptors, utils
from brainglobe_atlasapi.utils import (
    _rich_atlas_metadata,
    check_gin_status,
    check_internet_connection,
)

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
    fn_update : Callable
        Handler function to update during download. Takes completed and total
        bytes.

    """

    atlas_name = None
    _remote_url_base = descriptors.remote_url_base

    def __init__(
        self,
        atlas_name,
        brainglobe_dir=None,
        interm_download_dir=None,
        check_latest=True,
        config_dir=None,
        fn_update=None,
    ):
        self.atlas_name = atlas_name
        self.fn_update = fn_update

        # Read BrainGlobe configuration file:
        conf = config.read_config(config_dir)

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
            if self.remote_version is None:
                check_internet_connection(raise_error=True)
                check_gin_status(raise_error=True)

                # If internet and GIN are up, then the atlas name was invalid
                raise ValueError(f"{atlas_name} is not a valid atlas name!")
            else:
                self.download_extract_file()

        # Instantiate after eventual download:
        super().__init__(self.brainglobe_dir / self.local_full_name)

        if check_latest:
            self.check_latest_version()

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

        try:
            # Grasp remote version
            versions_conf = utils.conf_from_url(remote_url)
        except requests.ConnectionError:
            return None

        try:
            return _version_tuple_from_str(
                versions_conf["atlases"][self.atlas_name]
            )
        except KeyError:
            return None

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
                f"Multiple versions of atlas {self.atlas_name} in "
                f"{self.brainglobe_dir}"
            )
        # If no one exist, return None:
        elif len(candidate_dirs) == 0:
            return
        # Else, return actual name:
        else:
            return candidate_dirs[0].name

    @property
    def remote_url(self):
        """Format complete url for download."""

        if self.remote_version is not None:
            name = (
                f"{self.atlas_name}_v{self.remote_version[0]}."
                f"{self.remote_version[1]}.tar.gz"
            )

            return self._remote_url_base.format(name)

    def download_extract_file(self):
        """Download and extract atlas from remote url."""
        check_internet_connection()
        check_gin_status()

        # Get path to folder where data will be saved
        destination_path = self.interm_download_dir / COMPRESSED_FILENAME

        # Try to download atlas data
        utils.retrieve_over_http(
            self.remote_url, destination_path, self.fn_update
        )

        # Uncompress in brainglobe path:
        tar = tarfile.open(destination_path)
        tar.extractall(path=self.brainglobe_dir)
        tar.close()

        destination_path.unlink()

    def check_latest_version(
        self, print_warning: bool = True
    ) -> Optional[bool]:
        """
        Checks if the local version is the latest available
        and prompts the user to update if not.

        Parameters
        ----------
        print_warning : bool, optional
            If True, prints a message if the local version is not the latest,
            by default True. Useful to turn off, e.g. when the user is updating
            the atlas

        Returns
        -------
        Optional[bool]
            Returns False if the local version is not the latest,
            True if it is, and None if we are offline.
        """

        # Cache remote version to avoid multiple requests
        remote_version = self.remote_version
        # If we are offline, return None
        if remote_version is None:
            return

        local = _version_str_from_tuple(self.local_version)
        online = _version_str_from_tuple(remote_version)

        if local != online:
            if print_warning:
                rprint(
                    "[b][magenta2]brainglobe_atlasapi[/b]: "
                    f"[b]{self.atlas_name}[/b] version [b]{local}[/b] "
                    f"is not the latest available ([b]{online}[/b]). "
                    "To update the atlas run in the terminal:[/magenta2]\n"
                    f" [gold1]brainglobe update -a {self.atlas_name}[/gold1]"
                )
            return False
        return True

    def __repr__(self):
        """Fancy print providing atlas information."""
        name_split = self.atlas_name.split("_")
        pretty_name = "{} {} atlas (res. {})".format(*name_split)
        return pretty_name

    def __str__(self):
        """
        If the atlas metadat are to be printed
        with the built in print function instead of rich's, then
        print the rich panel as a string.

        It will miss the colors.

        """
        buf = StringIO()
        _console = Console(file=buf, force_jupyter=False)
        _console.print(self)

        return buf.getvalue()

    def __rich_console__(self, *args):
        """
        Method for rich API's console protocol.
        Prints the atlas metadata as a table nested in a panel
        """
        panel = _rich_atlas_metadata(self.atlas_name, self.metadata)
        yield panel
