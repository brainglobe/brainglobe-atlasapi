from rich import print as rprint
import shutil

import bg_atlasapi
from bg_atlasapi.bg_atlas import _version_str_from_tuple
from bg_atlasapi.list_atlases import get_downloaded_atlases


def update_atlas(atlas_name=None, force=False):
    """
        Updates a bg_atlasapi atlas from the latest
        available version online. 

        Arguments:
        ----------
        atlas_name: str, name of the atlas to update
        force: bool, if False it checks if the atlas is already
            at the latest version and doesn't update if
            that's the case.
    """
    # Check input
    if not isinstance(atlas_name, str):
        raise ValueError(f"atlas name should be a string, not {atlas_name}")

    # Get atlas class
    atlasclass = bg_atlasapi.get_atlas_class_from_name(atlas_name)
    if atlasclass is None:
        return

    atlas = atlasclass()

    # Check if we need to update
    if not force:
        if atlas.check_latest_version():
            rprint(
                f"[b][magenta2]bg_atlasapi: {atlas.atlas_name} is already updated "
                + f"(version: {_version_str_from_tuple(atlas.local_version)})[/b]"
            )
            return

    # Delete atlas folder
    rprint(
        f"[b][magenta2]bg_atlasapi: updating {atlas.atlas_name}[/magenta2][/b]"
    )
    fld = atlas.brainglobe_dir / atlas.local_full_name
    shutil.rmtree(fld)
    if fld.exists():
        raise ValueError(
            "Something went wrong while tryint to delete the old version of the atlas, aborting."
        )

    # Download again
    atlas.download_extract_file()

    # Check that everything went well
    atlasclass()
    rprint(
        f"[b][magenta2]bg_atlasapi: {atlas.atlas_name} updated to version: "
        + f"{_version_str_from_tuple(atlas.remote_version)}[/magenta2][/b]"
    )


def install_atlas(atlas_name):
    """
        Installs a BrainGlobe atlas from the latest
        available version online. 

        Arguments:
        ----------
        atlas_name: str, name of the atlas to update
    """

    # Check input
    if not isinstance(atlas_name, str):
        raise ValueError(f"atlas name should be a string, not {atlas_name}")

    # Check if already downloaded
    atlases = get_downloaded_atlases().keys()
    if atlas_name in atlases:
        rprint(
            f"[b][magenta2]Bg_atlasapi: installing {atlas_name}: atlas already installed![/magenta2][/b]"
        )
        return

    # Get atlas class
    atlasclass = bg_atlasapi.get_atlas_class_from_name(atlas_name)
    if atlasclass is None:
        raise ValueError(
            f"We could not find a class for the altas named pased: {atlas_name}"
        )

    # Instantiate class to download if not present
    atlasclass()
