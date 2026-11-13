"""Provides functionality to update the atlas."""

import shutil

from rich import print as rprint

from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import (
    folder_version_to_dotted,
    get_downloaded_atlases,
)


def update_atlas(atlas_name, force=False, fn_update=None, remote=None):
    """Update a brainglobe_atlasapi atlas from the latest
    available version online.

    Parameters
    ----------
    atlas_name: str
        Name of the atlas to update.
    force: bool
        If False it checks if the atlas is already at the latest version
        and doesn't update if that's the case.
    fn_update : Callable, Optional
        A callback function to update progress during download.
    remote : str, Optional
        An ``s3://bucket/prefix`` to fetch the atlas from instead of
        searching the configured remotes.
    """
    atlas = BrainGlobeAtlas(
        atlas_name=atlas_name,
        check_latest=False,
        fn_update=fn_update,
        remote=remote,
    )

    is_latest_version = atlas.check_latest_version(print_warning=False)

    if force and is_latest_version:
        # Delete atlas folder to force update
        atlas_folder = (atlas.brainglobe_dir / atlas.local_full_name).parent
        shutil.rmtree(atlas_folder)
        if atlas_folder.exists():
            raise ValueError(
                "Something went wrong while trying to delete the old "
                "version of the atlas, aborting."
            )

    elif is_latest_version:
        rprint(
            f"[b][magenta2]brainglobe_atlasapi: {atlas.atlas_name} "
            "is already updated "
            f"(version: {folder_version_to_dotted(atlas._local_version_str)})"
            "[/b]"
        )

        return

    rprint(
        "[b][magenta2]brainglobe_atlasapi: "
        f"updating {atlas.atlas_name}[/magenta2][/b]"
    )

    # Download again
    atlas.download()

    # Check that everything went well
    rprint(
        "[b][magenta2]brainglobe_atlasapi: "
        f"{atlas.atlas_name} updated to version: "
        f"{folder_version_to_dotted(atlas._remote_version_str)}[/magenta2][/b]"
    )


def install_atlas(atlas_name, fn_update=None, remote=None):
    """Installs a BrainGlobe atlas from the latest
    available version online.

    Parameters
    ----------
    atlas_name : str
        Name of the atlas to update.
    fn_update : Callable, Optional
        A callback function to update progress during download.
    remote : str, Optional
        An ``s3://bucket/prefix`` to fetch the atlas from instead of
        searching the configured remotes.
    """
    # Check input:
    if not isinstance(atlas_name, str):
        raise TypeError(
            f"Atlas name should be a string, not a "
            f"{type(atlas_name).__name__}."
        )

    # Check if already downloaded. get_downloaded_atlases only sees the
    # default cache; with an explicit remote, BrainGlobeAtlas checks that
    # remote's cache itself.
    if remote is None and atlas_name in get_downloaded_atlases():
        rprint(
            f"[b][magenta2]brainglobe_atlasapi: installing {atlas_name}: "
            "atlas already installed![/magenta2][/b]"
        )
        return

    # Istantiate to download:
    BrainGlobeAtlas(
        atlas_name,
        fn_update=fn_update,
        remote=remote,
    )
