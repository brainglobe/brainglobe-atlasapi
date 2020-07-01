from rich import print as rprint
import shutil
import click

import brainatlas_api
from brainatlas_api.bg_atlas import _version_str_from_tuple


def update_atlas(atlas_name=None, force=False):
    """
        Updates a brainatlas_api atlas from the latest
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
    atlasclass = brainatlas_api.get_atlas_class_from_name(atlas_name)
    if atlasclass is None:
        return

    atlas = atlasclass()

    # Check if we need to update
    if not force:
        if atlas.check_latest_version():
            rprint(
                f"[b][magenta2]Brainatlas_api: {atlas.atlas_name} is already updated "
                + f"(version: {_version_str_from_tuple(atlas.local_version)})[/b]"
            )
            return

    # Delete atlas folder
    rprint(
        f"[b][magenta2]Brainatlas_api: updating {atlas.atlas_name}[/magenta2][/b]"
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
        f"[b][magenta2]Brainatlas_api: {atlas.atlas_name} updated to version: "
        + f"{_version_str_from_tuple(atlas.remote_version)}[/magenta2][/b]"
    )


@click.command()
@click.option("-a", "--atlas_name")
@click.option("-f", "--force", is_flag=True)
def cli_update_atlas_command(atlas_name, force=False):
    return update_atlas(atlas_name, force=force)
