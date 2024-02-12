"""
    Some functionality to list all available and downloaded brainglobe atlases
"""

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from brainglobe_atlasapi import config, descriptors, utils


def get_downloaded_atlases():
    """Get a list of all the downloaded atlases and their version.

    Returns
    -------
    list
        A list of tuples with the locally available atlases and their version
    """

    # Get brainglobe directory:
    brainglobe_dir = config.get_brainglobe_dir()

    return [
        f.name.split("_v")[0]
        for f in brainglobe_dir.glob("*_*_*_v*")
        if f.is_dir()
    ]


def get_local_atlas_version(atlas_name):
    """Get version of a downloaded available atlas.

    Arguments
    ---------
    atlas_name : str
        Name of the atlas.

    Returns
    -------
    str
        Version of atlas.
    """

    brainglobe_dir = config.get_brainglobe_dir()
    return [
        f.name.split("_v")[1]
        for f in brainglobe_dir.glob(f"*{atlas_name}*")
        if f.is_dir()
    ][0]


def get_all_atlases_lastversions():
    """Read from URL all available last versions"""
    available_atlases = utils.conf_from_url(
        descriptors.remote_url_base.format("last_versions.conf")
    )
    available_atlases = dict(available_atlases["atlases"])
    return available_atlases


def get_atlases_lastversions():
    """
    Returns
    -------
    dict
        A dictionary with metadata about already installed atlases.
    """

    available_atlases = get_all_atlases_lastversions()

    # Get downloaded atlases looping over folders in brainglobe directory:
    atlases = {}
    for name in get_downloaded_atlases():
        if name in available_atlases.keys():
            local_version = get_local_atlas_version(name)
            atlases[name] = dict(
                downloaded=True,
                local=name,
                version=local_version,
                latest_version=str(available_atlases[name]),
                updated=str(available_atlases[name]) == local_version,
            )
    return atlases


def show_atlases(show_local_path=False):
    """Prints a formatted table with the name and version of local (downloaded)
    and online (available) atlases. To do so, dowload info on
    the latest atlas version and compares it with what it's stored
    locally.

    Arguments
    ---------
    show_local_path : bool
        If true, local path of the atlases are in the table with the rest
        (optional, default=False).

    """

    available_atlases = get_all_atlases_lastversions()

    # Get local atlases:
    atlases = get_atlases_lastversions()

    # Get atlases not yet downloaded:
    for atlas in available_atlases.keys():
        if atlas not in atlases.keys():
            atlases[str(atlas)] = dict(
                downloaded=False,
                local="",
                version="",
                latest_version=str(available_atlases[atlas]),
                updated=None,
            )

    # Print table:
    table = Table(
        show_header=True,
        header_style="bold green",
        show_lines=True,
        expand=False,
        box=None,
    )

    table.add_column("Name", no_wrap=True, width=32)
    table.add_column("Downloaded", justify="center")
    table.add_column("Updated", justify="center")
    table.add_column("Local version", justify="center")
    table.add_column("Latest version", justify="center")
    if show_local_path:
        table.add_column("Local path")

    for n, (atlas, info) in enumerate(atlases.items()):
        if info["downloaded"]:
            downloaded = "[green]:heavy_check_mark:[/green]"

            if info["version"] == info["latest_version"]:
                updated = "[green]:heavy_check_mark:[/green]"
            else:
                updated = "[red dim]x"

        else:
            downloaded = ""
            updated = ""

        row = [
            "[bold]" + atlas,
            downloaded,
            updated,
            (
                "[#c4c4c4]" + info["version"]
                if "-" not in info["version"]
                else ""
            ),
            "[#c4c4c4]" + info["latest_version"],
        ]

        if show_local_path:
            row.append(info["local"])

        table.add_row(*row)
    rprint(
        Panel.fit(
            table,
            width=88,
            title="Brainglobe Atlases",
        )
    )
