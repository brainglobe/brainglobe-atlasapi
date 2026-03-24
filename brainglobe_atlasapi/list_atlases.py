"""
Functionality to list all available and downloaded
brainglobe atlases.
"""

from typing import Any, Dict, List, Optional

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from brainglobe_atlasapi import config, descriptors, utils


def get_downloaded_atlases() -> List[str]:
    """Get a list of all the downloaded atlases.

    Returns
    -------
    List[str]
        A list of the locally available atlases.
    """
    # Get brainglobe directory:
    brainglobe_dir = config.get_brainglobe_dir()
    atlases_dir = brainglobe_dir / descriptors.V2_ATLAS_ROOTDIR

    downloaded_atlases = []

    if not atlases_dir.exists():
        return downloaded_atlases

    for f in atlases_dir.iterdir():
        if f.is_dir():
            downloaded_atlases.append(f.name)

    sorted_atlases = sorted(downloaded_atlases)

    return sorted_atlases


def get_local_atlas_version(atlas_name: str) -> Optional[str]:
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
    atlas_dir = brainglobe_dir / descriptors.V2_ATLAS_ROOTDIR / atlas_name
    atlas_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        available_versions = [
            p.name for p in atlas_dir.iterdir() if p.is_dir()
        ]
        latest_version = utils.get_latest_version(available_versions)
        return latest_version
    except (IndexError, FileNotFoundError, ValueError):
        print(f"No atlas found with the name: {atlas_name}")
        return None


def get_all_atlases_lastversions() -> Dict[str, Any]:
    """Read from URL or local cache all available last versions."""
    v2_dir = descriptors.V2_ATLAS_ROOTDIR
    cache_path = config.get_brainglobe_dir() / v2_dir / "last_versions.conf"
    custom_path = config.get_brainglobe_dir() / v2_dir / "custom_atlases.conf"

    if utils.check_internet_connection(raise_error=False):
        official_atlases = utils.conf_from_url(
            descriptors.remote_url_s3_http.format(
                f"{v2_dir}/last_versions.conf"
            ),
            cache_path,
        )
    else:
        print("Cannot fetch latest atlas versions from the server.")
        official_atlases = utils.conf_from_file(cache_path)
    try:
        custom_atlases = utils.conf_from_file(custom_path)
    except FileNotFoundError:
        return dict(official_atlases["atlases"])
    return {**official_atlases["atlases"], **custom_atlases["atlases"]}


def get_atlases_lastversions() -> Dict[str, Dict[str, Any]]:
    """
    Return a dictionary of atlas metadata for the latest versions of all
    available atlases.

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
                updated=str(available_atlases[name]).replace(".", "_")
                == local_version,
            )
    return atlases


def show_atlases(show_local_path: bool = False, table_width: int = 88) -> None:
    """
    Print a formatted table with the name and version of local (downloaded)
    and online (available) atlases.

    Parameters
    ----------
    show_local_path : bool, optional
        If True, includes the local path of the atlases
        in the table (default is False).
    table_width : int, optional
        The width of the table to be printed (default is 88).

    Returns
    -------
    None

    """
    available_atlases = get_all_atlases_lastversions()

    # Get local atlases
    downloaded_atlases = get_atlases_lastversions()

    # Get atlases not yet downloaded
    non_downloaded_atlases = {}
    for atlas in available_atlases.keys():
        if atlas not in downloaded_atlases.keys():
            non_downloaded_atlases[str(atlas)] = dict(
                downloaded=False,
                local="",
                version="",
                latest_version=str(available_atlases[atlas]),
                updated=None,
            )

    # Create table
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

    # Add downloaded atlases (sorted) to the table first
    for atlas_name in sorted(downloaded_atlases.keys()):
        atlas = downloaded_atlases[atlas_name]
        table = add_atlas_to_row(
            atlas_name, atlas, table, show_local_path=show_local_path
        )

    # Then add non-download atlases (sorted) to the table
    for atlas_name in sorted(non_downloaded_atlases.keys()):
        atlas = non_downloaded_atlases[atlas_name]
        table = add_atlas_to_row(
            atlas_name, atlas, table, show_local_path=show_local_path
        )

    # Print the resulting table
    rprint(
        Panel.fit(
            table,
            width=table_width,
            title="Brainglobe Atlases",
        )
    )


def add_atlas_to_row(
    atlas: str,
    info: Dict[str, Any],
    table: Table,
    show_local_path: bool = False,
) -> Table:
    """
    Add information about each atlas to a row of the rich table.

    Parameters
    ----------
    atlas : str
        The name of the atlas.
    info : dict
        A dictionary containing information about the atlas.
    table : rich.table.Table
        The table to which the row will be added.
    show_local_path : bool, optional
        If True, includes the local path of the atlas
        in the row (default is False).

    Returns
    -------
    rich.table.Table
        The updated table with the new row added.
    -------

    """
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
        ("[#c4c4c4]" + info["version"] if "-" not in info["version"] else ""),
        "[#c4c4c4]" + info["latest_version"],
    ]

    if show_local_path:
        row.append(info["local"])

    table.add_row(*row)

    return table
