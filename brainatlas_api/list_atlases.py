from pathlib import Path
from rich.table import Table, box, Style
from rich import print as rprint
import click

from bg_atlasapi import config
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bg_atlasapi import utils


"""
    Some functionality to list all available and downloaded brainglobe atlases
"""


def show_atlases(show_local_path=False):
    """ 
        Print's a formatted table with the name and version of local (downloaded)
        and online (available) atlases.

        Downloads the latest atlas version and compares it with what it's stored
        locally. 
    """
    if not utils.check_internet_connection():
        print(
            "Sorry, we need a working internet connection to retriev the latest metadata"
        )
        return

    # --------------------------- Get available_atlases -------------------------- #
    available_atlases = utils.conf_from_url(
        BrainGlobeAtlas._remote_url_base.format("last_versions.conf")
    )
    available_atlases = dict(available_atlases["atlases"])

    # ----------------------------- Get local atlases ---------------------------- #
    # Get brainglobe directory
    conf = config.read_config()
    brainglobe_dir = Path(conf["default_dirs"]["brainglobe_dir"])

    # Get downloaded atlases
    atlases = {}
    for elem in brainglobe_dir.iterdir():
        if elem.is_dir():
            name = elem.name.split("_v")[0]
            if name in available_atlases.keys():
                atlases[name] = dict(
                    downloaded=True,
                    local=str(elem),
                    version=elem.name.split("_v")[-1],
                    latest_version=str(available_atlases[name]),
                    updated=str(available_atlases[name])
                    == elem.name.split("_v")[-1],
                )

    # ---------------------- Get atlases not yet downloaded ---------------------- #
    for atlas in available_atlases.keys():
        if atlas not in atlases.keys():
            atlases[str(atlas)] = dict(
                downloaded=False,
                local="[red]---[/red]",
                version="[red]---[/red]",
                latest_version=str(available_atlases[str(name)]),
                updated=None,
            )

    # -------------------------------- print table ------------------------------- #
    table = Table(
        show_header=True,
        header_style="bold green",
        title="\n\nBrainglobe Atlases",
        expand=False,
        box=box.ROUNDED,
    )

    table.add_column("Name", no_wrap=True, width=32)
    table.add_column("Downloaded", justify="center")
    table.add_column("Local version", justify="center")
    table.add_column("Latest version", justify="center")
    if show_local_path:
        table.add_column("Local path")

    for n, (atlas, info) in enumerate(atlases.items()):
        if info["downloaded"]:
            downloaded = "[green]:heavy_check_mark:[/green]"
        else:
            downloaded = "[red]---[/red]"

        row = [
            "[b]" + atlas + "[/b]",
            downloaded,
            info["version"],
            info["latest_version"],
        ]

        if show_local_path:
            row.append(info["local"])

        table.add_row(*row,)

        if info["updated"] is not None:
            if not info["updated"]:
                table.row_styles.append(
                    Style(color="black", bgcolor="magenta2")
                )

    rprint(table)


@click.command()
@click.option("-s", "--show_local_path", is_flag=True)
def cli_show_atlases(show_local_path=False):
    return show_atlases(show_local_path=show_local_path)
