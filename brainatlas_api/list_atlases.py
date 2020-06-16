from pathlib import Path
from rich.table import Table
from rich import print as rprint

from brainatlas_api import config
from brainatlas_api import bg_atlas


"""
    Some functionality to list all available and downloaded brainglobe atlases
"""


def list_atlases():
    # Parse config
    conf = config.read_config()
    brainglobe_dir = Path(conf["default_dirs"]["brainglobe_dir"])

    # ----------------------------- Get local atlases ---------------------------- #
    atlases = {}
    for elem in brainglobe_dir.iterdir():
        if elem.is_dir():
            atlases[elem.name] = dict(
                downloaded=True,
                local=str(elem),
                online=bg_atlas.BrainGlobeAtlas._remote_url_base.format(
                    elem.name
                ),
            )

    # ---------------------- Get atlases not yet downloaded ---------------------- #
    available_atlases = [
        cls for cls in map(bg_atlas.__dict__.get, bg_atlas.__all__)
    ]
    for atlas in available_atlases:
        name = f"{atlas.atlas_name}_v{atlas.version}"
        if name not in atlases.keys():
            atlases[str(name)] = dict(
                downloaded=False,
                local="[red]---[/red]",
                online=atlas._remote_url_base.format(name),
            )

    # -------------------------------- print table ------------------------------- #
    table = Table(
        show_header=True,
        header_style="bold green",
        title="\n\nBrainglobe Atlases",
    )
    table.add_column("Name")
    table.add_column("Downloaded")
    table.add_column("Local path")
    table.add_column("Online path", style="dim")

    for atlas, info in atlases.items():
        if info["downloaded"]:
            downloaded = "[green]:heavy_check_mark:[/green]"
        else:
            downloaded = "[red]---[/red]"
        table.add_row(
            "[b]" + atlas + "[/b]", downloaded, info["local"], info["online"]
        )

    rprint(table)
