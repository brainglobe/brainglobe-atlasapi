from rich import print as rprint

from bg_atlasapi import bg_atlas
from bg_atlasapi.list_atlases import show_atlases

available_atlases = [
    cls for cls in map(bg_atlas.__dict__.get, bg_atlas.__all__)
]


def get_atlas_class_from_name(name):
    names = [atlas.atlas_name for atlas in available_atlases]

    atlases = {n: a for n, a in zip(names, available_atlases)}

    if name in atlases.keys():
        return atlases[name]
    else:
        rprint(
            f"[red1][b]Brainglobe_api[/b]: Could not find atlas with name {name}. Available atlases:[red1]"
        )
        show_atlases()
        return None
