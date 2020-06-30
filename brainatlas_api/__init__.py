from brainatlas_api import bg_atlas
from brainatlas_api.list_atlases import list_atlases

available_atlases = [
    cls for cls in map(bg_atlas.__dict__.get, bg_atlas.__all__)
]


def get_atlas_class_from_name(name):
    names = [atlas.atlas_name for atlas in available_atlases]

    atlases = {n: a for n, a in zip(names, available_atlases)}

    if name in atlases.keys():
        return atlases[name]
    else:
        print(f"Could not find atlas with name {name}. Available atlases:\n")
        list_atlases()
        return None
