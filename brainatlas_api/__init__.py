from brainatlas_api import bg_atlas
from brainatlas_api.list_atlases import list_atlases

available_atlases = [
    cls for cls in map(bg_atlas.__dict__.get, bg_atlas.__all__)
]


def get_atlas_class_from_name(name):
    a = 1
