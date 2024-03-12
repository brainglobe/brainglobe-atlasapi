from importlib.metadata import PackageNotFoundError, metadata

try:
    __version__ = metadata("brainglobe-atlasapi")["Version"]
    __author__ = metadata("brainglobe-atlasapi")["Author-email"]
    del metadata
except PackageNotFoundError:
    # package is not installed
    pass


from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
from brainglobe_atlasapi.list_atlases import show_atlases
