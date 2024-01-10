from importlib.metadata import PackageNotFoundError, metadata

try:
    __version__ = metadata("bg-atlasapi")["Version"]
    __author__ = metadata("bg-atlasapi")["Author"]
    del metadata
except PackageNotFoundError:
    # package is not installed
    pass


from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bg_atlasapi.list_atlases import show_atlases
