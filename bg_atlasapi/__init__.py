from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bg-atlasapi")
    del version
except PackageNotFoundError:
    # package is not installed
    pass

__author__ = "BrainGlobe Developers"


from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bg_atlasapi.list_atlases import show_atlases
