__author__ = """brainglobe"""
__version__ = "0.2.0"

import pyinspect
pyinspect.install_traceback()

from bg_atlasapi.bg_atlas import BrainGlobeAtlas
from bg_atlasapi.list_atlases import show_atlases
