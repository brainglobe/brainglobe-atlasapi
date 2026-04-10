# %% Imports

from pathlib import Path
import numpy as np
import pandas as pd
import pooch

# %% Metadata
__version__ = 0
ATLAS_NAME = "charm_macaque"
CITATION = ""
SPECIES = "macaca mulatta"
ATLAS_LINK = ""
ATLAS_FILE_URL = ""
ORIENTATION = ""
ROOT_ID = 0
RESOLUTION = 250  # microns
ATLAS_PACKAGER = ""

# %% Download files
BG_ROOT_DIR = Path.home() / "brainglobe_workingdir" / ATLAS_NAME
DOWNLOAD_DIR = BG_ROOT_DIR / "downloads"

NMT_SYM_URL = "https://afni.nimh.nih.gov/pub/dist/atlases/macaque/nmt/NMT_v2.0_sym.tgz"

def download_resources():
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    nmt_files = pooch.retrieve(
        url=NMT_SYM_URL,
        known_hash=None,
        path=DOWNLOAD_DIR,
        fname="NMT_v2.0_sym.tgz",
        processor=pooch.Untar(extract_dir="NMT_v2.0_sym"),
        progressbar=True,
    )


    return {
        "nmt_dir": DOWNLOAD_DIR / "NMT_v2.0_sym",
        "nmt_files": [Path(p) for p in nmt_files],
    }


resources = download_resources()

print("NMT dir:", resources["nmt_dir"])
print("Number of extracted NMT files:", len(resources["nmt_files"]))

print("\nFirst few NMT files:")
for p in resources["nmt_files"][:10]:
    print(p)
