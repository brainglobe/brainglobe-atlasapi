__version__ = "0"

import csv
import glob as glob
import json
from pathlib import Path

import pooch

from brainglobe_atlasapi import utils

# from skimage import io
"""from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)"""


def create_atlas(working_dir, resolution):

    HIERARCHY_FILE_URL = "https://raw.githubusercontent.com/noisyneuron/cuttlebase-util/main/data/brain-hierarchy.csv"
    BRAIN_SCENE_URL = "https://raw.githubusercontent.com/noisyneuron/cuttlebase-util/main/data/brain-scene.json"

    download_dir_path = working_dir / "downloads"
    download_dir_path.mkdir(exist_ok=True)
    atlas_path = download_dir_path / "atlas_files"

    # download hierarchy files
    utils.check_internet_connection()
    hierarchy_path = pooch.retrieve(
        HIERARCHY_FILE_URL,
        known_hash="023418e626bdefbd177d4bb8c08661bd63a95ccff47720e64bb7a71546935b77",
        progressbar=True,
    )

    # create dictionaries
    print("Creating structure tree")
    with open(
        hierarchy_path, mode="r", encoding="utf-8-sig"
    ) as cuttlefish_file:
        cuttlefish_dict_reader = csv.DictReader(cuttlefish_file)

        # empty list to populate with dictionaries
        hierarchy = []

        # parse through csv file and populate hierarchy list
        for row in cuttlefish_dict_reader:
            hierarchy.append(row)

    # remove 'hasSides' and 'function' keys, reorder and rename the remaining keys
    for i in range(0, len(hierarchy)):
        hierarchy[i]["acronym"] = hierarchy[i].pop("abbreviation")
        hierarchy[i].pop("hasSides")
        hierarchy[i].pop("function")
        hierarchy[i]["structure_id_path"] = list(
            (map(int, (hierarchy[i]["index"].split("-"))))
        )
        hierarchy[i]["structure_id_path"].insert(0, 999)
        hierarchy[i].pop("index")
        path_string = [str(i) for i in hierarchy[i]["structure_id_path"]]
        hierarchy[i]["id"] = int("".join(path_string))
        hierarchy[i]["parent_structure_id"] = int(str(hierarchy[i]["id"])[:-1])
        prev = ""
        for index, id in enumerate(hierarchy[i]["structure_id_path"]):
            hierarchy[i]["structure_id_path"][index] = str(prev) + str(id)
            prev = hierarchy[i]["structure_id_path"][index]

    # add the 'root' structure
    hierarchy.append(
        {
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "id": 999,
            "parent_structure_id": None,
        }
    )

    # download region colour data
    brain_scene_path = pooch.retrieve(
        BRAIN_SCENE_URL,
        known_hash="057fe98ea5ae24c5f9a10aebec072a12f6df19447c3c027f0f12ddba61a1bb90",
        progressbar=True,
    )

    # apply colour map to each region
    print("Applying colours:")
    f = open(brain_scene_path)
    brain_scene = json.load()
    colormap = brain_scene["params"]["colors"]

    print(colormap)
    f.close()
    # check the transformed version of the hierarchy.csv file
    # print(hierarchy)

    return None


if __name__ == "__main__":
    res = 2, 2, 2
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)
