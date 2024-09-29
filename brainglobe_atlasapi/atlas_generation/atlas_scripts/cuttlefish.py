__version__ = "0"

import csv
import glob as glob
import json
from pathlib import Path

import pooch
import re

from brainglobe_atlasapi import utils
from brainglobe_utils.IO.image import load


import pandas as pd

# from skimage import io
"""from brainglobe_atlasapi.atlas_generation.mesh_utils import (
    Region,
    create_region_mesh,
)"""


def hex_to_rgb(hex):
    hex = hex.lstrip("#")
    rgb = []
    for i in (0, 2, 4):
        decimal = int(hex[i : i + 2], 16)
        rgb.append(decimal)

    return rgb


def create_atlas(working_dir, resolution):

    HIERARCHY_FILE_URL = "https://raw.githubusercontent.com/noisyneuron/cuttlebase-util/main/data/brain-hierarchy.csv"
    TEMPLATE_URL = r"https://www.dropbox.com/scl/fo/fz8gnpt4xqduf0dnmgrat/ABflM0-v-b4_2WthGaeYM4s/Averaged%2C%20template%20brain/2023_FINAL-Cuttlebase_warped_template.nii.gz?rlkey=eklemeh57slu7v6j1gphqup4z&dl=1"
    ANNOTATION_URL = r"https://www.dropbox.com/scl/fo/fz8gnpt4xqduf0dnmgrat/ALfSeAj81IM0v56bEeoTfUQ/Averaged%2C%20template%20brain/2023_FINAL-Cuttlebase_warped_template_lobe-labels.nii.seg.nrrd?rlkey=eklemeh57slu7v6j1gphqup4z&dl=1"

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
    
    # import cuttlefish .nrrd file 
    annotation_path = pooch.retrieve(
        ANNOTATION_URL,
        known_hash="768973251b179902ab48499093a4cc870cb6507c09ce46ff76b8203daf243f82",
        progressbar=True,
    )
    
    import nrrd
    # process brain annotation file. There are a total of 70 segments. 
    print("Processing brain annotations:")
    readdata, header = nrrd.read(annotation_path)
    
    # Extract annotation mapping information from nrrd headers, to be applied to hierarchy file later. 
    mapping = []
    for n in range(0,70):
        mapping.append({'color':header[f'Segment{n}_Color'], 'ID':header[f'Segment{n}_LabelValue'], 'acronym':header[f'Segment{n}_Name']})
    
    # convert the color information stored as a string of 3 RGB floats into a list of 3 RGB integers from 0 to 255.
    for index, Map in enumerate(mapping):
        mapping[index]['color'] = Map['color'].split(' ')
        mapping[index]['color'] = list(map(float, mapping[index]['color']))
        mapping[index]['color'] = [int(255*x) for x in mapping[index]['color']]
    
    #print(mapping)
    #df = pd.DataFrame(mapping)
    #df.to_csv('mappingtest.csv')
    
    

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
            if row['hasSides'] == 'Y':
                leftSide = dict(row)
                leftSide['abbreviation'] = leftSide['abbreviation'] + 'l'
                leftSide['name'] = leftSide['name'] + ' (left)'
                
                rightSide = dict(row)
                rightSide['abbreviation'] = rightSide['abbreviation'] + 'r'
                rightSide['name'] = rightSide['name'] + ' (right)'
                
                hierarchy.append(leftSide)
                hierarchy.append(rightSide)
            else:
                hierarchy.append(row)
    
    # use layer1 and layer2 to give IDs to regions which do not have existing IDs. 
    layer1 = 100
    layer2 = 200
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
        if len(hierarchy[i]['structure_id_path']) < 4 and hierarchy[i]['structure_id_path'][-2] != 3:
            if len(hierarchy[i]['structure_id_path']) == 3:
                hierarchy[i]['ID'] = layer2
                layer2 += 1
            elif len(hierarchy[i]['structure_id_path']) == 2:
                hierarchy[i]['ID'] = layer1
                layer1 += 1
        if hierarchy[i]['acronym'] == 'SB': 
            hierarchy[i]['ID'] = 71
        elif hierarchy[i]['acronym'] == 'IB':
            hierarchy[i]['ID'] = 72
            
    
    # remove erroneous key for the VS region (error due to commas being included in the 'function' column)
    hierarchy[-3].pop(None)
    hierarchy[-4].pop(None)
    
    # add the 'root' structure
    hierarchy.append(
        {
            "name": "root",
            "acronym": "root",
            "structure_id_path": [999],
            "ID": 999,
            "parent_structure_id": None,
        }
    )


    # apply colour and ID map to each region
    for index, region in enumerate(hierarchy):
        for Map in mapping: 
            if region['acronym'] == Map['acronym']:
                hierarchy[index]['rgb_triplet'] = Map['color']
                hierarchy[index]['ID'] = int(Map['ID'])
    
    # amend each region's structure_id_path by iterating through entire list, and replacing dummy values with actual ID values. 
    for i in range(0, len(hierarchy)):
        if len(hierarchy[i]['structure_id_path']) == 2:
            hierarchy[i]['structure_id_path'][1] = hierarchy[i]['ID']
            len2_shortest_index = i
        
        elif len(hierarchy[i]['structure_id_path']) == 3:
            hierarchy[i]['structure_id_path'][1] = hierarchy[len2_shortest_index]['ID']
            hierarchy[i]['structure_id_path'][2] = hierarchy[i]['ID']
            len3_shortest_index = i
            
        elif len(hierarchy[i]['structure_id_path']) == 4:
            hierarchy[i]['structure_id_path'][1] = hierarchy[len2_shortest_index]['ID']
            hierarchy[i]['structure_id_path'][2] = hierarchy[len3_shortest_index]['ID']
            hierarchy[i]['structure_id_path'][3] = hierarchy[i]["ID"]
        # find parent_structure_id using resulting structure_id_path
        if hierarchy[i]['name'] != 'root':
            hierarchy[i]['parent_structure_id'] = hierarchy[i]['structure_id_path'][-2]
    
    
    # original atlas does not give colours to some regions, so we give random RGB triplets to regions without specified RGB triplet values
    random_rgb_triplets = [[156, 23, 189],[45, 178, 75],[231, 98, 50],[12, 200, 155],[87, 34, 255],[190, 145, 66],[64, 199, 225],
    [255, 120, 5],[10, 45, 90],[145, 222, 33],[35, 167, 204],[76, 0, 89], [27, 237, 236], [255, 255, 255]]
    
    n = 0
    for index, region in enumerate(hierarchy):
        if "rgb_triplet" not in region:
            hierarchy[index]["rgb_triplet"] = random_rgb_triplets[n]
            n = n + 1

    # give filler acronyms for regions without specified acronyms
    missing_acronyms = ['SpEM', 'VLC', 'BsLC', 'SbEM', 'PLC', 'McLC', 'PvLC', 'BLC', 'PeM', 
                        'OTC', 'NF']
    n = 0
    for index, region in enumerate(hierarchy):
        if hierarchy[index]['acronym'] == '':
            hierarchy[index]['acronym'] = missing_acronyms[n]
            n = n+1
    
    
    # import cuttlefish .nii file 
    template_path = pooch.retrieve(
        TEMPLATE_URL,
        known_hash="195125305a11abe6786be1b32830a8aed1bc8f68948ad53fa84bf74efe7cbe9c",
        progressbar=True,
    )

    # process brain template MRI file
    print("Processing brain template:")
    brain_template = load.load_nii(template_path)
    
    
    # check the transformed version of the hierarchy.csv file
    print(hierarchy)
    #df = pd.DataFrame(hierarchy)
    #df.to_csv('hierarchy_test.csv')
    
    return None


if __name__ == "__main__":
    res = 2, 2, 2
    home = str(Path.home())
    bg_root_dir = Path.home() / "brainglobe_workingdir"
    bg_root_dir.mkdir(exist_ok=True, parents=True)

    create_atlas(bg_root_dir, res)
