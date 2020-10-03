# BG-atlasAPI

[![Python Version](https://img.shields.io/pypi/pyversions/bg-atlasapi.svg)](https://pypi.org/project/bg-atlasapi)
[![PyPI](https://img.shields.io/pypi/v/bg-atlasapi.svg)](https://pypi.org/project/bg-atlasapi/)
[![Wheel](https://img.shields.io/pypi/wheel/bg-atlasapi.svg)](https://pypi.org/project/bg-atlasapi)
[![Development Status](https://img.shields.io/pypi/status/brainatlas-api.svg)](https://github.com/SainsburyWellcomeCentre/brainatlas-api)
[![Travis](https://img.shields.io/travis/com/brainglobe/bg-atlasapi?label=Travis%20CI)](
    https://travis-ci.com/brainglobe/bg-atlasapi)
[![Coverage Status](https://coveralls.io/repos/github/brainglobe/brainatlas-api/badge.svg?branch=master)](https://coveralls.io/github/brainglobe/brainatlas-api?branch=master)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

![banner](https://github.com/brainglobe/bg-atlasapi/blob/master/docs/img/banner.jpeg)


The brainglobe atlas API (BG-AtlasAPI) provides a common interface for programmers to download and process brain atlas data from multiple sources. 

## Atlases available

A number of atlases are in development, but those available currently are:
* The [Allen Mouse Brain Atlas](www.brain-map.org) at 10, 25 and 50 micron resolutions
* The [Allen Human Brain Atlas](www.brain-map.org) at 100 micron resolution
* The [Max Planck Zebrafish Brain Atlas](http://fishatlas.neuro.mpg.de) at 1 micron resolution
* The [Enhanced and Unified Mouse Brain Atlas](https://kimlab.io/brain-map/atlas/) at 25 micron resolution

## Installation
BG-AtlasAPI works with Python >3.6, and can be installed from PyPI with:
```bash
pip install bg-atlasapi
```

## Usage
Full information can be found in the [documentation](https://docs.brainglobe.info/bg-atlasapi/usage)
### Python API
**List of atlases**

To see a list of atlases use `bg_atlasapi.show_atlases`
```python
from bg_atlasapi import show_atlases
show_atlases()
#                                Brainglobe Atlases                               
# ╭──────────────────────────────────┬────────────┬───────────────┬──────────────╮
# │ Name                             │ Downloaded │ Local version │    Latest    │
# │                                  │            │               │   version    │
# ├──────────────────────────────────┼────────────┼───────────────┼──────────────┤
# │ allen_human_500um                │     ✔      │      0.1      │     0.1      │
# │ mpin_zfish_1um                   │     ✔      │      0.3      │     0.3      │
# │ allen_mouse_50um                 │     ✔      │      0.3      │     0.3      │
# │ kim_unified_25um                 │     ✔      │      0.1      │     0.1      │
# │ allen_mouse_25um                 │     ✔      │      0.3      │     0.3      │
# │ allen_mouse_10um                 │     ✔      │      0.3      │     0.3      │
# │ example_mouse_100um              │    ---     │      ---      │     0.3      │
# ╰──────────────────────────────────┴────────────┴───────────────┴──────────────╯
```

**Using the atlases**

All the features of each atlas can be accessed via the `BrainGlobeAtlas` class. 


e.g. for the 25um Allen Mouse Brain Atlas:

```python
from bg_atlasapi.bg_atlas import BrainGlobeAtlas
atlas = BrainGlobeAtlas("allen_mouse_25um")
```

The various files associated with the atlas can then be accessed as attributes of the class:
```python
# reference image
reference_image = atlas.reference
print(reference_image.shape)
# (528, 320, 456)

# annotation image
annotation_image = atlas.annotation
print(annotation_image.shape)
# (528, 320, 456)

# a hemispheres image (value 1 in left hemisphere, 2 in right) can be generated
hemispheres_image = atlas.hemispheres
print(hemispheres_image.shape)
# (528, 320, 456)
```

**Brain regions**

There are multiple ways to work with individual brain regions. To see a dataframe of each brain region, with it's unique ID, acronym and full name, use `atlas.lookup_df`:
```python
atlas.lookup_df.head(8)
#      acronym         id                           name
# 0       root        997                           root
# 1       grey          8  Basic cell groups and regions
# 2         CH        567                       Cerebrum
# 3        CTX        688                Cerebral cortex
# 4      CTXpl        695                 Cortical plate
# 5  Isocortex        315                      Isocortex
# 6        FRP        184  Frontal pole, cerebral cortex
# 7       FRP1         68          Frontal pole, layer 1
```

Each brain region can also be access by the acronym, e.g. for primary visual cortex (VISp):
```python
from pprint import pprint
VISp = atlas.structures["VISp"]
pprint(VISp)
# {'acronym': 'VISp',
#  'id': 385,
#  'mesh': None,
#  'mesh_filename': PosixPath('/home/user/.brainglobe/allen_mouse_25um_v0.3/meshes/385.obj'),
#  'name': 'Primary visual area',
#  'rgb_triplet': [8, 133, 140],
#  'structure_id_path': [997, 8, 567, 688, 695, 315, 669, 385]}
```

### Note on coordinates in `bg-atlasapi` 
Working with both image coordinates and cartesian coordinates in the same space can be confusing! In `bg-atlasapi`, the origin is always assumed to be in the upper left corner of the image (sectioning along the first dimension), the "ij" convention. This means that when plotting meshes and points using cartesian systems, you might encounter confusing behaviors coming from the fact that in cartesian plots one axis is inverted with respect to  ij coordinates (vertical axis increases going up, image row indexes increase going down). To make things as consistent as possible, in `bg-atlasapi` the 0 of the meshes coordinates is assumed to coincide with the 0 index of the images stack, and meshes coordinates increase following the direction stack indexes increase.
To deal with transformations between your data space and `bg-atlasapi`, you might find helpful the [bg-space](https://github.com/brainglobe/bg-space) package.

# Contributing to bg-atlasapi
**Contributors to bg-atlaspi are absolutely encouraged**, whether you want to fix bugs, add/request new features or simply ask questions.

If you would like to contribute to `bg-atlasapi` (or any of the downstream tools like [brainrender](https://github.com/BrancoLab/BrainRender) etc.\) please get in touch by opening a new issue or pull request on [GitHub](https://github.com/brainglobe/bg-atlasapi). 

Someone might have already asked a question you might have, so if you're not sure where to start, check out the [issues](https://github.com/brainglobe/bg-atlasapi/issues) \(and the issues of the other repositories\)
