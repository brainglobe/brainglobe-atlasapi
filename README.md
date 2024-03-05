# brainglobe-atlasapi

[![Python Version](https://img.shields.io/pypi/pyversions/brainglobe-atlasapi.svg)](https://pypi.org/project/brainglobe-atlasapi)
[![PyPI](https://img.shields.io/pypi/v/brainglobe-atlasapi.svg)](https://pypi.org/project/brainglobe-atlasapi/)
[![Wheel](https://img.shields.io/pypi/wheel/brainglobe-atlasapi.svg)](https://pypi.org/project/brainglobe-atlasapi)
[![Development Status](https://img.shields.io/pypi/status/brainatlas-api.svg)](https://github.com/SainsburyWellcomeCentre/brainatlas-api)
[![Downloads](https://pepy.tech/badge/brainglobe-atlasapi)](https://pepy.tech/project/brainglobe-atlasapi)
[![Tests](https://img.shields.io/github/actions/workflow/status/brainglobe/brainglobe-atlasapi/test_and_deploy.yml?branch=main)](
    https://github.com/brainglobe/brainglobe-atlasapi/actions)
[![codecov](https://codecov.io/gh/brainglobe/brainglobe-atlasapi/branch/master/graph/badge.svg?token=WTFPFW0TE4)](https://codecov.io/gh/brainglobe/brainglobe-atlasapi)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.02668/status.svg)](https://doi.org/10.21105/joss.02668)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)](https://docs.brainglobe.info/cellfinder/contributing)
[![Website](https://img.shields.io/website?up_message=online&url=https%3A%2F%2Fbrainglobe.info)](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html)
[![Twitter](https://img.shields.io/twitter/follow/brain_globe?style=social)](https://twitter.com/brain_globe)

The brainglobe atlas API (brainglobe-atlasapi) provides a common interface for programmers to download and process brain atlas data from multiple sources.

## Atlases available

A number of atlases are in development, but those available currently are:

- [Allen Mouse Brain Atlas](https://doi.org/10.1016/j.cell.2020.04.007) at 10, 25, 50 and 100 micron resolutions
- [Allen Human Brain Atlas](https://www.brain-map.org) at 100 micron resolution
- [Max Planck Zebrafish Brain Atlas](http://fishatlas.neuro.mpg.de) at 1 micron resolution
- [Enhanced and Unified Mouse Brain Atlas](https://kimlab.io/brain-map/atlas/) at 10, 25, 50 and 100 micron resolutions
- [Smoothed version of the Kim et al. mouse reference atlas](https://doi.org/10.1016/j.celrep.2014.12.014) at 10, 25, 50 and 100 micron resolutions
- [Gubra's LSFM mouse brain atlas](https://doi.org/10.1007/s12021-020-09490-8) at 20 micron resolution
- [3D version of the Allen mouse spinal cord atlas](https://doi.org/10.1101/2021.05.06.443008) at 20 x 10 x 10 micron resolution
- [AZBA: A 3D Adult Zebrafish Brain Atlas](https://doi.org/10.1101/2021.05.04.442625) at 4 micron resolution
- [Waxholm Space atlas of the Sprague Dawley rat brain](https://doi.org/10.1038/s41592-023-02034-3) at 39 micron resolution
- [3D Edge-Aware Refined Atlases Derived from the Allen Developing Mouse Brain Atlases](https://doi.org/10.7554/eLife.61408) (E13, E15, E18, P4, P14, P28 & P56)
- [Princeton Mouse Brain Atlas](https://brainmaps.princeton.edu/2020/09/princeton-mouse-brain-atlas-links) at 20 micron resolution
- [Kim Lab Developmental CCF (P56)](https://data.mendeley.com/datasets/2svx788ddf/1) at 10 micron resolution with 8 reference images - STP, LSFM (iDISCO) and MRI (a0, adc, dwo, fa, MTR, T2)

## Installation

brainglobe-atlasapi works with Python >3.6, and can be installed from PyPI with:

```bash
pip install brainglobe-atlasapi
```

## Usage

Full information can be found in the [documentation](https://brainglobe.info/documentation/brainglobe-atlasapi/index.html)

### Python API

#### List of atlases

To see a list of atlases use `brainglobe_atlasapi.show_atlases`

```python
from brainglobe_atlasapi import show_atlases
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

#### Using the atlases

All the features of each atlas can be accessed via the `BrainGlobeAtlas` class.

e.g. for the 25um Allen Mouse Brain Atlas:

```python
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas
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

#### Brain regions

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

### Note on coordinates in `brainglobe-atlasapi`

Working with both image coordinates and cartesian coordinates in the same space can be confusing!
In `brainglobe-atlasapi`, the origin is always assumed to be in the upper left corner of the image (sectioning along the first dimension), the "ij" convention.
This means that when plotting meshes and points using cartesian systems, you might encounter confusing behaviors coming from the fact that in cartesian plots one axis is inverted with respect to  ij coordinates (vertical axis increases going up, image row indexes increase going down).
To make things as consistent as possible, in `brainglobe-atlasapi` the 0 of the meshes coordinates is assumed to coincide with the 0 index of the images stack, and meshes coordinates increase following the direction stack indexes increase.
To deal with transformations between your data space and `brainglobe-atlasapi`, you might find the [brainglobe-space](https://github.com/brainglobe/brainglobe-space) package helpful.

## Contributing to brainglobe-atlasapi

**Contributors to bg-atlaspi are absolutely encouraged**, whether you want to fix bugs, add/request new features or simply ask questions.

If you would like to contribute to `brainglobe-atlasapi` (or any of the downstream tools like [brainrender](https://github.com/brainglobe/brainrender) etc.) please get in touch by opening a new issue or pull request on [GitHub](https://github.com/brainglobe/brainglobe-atlasapi).
Please also see the [developers guide](https://brainglobe.info/community/developers/index.html).

Someone might have already asked a question you might have, so if you're not sure where to start, check out the [issues](https://github.com/brainglobe/brainglobe-atlasapi/issues) (and the issues of the other repositories).

## Citation

If you find the BrainGlobe Atlas API useful, please cite the paper in your work:

>Claudi, F., Petrucco, L., Tyson, A. L., Branco, T., Margrie, T. W. and Portugues, R. (2020). BrainGlobe Atlas API: a common interface for neuroanatomical atlases. Journal of Open Source Software, 5(54), 2668, https://doi.org/10.21105/joss.02668

**Don't forget to cite the developers of the atlas that you used!**

---

# Atlas Generation and Adding a New Atlas

For full instructions to add a new BrainGlobe atlas, please see [here](https://brainglobe.info/documentation/brainglobe-atlasapi/adding-a-new-atlas.html).

The `brainglobe_atlasapi.atlas_generation` submodule contains code for the generation of cleaned-up data, for the main `brainglobe_atlasapi` module.
This code was previously the `bg-atlasgen` module.

## To contribute

1. Fork this repo
2. Clone your repo
3. Run `git clone https://github.com/brainglobe/brainglobe-atlasapi`
4. Install an editable version of the package; by running `pip install -e .` within the cloned directory
5. Create a script to package your atlas, and place into `brainglobe_atlasapi/atlas_generation/atlas_scripts`. Please see other scripts for examples.

Your script should contain everything required to run.
The raw data should be hosted on a publicly accessible repository so that anyone can run the script to recreate the atlas.

If you need to add any dependencies, please add them as an extra in the `pyproject.toml` file, e.g.:

```python
[project.optional-dependencies]
allenmouse = ["allensdk"]
newatlas = ["dependency_1", "dependency_2"]
```
