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

| Atlas Name | Resolution | Ages | Reference Images | Name in API
| --- |  --- | --- | --- | --- |
| [Allen Mouse Brain Atlas](https://doi.org/10.1016/j.cell.2020.04.007) | 10, 25, 50, and 100 micron | P56 | STPT | allen_mouse_10um, allen_mouse_25um, allen_mouse_100um
| [Allen Human Brain Atlas](https://www.brain-map.org) | 500 micron | Adult | MRI | allen_human_500um
| [Max Planck Zebrafish Brain Atlas](http://fishatlas.neuro.mpg.de) | 1 micron | 6-dpf | FISH | mpin_zfish_1um
| [Enhanced and Unified Mouse Brain Atlas](https://kimlab.io/brain-map/atlas/) | 10, 25, 50, and 100 micron | P56 | STPT | kim_mouse_10um, kim_mouse_25um, kim_mouse_50um, kim_mouse_100um
| [Smoothed version of the Kim et al. mouse reference atlas](https://doi.org/10.1016/j.celrep.2014.12.014) | 10, 25, 50 and 100 micron | P56 | STPT | osten_mouse_10um, osten_mouse_25um, osten_mouse_50um, osten_mouse_100um
| [Gubra's LSFM mouse brain atlas](https://doi.org/10.1007/s12021-020-09490-8) | 20 micron | 8 to 10 weeks post natal | LSFM | perens_lsfm_mouse_20um
| [3D version of the Allen mouse spinal cord atlas](https://doi.org/10.1101/2021.05.06.443008) | 20 x 10 x 10 micron | Adult | Nissl | allen_cord_20um
| [AZBA: A 3D Adult Zebrafish Brain Atlas](https://doi.org/10.1101/2021.05.04.442625) | 4 micron | 15-16 weeks post natal | LSFM | azba_zfish_4um
| [Waxholm Space atlas of the Sprague Dawley rat brain](https://doi.org/10.1038/s41592-023-02034-3) | 39 micron | P80  | MRI | whs_sd_rat_39um
| [3D Edge-Aware Refined Atlases Derived from the Allen Developing Mouse Brain Atlases](https://doi.org/10.7554/eLife.61408) | 16, 16.75, and 25 micron | E13, E15, E18, P4, P14, P28 & P56 | Nissl | admba_3d_e11_5_mouse_16um, admba_3d_e13_5_mouse_16um, admba_3d_e15_5_mouse_16um, admba_3d_e18_5_mouse_16um, admba_3d_p14_mouse_16.752um, admba_3d_p28_mouse_16.752um, admba_3d_p4_mouse_16.752um, admba_3d_p56_mouse_25um
| [Princeton Mouse Brain Atlas](https://brainmaps.princeton.edu/2020/09/princeton-mouse-brain-atlas-links) | 20 micron | >P56 (older animals included) | LSFM | princeton_mouse_20um
| [Kim Lab Developmental CCF](https://data.mendeley.com/datasets/2svx788ddf/1) | 10 micron | P56  | STP, LSFM (iDISCO) and MRI (a0, adc, dwo, fa, MTR, T2) | kim_dev_mouse_stp_10um, kim_dev_mouse_idisco_10um, kim_dev_mouse_mri_a0_10um, kim_dev_mouse_mri_adc_10um, kim_dev_mouse_mri_dwi_10um, kim_dev_mouse_mri_fa_10um, kim_dev_mouse_mri_mtr_10um, kim_dev_mouse_mri_t2_10um
| [Blind Mexican Cavefish Brain Atlas](https://doi.org/10.7554/eLife.80777) | 2 micron | 6 days post fertilisation | IHC | sju_cavefish_2um
| [BlueBrain Barrel Cortex Atlas](https://doi.org/10.1162/imag_a_00209) | 10 and 25 micron | P56 | STPT | allen_mouse_bluebrain_barrels_10um, allen_mouse_bluebrain_barrels_25um
| [UNAM Axolotl Brain Atlas](https://doi.org/10.1038/s41598-021-89357-3) | 40 micron | ~ 3 months post hatching | MRI | unam_axolotl_40um
| [Prairie Vole Brain Atlas](https://doi.org/10.7554/eLife.87029.3.sa0) | 25 micron | Unknown | LSFM | prairie_vole_25um

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

## Seeking help or contributing
We are always happy to help users of our tools, and welcome any contributions. If you would like to get in contact with us for any reason, please see the [contact page of our website](https://brainglobe.info/contact.html).

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
