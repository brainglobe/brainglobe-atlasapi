# BG-AtlasGen

Utilities and scripts for the generation of cleaned-up data for the `bg-atlasapi` module.


### To contribute
1) Fork this repo

2) Clone your repo 
```bash
git clone https://github.com/USERNAME/bg-atlasgen
```

3) Install an editable version
```bash
cd bg-atlasgen
pip install -e . 
```
4) Create a script to package your atlas, and place into 
`bg_atlasgen/atlas_scripts`. Please see other scripts for examples.

Your script should contain everything required to run. The raw data should be 
hosted on a publicly accessible repository so that anyone can run the script
 to recreate the atlas.

If you need to add any dependencies, please add them as an extra in the 
setup.py file, e.g.:

```python
extras_require={"allenmouse": ["allensdk"],
                "newatlas": ["dep1", "dep2"]}
```
