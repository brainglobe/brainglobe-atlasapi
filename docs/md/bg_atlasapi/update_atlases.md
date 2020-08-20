



Contents
========

* [**`update_atlas`** [#8]](#update_atlas-8)
* [**`install_atlas`** [#53]](#install_atlas-53)


&nbsp;

--------
# **`update_atlas`** [#8]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/update_atlases.py#L8) online

```python
def update_atlas(atlas_name, force=False):
```

&nbsp;  
docstring:

```text
Updates a bg_atlasapi atlas from the latest

available version online.

Arguments:

----------

atlas_name: str

Name of the atlas to update.

force: bool

If False it checks if the atlas is already at the latest version

and doesn't update if that's the case.

```

&nbsp;

--------
# **`install_atlas`** [#53]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/update_atlases.py#L53) online

```python
def install_atlas(atlas_name):
```

&nbsp;  
docstring:

```text
Installs a BrainGlobe atlas from the latest

available version online.

Arguments

---------

atlas_name : str

Name of the atlas to update.

```