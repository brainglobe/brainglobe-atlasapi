



Contents
========

* [**Atlas**](#atlas)
	* [**`__init__`** [#28]](#__init__-28)
	* [**`resolution`** [#57]](#resolution-57)
	* [**`hierarchy`** [#63]](#hierarchy-63)
	* [**`lookup_df`** [#71]](#lookup_df-71)
	* [**`reference`** [#85]](#reference-85)
	* [**`annotation`** [#91]](#annotation-91)
	* [**`hemispheres`** [#97]](#hemispheres-97)
	* [**`hemisphere_from_coords`** [#122]](#hemisphere_from_coords-122)
	* [**`structure_from_coords`** [#148]](#structure_from_coords-148)
	* [**`_get_from_structure`** [#183]](#_get_from_structure-183)
	* [**`mesh_from_structure`** [#205]](#mesh_from_structure-205)
	* [**`meshfile_from_structure`** [#208]](#meshfile_from_structure-208)
	* [**`root_mesh`** [#211]](#root_mesh-211)
	* [**`root_meshfile`** [#214]](#root_meshfile-214)
	* [**`_idx_from_coords`** [#217]](#_idx_from_coords-217)
	* [**`get_structure_ancestors`** [#224]](#get_structure_ancestors-224)
	* [**`get_structure_descendants`** [#235]](#get_structure_descendants-235)


&nbsp;

--------
# **Atlas**


```
Base class to handle atlases in BrainGlobe.

Parameters
----------
path : str or Path object
    path to folder containing data info.
```

&nbsp;
## **`__init__`** [#28]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L28) online

```python
def __init__(self, path):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`resolution`** [#57]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L57) online

```python
def resolution(self):
```

&nbsp;  
docstring:

```text
Make resolution more accessible from class.

```

&nbsp;
## **`hierarchy`** [#63]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L63) online

```python
def hierarchy(self):
```

&nbsp;  
docstring:

```text
Returns a Treelib.tree object with structures hierarchy.

```

&nbsp;
## **`lookup_df`** [#71]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L71) online

```python
def lookup_df(self):
```

&nbsp;  
docstring:

```text
Returns a dataframe with id, acronym and name for each structure.

```

&nbsp;
## **`reference`** [#85]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L85) online

```python
def reference(self):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`annotation`** [#91]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L91) online

```python
def annotation(self):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`hemispheres`** [#97]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L97) online

```python
def hemispheres(self):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`hemisphere_from_coords`** [#122]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L122) online

```python
def hemisphere_from_coords(self, coords, microns=False,
    as_string=False):
```

&nbsp;  
docstring:

```text
Get the hemisphere from a coordinate triplet.

Parameters

----------

coords : tuple or list or numpy array

Triplet of coordinates. Default in voxels, can be microns if

microns=True

microns : bool

If true, coordinates are interpreted in microns.

as_string : bool

If true, returns "left" or "right".

Returns

-------

int or string

Hemisphere label.

```

&nbsp;
## **`structure_from_coords`** [#148]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L148) online

```python
def structure_from_coords(self, coords, microns=False,
    as_acronym=False, hierarchy_lev=None):
```

&nbsp;  
docstring:

```text
Get the structure from a coordinate triplet.

Parameters

----------

coords : tuple or list or numpy array

Triplet of coordinates.

microns : bool

If true, coordinates are interpreted in microns.

as_acronym : bool

If true, the region acronym is returned.

hierarchy_lev : int or None

If specified, return parent node at thi hierarchy level.

Returns

-------

int or string

Structure containing the coordinates.

```

&nbsp;
## **`_get_from_structure`** [#183]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L183) online

```python
def _get_from_structure(self, structure, key):
```

&nbsp;  
docstring:

```text
Internal interface to the structure dict. It support querying with a

single structure id or a list of ids.

Parameters

----------

structure : int or str or list

Valid id or acronym, or list if ids or acronyms.

key : str

Key for the Structure dictionary (eg "name" or "rgb_triplet").

Returns

-------

value or list of values

If structure is a list, returns list.

```

&nbsp;
## **`mesh_from_structure`** [#205]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L205) online

```python
def mesh_from_structure(self, structure):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`meshfile_from_structure`** [#208]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L208) online

```python
def meshfile_from_structure(self, structure):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`root_mesh`** [#211]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L211) online

```python
def root_mesh(self):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`root_meshfile`** [#214]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L214) online

```python
def root_meshfile(self):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`_idx_from_coords`** [#217]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L217) online

```python
def _idx_from_coords(self, coords, microns):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`get_structure_ancestors`** [#224]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L224) online

```python
def get_structure_ancestors(self, structure):
```

&nbsp;  
docstring:

```text
Returns a list of acronyms for all

ancestors of a given structure

```

&nbsp;
## **`get_structure_descendants`** [#235]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/core.py#L235) online

```python
def get_structure_descendants(self, structure):
```

&nbsp;  
docstring:

```text
Returns a list of acronyms for all

descendants of a given structure

```