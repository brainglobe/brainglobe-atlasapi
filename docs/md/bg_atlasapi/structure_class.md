



Contents
========

* [**Structure**](#structure)
	* [**`__getitem__`** [#11]](#__getitem__-11)
* [**StructuresDict**](#structuresdict)
	* [**`__init__`** [#39]](#__init__-39)
	* [**`__getitem__`** [#51]](#__getitem__-51)
	* [**`__repr__`** [#69]](#__repr__-69)


&nbsp;

--------
# **Structure**


```
Class implementing the lazy loading of a mesh if the dictionary is
queried for it.
```

&nbsp;
## **`__getitem__`** [#11]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_class.py#L11) online

```python
def __getitem__(self, item):
```

&nbsp;  
docstring:

no docstring

&nbsp;

--------
# **StructuresDict**


```
Class to handle dual indexing by either acronym or id.

Parameters
----------
mesh_path : str or Path object
    path to folder containing all meshes .obj files
```

&nbsp;
## **`__init__`** [#39]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_class.py#L39) online

```python
def __init__(self, structures_list):
```

&nbsp;  
docstring:

no docstring

&nbsp;
## **`__getitem__`** [#51]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_class.py#L51) online

```python
def __getitem__(self, item):
```

&nbsp;  
docstring:

```text
Core implementation of the class support for different indexing.

Parameters

----------

item :

Returns

-------

```

&nbsp;
## **`__repr__`** [#69]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_class.py#L69) online

```python
def __repr__(self):
```

&nbsp;  
docstring:

```text
String representation of the class, print all regions names

```