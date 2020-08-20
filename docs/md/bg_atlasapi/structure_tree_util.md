



Contents
========

* [**`child_ids`** [#6]](#child_ids-6)
* [**`get_structures_tree`** [#15]](#get_structures_tree-15)
* [**`add_descendants_to_tree`** [#19]](#add_descendants_to_tree-19)


&nbsp;

--------
# **`child_ids`** [#6]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_tree_util.py#L6) online

```python
def child_ids(structure, structure_list):
```

&nbsp;  
docstring:

no docstring

&nbsp;

--------
# **`get_structures_tree`** [#15]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_tree_util.py#L15) online

```python
def get_structures_tree(structures_list):
```

&nbsp;  
docstring:

```text
Creates a 'tree' graph with the hierarchical organisation of all
    structures

```

&nbsp;

--------
# **`add_descendants_to_tree`** [#19]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/structure_tree_util.py#L19) online

```python
def add_descendants_to_tree(structures_list, id_to_acronym_map, tree,
    structure_id, parent_id):
```

&nbsp;  
docstring:

```text
Recursively goes through all the the descendants of a region and adds
    them to the tree

```