



Contents
========

* [**`bg_cli`** [#7]](#bg_cli-7)


&nbsp;

--------
# **`bg_cli`** [#7]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/cli.py#L7) online

```python
def bg_cli(command, atlas_name=None, force=False, show=False,
    key=None, value=None):
```

&nbsp;  
docstring:

```text
Command line dispatcher. Given a command line call to `brainglobe`

it calls the correct function, depending on which `command` was
    passed.

Arguments:

----------

command: str. Name of the command:

- list: list available atlases

- install: isntall new atlas

- update: update an installed atlas

- config: modify config

show: bool. If True when using `list` shows the local path of
    installed atlases

and when using 'config' it prints the modify config results.

atlas_name: ts. Used with `update` and `install`, name of the atlas to
    install

force: bool, used with `update`. If True it forces the update

```