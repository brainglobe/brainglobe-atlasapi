



Contents
========

* [**`write_default_config`** [#19]](#write_default_config-19)
* [**`read_config`** [#40]](#read_config-40)
* [**`write_config_value`** [#64]](#write_config_value-64)
* [**`get_brainglobe_dir`** [#88]](#get_brainglobe_dir-88)
* [**`cli_modify_config`** [#100]](#cli_modify_config-100)
* [**`_print_config`** [#117]](#_print_config-117)


&nbsp;

--------
# **`write_default_config`** [#19]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L19) online

```python
def write_default_config(path=CONFIG_PATH,
    template=TEMPLATE_CONF_DICT):
```

&nbsp;  
docstring:

```text
Write configuration file at first repo usage. In this way,

we don't need to keep a confusing template config file in the repo.

Parameters

----------

path : Path object

Path of the config file (optional).

template : dict

Template of the config file to be written (optional).

```

&nbsp;

--------
# **`read_config`** [#40]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L40) online

```python
def read_config(path=CONFIG_PATH):
```

&nbsp;  
docstring:

```text
Read BrainGlobe config.

Parameters

----------

path : Path object

Path of the config file (optional).

Returns

-------

ConfigParser object

brainglobe configuration

```

&nbsp;

--------
# **`write_config_value`** [#64]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L64) online

```python
def write_config_value(key, val, path=CONFIG_PATH):
```

&nbsp;  
docstring:

```text
Write a new value in the config file. To make things simple, ignore

sections and look directly for matching parameters names.

Parameters

----------

key : str

Name of the parameter to configure.

val :

New value.

path : Path object

Path of the config file (optional).

```

&nbsp;

--------
# **`get_brainglobe_dir`** [#88]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L88) online

```python
def get_brainglobe_dir():
```

&nbsp;  
docstring:

```text
Return brainglobe default directory.

Returns

-------

Path object

default BrainGlobe directory with atlases

```

&nbsp;

--------
# **`cli_modify_config`** [#100]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L100) online

```python
def cli_modify_config(key=0, value=0, show=False):
```

&nbsp;  
docstring:

no docstring

&nbsp;

--------
# **`_print_config`** [#117]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/config.py#L117) online

```python
def _print_config():
```

&nbsp;  
docstring:

```text
Print configuration.

```