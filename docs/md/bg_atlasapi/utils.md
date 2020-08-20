



Contents
========

* [**`check_internet_connection`** [#14]](#check_internet_connection-14)
* [**`retrieve_over_http`** [#39]](#retrieve_over_http-39)
* [**`conf_from_url`** [#71]](#conf_from_url-71)
* [**`get_latest_atlases_version`** [#90]](#get_latest_atlases_version-90)
* [**`read_json`** [#98]](#read_json-98)
* [**`read_tiff`** [#104]](#read_tiff-104)


&nbsp;

--------
# **`check_internet_connection`** [#14]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L14) online

```python
def check_internet_connection(url='http://www.google.com/', timeout=5,
    raise_error=True):
```

&nbsp;  
docstring:

```text
Check that there is an internet connection

url : str

url to use for testing (Default value = 'http://www.google.com/')

timeout : int

timeout to wait for [in seconds] (Default value = 5).

raise_error : bool

if false, warning but no error.

```

&nbsp;

--------
# **`retrieve_over_http`** [#39]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L39) online

```python
def retrieve_over_http(url, output_file_path):
```

&nbsp;  
docstring:

```text
Download file from remote location, with progress bar.

Parameters

----------

url : str

Remote URL.

output_file_path : str or Path

Full file destination for download.

```

&nbsp;

--------
# **`conf_from_url`** [#71]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L71) online

```python
def conf_from_url(url):
```

&nbsp;  
docstring:

```text
Read conf file from an URL.

Parameters

----------

url : str

conf file url (in a repo, make sure the "raw" url is passed)

Returns

-------

conf object

```

&nbsp;

--------
# **`get_latest_atlases_version`** [#90]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L90) online

```python
def get_latest_atlases_version():
```

&nbsp;  
docstring:

no docstring

&nbsp;

--------
# **`read_json`** [#98]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L98) online

```python
def read_json(path):
```

&nbsp;  
docstring:

no docstring

&nbsp;

--------
# **`read_tiff`** [#104]
  
Check the [***``source code``***](https://github.com/brainglobe/bg-atlasapi/blob/master/bg_atlasapi/utils.py#L104) online

```python
def read_tiff(path):
```

&nbsp;  
docstring:

no docstring