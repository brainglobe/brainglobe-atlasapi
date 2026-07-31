"""Test the BrainGlobeAtlas class."""

import shutil
from types import SimpleNamespace
from unittest.mock import PropertyMock, patch

import pytest

import brainglobe_atlasapi
from brainglobe_atlasapi import bg_atlas
from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas


def test_versions(atlas):
    """Assert local and remote versions are equal."""
    assert atlas.local_version == atlas.remote_version


def test_remote_version_connection_error():
    """Test handling a connection error when fetching the remote version."""
    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=False
    ):
        atlas = object.__new__(BrainGlobeAtlas)
        atlas._remote_version = None
        assert atlas.remote_version is None


def test_remote_version_missing_from_assumed_root_raises_file_not_found():
    """A cached atlas whose assumed root doesn't host it fails cleanly.

    Regression test: root resolution is skipped when the atlas is already
    found in the local cache, so ``remote_version`` keeps whichever root
    was assumed (the default). If that root doesn't actually have this
    atlas name, ``fs.ls`` on the nonexistent S3 prefix returns ``[]``
    rather than raising, which previously fell through to
    ``get_latest_version([])`` and raised a bare ``IndexError``. This must
    surface as ``FileNotFoundError`` instead, since callers such as
    ``update_atlas`` depend on that type.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version = None
    atlas._requested_version = None
    atlas._remote_root = bg_atlas.descriptors.DEFAULT_REMOTE_ROOT
    atlas.atlas_name = "not_actually_here"
    atlas.fs = SimpleNamespace(ls=lambda path: [])

    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    ):
        with pytest.raises(
            FileNotFoundError, match="is not a valid atlas name!"
        ):
            atlas.remote_version


@pytest.mark.parametrize(
    "local_version, remote_version, expected",
    [
        pytest.param((1, 0), (2, 0), False, id="local < remote"),
        pytest.param((1, 0), (1, 0), True, id="local = remote"),
        pytest.param((1, 0), None, None, id="no remote version"),
    ],
)
def test_check_latest_version_local(local_version, remote_version, expected):
    """Test `check_latest_version` method.

    Parameters
    ----------
    local_version : tuple
        The mock local version.
    remote_version : tuple
        The mock remote version.
    expected : bool or None
        The expected result of `check_latest_version`.
    """
    with (
        patch.object(
            BrainGlobeAtlas, "local_version", new_callable=PropertyMock
        ) as mock_local_version,
        patch.object(
            BrainGlobeAtlas, "remote_version", new_callable=PropertyMock
        ) as mock_remote_version,
    ):
        mock_local_version.return_value = local_version
        mock_remote_version.return_value = remote_version
        atlas = object.__new__(BrainGlobeAtlas)
        assert atlas.check_latest_version(print_warning=False) == expected


@pytest.mark.parametrize(
    "atlas_name, expected_repr",
    [
        pytest.param(
            "nadkarni_mri_mouselemur_91um",
            "nadkarni mri mouselemur atlas (res. 91um)",
            id="nadkarni_mri_mouselemur_91um",
        ),
        pytest.param(
            "example_mouse_100um",
            "example mouse atlas (res. 100um)",
            id="example_mouse_100um",
        ),
        pytest.param(
            "axolotl_50um", "axolotl atlas (res. 50um)", id="axolotl_50um"
        ),
    ],
)
def test_repr(atlas_name, expected_repr):
    """Test `BrainGlobeAtlas` `__repr__` method.

    Parameters
    ----------
    atlas_name : str
        The name of the atlas.
    expected_repr : str
        The expected string representation.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    assert repr(atlas) == expected_repr


def test_str(atlas, capsys):
    """Test `BrainGlobeAtlas` `__str__` method."""
    print(atlas)
    captured = capsys.readouterr()
    expected_doi = "https://doi.org/10.1016/j.cell.2020.04.007"
    assert expected_doi in captured.out
    assert captured.err == ""


def test_local_search(tmpdir):
    """Test local atlas search and handling of multiple versions."""
    atlas_name = "example_mouse_100um"
    temp_brainglobe_dir = tmpdir.mkdir("brainglobe")

    atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=temp_brainglobe_dir,
    )

    assert atlas.atlas_name in atlas.local_full_name

    # Make a copy:
    new_version = "4_5"
    old_version = "_".join(str(v) for v in atlas.local_version)
    atlas_path = atlas.local_full_name.split("/")
    atlas_root = "/".join(atlas_path[:-2])
    new_path = atlas.root_dir / atlas_root / new_version
    old_path = atlas.root_dir / atlas_root / old_version
    shutil.copytree(old_path, new_path)

    # Should find the latest version:
    atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=temp_brainglobe_dir,
    )
    assert atlas.local_full_name == f"{atlas_root}/{new_version}/manifest.json"


@pytest.mark.parametrize(
    "version_str, expected",
    [
        ("3.0", (3, 0)),
        ("3_0", (3, 0)),
        ("2017", (2017,)),
        ("2024-05", (2024, 5)),
    ],
)
def test_version_tuple_from_str_separators(version_str, expected):
    """Version strings parse across ``.``, ``_`` and ``-`` separators.

    Parameters
    ----------
    version_str : str
        Version string as stored in a manifest or folder name.
    expected : tuple of int
        Expected numeric tuple.
    """
    assert bg_atlas._version_tuple_from_str(version_str) == expected


@pytest.mark.parametrize(
    "atlas_name, resolution, expected",
    [
        (
            "example_mouse_100um",
            [100.0, 100.0, 100.0],
            "example mouse atlas (res. 100um)",
        ),
        (
            "allen-adult-mouse-ccf-atlas",
            [25.0, 25.0, 25.0],
            "allen-adult-mouse-ccf-atlas (res. 25.0um)",
        ),
    ],
)
def test_repr_handles_names_without_underscore(
    atlas_name, resolution, expected
):
    """Names with no underscore keep their full name in the repr.

    Parameters
    ----------
    atlas_name : str
        Atlas name under test.
    resolution : list of float
        Resolution recorded in the metadata.
    expected : str
        Expected repr text.
    """
    instance = BrainGlobeAtlas.__new__(BrainGlobeAtlas)
    instance.atlas_name = atlas_name
    instance.metadata = {"resolution": resolution}
    assert repr(instance) == expected


class _FakeFS:
    """Filesystem stub that reports a fixed set of existing paths."""

    def __init__(self, existing):
        self.existing = set(existing)

    def exists(self, path):
        return path in self.existing


def test_resolve_root_prefers_declaration_order():
    """A name present in both roots resolves to the first declared."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS(
        [
            "s3://brainglobe/atlas-rc2/atlases/shared_name",
            "s3://other/prefix/atlases/shared_name",
        ]
    )
    assert bg_atlas._resolve_remote_root(fs, "shared_name", roots) == (
        "brainglobe-atlasapi",
        "s3://brainglobe/atlas-rc2",
    )


def test_resolve_root_falls_through_to_later_root():
    """A name only in a later root resolves to that root."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS(["s3://other/prefix/atlases/allen-adult-mouse-ccf-atlas"])
    assert bg_atlas._resolve_remote_root(
        fs, "allen-adult-mouse-ccf-atlas", roots
    ) == ("allen", "s3://other/prefix")


def test_resolve_root_unknown_name_lists_roots():
    """An unknown name reports every root that was searched."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    with pytest.raises(ValueError) as error:
        bg_atlas._resolve_remote_root(_FakeFS([]), "nope", roots)
    message = str(error.value)
    assert "nope" in message
    assert "s3://brainglobe/atlas-rc2" in message
    assert "s3://other/prefix" in message


def test_default_root_cache_path_is_unchanged(atlas):
    """The default root keeps the historical cache directory name.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture.
    """
    assert atlas.brainglobe_dir.name == "brainglobe-atlasapi"
    assert atlas.root_dir == atlas.brainglobe_dir


def test_resolve_non_default_root_switches_cache_dir(tmp_path, monkeypatch):
    """Resolving to a non-default root swaps the on-disk cache directory.

    Root resolution is deferred until the atlas is not found under the
    default root's local cache, so this stubs out every network-facing
    call to exercise the directory swap without touching S3.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to stub S3 access.
    """

    class _StubFS:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self, path):
            return False

        def ls(self, path):
            raise FileNotFoundError(path)

    monkeypatch.setattr(bg_atlas.s3fs, "S3FileSystem", _StubFS)
    monkeypatch.setattr(bg_atlas, "check_s3_status", lambda **_: True)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remote_roots",
        lambda *_: {
            "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
            "allen": "s3://other/prefix",
        },
    )
    monkeypatch.setattr(
        bg_atlas,
        "_resolve_remote_root",
        lambda fs, name, roots: ("allen", "s3://other/prefix"),
    )

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas("not_cached_anywhere", brainglobe_dir=tmp_path)

    assert (tmp_path / "allen").is_dir()
