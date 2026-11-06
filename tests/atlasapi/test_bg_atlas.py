"""Test the BrainGlobeAtlas class."""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

import brainglobe_atlasapi
from brainglobe_atlasapi import bg_atlas, descriptors
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
        atlas._remote_root = bg_atlas.descriptors.DEFAULT_REMOTE_ROOT
        atlas._reachable_roots = set()
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
    atlas._remote_root = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE_ROOT
    atlas._reachable_roots = set()
    atlas.atlas_name = "not_actually_here"
    atlas.fs = SimpleNamespace(exists=lambda path: True, ls=lambda path: [])

    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    ):
        with pytest.raises(
            FileNotFoundError, match="is not a valid atlas name!"
        ):
            atlas.remote_version


def test_remote_version_carries_dashed_folder_string():
    """A dashed atlas-assets folder name is carried through verbatim.

    Regression test: ``remote_version`` must record the resolved S3
    folder name in ``_remote_version_str`` as-is, not reconstruct it
    from the parsed tuple: rebuilding it from ``(2024, 5)`` would yield
    ``"2024_5"``, which is not the real folder ``"2024-05"``.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version = None
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas._remote_root = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE_ROOT
    atlas._reachable_roots = set()
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas.fs = SimpleNamespace(
        exists=lambda path: True,
        ls=lambda path: [f"{path}/2017", f"{path}/2024-05"],
    )

    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    ):
        assert atlas.remote_version == (2024, 5)

    assert atlas._remote_version_str == "2024-05"


def test_remote_version_ignores_non_version_entries():
    """Stray files next to the version folders are not parsed as versions."""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version = None
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas._remote_root = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE_ROOT
    atlas._reachable_roots = set()
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas.fs = SimpleNamespace(
        ls=lambda path: [f"{path}/2017", f"{path}/README.md", f"{path}/tmp"],
    )

    with patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    ):
        assert atlas.remote_version == (2017,)


def _atlas_for_remote_version(
    atlas_name, brainglobe_dir, local_full_name=None
):
    """Create a bare BrainGlobeAtlas with a mocked filesystem."""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    atlas.brainglobe_dir = brainglobe_dir
    atlas._remote_root = bg_atlas.descriptors.DEFAULT_REMOTE_ROOT
    atlas._reachable_roots = set()
    atlas._remote_version = None
    atlas._requested_version = None
    atlas._local_full_name = local_full_name
    atlas.fs = MagicMock()
    atlas.fs.exists.return_value = True
    # An in-progress version is present on the remote
    atlas.fs.ls.return_value = [
        f"brainglobe/atlas/atlases/{atlas_name}/1_0",
        f"brainglobe/atlas/atlases/{atlas_name}/2_0",
    ]
    return atlas


def test_remote_version_uses_last_versions_conf(tmp_path):
    """Test the latest remote version comes from last_versions.conf,
    ignoring newer versions on the remote that are not yet released.
    """
    atlas = _atlas_for_remote_version("example_mouse_100um", tmp_path)
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={"example_mouse_100um": "1.0"},
        ),
    ):
        assert atlas.remote_version == (1, 0)


def test_remote_version_unreleased_atlas(tmp_path):
    """Test an atlas absent from last_versions.conf is not fetched."""
    atlas = _atlas_for_remote_version("example_mouse_100um", tmp_path)
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={},
        ),
    ):
        with pytest.raises(FileNotFoundError, match="no released version"):
            atlas.remote_version


def test_remote_version_unreleased_atlas_local(tmp_path):
    """Test an atlas absent from last_versions.conf but available locally
    returns no remote version.
    """
    atlas = _atlas_for_remote_version(
        "example_mouse_100um",
        tmp_path,
        local_full_name="atlases/example_mouse_100um/2_0/manifest.json",
    )
    with (
        patch.object(
            brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
        ),
        patch.object(
            brainglobe_atlasapi.bg_atlas,
            "get_all_atlases_lastversions",
            return_value={},
        ),
    ):
        assert atlas.remote_version is None


@pytest.mark.parametrize(
    "local_folder, remote_folder, expected",
    [
        pytest.param("1_0", "2_0", False, id="local < remote"),
        pytest.param("1_0", "1_0", True, id="local = remote"),
        pytest.param("2024-05", "2024_05", False, id="separators differ"),
        pytest.param("1_0", None, None, id="no remote version"),
    ],
)
def test_check_latest_version_local(local_folder, remote_folder, expected):
    """Test `check_latest_version` compares version folder names exactly.

    Parameters
    ----------
    local_folder : str
        The local version folder name.
    remote_folder : str or None
        The remote version folder name, or None when offline.
    expected : bool or None
        The expected result of `check_latest_version`.
    """
    remote_version = (
        None
        if remote_folder is None
        else bg_atlas.version_tuple_from_str(remote_folder)
    )
    with patch.object(
        BrainGlobeAtlas,
        "remote_version",
        new_callable=PropertyMock,
        return_value=remote_version,
    ):
        atlas = object.__new__(BrainGlobeAtlas)
        atlas._local_version_str = local_folder
        atlas._remote_version_str = remote_folder
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


def test_local_full_name_matches_dashed_version_folders(tmp_path):
    r"""The "latest cached" branch must see dashed version folders too.

    Regression test: the discovery regex in ``local_full_name`` only
    accepted ``\d+(?:_\d+)?``, which matches BrainGlobe folders like
    ``3_0``/``2017`` but not atlas-assets folders like ``2024-05``. That
    made a dashed-version atlas invisible to local-cache discovery even
    though its manifest was on disk.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._local_full_name = None
    atlas._local_version_str = None
    atlas._requested_version = None
    atlas.atlas_name = "dashed_version_atlas"
    atlas.brainglobe_dir = tmp_path

    atlas_dir = tmp_path / bg_atlas.V3_ATLAS_ROOTDIR / atlas.atlas_name
    for version in ("2024-05", "2026-03"):
        version_dir = atlas_dir / version
        version_dir.mkdir(parents=True)
        (version_dir / "manifest.json").write_text("{}")

    assert atlas.local_full_name == (
        f"{bg_atlas.V3_ATLAS_ROOTDIR}/{atlas.atlas_name}/2026-03/manifest.json"
    )
    assert atlas._local_version_str == "2026-03"


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

    def ls(self, path):
        if path not in self.existing:
            raise FileNotFoundError(path)
        return [f"{path}/1_0"]


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
    assert (
        bg_atlas._resolve_remote_root(fs, "shared_name", roots, lambda _: True)
        == "s3://brainglobe/atlas-rc2"
    )


def test_resolve_root_falls_through_to_later_root():
    """A name only in a later root resolves to that root."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS(["s3://other/prefix/atlases/allen-adult-mouse-ccf-atlas"])
    assert (
        bg_atlas._resolve_remote_root(
            fs, "allen-adult-mouse-ccf-atlas", roots, lambda _: True
        )
        == "s3://other/prefix"
    )


def test_resolve_root_unknown_name_lists_roots():
    """An unknown name reports every root that was searched."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    with pytest.raises(ValueError) as error:
        bg_atlas._resolve_remote_root(
            _FakeFS([]), "nope", roots, lambda _: True
        )
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
        lambda fs, name, roots, is_reachable: "s3://other/prefix",
    )

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas("not_cached_anywhere", brainglobe_dir=tmp_path)

    assert (tmp_path / "prefix").is_dir()


def test_atlas_cached_under_other_root_loads_offline(tmp_path, monkeypatch):
    """A copy cached under a non-default root is found without S3.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to forbid S3 access and skip loading the stub atlas.
    """
    manifest = tmp_path / "prefix" / "atlases" / "cached_atlas" / "1_0"
    manifest.mkdir(parents=True)
    (manifest / "manifest.json").write_text("{}")

    def fail(*args, **kwargs):
        raise AssertionError("no network call should be made")

    monkeypatch.setattr(bg_atlas, "check_s3_status", fail)
    monkeypatch.setattr(bg_atlas, "_resolve_remote_root", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remote_roots",
        lambda *_: {
            "brainglobe-atlasapi": "s3://brainglobe/atlas",
            "allen": "s3://other/prefix",
        },
    )
    monkeypatch.setattr(bg_atlas, "_normalize_manifest", lambda *_: {})
    monkeypatch.setattr(
        bg_atlas.core.Atlas, "__init__", lambda self, *a, **k: None
    )

    atlas = BrainGlobeAtlas(
        "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
    )

    assert atlas._remote_root == "s3://other/prefix"
    assert atlas.brainglobe_dir == tmp_path / "prefix"
    record = tmp_path / "prefix" / descriptors.REMOTE_ROOT_FILENAME
    assert record.read_text() == "s3://other/prefix\n"
    assert atlas.local_full_name == "atlases/cached_atlas/1_0/manifest.json"


def test_atlas_cached_under_unconfigured_root_loads_offline(
    tmp_path, monkeypatch
):
    """A cache whose root is only recorded, not configured, is found.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to forbid S3 access and skip loading the stub atlas.
    """
    cache = tmp_path / "custom"
    manifest = cache / "atlases" / "cached_atlas" / "1_0"
    manifest.mkdir(parents=True)
    (manifest / "manifest.json").write_text("{}")
    (cache / descriptors.REMOTE_ROOT_FILENAME).write_text(
        "s3://bucket/custom\n"
    )

    def fail(*args, **kwargs):
        raise AssertionError("no network call should be made")

    monkeypatch.setattr(bg_atlas, "check_s3_status", fail)
    monkeypatch.setattr(bg_atlas, "_resolve_remote_root", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remote_roots",
        lambda *_: {"brainglobe-atlasapi": "s3://brainglobe/atlas"},
    )
    monkeypatch.setattr(bg_atlas, "_normalize_manifest", lambda *_: {})
    monkeypatch.setattr(
        bg_atlas.core.Atlas, "__init__", lambda self, *a, **k: None
    )

    atlas = BrainGlobeAtlas(
        "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
    )

    assert atlas._remote_root == "s3://bucket/custom"
    assert atlas.brainglobe_dir == cache


def test_resolve_root_skips_unreachable_root():
    """A root whose bucket is down does not stop the others being searched."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS(
        [
            "s3://brainglobe/atlas/atlases/shared_name",
            "s3://other/prefix/atlases/shared_name",
        ]
    )
    assert (
        bg_atlas._resolve_remote_root(
            fs, "shared_name", roots, lambda r: r == "s3://other/prefix"
        )
        == "s3://other/prefix"
    )


def test_resolve_root_probes_lazily():
    """Roots after the one holding the atlas are never probed."""
    roots = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas",
        "allen": "s3://other/prefix",
    }
    probed = []

    def is_reachable(remote_root):
        probed.append(remote_root)
        return True

    fs = _FakeFS(["s3://brainglobe/atlas/atlases/shared_name"])
    bg_atlas._resolve_remote_root(fs, "shared_name", roots, is_reachable)
    assert probed == ["s3://brainglobe/atlas"]


def test_resolve_root_with_nothing_reachable_returns_none():
    """Offline, resolution defers to the caller rather than raising."""
    roots = {"brainglobe-atlasapi": "s3://brainglobe/atlas"}
    assert (
        bg_atlas._resolve_remote_root(
            _FakeFS([]), "shared_name", roots, lambda _: False
        )
        is None
    )


def test_remote_root_override_skips_config_search(tmp_path, monkeypatch):
    """An explicit remote_root takes full priority over configured roots.

    It must be the only root searched: the config/multi-root resolution
    is skipped entirely, and its local cache directory is named after the
    last URL segment rather than any configured root key.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to stub S3 access and spy on the config/resolution calls.
    """

    class _StubFS:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self, path):
            return False

        def ls(self, path):
            raise FileNotFoundError(path)

    resolve_spy = MagicMock()
    roots_spy = MagicMock()

    monkeypatch.setattr(bg_atlas.s3fs, "S3FileSystem", _StubFS)
    monkeypatch.setattr(bg_atlas, "check_s3_status", lambda **_: True)
    monkeypatch.setattr(bg_atlas, "_resolve_remote_root", resolve_spy)
    monkeypatch.setattr(bg_atlas.config, "get_remote_roots", roots_spy)

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas(
            "not_cached_anywhere",
            brainglobe_dir=tmp_path,
            remote_root="s3://custom-bucket/prefix",
        )

    resolve_spy.assert_not_called()
    roots_spy.assert_not_called()
    assert (tmp_path / "prefix").is_dir()


@pytest.mark.parametrize(
    ["remote_root", "expected"],
    [
        ("s3://brainglobe/atlas", "brainglobe-atlasapi"),
        ("s3://allen-atlas-assets", "allen-atlas-assets"),
        ("s3://brainglobe/atlas-rc2", "atlas-rc2"),
    ],
)
def test_cache_key_is_last_url_segment(remote_root, expected):
    """Only the BrainGlobe root maps to brainglobe-atlasapi.

    Parameters
    ----------
    remote_root : str
        Remote root under test.
    expected : str
        Expected cache directory name.
    """
    assert bg_atlas._cache_key(remote_root) == expected


def test_cache_key_refuses_the_default_directory():
    """Another root ending in brainglobe-atlasapi cannot take its place."""
    with pytest.raises(ValueError, match="reserved"):
        bg_atlas._cache_key("s3://someone/brainglobe-atlasapi")


def test_explicit_default_root_uses_default_cache(tmp_path, monkeypatch):
    """Passing the BrainGlobe root explicitly reuses the default cache.

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

        def ls(self, path):
            raise FileNotFoundError(path)

    monkeypatch.setattr(bg_atlas.s3fs, "S3FileSystem", _StubFS)
    monkeypatch.setattr(bg_atlas, "check_s3_status", lambda **_: True)

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas(
            "not_cached_anywhere",
            brainglobe_dir=tmp_path,
            remote_root="s3://brainglobe/atlas/",
        )

    record = (
        tmp_path / "brainglobe-atlasapi" / descriptors.REMOTE_ROOT_FILENAME
    )
    # A failed lookup must not claim the cache directory.
    assert not record.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["brainglobe-atlasapi"]


def test_cache_recording_another_root_raises(tmp_path, monkeypatch):
    """A cache directory recording a different root is never reused.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to forbid S3 access.
    """
    manifest = tmp_path / "prefix" / "atlases" / "cached_atlas" / "1_0"
    manifest.mkdir(parents=True)
    (manifest / "manifest.json").write_text("{}")
    record = tmp_path / "prefix" / descriptors.REMOTE_ROOT_FILENAME
    record.write_text("s3://somebody-else/prefix\n")

    def fail(*args, **kwargs):
        raise AssertionError("no network call should be made")

    monkeypatch.setattr(bg_atlas, "check_s3_status", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remote_roots",
        lambda *_: {"allen": "s3://other/prefix"},
    )

    with pytest.raises(ValueError, match="s3://somebody-else/prefix"):
        BrainGlobeAtlas(
            "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
        )


def test_check_latest_version_compares_folder_strings(atlas, monkeypatch):
    """Dashed versions compare and print without being reconstructed.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to inject dashed version strings.
    """
    monkeypatch.setattr(atlas, "_local_version_str", "2024-05")
    monkeypatch.setattr(atlas, "_remote_version_str", "2024-05")
    monkeypatch.setattr(
        type(atlas), "remote_version", property(lambda self: (2024, 5))
    )
    assert atlas.check_latest_version() is True

    monkeypatch.setattr(atlas, "_remote_version_str", "2026-03")
    assert atlas.check_latest_version(print_warning=False) is False
