"""Test the BrainGlobeAtlas class."""

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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
        atlas._remote_version_str = None
        atlas._remote = bg_atlas.descriptors.DEFAULT_REMOTE
        atlas._reachable_remotes = set()
        assert atlas.remote_version is None


def test_remote_version_missing_from_assumed_remote(mocker):
    """A cached atlas whose remote doesn't host it fails cleanly.

    Regression test: remote resolution is skipped when the atlas is already
    found in a local cache, so ``remote_version`` keeps that cache's
    non-default remote. If that remote doesn't actually have this atlas name,
    ``fs.ls`` on the nonexistent S3 prefix returns ``[]`` rather than
    raising, which previously fell through to ``get_latest_version([])``
    and raised a bare ``IndexError``. There is then no remote version, and
    resolving one for a download raises ``FileNotFoundError``, which
    callers such as ``update_atlas`` depend on. (On the default remote,
    ``last_versions.conf`` is consulted instead, so this path is never
    reached there.)
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas._remote = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE
    atlas._reachable_remotes = set()
    atlas.atlas_name = "not_actually_here"
    atlas.fs = SimpleNamespace(ls=lambda path: [])

    mocker.patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    )
    assert atlas.remote_version is None
    with pytest.raises(FileNotFoundError, match="is not a valid atlas name!"):
        atlas._resolve_remote_version()


def test_remote_version_carries_dashed_folder_string(mocker):
    """A dashed atlas-assets folder name is carried through verbatim.

    Regression test: ``remote_version`` must record the resolved S3
    folder name in ``_remote_version_str`` as-is, not reconstruct it
    from the parsed tuple: rebuilding it from ``(2024, 5)`` would yield
    ``"2024_5"``, which is not the real folder ``"2024-05"``.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas._remote = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE
    atlas._reachable_remotes = set()
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas.fs = SimpleNamespace(
        ls=lambda path: [f"{path}/2017", f"{path}/2024-05"],
    )

    mocker.patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    )
    assert atlas.remote_version == (2024, 5)
    assert atlas._remote_version_str == "2024-05"


def test_remote_version_ignores_non_version_entries(mocker):
    """Stray files next to the version folders are not parsed as versions."""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas._remote = bg_atlas.descriptors.ATLAS_ASSETS_REMOTE
    atlas._reachable_remotes = set()
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas.fs = SimpleNamespace(
        ls=lambda path: [f"{path}/2017", f"{path}/README.md", f"{path}/tmp"],
    )

    mocker.patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    )
    assert atlas.remote_version == (2017,)


def _atlas_for_remote_version(atlas_name, brainglobe_dir):
    """Create a bare BrainGlobeAtlas with a mocked filesystem."""
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = atlas_name
    atlas.brainglobe_dir = brainglobe_dir
    atlas._remote = bg_atlas.descriptors.DEFAULT_REMOTE
    atlas._reachable_remotes = set()
    atlas._remote_version_str = None
    atlas._requested_version = None
    atlas.fs = MagicMock()
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
    """Test an atlas absent from last_versions.conf has no remote version,
    and resolving one (the first step of a download) raises.

    Parameters
    ----------
    tmp_path : Path
        Temporary brainglobe directory.
    """
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
        assert atlas.remote_version is None
        with pytest.raises(FileNotFoundError, match="no released version"):
            atlas._resolve_remote_version()


@pytest.mark.parametrize(
    "requested, expected",
    [
        pytest.param("2_0", (2, 0), id="on remote"),
        pytest.param("3_0", None, id="not on remote"),
    ],
)
def test_remote_version_requested(tmp_path, requested, expected, mocker):
    """Test a requested version is taken from the remote folder listing.

    Parameters
    ----------
    tmp_path : Path
        Temporary brainglobe directory.
    requested : str
        Requested version folder name.
    expected : tuple or None
        The expected remote version.
    mocker : pytest_mock.MockerFixture
        Used to stub the S3 status check.
    """
    atlas = _atlas_for_remote_version("example_mouse_100um", tmp_path)
    atlas._requested_version = requested
    mocker.patch.object(
        brainglobe_atlasapi.bg_atlas, "check_s3_status", return_value=True
    )
    assert atlas.remote_version == expected
    if expected is None:
        with pytest.raises(FileNotFoundError, match="not found in remote"):
            atlas._resolve_remote_version()


@pytest.mark.parametrize(
    "local_folder, remote_folder, expected",
    [
        pytest.param("1_0", "2_0", False, id="local < remote"),
        pytest.param("1_0", "1_0", True, id="local = remote"),
        pytest.param("2024-05", "2024-05", True, id="dashed equal"),
        pytest.param("2024-05", "2026-03", False, id="dashed local < remote"),
        pytest.param("2024-05", "2024_05", False, id="separators differ"),
        pytest.param("1_0", None, None, id="no remote version"),
    ],
)
def test_check_latest_version_local(
    local_folder, remote_folder, expected, mocker
):
    """Test `check_latest_version` compares version folder names exactly.

    Parameters
    ----------
    local_folder : str
        The local version folder name.
    remote_folder : str or None
        The remote version folder name, or None when offline.
    expected : bool or None
        The expected result of `check_latest_version`.
    mocker : pytest_mock.MockerFixture
        Used to stub the ``remote_version`` property.
    """
    remote_version = (
        None
        if remote_folder is None
        else bg_atlas.version_tuple_from_str(remote_folder)
    )
    mocker.patch.object(
        BrainGlobeAtlas,
        "remote_version",
        new_callable=mocker.PropertyMock,
        return_value=remote_version,
    )
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


def test_local_search(tmpdir, atlas_path):
    """Test local atlas search and handling of multiple versions."""
    atlas_name = "example_mouse_100um"
    temp_brainglobe_dir = tmpdir.mkdir("brainglobe")
    # The search is local, so start from the cache rather than downloading.
    shutil.copytree(atlas_path, temp_brainglobe_dir / atlas_path.name)

    atlas = BrainGlobeAtlas(
        atlas_name,
        brainglobe_dir=temp_brainglobe_dir,
        check_latest=False,
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
        check_latest=False,
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
        (
            "allen-dev-mouse-e11pt5-nissl-atlas",
            [16.0, 16.0, 20.0],
            "allen-dev-mouse-e11pt5-nissl-atlas (res. 16.0um)",
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


def test_default_root_cache_path_is_unchanged(atlas):
    """The default remote keeps the historical cache directory name.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture.
    """
    assert atlas.brainglobe_dir.name == "brainglobe-atlasapi"
    assert atlas.root_dir == atlas.brainglobe_dir


def test_resolve_non_default_root_switches_cache_dir(tmp_path, monkeypatch):
    """Resolving to a non-default remote swaps the on-disk cache directory.

    Remote resolution is deferred until the atlas is not found under the
    default remote's local cache, so this stubs out every network-facing
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
        "get_remotes",
        lambda *_: {
            "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
            "allen": "s3://other/prefix",
        },
    )
    monkeypatch.setattr(
        bg_atlas,
        "resolve_remote",
        lambda fs, name, remotes, is_reachable: "s3://other/prefix",
    )

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas("not_cached_anywhere", brainglobe_dir=tmp_path)

    assert (tmp_path / "prefix").is_dir()


def test_atlas_cached_under_other_root_loads_offline(tmp_path, monkeypatch):
    """A copy cached under a non-default remote is found without S3.

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
    monkeypatch.setattr(bg_atlas, "resolve_remote", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remotes",
        lambda *_: {
            "brainglobe-atlasapi": "s3://brainglobe/atlas",
            "allen": "s3://other/prefix",
        },
    )
    monkeypatch.setattr(
        bg_atlas.core.Atlas, "__init__", lambda self, *a, **k: None
    )

    atlas = BrainGlobeAtlas(
        "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
    )

    assert atlas._remote == "s3://other/prefix"
    assert atlas.brainglobe_dir == tmp_path / "prefix"
    record = tmp_path / "prefix" / descriptors.REMOTE_FILENAME
    assert record.read_text() == "s3://other/prefix\n"
    assert atlas.local_full_name == "atlases/cached_atlas/1_0/manifest.json"


def test_atlas_cached_under_unconfigured_root_loads_offline(
    tmp_path, monkeypatch
):
    """A cache whose remote is only recorded, not configured, is found.

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
    (cache / descriptors.REMOTE_FILENAME).write_text("s3://bucket/custom\n")

    def fail(*args, **kwargs):
        raise AssertionError("no network call should be made")

    monkeypatch.setattr(bg_atlas, "check_s3_status", fail)
    monkeypatch.setattr(bg_atlas, "resolve_remote", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remotes",
        lambda *_: {"brainglobe-atlasapi": "s3://brainglobe/atlas"},
    )
    monkeypatch.setattr(
        bg_atlas.core.Atlas, "__init__", lambda self, *a, **k: None
    )

    atlas = BrainGlobeAtlas(
        "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
    )

    assert atlas._remote == "s3://bucket/custom"
    assert atlas.brainglobe_dir == cache


def test_remote_override_skips_config_search(tmp_path, monkeypatch, mocker):
    """An explicit remote takes full priority over configured remotes.

    It must be the only remote searched: the config/multi-remote resolution
    is skipped entirely, and its local cache directory is named after the
    last URL segment rather than any configured remote key.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    monkeypatch : pytest.MonkeyPatch
        Used to stub S3 access.
    mocker : pytest_mock.MockerFixture
        Used to spy on the config/resolution calls.
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
    resolve_spy = mocker.patch.object(bg_atlas, "resolve_remote")
    remotes_spy = mocker.patch.object(bg_atlas.config, "get_remotes")

    with pytest.raises(FileNotFoundError):
        BrainGlobeAtlas(
            "not_cached_anywhere",
            brainglobe_dir=tmp_path,
            remote="s3://custom-bucket/prefix",
        )

    resolve_spy.assert_not_called()
    remotes_spy.assert_not_called()
    assert (tmp_path / "prefix").is_dir()


def test_explicit_default_root_uses_default_cache(tmp_path, monkeypatch):
    """Passing the BrainGlobe remote explicitly reuses the default cache.

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
            remote="s3://brainglobe/atlas/",
        )

    record = tmp_path / "brainglobe-atlasapi" / descriptors.REMOTE_FILENAME
    # A failed lookup must not claim the cache directory.
    assert not record.exists()
    assert [p.name for p in tmp_path.iterdir()] == ["brainglobe-atlasapi"]


def test_cache_recording_another_root_raises(tmp_path, monkeypatch):
    """A cache directory recording a different remote is never reused.

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
    record = tmp_path / "prefix" / descriptors.REMOTE_FILENAME
    record.write_text("s3://somebody-else/prefix\n")

    def fail(*args, **kwargs):
        raise AssertionError("no network call should be made")

    monkeypatch.setattr(bg_atlas, "check_s3_status", fail)
    monkeypatch.setattr(
        bg_atlas.config,
        "get_remotes",
        lambda *_: {"allen": "s3://other/prefix"},
    )

    with pytest.raises(ValueError, match="s3://somebody-else/prefix"):
        BrainGlobeAtlas(
            "cached_atlas", brainglobe_dir=tmp_path, check_latest=False
        )


def test_check_latest_version_warning_prints_folder_strings(mocker, capsys):
    """The update warning shows dashed versions as the folders name them.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to stub the ``remote_version`` property.
    capsys : pytest.CaptureFixture
        Captures the printed warning.
    """
    mocker.patch.object(
        BrainGlobeAtlas,
        "remote_version",
        new_callable=mocker.PropertyMock,
        return_value=(2026, 3),
    )
    atlas = object.__new__(BrainGlobeAtlas)
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas._local_version_str = "2024-05"
    atlas._remote_version_str = "2026-03"

    assert atlas.check_latest_version() is False
    out = capsys.readouterr().out
    assert "2024-05" in out
    assert "2026-03" in out


def test_is_reachable_remembers_only_success(mocker):
    """A failed probe is retried later; a successful one is not repeated.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Used to stub the S3 status check.
    """
    status = mocker.patch.object(
        bg_atlas, "check_s3_status", side_effect=[False, True]
    )
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._reachable_remotes = set()
    remote = descriptors.ATLAS_ASSETS_REMOTE

    assert atlas._is_reachable(remote) is False
    assert atlas._is_reachable(remote) is True
    # A third probe would exhaust side_effect and raise StopIteration.
    assert atlas._is_reachable(remote) is True
    assert status.call_count == 2


@pytest.mark.parametrize(
    "requested, expected",
    [
        pytest.param("2024-05", (2024, 5), id="exact folder name"),
        # What __init__ stores for version="2024.05".
        pytest.param("2024_05", None, id="2024.05 is not 2024-05"),
    ],
)
def test_requested_version_matches_non_default_root_exactly(
    requested, expected, mocker
):
    """Only the BrainGlobe remote normalises version separators.

    Folders on other remotes are named by their owners, so a requested
    version must match the folder name exactly: there is no guarantee
    that ``2024.05`` and ``2024-05`` are the same version.

    Parameters
    ----------
    requested : str
        Requested version, as stored by ``__init__``.
    expected : tuple or None
        The expected remote version.
    mocker : pytest_mock.MockerFixture
        Used to stub the S3 status check.
    """
    atlas = object.__new__(BrainGlobeAtlas)
    atlas._remote_version_str = None
    atlas._requested_version = requested
    atlas._remote = descriptors.ATLAS_ASSETS_REMOTE
    atlas._reachable_remotes = set()
    atlas.atlas_name = "allen-adult-mouse-ccf-atlas"
    atlas.fs = SimpleNamespace(
        ls=lambda path: [f"{path}/2017", f"{path}/2024-05"],
    )
    mocker.patch.object(bg_atlas, "check_s3_status", return_value=True)

    assert atlas.remote_version == expected
    if expected is None:
        with pytest.raises(FileNotFoundError, match="not found in remote"):
            atlas._resolve_remote_version()
