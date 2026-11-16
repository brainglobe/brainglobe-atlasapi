"""Test locating remotes and their local caches."""

import pytest

from brainglobe_atlasapi.remotes import (
    cache_key,
    known_remotes,
    record_remote,
    resolve_remote,
)


class _FakeFS:
    """Filesystem stub that reports a fixed set of existing paths."""

    def __init__(self, existing):
        self.existing = set(existing)

    def ls(self, path):
        if path not in self.existing:
            raise FileNotFoundError(path)
        return [f"{path}/1_0"]


def test_resolve_remote_prefers_declaration_order():
    """A name present in both remotes resolves to the first declared."""
    remotes = {
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
        resolve_remote(fs, "shared_name", remotes, lambda _: True)
        == "s3://brainglobe/atlas-rc2"
    )


def test_resolve_remote_falls_through_to_later_remote():
    """A name only in a later remote resolves to that remote."""
    remotes = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS(["s3://other/prefix/atlases/allen-adult-mouse-ccf-atlas"])
    assert (
        resolve_remote(
            fs, "allen-adult-mouse-ccf-atlas", remotes, lambda _: True
        )
        == "s3://other/prefix"
    )


def test_resolve_remote_unknown_name_lists_remotes():
    """An unknown name reports every remote that was searched."""
    remotes = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas-rc2",
        "allen": "s3://other/prefix",
    }
    fs = _FakeFS([])
    with pytest.raises(ValueError) as error:
        resolve_remote(fs, "nope", remotes, lambda _: True)
    message = str(error.value)
    assert "nope" in message
    assert "s3://brainglobe/atlas-rc2" in message
    assert "s3://other/prefix" in message


def test_resolve_remote_skips_unreachable_remote():
    """A remote whose bucket is down does not stop the others being tried."""
    remotes = {
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
        resolve_remote(
            fs, "shared_name", remotes, lambda r: r == "s3://other/prefix"
        )
        == "s3://other/prefix"
    )


def test_resolve_remote_probes_lazily():
    """Remotes after the one holding the atlas are never probed."""
    remotes = {
        "brainglobe-atlasapi": "s3://brainglobe/atlas",
        "allen": "s3://other/prefix",
    }
    probed = []

    def is_reachable(remote):
        probed.append(remote)
        return True

    fs = _FakeFS(["s3://brainglobe/atlas/atlases/shared_name"])
    resolve_remote(fs, "shared_name", remotes, is_reachable)
    assert probed == ["s3://brainglobe/atlas"]


def test_resolve_remote_with_nothing_reachable_returns_none():
    """Offline, resolution defers to the caller rather than raising."""
    remotes = {"brainglobe-atlasapi": "s3://brainglobe/atlas"}
    assert (
        resolve_remote(_FakeFS([]), "shared_name", remotes, lambda _: False)
        is None
    )


@pytest.mark.parametrize(
    ["remote", "expected"],
    [
        ("s3://brainglobe/atlas", "brainglobe-atlasapi"),
        ("s3://allen-atlas-assets", "allen-atlas-assets"),
        ("s3://brainglobe/atlas-rc2", "atlas-rc2"),
    ],
)
def test_cache_key_is_last_url_segment(remote, expected):
    """Only the BrainGlobe remote maps to brainglobe-atlasapi.

    Parameters
    ----------
    remote : str
        Remote under test.
    expected : str
        Expected cache directory name.
    """
    assert cache_key(remote) == expected


def test_cache_key_refuses_the_default_directory():
    """Another remote ending in brainglobe-atlasapi cannot take its place."""
    with pytest.raises(ValueError, match="reserved"):
        cache_key("s3://someone/brainglobe-atlasapi")


def test_known_remotes_lists_configured_then_recorded(tmp_path):
    """Config order comes first; a recorded remote is listed only once.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    """
    for directory, remote in [("b", "s3://x/b"), ("c", "s3://x/c")]:
        (tmp_path / directory).mkdir()
        record_remote(tmp_path / directory, remote)

    configured = {"first": "s3://x/z", "second": "s3://x/b"}
    assert known_remotes(tmp_path, configured) == [
        "s3://x/z",
        "s3://x/b",
        "s3://x/c",
    ]
