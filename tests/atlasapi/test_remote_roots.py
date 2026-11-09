"""Test locating remote roots and their local caches."""

import pytest

from brainglobe_atlasapi import remote_roots


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
        remote_roots.resolve_remote_root(
            fs, "shared_name", roots, lambda _: True
        )
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
        remote_roots.resolve_remote_root(
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
        remote_roots.resolve_remote_root(
            _FakeFS([]), "nope", roots, lambda _: True
        )
    message = str(error.value)
    assert "nope" in message
    assert "s3://brainglobe/atlas-rc2" in message
    assert "s3://other/prefix" in message


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
        remote_roots.resolve_remote_root(
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
    remote_roots.resolve_remote_root(fs, "shared_name", roots, is_reachable)
    assert probed == ["s3://brainglobe/atlas"]


def test_resolve_root_with_nothing_reachable_returns_none():
    """Offline, resolution defers to the caller rather than raising."""
    roots = {"brainglobe-atlasapi": "s3://brainglobe/atlas"}
    assert (
        remote_roots.resolve_remote_root(
            _FakeFS([]), "shared_name", roots, lambda _: False
        )
        is None
    )


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
    assert remote_roots.cache_key(remote_root) == expected


def test_cache_key_refuses_the_default_directory():
    """Another root ending in brainglobe-atlasapi cannot take its place."""
    with pytest.raises(ValueError, match="reserved"):
        remote_roots.cache_key("s3://someone/brainglobe-atlasapi")


def test_known_remote_roots_lists_configured_then_recorded(tmp_path):
    """Config order comes first; a recorded root is listed only once.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Empty temporary directory used as the brainglobe_dir.
    """
    for directory, root in [("b", "s3://x/b"), ("c", "s3://x/c")]:
        (tmp_path / directory).mkdir()
        remote_roots.record_remote_root(tmp_path / directory, root)

    configured = {"first": "s3://x/z", "second": "s3://x/b"}
    assert remote_roots.known_remote_roots(tmp_path, configured) == [
        "s3://x/z",
        "s3://x/b",
        "s3://x/c",
    ]
