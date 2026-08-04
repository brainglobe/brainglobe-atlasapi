"""Tests that remote assets are read from the atlas' own remote root."""

import pytest

from brainglobe_atlasapi import descriptors, structure_class
from brainglobe_atlasapi.structure_class import StructuresDict

structures_list = [
    {
        "acronym": "root",
        "id": 997,
        "name": "root",
        "structure_id_path": [997],
        "rgb_triplet": [255, 255, 255],
        "mesh_filename": None,
    },
]


def test_hemispheres_incomplete_cache_raises(atlas, monkeypatch):
    """An asymmetric atlas missing a cached, bucket-held asset raises.

    The local cache lacking an asset the bucket does hold means the cache
    is incomplete; synthesizing hemispheres there would silently return
    wrong values.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture, whose hemispheres asset is not cached.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to force the asymmetric, bucket-present combination.
    """
    monkeypatch.setitem(atlas.metadata, "symmetric", False)
    atlas._hemispheres = None
    monkeypatch.setattr(atlas.fs, "exists", lambda path: True)

    with pytest.raises(FileNotFoundError):
        atlas.hemispheres


def test_hemispheres_cached_asset_skips_bucket(asymmetric_atlas, monkeypatch):
    """A complete cache is read without consulting the bucket.

    Parameters
    ----------
    asymmetric_atlas : BrainGlobeAtlas
        Asymmetric atlas fixture with a cached hemispheres asset.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to fail the test on any bucket call.
    """

    def fail(*args, **kwargs):
        raise AssertionError("bucket must not be consulted")

    asymmetric_atlas._hemispheres = None
    monkeypatch.setattr(asymmetric_atlas.fs, "exists", fail)

    assert asymmetric_atlas.hemispheres is not None


def _capture_mesh_url(monkeypatch, requested):
    """Monkeypatch `s3fs` so mesh lookups record their URL and stop."""

    class FakeS3FileSystem:
        def __init__(self, *args, **kwargs):
            pass

        def exists(self, path):
            requested.append(path)
            return False

    monkeypatch.setattr(structure_class.s3fs, "S3FileSystem", FakeS3FileSystem)


def test_mesh_download_uses_custom_remote_root(tmp_path, monkeypatch):
    """A non-default remote root is used for the mesh download."""
    requested = []
    _capture_mesh_url(monkeypatch, requested)

    struct_dict = StructuresDict(
        structures_list, remote_root="s3://other-bucket/atlas-assets"
    )

    with pytest.raises(FileNotFoundError):
        struct_dict["root"]._download_mesh(tmp_path / "997")

    assert requested[0].startswith("s3://other-bucket/atlas-assets/")


def test_mesh_download_defaults_to_brainglobe_root(tmp_path, monkeypatch):
    """With no remote root supplied, the BrainGlobe bucket is used."""
    requested = []
    _capture_mesh_url(monkeypatch, requested)

    struct_dict = StructuresDict(structures_list)

    with pytest.raises(FileNotFoundError):
        struct_dict["root"]._download_mesh(tmp_path / "997")

    assert requested[0].startswith(f"{descriptors.DEFAULT_REMOTE_ROOT}/")
