"""
Tests for overwrite and early-exit behaviour in wrapup_atlas_from_data.

These tests verify that atlas generation fails early when output already
exists, and that the overwrite flag correctly replaces existing output.
"""

import numpy as np
import pytest

from brainglobe_atlasapi.atlas_generation.wrapup import (
    ATLAS_VERSION,
    wrapup_atlas_from_data,
)


def _minimal_valid_inputs(tmp_path):
    """
    Return a minimal set of valid inputs required to run wrapup
    past the overwrite logic without performing actual atlas gen.
    """
    return dict(
        atlas_name="test_mouse",
        atlas_minor_version=0,
        citation="unpublished",
        atlas_link="http://example.com",
        species="Mouse (Mus musculus)",
        resolution=(25, 25, 25),
        orientation="asr",
        root_id=0,
        reference_stack=np.zeros((2, 2, 2)),
        annotation_stack=np.zeros((2, 2, 2), dtype=int),
        structures_list=[
            {
                "id": 0,
                "acronym": "root",
                "name": "root",
                "rgb_triplet": [255, 255, 255],
                "structure_id_path": [0],
            }
        ],
        meshes_dict={},
        working_dir=tmp_path,
    )


def _expected_atlas_dir(tmp_path):
    """Return the path where wrapup writes the atlas manifest."""
    atlas_version = f"{ATLAS_VERSION}.0".replace(".", "_")
    return (
        tmp_path
        / "brainglobe-atlasapi"
        / "atlases"
        / "test_mouse_25um"
        / atlas_version
    )


def test_wrapup_fails_if_output_exists(tmp_path):
    """Fail early if atlas output already exists and overwrite=False."""
    atlas_dir = _expected_atlas_dir(tmp_path)
    atlas_dir.mkdir(parents=True)

    kwargs = _minimal_valid_inputs(tmp_path)

    with pytest.raises(FileExistsError, match="Atlas output already exists"):
        wrapup_atlas_from_data(**kwargs, overwrite=False)


def test_wrapup_overwrites_existing_output(tmp_path, monkeypatch):
    """Overwrite existing atlas output when overwrite=True."""
    atlas_dir = _expected_atlas_dir(tmp_path)
    atlas_dir.mkdir(parents=True)
    (atlas_dir / "old_file.txt").write_text("old")

    from brainglobe_atlasapi.atlas_generation import wrapup

    # Stub out heavy I/O that runs after the overwrite check.
    class _FakeImage:
        data = type("arr", (), {"shape": (2, 2, 2)})()

    class _FakeMultiscale:
        images = [_FakeImage()]

    monkeypatch.setattr(
        wrapup, "_save_template_data", lambda *a, **kw: _FakeMultiscale()
    )
    monkeypatch.setattr(
        wrapup, "_save_annotation_data", lambda *a, **kw: (None, None)
    )
    monkeypatch.setattr(
        wrapup, "_save_additional_references", lambda *a, **kw: None
    )
    monkeypatch.setattr(wrapup, "get_all_validation_functions", lambda: [])
    monkeypatch.setattr(
        wrapup,
        "BrainGlobeAtlas",
        lambda *args, **kwargs: None,
    )

    kwargs = _minimal_valid_inputs(tmp_path)
    wrapup_atlas_from_data(**kwargs, overwrite=True)

    assert atlas_dir.exists()
    assert not (atlas_dir / "old_file.txt").exists()
