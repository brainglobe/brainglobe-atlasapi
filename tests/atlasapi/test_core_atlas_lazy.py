"""Test lazy atlas array loading."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import dask.array as da
import numpy as np
import pytest

from brainglobe_atlasapi import BrainGlobeAtlas, core


def _multiscale(data):
    """Build a tiny ngff-zarr-like multiscale object."""
    return SimpleNamespace(
        images=[SimpleNamespace(data=data)],
        metadata=SimpleNamespace(datasets=[SimpleNamespace(path="s0")]),
    )


def _atlas(tmp_path, monkeypatch, data):
    """Build a minimal Atlas instance without running full init."""
    atlas = object.__new__(core.Atlas)
    atlas.lazy = True
    atlas.root_dir = tmp_path
    atlas.metadata = {
        "annotation_set": {
            "template": {"location": "/template"},
            "location": "/annotation",
        },
        "shape": [2, 2, 2],
        "symmetric": True,
    }
    atlas.space = SimpleNamespace(
        axes_order=["superior", "anterior", "frontal"]
    )
    atlas.fs = MagicMock()
    atlas._template = None
    atlas._annotation = None
    atlas._hemispheres = None
    atlas._template_pyramid_level = 0
    atlas._annotation_pyramid_level = 0
    atlas._annotation_masks_pyramid_level = 0
    atlas._annotation_mapping = {2: 0}
    atlas.structures = {
        "leaf": {"id": 2},
        2: {"id": 2, "acronym": "leaf", "structure_id_path": [2]},
    }
    monkeypatch.setattr(
        core.nz, "from_ngff_zarr", lambda path: _multiscale(data)
    )
    return atlas


@pytest.mark.parametrize(
    "stack_name, val",
    [
        ("template", [[[155, 146], [157, 153]], [[151, 148], [154, 153]]]),
        ("annotation", [[[59, 59], [59, 59]], [[59, 59], [59, 59]]]),
        ("hemispheres", [[[2, 1], [2, 1]], [[2, 1], [2, 1]]]),
    ],
)
def test_example_atlas_lazy_properties_are_dask_arrays(atlas, stack_name, val):
    """BrainGlobeAtlas(..., lazy=True) works on the example atlas."""
    lazy_atlas = BrainGlobeAtlas(
        atlas.atlas_name, check_latest=False, lazy=True
    )
    loaded_stack = getattr(lazy_atlas, stack_name)

    assert isinstance(loaded_stack, da.Array)
    assert np.allclose(loaded_stack[65:67, 39:41, 56:58].compute(), val)


def test_lazy_coord_lookups_return_scalars(tmp_path, monkeypatch):
    """lazy=True keeps coordinate helper return values eager/scalar."""
    data = da.from_array(np.full((2, 2, 2), 2, dtype=np.uint8))
    atlas = _atlas(tmp_path, monkeypatch, data)

    structure = atlas.structure_from_coords((0, 0, 0))

    assert isinstance(structure, int)
    assert structure == 2
    assert atlas.structure_from_coords((0, 0, 0), as_acronym=True) == "leaf"
    assert atlas.hemisphere_from_coords((0, 0, 1), as_string=True) == "left"


def test_lazy_get_structure_mask_returns_dask_array(tmp_path, monkeypatch):
    """lazy=True returns a dask array for structure masks."""
    data = da.from_array(np.ones((1, 2, 2, 2), dtype=np.uint8))
    atlas = _atlas(tmp_path, monkeypatch, data)

    mask = atlas.get_structure_mask("leaf")

    assert isinstance(mask, da.Array)
    assert mask.compute()[0, 0, 0] == 1
