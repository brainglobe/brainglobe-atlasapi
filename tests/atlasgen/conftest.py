from pathlib import Path

import pytest

from brainglobe_atlasapi import BrainGlobeAtlas


@pytest.fixture(autouse=True)
def setup_preexisting_local_atlases():
    """Automatically setup all tests to have three downloaded atlases
    in the test user data."""
    preexisting_atlases = [
        ("example_mouse_100um", "v1.2"),
        ("allen_mouse_100um", "v1.2"),
        ("kim_dev_mouse_e11-5_mri-adc_31.5um", "v1.3"),
    ]
    for atlas_name, version in preexisting_atlases:
        if not Path.exists(
            Path.home() / f".brainglobe/{atlas_name}_{version}"
        ):
            _ = BrainGlobeAtlas(atlas_name)


@pytest.fixture
def structures():
    """List of structures for testing.

    Structure tree:

    root (999)
    └── o (101)
      ├── aon (5)
      └── on (1)

    """
    structure101 = {
        "id": 101,
        "acronym": "o",
        "name": "olfactory system",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 999, 101],
    }
    structure1 = {
        "id": 1,
        "acronym": "on",
        "name": "olfactory nerve",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 101, 1],
    }
    structure5 = {
        "id": 5,
        "acronym": "aon",
        "name": "anterior olfactory nucleus",
        "rgb_triplet": [255, 0, 0],
        "structure_id_path": [999, 101, 5],
    }
    root = {
        "name": "root",
        "acronym": "root",
        "id": 999,
        "rgb_triplet": [255, 255, 255],
        "structure_id_path": [999],
    }

    return [
        structure101,
        structure1,
        structure5,
        root,
    ]
