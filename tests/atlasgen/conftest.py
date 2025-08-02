"""Define test fixtures for atlas generation tests.

Provide commonly used fixtures and setup for testing `brainglobe_atlasapi`
functionality.
"""

from pathlib import Path

import pytest

from brainglobe_atlasapi import BrainGlobeAtlas


@pytest.fixture(autouse=True)
def setup_preexisting_local_atlases():
    """Set up all tests to have three downloaded atlases in the test user data.

    Automatically downloads and sets up predefined atlases for testing
    purposes, ensuring they are available locally before test execution.
    """
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
    """Provide a list of structures for testing.

    >>> The structure tree is defined as:
    root (999)
    └── o (101)
        ├── aon (5)
        └── on (1)

    Returns
    -------
    list
        A list of dictionaries, where each dictionary represents a structure
        with its ID, acronym, name, RGB triplet, and structure ID path.
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
