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
