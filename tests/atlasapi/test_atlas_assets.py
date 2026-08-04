"""Tests for reading atlases that follow the atlas-assets specification."""

import json
from pathlib import Path

import pytest

from brainglobe_atlasapi import bg_atlas

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture()
def atlas_assets_manifest():
    """Return a real atlas-assets atlas manifest.

    Returns
    -------
    dict
        Parsed manifest contents.
    """
    return json.loads((DATA_DIR / "manifest_atlas_assets.json").read_text())


@pytest.fixture()
def ome_attrs():
    """Return the OME attributes block of an atlas-assets zarr.

    Returns
    -------
    dict
        Parsed ``attributes.ome`` contents.
    """
    return json.loads((DATA_DIR / "ome_attrs_atlas_assets.json").read_text())


@pytest.mark.parametrize(
    "atlas_name, expected",
    [
        ("allen-adult-mouse-ccf-atlas", "Mus musculus"),
        ("allen-dev-mouse-e11pt5-nissl-atlas", "Mus musculus"),
        ("hmba-adult-human-homba-atlas", "Homo sapiens"),
        ("hmba-adult-marmoset-homba-atlas", "Callithrix jacchus"),
        ("hmba-adult-rhesusmacaque-homba-atlas", "Macaca mulatta"),
        (
            "hmba-adult-cynomolgusmacaque-homba-atlas",
            "Macaca fascicularis",
        ),
        ("allen-dev-P4-mouse-atlas", "Mus musculus"),
        ("some-unknown-organism-atlas", None),
    ],
)
def test_species_from_name(atlas_name, expected):
    """Species is matched by vocabulary, not by segment position.

    Parameters
    ----------
    atlas_name : str
        Atlas name under test.
    expected : str or None
        Expected binomial name, or None when no segment matches.
    """
    assert bg_atlas._species_from_name(atlas_name) == expected


def test_orientation_from_axes(ome_attrs):
    """Anatomical axis directions map to brainglobe-space letters.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert bg_atlas._orientation_from_axes(ome_attrs) == "asl"


def test_pyramid_level_from_attrs(ome_attrs):
    """Resolution in micrometres selects the matching dataset.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert bg_atlas._pyramid_level_from_attrs(ome_attrs, 10) == (0, "s0")
    assert bg_atlas._pyramid_level_from_attrs(ome_attrs, 25) == (1, "s1")


def test_pyramid_level_from_attrs_invalid(ome_attrs):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._pyramid_level_from_attrs(ome_attrs, 37)
    assert "37" in str(error.value)


def test_brainglobe_manifest_passes_through():
    """A BrainGlobe manifest is returned unchanged."""
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    assert bg_atlas._normalize_manifest(raw, None, Path(".")) == raw


def test_resolution_required_for_multi_scale(atlas_assets_manifest):
    """Omitting resolution on a multi-scale atlas fails loudly.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(atlas_assets_manifest, None, Path("."))
    assert "resolution=" in str(error.value)
    assert "100" in str(error.value)


def test_resolution_must_be_an_available_scale(atlas_assets_manifest):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(atlas_assets_manifest, 37, Path("."))
    assert "37" in str(error.value)


def test_resolution_rejected_when_it_contradicts_manifest():
    """A resolution disagreeing with a BrainGlobe manifest fails."""
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    with pytest.raises(ValueError) as error:
        bg_atlas._normalize_manifest(raw, 10, Path("."))
    assert "25.0" in str(error.value)
