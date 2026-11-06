"""Tests for converting atlas manifests into ``core.Atlas`` metadata."""

import copy
import json
from pathlib import Path

import pytest

from brainglobe_atlasapi import manifest

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
    assert manifest._species_from_name(atlas_name) == expected


def test_orientation_from_axes(ome_attrs):
    """Anatomical axis directions map to brainglobe-space letters.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert manifest._orientation_from_axes(ome_attrs) == "asl"


def test_origin_letter_is_where_each_axis_starts():
    """Each direction's origin letter is the side it starts from."""
    start_letter = {
        "anterior": "a",
        "rostral": "a",
        "posterior": "p",
        "caudal": "p",
        "superior": "s",
        "dorsal": "s",
        "inferior": "i",
        "ventral": "i",
        "left": "l",
        "right": "r",
    }
    for direction, letter in manifest._ORIGIN_BY_AXIS_DIRECTION.items():
        start = direction.split("-to-")[0]
        assert letter == start_letter[start], direction


def test_pyramid_level_from_attrs(ome_attrs):
    """Resolution in micrometres selects the matching dataset.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    assert manifest._pyramid_level_from_attrs(ome_attrs, 10) == (0, "s0")
    assert manifest._pyramid_level_from_attrs(ome_attrs, 25) == (1, "s1")


def test_pyramid_level_from_attrs_invalid(ome_attrs):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    ome_attrs : dict
        OME attributes block fixture.
    """
    with pytest.raises(ValueError) as error:
        manifest._pyramid_level_from_attrs(ome_attrs, 37)
    assert "37" in str(error.value)


def test_brainglobe_manifest_passes_through():
    """A BrainGlobe manifest is returned unchanged.

    ``normalize_manifest`` returns the same object for a BrainGlobe
    manifest (``raw is result``), so comparing the result against
    ``raw`` after the call would pass even if the function mutated
    ``raw`` in place. Comparing against an untouched deep copy, taken
    before the call, actually exercises that guarantee.
    """
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    untouched = copy.deepcopy(raw)
    result = manifest.normalize_manifest(raw, None, Path("."))
    assert result == untouched


def test_resolution_defaults_to_finest_for_multi_scale(
    atlas_assets_manifest, capsys
):
    """A multi-scale atlas with no resolution picks the finest, and says so.

    As in the single-scale test below, there is no zarr store on disk, so
    execution fails at the read that follows the resolution choice.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    capsys : pytest.CaptureFixture
        Captures the printed note.
    """
    with pytest.raises(FileNotFoundError):
        manifest.normalize_manifest(atlas_assets_manifest, None, Path("."))
    printed = capsys.readouterr().out
    assert "finest, 10 um" in printed
    assert "resolution=" in printed


def test_resolution_defaulted_for_single_scale(atlas_assets_manifest):
    """A single-scale atlas-assets manifest defaults resolution.

    The zarr read that follows resolution selection needs a real store
    on disk, which this test does not provide, so it can only exercise
    the scales-defaulting decision itself: a single-scale manifest with
    resolution=None must not raise, proving resolution was defaulted
    rather than left unset. Execution then reaches the (missing) zarr
    store and fails for that unrelated reason instead.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    atlas_assets_manifest["annotation_sets"][0]["scales"] = [25]
    with pytest.raises(FileNotFoundError):
        manifest.normalize_manifest(atlas_assets_manifest, None, Path("."))


def test_resolution_must_be_an_available_scale(atlas_assets_manifest):
    """An unavailable resolution fails loudly.

    Parameters
    ----------
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        manifest.normalize_manifest(atlas_assets_manifest, 37, Path("."))
    assert "37" in str(error.value)


def test_resolution_rejected_when_it_contradicts_manifest():
    """A resolution disagreeing with a BrainGlobe manifest fails."""
    raw = json.loads((DATA_DIR / "manifest_brainglobe.json").read_text())
    with pytest.raises(ValueError) as error:
        manifest.normalize_manifest(raw, 10, Path("."))
    assert "25.0" in str(error.value)


def test_scale_from_flat_transforms():
    """OME 0.5 puts the scale at the top level of the list."""
    transforms = [
        {"type": "scale", "scale": [0.025, 0.025, 0.025]},
        {"type": "translation", "translation": [0, 0, 0]},
    ]
    assert manifest._scale_from_transforms(transforms) == [0.025, 0.025, 0.025]


def test_scale_from_sequence_transforms():
    """OME 0.6 nests the scale inside a sequence transformation."""
    transforms = [
        {
            "type": "sequence",
            "transformations": [{"type": "scale", "scale": [0.1, 0.1, 0.1]}],
        }
    ]
    assert manifest._scale_from_transforms(transforms) == [0.1, 0.1, 0.1]


def test_scale_from_transforms_missing():
    """A transform list with no scale fails loudly."""
    with pytest.raises(ValueError) as error:
        manifest._scale_from_transforms([{"type": "translation"}])
    assert "scale" in str(error.value)
