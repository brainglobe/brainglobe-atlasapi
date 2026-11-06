"""Tests for converting atlas manifests into ``core.Atlas`` metadata.

Manifests and OME attributes are the real ones, read from the local atlas
cache. ``test_cached_metadata_matches_remote`` keeps that cache honest, so a
format change on a remote fails here rather than going unnoticed.
"""

import copy
from pathlib import Path

import pytest
import s3fs

from brainglobe_atlasapi import descriptors, manifest
from brainglobe_atlasapi.utils import read_json


def _raw_manifest(atlas):
    """Read an atlas's manifest from the cache, before normalisation.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Atlas whose manifest to read.

    Returns
    -------
    dict
        Parsed manifest contents.
    """
    return read_json(atlas.brainglobe_dir / atlas.local_full_name)


def _annotation_ome_attrs(atlas, raw_manifest):
    """Read the OME attributes block of an atlas's cached annotation zarr.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Atlas whose annotation to read.
    raw_manifest : dict
        Parsed manifest of the atlas, in either shape.

    Returns
    -------
    dict
        Parsed ``attributes.ome`` contents.
    """
    location = manifest.get_annotation_set(raw_manifest)["location"][1:]
    zarr_json = (
        atlas.root_dir
        / location
        / descriptors.V3_ANNOTATION_NAME
        / "zarr.json"
    )
    return read_json(zarr_json)["attributes"]["ome"]


@pytest.fixture()
def atlas_assets_manifest(atlas_assets_atlas):
    """Return a real atlas-assets atlas manifest.

    Parameters
    ----------
    atlas_assets_atlas : BrainGlobeAtlas
        Shared atlas-assets atlas fixture.

    Returns
    -------
    dict
        Parsed manifest contents.
    """
    return _raw_manifest(atlas_assets_atlas)


@pytest.fixture()
def brainglobe_manifest(atlas):
    """Return a real BrainGlobe atlas manifest.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture.

    Returns
    -------
    dict
        Parsed manifest contents.
    """
    return _raw_manifest(atlas)


@pytest.fixture()
def ome_attrs(atlas_assets_atlas, atlas_assets_manifest):
    """Return the OME attributes block of a real atlas-assets zarr.

    Parameters
    ----------
    atlas_assets_atlas : BrainGlobeAtlas
        Shared atlas-assets atlas fixture.
    atlas_assets_manifest : dict
        atlas-assets manifest fixture.

    Returns
    -------
    dict
        Parsed ``attributes.ome`` contents.
    """
    return _annotation_ome_attrs(atlas_assets_atlas, atlas_assets_manifest)


@pytest.fixture()
def brainglobe_ome_attrs(atlas, brainglobe_manifest):
    """Return the OME attributes block of a real BrainGlobe zarr.

    Parameters
    ----------
    atlas : BrainGlobeAtlas
        Default test atlas fixture.
    brainglobe_manifest : dict
        BrainGlobe manifest fixture.

    Returns
    -------
    dict
        Parsed ``attributes.ome`` contents.
    """
    return _annotation_ome_attrs(atlas, brainglobe_manifest)


@pytest.mark.parametrize("atlas_fixture", ["atlas", "atlas_assets_atlas"])
def test_cached_metadata_matches_remote(request, atlas_fixture):
    """The cached metadata tests rely on is what the bucket holds today.

    Version folders can change in place, and a cached component is never
    downloaded again, so without this a long-lived cache would hide a format
    change on the remote. Costs a few small requests per test session.

    Parameters
    ----------
    request : pytest.FixtureRequest
        Used to look up the atlas fixture by name.
    atlas_fixture : str
        Name of the fixture providing the atlas to check.
    """
    atlas = request.getfixturevalue(atlas_fixture)
    annotation_set = atlas.metadata["annotation_set"]
    annotation = annotation_set["location"][1:]
    template = annotation_set["template"]["location"][1:]
    terminology = atlas.metadata["terminology"]["location"][1:]
    cached_files = [
        atlas.local_full_name,
        f"{annotation}/{descriptors.V3_ANNOTATION_NAME}/zarr.json",
        f"{template}/{descriptors.V3_TEMPLATE_NAME}/zarr.json",
        f"{terminology}/{descriptors.V3_TERMINOLOGY_NAME}",
    ]

    fs = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
    stale = [
        path
        for path in cached_files
        if (atlas.root_dir / path).read_bytes()
        != fs.cat_file(f"{atlas._remote_root}/{path}")
    ]

    assert not stale, (
        f"{atlas._remote_root} changed in place. Delete {atlas.root_dir} "
        f"(in CI, the atlas cache) to test against it. Differs: {stale}"
    )


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


def test_orientation_from_flat_axes(brainglobe_ome_attrs, brainglobe_manifest):
    """OME 0.5 axes on the multiscale give the manifest's orientation.

    Parameters
    ----------
    brainglobe_ome_attrs : dict
        OME 0.5 attributes block fixture.
    brainglobe_manifest : dict
        BrainGlobe manifest fixture.
    """
    multiscale = brainglobe_ome_attrs["multiscales"][0]
    assert manifest._intrinsic_axes(multiscale) == multiscale["axes"]
    assert (
        manifest._orientation_from_axes(brainglobe_ome_attrs)
        == brainglobe_manifest["orientation"]
    )


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


def test_brainglobe_manifest_passes_through(brainglobe_manifest):
    """A BrainGlobe manifest is returned unchanged.

    ``normalize_manifest`` returns the same object for a BrainGlobe
    manifest (``raw is result``), so comparing the result against
    ``raw`` after the call would pass even if the function mutated
    ``raw`` in place. Comparing against an untouched deep copy, taken
    before the call, actually exercises that guarantee.

    Parameters
    ----------
    brainglobe_manifest : dict
        BrainGlobe manifest fixture.
    """
    untouched = copy.deepcopy(brainglobe_manifest)
    result = manifest.normalize_manifest(brainglobe_manifest, None, Path("."))
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


def test_resolution_rejected_when_it_contradicts_manifest(brainglobe_manifest):
    """A resolution disagreeing with a BrainGlobe manifest fails.

    Parameters
    ----------
    brainglobe_manifest : dict
        BrainGlobe manifest fixture.
    """
    with pytest.raises(ValueError) as error:
        manifest.normalize_manifest(brainglobe_manifest, 10, Path("."))
    assert str(brainglobe_manifest["resolution"][0]) in str(error.value)


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
