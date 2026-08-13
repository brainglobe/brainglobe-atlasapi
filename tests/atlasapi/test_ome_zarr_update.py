"""Tests for the temporary atlas-assets OME-Zarr metadata shim.

Delete alongside ``brainglobe_atlasapi._ome_zarr_repair``.
"""

import copy
import json

import ngff_zarr as nz
import pytest
import zarr

from brainglobe_atlasapi._ome_zarr_update import update_ome_zarr_attributes

SCALES = [0.01, 0.025, 0.05]


def _axes():
    return [
        {
            "name": "z",
            "type": "space",
            "unit": "millimeter",
            "orientation": {
                "type": "anatomical",
                "value": "anterior-to-posterior",
            },
        },
        {
            "name": "y",
            "type": "space",
            "unit": "millimeter",
            "orientation": {
                "type": "anatomical",
                "value": "dorsal-to-ventral",
            },
        },
        {
            "name": "x",
            "type": "space",
            "unit": "millimeter",
            "orientation": {"type": "anatomical", "value": "left-to-right"},
        },
    ]


def _valid_ome():
    """Build the metadata the atlas-assets bucket ought to be writing."""
    return {
        "version": "0.6",
        "multiscales": [
            {
                "datasets": [
                    {
                        "path": f"s{i}",
                        "coordinateTransformations": [
                            {
                                "type": "sequence",
                                "input": {"path": f"s{i}"},
                                "output": {"name": "intrinsic"},
                                "transformations": [
                                    {"type": "scale", "scale": [scale] * 3}
                                ],
                            }
                        ],
                    }
                    for i, scale in enumerate(SCALES)
                ],
                "name": "/",
                "axes": _axes(),
                "coordinateTransformations": [
                    {
                        "type": "sequence",
                        "input": "intrinsic",
                        "output": "mm RAS",
                        "transformations": [
                            {
                                "type": "scale",
                                "scale": [1.0, 1.0, 1.0],
                            }
                        ],
                    }
                ],
                "coordinateSystems": [
                    {"name": "intrinsic", "axes": _axes()},
                    {"name": "mm RAS", "axes": _axes()},
                ],
            }
        ],
    }


def _fully_wrapped_ome():
    """Build the shim's fixed point.

    The bucket's corrected reference file leaves the multiscale-level
    ``input``/``output`` as bare strings, which ``ngff-zarr`` accepts.
    The shim wraps them anyway, so only this shape survives a repair
    untouched.
    """
    ome = _valid_ome()
    transform = ome["multiscales"][0]["coordinateTransformations"][0]
    transform["input"] = {"name": "intrinsic"}
    transform["output"] = {"name": "mm RAS"}
    return ome


def _break_version(ome):
    ome["version"] = "0.6.dev3"


def _break_references(ome):
    for dataset in ome["multiscales"][0]["datasets"]:
        for transform in dataset["coordinateTransformations"]:
            transform["input"] = transform["input"]["path"]
            transform["output"] = transform["output"]["name"]


def _break_coordinate_systems(ome):
    ome["coordinateSystems"] = ome["multiscales"][0].pop("coordinateSystems")


BREAKAGES = {
    "version": [_break_version],
    "references": [_break_references],
    "coordinate_systems": [_break_coordinate_systems],
    "all_three": [
        _break_version,
        _break_references,
        _break_coordinate_systems,
    ],
}


@pytest.fixture
def group_path(tmp_path):
    """Create an OME-Zarr group with tiny arrays and valid v0.6 metadata."""
    path = tmp_path / "template.ome.zarr"
    group = zarr.open_group(str(path), mode="w")
    for i in range(len(SCALES)):
        group.create_array(f"s{i}", shape=(4, 4, 4), dtype="uint16")

    metadata = json.loads((path / "zarr.json").read_text())
    metadata["attributes"] = {"ome": _valid_ome()}
    (path / "zarr.json").write_text(json.dumps(metadata, indent=2))
    return path


def _write_ome(group_path, ome):
    metadata = json.loads((group_path / "zarr.json").read_text())
    metadata["attributes"]["ome"] = ome
    (group_path / "zarr.json").write_text(json.dumps(metadata, indent=2))


def test_normalised_metadata_is_left_alone(group_path):
    """A fully normalised group is untouched, byte for byte."""
    _write_ome(group_path, _fully_wrapped_ome())
    before = (group_path / "zarr.json").read_bytes()

    assert update_ome_zarr_attributes(group_path) is False
    assert (group_path / "zarr.json").read_bytes() == before

    nz.from_ngff_zarr(str(group_path))


@pytest.mark.parametrize("name", list(BREAKAGES))
def test_malformed_metadata_becomes_readable(group_path, name):
    """Each malformation is independently fatal, and independently fixed."""
    ome = _valid_ome()
    for break_it in BREAKAGES[name]:
        break_it(ome)
    _write_ome(group_path, ome)

    with pytest.raises(Exception):
        nz.from_ngff_zarr(str(group_path))

    assert update_ome_zarr_attributes(group_path) is True

    multiscale = nz.from_ngff_zarr(str(group_path))
    assert multiscale.images[0].scale == {"z": 0.01, "y": 0.01, "x": 0.01}


def test_repair_is_idempotent(group_path):
    """A second pass finds nothing left to change."""
    ome = _valid_ome()
    for break_it in BREAKAGES["all_three"]:
        break_it(ome)
    _write_ome(group_path, ome)

    assert update_ome_zarr_attributes(group_path) is True
    repaired = (group_path / "zarr.json").read_bytes()

    assert update_ome_zarr_attributes(group_path) is False
    assert (group_path / "zarr.json").read_bytes() == repaired


def test_references_are_wrapped_beyond_the_dataset_level(group_path):
    """Bare-string endpoints are wrapped wherever they appear.

    The bucket only malforms the dataset-level sequences, but the shim
    normalises the multiscale-level one too, resolving each reference
    against the declared array paths.
    """
    _write_ome(group_path, _valid_ome())
    update_ome_zarr_attributes(group_path)

    ome = json.loads((group_path / "zarr.json").read_text())["attributes"][
        "ome"
    ]
    transform = ome["multiscales"][0]["coordinateTransformations"][0]

    assert transform["input"] == {"name": "intrinsic"}
    assert transform["output"] == {"name": "mm RAS"}


def test_coordinate_systems_kept_when_there_is_no_multiscale(tmp_path):
    """Without a multiscale to move them into, the block stays put."""
    path = tmp_path / "empty.ome.zarr"
    zarr.open_group(str(path), mode="w")
    systems = [{"name": "intrinsic", "axes": _axes()}]
    _write_ome(path, {"version": "0.6", "coordinateSystems": systems})

    assert update_ome_zarr_attributes(path) is False

    ome = json.loads((path / "zarr.json").read_text())["attributes"]["ome"]
    assert ome["coordinateSystems"] == systems


def test_missing_or_non_ome_group_is_a_no_op(tmp_path):
    """Safe to call on anything that is not an OME-Zarr group."""
    assert update_ome_zarr_attributes(tmp_path / "nonexistent") is False

    plain = tmp_path / "plain.zarr"
    zarr.open_group(str(plain), mode="w")
    assert update_ome_zarr_attributes(plain) is False


def test_release_versions_are_not_rewritten(group_path):
    """Only ``.devN`` suffixes are stripped."""
    ome = _valid_ome()
    ome["version"] = "0.5"
    _write_ome(group_path, ome)

    update_ome_zarr_attributes(group_path)

    reread = json.loads((group_path / "zarr.json").read_text())
    assert reread["attributes"]["ome"]["version"] == "0.5"


def test_repairs_the_real_remote_shape(group_path):
    """End to end on a faithful copy of what the bucket serves."""
    remote = _valid_ome()
    for break_it in BREAKAGES["all_three"]:
        break_it(remote)
    _write_ome(group_path, copy.deepcopy(remote))

    assert update_ome_zarr_attributes(group_path) is True

    ome = json.loads((group_path / "zarr.json").read_text())["attributes"][
        "ome"
    ]
    assert ome["version"] == "0.6"
    assert "coordinateSystems" not in ome
    assert "coordinateSystems" in ome["multiscales"][0]
    dataset = ome["multiscales"][0]["datasets"][0]
    assert dataset["coordinateTransformations"][0]["input"] == {"path": "s0"}
    assert dataset["coordinateTransformations"][0]["output"] == {
        "name": "intrinsic"
    }
