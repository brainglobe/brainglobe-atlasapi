"""Test the StructuresDict class for handling atlas structures."""

import meshio as mio
import pytest

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_atlasapi.utils import read_json

structures_list = [
    {
        "acronym": "root",
        "id": 997,
        "name": "root",
        "structure_id_path": [997],
        "rgb_triplet": [255, 255, 255],
        "mesh_filename": None,
    },
    {
        "acronym": "grey",
        "id": 8,
        "name": "Basic cell groups and regions",
        "structure_id_path": [997, 8],
        "rgb_triplet": [191, 218, 227],
        "mesh_filename": None,
    },
    {
        "acronym": "CH",
        "id": 567,
        "name": "Cerebrum",
        "structure_id_path": [997, 8, 567],
        "rgb_triplet": [176, 240, 255],
        "mesh_filename": None,
    },
]


@pytest.mark.filterwarnings("ignore:No mesh filename for region root")
def test_structure_indexing():
    """Test various indexing methods for StructuresDict.

    Verify that structures can be accessed by integer ID, float ID,
    and string ID, and that mesh loading errors are handled.
    """
    structures_dict = StructuresDict(structures_list)
    print(structures_dict)
    assert structures_dict[997] == structures_dict["root"]
    assert structures_dict[997.0] == structures_dict["root"]
    assert structures_dict["997"] == structures_dict["root"]

    with pytest.raises(mio.ReadError) as error:
        structures_dict["997"]["mesh_filename"] = "wrong_filename.smtg"
        _ = structures_dict["997"]["mesh"]
    print(str(error))
    assert "" in str(error)


def test_mesh_loading(atlas_path):
    """Load meshes from a StructuresDict and verify type.

    Parameters
    ----------
    atlas_path : Path
        Path to the test atlas directory.
    """
    structures_list_real = read_json(
        atlas_path / descriptors.STRUCTURES_FILENAME
    )

    # Add entry for file paths:
    for struct in structures_list_real:
        struct["mesh_filename"] = (
            atlas_path
            / descriptors.MESHES_DIRNAME
            / "{}.obj".format(struct["id"])
        )

    struct_dict = StructuresDict(structures_list_real)
    assert isinstance(struct_dict["997"]["mesh"], mio.Mesh)
