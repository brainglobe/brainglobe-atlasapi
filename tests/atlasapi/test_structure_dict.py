"""Test the StructuresDict class for handling atlas structures."""

import meshio as mio
import pytest

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.structure_class import StructuresDict
from brainglobe_atlasapi.utils import load_structures_from_csv

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


@pytest.mark.filterwarnings("ignore:No valid mesh for region root")
def test_structure_indexing(atlas_path):
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
        bad_path = (
            atlas_path
            / "annotation-sets"
            / "example_mouse-annotation"
            / "1_2"
            / "meshes"
            / "998"
        )
        structures_dict["root"]["mesh_filename"] = bad_path
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
    structures_list_real = load_structures_from_csv(
        atlas_path
        / "terminologies"
        / "example_mouse-terminology"
        / "1_2"
        / descriptors.V2_TERMINOLOGY_NAME
    )

    mesh_root_path = (
        atlas_path
        / "annotation-sets"
        / "example_mouse-annotation"
        / "1_2"
        / descriptors.V2_MESHES_DIRECTORY
    )

    # Add entry for file paths:
    for struct in structures_list_real:
        struct["mesh_filename"] = mesh_root_path / f"{struct['id']}"

    struct_dict = StructuresDict(structures_list_real)
    assert isinstance(struct_dict["997"]["mesh"], mio.Mesh)
