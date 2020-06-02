from brainatlas_api.structure_class import StructuresDict


struct_list = [
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


def test_structure_indexing():
    struct_dict = StructuresDict(struct_list)
    assert struct_dict[997] == struct_dict["root"]
    assert struct_dict[997.0] == struct_dict["root"]
    assert struct_dict["997"] == struct_dict["root"]
