from brainatlas_api.atlas_gen.descriptors import (
    STRUCTURE_TEMPLATE as STEMPLATE,
)


def check_struct_consistency(structures):
    """Ensures internal consistency of the structures list
    Parameters
    ----------
    structures

    Returns
    -------

    """
    assert type(structures) == list
    assert type(structures[0]) == dict

    # Check that all structures have the correct keys and value types:
    for struct in structures:
        try:
            assert struct.keys() == STEMPLATE.keys()
            assert [
                isinstance(v, type(STEMPLATE[k])) for k, v in struct.items()
            ]
        except AssertionError:
            raise AssertionError(
                f"Inconsistencies found for structure {struct}"
            )
