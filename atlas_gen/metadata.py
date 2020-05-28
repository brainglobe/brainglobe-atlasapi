import requests
import re
from requests.exceptions import MissingSchema, InvalidURL, ConnectionError


def generate_metadata_dict(
    name, citation, atlas_link, species, symmetric, resolution, version, shape
):

    # We ask for a rigid naming convention to be followed:
    parsename = name.split("_")
    assert len(parsename) >= 3
    assert re.match("[0-9]+um", parsename[-1])

    # Control version formatting:
    assert re.match("[0-9]+\\.[0-9]+", version)

    # We ask for DOI and correct link only if atlas is published:
    if citation != "unpublished":
        assert "doi" in citation

        # Test url:
        try:
            _ = requests.get(atlas_link)
        except (MissingSchema, InvalidURL, ConnectionError):
            raise InvalidURL(
                "Ensure that the url is valid and formatted correctly!"
            )

    # Enforce correct format for symmetric, resolution and shape:
    assert type(symmetric) == bool
    assert len(resolution) == 3
    assert len(shape) == 3

    resolution = tuple([float(v) for v in resolution])
    shape = tuple(int(v) for v in shape)

    return dict(
        name=name,
        citation=citation,
        atlas_link=atlas_link,
        species=species,
        symmetric=symmetric,
        resolution=resolution,
        version=version,
        shape=shape,
    )
