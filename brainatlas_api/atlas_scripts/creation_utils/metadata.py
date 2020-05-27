import requests

# Entries and types from this template will be used to check atlas info
# consistency. Please keep updated both this and the function when changing the structure.

METADATA_TEMPLATE = {
    "name": "test",
    "citation": "Kunst et al 2019, https://doi.org/10.1016/j.neuron.2019.04.034",
    "atlas_link": "https://fishatlas.neuro.mpg.de",
    "species": "Danio rerio",
    "symmetric": False,
    "resolution": (0.994, 1., 0.994),
    "shape": (100, 50, 100),
    "version": "0.0",
}

def generate_metadata_dict(name,
                           citation,
                           atlas_link,
                           species,
                           symmetric,
                           resolution,
                           shape):

    if "doi" not in citation:
        if "unpublished" not in citation:
            raise ValueError("The citation field should contained a doi or specify 'unpublished'")

    # Enforce correct format for resolution and shape:
    resolution = tuple([float(v) for v in resolution])
    shape = tuple(int(v)for v in shape)

    metadata_dict = dict(name=name,
                         citation=citation,
                         atlas_link=atlas_link,
                         species=species,
                         symmetric=symmetric,
                         resolution=resolution,
                         shape=shape)



