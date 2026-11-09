"""Atlas generation script for the symmetric NMT CHARM and SARM atlases.

The symmetric and asymmetric NMT v2.1 datasets share the same layout, so
this reuses the asymmetric script, swapping in the symmetric dataset and
packaging without a hemisphere map.
"""

try:
    from atlas_scripts.nmt_charm_sarm_asym_macaque import create_atlases
except ModuleNotFoundError:  # run directly as ``python <this file>``
    from nmt_charm_sarm_asym_macaque import create_atlases

### Metadata
__version__ = 0


if __name__ == "__main__":
    create_atlases("sym", minor_version=__version__)
