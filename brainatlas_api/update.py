from rich import print as rprint
import shutil
import argparse

import brainatlas_api
from brainatlas_api.bg_atlas import _version_str_from_tuple


def update_atlas(atlas_name):
    # Check input
    if not isinstance(atlas_name, str):
        raise ValueError("atlas name should be a string")

    # Get atlas class
    atlasclass = brainatlas_api.get_atlas_class_from_name(atlas_name)
    if atlasclass is None:
        return

    atlas = atlasclass()

    # Check if we need to update
    if atlas.check_lateset_version():
        rprint(
            f"[b][magenta2]Brainatlas_api: {atlas.atlas_name} is already updated "
            + f"(version: {_version_str_from_tuple(atlas.local_version)})[/b]"
        )
        # return

    # Delete atlas folder
    rprint(
        f"[b][magenta2]Brainatlas_api: updating {atlas.atlas_name}[/magenta2][/b]"
    )
    fld = atlas.brainglobe_dir / atlas.local_full_name
    shutil.rmtree(fld)

    # Download again
    atlas.download_extract_file()

    # Check that everything went well
    atlasclass()
    rprint(
        f"[b][magenta2]Brainatlas_api: {atlas.atlas_name} updated to version: "
        + f"{_version_str_from_tuple(atlas.remote_version)}[/magenta2][/b]"
    )


def update_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-a",
        "--atlas",
        dest="atlas",
        required=True,
        help="name of the atlas to update",
        type=str,
    )

    return parser


def main():
    args = update_parser().parse_args()
    update_atlas(args.atlas)
    return
