import configparser
import errno
import os
import shutil
import stat
from importlib import import_module
from pathlib import Path

from git import Repo
from git.exc import GitCommandError

import brainglobe_atlasapi.atlas_generation
from brainglobe_atlasapi.utils import (
    atlas_name_from_repr,
    atlas_repr_from_name,
)

# Main dictionary specifying which atlases to generate
# and with which resolutions:
GENERATION_DICT = dict(
    mpin_zfish=[1],
    allen_mouse=[10, 25, 50, 100],
    kim_mouse=[10, 25, 50, 100],
    osten_mouse=[10, 25, 50, 100],
    example_mouse=[100],
)


CWD = Path.home() / "bg_auto"


def handleRemoveReadonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)
    else:
        raise


def delete_folder(path):
    shutil.rmtree(path, ignore_errors=False, onerror=handleRemoveReadonly)


CWD.mkdir(exist_ok=True)


if __name__ == "__main__":
    repo_path = CWD / "atlas_repo"
    if repo_path.exists():
        repo = Repo(repo_path)
        repo.git.pull()
    else:
        repo_path.mkdir(exist_ok=True)

        print("Cloning atlases repo...")
        repo = Repo.clone_from(
            "https://gin.g-node.org/brainglobe/atlases", repo_path
        )
    # us = input("GIN-GNode user: ")
    # pw = input("GIN-GNode password: ")

    # Read last versions from conf file:
    conf = configparser.ConfigParser()
    conf.read(repo_path / "last_versions.conf")

    # Find all atlases representation given the names in the conf:
    atlases_repr = dict()
    for k in conf["atlases"].keys():
        repr = atlas_repr_from_name(k)
        # Read versions from conf:
        repr["major_vers"], repr["minor_vers"] = conf["atlases"][k].split(".")
        # Add as entries in a dict:
        atlases_repr[repr.pop("name")] = repr

    # Major version is given by version of the atlas_gen module:
    brainglobe_atlasapi.atlas_generation_version = (
        brainglobe_atlasapi.atlas_generation.__version__
    )

    # Path to the scripts to generate the atlases:
    atlas_gen_path = Path(__file__).parent
    scripts_path = atlas_gen_path / "atlas_scripts"

    # Loop over the entries from the GENERATION_DICT configuration dict
    commit_log = "Updated: "
    for name, resolutions in GENERATION_DICT.items():
        status = atlases_repr[name]
        module = import_module(
            f"brainglobe_atlasapi.atlas_generation.atlas_scripts.{name}"
        )
        script_version = module.__version__

        if brainglobe_atlasapi.atlas_generation_version > status[
            "major_vers"
        ] or (
            brainglobe_atlasapi.atlas_generation_version
            == status["major_vers"]
            and script_version > status["minor_vers"]
        ):
            # Loop over all resolutions:
            for resolution in resolutions:
                print(f"Generating {name}, {resolution} um...")

                # Make working directory for atlas generation:
                temp_dir = CWD / f"tempdir_{name}_{resolution}"
                temp_dir.mkdir(exist_ok=True)

                # Create and compress atlas:
                output_filename = module.create_atlas(temp_dir, resolution)

                # Move atlas to repo:
                shutil.move(str(output_filename), repo_path)
                # delete_folder(temp_dir)

                # Update config file with new version:
                k = atlas_name_from_repr(name, resolution)
                conf["brainglobe_atlasapi.atlas_generation"] = str(
                    f"{brainglobe_atlasapi.atlas_generation_version}.{script_version}"
                )
                with open(repo_path / "last_versions.conf", "w") as f:
                    conf.write(f)

                # Update log for commit message:
                commit_log += f"{output_filename.stem}, "

    # Commit and push:
    try:
        repo.git.add(".")
        repo.git.commit("-m", commit_log)
    except GitCommandError:
        pass

    repo.git.push()

    # Clear folder:
    repo.close()
    # delete_folder(repo_path)
