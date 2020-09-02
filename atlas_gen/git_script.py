from git import Repo
from pathlib import Path
import tempfile
from bg_atlasapi.config import read_config
import configparser
import atlas_gen
from importlib import import_module
import shutil
from bg_atlasapi.utils import atlas_name_from_repr, atlas_repr_from_name

GENERATION_DICT = dict(example_mouse=[100])


cwd = Path.home() / "bg_auto"
cwd.mkdir(exist_ok=True)


if __name__ == "__main__":
    repo_path = cwd / "atlas_repo"
    atlas_gen_path = Path(__file__).parent
    # Repo.clone_from("https://gin.g-node.org/vigji/bg_test", repo_path)
    # us = input("GIN-GNode user: ")  # Python 3
    # pw = input("GIN-GNode password: ")  # Python 3
    # print(us, pw)

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
    bg_atlasgen_version = atlas_gen.__version__

    # Path to the scripts to generate the atlases:
    scripts_path = atlas_gen_path / "atlas_scripts"

    # Loop over the entries from the GENERATION_DICT configuration dict
    for name, resolutions in GENERATION_DICT.items():
        status = atlases_repr[name]
        module = import_module(f"atlas_gen.atlas_scripts.{name}")
        script_version = module.__version__

        if bg_atlas_version >= status["major_vers"] and \
                script_version > status["minor_vers"]:
            print(name, module.create_atlas)

            for resolution in resolutions:
                temp_dir = cwd / "tempdir"
                temp_dir.mkdir(exist_ok=True)

                output_filename = module.create_atlas(temp_dir, resolution)

                shutil.move(str(output_filename), repo_path)
                shutil.rmtree(temp_dir)

                k = atlas_name_from_repr(name, resolution)
                conf["atlases"][k] = str(f"{bg_atlasgen_version}.{script_version}")

                with open(repo_path / "last_versions.conf", "w") as f:
                    conf.write(f)
