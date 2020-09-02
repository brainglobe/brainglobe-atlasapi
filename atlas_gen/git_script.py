from git import Repo
from pathlib import Path
import tempfile
from bg_atlasapi.config import read_config
import configparser
import atlas_gen
from importlib import import_module

from atlas_gen.atlas_scripts.example_mouse import create_atlas


generation_dict = dict(example_mouse=[100])


def get_atlas_repr(name):
    parts = name.split("_")
    # if atlas name with no version:
    version_str = parts.pop() if not parts[-1].endswith("um") else None
    resolution_str = parts.pop()

    atlas_name = "_".join(parts)
    if version_str:
        major_vers, minor_vers = version_str[2:].split(".")
    else:
        major_vers, minor_vers = None, None
    return dict(name=atlas_name,
                major_vers=major_vers,
                minor_vers=minor_vers,
                resolution=resolution_str[:-2])



cwd = Path.home() / "bg_auto"
cwd.mkdir(exist_ok=True)


if __name__ == "__main__":
    repo_path = cwd / "atlas_repo"
    atlas_gen_path = Path(__file__).parent
    # Repo.clone_from("https://gin.g-node.org/vigji/bg_test", repo_path)
    # us = input("GIN-GNode user: ")  # Python 3
    # pw = input("GIN-GNode password: ")  # Python 3
    # print(us, pw)

    conf = configparser.ConfigParser()
    conf.read(repo_path / "last_versions.conf")
    atlases_status = dict()
    for k in conf["atlases"].keys():
        repr = get_atlas_repr(k)

        # Read versions from conf:
        major_vers, minor_vers = conf["atlases"][k].split(".")
        repr["major_vers"] = major_vers
        repr["minor_vers"] = minor_vers
        atlases_status[repr.pop("name")] =repr

    bg_atlas_version = atlas_gen.__version__

    scripts_path = atlas_gen_path / "atlas_scripts"

    for n, res in generation_dict.items():
        # print(next(scripts_path.glob(f"{n}.py")))
        # print(n)
        status = atlases_status[n]
        mod = import_module(f"atlas_gen.atlas_scripts.{n}")
        script_version = mod.__version__
        if bg_atlas_version >= status["major_vers"] and \
                script_version > status["minor_vers"]:
            print(n, mod.create_atlas)



