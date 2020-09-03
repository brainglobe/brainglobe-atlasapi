from git import Repo
from pathlib import Path
import configparser
import bg_atlasgen
from importlib import import_module
import shutil
from bg_atlasapi.utils import atlas_name_from_repr, atlas_repr_from_name

# Main dictionary specifying which atlases to generate
# and with which resolutions:
GENERATION_DICT = dict(example_mouse=[100])


cwd = Path.home() / "bg_auto"
cwd.mkdir(exist_ok=True)


if __name__ == "__main__":
    repo_path = cwd / "atlas_repo"
    repo_path.mkdir(exist_ok=True)

    print("Cloning atlases repo...")
    repo = Repo.clone_from("https://gin.g-node.org/vigji/bg_test", repo_path)
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
    bg_atlasgen_version = bg_atlasgen.__version__

    # Path to the scripts to generate the atlases:
    atlas_gen_path = Path(__file__).parent
    scripts_path = atlas_gen_path / "atlas_scripts"

    # Loop over the entries from the GENERATION_DICT configuration dict
    commit_log = "Updated: "
    for name, resolutions in GENERATION_DICT.items():
        status = atlases_repr[name]
        module = import_module(f"atlas_gen.atlas_scripts.{name}")
        script_version = module.__version__

        if bg_atlasgen_version >= status["major_vers"] and \
                script_version > status["minor_vers"]:

            # Loop over all resolutions:
            for resolution in resolutions:
                print(f"Generating {name}, {resolution} um...")

                # Make working directory for atlas generation:
                temp_dir = cwd / "tempdir"
                temp_dir.mkdir(exist_ok=True)

                # Create and compress atlas:
                output_filename = module.create_atlas(temp_dir, resolution)

                # Move atlas to repo:
                shutil.move(str(output_filename), repo_path)
                shutil.rmtree(temp_dir)

                # Update config file with new version:
                k = atlas_name_from_repr(name, resolution)
                conf["atlases"][k] = str(f"{bg_atlasgen_version}.{script_version}")
                with open(repo_path / "last_versions.conf", "w") as f:
                    conf.write(f)

                # Update log for commit message:
                commit_log += f"{output_filename}.name, "

    # Commit and push:
    repo.git.add(".")
    repo.git.commit('-m', commit_log)
    repo.git.push()

    # Clear folder:
    shutil.rmtree(repo_path)
