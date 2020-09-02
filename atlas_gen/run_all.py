import atlas_gen.atlas_scripts as atlas_scripts
import pkgutil
from importlib import import_module
from pathlib import Path
import shutil


# A global working directory:
temp_root_dir = Path.home() / "temp_brainglobe_workingdir"
temp_root_dir.mkdir(exist_ok=True)

# Directory where final atlases will be stored (and synch remotely):
dest_root_dir = Path.home() / "final_brainglobe_workingdir"
dest_root_dir.mkdir(exist_ok=True)

# Here we can map multiple resolutions for each script.
# It could be expanded to multiplex more params.
resolutions_dict = dict(allen_mouse_atlas=[25, 100])

# List over modules in the atlas_scripts folder:
for (_, module_name, _) in pkgutil.iter_modules(atlas_scripts.__path__):
    print(module_name)
    # Import the module:
    module = import_module(f"atlas_gen.atlas_scripts.{module_name}")

    # If create function is available:
    if "create_atlas" in dir(module):

        # If multiple resolutions are required:
        if module_name in resolutions_dict.keys():
            for res_um in resolutions_dict[module_name]:
                # Make working directory for this atlas:
                bg_root_dir = temp_root_dir / f"{module_name}_{res_um}um"
                bg_root_dir.mkdir(exist_ok=True)

                module.create_atlas(
                    version=4, res_um=res_um, bg_root_dir=bg_root_dir
                )

                compressed_file = next(
                    bg_root_dir.glob("*_*_[0-9]*um_*.*.tar.gz")
                )
                shutil.move(str(compressed_file), str(dest_root_dir))
        else:
            module.create_atlas(version=4, bg_root_dir=bg_root_dir)

            compressed_file = next(bg_root_dir.glob("*_*_[0-9]*um_*.*.tar.gz"))
            shutil.move(str(compressed_file), str(dest_root_dir))
