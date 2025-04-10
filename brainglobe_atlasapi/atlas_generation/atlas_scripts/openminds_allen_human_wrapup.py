import json
from datetime import datetime


def wrapup_atlas_to_openminds(
    atlas_name,
    version,
    species,
    citation,
    atlas_link,
    orientation,
    resolution,
    root_id,
    structures_list,
    meshes_dict,
    working_dir,
):
    """
    Wrap up atlas data into an OpenMINDS SANDS compliant JSON file.
    """
    # Construct the atlas metadata according to OpenMINDS SANDS schema.
    # Note: This is a simplified mapping – please verify with the schema.
    openminds_atlas = {
        "identifier": f"{atlas_name.lower()}_v{version}",
        "name": atlas_name,
        "version": str(version),
        "species": species,
        "citation": citation,
        "referenceAtlasLink": atlas_link,
        "coordinateSystem": orientation,  # Assuming orientation maps directly.
        "resolution": {
            "x": resolution[0],
            "y": resolution[1],
            "z": resolution[2],
        },
        "creationDate": datetime.utcnow().isoformat() + "Z",
        "structures": [],
    }

    # Iterate over structures_list to build structures entry.
    # I assume each structure is a dict with keys: 'id', 'name', 'acronym'
    # and that meshes_dict maps structure IDs to file paths.
    for s in structures_list:
        struct_id = s.get("id")
        structure_entry = {
            "identifier": f"{atlas_name.lower()}_structure_{struct_id}",
            "name": s.get("name"),
            "acronym": s.get("acronym"),
            "meshFile": (
                str(meshes_dict.get(struct_id))
                if meshes_dict.get(struct_id)
                else None
            ),
            "structureIdPath": s.get("structure_id_path", []),
            # Additional fields like color or other
            #  properties could be added here.
        }
        openminds_atlas["structures"].append(structure_entry)

    # Define output filename based on atlas name and version.
    output_filename = working_dir / f"{atlas_name}_v{version}_openminds.json"

    with open(output_filename, "w") as f:
        json.dump(openminds_atlas, f, indent=4)

    print(f"Atlas written to OpenMINDS SANDS format at: {output_filename}")
    return output_filename
