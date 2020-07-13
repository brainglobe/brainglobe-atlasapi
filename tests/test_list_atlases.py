from bg_atlasapi.list_atlases import (
    get_downloaded_atlases,
    get_atlases_lastversions,
    show_atlases,
    get_local_atlas_version,
)


def test_get_downloaded_atlases():
    available_atlases = get_downloaded_atlases()

    # Check that example is listed:
    assert "example_mouse_100um" in available_atlases

    # Check that all listed ar valid atlas names:
    assert all([len(a.split("_")) == 3 for a in available_atlases])


def test_get_local_atlas_version():
    v = get_local_atlas_version("example_mouse_100um")

    assert len(v.split(".")) == 2


def test_lastversions():
    last_versions = get_atlases_lastversions()
    example_atlas = last_versions["example_mouse_100um"]

    local_v = get_local_atlas_version("example_mouse_100um")

    assert example_atlas["version"] == local_v
    assert all(
        [
            int(l) <= int(r)
            for l, r in zip(
                example_atlas["latest_version"].split("."), local_v.split(".")
            )
        ]
    )


def test_show_atlases():
    # TODO add more valid testing than just look for errors when running:
    show_atlases(show_local_path=True)
