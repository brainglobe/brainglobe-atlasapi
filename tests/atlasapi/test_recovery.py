from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

def test_recovers_missing_metadata(mocker):
    """
    Test that BrainGlobeAtlas recovers from missing metadata by re-downloading.
    """

    # Mock Atlas.__init__: fail once, succeed second time
    mock_atlas_init = mocker.patch(
        "brainglobe_atlasapi.core.Atlas.__init__",
        side_effect=[FileNotFoundError("Missing metadata"), None],
        autospec=True,
    )

    # Mock local_full_name to simulate a valid atlas folder
    mocker.patch(
        "brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas.local_full_name",
        new_callable=mocker.PropertyMock,
        return_value="example_mouse_100um_v1.0",
    )

    # Prevent real IO
    mock_download = mocker.patch(
        "brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas.download_extract_file"
    )
    mocker.patch(
        "brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas.check_latest_version"
    )

    mock_rmtree = mocker.patch(
        "brainglobe_atlasapi.bg_atlas.shutil.rmtree"
    )

    # Act
    BrainGlobeAtlas("example_mouse_100um", check_latest=False)

    # Assert
    assert mock_atlas_init.call_count == 2
    mock_download.assert_called_once()


