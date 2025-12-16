from brainglobe_atlasapi.bg_atlas import BrainGlobeAtlas

def test_recovers_missing_metadata(mocker):
    """
    Test that BrainGlobeAtlas recovers from missing metadata by re-downloading.
    """
    
    # Simulate failure on first init, success on second
    # We mock 'brainglobe_atlasapi.core.Atlas.__init__' specifically.
    mock_atlas_init = mocker.patch(
        "brainglobe_atlasapi.core.Atlas.__init__",
        side_effect=[FileNotFoundError("Missing metadata"), None],
        autospec=True,
    )

    # Prevent real IO - patch only relevant methods
    mock_download = mocker.patch(
        "brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas.download_extract_file"
    )
    mocker.patch(
        "brainglobe_atlasapi.bg_atlas.BrainGlobeAtlas.check_latest_version"
    )
    # Patch shutil.rmtree to verify clean up without touching filesystem
    mock_rmtree = mocker.patch(
        "brainglobe_atlasapi.bg_atlas.shutil.rmtree"
    )

    
    # Act
    BrainGlobeAtlas("example_mouse_100um", check_latest=False)
    
    # Assert recovery behavior
    mock_download.assert_called_once()
    assert mock_atlas_init.call_count == 2
    mock_rmtree.assert_called_once()
