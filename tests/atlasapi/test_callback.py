"""Test the fsspec callbacks used to report download progress."""

import fsspec
import pytest

from brainglobe_atlasapi.callback import AtlasCallback


@pytest.mark.parametrize(
    "recursive",
    [pytest.param(False, id="one file"), pytest.param(True, id="directory")],
)
def test_atlas_callback_reports_file_counts(tmp_path, recursive):
    """`AtlasCallback` reports fetched file counts to `fn_update`.

    An atlas is an OME-Zarr store, so byte progress would tick through
    thousands of small chunks. The file count is the useful proxy.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary path to fetch files into.
    recursive : bool
        Whether to fetch a whole directory or a single file.
    """
    calls = []
    memory_fs = fsspec.filesystem("memory")
    memory_fs.pipe("/atlas/first.json", b"a" * 4096)
    memory_fs.pipe("/atlas/second.json", b"b" * 2048)

    source = "/atlas/" if recursive else "/atlas/first.json"
    memory_fs.get(
        source,
        str(tmp_path / "destination"),
        recursive=recursive,
        callback=AtlasCallback(
            lambda completed, total: calls.append((completed, total))
        ),
    )

    assert calls, "fn_update was never called"
    assert all(0 <= completed <= total for completed, total in calls)

    totals = {total for _, total in calls}
    assert len(totals) == 1, f"the total moved during the fetch: {totals}"
    total = totals.pop()

    # The total is a file count, so it stays small and is never one of the
    # 4096 or 2048 byte sizes written above. The exact number is left to
    # fsspec, which counts the directory entry as well as its files.
    assert total <= 10
    assert total not in (4096, 2048)
    assert calls[-1] == (total, total)


def test_atlas_callback_without_fn_update(tmp_path):
    """`AtlasCallback` still transfers files when no handler is given."""
    memory_fs = fsspec.filesystem("memory")
    memory_fs.pipe("/atlas/first.json", b"a" * 8)

    memory_fs.get(
        "/atlas/first.json",
        str(tmp_path / "first.json"),
        callback=AtlasCallback(),
    )

    assert (tmp_path / "first.json").read_bytes() == b"a" * 8
