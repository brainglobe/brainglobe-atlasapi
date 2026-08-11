"""Module containing the fsspec callbacks used to report download progress."""

from collections.abc import Callable
from typing import Optional

from fsspec.callbacks import TqdmCallback


class AtlasCallback(TqdmCallback):
    """Show a tqdm bar for an atlas download and report progress onwards.

    An atlas is an OME-Zarr store, so this reports the number of files fetched
    rather than the number of bytes. Bytes would tick through thousands of
    small chunks and read as a stalling bar.

    Parameters
    ----------
    fn_update : Callable, optional
        Handler called as ``fn_update(completed, total)`` with the number of
        files fetched so far and the total number of files to fetch. If None,
        only the tqdm progress bar is shown.
    """

    def __init__(
        self,
        fn_update: Optional[Callable[[int, int], None]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.fn_update = fn_update

    def call(self, *args, **kwargs):
        """Advance the tqdm bar, then report the file count to the handler.

        ``TqdmCallback`` overrides :meth:`fsspec.callbacks.Callback.call` with
        a version that never runs ``hooks``, so the handler has to be called
        from here rather than registered as one.
        """
        super().call(*args, **kwargs)
        if self.fn_update:
            self.fn_update(self.value, self.size or 0)
