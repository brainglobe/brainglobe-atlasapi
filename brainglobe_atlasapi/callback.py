"""Module containing the fsspec callbacks used to report download progress."""

from collections.abc import Callable
from typing import Optional

from fsspec.callbacks import TqdmCallback


class _ProgressCallback(TqdmCallback):
    """Show a tqdm bar and report progress to an external handler.

    ``fsspec`` drives the callback passed to ``get`` with a file count, and
    that count is what is reported. An atlas is an OME-Zarr store, so a byte
    count would tick through thousands of small chunks and read as a stalling
    bar, while the file count advances once per file.

    ``TqdmCallback`` replaces :meth:`fsspec.callbacks.Callback.call` with one
    that only drives the bar and never runs ``hooks``, so the handler is
    invoked from an override rather than passed in as a hook.

    Parameters
    ----------
    fn_update : Callable
        Handler called as ``fn_update(completed, total)`` with the number of
        files fetched so far and the total number of files to fetch.
    """

    def __init__(self, fn_update: Callable[[int, int], None], **kwargs):
        super().__init__(**kwargs)
        self.fn_update = fn_update

    def call(self, *args, **kwargs):
        """Advance the tqdm bar, then report the file count to the handler."""
        super().call(*args, **kwargs)
        self.fn_update(self.value, self.size or 0)


def _download_callback(
    fn_update: Optional[Callable[[int, int], None]],
) -> TqdmCallback:
    """Build the fsspec callback used for a single atlas download step.

    Parameters
    ----------
    fn_update : Callable, optional
        Handler to report download progress to. If None, only the tqdm
        progress bar is shown.

    Returns
    -------
    TqdmCallback
        The callback to pass to ``fsspec``.
    """
    if fn_update is None:
        return TqdmCallback()
    return _ProgressCallback(fn_update)
