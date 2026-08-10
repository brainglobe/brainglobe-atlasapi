"""Module containing the fsspec callbacks used to report download progress."""

from collections.abc import Callable
from typing import Optional

from fsspec.callbacks import Callback, TqdmCallback


class _ProgressCallback(TqdmCallback):
    """Show a tqdm bar and report progress to an external handler.

    ``fsspec`` drives the callback passed to ``get`` with a file count, and
    asks it for a per-file callback through ``branched`` which is driven with
    a byte count. ``fn_update`` is documented to take completed and total
    bytes, so it is attached to the branched callbacks.

    Parameters
    ----------
    fn_update : Callable
        Handler called as ``fn_update(completed, total)`` with the bytes
        transferred so far and the total size of the file being fetched.
    """

    def __init__(self, fn_update: Callable[[int, int], None], **kwargs):
        super().__init__(**kwargs)
        self.fn_update = fn_update

    def branched(self, path_1, path_2, **kwargs):
        """Return a per-file callback that reports bytes to ``fn_update``."""
        kwargs["callback"] = Callback(
            hooks={
                "fn_update": lambda size, value, **_: self.fn_update(
                    value, size or 0
                )
            }
        )
        return super().branched(path_1, path_2, **kwargs)


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
