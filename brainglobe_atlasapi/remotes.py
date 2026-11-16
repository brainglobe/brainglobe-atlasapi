"""Work out which remote an atlas comes from and where it is cached.

Each remote is mirrored in its own directory under the brainglobe
directory, holding a record of the remote it mirrors.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.descriptors import V3_ATLAS_ROOTDIR


def cache_key(remote: str) -> str:
    """Return the local cache directory name for a remote.

    The BrainGlobe remote keeps its historical directory; every other remote
    is cached under the last segment of its URL, e.g.
    ``s3://bucket/prefix/allen-atlas-assets-rc11`` under
    ``allen-atlas-assets-rc11``.

    Parameters
    ----------
    remote : str
        Remote, without a trailing slash.

    Returns
    -------
    str
        Directory name, relative to the brainglobe directory.

    Raises
    ------
    ValueError
        If a non-default remote would take the default remote's directory.
    """
    if remote == descriptors.DEFAULT_REMOTE:
        return descriptors.DEFAULT_REMOTE_KEY
    key = remote.removeprefix("s3://").rsplit("/", 1)[-1]
    if key == descriptors.DEFAULT_REMOTE_KEY:
        raise ValueError(
            f"{remote} would share the cache directory reserved "
            f"for {descriptors.DEFAULT_REMOTE}."
        )
    return key


def check_remote(cache_dir: Path, remote: str) -> None:
    """Check a cache directory does not mirror a different remote.

    Two remotes sharing a last URL segment map to the same directory; the
    record written by ``record_remote`` keeps them from mixing.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote``.
    remote : str
        Remote, without a trailing slash.

    Raises
    ------
    ValueError
        If the directory already records a different remote.
    """
    record = cache_dir / descriptors.REMOTE_FILENAME
    if record.exists():
        recorded = record.read_text().strip()
        if recorded != remote:
            raise ValueError(
                f"{cache_dir} holds atlases from {recorded}, not "
                f"{remote}. Remotes sharing a last URL segment "
                "cannot share a cache directory."
            )


def record_remote(cache_dir: Path, remote: str) -> None:
    """Record which remote a cache directory mirrors, if unrecorded.

    Called only once the directory holds an atlas, so a failed lookup
    (a typo'd remote or atlas name) never claims the directory.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote``.
    remote : str
        Remote, without a trailing slash.
    """
    record = cache_dir / descriptors.REMOTE_FILENAME
    if record.exists():
        return
    try:
        record.write_text(f"{remote}\n")
    except OSError:
        # A read-only shared cache still loads; it just goes unrecorded.
        pass


def resolve_remote(
    fs, atlas_name: str, remotes, is_reachable: Callable[[str], bool]
) -> Optional[str]:
    """Find which configured remote holds an atlas.

    Remotes are searched in declaration order, so the first-declared remote
    wins when a name exists in more than one bucket. Each remote is probed
    for reachability only when its turn comes, and unreachable remotes are
    skipped, so finding an atlas on the first remote costs two requests.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem used to list the atlas directory.
    atlas_name : str
        Name of the atlas to locate.
    remotes : dict
        Mapping of remote key to remote, in resolution order.
    is_reachable : callable
        Returns whether a remote answers.

    Returns
    -------
    str or None
        The winning remote, or None if no remote is reachable.

    Raises
    ------
    ValueError
        If the atlas name is present in none of the reachable remotes.
    """
    searched = []
    for remote in remotes.values():
        if not is_reachable(remote):
            continue
        searched.append(remote)
        try:
            if fs.ls(f"{remote}/{V3_ATLAS_ROOTDIR}/{atlas_name}"):
                return remote
        except FileNotFoundError:
            # Not on this remote (ls may raise or return []): try the next.
            pass

    if not searched:
        return None

    raise ValueError(
        f"{atlas_name} is not a valid atlas name! "
        f"Searched remotes: {', '.join(searched)}"
    )


def known_remotes(base_dir: Path, remotes: dict) -> list:
    """List the remotes whose local cache may hold an atlas.

    Parameters
    ----------
    base_dir : Path
        Directory holding one cache directory per remote.
    remotes : dict
        Mapping of remote key to configured remote.

    Returns
    -------
    list of str
        Configured remotes in config order, then any other remote recorded
        under ``base_dir``, without duplicates.
    """
    records = sorted(base_dir.glob(f"*/{descriptors.REMOTE_FILENAME}"))
    recorded_remotes = [record.read_text().strip() for record in records]
    return list(dict.fromkeys([*remotes.values(), *recorded_remotes]))
