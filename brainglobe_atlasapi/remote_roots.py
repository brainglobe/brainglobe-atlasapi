"""Work out which remote root an atlas comes from and where it is cached.

Each remote root is mirrored in its own directory under the brainglobe
directory, holding a record of the root it mirrors.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Optional

from brainglobe_atlasapi import descriptors
from brainglobe_atlasapi.descriptors import V3_ATLAS_ROOTDIR


def cache_key(remote_root: str) -> str:
    """Return the local cache directory name for a remote root.

    The BrainGlobe root keeps its historical directory; every other root
    is cached under the last segment of its URL, e.g.
    ``s3://bucket/prefix/allen-atlas-assets-rc11`` under
    ``allen-atlas-assets-rc11``.

    Parameters
    ----------
    remote_root : str
        Remote root, without a trailing slash.

    Returns
    -------
    str
        Directory name, relative to the brainglobe directory.

    Raises
    ------
    ValueError
        If a non-default root would take the default root's directory.
    """
    if remote_root == descriptors.DEFAULT_REMOTE_ROOT:
        return descriptors.DEFAULT_ROOT_KEY
    key = remote_root.removeprefix("s3://").rsplit("/", 1)[-1]
    if key == descriptors.DEFAULT_ROOT_KEY:
        raise ValueError(
            f"{remote_root} would share the cache directory reserved "
            f"for {descriptors.DEFAULT_REMOTE_ROOT}."
        )
    return key


def check_remote_root(cache_dir: Path, remote_root: str) -> None:
    """Check a cache directory does not mirror a different remote root.

    Two roots sharing a last URL segment map to the same directory; the
    record written by ``record_remote_root`` keeps them from mixing.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote_root``.
    remote_root : str
        Remote root, without a trailing slash.

    Raises
    ------
    ValueError
        If the directory already records a different remote root.
    """
    record = cache_dir / descriptors.REMOTE_ROOT_FILENAME
    if record.exists():
        recorded = record.read_text().strip()
        if recorded != remote_root:
            raise ValueError(
                f"{cache_dir} holds atlases from {recorded}, not "
                f"{remote_root}. Remote roots sharing a last URL segment "
                "cannot share a cache directory."
            )


def record_remote_root(cache_dir: Path, remote_root: str) -> None:
    """Record which remote root a cache directory mirrors, if unrecorded.

    Called only once the directory holds an atlas, so a failed lookup
    (a typo'd root or atlas name) never claims the directory.

    Parameters
    ----------
    cache_dir : Path
        Local cache directory for ``remote_root``.
    remote_root : str
        Remote root, without a trailing slash.
    """
    record = cache_dir / descriptors.REMOTE_ROOT_FILENAME
    if record.exists():
        return
    try:
        record.write_text(f"{remote_root}\n")
    except OSError:
        # A read-only shared cache still loads; it just goes unrecorded.
        pass


def resolve_remote_root(
    fs, atlas_name: str, roots, is_reachable: Callable[[str], bool]
) -> Optional[str]:
    """Find which configured root holds an atlas.

    Roots are searched in declaration order, so the first-declared root
    wins when a name exists in more than one bucket. Each root is probed
    for reachability only when its turn comes, and unreachable roots are
    skipped, so finding an atlas on the first root costs two requests.

    Parameters
    ----------
    fs : s3fs.S3FileSystem
        Filesystem used to list the atlas directory.
    atlas_name : str
        Name of the atlas to locate.
    roots : dict
        Mapping of root key to remote root, in resolution order.
    is_reachable : callable
        Returns whether a remote root answers.

    Returns
    -------
    str or None
        The winning remote root, or None if no root is reachable.

    Raises
    ------
    ValueError
        If the atlas name is present in none of the reachable roots.
    """
    searched = []
    for remote_root in roots.values():
        if not is_reachable(remote_root):
            continue
        searched.append(remote_root)
        try:
            if fs.ls(f"{remote_root}/{V3_ATLAS_ROOTDIR}/{atlas_name}"):
                return remote_root
        except FileNotFoundError:
            # Not on this root (ls may raise or return []): try the next.
            pass

    if not searched:
        return None

    raise ValueError(
        f"{atlas_name} is not a valid atlas name! "
        f"Searched remote roots: {', '.join(searched)}"
    )


def known_remote_roots(base_dir: Path, roots: dict) -> list:
    """List the remote roots whose local cache may hold an atlas.

    Parameters
    ----------
    base_dir : Path
        Directory holding one cache directory per remote root.
    roots : dict
        Mapping of root key to configured remote root.

    Returns
    -------
    list of str
        Configured roots in config order, then any other root recorded
        under ``base_dir``, without duplicates.
    """
    records = sorted(base_dir.glob(f"*/{descriptors.REMOTE_ROOT_FILENAME}"))
    recorded_roots = [record.read_text().strip() for record in records]
    return list(dict.fromkeys([*roots.values(), *recorded_roots]))
