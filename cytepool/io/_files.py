"""Which local file a path names, and whether a file changed while it was read.

The reader of Zarr stores and the cube files share these:

- ``local_path``: the absolute path that a str or os.PathLike names, refusing bytes
  and URLs.
- ``file_stamp``: the device, inode, size and modification time of a file.
- ``same_file``: whether two stamps describe one unchanged file.
"""

import os
from pathlib import Path
from typing import Any


def local_path(path: Any) -> Path:
    if not isinstance(path, str | os.PathLike):
        raise TypeError(f"path must be a str or os.PathLike, got {type(path).__name__}")
    text = os.fspath(path)
    if not isinstance(text, str):
        raise TypeError("path must be text, not bytes")
    if "://" in text:
        raise ValueError(
            f"{text!r} is a URL; only local directories are supported (zarr-python "
            "reads remote stores)"
        )
    return Path(os.path.abspath(text))


def file_stamp(path: str | os.PathLike[str]) -> tuple[int, int, int, int]:
    """Return the device, inode, size and modification time of a file."""
    status = os.stat(path)
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def same_file(old: tuple[int, ...], new: tuple[int, ...]) -> bool:
    """Return whether two stamps describe one unchanged file.

    The inode is compared only on the same device: another mount of the same files,
    in another process or on another machine, may number inodes differently.
    """
    return old == new if old[0] == new[0] else old[2:] == new[2:]
