"""Statistics on pseudobulk and bulk omics data."""

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _distribution_version


def _installed_version() -> str:
    # The metadata of an install names the commit it was built from. The version file
    # that the build also writes can lag behind it: an install reused from the cache
    # leaves the file of a later build in place.
    try:
        return _distribution_version("cytepool")
    except _PackageNotFoundError:  # a source tree on the path, never installed
        try:
            from ._version import version
        except ImportError:
            return "0+unknown"
        return version


__version__ = _installed_version()

__all__ = ["__version__"]
