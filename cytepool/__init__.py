"""Statistics on pseudobulk and bulk omics data."""

try:
    from ._version import version as __version__
except ImportError:  # a source tree that was never installed has no version file
    __version__ = "0+unknown"

__all__ = ["__version__"]
