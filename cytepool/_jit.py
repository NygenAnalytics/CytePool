"""Just-in-time compilation for the package's numerical routines.

Routines are compiled by numba on first use and cached on disk. The cache
directory is keyed by the package version, a digest of every source file in
the package, the numba version, and the numba settings that change generated
code. Numba checks only the source file of the function it loads, so a cached
routine would otherwise keep running stale machine code after a routine that
it calls, defined in another file, has changed.

Caching never fails a computation: when no cache directory is writable, or the
installed numba no longer provides the interfaces used here, routines are
compiled in every process instead.

Parallel routines use numba's threads; ``thread_limit`` caps how many of them one
call uses.
"""

import hashlib
import logging
import operator
import os
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any

import numba

from . import __version__

logger = logging.getLogger(__name__)

_PACKAGE_ROOT = Path(__file__).resolve().parent


def jit(function: Callable[..., Any] | None = None, /, **options: Any) -> Any:
    """Compile a function in numba's nopython mode with the package cache.

    Use it as ``@jit`` or with numba options, as in ``@jit(parallel=True)``.

    Args:
        function: The function to compile, when used without options.
        options: Options passed to ``numba.njit``, other than ``cache``.

    Returns:
        A numba dispatcher, or a decorator that returns one.
    """
    if "cache" in options:
        raise TypeError("jit manages the cache itself; do not pass cache")

    def decorate(func: Callable[..., Any]) -> Any:
        dispatcher = numba.njit(**options)(func)
        _attach_cache(dispatcher)
        return dispatcher

    return decorate if function is None else decorate(function)


@contextmanager
def thread_limit(n_threads: int | None) -> Iterator[int]:
    """Cap the threads that parallel routines use, then restore the previous cap.

    numba keeps the thread count per calling thread, so the cap applies to
    parallel routines called from this thread inside the block, and other threads
    are not affected. The previous value is restored on exit, also when the block
    raises.

    Args:
        n_threads: The most threads to use, at least 1. numba starts
            ``numba.config.NUMBA_NUM_THREADS`` threads (set by the environment
            variable of that name) and cannot use more, so a larger value uses all
            of them. ``None`` keeps the current setting.

    Yields:
        The number of threads parallel routines use inside the block.

    Raises:
        TypeError: If ``n_threads`` is not an integer or None.
        ValueError: If ``n_threads`` is below 1.

    Example:
        >>> with thread_limit(1) as used:
        ...     used
        1
    """
    if n_threads is None:
        yield numba.get_num_threads()
        return
    if isinstance(n_threads, bool):
        raise TypeError("n_threads must be a positive integer or None, not bool")
    try:
        limit = operator.index(n_threads)
    except TypeError:
        raise TypeError(
            "n_threads must be a positive integer or None, got "
            f"{type(n_threads).__name__}"
        ) from None
    if limit < 1:
        raise ValueError(f"n_threads must be at least 1, got {limit}")
    previous = numba.get_num_threads()
    numba.set_num_threads(min(limit, numba.config.NUMBA_NUM_THREADS))
    try:
        yield numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)


@cache
def cache_key() -> str:
    """Return the key that names this installation's cache directory."""
    parts = (
        __version__,
        _source_digest(_PACKAGE_ROOT),
        numba.__version__,
        f"boundscheck={numba.config.BOUNDSCHECK}",
        f"opt={numba.config.OPT}",
    )
    return hashlib.sha256("\n".join(map(str, parts)).encode()).hexdigest()[:16]


@cache
def cache_directory() -> Path | None:
    """Return the writable cache directory for compiled routines, or None.

    The directory is ``cytepool/<key>`` under ``NUMBA_CACHE_DIR`` when it is
    set, then under the user cache directory, then under the temporary
    directory; the first writable one is used.
    """
    key = cache_key()
    for base in _base_directories():
        directory = base / "cytepool" / key
        try:
            directory.mkdir(parents=True, exist_ok=True)
            tempfile.TemporaryFile(dir=directory).close()
        except OSError:
            continue
        return directory
    return None


def _base_directories() -> list[Path]:
    bases = []
    if numba_cache := os.environ.get("NUMBA_CACHE_DIR"):
        bases.append(Path(numba_cache))
    if user_cache := os.environ.get("XDG_CACHE_HOME"):
        bases.append(Path(user_cache))
    else:
        try:
            bases.append(Path.home() / ".cache")
        except RuntimeError:  # no home directory can be determined
            pass
    bases.append(Path(tempfile.gettempdir()))
    return bases


def _source_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix()
        # The version file is rewritten on every install; the version is keyed
        # separately.
        if relative == "_version.py":
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _attach_cache(dispatcher: Any) -> None:
    try:
        from numba.core.dispatcher import Dispatcher

        # With NUMBA_DISABLE_JIT set, numba returns the Python function.
        if not isinstance(dispatcher, Dispatcher):
            return
        dispatcher._cache = _package_cache_class()(dispatcher.py_func)
    except Exception as error:
        logger.debug("%s is compiled in every process: %s", dispatcher, error)


@cache
def _package_cache_class() -> type:
    from numba.core import caching

    class PackageLocator(caching._CacheLocator):
        """Places the cache files of package routines under the keyed directory."""

        def __init__(self, py_func: Any, py_file: str, directory: Path) -> None:
            self._py_file = py_file
            self._lineno = py_func.__code__.co_firstlineno
            self._cache_path = str(directory)

        def get_cache_path(self) -> str:
            return self._cache_path

        def get_source_stamp(self) -> str:
            # The directory key already covers every source file, so file
            # times, which a fresh checkout changes, play no part.
            return cache_key()

        def get_disambiguator(self) -> str:
            return str(self._lineno)

        @classmethod
        def from_function(cls, py_func: Any, py_file: str) -> "PackageLocator | None":
            directory = cache_directory()
            if directory is None:
                return None
            try:
                relative = Path(py_file).resolve().relative_to(_PACKAGE_ROOT)
            except ValueError:
                return None
            locator = cls(py_func, py_file, directory / relative.with_suffix(""))
            try:
                locator.ensure_cache_path()
            except OSError:
                return None
            return locator

    class PackageCacheImpl(caching.CompileResultCacheImpl):
        _locator_classes = [PackageLocator]

    class PackageFunctionCache(caching.FunctionCache):
        _impl_class = PackageCacheImpl

    return PackageFunctionCache
