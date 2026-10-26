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
call uses, ``by_chunks`` runs a loop over items in parallel or on the calling
thread, and ``by_runs`` runs a routine that releases the GIL over runs of items on
Python threads.
"""

import gc
import hashlib
import logging
import os
import tempfile
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import cache
from pathlib import Path
from typing import Any, cast

import numba

from ._options import index_integer

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


def jit_numpy_errors[F: Callable[..., Any]](function: F) -> F:
    """Compile a routine with numba; division by zero gives inf or NaN, as in numpy.

    The compiled routine keeps the Python signature for type checking.
    """
    return cast(F, jit(error_model="numpy")(function))


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
    limit = index_integer(n_threads, "n_threads", expected="a positive integer or None")
    if limit < 1:
        raise ValueError(f"n_threads must be at least 1, got {limit}")
    previous = numba.get_num_threads()
    numba.set_num_threads(min(limit, numba.config.NUMBA_NUM_THREADS))
    try:
        yield numba.get_num_threads()
    finally:
        numba.set_num_threads(previous)


def by_chunks(
    parallel: Callable[..., Any],
    serial: Callable[..., Any],
    arguments: tuple[Any, ...],
    n_items: int,
) -> Any:
    """Run a compiled loop over items in parallel, or on the calling thread alone.

    ``parallel(*arguments, n_chunks)`` runs the loop in ``n_chunks`` runs of
    consecutive items on numba's threads, and ``serial(*arguments)`` runs it in
    order; both do the same operations for every item, so the result does not
    depend on the number of threads or chunks. With more than one thread and more
    than one item, the parallel form runs with about four chunks per thread, which
    balances the work. Otherwise the serial form runs without a parallel region,
    which forked processes and several Python threads at once can run whatever
    numba's threading layer.
    """
    threads = numba.get_num_threads()
    if threads > 1 and n_items > 1:
        return parallel(*arguments, max(1, min(n_items, 4 * threads)))
    return serial(*arguments)


def by_runs(run: Callable[[int, int], Any], n_items: int) -> None:
    """Run a compiled routine over runs of consecutive items on Python threads.

    ``run(start, stop)`` computes the items ``start:stop`` and releases the GIL while
    it works, as a routine compiled with ``nogil=True`` does. With more than one of
    numba's threads and more than one item, the items are split into about four runs
    per thread. The first run goes on the calling thread, so that the routine is
    compiled, or loaded from the cache, there and never on a worker thread; the
    others go to a pool of Python threads, with the garbage collector paused until
    they finish. A collection that started on a worker thread would run the
    finalizers of the objects it frees on that thread, and some objects may only be
    finalized on the thread that made them: those of Tk, which matplotlib's
    interactive backend makes, raise an error there, and a cold compile on a worker
    thread once crashed the interpreter in the collector. Each item is computed by
    the same operations whatever its run, so the result does not depend on the
    number of threads.
    """
    threads = min(numba.get_num_threads(), n_items)
    if threads <= 1:
        run(0, n_items)
        return
    n_runs = min(n_items, 4 * threads)
    bounds = [
        (k * n_items // n_runs, (k + 1) * n_items // n_runs) for k in range(n_runs)
    ]
    run(*bounds[0])
    collecting = gc.isenabled()
    gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            for _ in pool.map(lambda bound: run(*bound), bounds[1:]):
                pass
    finally:
        if collecting:
            gc.enable()


@cache
def cache_key() -> str:
    """Return the key that names this installation's cache directory.

    The key follows what the machine code depends on: the package's source, numba's
    version and the compilation settings. It does not follow the version of
    CytePool, which names the commit, so that a commit that leaves the code as it
    was keeps the cache.
    """
    parts = (
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
        # The version file is rewritten on every install and changes nothing
        # that is compiled.
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
        """Places the cache files of package routines under the keyed directory.

        It follows numba's ``UserProvidedCacheLocator``, as ``THIRD_PARTY_NOTICES.md``
        records.
        """

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
