import gc
import importlib.util
import os
import sys
import tempfile
import threading
from pathlib import Path
from types import ModuleType

import numba
import numpy as np
import pytest
from numba.core.caching import NullCache

import cytepool
from cytepool import _jit

SOURCE = """
from cytepool._jit import jit


@jit
def weighted_total(values, weights):
    total = 0.0
    for i in range(values.shape[0]):
        total += values[i] * weights[i]
    return total
"""


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A stand-in package root holding one compiled routine, with a fresh cache."""
    root = tmp_path / "package"
    root.mkdir()
    (root / "routines.py").write_text(SOURCE)
    monkeypatch.setattr(_jit, "_PACKAGE_ROOT", root.resolve())
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path / "cache"))
    _clear_memoized()
    yield root
    _clear_memoized()


def _clear_memoized() -> None:
    _jit.cache_key.cache_clear()
    _jit.cache_directory.cache_clear()


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules.pop(name, None)
    return module


def _call(module: ModuleType) -> float:
    values = np.array([1.0, 2.0, 3.0])
    weights = np.array([0.5, 0.25, 2.0])
    result = module.weighted_total(values, weights)
    assert result == pytest.approx(float(values @ weights), rel=1e-15)
    return float(result)


def test_compiled_routines_are_saved_and_reloaded_from_the_keyed_directory(
    package: Path, tmp_path: Path
) -> None:
    first = _load(package / "routines.py", "routines_first")
    _call(first)
    directory = tmp_path / "cache" / "cytepool" / _jit.cache_key() / "routines"
    assert sorted(p.suffix for p in directory.iterdir()) == [".nbc", ".nbi"]
    assert sum(first.weighted_total.stats.cache_misses.values()) == 1

    # A dispatcher created later, as in a new process, loads the machine code.
    second = _load(package / "routines.py", "routines_second")
    _call(second)
    assert sum(second.weighted_total.stats.cache_hits.values()) == 1
    assert sum(second.weighted_total.stats.cache_misses.values()) == 0


def test_editing_any_source_file_starts_a_new_cache(package: Path) -> None:
    _call(_load(package / "routines.py", "routines_before"))
    key_before = _jit.cache_key()

    # The routine itself is unchanged; only another file of the package is.
    (package / "helpers.py").write_text("SCALE = 2.0\n")
    _clear_memoized()
    assert _jit.cache_key() != key_before

    after = _load(package / "routines.py", "routines_after")
    _call(after)
    assert sum(after.weighted_total.stats.cache_hits.values()) == 0
    assert sum(after.weighted_total.stats.cache_misses.values()) == 1


def test_a_new_version_of_the_same_code_keeps_the_cache(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Every commit gets its own version; one that leaves the code as it was, such as
    # a change to the documentation, needs no new compilation.
    (package / "_version.py").write_text('version = "0.1.dev38+g3496bb80b"\n')
    _call(_load(package / "routines.py", "routines_before"))
    key_before = _jit.cache_key()

    (package / "_version.py").write_text('version = "0.1.dev39+g0123456"\n')
    monkeypatch.setattr(cytepool, "__version__", "0.1.dev39+g0123456")
    monkeypatch.setattr(_jit, "__version__", "0.1.dev39+g0123456", raising=False)
    _clear_memoized()
    assert _jit.cache_key() == key_before

    after = _load(package / "routines.py", "routines_after")
    _call(after)
    assert sum(after.weighted_total.stats.cache_hits.values()) == 1
    assert sum(after.weighted_total.stats.cache_misses.values()) == 0


def test_bounds_checking_has_its_own_cache(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(numba.config, "BOUNDSCHECK", False)
    without_checks = _jit.cache_key()
    _clear_memoized()
    monkeypatch.setattr(numba.config, "BOUNDSCHECK", True)
    assert _jit.cache_key() != without_checks


@pytest.mark.skipif(
    os.name != "posix" or os.geteuid() == 0, reason="needs file permissions that bind"
)
def test_without_a_writable_directory_routines_compile_uncached(
    package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    read_only = tmp_path / "read-only"
    read_only.mkdir()
    read_only.chmod(0o500)
    try:
        monkeypatch.setenv("NUMBA_CACHE_DIR", str(read_only))
        monkeypatch.setenv("XDG_CACHE_HOME", str(read_only))
        monkeypatch.setattr(tempfile, "tempdir", str(read_only))
        _clear_memoized()
        assert _jit.cache_directory() is None

        module = _load(package / "routines.py", "routines_uncached")
        _call(module)
        assert isinstance(module.weighted_total._cache, NullCache)
        assert list(read_only.iterdir()) == []
    finally:
        read_only.chmod(0o700)


def test_cache_option_is_rejected() -> None:
    with pytest.raises(TypeError, match="cache"):
        _jit.jit(cache=True)


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to tell caps apart"
)
def test_thread_limit_caps_threads_and_restores_them() -> None:
    launched = numba.config.NUMBA_NUM_THREADS
    before = numba.get_num_threads()
    with _jit.thread_limit(1) as used:
        assert used == 1
        assert numba.get_num_threads() == 1
        with _jit.thread_limit(launched) as inner:
            assert inner == numba.get_num_threads() == launched
        assert numba.get_num_threads() == 1
    assert numba.get_num_threads() == before
    # numba cannot use more threads than it launched, so a larger cap uses them all.
    with _jit.thread_limit(launched + 5) as used:
        assert used == numba.get_num_threads() == launched
    with _jit.thread_limit(None) as used:
        assert used == numba.get_num_threads() == before
    assert numba.get_num_threads() == before


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to tell caps apart"
)
def test_thread_limit_applies_to_the_calling_thread_only() -> None:
    before = numba.get_num_threads()
    entered, release = threading.Event(), threading.Event()
    seen: dict[str, int] = {}

    def capped() -> None:
        with _jit.thread_limit(1):
            seen["inside"] = numba.get_num_threads()
            entered.set()
            release.wait(timeout=10)

    worker = threading.Thread(target=capped)
    worker.start()
    try:
        assert entered.wait(timeout=10)
        assert seen["inside"] == 1
        assert numba.get_num_threads() == before
    finally:
        release.set()
        worker.join(timeout=10)


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to tell caps apart"
)
def test_thread_limit_restores_threads_after_an_exception() -> None:
    before = numba.get_num_threads()
    with pytest.raises(RuntimeError, match="inside the block"):
        with _jit.thread_limit(1):
            assert numba.get_num_threads() == 1
            raise RuntimeError("inside the block")
    assert numba.get_num_threads() == before


@pytest.mark.parametrize(
    ("n_threads", "error", "message"),
    [
        (0, ValueError, "n_threads must be at least 1, got 0"),
        (-2, ValueError, "at least 1, got -2"),
        (1.5, TypeError, "an integer or None, got float"),
        ("2", TypeError, "got str"),
        (True, TypeError, "got bool"),
    ],
)
def test_thread_limit_rejects_invalid_counts(
    n_threads: object, error: type, message: str
) -> None:
    before = numba.get_num_threads()
    with pytest.raises(error, match=message):
        with _jit.thread_limit(n_threads):  # type: ignore[arg-type]
            pass
    assert numba.get_num_threads() == before


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to use a pool"
)
def test_by_runs_covers_every_item_once_and_starts_on_the_calling_thread() -> None:
    calls: list[tuple[int, int, int]] = []
    lock = threading.Lock()

    def run(start: int, stop: int) -> None:
        with lock:
            calls.append((start, stop, threading.get_ident()))

    with _jit.thread_limit(2):
        _jit.by_runs(run, 23)
    items = sorted(i for start, stop, _ in calls for i in range(start, stop))
    assert items == list(range(23))
    # Two threads give eight runs, the first of them on this thread, where the
    # routine compiles before any worker uses it.
    assert len(calls) == 8
    assert calls[0][:2] == (0, 2) and calls[0][2] == threading.get_ident()


def test_by_runs_on_one_thread_runs_everything_here() -> None:
    calls: list[tuple[int, int, int]] = []
    with _jit.thread_limit(1):
        _jit.by_runs(
            lambda start, stop: calls.append((start, stop, threading.get_ident())), 5
        )
    assert calls == [(0, 5, threading.get_ident())]


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to use a pool"
)
def test_by_runs_finalizes_the_garbage_of_its_workers_on_the_calling_thread() -> None:
    """Objects that only the collector frees are finalized where a collection runs.
    With the collector running at almost every allocation, the workers' garbage
    would be finalized on the workers; by_runs pauses it, so they are finalized
    here, where objects such as Tk's must be."""
    finalized_on: list[int] = []

    class Cycle:
        def __init__(self) -> None:
            self.itself = self

        def __del__(self) -> None:
            finalized_on.append(threading.get_ident())

    def run(start: int, stop: int) -> None:
        for _ in range(200):
            Cycle()

    threshold = gc.get_threshold()
    gc.collect()
    gc.set_threshold(1)
    try:
        with _jit.thread_limit(2):
            _jit.by_runs(run, 16)
    finally:
        gc.set_threshold(*threshold)
    gc.collect()
    assert len(finalized_on) == 8 * 200
    assert set(finalized_on) == {threading.get_ident()}
    assert gc.isenabled()
