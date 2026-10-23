import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from types import ModuleType

import numba
import numpy as np
import pytest
from numba.core.caching import NullCache

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
