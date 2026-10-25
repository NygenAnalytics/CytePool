import contextlib
import gzip
import importlib
import json
import os
import pickle
import shutil
import sys
import tracemalloc
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.cube import Cube
from cytepool.io import cube_files, read_cube, write_cube
from cytepool.io import zarr as zarr_module
from cytepool.io.zarr import open_array, write_array

FIXTURES = Path(__file__).parent / "data" / "cube_files"
try:
    importlib.import_module("compression.zstd")
    COMPRESSIONS: list[Any] = ["gzip", None, "zstd"]
except ImportError:  # Python 3.13
    COMPRESSIONS = ["gzip", None]
LAYER_AXES = ["groups", "samples", "features"]
SAMPLES = ["S1", "S2", "S3", "S4"]
GROUPS = ["B", "T", "NK"]
# Cells per sample (rows) and group (columns); no sample has NK cells.
CELLS = np.array([[5, 3, 0], [2, 0, 0], [7, 1, 0], [4, 6, 0]])


def simulate_cube(n_features: int = 6, seed: int = 3) -> Cube:
    """Sums over simulated cells, with table columns of every type cube files store."""
    rng = np.random.default_rng(seed)
    shape = (len(GROUPS), len(SAMPLES), n_features)
    counts = np.zeros(shape, dtype=np.uint16)
    detected = np.zeros(shape, dtype=np.int8)
    log_sum, log_sum_sq, linear_sum = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    for s in range(len(SAMPLES)):
        for g in range(len(GROUPS)):
            cells = rng.poisson(2.0, size=(CELLS[s, g], n_features))
            linear = cells * 1e4 / np.maximum(cells.sum(axis=1, keepdims=True), 1)
            counts[g, s] = cells.sum(axis=0)
            detected[g, s] = (cells > 0).sum(axis=0)
            log_sum[g, s] = np.log1p(linear).sum(axis=0)
            log_sum_sq[g, s] = (np.log1p(linear) ** 2).sum(axis=0)
            linear_sum[g, s] = linear.sum(axis=0)
    visits = np.array(["2024-03-01T10:00", "NaT", "1999-12-31", "2024-02-29"], "M8[ms]")
    samples = {
        "sample_id": SAMPLES,
        "donor": ["D1", None, "D1", "D2"],
        "note": ["", "ünïcode", None, "a/b"],
        "age": [41.5, np.nan, -0.0, 7.0],
        "batch": np.array([1, -2, 3, 127], dtype=np.int8),
        "visit": visits,
        "Treated?": [True, False, True, True],
        "CD4/CD8": np.array([1.5, 0.25, 2.0, np.inf], dtype=np.float32),
    }
    groups = {
        "group_id": GROUPS,
        "cell_ontology_id": ["CL:0000236", None, "CL:0000623"],
        "parent": [None] * 3,  # a text column with no values
        "reads": np.array([2**64 - 1, 0, 12], dtype=np.uint64),
        "phase": np.array([1 + 2j, np.nan, -1j], dtype=np.complex64),
        "lifetime": np.array([3600, -1, 0], dtype="m8[s]"),
    }
    features = {
        "feature_id": [f"G{j}" for j in range(n_features)],
        "symbol": [None if j == 1 else f"GENE{j}" for j in range(n_features)],
        "measured": np.arange(n_features) != 2,
        "gc": np.linspace(0.3, 0.6, n_features).astype(np.float16),
        "length": (np.arange(n_features) * 100 - 50).astype(np.int16),
    }
    layers = {
        "counts": counts,
        "detected": detected,
        "log_sum": log_sum,
        "log_sum_sq": log_sum_sq,
        "linear_sum": linear_sum,
    }
    return Cube(
        layers,
        samples=samples,
        groups=groups,
        features=features,
        n_cells=CELLS,
        normalization={"method": "library_size", "target_sum": 1e4, "log": "natural"},
        provenance={"source": "simulated cells", "operations": [{"operation": "x"}]},
    )


@pytest.fixture(scope="module")
def cube() -> Cube:
    return simulate_cube()


@pytest.fixture(scope="module")
def store(cube: Cube, tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("store") / "cells.cube.zarr"
    write_cube(cube, path)
    return path


def assert_same_bits(actual: Any, expected: Any) -> None:
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert actual.tobytes() == expected.tobytes()


def assert_same_cube(actual: Cube, expected: Cube) -> None:
    """Assert equal tables (floats bit for bit), cells, layers, and JSON metadata."""
    for name in ("samples", "groups", "features"):
        mine, theirs = getattr(actual, name), getattr(expected, name)
        assert mine.equals(theirs), name
        for column in theirs.columns:
            if theirs[column].dtype.kind != "O":
                assert_same_bits(mine[column], theirs[column])
    if expected.n_cells is None:
        assert actual.n_cells is None
    else:
        assert_same_bits(actual.n_cells, expected.n_cells)
    assert actual.layer_names == expected.layer_names
    for name in expected.layer_names:
        assert_same_bits(actual.layer(name), expected.layer(name))
    assert actual.normalization == expected.normalization
    assert actual.provenance == expected.provenance


def stored_files(path: Path) -> set[str]:
    return {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}


_opened: list[str] | None = None


def _record_open(event: str, args: tuple[Any, ...]) -> None:
    if event == "open" and _opened is not None:
        _opened.append(str(args[0]))


sys.addaudithook(_record_open)


@contextlib.contextmanager
def opened_files() -> Iterator[list[str]]:
    """Record the paths of the files opened inside the block (PEP 578 audit events)."""
    global _opened
    _opened = []
    try:
        yield _opened
    finally:
        _opened = None


# Round trips.


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_round_trip_keeps_every_layer_column_and_attribute(
    cube: Cube, tmp_path: Path, compression: str | None
) -> None:
    write_cube(cube, tmp_path / "c.cube.zarr", compression=compression)
    back = read_cube(tmp_path / "c.cube.zarr")
    assert "counts (uint16, lazy)" in repr(back)
    assert_same_cube(back, cube)
    assert_same_cube(back.to_memory(), cube)


def test_bulk_cubes_and_cubes_with_empty_axes_round_trip(tmp_path: Path) -> None:
    rng = np.random.default_rng(4)
    bulk = Cube(
        {"counts": rng.gamma(2.0, 3.0, size=(1, 5, 4))},
        samples={"sample_id": [f"S{i}" for i in range(5)]},
        groups={"group_id": ["bulk"]},
        features={"feature_id": list("abcd")},
    )
    empty = {"counts": np.zeros((2, 0, 3), dtype=np.int64)}
    no_samples = Cube(
        empty,
        samples={"sample_id": [], "site": []},
        groups={"group_id": ["a", "b"]},
        features={"feature_id": list("xyz"), "measured": [True, False, True]},
        n_cells=np.zeros((0, 2), dtype=np.int64),
    )
    no_groups = Cube(
        {"counts": np.zeros((0, 2, 3), dtype=np.int32)},
        samples={"sample_id": ["s", "t"]},
        groups={"group_id": [], "parent": []},
        features={"feature_id": list("xyz")},
        n_cells=np.zeros((2, 0), dtype=np.int64),
    )
    no_features = Cube(
        {"counts": np.zeros((1, 2, 0))},
        samples={"sample_id": ["s", "t"]},
        groups={"group_id": ["g"]},
        features={"feature_id": [], "symbol": []},
    )
    for name, original in [
        ("bulk", bulk),
        ("no_samples", no_samples),
        ("no_groups", no_groups),
        ("no_features", no_features),
    ]:
        write_cube(original, tmp_path / name)
        assert_same_cube(read_cube(tmp_path / name), original)
    assert not (tmp_path / "bulk" / "n_cells").exists()
    attributes = json.loads((tmp_path / "bulk" / "zarr.json").read_text())
    assert attributes["attributes"]["has_n_cells"] is False


def test_n_cells_beyond_int32_are_stored_as_int64(tmp_path: Path) -> None:
    original = Cube(
        {"counts": np.array([[[5]]])},
        samples={"sample_id": ["s"]},
        groups={"group_id": ["g"]},
        features={"feature_id": ["f"]},
        n_cells=np.array([[2**31]]),
    )
    write_cube(original, tmp_path / "c")
    document = json.loads((tmp_path / "c" / "n_cells" / "zarr.json").read_text())
    assert document["data_type"] == "int64"
    assert_same_cube(read_cube(tmp_path / "c"), original)


def test_the_largest_int64_round_trips_and_sums_beyond_it_raise(tmp_path: Path) -> None:
    largest = 2**63 - 1
    original = Cube(
        {"counts": np.array([[[largest, 0]], [[2**62, 2]], [[2**62, 0]]], np.uint64)},
        samples={"sample_id": ["s"]},
        groups={"group_id": ["a", "b", "c"]},
        features={"feature_id": ["x", "y"]},
    )
    write_cube(original, tmp_path / "c.cube.zarr")
    back = read_cube(tmp_path / "c.cube.zarr")
    assert_same_cube(back, original)
    assert back.library_sizes()[0].tolist() == [largest, 2**62 + 2, 2**62]
    merged = back.regroup({"bc": ["b", "c"]})  # 2**62 + 2**62 is 2**63, beyond int64
    assert merged.layer("counts", "a").tolist() == [[largest, 0]]
    with pytest.raises(OverflowError, match="merged group 'bc' at sample 's', feature"):
        merged.layer("counts", "bc")
    with pytest.raises(OverflowError, match="sums to 9223372036854775808, beyond 2"):
        write_cube(merged, tmp_path / "merged.cube.zarr")
    assert [p.name for p in tmp_path.iterdir()] == ["c.cube.zarr"]


def test_values_beyond_int64_in_a_file_written_elsewhere_raise(
    tmp_path: Path,
) -> None:
    original = Cube(
        {"counts": np.array([[[3, 2**63 - 1]]], dtype=np.uint64)},
        samples={"sample_id": ["s"]},
        groups={"group_id": ["g"]},
        features={"feature_id": ["x", "y"]},
        n_cells=np.array([[5]]),
    )
    path = tmp_path / "c.cube.zarr"
    write_cube(original, path, compression=None)
    chunk = path / "layers" / "counts" / "c" / "0" / "0" / "0"
    chunk.write_bytes(np.array([3, 2**63], dtype="<u8").tobytes())
    back = read_cube(path)  # layers are read when they are used
    with pytest.raises(
        ValueError,
        match=r"'counts' has a value above 2\*\*63 - 1 \(9223372036854775808\) at "
        r"group 'g', sample 's', feature 'y'",
    ):
        back.layer("counts", "g")
    shutil.rmtree(path / "n_cells")
    write_array(
        path / "n_cells",
        np.array([[2**63]], dtype=np.uint64),
        chunks=(1, 1),
        dimension_names=["samples", "groups"],
    )
    with pytest.raises(ValueError, match=r"n_cells is above 2\*\*63 - 1") as raised:
        read_cube(path)
    assert raised.value.__notes__ == [f"(while reading the cube file {path})"]


def test_a_lazy_cube_can_be_written_back_over_its_own_file(
    cube: Cube, tmp_path: Path
) -> None:
    def derive(source: Cube) -> Cube:
        chosen = source.select(samples=["S4", "S2", "S1"], features=[5, 0, 3])
        return chosen.regroup({"lymphoid": ["B", "T", "NK"]})

    path = tmp_path / "c.cube.zarr"
    write_cube(cube, path)
    derived = derive(read_cube(path))
    write_cube(derived, path, overwrite=True)
    assert_same_cube(read_cube(path), derive(cube))


def test_a_pickled_cube_reads_from_its_file(store: Path, cube: Cube) -> None:
    copy = pickle.loads(pickle.dumps(read_cube(store)))
    assert_same_bits(copy.layer("log_sum", "T"), cube.layer("log_sum", "T"))


# The layout on disk, against the documented format.


def test_the_layout_on_disk_follows_the_documented_format(
    store: Path, cube: Cube
) -> None:
    root = json.loads((store / "zarr.json").read_text("utf-8"))
    assert root == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "format": "cytepool.cube",
            "format_version": 1,
            "axes": ["groups", "samples", "features"],
            "layers": ["counts", "detected", "log_sum", "log_sum_sq", "linear_sum"],
            "has_n_cells": True,
            "normalization": cube.normalization,
            "provenance": cube.provenance,
        },
    }
    samples = json.loads((store / "samples" / "zarr.json").read_text("utf-8"))
    assert samples["attributes"] == {
        "columns": list(cube.samples.columns),
        "arrays": ["sample_id", "donor", "note", "age", "batch", "visit"]
        + ["column_6", "column_7"],
    }
    donor = open_array(store / "samples" / "donor")
    assert donor.dtype == np.int32 and donor.dimension_names == ("samples",)
    assert np.asarray(donor).tolist() == [0, -1, 0, 1]
    assert donor.attributes == {"levels": ["D1", "D2"]}
    note = open_array(store / "samples" / "note")
    assert np.asarray(note).tolist() == [0, 1, -1, 2]
    assert note.attributes == {"levels": ["", "ünïcode", "a/b"]}
    visit = open_array(store / "samples" / "visit")
    assert visit.attributes == {"numpy_dtype": "datetime64[ms]"}
    expected_ms = [1709287200000, -(2**63), 946598400000, 1709164800000]
    assert np.asarray(visit).tolist() == expected_ms
    lifetime = open_array(store / "groups" / "lifetime")
    assert lifetime.attributes == {"numpy_dtype": "timedelta64[s]"}
    cells = open_array(store / "n_cells")
    assert cells.dtype == np.int32 and cells.dimension_names == ("samples", "groups")
    assert np.asarray(cells).tolist() == CELLS.tolist()
    counts = json.loads((store / "layers" / "counts" / "zarr.json").read_text())
    assert counts["data_type"] == "uint16" and counts["shape"] == [3, 4, 6]
    assert counts["chunk_grid"]["configuration"]["chunk_shape"] == [1, 4, 6]
    assert counts["dimension_names"] == LAYER_AXES
    assert counts["codecs"] == [
        {"name": "bytes", "configuration": {"endian": "little"}},
        {"name": "gzip", "configuration": {"level": 4}},
    ]
    # No sample has NK cells, so the NK group of every layer is zero and not stored.
    layer_files = {name for name in stored_files(store) if name.startswith("layers/")}
    assert layer_files == {
        f"layers/{layer}/{part}"
        for layer in cube.layer_names
        for part in ("zarr.json", "c/0/0/0", "c/1/0/0")
    } | {"layers/zarr.json"}
    assert_same_bits(read_cube(store).layer("counts", "NK"), np.zeros((4, 6), "u2"))


def test_array_names_are_portable_and_unique(tmp_path: Path) -> None:
    names = [
        "sample_id",
        "Age",
        "age",
        "CD4/CD8",
        "Zarr.json",  # collides with zarr.json where case is ignored
        "__x",
        "con",
        "nul.txt",
        "column_3",
        ".hidden",
        "x.",
        "é",
        "a" * 65,
        "a" * 64,
        "ok-name.v2",
        "-dash",
    ]
    samples = {name: [f"S{i}" for i in range(2)] for name in names}
    original = Cube(
        {"counts": np.zeros((1, 2, 1), dtype=np.int64)},
        samples=samples,
        groups={"group_id": ["g"]},
        features={"feature_id": ["f"]},
    )
    write_cube(original, tmp_path / "c")
    group = json.loads((tmp_path / "c" / "samples" / "zarr.json").read_text("utf-8"))
    assert group["attributes"]["arrays"] == [
        "sample_id",
        *(f"column_{i}" for i in range(1, 13)),
        "a" * 64,
        "ok-name.v2",
        "column_15",
    ]
    assert read_cube(tmp_path / "c").samples.equals(original.samples)


@pytest.mark.parametrize(
    ("compression", "level", "codec"),
    [
        (None, None, None),
        ("gzip", 1, {"name": "gzip", "configuration": {"level": 1}}),
    ],
)
def test_compression_choices_reach_every_array(
    cube: Cube, tmp_path: Path, compression: str | None, level: Any, codec: Any
) -> None:
    write_cube(cube, tmp_path / "c", compression=compression, level=level)
    arrays = 0
    for metadata in (tmp_path / "c").rglob("zarr.json"):
        document = json.loads(metadata.read_text())
        if document["node_type"] == "array":
            arrays += 1
            one_byte = np.dtype(document["data_type"]).itemsize == 1
            serializer = {"name": "bytes", "configuration": {"endian": "little"}}
            expected = [{"name": "bytes"} if one_byte else serializer]
            assert document["codecs"] == expected + ([codec] if codec else []), metadata
    assert arrays == 25  # 8 + 6 + 5 columns, n_cells and 5 layers


def test_features_beyond_one_chunk(tmp_path: Path) -> None:
    original = simulate_cube(n_features=4100)
    write_cube(original, tmp_path / "c")
    counts = open_array(tmp_path / "c" / "layers" / "counts")
    assert counts.chunks == (1, 4, 4096)
    assert {"c/0/0/0", "c/0/0/1"} <= stored_files(tmp_path / "c" / "layers" / "counts")
    back = read_cube(tmp_path / "c")
    assert_same_bits(back.layer("detected", "T"), original.layer("detected", "T"))
    assert_same_bits(back.layer("linear_sum"), original.layer("linear_sum"))


# Reading what zarr-python wrote.


def test_reads_the_cube_file_zarr_python_wrote() -> None:
    # validation/cube_files.py had zarr-python copy a cube file node by node, with
    # chunks halving every axis, big-endian bytes and gzip level 1, and saved the
    # contents of the original cube next to it.
    back = read_cube(FIXTURES / "tiny.cube.zarr")
    text = json.loads((FIXTURES / "expected.json").read_text("utf-8"))
    with np.load(FIXTURES / "expected.npz") as data:
        arrays = dict(data)
    layers = open_array(FIXTURES / "tiny.cube.zarr" / "layers" / "counts")
    assert layers.chunks == (2, 2, 4)
    assert text["attributes"]["layers"] == list(back.layer_names)
    assert back.normalization == text["attributes"]["normalization"]
    assert back.provenance == text["attributes"]["provenance"]
    for axis in ("samples", "groups", "features"):
        table = getattr(back, axis)
        assert list(table.columns) == text[axis]
        for position, name in enumerate(table.columns):
            key = f"{axis}:{position}"
            if key in text:
                assert table[name].tolist() == text[key]
            else:
                assert_same_bits(table[name], arrays[key])
    assert_same_bits(back.n_cells, arrays["n_cells"].astype(np.int64))
    for name in back.layer_names:
        assert_same_bits(back.layer(name), arrays[f"layer:{name}"])


# Laziness.


def test_opening_reads_no_layer_chunks_and_a_group_reads_only_its_own(
    store: Path, cube: Cube
) -> None:
    with opened_files() as opened:
        back = read_cube(store)
    assert str(store / "zarr.json") in opened
    assert not [path for path in opened if {"layers", "c"} <= set(Path(path).parts)]
    with opened_files() as opened:
        block = back.layer("counts", "T")
    assert opened == [str(store / "layers" / "counts" / "c" / "1" / "0" / "0")]
    assert_same_bits(block, cube.layer("counts", "T"))
    with opened_files() as opened:
        back.variance("B")
    assert sorted(opened) == [
        str(store / "layers" / name / "c" / "0" / "0" / "0")
        for name in ("log_sum", "log_sum_sq")
    ]


@pytest.fixture(scope="module")
def narrow(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Cube]:
    """A cube of 10 features saved in layer chunks of 3: features 0-2, 3-5, 6-8, 9."""
    simulated = simulate_cube(n_features=10)
    cube = Cube(
        {name: simulated.layer(name) for name in simulated.layer_names},
        samples=simulated.samples.with_columns(pool=["P1", "P2", "P1", "P1"]),
        groups=simulated.groups,
        features=simulated.features,
        n_cells=simulated.n_cells,
        normalization=simulated.normalization,
        provenance=simulated.provenance,
    )
    path = tmp_path_factory.mktemp("narrow") / "c.cube.zarr"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cube_files, "FEATURES_PER_CHUNK", 3)
        write_cube(cube, path)
    return path, cube


def chunk_files(
    path: Path, layer: str, groups: list[int], chunks: list[int]
) -> list[str]:
    """The paths of the chunks of the given groups and feature chunks, sorted."""
    root = path / "layers" / layer / "c"
    return sorted(str(root / str(g) / "0" / str(j)) for g in groups for j in chunks)


@pytest.mark.parametrize(
    ("selection", "groups", "chunks"),
    [
        pytest.param({"features": [4]}, [0, 1, 2], [1], id="one feature"),
        pytest.param({"features": [9, 0, 4]}, [0, 1, 2], [0, 1, 3], id="unsorted"),
        pytest.param({"features": [2, 3]}, [0, 1, 2], [0, 1], id="across chunks"),
        pytest.param({"features": np.arange(10) >= 6}, [0, 1, 2], [2, 3], id="mask"),
        pytest.param({"features": ["G8", "G7"]}, [0, 1, 2], [2], id="ids"),
        pytest.param({"features": 5, "groups": ["T"]}, [1], [1], id="one group"),
        pytest.param(
            {"samples": ["S3", "S1"], "features": [6, 1]},
            [0, 1, 2],
            [0, 2],
            id="samples",
        ),
        pytest.param(
            {"samples": ["S2"], "groups": [2, 0]},
            [0, 2],
            [0, 1, 2, 3],
            id="all features",
        ),
        pytest.param({"features": []}, [], [], id="no features"),
    ],
)
def test_selections_of_a_saved_cube_read_only_the_chunks_that_hold_them(
    narrow: tuple[Path, Cube], selection: dict, groups: list[int], chunks: list[int]
) -> None:
    path, cube = narrow
    chosen, expected = read_cube(path).select(**selection), cube.select(**selection)
    for name in cube.layer_names:
        with opened_files() as opened:
            values = chosen.layer(name)
        # Each chunk once, opened even where it is not stored (the NK group).
        assert sorted(opened) == chunk_files(path, name, groups, chunks)
        assert_same_bits(values, expected.layer(name))


def test_derived_cubes_of_a_saved_cube_read_only_the_chunks_they_need(
    narrow: tuple[Path, Cube],
) -> None:
    path, cube = narrow
    derivations: list[tuple[Callable[[Cube], Cube], list[int], list[int]]] = [
        (
            lambda c: c.regroup({"lymphoid": ["B", "T"]}).select(features=[7, 2]),
            [0, 1, 2],
            [0, 2],
        ),
        (
            lambda c: c.select(features=[9, 3]).collapse_samples("pool"),
            [0, 1, 2],
            [1, 3],
        ),
        (
            lambda c: c.collapse_samples("pool").select(
                samples=["P2"], features=[0], groups=["NK", "B"]
            ),
            [0, 2],
            [0],
        ),
    ]
    for derive, groups, chunks in derivations:
        derived, expected = derive(read_cube(path)), derive(cube)
        with opened_files() as opened:
            values = derived.layer("counts")
        assert sorted(opened) == chunk_files(path, "counts", groups, chunks)
        assert_same_bits(values, expected.layer("counts"))
        assert_same_cube(derived, expected)
        assert_same_cube(pickle.loads(pickle.dumps(derived)), expected)


def test_reads_of_a_saved_cube_hold_what_they_return_and_about_one_chunk(
    tmp_path: Path,
) -> None:
    # One group of 256 samples x 4,096 features of int32 takes 4 MiB, saved in 16
    # chunks of 256 features (256 KiB). A read holds what it returns, the chunk it
    # copies from and the part of it a selection takes. Reading whole groups for a
    # few features would take 4 MiB; copying what the reader returns, or reordering
    # features after the read, would hold a whole group twice.
    counts = np.random.default_rng(5).integers(0, 50, (1, 256, 4096), dtype=np.int32)
    original = Cube(
        {"counts": counts},
        samples={"sample_id": [f"s{i}" for i in range(256)]},
        groups={"group_id": ["g"]},
        features={"feature_id": [f"f{j}" for j in range(4096)]},
    )
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(cube_files, "FEATURES_PER_CHUNK", 256)
        write_cube(original, tmp_path / "c.cube.zarr", compression=None)
    back = read_cube(tmp_path / "c.cube.zarr")
    chunk, group = 2**18, 2**22
    shuffled = np.random.default_rng(6).permutation(4096)
    reversed_samples = np.arange(256)[::-1]
    reads: list[tuple[Cube, Callable[[Cube], np.ndarray], np.ndarray, float]] = [
        # Features in two chunks, then in four chunks and out of order.
        (
            back.select(features=[1000, 1500]),
            lambda c: c.layer("counts"),
            counts[:, :, [1000, 1500]],
            2 * chunk,
        ),
        (
            back.select(features=[4000, 10, 1000, 2000]),
            lambda c: c.layer("counts"),
            counts[:, :, [4000, 10, 1000, 2000]],
            2 * chunk,
        ),
        # The whole group, the whole layer, every feature shuffled, and every
        # sample in reverse order (read as the selection's one group).
        (back, lambda c: c.layer("counts", 0), counts[0], 1.25 * group),
        (back, lambda c: c.layer("counts"), counts, 1.25 * group),
        (
            back.select(features=shuffled),
            lambda c: c.layer("counts", 0),
            counts[0][:, shuffled],
            1.25 * group,
        ),
        (
            back.select(samples=reversed_samples),
            lambda c: c.layer("counts"),
            counts[:, reversed_samples],
            1.25 * group,
        ),
    ]
    for chosen, read, expected, bound in reads:
        tracemalloc.start()
        try:
            values = read(chosen)
            peak = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        assert_same_bits(values, expected)
        assert values.flags.c_contiguous and not values.flags.writeable
        assert peak < bound


def test_merging_groups_of_a_saved_cube_holds_the_sum_and_one_group(
    tmp_path: Path,
) -> None:
    # Four groups of 64 samples x 4,096 uint32 counts, 1 MiB each and one chunk
    # each. Merging them holds the int64 sum (2 MiB), the group being read and its
    # chunk; keeping the first group, or an int64 copy of each, would take more.
    counts = np.random.default_rng(7).integers(0, 50, (4, 64, 4096), dtype=np.uint32)
    groups = ["a", "b", "c", "d"]
    original = Cube(
        {"counts": counts},
        samples={"sample_id": [f"s{i}" for i in range(64)]},
        groups={"group_id": groups},
        features={"feature_id": [f"f{j}" for j in range(4096)]},
    )
    write_cube(original, tmp_path / "c.cube.zarr", compression=None)
    merged = read_cube(tmp_path / "c.cube.zarr").regroup({"all": groups})
    tracemalloc.start()
    try:
        values = merged.layer("counts", "all")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert_same_bits(values, counts.sum(axis=0, dtype=np.int64))
    assert peak < 4.5 * 2**20


def test_layers_kept_from_a_saved_cube_cannot_be_made_writable(store: Path) -> None:
    # The cube keeps the arrays a cube file's reader returns without a copy. The
    # read-only flag it sets on them holds for the views it gives out only because
    # those arrays own their memory: a read-only view of a writable array can be
    # made writable again.
    back = read_cube(store)
    kept = [
        back.to_memory(),
        back.select(groups=["T"]).to_memory(),  # keeps the one group as it was read
        back.select(groups=["T"], features=[3, 1]).to_memory(),
        back.select(samples=["S2", "S1"]).to_memory(),
    ]
    for cube in kept:
        for name in cube.layer_names:
            for values in (cube.layer(name), cube.layer(name, 0)):
                with pytest.raises(ValueError, match="WRITEABLE"):
                    values.flags.writeable = True


def test_a_cube_whose_file_was_replaced_stops_reading(
    cube: Cube, tmp_path: Path
) -> None:
    path = tmp_path / "c.cube.zarr"
    write_cube(cube, path)
    for metadata in path.rglob("zarr.json"):  # as if written a while ago
        os.utime(metadata, ns=(0, metadata.stat().st_mtime_ns - 10**10))
    old = read_cube(path)
    write_cube(cube.select(samples=["S2", "S1", "S4", "S3"]), path, overwrite=True)
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        old.layer("counts", "B")
    reopened = read_cube(path)
    shutil.rmtree(path)
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        reopened.layer("counts", "B")


# Overwriting and failed writes.


def test_refuses_to_overwrite_unless_asked(cube: Cube, tmp_path: Path) -> None:
    path = tmp_path / "c.cube.zarr"
    write_cube(cube, path)
    smaller = cube.select(groups=["T"])
    with pytest.raises(FileExistsError, match="already exists; pass overwrite=True"):
        write_cube(smaller, path)
    assert read_cube(path).n_groups == 3
    write_cube(smaller, path, overwrite=True)
    assert read_cube(path).groups["group_id"].tolist() == ["T"]
    (tmp_path / "empty").mkdir()
    write_cube(smaller, tmp_path / "empty", overwrite=True)
    assert read_cube(tmp_path / "empty").n_groups == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["c.cube.zarr", "empty"]


def test_overwrite_replaces_only_cube_files_and_empty_directories(
    cube: Cube, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def directory(name: str, files: dict[str, str]) -> Path:
        path = tmp_path / name
        for relative, text in files.items():
            (path / relative).parent.mkdir(parents=True, exist_ok=True)
            (path / relative).write_text(text)
        return path

    project = directory("project", {"zarr.json": "not json", "thesis.tex": "x"})
    other = directory(
        "other.zarr", {"zarr.json": '{"zarr_format": 3, "node_type": "group"}'}
    )
    (other / "X").mkdir()
    write_array(tmp_path / "array", np.ones(3), chunks=(3,))
    shutil.copytree(store, tmp_path / "copy.cube.zarr")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "link").symlink_to(project)
    cases = [
        ("project", "is not a cube file"),  # a stray zarr.json is not a store
        ("other.zarr", "is not a cube file"),  # another Zarr group, such as AnnData
        ("copy.cube.zarr/layers", "is not a cube file"),  # a node inside a cube file
        ("array", "is not a cube file"),
        ("notes.txt", "is a file"),
        ("link", "is a symbolic link"),
        (".", "is the working directory or holds it"),
    ]
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    monkeypatch.chdir(tmp_path / "copy.cube.zarr")
    for name, message in cases:
        target = Path(name) if name == "." else tmp_path / name
        with pytest.raises(FileExistsError, match=message):
            write_cube(cube, target, overwrite=True)
    with pytest.raises(FileExistsError, match="is the working directory or holds it"):
        write_cube(cube, tmp_path, overwrite=True)
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before  # nothing changed
    # Without a working directory (it was deleted), the guard has nothing to protect.
    gone = tmp_path / "gone"
    gone.mkdir()
    monkeypatch.chdir(gone)
    gone.rmdir()
    write_cube(cube, tmp_path / "copy.cube.zarr", overwrite=True)
    # A cube file of any version is replaced.
    metadata = tmp_path / "copy.cube.zarr" / "zarr.json"
    document = json.loads(metadata.read_text())
    document["attributes"]["format_version"] = 2
    metadata.write_text(json.dumps(document))
    monkeypatch.chdir(tmp_path)
    write_cube(cube.select(groups=["B"]), tmp_path / "copy.cube.zarr", overwrite=True)
    assert read_cube(tmp_path / "copy.cube.zarr").n_groups == 1


class FailingLayer:
    """A lazy layer whose third group breaks a constraint (a negative count)."""

    def __init__(self, values: np.ndarray) -> None:
        self.values = values.astype(np.int64)
        self.values[2, 0, 0] = -1
        self.shape, self.dtype = self.values.shape, self.values.dtype

    def __getitem__(self, group: int) -> np.ndarray:
        return self.values[group]


def test_a_failed_write_leaves_no_store_and_keeps_the_old_one(
    cube: Cube, tmp_path: Path
) -> None:
    failing = Cube(
        {"counts": FailingLayer(cube.layer("counts"))},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    with pytest.raises(ValueError, match="'counts' has a negative value"):
        write_cube(failing, tmp_path / "new.cube.zarr")
    assert list(tmp_path.iterdir()) == []
    write_cube(cube, tmp_path / "old.cube.zarr")
    with pytest.raises(ValueError, match="'counts' has a negative value"):
        write_cube(failing, tmp_path / "old.cube.zarr", overwrite=True)
    assert [p.name for p in tmp_path.iterdir()] == ["old.cube.zarr"]
    assert_same_cube(read_cube(tmp_path / "old.cube.zarr"), cube)


def test_the_store_is_not_a_group_until_everything_else_is_written(
    cube: Cube, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen = []
    write = cube_files.write_array

    def watching_write(path: Path, *args: Any, **kwargs: Any) -> None:
        root = next(p for p in path.parents if ".partial-" in p.name)
        seen.append((root.parent == tmp_path, (root / "zarr.json").exists()))
        write(path, *args, **kwargs)

    monkeypatch.setattr(cube_files, "write_array", watching_write)
    write_cube(cube, tmp_path / "c.cube.zarr")
    # Every array went into the hidden partial directory next to the target, which
    # had no root metadata while they were written.
    assert seen == [(True, False)] * 25
    assert [p.name for p in tmp_path.iterdir()] == ["c.cube.zarr"]


def test_overwrite_puts_the_old_store_back_if_the_new_one_cannot_move_in(
    cube: Cube, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.cube.zarr"
    write_cube(cube, path)
    rename = os.rename

    def failing_rename(source: Any, target: Any) -> None:
        if ".partial-" in str(source):
            raise OSError("simulated failure")
        rename(source, target)

    monkeypatch.setattr(cube_files.os, "rename", failing_rename)
    with pytest.raises(OSError, match="simulated failure"):
        write_cube(cube.select(groups=["B"]), path, overwrite=True)
    monkeypatch.undo()
    assert [p.name for p in tmp_path.iterdir()] == ["c.cube.zarr"]
    assert_same_cube(read_cube(path), cube)

    # If the old store cannot be put back either, the error says where it is.
    def failing_renames(source: Any, target: Any) -> None:
        if ".partial-" in str(source) or ".replaced-" in str(source):
            raise OSError("simulated failure")
        rename(source, target)

    monkeypatch.setattr(cube_files.os, "rename", failing_renames)
    with pytest.raises(OSError, match="simulated failure") as raised:
        write_cube(cube.select(groups=["B"]), path, overwrite=True)
    monkeypatch.undo()
    (backup,) = tmp_path.iterdir()
    assert backup.name.startswith(".c.cube.zarr.replaced-")
    assert raised.value.__notes__ == [f"(the previous store was left at {backup})"]
    assert_same_cube(read_cube(backup), cube)


def test_a_store_that_appears_during_the_write_is_kept(
    cube: Cube, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_store = cube_files._write_store
    target = tmp_path / "c.cube.zarr"

    def write_and_race(*args: Any) -> None:
        write_store(*args)
        target.mkdir()
        (target / "theirs.txt").write_text("another writer")

    monkeypatch.setattr(cube_files, "_write_store", write_and_race)
    with pytest.raises(
        FileExistsError, match="appeared while the cube was being written"
    ):
        write_cube(cube, target)
    assert [p.name for p in tmp_path.iterdir()] == ["c.cube.zarr"]
    assert (target / "theirs.txt").read_text() == "another writer"


def test_a_store_replaced_while_it_is_read_raises(
    cube: Cube, store: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "c.cube.zarr"
    shutil.copytree(store, path)
    reordered = cube.select(samples=["S4", "S3", "S2", "S1"])
    open_layer = cube_files.open_array

    def open_after_a_replacement(array: Path) -> Any:
        if array.name == "counts":  # the tables are read; now another writer replaces
            monkeypatch.setattr(cube_files, "open_array", open_layer)
            write_cube(reordered, path, overwrite=True)
        return open_layer(array)

    monkeypatch.setattr(cube_files, "open_array", open_after_a_replacement)
    with pytest.raises(
        RuntimeError, match="was replaced or removed while it was being read"
    ):
        read_cube(path)
    monkeypatch.undo()
    assert_same_cube(read_cube(path), reordered)


def test_layer_chunks_hold_every_sample_up_to_256_mib() -> None:
    assert cube_files._layer_chunks(12, 5000, 8) == (1, 12, 4096)
    assert cube_files._layer_chunks(12, 7, 2) == (1, 12, 7)
    assert cube_files._layer_chunks(0, 0, 8) == (1, 1, 1)
    assert cube_files._layer_chunks(8192, 60_000, 8) == (1, 8192, 4096)  # 256 MiB
    # 200,000 samples x 4,096 float64 values would take 6.1 GiB in one chunk.
    assert cube_files._layer_chunks(200_000, 60_000, 8) == (1, 8192, 4096)
    assert cube_files._layer_chunks(200_000, 60_000, 1) == (1, 65_536, 4096)
    assert cube_files.LAYER_CHUNK_BYTES <= zarr_module.MAX_CHUNK_BYTES


# Clear errors.


def test_write_cube_rejects_invalid_arguments(
    cube: Cube, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for arguments, error, message in [
        (("not a cube", tmp_path / "a"), TypeError, "saves a Cube, got str"),
        ((cube, tmp_path / "missing" / "a"), FileNotFoundError, "does not exist"),
        ((cube, "s3://bucket/a.cube.zarr"), ValueError, "is a URL"),
    ]:
        with pytest.raises(error, match=message):
            write_cube(*arguments)
    for options, error, message in [
        ({"compression": "lz4"}, ValueError, "'gzip', 'zstd' or None, got 'lz4'"),
        ({"level": 10}, ValueError, "gzip level must be from 0 to 9"),
        ({"compression": None, "level": 1}, ValueError, "level applies only"),
        ({"overwrite": "yes"}, TypeError, "overwrite must be True or False"),
    ]:
        with pytest.raises(error, match=message):
            write_cube(cube, tmp_path / "a", **options)
    monkeypatch.setattr(zarr_module, "_zstd", lambda: None)
    with pytest.raises(ValueError, match="needs the compression.zstd module"):
        write_cube(cube, tmp_path / "a", compression="zstd")
    assert list(tmp_path.iterdir()) == []


@pytest.mark.skipif(
    np.dtype(np.longdouble).itemsize <= 8, reason="long double is float64 here"
)
def test_columns_without_a_zarr_data_type_are_named(cube: Cube, tmp_path: Path) -> None:
    samples = cube.samples.with_columns(weight=np.ones(4, dtype=np.longdouble))
    odd = Cube(
        {"counts": cube.layer("counts")},
        samples=samples,
        groups=cube.groups,
        features=cube.features,
    )
    with pytest.raises(TypeError, match="not a Zarr v3 core data type") as raised:
        write_cube(odd, tmp_path / "a")
    assert "(while writing the samples column 'weight')" in raised.value.__notes__
    assert list(tmp_path.iterdir()) == []


@pytest.fixture(scope="module")
def editable_store(store: Path, tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A copy of the store that a test may change, if it puts the original back."""
    path = tmp_path_factory.mktemp("editable") / "c.cube.zarr"
    shutil.copytree(store, path)
    return path


def edit(document: dict, path: str, value: Any) -> None:
    """Set (or, for a value of ..., delete) the entry at a dotted path."""
    *parents, last = path.split(".")
    for key in parents:
        document = document[key if not key.isdigit() else int(key)]
    if value is ...:
        del document[last]
    else:
        document[last if not last.isdigit() else int(last)] = value


@pytest.mark.parametrize(
    ("node", "entry", "value", "error", "message"),
    [
        ("", "attributes.format", ..., ValueError, "a Zarr group but not a cube file"),
        ("", "attributes.format", "anndata", ValueError, "holds the format 'anndata'"),
        (
            "",
            "attributes.format_version",
            2,
            ValueError,
            "version 2, but this version of CytePool reads version 1; update CytePool",
        ),
        ("", "attributes.format_version", 0, ValueError, "version 0, but this"),
        (
            "",
            "attributes.format_version",
            "1",
            ValueError,
            "must be an integer, got '1'",
        ),
        ("", "attributes.format_version", True, ValueError, "integer, got True"),
        ("", "attributes.layers", ..., ValueError, "lacks the cube attribute 'layers'"),
        ("", "attributes.axes", LAYER_AXES[::-1], ValueError, "axes must be"),
        ("", "attributes.layers", ["counts", "spliced"], ValueError, "distinct names"),
        ("", "attributes.layers", ["counts", "counts"], ValueError, "distinct names"),
        ("", "attributes.layers", ["../samples"], ValueError, "distinct names"),
        ("", "attributes.has_n_cells", False, ValueError, "has an n_cells array"),
        ("", "attributes.has_n_cells", "yes", ValueError, "true or false"),
        ("", "attributes.normalization", [], ValueError, "an object or null"),
        ("", "attributes.provenance", None, ValueError, "provenance must be an object"),
        ("samples", "attributes.arrays.1", "../groups", ValueError, "not a valid"),
        ("samples", "attributes.arrays", ..., ValueError, "the same length"),
        (
            "samples",
            "attributes.columns.1",
            "sample_id",
            ValueError,
            "column 'sample_id' twice",
        ),
        (
            "samples/donor",
            "attributes.levels",
            ["D1", "D1"],
            ValueError,
            "must not repeat",
        ),
        (
            "samples/donor",
            "attributes.levels",
            ["D1"],
            ValueError,
            "code 1, but codes run from -1",
        ),
        ("samples/donor", "attributes.levels", [1, 2], ValueError, "a list of strings"),
        ("samples/donor", "attributes.levels", None, ValueError, "a list of strings"),
        (
            "samples/visit",
            "attributes.numpy_dtype",
            None,
            ValueError,
            "must name a datetime64",
        ),
        (
            "samples/donor",
            "data_type",
            "int64",
            ValueError,
            "codes must be int32, not int64",
        ),
        (
            "samples/donor",
            "attributes.numpy_dtype",
            "M8[s]",
            ValueError,
            "both 'levels'",
        ),
        (
            "samples/visit",
            "attributes.numpy_dtype",
            "<M8[ms]",
            ValueError,
            "must name a datetime64",
        ),
        (
            "samples/visit",
            "attributes.numpy_dtype",
            "int64",
            ValueError,
            "must name a datetime64",
        ),
        (
            "samples/visit",
            "data_type",
            "uint64",
            ValueError,
            "stored as int64, not uint64",
        ),
        (
            "samples/age",
            "dimension_names",
            ["groups"],
            ValueError,
            r"must be \['samples'\]",
        ),
        (
            "layers/counts",
            "dimension_names",
            ["samples", "groups", "features"],
            ValueError,
            "dimension_names must be",
        ),
        (
            "layers/counts",
            "shape",
            [3, 4, 5],
            ValueError,
            r"\(n_groups, n_samples, n_features\) = \(3, 4, 6\), got \(3, 4, 5\)",
        ),
        (
            "layers/log_sum",
            "data_type",
            "float32",
            TypeError,
            "'log_sum' must have float64 dtype",
        ),
        ("n_cells", "data_type", "float64", ValueError, "n_cells must hold integers"),
        (
            "layers",
            "node_type",
            "array",
            ValueError,
            "describes a Zarr array; expected a group",
        ),
    ],
)
def test_files_that_are_not_valid_cube_files_raise_clear_errors(
    editable_store: Path,
    node: str,
    entry: str,
    value: Any,
    error: type,
    message: str,
) -> None:
    metadata = editable_store / node / "zarr.json"
    original = metadata.read_bytes()
    document = json.loads(original)
    edit(document, entry, value)
    metadata.write_text(json.dumps(document))
    try:
        with pytest.raises(error, match=message):
            read_cube(editable_store)
    finally:
        metadata.write_bytes(original)


def test_errors_from_the_cube_name_the_file(store: Path, tmp_path: Path) -> None:
    path = tmp_path / "c.cube.zarr"
    shutil.copytree(store, path)
    shutil.rmtree(path / "samples" / "sample_id")
    codes = np.array([0, 0, 1, 2], dtype=np.int32)  # S1 twice
    write_array(
        path / "samples" / "sample_id",
        codes,
        chunks=(4,),
        attributes={"levels": ["S1", "S2", "S3"]},
        dimension_names=["samples"],
    )
    with pytest.raises(ValueError, match="repeats the id 'S1'") as raised:
        read_cube(path)
    assert raised.value.__notes__ == [f"(while reading the cube file {path})"]
    shutil.rmtree(path / "n_cells")
    with pytest.raises(FileNotFoundError, match="n_cells does not exist"):
        read_cube(path)


def test_paths_that_hold_no_cube_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        read_cube(tmp_path / "missing.cube.zarr")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / ".zgroup").write_text('{"zarr_format": 2}')
    with pytest.raises(ValueError, match="is a Zarr v2 store"):
        read_cube(tmp_path / "old")
    write_array(tmp_path / "array", np.zeros(3), chunks=(3,))
    with pytest.raises(ValueError, match="describes a Zarr array; expected a group"):
        read_cube(tmp_path / "array")
    with pytest.raises(ValueError, match="is a URL"):
        read_cube("hf://datasets/x/y.cube.zarr")


def test_a_value_corrupted_on_disk_raises_when_its_group_is_read(
    store: Path, tmp_path: Path, cube: Cube
) -> None:
    path = tmp_path / "c.cube.zarr"
    shutil.copytree(store, path)
    chunk = path / "layers" / "counts" / "c" / "1" / "0" / "0"
    values = cube.layer("counts", "T").astype("<u2")
    values[1, 0] = 9  # S2 has no T cells, so every layer must be zero there
    chunk.write_bytes(gzip.compress(values.tobytes()))
    back = read_cube(path)
    assert_same_bits(back.layer("counts", "B"), cube.layer("counts", "B"))
    # Every read that takes the entry checks it; a selection without it reads.
    message = "'counts' is 9 at group 'T', sample 'S2', feature 'G0'"
    for read in (
        lambda: back.layer("counts", "T"),
        lambda: back.layer("counts"),
        back.to_memory,
        lambda: back.select(features=["G3", "G0"]).layer("counts", "T"),
    ):
        with pytest.raises(ValueError, match=message):
            read()
    assert_same_bits(
        back.select(features=["G3", "G1"]).layer("counts", "T"),
        cube.layer("counts", "T")[:, [3, 1]],
    )
