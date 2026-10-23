import contextlib
import gzip
import importlib
import json
import math
import os
import pickle
import struct
import sys
import tracemalloc
import zlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.io import zarr as zarr_module
from cytepool.io.zarr import (
    DATA_TYPES,
    check_compression,
    open_array,
    read_group_attributes,
    write_array,
    write_group,
)

FIXTURES = Path(__file__).parent / "data" / "cube_files"
FIXTURE_ARRAYS = sorted(path.name for path in (FIXTURES / "arrays").iterdir())
try:
    zstd: Any = importlib.import_module("compression.zstd")
except ImportError:  # Python 3.13
    zstd = None
needs_zstd = pytest.mark.skipif(zstd is None, reason="zstd needs Python 3.14 or newer")
REMOVE = object()  # deletes a field in make_array
LITTLE = {"name": "bytes", "configuration": {"endian": "little"}}


def regular(chunk_shape: list[int]) -> dict[str, Any]:
    return {"name": "regular", "configuration": {"chunk_shape": chunk_shape}}


def make_array(
    path: Path, chunks: dict[str, bytes] | None = None, **fields: Any
) -> Path:
    """Write a zarr.json (an int16 array of shape (2, 3) unless changed) and chunks."""
    document = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 3],
        "data_type": "int16",
        "chunk_grid": regular([1, 2]),
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [LITTLE],
    }
    document.update(fields)
    document = {key: value for key, value in document.items() if value is not REMOVE}
    path.mkdir(parents=True)
    (path / "zarr.json").write_text(json.dumps(document))
    for key, data in (chunks or {}).items():
        (path / key).parent.mkdir(parents=True, exist_ok=True)
        (path / key).write_bytes(data)
    return path


def assert_same_bits(actual: Any, expected: Any) -> None:
    """Assert equal dtype, shape and bytes, so NaN payloads and -0.0 count."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.dtype == expected.dtype.newbyteorder("=")
    assert actual.shape == expected.shape
    assert actual.tobytes() == expected.astype(actual.dtype).tobytes()


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


# Arrays written by zarr-python.


@pytest.fixture(scope="module")
def expected() -> dict[str, np.ndarray]:
    with np.load(FIXTURES / "expected.npz") as data:
        return dict(data)


@pytest.mark.parametrize("name", FIXTURE_ARRAYS)
def test_reads_the_arrays_zarr_python_wrote(
    name: str, expected: dict[str, np.ndarray]
) -> None:
    # Each array mixes a data type with a byte order, compressors, a chunk key
    # encoding and a fill value; validation/cube_files.py wrote them with zarr-python
    # and saved the values it gave it.
    path = FIXTURES / "arrays" / name
    document = json.loads((path / "zarr.json").read_text())
    if zstd is None and "zstd" in [codec["name"] for codec in document["codecs"]]:
        with pytest.raises(ValueError, match="zstd codec, which needs the compression"):
            open_array(path)
        return
    array = open_array(path)
    values = expected[f"array:{name}"]
    assert_same_bits(np.asarray(array), values)
    keys = [-1, (slice(1, 4), slice(None, None, -1)), (Ellipsis, 1), (slice(4, 0, -3),)]
    for key in keys:
        assert_same_bits(array[key], values[key])
    # Some chunks equal the fill value, so zarr-python did not store them.
    stored = [p for p in path.rglob("*") if p.is_file() and p.name != "zarr.json"]
    chunk_shape = document["chunk_grid"]["configuration"]["chunk_shape"]
    grid = [math.ceil(n / c) for n, c in zip(document["shape"], chunk_shape)]
    assert 0 < len(stored) < math.prod(grid)


# Selections against numpy.


@pytest.fixture(scope="module")
def grid(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, np.ndarray]:
    values = np.arange(7 * 8 * 3, dtype=np.int32).reshape(7, 8, 3) - 50
    path = tmp_path_factory.mktemp("grid") / "values"
    write_array(path, values, chunks=(2, 3, 2), compression=None)
    return open_array(path), values


@pytest.mark.parametrize(
    "key",
    [
        0,
        -1,
        np.int64(6),
        (2, 3),
        (1, 2, 0),
        (np.int8(-1), -8, -3),
        (slice(None),),
        (slice(1, 6), slice(2, 8, 3)),
        (slice(None, None, -1),),
        (slice(6, 0, -2), slice(None), 1),
        (Ellipsis, 2),
        (1, Ellipsis),
        (Ellipsis,),
        (slice(2, 2),),
        (slice(-3, None), slice(-100, 100)),
        (slice(None, None, 5), slice(1, None, 4), slice(None, None, -2)),
        (slice(5, 1, -1), slice(7, None, -3), slice(0, 3, 2)),
        (slice(None, None, 10**20), slice(None, None, -(10**20))),
    ],
)
def test_integer_and_slice_selections_match_numpy(
    grid: tuple[Any, np.ndarray], key: Any
) -> None:
    array, values = grid
    result, expected_result = array[key], values[key]
    assert type(result) is type(expected_result)  # a numpy scalar for (1, 2, 0)
    assert_same_bits(result, expected_result)


def test_reads_open_only_the_chunks_a_selection_touches(
    grid: tuple[Any, np.ndarray],
) -> None:
    array, values = grid
    with opened_files() as opened:
        block = array[3]  # row 3 lies in the second row of chunks
    assert_same_bits(block, values[3])
    root = array.path / "c"
    expected_chunks = {root / "1" / str(j) / str(k) for j in range(3) for k in range(2)}
    assert {Path(path) for path in opened} == expected_chunks
    with opened_files() as opened:
        array[0, 4, 2]
    assert [Path(path) for path in opened] == [root / "0" / "1" / "1"]
    with opened_files() as opened:
        assert array[2:2].shape == (0, 8, 3)
    assert opened == []


def test_selection_errors(grid: tuple[Any, np.ndarray]) -> None:
    array, _ = grid
    for key, error, message in [
        ([0, 1], TypeError, "integers, slices and '...', not list"),
        (np.array([0, 1]), TypeError, "not ndarray"),
        (None, TypeError, "not NoneType"),
        (True, TypeError, "not booleans"),
        ((0, 0, 0, 0), IndexError, "the array has 3 dimensions but 4 were indexed"),
        ((7,), IndexError, "index 7 is out of bounds for axis 0 with size 7"),
        ((0, -9), IndexError, "index -9 is out of bounds for axis 1 with size 8"),
        ((Ellipsis, 0, Ellipsis), IndexError, "a single ellipsis"),
        (slice(0.5, 2), TypeError, "slice indices must be integers"),
    ]:
        with pytest.raises(error, match=message):
            array[key]
    with pytest.raises(ValueError, match="cannot be converted without a copy"):
        np.asarray(array, copy=False)
    assert np.asarray(array, dtype=np.float64).dtype == np.float64


# Writing: the bytes the specification defines.


def test_writes_the_metadata_and_bytes_the_specification_defines(
    tmp_path: Path,
) -> None:
    values = np.array([[1, 2, 3], [0, 0, -2]], dtype=">i2")  # big-endian input
    path = tmp_path / "values"
    write_array(
        path,
        values,
        chunks=(1, 2),
        compression=None,
        attributes={"unit": "count"},
        dimension_names=["row", None],
    )
    assert json.loads((path / "zarr.json").read_text()) == {
        "zarr_format": 3,
        "node_type": "array",
        "shape": [2, 3],
        "data_type": "int16",
        "chunk_grid": regular([1, 2]),
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": 0,
        "codecs": [LITTLE],
        "attributes": {"unit": "count"},
        "dimension_names": ["row", None],
    }
    files = {
        p.relative_to(path).as_posix(): p.read_bytes()
        for p in path.rglob("*")
        if p.is_file() and p.name != "zarr.json"
    }
    # Little-endian two's complement in row-major order; edge chunks are padded with
    # the fill value, and chunk (1, 0) holds only zeros, so it is not stored.
    assert files == {
        "c/0/0": b"\x01\x00\x02\x00",
        "c/0/1": b"\x03\x00\x00\x00",
        "c/1/1": b"\xfe\xff\x00\x00",
    }


@pytest.mark.parametrize(
    ("dtype", "fill_value", "serializer"),
    [
        ("bool", False, {"name": "bytes"}),
        ("uint8", 0, {"name": "bytes"}),
        ("int64", 0, LITTLE),
        ("float16", 0.0, LITTLE),
        ("complex64", [0.0, 0.0], LITTLE),
    ],
)
def test_fill_value_and_byte_order_metadata(
    tmp_path: Path, dtype: str, fill_value: Any, serializer: dict
) -> None:
    write_array(tmp_path / "a", np.ones(3, dtype=dtype), chunks=(2,), compression=None)
    document = json.loads((tmp_path / "a" / "zarr.json").read_text())
    assert document["fill_value"] == fill_value
    assert type(document["fill_value"]) is type(fill_value)
    assert document["codecs"] == [serializer]  # one-byte types have no byte order


def test_gzip_chunks_are_reproducible_gzip_members(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    for name, level in (("default", None), ("best", 9)):
        write_array(tmp_path / name, values, chunks=(3, 4), level=level)
        data = (tmp_path / name / "c" / "0" / "0").read_bytes()
        # RFC 1952: magic bytes, deflate, and a zero modification time, so that the
        # same array gives the same bytes on every run.
        assert data[:3] == b"\x1f\x8b\x08"
        assert data[4:8] == b"\x00\x00\x00\x00"
        assert gzip.decompress(data) == values.astype("<f8").tobytes()
        codecs = json.loads((tmp_path / name / "zarr.json").read_text())["codecs"]
        assert codecs == [
            LITTLE,
            {"name": "gzip", "configuration": {"level": level or 4}},
        ]


@needs_zstd
def test_zstd_chunks_are_frames_with_checksums(tmp_path: Path) -> None:
    values = np.arange(12, dtype=np.int32)
    write_array(tmp_path / "a", values, chunks=(12,), compression="zstd", level=-5)
    data = (tmp_path / "a" / "c" / "0").read_bytes()
    assert data[:4] == b"\x28\xb5\x2f\xfd"  # RFC 8878 magic number
    assert data[4] & 0x04  # the frame header's content checksum flag
    assert zstd.decompress(data) == values.astype("<i4").tobytes()
    codecs = json.loads((tmp_path / "a" / "zarr.json").read_text())["codecs"]
    zstd_codec = {"name": "zstd", "configuration": {"level": -5, "checksum": True}}
    assert codecs == [LITTLE, zstd_codec]


COMPRESSIONS = ["gzip", None, pytest.param("zstd", marks=needs_zstd)]


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("dtype", list(DATA_TYPES.values()), ids=list(DATA_TYPES))
def test_every_data_type_round_trips_bit_for_bit(
    tmp_path: Path, dtype: Any, compression: str | None
) -> None:
    rng = np.random.default_rng(7)
    if dtype.kind == "b":
        values = rng.random((5, 7)) < 0.5
    elif dtype.kind in "iu":
        limits = np.iinfo(dtype)
        values = rng.integers(limits.min, limits.max, (5, 7), dtype, endpoint=True)
        values[0, :2] = limits.min, limits.max
    else:
        values = (rng.standard_normal((5, 7)) * 1e3).astype(dtype)
        if dtype.kind == "c":
            values += 1j * values[::-1]
        part = np.finfo(dtype)  # of the real and imaginary parts, for complex
        bits = values.view(f"u{part.bits // 8}").reshape(-1)  # writes go to values
        bits[0] = ((1 << part.nexp) - 1) << part.nmant | 5  # a NaN with a payload
        bits[1] = 1 << (part.bits - 1)  # negative zero
    values[3:] = 0  # with chunks of two rows, the last row of chunks holds zeros
    write_array(tmp_path / "a", values, chunks=(2, 3), compression=compression)
    stored = {
        p.relative_to(tmp_path / "a").as_posix()
        for p in (tmp_path / "a").rglob("*")
        if p.is_file() and p.name != "zarr.json"
    }
    nonzero = {
        f"c/{i}/{j}"
        for i in range(3)
        for j in range(3)
        if values[2 * i : 2 * i + 2, 3 * j : 3 * j + 3].view(np.uint8).any()
    }
    assert stored == nonzero and not any(key.startswith("c/2/") for key in stored)
    assert_same_bits(np.asarray(open_array(tmp_path / "a")), values)


def test_zero_dimensional_and_empty_arrays(tmp_path: Path) -> None:
    write_array(tmp_path / "scalar", np.array(2.5), chunks=(), compression=None)
    assert (tmp_path / "scalar" / "c").read_bytes() == struct.pack("<d", 2.5)
    scalar = open_array(tmp_path / "scalar")
    assert scalar[()] == 2.5 and type(scalar[()]) is np.float64
    assert isinstance(scalar[...], np.ndarray) and scalar[...].shape == ()
    with pytest.raises(TypeError, match="len"):
        len(scalar)
    write_array(tmp_path / "empty", np.zeros((0, 3), np.uint16), chunks=(1, 3))
    assert [p.name for p in (tmp_path / "empty").iterdir()] == ["zarr.json"]
    empty = open_array(tmp_path / "empty")
    assert np.asarray(empty).shape == (0, 3) and len(empty) == 0


def test_write_array_reads_array_likes_one_row_of_chunks_at_a_time(
    tmp_path: Path,
) -> None:
    values = np.arange(30, dtype=np.int64).reshape(5, 6)
    write_array(tmp_path / "source", values, chunks=(2, 4))
    source = open_array(tmp_path / "source")
    with opened_files() as opened:
        write_array(tmp_path / "copy", source, chunks=(2, 6), compression=None)
    reads = [path for path in opened if "/source/" in path]
    assert sorted(reads) == sorted(set(reads)) and len(reads) == 6  # each chunk once
    assert_same_bits(np.asarray(open_array(tmp_path / "copy")), values)


# Fill values, chunk keys and codec chains, from hand-written metadata.


@pytest.mark.parametrize(
    ("data_type", "fill_value", "little_endian_hex"),
    [
        ("bool", True, "01"),
        ("int8", -128, "80"),
        ("uint64", 2**64 - 1, "ff" * 8),
        ("int64", -(2**63), "00" * 7 + "80"),
        ("float64", "NaN", "000000000000f87f"),
        ("float32", "NaN", "0000c07f"),
        ("float16", "NaN", "007e"),
        ("float64", "Infinity", "000000000000f07f"),
        ("float32", "-Infinity", "000080ff"),
        ("float64", "0x7ff0000000000001", "010000000000f07f"),  # signalling NaN
        ("float32", "0x80000000", "00000080"),  # negative zero
        ("float16", "0x3c00", "003c"),  # 1.0
        ("float64", 0.1, "9a9999999999b93f"),
        ("float32", 0.1, "cdcccc3d"),  # rounded to the nearest float32
        ("float32", 1e300, "0000807f"),  # beyond float32: infinity
        ("float64", 3, "0000000000000840"),
        ("complex64", [1.5, "NaN"], "0000c03f0000c07f"),
        (
            "complex128",
            ["-Infinity", "0x" + "0" * 15 + "1"],
            "000000000000f0ff01" + "00" * 7,
        ),
    ],
)
def test_chunks_that_are_not_stored_read_as_the_fill_value(
    tmp_path: Path, data_type: str, fill_value: Any, little_endian_hex: str
) -> None:
    serializer = {"name": "bytes"} if data_type in ("bool", "int8") else LITTLE
    path = make_array(
        tmp_path / "a",
        shape=[3],
        chunk_grid=regular([2]),
        data_type=data_type,
        fill_value=fill_value,
        codecs=[serializer],
    )
    array = open_array(path)
    little = np.dtype(data_type).newbyteorder("<")
    assert [value.astype(little).tobytes().hex() for value in np.asarray(array)] == [
        little_endian_hex
    ] * 3
    assert np.asarray(array.fill_value).astype(little).tobytes().hex() == (
        little_endian_hex
    )


@pytest.mark.parametrize(
    ("encoding", "key"),
    [
        ({"name": "default", "configuration": {"separator": "/"}}, "c/1/0"),
        ({"name": "default", "configuration": {"separator": "."}}, "c.1.0"),
        ("default", "c/1/0"),
        ({"name": "v2", "configuration": {"separator": "/"}}, "1/0"),
        ({"name": "v2"}, "1.0"),
    ],
)
def test_chunk_key_encodings_and_big_endian_bytes(
    tmp_path: Path, encoding: Any, key: str
) -> None:
    path = make_array(
        tmp_path / "a",
        shape=[2, 2],
        data_type="int32",
        chunk_grid=regular([1, 2]),
        chunk_key_encoding=encoding,
        fill_value=-1,
        codecs=[{"name": "bytes", "configuration": {"endian": "big"}}],
        chunks={key: b"\x00\x00\x00\x05\xff\xff\xff\xfe"},
    )
    assert np.asarray(open_array(path)).tolist() == [[-1, -1], [5, -2]]


@pytest.mark.parametrize(("encoding", "key"), [("default", "c"), ("v2", "0")])
def test_zero_dimensional_chunk_keys(tmp_path: Path, encoding: str, key: str) -> None:
    path = make_array(
        tmp_path / "a",
        shape=[],
        chunk_grid=regular([]),
        data_type="float64",
        fill_value=1.5,
        chunk_key_encoding=encoding,
        codecs=[LITTLE],
        chunks={key: struct.pack("<d", -2.5)},
    )
    assert open_array(path)[()] == -2.5


def test_compressor_chains_and_gzip_members(tmp_path: Path) -> None:
    raw = np.array([1, 2, 3, 4], dtype="<i4").tobytes()
    gzip_codec = {"name": "gzip", "configuration": {"level": 1}}
    fields = {"shape": [4], "data_type": "int32", "chunk_grid": regular([4])}
    cases = [
        # Two members in one stream, as RFC 1952 allows.
        (
            "members",
            [LITTLE, gzip_codec],
            gzip.compress(raw[:6]) + gzip.compress(raw[6:]),
        ),
        ("twice", [LITTLE, gzip_codec, gzip_codec], gzip.compress(gzip.compress(raw))),
        ("short", [LITTLE, "gzip"], gzip.compress(raw)),  # a codec given by name
    ]
    if zstd is not None:
        chain = [LITTLE, gzip_codec, {"name": "zstd", "configuration": {"level": 0}}]
        cases.append(("zstd", chain, zstd.compress(gzip.compress(raw))))
    for name, codecs, data in cases:
        path = make_array(
            tmp_path / name, codecs=codecs, chunks={"c/0": data}, **fields
        )
        assert np.asarray(open_array(path)).tolist() == [1, 2, 3, 4], name


def test_fields_marked_must_understand_false_are_ignored(tmp_path: Path) -> None:
    extension = {"must_understand": False, "anything": [1, 2]}
    path = make_array(tmp_path / "a", future_extension=extension, attributes={"a": 1})
    array = open_array(path)
    assert array.attributes == {"a": 1} and array.dimension_names is None
    array.attributes["a"] = 2  # a copy
    assert array.attributes == {"a": 1}


# Strict reading.


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        (
            {"codecs": [LITTLE, {"name": "blosc", "configuration": {"clevel": 5}}]},
            "codec 'blosc'",
        ),
        ({"codecs": [LITTLE, {"name": "crc32c"}]}, "codec 'crc32c'"),
        (
            {
                "codecs": [
                    {"name": "transpose", "configuration": {"order": [1, 0]}},
                    LITTLE,
                ]
            },
            "codec 'transpose'",
        ),
        (
            {
                "codecs": [
                    {
                        "name": "sharding_indexed",
                        "configuration": {"chunk_shape": [1, 1]},
                    }
                ]
            },
            "codec 'sharding_indexed'",
        ),
        ({"data_type": "string"}, "data type 'string'"),
        ({"data_type": "r16"}, "data type 'r16'"),
        (
            {"data_type": {"name": "numpy.datetime64", "configuration": {"unit": "s"}}},
            "data type .*numpy.datetime64",
        ),
        (
            {"chunk_grid": {"name": "rectilinear", "configuration": {}}},
            "chunk grid 'rectilinear'",
        ),
        ({"chunk_key_encoding": {"name": "mapped"}}, "chunk key encoding 'mapped'"),
        (
            {"storage_transformers": [{"name": "caching"}]},
            "storage transformer 'caching'",
        ),
    ],
)
def test_unsupported_features_raise_errors_that_name_them(
    tmp_path: Path, fields: dict, message: str
) -> None:
    path = make_array(tmp_path / "a", **fields)
    with pytest.raises(ValueError, match=f"{message}.*read the store with zarr-python"):
        open_array(path)


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({"zarr_format": 2}, "zarr_format 2; this reader reads Zarr v3"),
        ({"zarr_format": True}, "zarr_format True"),
        ({"node_type": "group"}, "describes a Zarr group; expected an array"),
        ({"node_type": "table"}, "node_type 'table'"),
        ({"fill_value": REMOVE}, "lacks the required field 'fill_value'"),
        ({"spare": 1}, "field 'spare', which this reader does not know"),
        ({"spare": {"must_understand": True}}, "field 'spare'"),
        ({"shape": [2.0, 3]}, "shape must be a list of integers of at least 0"),
        ({"shape": [True, 3]}, "shape must be a list of integers"),
        ({"shape": [-1, 3]}, "shape must be a list of integers"),
        (
            {"chunk_grid": regular([0, 2])},
            "chunk_shape must be a list of integers of at least 1",
        ),
        ({"chunk_grid": regular([2])}, "has 1 dimensions, but the array has 2"),
        ({"chunk_grid": {"name": "regular"}}, "needs a chunk_shape"),
        (
            {
                "chunk_grid": {
                    "name": "regular",
                    "configuration": {"chunk_shape": [1, 2], "x": 0},
                }
            },
            "unknown member 'x'",
        ),
        (
            {
                "chunk_key_encoding": {
                    "name": "default",
                    "configuration": {"separator": "-"},
                }
            },
            "separator must be '/' or '.'",
        ),
        ({"fill_value": 2**15}, "integer from -32768 to 32767, got 32768"),
        ({"fill_value": 1.0}, "integer from -32768"),
        ({"fill_value": False}, "integer from -32768"),
        (
            {"data_type": "bool", "fill_value": 0, "codecs": [{"name": "bytes"}]},
            "true or false",
        ),
        ({"data_type": "float32", "fill_value": "nan"}, "must be a number, 'NaN'"),
        (
            {"data_type": "float32", "fill_value": "0x100000000"},
            "hexadecimal bit pattern",
        ),
        (
            {"data_type": "float32", "fill_value": "0x7fc0_0000"},
            "hexadecimal bit pattern",
        ),
        # zarr-python reads this as a float32 NaN widened to float64; the
        # specification asks for all 16 digits of a float64.
        ({"data_type": "float64", "fill_value": "0x7fc00000"}, "pattern of 16 digits"),
        ({"data_type": "float32", "fill_value": None}, "must be a number"),
        (
            {"data_type": "complex64", "fill_value": 1.0},
            "list of a real and an imaginary part",
        ),
        ({"codecs": []}, "codecs must be a non-empty list"),
        (
            {"codecs": [{"name": "gzip", "configuration": {"level": 1}}]},
            "must begin with 'bytes', not 'gzip'",
        ),
        ({"codecs": [{"name": "bytes"}]}, "needs an 'endian'"),
        (
            {"codecs": [{"name": "bytes", "configuration": {"endian": "middle"}}]},
            "endian must be 'little' or 'big'",
        ),
        ({"codecs": [LITTLE, LITTLE]}, "may appear only once"),
        (
            {"codecs": [LITTLE, {"name": "gzip", "configuration": {"level": 10}}]},
            "gzip level must be an integer from 0 to 9",
        ),
        (
            {
                "codecs": [
                    LITTLE,
                    {"name": "gzip", "configuration": {"level": 1, "x": 0}},
                ]
            },
            "unknown member 'x'",
        ),
        ({"codecs": [LITTLE | {"spare": 1}]}, "unknown member 'spare'"),
        ({"codecs": [7]}, "must be a name or an object"),
        ({"dimension_names": ["row"]}, "dimension_names must give a string or null"),
        ({"attributes": []}, "attributes must be an object"),
        ({"storage_transformers": {}}, "storage_transformers must be a list"),
    ],
)
def test_invalid_metadata_is_rejected(
    tmp_path: Path, fields: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        open_array(make_array(tmp_path / "a", **fields))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (b'{"zarr_format": 3, "fill_value": NaN}', "NaN is not JSON"),
        (b'{"zarr_format": 3, "zarr_format": 3}', "repeats the key 'zarr_format'"),
        (b"[]", "must hold a JSON object"),
        (b"\xff", "is not a valid JSON document"),
        (b'{"zarr_format": 3', "is not a valid JSON document"),
        (b"\xef\xbb\xbf{}", "is not a valid JSON document"),  # a byte order mark
    ],
)
def test_metadata_must_be_strict_json(
    tmp_path: Path, text: bytes, message: str
) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "zarr.json").write_bytes(text)
    with pytest.raises(ValueError, match=message):
        open_array(tmp_path / "a")


def corrupt_chunks() -> list[tuple[str, list[Any], bytes, str]]:
    raw = np.array([1, 2], dtype="<i4").tobytes()
    packed = gzip.compress(raw)
    cases = [
        (
            "short",
            [LITTLE],
            raw[:3],
            "holds 3 bytes after decoding, but a chunk of shape",
        ),
        ("truncated", [LITTLE, "gzip"], packed[:-6], "ends inside a gzip stream"),
        (
            "crc",
            [LITTLE, "gzip"],
            packed[:-8] + bytes([packed[-8] ^ 1]) + packed[-7:],
            "not valid gzip data",
        ),
        ("trailing", [LITTLE, "gzip"], packed + b"junk", "not valid gzip data"),
        (
            "bomb",
            [LITTLE, "gzip"],
            gzip.compress(bytes(10**6)),
            "decodes to more than the 8 bytes a chunk can hold",
        ),
        ("empty", [LITTLE, "gzip"], b"", "ends inside a gzip stream"),
    ]
    if zstd is not None:
        frame = zstd.compress(raw, options={zstd.CompressionParameter.checksum_flag: 1})
        cases += [
            (
                "zstd checksum",
                [LITTLE, "zstd"],
                frame[:-1] + bytes([frame[-1] ^ 1]),
                "not valid zstd data",
            ),
            (
                "zstd truncated",
                [LITTLE, "zstd"],
                frame[:-4],
                "ends inside a zstd stream",
            ),
            (
                "zstd bomb",
                [LITTLE, "zstd"],
                zstd.compress(bytes(10**6)),
                "decodes to more than the 8 bytes a chunk can hold",
            ),
        ]
    return cases


@pytest.mark.parametrize(("name", "codecs", "data", "message"), corrupt_chunks())
def test_corrupt_chunks_raise_when_read(
    tmp_path: Path, name: str, codecs: list[Any], data: bytes, message: str
) -> None:
    fields = {"shape": [2], "data_type": "int32", "chunk_grid": regular([2])}
    path = make_array(tmp_path / "a", codecs=codecs, chunks={"c/0": data}, **fields)
    with pytest.raises(ValueError, match=message):
        np.asarray(open_array(path))


def bounded_peak(read: Any, message: str) -> int:
    """Return the peak of Python's allocations while ``read()`` raises ``message``."""
    tracemalloc.start()
    try:
        with pytest.raises(ValueError, match=message):
            read()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("codec", ["gzip", pytest.param("zstd", marks=needs_zstd)])
def test_decoding_stops_just_past_the_size_of_a_chunk(
    tmp_path: Path, codec: str
) -> None:
    # 16 MiB of zeros compress to a few kilobytes, small enough to be a chunk of
    # 256 KiB; decoding must stop just past 256 KiB instead of producing 16 MiB.
    zeros = bytes(16 * 2**20)
    packed = gzip.compress(zeros) if codec == "gzip" else zstd.compress(zeros)
    fields = {"shape": [2**16], "data_type": "int32", "chunk_grid": regular([2**16])}
    path = make_array(
        tmp_path / "a", codecs=[LITTLE, codec], chunks={"c/0": packed}, **fields
    )
    message = f"decodes to more than the {2**18:,} bytes a chunk can hold"
    assert bounded_peak(lambda: open_array(path)[0], message) < 2**20


def test_every_decoding_step_and_the_file_read_are_bounded(tmp_path: Path) -> None:
    fields = {"shape": [2], "data_type": "int32", "chunk_grid": regular([2])}
    # The outer of two gzip steps decodes to 4 MiB; a valid inner stream for a chunk
    # of 8 bytes is at most 8 + 1 + 65,536 bytes long.
    nested = gzip.compress(bytes(4 * 2**20))  # 4,098 bytes
    codecs = [LITTLE, "gzip", "gzip"]
    path = make_array(
        tmp_path / "nested", codecs=codecs, chunks={"c/0": nested}, **fields
    )
    message = "decodes to more than the 65,545 bytes a chunk can hold"
    assert bounded_peak(lambda: open_array(path)[0], message) < 2**20
    # A stored chunk larger than any encoding of its chunk is refused unread.
    path = make_array(tmp_path / "large", codecs=[LITTLE], **fields)
    (path / "c").mkdir()
    with open(path / "c" / "0", "wb") as handle:
        handle.truncate(300 * 2**20)  # sparse: no blocks on disk
    message = "holds 314,572,800 bytes, more than any valid encoding of a chunk of 8"
    assert bounded_peak(lambda: open_array(path)[0], message) < 2**20
    # A small file is read as it is: a declared chunk of 128 MiB costs no more.
    fields_128 = {
        "shape": [2**25],
        "data_type": "int32",
        "chunk_grid": regular([2**25]),
    }
    truncated = {"c/0": gzip.compress(b"x")[:10]}
    path = make_array(
        tmp_path / "small", codecs=[LITTLE, "gzip"], chunks=truncated, **fields_128
    )
    message = "ends inside a gzip stream"
    assert bounded_peak(lambda: open_array(path)[0], message) < 2**20
    # Declared chunks beyond MAX_CHUNK_BYTES are refused when the array is opened.
    path = make_array(tmp_path / "huge", **(fields | {"chunk_grid": regular([2**40])}))
    with pytest.raises(ValueError, match="declares chunks of 4,398,046,511,104 bytes"):
        open_array(path)


def fixed_huffman_gzip(data: np.ndarray) -> bytes:
    """A gzip member (RFC 1952) whose one deflate block codes every byte, all of them
    at least 144, as a 9-bit fixed Huffman literal (RFC 1951, 3.2.6): 1/8 larger."""
    codes = data.astype(np.uint16) - 144 + 0b110010000
    literal_bits = (
        codes[:, None] >> np.arange(8, -1, -1)
    ) & 1  # most significant first
    bits = np.concatenate([[1, 1, 0], literal_bits.ravel(), np.zeros(7, np.uint16)])
    deflate = np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()
    raw = data.tobytes()
    trailer = struct.pack("<II", zlib.crc32(raw), len(raw))
    return b"\x1f\x8b\x08\x00\x00\x00\x00\x00\x00\xff" + deflate + trailer


def test_gzip_streams_larger_than_their_data_are_read(tmp_path: Path) -> None:
    # Encoders without a fallback to stored blocks grow incompressible data.
    values = np.random.default_rng(5).integers(144, 256, 2**16).astype(np.uint8)
    stream = fixed_huffman_gzip(values)
    assert zlib.decompress(stream, wbits=31) == values.tobytes()
    assert len(stream) > 1.12 * values.size
    fields = {"shape": [2**16], "data_type": "uint8", "chunk_grid": regular([2**16])}
    codecs = [{"name": "bytes"}, "gzip"]
    path = make_array(tmp_path / "a", codecs=codecs, chunks={"c/0": stream}, **fields)
    assert_same_bits(np.asarray(open_array(path)), values)


def test_a_chunk_holds_at_most_1024_gzip_members(tmp_path: Path) -> None:
    # A chunk of 64 KiB, so that 1,024 empty members (20 bytes each) fit the file.
    raw = np.arange(2**14, dtype="<i4").tobytes()
    fields = {"shape": [2**14], "data_type": "int32", "chunk_grid": regular([2**14])}
    empty = gzip.compress(b"")
    for count, ok in ((1023, True), (1024, False)):
        members = gzip.compress(raw) + empty * count
        path = make_array(
            tmp_path / str(count),
            codecs=[LITTLE, "gzip"],
            chunks={"c/0": members},
            **fields,
        )
        if ok:
            assert np.asarray(open_array(path)).tolist() == list(range(2**14))
        else:
            with pytest.raises(ValueError, match="holds more than 1,024 gzip members"):
                open_array(path)[0]


def test_bool_chunks_hold_only_zeros_and_ones(tmp_path: Path) -> None:
    fields = {"shape": [2], "data_type": "bool", "chunk_grid": regular([2])}
    fields |= {"fill_value": False}
    codecs = [{"name": "bytes"}]
    path = make_array(
        tmp_path / "a", codecs=codecs, chunks={"c/0": b"\x01\x02"}, **fields
    )
    with pytest.raises(ValueError, match="bool values other than 0 and 1"):
        open_array(path)[0]
    path = make_array(tmp_path / "b", codecs=codecs, chunks={"c/0/x": b""}, **fields)
    with pytest.raises(ValueError, match="'c/0' of the array at .* is not a file"):
        open_array(path)[0]


def test_paths_that_are_not_zarr_v3_nodes(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="does not exist"):
        open_array(tmp_path / "missing")
    (tmp_path / "plain").mkdir()
    with pytest.raises(
        FileNotFoundError, match="has no zarr.json, so it is not a Zarr v3"
    ):
        open_array(tmp_path / "plain")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / ".zarray").write_text("{}")
    with pytest.raises(ValueError, match="is a Zarr v2 store"):
        open_array(tmp_path / "old")
    (tmp_path / "file").write_text("x")
    with pytest.raises(NotADirectoryError, match="is a file"):
        open_array(tmp_path / "file")
    with pytest.raises(ValueError, match="is a URL"):
        open_array("s3://bucket/cube.zarr")
    with pytest.raises(TypeError, match="path must be a str or os.PathLike"):
        open_array(3)


def test_zstd_needs_python_3_14(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(zarr_module, "_zstd", lambda: None)
    path = make_array(
        tmp_path / "a", codecs=[LITTLE, {"name": "zstd", "configuration": {"level": 3}}]
    )
    message = "zstd codec, which needs the compression.zstd module of Python 3.14 or "
    with pytest.raises(ValueError, match=message + "newer; read the store with zarr"):
        open_array(path)
    with pytest.raises(
        ValueError, match="needs the compression.zstd module of Python 3.14"
    ):
        write_array(tmp_path / "b", np.zeros(2), chunks=(2,), compression="zstd")
    assert not (tmp_path / "b").exists()


# Changes on disk, pickling and the array's description.


def test_reads_fail_once_the_metadata_changes_or_disappears(tmp_path: Path) -> None:
    write_array(tmp_path / "a", np.arange(4), chunks=(2,))
    array = open_array(tmp_path / "a")
    metadata = tmp_path / "a" / "zarr.json"
    status = metadata.stat()
    os.utime(metadata, ns=(status.st_atime_ns, status.st_mtime_ns - 10**9))
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        array[0]
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        np.asarray(array)
    reopened = open_array(tmp_path / "a")
    # A file written anew with the same bytes and modification time, as a coarse
    # file system clock allows, is still a different file: it has another inode.
    status = metadata.stat()
    replacement = tmp_path / "zarr.json.new"
    replacement.write_bytes(metadata.read_bytes())
    os.utime(replacement, ns=(status.st_atime_ns, status.st_mtime_ns))
    os.replace(replacement, metadata)
    with pytest.raises(RuntimeError, match="changed after it was opened"):
        reopened[0]
    metadata.unlink()
    with pytest.raises(FileNotFoundError, match="no longer exists"):
        reopened[0]


def test_inodes_count_only_on_the_same_device() -> None:
    same = zarr_module._same_file
    stamp = (5, 100, 512, 10**18)  # device, inode, size, modification time
    assert same(stamp, stamp)
    assert not same(stamp, (5, 101, 512, 10**18))  # replaced on the same device
    # Another mount (another process or machine) may number inodes differently.
    assert same(stamp, (6, 999, 512, 10**18))
    assert not same(stamp, (6, 999, 512, 10**18 + 1))
    assert not same(stamp, (6, 100, 513, 10**18))


def test_a_chunk_that_changes_while_it_is_read_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_array(tmp_path / "a", np.arange(4, dtype=np.int64), chunks=(4,))
    array = open_array(tmp_path / "a")
    fstat = os.fstat

    def shrunk(descriptor: int) -> Any:  # the file grew after its size was taken
        status = fstat(descriptor)
        return type("Status", (), {"st_size": status.st_size - 1})()

    monkeypatch.setattr(zarr_module.os, "fstat", shrunk)
    with pytest.raises(ValueError, match="changed while it was being read"):
        array[0]


def test_arrays_pickle_and_describe_themselves(tmp_path: Path) -> None:
    values = np.arange(10, dtype=np.uint16).reshape(2, 5)
    write_array(
        tmp_path / "a",
        values,
        chunks=(1, 4),
        attributes={"k": [1]},
        dimension_names=["x", "y"],
    )
    array = open_array(tmp_path / "a")
    assert (array.shape, array.dtype, array.ndim, array.chunks) == (
        (2, 5),
        np.uint16,
        2,
        (1, 4),
    )
    assert (len(array), array.fill_value, array.dimension_names) == (2, 0, ("x", "y"))
    assert array.path == tmp_path / "a"
    where = str(tmp_path / "a")
    expected = f"<ZarrArray uint16 (2, 5), chunks (1, 4), bytes + gzip, at {where!r}>"
    assert repr(array) == expected
    assert_same_bits(np.asarray(pickle.loads(pickle.dumps(array))), values)


# Groups and arguments.


def test_groups_and_their_attributes(tmp_path: Path) -> None:
    write_group(tmp_path / "g", {"title": "ünïcode", "n": [1, None]})
    assert json.loads((tmp_path / "g" / "zarr.json").read_text("utf-8")) == {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {"title": "ünïcode", "n": [1, None]},
    }
    assert read_group_attributes(tmp_path / "g") == {"title": "ünïcode", "n": [1, None]}
    with pytest.raises(FileExistsError):
        write_group(tmp_path / "g")
    document = {"zarr_format": 3, "node_type": "group"}
    consolidated = {"kind": "inline", "must_understand": False, "metadata": {}}
    for name, extra in (("none", None), ("inline", consolidated)):
        (tmp_path / name).mkdir()
        text = json.dumps(document | {"consolidated_metadata": extra})
        (tmp_path / name / "zarr.json").write_text(text)
        assert read_group_attributes(tmp_path / name) == {}
    write_array(tmp_path / "array", np.zeros(2), chunks=(2,))
    with pytest.raises(ValueError, match="describes a Zarr array; expected a group"):
        read_group_attributes(tmp_path / "array")


def test_write_array_rejects_invalid_arguments(tmp_path: Path) -> None:
    values = np.zeros((2, 3))
    for kwargs, error, message in [
        (
            {"data": np.zeros(2, dtype="M8[s]")},
            TypeError,
            "datetime64.* is not a Zarr v3 core data type",
        ),
        (
            {"data": np.array(["a"], dtype=object)},
            TypeError,
            "not a Zarr v3 core data type",
        ),
        (
            {"chunks": (2,)},
            ValueError,
            "a positive integer for each of the 2 dimensions",
        ),
        ({"chunks": (0, 1)}, ValueError, "a positive integer"),
        ({"chunks": (True, 1)}, ValueError, "a positive integer"),
        (
            {"dimension_names": ["a"]},
            ValueError,
            "dimension_names must give a str or None",
        ),
        (
            {"attributes": {"x": math.nan}},
            ValueError,
            "attributes must be plain JSON values",
        ),
        (
            {"attributes": {"x": object()}},
            TypeError,
            "attributes must be plain JSON values",
        ),
        ({"attributes": {"x": "\udc80"}}, ValueError, "lone surrogate"),
        (
            {"compression": "lz4"},
            ValueError,
            "compression must be 'gzip', 'zstd' or None",
        ),
        (
            {"data": [1.0, 2.0]},
            TypeError,
            "data must be a numpy array or an array-like",
        ),
        ({"chunks": 2}, TypeError, "chunks must be a list or tuple, got 2"),
        (
            {"dimension_names": "xy"},
            TypeError,
            "dimension_names must be a list or tuple",
        ),
        (
            {"attributes": [("a", 1)]},
            TypeError,
            "attributes must be a mapping, got list",
        ),
        ({"chunks": (2**28, 3)}, ValueError, "hold more than MAX_CHUNK_BYTES"),
    ]:
        arguments = {"data": values, "chunks": (1, 3)} | kwargs
        with pytest.raises(error, match=message):
            write_array(tmp_path / "a", **arguments)
        assert not (tmp_path / "a").exists()
    write_array(tmp_path / "a", values, chunks=(1, 3))
    with pytest.raises(FileExistsError):
        write_array(tmp_path / "a", values, chunks=(1, 3))

    class Rows:
        shape, dtype = (2, 3), np.dtype(np.float64)

        def __getitem__(self, rows: slice) -> np.ndarray:
            return np.zeros((1, 3))

    with pytest.raises(ValueError, match=r"data\[0:2\] gave shape \(1, 3\)"):
        write_array(tmp_path / "rows", Rows(), chunks=(2, 3))


def test_check_compression() -> None:
    assert check_compression("gzip", None) == 4
    assert check_compression("gzip", np.int64(9)) == 9
    assert check_compression(None, None) is None
    if zstd is not None:
        assert check_compression("zstd", None) == 3
        assert check_compression("zstd", -131072) == -131072
    for compression, level, error, message in [
        (None, 1, ValueError, "level applies only with compression"),
        ("gzip", 10, ValueError, "the gzip level must be from 0 to 9, got 10"),
        ("gzip", -1, ValueError, "from 0 to 9"),
        ("gzip", 1.0, TypeError, "level must be an integer, got float"),
        ("gzip", True, TypeError, "level must be an integer, got bool"),
        (
            "GZIP",
            None,
            ValueError,
            "compression must be 'gzip', 'zstd' or None, got 'GZIP'",
        ),
        (1, None, TypeError, "compression must be 'gzip', 'zstd' or None, got int"),
    ]:
        with pytest.raises(error, match=message):
            check_compression(compression, level)
