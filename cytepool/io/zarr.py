"""A minimal reader and writer for Zarr v3 arrays and groups on a local file system.

This module implements, with numpy and the standard library only, the part of the
Zarr core specification, version 3.1 (Miles et al. 2025), that cube files need, with
the ``bytes`` and ``gzip`` codecs of the Zarr specifications (Zarr developers 2023)
and the ``zstd`` codec of its registered extensions (Maitin-Shepard 2025).

Reading supports:

- group and array metadata documents (``zarr.json``);
- the core data types bool, int8 to int64, uint8 to uint64, float16, float32,
  float64, complex64 and complex128, the floating-point types in the binary formats
  of IEEE 754 (IEEE 2019);
- the regular chunk grid, and the default and v2 chunk key encodings with either
  separator;
- fill values in every form the specification permits; a chunk that is not stored
  reads as the fill value;
- the ``bytes`` codec in either byte order, followed by any sequence of ``gzip``
  codecs, whose chunks are gzip streams (RFC 1952) of DEFLATE data (RFC 1951; both
  Deutsch 1996), and ``zstd`` codecs, whose chunks are Zstandard frames (RFC 8878;
  Collet and Kucherawy 2021); ``zstd`` needs the ``compression.zstd`` module of
  Python 3.14 or newer.

Anything else, such as another data type, the ``blosc``, ``crc32c``, ``transpose`` or
``sharding_indexed`` codec, another chunk grid or a storage transformer, raises a
``ValueError`` that names it; zarr-python reads such stores.

A read opens only the chunks that hold a selected element and decodes each of them once,
one at a time, so it holds the result and about one chunk: compressed chunks are read
from disk and decoded in pieces of at most 256 KiB. A zstd decoder also keeps its
frame's window, outside Python's allocations: up to the size of the chunk, and 2 MiB at
zstd's default level of 3. A chunk compressed twice holds the output of both steps.
Selections take integers and slices, and, through ``ZarrArray.oindex``, integer arrays
and boolean masks along any axes.

Reading is strict. A metadata document must be a JSON object (Bray 2017) without
repeated keys or NaN and Infinity tokens, and every field must be valid; a field the
specification does not define is an error unless its value is an object with
``"must_understand": false``. A chunk must decode to exactly the bytes its shape
needs, compressed streams must be complete and pass their checksums, and bool chunks
may hold only the bytes 0 and 1. Memory and time stay bounded on malformed files:
arrays whose chunks would hold more than 2 GiB (``MAX_CHUNK_BYTES``) are refused, at
most the bytes a valid encoding of a chunk can take are read from disk, every
decoding step stops just past the most bytes a valid chunk can produce, and a chunk
may hold at most 1,024 gzip members or zstd frames (writers use one).

Writing produces arrays with a regular chunk grid, the default chunk key encoding with
``/``, a fill value of zero (``false`` for bool) and the ``bytes`` codec, little
endian, followed by ``gzip``, ``zstd`` (with a checksum) or no compression. A chunk
whose bytes are all zero equals the fill value and is not stored.

References:
    Miles A, Striebel J, Rzepka N, Maitin-Shepard J, Moore J, eds (2025). Zarr core
    specification, version 3.1. https://zarr-specs.readthedocs.io/en/latest/v3/core/

    Zarr developers (2023). Bytes codec, version 1.0, and gzip codec, version 1.0. Zarr
    specifications, https://zarr-specs.readthedocs.io/en/latest/v3/codecs/

    Maitin-Shepard J (2025). Zstd codec. Zarr extensions,
    https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zstd

    Deutsch P (1996). GZIP file format specification version 4.3. RFC 1952, RFC Editor.
    doi:10.17487/RFC1952

    Deutsch P (1996). DEFLATE Compressed Data Format Specification version 1.3.
    RFC 1951, RFC Editor. doi:10.17487/RFC1951

    Collet Y, Kucherawy M (2021). Zstandard Compression and the 'application/zstd' Media
    Type. RFC 8878, RFC Editor. doi:10.17487/RFC8878

    Bray T (2017). The JavaScript Object Notation (JSON) Data Interchange Format.
    RFC 8259, RFC Editor. doi:10.17487/RFC8259

    IEEE (2019). IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019.
    doi:10.1109/IEEESTD.2019.8766229
"""

import gzip
import importlib
import itertools
import json
import math
import operator
import os
import re
import zlib
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from functools import cache
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, Literal, NamedTuple

import numpy as np

from .._masked_arrays import reject_masked, unmasked_array

METADATA_FILE = "zarr.json"
"""The name of the metadata document of every Zarr v3 group and array."""

DATA_TYPES: Mapping[str, np.dtype] = MappingProxyType(
    {
        name: np.dtype(name)
        for name in (
            "bool",
            "int8",
            "int16",
            "int32",
            "int64",
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "float16",
            "float32",
            "float64",
            "complex64",
            "complex128",
        )
    }
)
"""The Zarr v3 core data types this module reads and writes, by name."""

DEFAULT_LEVELS: Mapping[str, int] = MappingProxyType({"gzip": 4, "zstd": 3})
"""The compression level ``write_array`` uses when none is given."""

MAX_CHUNK_BYTES = 2**31
"""The largest chunk, in bytes once decoded, that this module reads or writes."""

_LEVEL_RANGES = {"gzip": (0, 9), "zstd": (-131072, 22)}
_TYPE_NAMES = (
    "bool, int8 to int64, uint8 to uint64, float16 to float64, complex64 and complex128"
)
_ARRAY_FIELDS = frozenset(
    {
        "zarr_format",
        "node_type",
        "shape",
        "data_type",
        "chunk_grid",
        "chunk_key_encoding",
        "fill_value",
        "codecs",
        "attributes",
        "storage_transformers",
        "dimension_names",
    }
)
_REQUIRED_ARRAY_FIELDS = (
    "shape",
    "data_type",
    "chunk_grid",
    "chunk_key_encoding",
    "fill_value",
    "codecs",
)
_GROUP_FIELDS = frozenset(
    {"zarr_format", "node_type", "attributes", "consolidated_metadata"}
)
_HEX_DIGITS = re.compile(r"[0-9a-fA-F]+")
_FIRST_PIECE = 4096  # input a decoder gets first in each gzip member or zstd frame
# The most input a decoder gets and the most output it returns at a time, and the
# most of a chunk file read at a time.
_LAST_PIECE = 1 << 18
_MAX_MEMBERS = 1024  # gzip members or zstd frames in one chunk; writers use one
_USE_ZARR_PYTHON = "read the store with zarr-python instead"


class _ArrayMetadata:
    """The parsed metadata document of an array."""

    __slots__ = (
        "shape",
        "dtype",
        "chunks",
        "separator",
        "v2_keys",
        "fill_value",
        "byte_order",
        "compressors",
        "attributes",
        "dimension_names",
        "limits",
    )

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
        separator: str,
        v2_keys: bool,
        fill_value: np.ndarray,
        byte_order: Literal["<", ">"],
        compressors: tuple[str, ...],
        attributes: dict[str, Any],
        dimension_names: tuple[str | None, ...] | None,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.chunks = chunks
        self.separator = separator
        self.v2_keys = v2_keys
        self.fill_value = fill_value
        self.byte_order = byte_order
        self.compressors = compressors
        self.attributes = attributes
        self.dimension_names = dimension_names
        # The most bytes the bytes codec takes (a whole chunk), then the most each
        # compressor outwards may decode to, and last the most a stored chunk holds.
        # Deflate with fixed Huffman codes can grow incompressible data by 1/8, and
        # gzip and zstd framing adds a few bytes per block and member, so each step
        # allows 1/8 plus 64 KiB.
        limits = [math.prod(chunks) * dtype.itemsize]
        for _ in compressors:
            limits.append(limits[-1] + limits[-1] // 8 + 65536)
        self.limits = tuple(limits)

    def chunk_key(self, index: tuple[int, ...]) -> str:
        """Return the store key of the chunk at a grid index."""
        if self.v2_keys:
            return self.separator.join(map(str, index)) if index else "0"
        return self.separator.join(("c", *map(str, index)))


class _Piece(NamedTuple):
    """The part of one chunk that a selection takes along one axis."""

    chunk: int
    source: slice | np.ndarray  # positions within the chunk
    target: slice | np.ndarray  # positions within the selection


class ZarrArray:
    """A Zarr v3 array in a local directory, read chunk by chunk.

    Open one with ``open_array``. Indexing with integers and slices, as in
    ``array[2]``, ``array[1, :, 100:200]`` or ``array[..., ::2]``, reads only the chunks
    the selection touches and returns a new numpy array in native byte order, which
    owns its memory (a numpy scalar when every axis gets an integer, as in numpy).
    ``array.oindex[...]`` also takes, along any axis, a 1-D integer array or a boolean
    mask (as a list or a numpy array), and selects every combination of the positions
    each axis selects (orthogonal indexing, as in zarr-python):
    ``array.oindex[0, [3, 1], [5, 9]]`` is
    ``numpy.asarray(array)[0][numpy.ix_([3, 1], [5, 9])]``. Integer arrays may be in
    any order, repeat positions and count negative positions from the end.
    ``numpy.asarray(array)`` reads the whole array. A chunk that is not stored reads as
    the fill value.

    Every read first checks that the array's ``zarr.json`` is still the file it was
    when the array was opened, and raises otherwise, so an array replaced on disk is
    never read through out-of-date metadata: on the same device the inode, size and
    modification time must match; through another mount, where inode numbers can
    differ, the size and modification time. A store replaced by an identical copy
    therefore also counts as changed on the same device.

    A ``ZarrArray`` holds no open files, so it can be pickled and read from another
    process that sees the same path.
    """

    __slots__ = ("_path", "_metadata", "_stamp")

    _path: Path
    _metadata: _ArrayMetadata
    _stamp: tuple[int, int, int, int]

    def __init__(
        self, path: Path, metadata: _ArrayMetadata, stamp: tuple[int, int, int, int]
    ) -> None:
        self._path = path
        self._metadata = metadata
        self._stamp = stamp

    @property
    def path(self) -> Path:
        """The array's directory."""
        return self._path

    @property
    def shape(self) -> tuple[int, ...]:
        """The array's shape."""
        return self._metadata.shape

    @property
    def dtype(self) -> np.dtype:
        """The data type of the values a read returns, in native byte order."""
        return self._metadata.dtype

    @property
    def ndim(self) -> int:
        """The number of dimensions."""
        return len(self._metadata.shape)

    @property
    def chunks(self) -> tuple[int, ...]:
        """The chunk shape of the regular chunk grid."""
        return self._metadata.chunks

    @property
    def fill_value(self) -> Any:
        """The value of elements in chunks that are not stored, as a numpy scalar."""
        return self._metadata.fill_value[()]

    @property
    def attributes(self) -> dict[str, Any]:
        """A copy of the array's attributes."""
        return deepcopy(self._metadata.attributes)

    @property
    def dimension_names(self) -> tuple[str | None, ...] | None:
        """The name of each dimension (None for an unnamed one), or None."""
        return self._metadata.dimension_names

    def __len__(self) -> int:
        """Return the length of the first dimension."""
        if not self._metadata.shape:
            raise TypeError("len() of a 0-d array")
        return self._metadata.shape[0]

    def __repr__(self) -> str:
        codecs = " + ".join(("bytes", *self._metadata.compressors))
        return (
            f"<ZarrArray {self.dtype} {self.shape}, chunks {self.chunks}, {codecs}, "
            f"at {str(self._path)!r}>"
        )

    def __getitem__(self, key: Any) -> Any:
        """Read the elements an integer, slice or tuple of them (with ``...``) selects.

        Raises:
            TypeError: If the key holds something other than integers, slices and
                one ellipsis, such as a mask, a list or ``None`` (``oindex`` takes
                masks and lists), or holds a numpy masked array.
            IndexError: If an integer is out of range or there are too many indices.
            ValueError: If a chunk is corrupt.
            RuntimeError: If the array changed on disk after it was opened.
        """
        return self._get(key, orthogonal=False)

    @property
    def oindex(self) -> "_OrthogonalIndexer":
        """Orthogonal indexing, which selects along each axis independently.

        ``array.oindex[key]`` reads what ``key`` selects. ``key`` gives each axis an
        integer, a slice, a 1-D integer array or a boolean mask with one value per
        position (arrays as lists or numpy arrays), and may hold one ``...``. The
        result has an axis for every axis that gets no integer, holding the selected
        positions in the order given, as in zarr-python's ``oindex``. It raises the
        errors of ``__getitem__``, and IndexError for an array that is not 1-D or a
        mask of the wrong length.
        """
        return _OrthogonalIndexer(self)

    def _get(self, key: Any, orthogonal: bool) -> Any:
        # operator.index, which also reads slice bounds, accepts a 0-d masked
        # integer and ignores its mask.
        reject_masked(key, "index")
        ranges, kept, scalar = _selection(key, self._metadata.shape, orthogonal)
        self._check_unchanged()
        values = self._read(ranges, kept)
        return values[()] if scalar else values

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        """Read the whole array, for ``numpy.asarray`` and ``numpy.array``."""
        if copy is False:
            raise ValueError(
                "a ZarrArray is read from disk into new memory, so it cannot be "
                "converted without a copy (copy=False)"
            )
        self._check_unchanged()
        shape = self._metadata.shape
        values = self._read([range(n) for n in shape], shape)
        return values if dtype is None else values.astype(dtype, copy=False)

    def _check_unchanged(self) -> None:
        try:
            stamp = _file_stamp(self._path / METADATA_FILE)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"the array at {self._path} no longer exists"
            ) from None
        if not _same_file(self._stamp, stamp):
            raise RuntimeError(
                f"the array at {self._path} changed after it was opened (its "
                f"{METADATA_FILE} was rewritten); open it again"
            )

    def _read(
        self, ranges: list[range | np.ndarray], shape: tuple[int, ...]
    ) -> np.ndarray:
        """Read what ``ranges`` select into a new array of ``shape``.

        ``shape`` leaves out the axes that an integer selected. The array owns its
        memory, so a read-only flag set on it holds for every view of it.
        """
        metadata = self._metadata
        result = np.empty(shape, metadata.dtype)
        if result.size == 0:
            return result
        # A view with an axis for every range, through which the chunks are copied.
        out = result.reshape(tuple(len(selected) for selected in ranges))
        per_axis = [
            _pieces(selected, size)
            for selected, size in zip(ranges, metadata.chunks, strict=True)
        ]
        # Each chunk index comes once, so every chunk is read and decoded once.
        for pieces in itertools.product(*per_axis):
            self._copy_chunk(out, pieces)
        return result

    def _copy_chunk(self, out: np.ndarray, pieces: tuple[_Piece, ...]) -> None:
        """Copy what a selection takes from one chunk into ``out``.

        The decoded chunk is freed on return, so a read holds one chunk at a time.
        """
        block = self._read_chunk(tuple(piece.chunk for piece in pieces))
        target = _index(tuple(piece.target for piece in pieces))
        if block is None:
            out[target] = self._metadata.fill_value
        else:
            out[target] = block[_index(tuple(piece.source for piece in pieces))]

    def _read_chunk(self, index: tuple[int, ...]) -> np.ndarray | None:
        """Read and decode the chunk at a grid index; None if it is not stored."""
        metadata = self._metadata
        key = metadata.chunk_key(index)
        where = f"the chunk {key!r} of the array at {self._path}"
        compressors = metadata.compressors
        try:
            with open(self._path / key, "rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if size > metadata.limits[-1]:
                    raise ValueError(
                        f"{where} holds {size:,} bytes, more than any valid encoding "
                        f"of a chunk of {metadata.limits[0]:,} bytes"
                    )
                if compressors:
                    # The outermost compressor decodes the file as it is read.
                    data: Any = _decode_stream(
                        _Input(b"", handle, size, where),
                        compressors[-1],
                        metadata.limits[len(compressors) - 1],
                        where,
                    )
                else:
                    data = handle.read(size + 1)  # read allocates what it is asked for
                    if len(data) != size:
                        raise ValueError(f"{where} changed while it was being read")
        except FileNotFoundError:
            return None
        except (IsADirectoryError, NotADirectoryError):
            raise ValueError(f"{where} is not a file") from None
        for position in reversed(range(len(compressors) - 1)):
            codec, limit = compressors[position], metadata.limits[position]
            data = _decode_stream(_Input(data), codec, limit, where)
        return _chunk_values(data, metadata, where)


class _OrthogonalIndexer:
    """What ``ZarrArray.oindex`` returns: ``array.oindex[key]`` reads a selection."""

    __slots__ = ("_array",)

    def __init__(self, array: ZarrArray) -> None:
        self._array = array

    def __getitem__(self, key: Any) -> Any:
        return self._array._get(key, orthogonal=True)


def open_array(path: str | os.PathLike[str]) -> ZarrArray:
    """Open a Zarr v3 array for reading.

    Args:
        path: The array's directory, which holds its ``zarr.json``.

    Returns:
        A ``ZarrArray``; nothing but the metadata is read.

    Raises:
        FileNotFoundError: If there is no array metadata at ``path``.
        ValueError: If the metadata is invalid or uses a feature this module does not
            support (the message names it).
    """
    directory = _local_path(path)
    document, stamp, where = _read_document(directory)
    return ZarrArray(directory, _parse_array(document, where), stamp)


def read_group_attributes(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read and check a Zarr v3 group's metadata and return its attributes.

    Consolidated metadata, if present, is ignored: each node is read from its own
    ``zarr.json``.

    Args:
        path: The group's directory, which holds its ``zarr.json``.

    Raises:
        FileNotFoundError: If there is no group metadata at ``path``.
        ValueError: If the metadata is invalid, describes an array, or belongs to a
            Zarr v2 store.
    """
    directory = _local_path(path)
    document, _, where = _read_document(directory)
    _check_node(document, "group", _GROUP_FIELDS, (), where)
    consolidated = document.get("consolidated_metadata")
    if consolidated is not None and not isinstance(consolidated, dict):
        raise ValueError(f"{where}: consolidated_metadata must be an object or null")
    return _attributes(document, where)


def write_group(
    path: str | os.PathLike[str], attributes: Mapping[str, Any] | None = None
) -> None:
    """Write a Zarr v3 group's metadata, creating its directory if needed.

    Args:
        path: The group's directory; its parent must exist.
        attributes: JSON-serialisable attributes.

    Raises:
        FileExistsError: If the directory already has a ``zarr.json``.
        ValueError, TypeError: If an attribute cannot be written as JSON.
    """
    directory = _local_path(path)
    document = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": dict(attributes or {}),
    }
    text = _json_bytes(document)
    directory.mkdir(exist_ok=True)
    _write_new_file(directory / METADATA_FILE, text)


def write_array(
    path: str | os.PathLike[str],
    data: Any,
    *,
    chunks: Sequence[int],
    compression: str | None = "gzip",
    level: int | None = None,
    attributes: Mapping[str, Any] | None = None,
    dimension_names: Sequence[str | None] | None = None,
) -> None:
    """Write an array as a new Zarr v3 array directory.

    Chunks are written first and ``zarr.json`` last, so a write that fails leaves a
    directory that does not open as an array. A chunk whose bytes are all zero is not
    stored, since zero is the fill value.

    Args:
        path: The directory to create; its parent must exist and it must not.
        data: A numpy array, or any object with ``shape`` and ``dtype`` whose
            ``data[start:stop]`` returns those rows along the first axis, such as a
            ``ZarrArray``. It is read one row of chunks at a time.
        chunks: The chunk shape: a positive length for each dimension.
        compression: ``"gzip"``, ``"zstd"`` or None (see ``check_compression``).
        level: The compression level, or None for the default.
        attributes: JSON-serialisable attributes.
        dimension_names: A name or None for each dimension.

    Raises:
        TypeError: If ``data`` has no shape and dtype, its data type is not a core
            data type, it is or gives rows as a numpy masked array, or another
            argument has the wrong type.
        ValueError: If an argument is invalid, a chunk would exceed
            ``MAX_CHUNK_BYTES``, or ``data`` gives rows of the wrong shape or dtype.
        FileExistsError: If ``path`` exists.
    """
    directory = _local_path(path)
    for argument, value in (
        ("data", data),
        ("chunks", chunks),
        ("dimension_names", dimension_names),
    ):
        reject_masked(value, argument)
    if not (hasattr(data, "shape") and hasattr(data, "dtype")):
        raise TypeError(
            "data must be a numpy array or an array-like with shape and dtype, got "
            f"{type(data).__name__}"
        )
    shape = tuple(operator.index(n) for n in data.shape)
    dtype = np.dtype(data.dtype)
    name = _data_type_name(dtype)
    for argument, value in (("chunks", chunks), ("dimension_names", dimension_names)):
        if value is not None and (
            isinstance(value, str) or not isinstance(value, Sequence)
        ):
            raise TypeError(f"{argument} must be a list or tuple, got {value!r}")
    if attributes is not None and not isinstance(attributes, Mapping):
        raise TypeError(
            f"attributes must be a mapping, got {type(attributes).__name__}"
        )
    chunk_shape = tuple(chunks)
    if len(chunk_shape) != len(shape) or not all(
        isinstance(n, int | np.integer) and not isinstance(n, bool) and n >= 1
        for n in chunk_shape
    ):
        raise ValueError(
            f"chunks must give a positive integer for each of the {len(shape)} "
            f"dimensions, got {chunks!r}"
        )
    chunk_shape = tuple(int(n) for n in chunk_shape)
    if math.prod(chunk_shape) * dtype.itemsize > MAX_CHUNK_BYTES:
        raise ValueError(
            f"chunks {chunk_shape} of {dtype} values hold more than MAX_CHUNK_BYTES "
            f"({MAX_CHUNK_BYTES:,} bytes), which open_array refuses; choose smaller "
            "chunks"
        )
    level = check_compression(compression, level)
    document: dict[str, Any] = {
        "zarr_format": 3,
        "node_type": "array",
        "shape": list(shape),
        "data_type": name,
        "chunk_grid": {
            "name": "regular",
            "configuration": {"chunk_shape": list(chunk_shape)},
        },
        "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}},
        "fill_value": _zero_fill_value(dtype),
        "codecs": _codec_metadata(dtype, compression, level),
        "attributes": dict(attributes or {}),
    }
    if dimension_names is not None:
        names = list(dimension_names)
        if len(names) != len(shape) or not all(
            item is None or isinstance(item, str) for item in names
        ):
            raise ValueError(
                f"dimension_names must give a str or None for each of the "
                f"{len(shape)} dimensions, got {dimension_names!r}"
            )
        document["dimension_names"] = names
    text = _json_bytes(document)  # fails on a bad attribute before writing anything
    directory.mkdir()
    _write_chunks(directory, data, shape, dtype, chunk_shape, compression, level)
    _write_new_file(directory / METADATA_FILE, text)


def check_compression(compression: str | None, level: int | None) -> int | None:
    """Check a compression and level for ``write_array`` and return the level used.

    Args:
        compression: ``"gzip"``, ``"zstd"`` (Python 3.14 or newer) or None for no
            compression.
        level: 0 to 9 for gzip, -131072 to 22 for zstd (0 asks zstd for its default),
            or None for ``DEFAULT_LEVELS`` (4 for gzip, 3 for zstd). It must be None
            without compression.

    Returns:
        The compression level, or None without compression.

    Raises:
        TypeError: If ``compression`` is not a string or None, or ``level`` is not an
            integer.
        ValueError: If ``compression`` is unknown or unavailable, or ``level`` is out
            of range or given without compression.
    """
    if compression is None:
        if level is not None:
            raise ValueError("level applies only with compression 'gzip' or 'zstd'")
        return None
    if not isinstance(compression, str):
        raise TypeError(
            "compression must be 'gzip', 'zstd' or None, got "
            f"{type(compression).__name__}"
        )
    if compression not in DEFAULT_LEVELS:
        raise ValueError(
            f"compression must be 'gzip', 'zstd' or None, got {compression!r}"
        )
    if compression == "zstd" and _zstd() is None:
        raise ValueError(
            "compression='zstd' needs the compression.zstd module of Python 3.14 or "
            "newer; use compression='gzip'"
        )
    if level is None:
        return DEFAULT_LEVELS[compression]
    if not isinstance(level, int | np.integer) or isinstance(level, bool):
        raise TypeError(f"level must be an integer, got {type(level).__name__}")
    low, high = _LEVEL_RANGES[compression]
    if not low <= level <= high:
        raise ValueError(
            f"the {compression} level must be from {low} to {high}, got {level}"
        )
    return int(level)


# Paths and metadata documents.


def _local_path(path: Any) -> Path:
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


def _file_stamp(path: str | os.PathLike[str]) -> tuple[int, int, int, int]:
    """Return the device, inode, size and modification time of a file."""
    status = os.stat(path)
    return status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns


def _same_file(old: tuple[int, ...], new: tuple[int, ...]) -> bool:
    """Return whether two stamps describe one unchanged file.

    The inode is compared only on the same device: another mount of the same files,
    in another process or on another machine, may number inodes differently.
    """
    return old == new if old[0] == new[0] else old[2:] == new[2:]


def _read_document(
    directory: Path,
) -> tuple[dict[str, Any], tuple[int, int, int, int], str]:
    """Read a node's zarr.json; return it, its size and mtime, and its path."""
    path = directory / METADATA_FILE
    try:
        with open(path, "rb") as handle:
            status = os.fstat(handle.fileno())
            raw = handle.read()
    except FileNotFoundError:
        if not os.path.lexists(directory):
            raise FileNotFoundError(f"{directory} does not exist") from None
        if any(os.path.lexists(directory / name) for name in (".zgroup", ".zarray")):
            raise ValueError(
                f"{directory} is a Zarr v2 store; this reader reads Zarr v3 "
                f"({METADATA_FILE}) only, and zarr-python can convert it"
            ) from None
        raise FileNotFoundError(
            f"{directory} has no {METADATA_FILE}, so it is not a Zarr v3 group or array"
        ) from None
    except NotADirectoryError:
        raise NotADirectoryError(
            f"{directory} is a file, not a Zarr group or array directory"
        ) from None
    except IsADirectoryError:
        raise ValueError(f"{path} is a directory, not a metadata document") from None
    where = str(path)
    try:
        document = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_repeated_keys,
            parse_constant=_reject_constant,
        )
    except (ValueError, RecursionError) as error:
        raise ValueError(f"{where} is not a valid JSON document: {error}") from None
    if not isinstance(document, dict):
        raise ValueError(f"{where} must hold a JSON object")
    stamp = (status.st_dev, status.st_ino, status.st_size, status.st_mtime_ns)
    return document, stamp, where


def _object_without_repeated_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"an object repeats the key {key!r}")
        result[key] = value
    return result


def _reject_constant(name: str) -> None:
    raise ValueError(
        f"{name} is not JSON; non-finite fill values are the strings 'NaN', "
        "'Infinity' and '-Infinity'"
    )


def _json_bytes(document: Mapping[str, Any]) -> bytes:
    try:
        text = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False)
        return (text + "\n").encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            "metadata must be valid Unicode text, but it holds the lone surrogate "
            f"{error.object[error.start : error.end]!r}"
        ) from None
    except (TypeError, ValueError) as error:
        raise type(error)(f"attributes must be plain JSON values: {error}") from None


def _write_new_file(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)


def _check_node(
    document: dict[str, Any],
    node_type: str,
    fields: frozenset[str],
    required: tuple[str, ...],
    where: str,
) -> None:
    zarr_format = document.get("zarr_format")
    if type(zarr_format) is not int or zarr_format != 3:
        raise ValueError(
            f"{where} has zarr_format {zarr_format!r}; this reader reads Zarr v3 "
            "(zarr_format 3)"
        )
    found = document.get("node_type")
    if found != node_type:
        if found in ("array", "group"):
            article = "an" if node_type == "array" else "a"
            raise ValueError(
                f"{where} describes a Zarr {found}; expected {article} {node_type}"
            )
        raise ValueError(f"{where} has node_type {found!r}; expected {node_type!r}")
    for field in required:
        if field not in document:
            raise ValueError(f"{where} lacks the required field {field!r}")
    for field, value in document.items():
        if field not in fields and not (
            isinstance(value, dict) and value.get("must_understand") is False
        ):
            raise ValueError(
                f"{where} has the field {field!r}, which this reader does not know; "
                f"{_USE_ZARR_PYTHON}"
            )


def _attributes(document: dict[str, Any], where: str) -> dict[str, Any]:
    attributes = document.get("attributes", {})
    if not isinstance(attributes, dict):
        raise ValueError(f"{where}: attributes must be an object")
    return attributes


def _parse_array(document: dict[str, Any], where: str) -> _ArrayMetadata:
    _check_node(document, "array", _ARRAY_FIELDS, _REQUIRED_ARRAY_FIELDS, where)
    shape = _integers(document["shape"], "shape", 0, where)
    dtype = _data_type(document["data_type"], where)
    chunks = _chunk_grid(document["chunk_grid"], len(shape), where)
    separator, v2_keys = _chunk_key_encoding(document["chunk_key_encoding"], where)
    chunk_bytes = math.prod(chunks) * dtype.itemsize
    if chunk_bytes > MAX_CHUNK_BYTES:
        raise ValueError(
            f"{where} declares chunks of {chunk_bytes:,} bytes, more than the "
            f"{MAX_CHUNK_BYTES:,} this reader decodes; {_USE_ZARR_PYTHON}"
        )
    fill_value = _fill_value(document["fill_value"], dtype, where)
    byte_order, compressors = _codecs(document["codecs"], dtype, where)
    transformers = document.get("storage_transformers", [])
    if not isinstance(transformers, list):
        raise ValueError(f"{where}: storage_transformers must be a list")
    if transformers:
        name, _ = _extension(transformers[0], "a storage transformer", where)
        raise ValueError(
            f"{where} uses the storage transformer {name!r}, which this reader does "
            f"not support; {_USE_ZARR_PYTHON}"
        )
    names = document.get("dimension_names")
    if names is not None and (
        not isinstance(names, list)
        or len(names) != len(shape)
        or not all(item is None or isinstance(item, str) for item in names)
    ):
        raise ValueError(
            f"{where}: dimension_names must give a string or null for each of the "
            f"{len(shape)} dimensions, got {names!r}"
        )
    return _ArrayMetadata(
        shape=shape,
        dtype=dtype,
        chunks=chunks,
        separator=separator,
        v2_keys=v2_keys,
        fill_value=fill_value,
        byte_order=byte_order,
        compressors=compressors,
        attributes=_attributes(document, where),
        dimension_names=None if names is None else tuple(names),
    )


def _integers(value: Any, field: str, minimum: int, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not all(
        type(item) is int and item >= minimum for item in value
    ):
        raise ValueError(
            f"{where}: {field} must be a list of integers of at least {minimum}, got "
            f"{value!r}"
        )
    return tuple(value)


def _extension(value: Any, what: str, where: str) -> tuple[str, dict[str, Any]]:
    """Return the name and configuration of an extension (a name or an object)."""
    if isinstance(value, str):
        return value, {}
    if not isinstance(value, dict):
        raise ValueError(
            f"{where}: {what} must be a name or an object with a 'name', got {value!r}"
        )
    for member in value:
        if member not in ("name", "configuration", "must_understand"):
            raise ValueError(f"{where}: {what} has the unknown member {member!r}")
    name = value.get("name")
    configuration = value.get("configuration", {})
    if not isinstance(name, str):
        raise ValueError(f"{where}: {what} needs a 'name' string, got {value!r}")
    if not isinstance(configuration, dict):
        raise ValueError(f"{where}: the configuration of {name!r} must be an object")
    if not isinstance(value.get("must_understand", True), bool):
        raise ValueError(f"{where}: must_understand of {name!r} must be true or false")
    return name, configuration


def _check_configuration(
    configuration: dict[str, Any], allowed: tuple[str, ...], name: str, where: str
) -> None:
    for member in configuration:
        if member not in allowed:
            raise ValueError(
                f"{where}: the {name} configuration has the unknown member {member!r}"
            )


def _data_type(value: Any, where: str) -> np.dtype:
    if isinstance(value, str) and value in DATA_TYPES:
        return DATA_TYPES[value]
    raise ValueError(
        f"{where} has the data type {value!r}, which this reader does not support "
        f"(it reads {_TYPE_NAMES}); {_USE_ZARR_PYTHON}"
    )


def _data_type_name(dtype: np.dtype) -> str:
    native = dtype.newbyteorder("=")
    for name, candidate in DATA_TYPES.items():
        if native == candidate:
            return name
    raise TypeError(
        f"dtype {dtype} is not a Zarr v3 core data type; arrays hold {_TYPE_NAMES}"
    )


def _chunk_grid(value: Any, ndim: int, where: str) -> tuple[int, ...]:
    name, configuration = _extension(value, "chunk_grid", where)
    if name != "regular":
        raise ValueError(
            f"{where} uses the chunk grid {name!r}, which this reader does not "
            f"support (it reads 'regular'); {_USE_ZARR_PYTHON}"
        )
    _check_configuration(configuration, ("chunk_shape",), "regular chunk grid", where)
    if "chunk_shape" not in configuration:
        raise ValueError(f"{where}: the regular chunk grid needs a chunk_shape")
    chunks = _integers(configuration["chunk_shape"], "chunk_shape", 1, where)
    if len(chunks) != ndim:
        raise ValueError(
            f"{where}: chunk_shape {list(chunks)} has {len(chunks)} dimensions, but "
            f"the array has {ndim}"
        )
    return chunks


def _chunk_key_encoding(value: Any, where: str) -> tuple[str, bool]:
    name, configuration = _extension(value, "chunk_key_encoding", where)
    if name not in ("default", "v2"):
        raise ValueError(
            f"{where} uses the chunk key encoding {name!r}, which this reader does "
            f"not support (it reads 'default' and 'v2'); {_USE_ZARR_PYTHON}"
        )
    _check_configuration(
        configuration, ("separator",), f"{name} chunk key encoding", where
    )
    separator = configuration.get("separator", "/" if name == "default" else ".")
    if separator not in ("/", "."):
        raise ValueError(
            f"{where}: the chunk key separator must be '/' or '.', got {separator!r}"
        )
    return separator, name == "v2"


def _fill_value(value: Any, dtype: np.dtype, where: str) -> np.ndarray:
    """Return the fill value as a 0-d array of the native dtype."""
    kind = dtype.kind
    if kind == "b":
        if type(value) is not bool:
            raise ValueError(
                f"{where}: the fill value of a bool array must be true or false, got "
                f"{value!r}"
            )
        return np.array(value, dtype=dtype)
    if kind in "iu":
        limits = np.iinfo(dtype)
        if type(value) is not int or not limits.min <= value <= limits.max:
            raise ValueError(
                f"{where}: the fill value of a {dtype} array must be an integer from "
                f"{limits.min} to {limits.max}, got {value!r}"
            )
        return np.array(value, dtype=dtype)
    if kind == "f":
        data = _float_fill_bytes(value, dtype, where)
    elif isinstance(value, list) and len(value) == 2:
        part = np.dtype(f"float{4 * dtype.itemsize}")
        data = b"".join(_float_fill_bytes(item, part, where) for item in value)
    else:
        raise ValueError(
            f"{where}: the fill value of a {dtype} array must be a list of a real and "
            f"an imaginary part, got {value!r}"
        )
    # From the bit pattern, so that hexadecimal NaN payloads survive unchanged.
    stored = np.frombuffer(data, dtype=dtype.newbyteorder("<")).reshape(())
    return stored.astype(dtype)


def _float_fill_bytes(value: Any, dtype: np.dtype, where: str) -> bytes:
    """Return a floating-point fill value as the little-endian bytes of ``dtype``."""
    little = dtype.newbyteorder("<")
    if type(value) is int or type(value) is float:
        try:
            number = float(value)
        except OverflowError:  # an integer beyond the float64 range
            number = math.copysign(math.inf, value)
        with np.errstate(over="ignore"):  # rounds to the nearest value, or infinity
            return np.array(number, dtype=little).tobytes()
    special = {"NaN": math.nan, "Infinity": math.inf, "-Infinity": -math.inf}
    if isinstance(value, str) and value in special:
        return np.array(special[value], dtype=little).tobytes()
    if (
        isinstance(value, str)
        and value.startswith("0x")
        and len(value) == 2 + 2 * dtype.itemsize
        and _HEX_DIGITS.fullmatch(value[2:])
    ):
        return int(value[2:], 16).to_bytes(dtype.itemsize, "little")
    raise ValueError(
        f"{where}: the fill value of a {dtype} array must be a number, 'NaN', "
        f"'Infinity', '-Infinity' or a hexadecimal bit pattern of "
        f"{2 * dtype.itemsize} digits such as '0x{'0' * 2 * dtype.itemsize}', got "
        f"{value!r}"
    )


def _codecs(
    value: Any, dtype: np.dtype, where: str
) -> tuple[Literal["<", ">"], tuple[str, ...]]:
    """Return the byte order of the bytes codec and the compressors that follow it."""
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where}: codecs must be a non-empty list")
    codecs = [
        _extension(item, f"codec {position}", where)
        for position, item in enumerate(value)
    ]
    name, configuration = codecs[0]
    if name in DEFAULT_LEVELS:
        raise ValueError(f"{where}: the codecs must begin with 'bytes', not {name!r}")
    if name != "bytes":
        raise ValueError(_unsupported_codec(name, where))
    _check_configuration(configuration, ("endian",), "bytes codec", where)
    endian = configuration.get("endian")
    if endian is None and dtype.itemsize > 1:
        raise ValueError(
            f"{where}: the bytes codec needs an 'endian' ('little' or 'big') for "
            f"{dtype} data"
        )
    if endian not in (None, "little", "big"):
        raise ValueError(
            f"{where}: the bytes codec's endian must be 'little' or 'big', got "
            f"{endian!r}"
        )
    compressors = []
    for name, configuration in codecs[1:]:
        if name == "bytes":
            raise ValueError(f"{where}: the bytes codec may appear only once, first")
        if name not in DEFAULT_LEVELS:
            raise ValueError(_unsupported_codec(name, where))
        allowed = ("level",) if name == "gzip" else ("level", "checksum")
        _check_configuration(configuration, allowed, f"{name} codec", where)
        level = configuration.get("level", DEFAULT_LEVELS[name])
        low, high = _LEVEL_RANGES[name]
        if type(level) is not int or not low <= level <= high:
            raise ValueError(
                f"{where}: the {name} level must be an integer from {low} to {high}, "
                f"got {level!r}"
            )
        if not isinstance(configuration.get("checksum", False), bool):
            raise ValueError(f"{where}: the zstd checksum must be true or false")
        if name == "zstd" and _zstd() is None:
            raise ValueError(
                f"{where} uses the zstd codec, which needs the compression.zstd module "
                f"of Python 3.14 or newer; {_USE_ZARR_PYTHON}"
            )
        compressors.append(name)
    return (">" if endian == "big" else "<"), tuple(compressors)


def _unsupported_codec(name: str, where: str) -> str:
    return (
        f"{where} uses the codec {name!r}, which this reader does not support (it "
        f"reads 'bytes' followed by any of 'gzip' and 'zstd'); {_USE_ZARR_PYTHON}"
    )


# Selections.


def _selection(
    key: Any, shape: tuple[int, ...], orthogonal: bool
) -> tuple[list[range | np.ndarray], tuple[int, ...], bool]:
    """Return what every axis selects, the result shape and whether it is a scalar.

    An axis selects a range of positions or, in an orthogonal selection, an array of
    positions in the order given.
    """
    keys = key if isinstance(key, tuple) else (key,)
    ellipses = sum(1 for item in keys if item is Ellipsis)
    if ellipses > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    named = len(keys) - ellipses
    if named > len(shape):
        raise IndexError(
            f"too many indices: the array has {len(shape)} dimensions but {named} "
            "were indexed"
        )
    if ellipses:
        at = next(i for i, item in enumerate(keys) if item is Ellipsis)
        keys = keys[:at] + (slice(None),) * (len(shape) - named) + keys[at + 1 :]
    else:
        keys = keys + (slice(None),) * (len(shape) - named)
    ranges: list[range | np.ndarray] = []
    kept = []
    for axis, (item, n) in enumerate(zip(keys, shape, strict=True)):
        if isinstance(item, slice):
            selected = range(*item.indices(n))
            ranges.append(selected)
            kept.append(len(selected))
            continue
        if orthogonal and (
            isinstance(item, list) or (isinstance(item, np.ndarray) and item.ndim)
        ):
            positions = _axis_positions(item, n, axis)
            ranges.append(positions)
            kept.append(positions.size)
            continue
        if isinstance(item, bool | np.bool_) or (
            isinstance(item, np.ndarray) and item.dtype.kind == "b"
        ):
            raise TypeError(
                "ZarrArray.oindex takes a mask as an array of booleans, not a single "
                "boolean"
                if orthogonal
                else "ZarrArray indices are integers and slices, not booleans; use "
                "array.oindex[...] to index an axis with a mask"
            )
        try:
            position = operator.index(item)
        except TypeError:
            raise TypeError(
                "ZarrArray.oindex indices are integers, slices, '...', and 1-D integer "
                f"arrays and boolean masks as lists or numpy arrays, not "
                f"{type(item).__name__}"
                if orthogonal
                else "ZarrArray indices are integers, slices and '...', not "
                f"{type(item).__name__}; use array.oindex[...] to index axes with "
                "integer arrays or masks, or read the array with numpy.asarray() for "
                "other indexing"
            ) from None
        if not -n <= position < n:
            raise IndexError(
                f"index {position} is out of bounds for axis {axis} with size {n}"
            )
        position %= n
        ranges.append(range(position, position + 1))
    scalar = not ellipses and not kept and named == len(shape)
    return ranges, tuple(kept), scalar


def _axis_positions(item: list[Any] | np.ndarray, n: int, axis: int) -> np.ndarray:
    """Return the positions an integer array or a boolean mask selects along an axis."""
    values = np.asarray(item)
    if values.ndim != 1:
        raise IndexError(
            f"oindex takes 1-D arrays, but axis {axis} got one of shape {values.shape}"
        )
    if values.dtype.kind == "b":
        if values.size != n:
            raise IndexError(
                f"a boolean mask for axis {axis} needs {n} values, got {values.size}"
            )
        return np.flatnonzero(values)
    if values.size == 0:  # [] is float64 in numpy
        return np.empty(0, dtype=np.intp)
    if values.dtype.kind not in "iu":
        raise TypeError(
            f"oindex takes integer arrays and boolean masks, but axis {axis} got "
            f"values of dtype {values.dtype}"
        )
    outside = (values < -n) | (values >= n)
    if outside.any():
        raise IndexError(
            f"index {values[outside][0]} is out of bounds for axis {axis} with size {n}"
        )
    positions: np.ndarray = values.astype(np.intp) % n
    return positions


def _pieces(selected: range | np.ndarray, size: int) -> list[_Piece]:
    """Split one axis of a selection by the chunks (of length ``size``) it touches.

    Each chunk comes once, with the positions it gives in the order the selection
    takes them, so positions in any order or repeated read every chunk once.
    """
    pieces: list[_Piece] = []
    if isinstance(selected, range) and selected.step == 1:
        if not selected:
            return pieces
        for chunk in range(selected.start // size, (selected.stop - 1) // size + 1):
            begin = max(selected.start, chunk * size)
            end = min(selected.stop, (chunk + 1) * size)
            pieces.append(
                _Piece(
                    chunk,
                    slice(begin - chunk * size, end - chunk * size),
                    slice(begin - selected.start, end - selected.start),
                )
            )
        return pieces
    if isinstance(selected, range):
        # An integer dtype keeps steps beyond int64, such as ::10**20, exact.
        indices = np.arange(selected.start, selected.stop, selected.step, dtype=np.intp)
    else:
        indices = selected
    if not indices.size:
        return pieces
    chunk_of = indices // size
    order = np.argsort(chunk_of, kind="stable")
    starts = np.flatnonzero(np.diff(chunk_of[order])) + 1
    for positions in np.split(order, starts):
        chunk = int(chunk_of[positions[0]])
        pieces.append(
            _Piece(
                chunk, _compact(indices[positions] - chunk * size), _compact(positions)
            )
        )
    return pieces


def _compact(positions: np.ndarray) -> slice | np.ndarray:
    """Return evenly spaced increasing positions as a slice, which numpy copies faster."""
    first = int(positions[0])
    if positions.size == 1:
        return slice(first, first + 1)
    step = int(positions[1]) - first
    if step > 0 and bool((np.diff(positions) == step).all()):
        return slice(first, int(positions[-1]) + 1, step)
    return positions


def _index(parts: tuple[slice | np.ndarray, ...]) -> Any:
    """Return a numpy index that takes every combination of what each axis selects."""
    if sum(isinstance(part, np.ndarray) for part in parts) <= 1:
        return parts  # numpy keeps a single array index in its axis's place
    return np.ix_(*(_positions(part) for part in parts))


def _positions(part: slice | np.ndarray) -> np.ndarray:
    if isinstance(part, slice):
        return np.arange(part.start, part.stop, part.step or 1)
    return part


# Chunks.


@cache
def _zstd() -> ModuleType | None:
    """Return the standard library's zstd module (Python 3.14 or newer), or None."""
    try:
        return importlib.import_module("compression.zstd")
    except ImportError:
        return None


def _chunk_values(data: Any, metadata: _ArrayMetadata, where: str) -> np.ndarray:
    """Return the decoded bytes of a chunk as its values, after checking them."""
    size = metadata.limits[0]
    if len(data) != size:
        raise ValueError(
            f"{where} holds {len(data)} bytes after decoding, but a chunk of shape "
            f"{metadata.chunks} and data type {metadata.dtype} has {size}"
        )
    if metadata.dtype.kind == "b" and np.frombuffer(data, dtype=np.uint8).max() > 1:
        raise ValueError(f"{where} has bool values other than 0 and 1")
    stored = metadata.dtype.newbyteorder(metadata.byte_order)
    return np.frombuffer(data, dtype=stored).reshape(metadata.chunks)


class _Input:
    """The input of one decoding step: bytes in memory, or a chunk file.

    A file is read in pieces of at most ``_LAST_PIECE`` bytes as the decoder asks for
    them, so little of it is in memory at a time, and reading raises if the file turns
    out shorter or longer than the ``size`` it had when it was opened. Pieces are
    views of what was read, and bytes the decoder hands back are put in front of the
    rest, so nothing but those bytes is copied.
    """

    __slots__ = ("_parts", "_handle", "_left", "_where")

    def __init__(
        self, data: Any, handle: Any = None, size: int = 0, where: str = ""
    ) -> None:
        # The bytes not decoded yet, the next ones last.
        self._parts = [memoryview(data)] if len(data) else []
        self._handle = handle
        self._left = size  # bytes of the file not read yet
        self._where = where
        if handle is not None and not size and handle.read(1):
            self._changed()

    def take(self, n: int) -> memoryview:
        """Remove and return up to the next ``n`` bytes; nothing only at the end."""
        if not self._parts and self._left:
            wanted = min(_LAST_PIECE, self._left)
            data = self._handle.read(wanted)
            self._left -= wanted
            if len(data) != wanted or (not self._left and self._handle.read(1)):
                self._changed()
            self._parts.append(memoryview(data))
        if not self._parts:
            return memoryview(b"")
        part = self._parts.pop()
        if len(part) > n:
            self._parts.append(part[n:])
            part = part[:n]
        return part

    def give_back(self, data: bytes) -> None:
        """Put bytes the decoder did not use in front of the others."""
        if data:
            self._parts.append(memoryview(data))

    def at_end(self) -> bool:
        return not self._parts and not self._left

    def _changed(self) -> None:
        raise ValueError(f"{self._where} changed while it was being read")


class _Output:
    """Decoded bytes, copied once into memory that never moves.

    They go into a bytearray while they fit in one piece, and beyond that into one
    buffer of ``limit`` bytes, the most a valid chunk decodes to, so a small or
    malformed stream costs only what it decodes to.
    """

    __slots__ = ("_small", "_buffer", "_limit", "size")

    def __init__(self, limit: int) -> None:
        self._small = bytearray()
        self._buffer: np.ndarray | None = None
        self._limit = limit
        self.size = 0

    def add(self, data: bytes) -> None:
        end = self.size + len(data)
        if self._buffer is None and end > _LAST_PIECE:
            self._buffer = np.empty(self._limit, dtype=np.uint8)
            self._buffer[: self.size] = memoryview(self._small)
            self._small = bytearray()
        if self._buffer is None:
            self._small += data
        else:
            self._buffer[self.size : end] = memoryview(data)
        self.size = end

    def value(self) -> bytearray | np.ndarray:
        """Return the decoded bytes, as a bytearray or a 1-D array of uint8."""
        return self._small if self._buffer is None else self._buffer[: self.size]


def _decode_stream(
    source: _Input, codec: str, limit: int, where: str
) -> bytearray | np.ndarray:
    """Decode the gzip members or zstd frames ``source`` holds, up to ``limit`` bytes.

    More output raises ValueError. The decoder gets the input in pieces that start
    small at each member or frame and double while it lasts, so that the leftover
    input it copies when a member ends is small; with at most ``_MAX_MEMBERS``
    members, decoding takes time in proportion to the input. It returns at most
    ``_LAST_PIECE`` bytes at a time; while it holds more output, it gets small pieces
    or none, so that the input it hands back stays small. Decoding therefore holds
    the output and about 1 MiB besides, and a zstd decoder also its window.
    """
    zstd = _zstd() if codec == "zstd" else None
    if codec == "zstd" and zstd is None:  # open_array refuses such arrays
        raise ValueError(f"{where} needs zstd, which needs Python 3.14 or newer")
    error_type = zlib.error if zstd is None else zstd.ZstdError
    output = _Output(limit)
    for _ in range(_MAX_MEMBERS):  # one gzip member or zstd frame at a time
        # wbits=31: the gzip format, with its CRC-32 checked; a zstd frame's
        # checksum, when present, is checked too.
        decoder: Any = (
            zlib.decompressobj(wbits=31) if zstd is None else zstd.ZstdDecompressor()
        )
        piece_size = _FIRST_PIECE
        held = False  # whether the decoder has output it could not return yet
        while True:
            if held and zstd is not None:
                piece: Any = b""  # zstd keeps the input it has not decoded
            else:
                piece = source.take(piece_size)
                if not piece and not held:
                    raise ValueError(
                        f"{where} ends inside a {codec} stream (truncated)"
                    )
            asked = min(limit - output.size + 1, _LAST_PIECE)
            try:
                decoded = decoder.decompress(piece, asked)
            except error_type as error:
                raise ValueError(
                    f"{where} is not valid {codec} data ({error})"
                ) from None
            if output.size + len(decoded) > limit:
                raise ValueError(
                    f"{where} decodes to more than the {limit:,} bytes a chunk can "
                    f"hold ({codec})"
                )
            output.add(decoded)
            if decoder.eof:
                source.give_back(decoder.unused_data)
                break
            if zstd is None:  # zlib hands back the input it has not decoded
                source.give_back(decoder.unconsumed_tail)
                held = len(decoded) == asked
            else:
                held = not decoder.needs_input
            piece_size = _FIRST_PIECE if held else min(2 * piece_size, _LAST_PIECE)
        if source.at_end():
            return output.value()
    unit = "members" if zstd is None else "frames"
    raise ValueError(f"{where} holds more than {_MAX_MEMBERS:,} {codec} {unit}")


def _zero_fill_value(dtype: np.dtype) -> Any:
    kind = dtype.kind
    if kind == "b":
        return False
    if kind in "iu":
        return 0
    return 0.0 if kind == "f" else [0.0, 0.0]


def _codec_metadata(
    dtype: np.dtype, compression: str | None, level: int | None
) -> list[dict[str, Any]]:
    codecs: list[dict[str, Any]] = [
        {"name": "bytes"}
        if dtype.itemsize == 1
        else {"name": "bytes", "configuration": {"endian": "little"}}
    ]
    if compression == "gzip":
        codecs.append({"name": "gzip", "configuration": {"level": level}})
    elif compression == "zstd":
        codecs.append(
            {"name": "zstd", "configuration": {"level": level, "checksum": True}}
        )
    return codecs


def _encoder(compression: str | None, level: int | None) -> Callable[[bytes], bytes]:
    if compression == "gzip":
        assert level is not None
        gzip_level = level
        # mtime=0 keeps the output the same from one run to the next.
        return lambda raw: gzip.compress(raw, compresslevel=gzip_level, mtime=0)
    if compression == "zstd":
        zstd = _zstd()
        assert zstd is not None
        options = {
            zstd.CompressionParameter.compression_level: level,
            zstd.CompressionParameter.checksum_flag: 1,
        }
        return lambda raw: bytes(zstd.compress(raw, options=options))
    return lambda raw: raw


def _write_chunks(
    directory: Path,
    data: Any,
    shape: tuple[int, ...],
    dtype: np.dtype,
    chunks: tuple[int, ...],
    compression: str | None,
    level: int | None,
) -> None:
    if 0 in shape:
        return
    encode = _encoder(compression, level)
    stored = dtype.newbyteorder("<")
    made: set[Path] = set()

    def write(index: tuple[int, ...], region: np.ndarray) -> None:
        if region.shape == chunks:
            buffer = np.asarray(region, dtype=stored, order="C")
        else:  # an edge chunk: the part outside the array holds the fill value
            buffer = np.zeros(chunks, dtype=stored)
            buffer[tuple(slice(0, n) for n in region.shape)] = region
        raw = buffer.tobytes()
        if not np.frombuffer(raw, dtype=np.uint8).any():
            return  # equal to the fill value, bit for bit
        parts = ("c", *map(str, index))
        parent = directory.joinpath(*parts[:-1])
        if parent not in made:
            parent.mkdir(parents=True, exist_ok=True)
            made.add(parent)
        _write_new_file(parent / parts[-1], encode(raw))

    if not shape:
        write((), unmasked_array(data, "data"))
        return
    grid = [math.ceil(n / c) for n, c in zip(shape, chunks, strict=True)]
    rows = chunks[0]
    for row in range(grid[0]):
        start, stop = row * rows, min(shape[0], (row + 1) * rows)
        block = unmasked_array(data[start:stop], f"data[{start}:{stop}]")
        expected = (stop - start, *shape[1:])
        if block.shape != expected or block.dtype.newbyteorder(
            "="
        ) != dtype.newbyteorder("="):
            raise ValueError(
                f"data[{start}:{stop}] gave shape {block.shape} and dtype "
                f"{block.dtype}; expected {expected} and {dtype}"
            )
        for rest in itertools.product(*(range(n) for n in grid[1:])):
            region = block[
                (
                    slice(None),
                    *(slice(j * c, (j + 1) * c) for j, c in zip(rest, chunks[1:])),
                )
            ]
            write((row, *rest), region)
