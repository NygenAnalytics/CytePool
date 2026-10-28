"""Cube files: a cube saved as a Zarr v3 group and read as it is used.

``write_cube`` saves a cube as a cube file: a directory, by convention named
``<name>.cube.zarr``, holding a Zarr v3 group, as the Zarr core specification (Miles et
al. 2025) defines it, that any Zarr v3 reader can open. ``read_cube`` opens one: it
reads the metadata and the sample, group and feature tables, and leaves the layers on
disk as ``ZarrArray`` objects that the cube reads one group at a time, and only the
chunks that hold the features it needs, so a cube far larger than memory can be opened
and analysed group by group, or for a few features of every group. Reading checks what
it reads against the digests stored with the file, and writing is atomic.

The format, ``cytepool.cube`` version 1, is specified, for programs that read or write
cube files, at https://docs.nygen.io/CytePool/reference/cube-files#cube-file-format.
Its arrays use the ``bytes`` and ``gzip`` codecs of the Zarr specifications (Zarr
developers 2023), or the ``zstd`` codec of its extensions (Maitin-Shepard 2025), which
store gzip streams (RFC 1952; Deutsch 1996) and Zstandard frames (RFC 8878; Collet and
Kucherawy 2021).

References:
    Miles A, Striebel J, Rzepka N, Maitin-Shepard J, Moore J, eds (2025). Zarr core
    specification, version 3.1. https://zarr-specs.readthedocs.io/en/latest/v3/core/

    Zarr developers (2023). Bytes codec, version 1.0, and gzip codec, version 1.0. Zarr
    specifications, https://zarr-specs.readthedocs.io/en/latest/v3/codecs/

    Maitin-Shepard J (2025). Zstd codec. Zarr extensions,
    https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/zstd

    Deutsch P (1996). GZIP file format specification version 4.3. RFC 1952, RFC Editor.
    doi:10.17487/RFC1952

    Collet Y, Kucherawy M (2021). Zstandard Compression and the 'application/zstd' Media
    Type. RFC 8878, RFC Editor. doi:10.17487/RFC8878
"""

import os
import re
import secrets
import shutil
import warnings
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..cube.cube import (
    LAYER_NAMES,
    Cube,
    _digest_of_read_group,
    _digest_record_of,
    _take_stored_digests,
)
from ..table import Table
from ._files import file_stamp, local_path, same_file
from .zarr import (
    METADATA_FILE,
    ZarrArray,
    check_compression,
    open_array,
    read_group_attributes,
    write_array,
    write_group,
)

FORMAT_NAME = "cytepool.cube"
"""The ``format`` attribute of a cube file's root group."""

FORMAT_VERSION = 1
"""The cube file format version this module writes and reads."""

AXES = ("groups", "samples", "features")
"""The axes of every layer, in order."""

FEATURES_PER_CHUNK = 4096
"""The most features a layer chunk holds."""

LAYER_CHUNK_BYTES = 2**28
"""The most bytes a layer chunk holds; only beyond it does a chunk split the samples."""

_TABLE_AXES = ("samples", "groups", "features")
_ROOT_ATTRIBUTES = ("axes", "layers", "has_n_cells", "normalization", "provenance")
_MAX_CHUNK_ELEMENTS = 2**20  # for table columns and n_cells
_PORTABLE_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,63}")
_POSITIONAL_NAME = re.compile(r"column_[0-9]+", re.IGNORECASE)
_WINDOWS_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"{prefix}{digit}" for prefix in ("COM", "LPT") for digit in range(10)}
)


def write_cube(
    cube: Cube,
    path: str | os.PathLike[str],
    *,
    compression: str | None = "gzip",
    compression_level: int | None = None,
    overwrite: bool = False,
) -> None:
    """Save a cube as a cube file, a Zarr v3 group.

    https://docs.nygen.io/CytePool/reference/cube-files#cube-file-format specifies
    the file.

    The store appears at ``path`` only once it is complete, so a write that fails
    leaves no store there; with ``overwrite=True`` it leaves the old store as it was.
    Lazy layers are read, and their values checked, one group at a time. Each group
    is hashed as it is written, and the digests are stored with the file, so the
    fingerprint of the saved cube needs no further pass over its values.

    Args:
        cube: The cube to save.
        path: The directory to create, by convention ending in ``.cube.zarr``. Its
            parent must exist.
        compression: ``"gzip"`` (the default), ``"zstd"`` (Python 3.14 or newer) or
            None for no compression.
        compression_level: The compression level: 0 to 9 for gzip (default 4),
            -131072 to 22 for zstd (default 3); None without compression.
        overwrite: Replace what is at ``path``, the whole directory, if it is a cube
            file (a Zarr group whose ``format`` is ``"cytepool.cube"``) or an empty
            directory. Nothing else is ever replaced, nor the working directory or
            one that holds it.

    Raises:
        TypeError: If ``cube`` is not a Cube, or an argument has the wrong type.
        ValueError: If ``compression`` or ``compression_level`` is invalid, ``path``
            is a URL, or a lazy layer gives a value that breaks a constraint.
        OverflowError: If a sum of a derived lazy layer does not fit (see "Sums" in
            ``Cube``).
        FileExistsError: If ``path`` exists and ``overwrite`` is False, or it is not
            something ``overwrite`` replaces.
        FileNotFoundError: If the parent directory of ``path`` does not exist.
    """
    if not isinstance(cube, Cube):
        raise TypeError(f"write_cube() saves a Cube, got {type(cube).__name__}")
    if not isinstance(overwrite, bool):
        raise TypeError(
            f"overwrite must be True or False, got {type(overwrite).__name__}"
        )
    level = check_compression(compression, compression_level)
    target = local_path(path)
    if not target.parent.is_dir():
        raise FileNotFoundError(f"the directory {target.parent} does not exist")
    _check_target(target, overwrite)
    while True:
        staging = target.with_name(f".{target.name}.partial-{secrets.token_hex(4)}")
        try:
            staging.mkdir()
            break
        except FileExistsError:
            continue
    try:
        _write_store(cube, staging, compression, level)
        _move_into_place(staging, target, overwrite)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def read_cube(path: str | os.PathLike[str]) -> Cube:
    """Open a cube file.

    The metadata and the sample, group and feature tables are read now. The layers
    stay on disk: when a method needs a group of a layer, the cube reads the chunks
    that hold the samples and features it needs, and checks those values then (see
    "Lazy layers" in ``Cube``). Opening a cube therefore takes little memory,
    whatever its size, and a cube derived with ``select()`` reads only the chunks that
    hold the selected features of the selected groups.

    The digests stored with the file give the cube's fingerprint without reading
    the layers. The tables, cell numbers and normalization are checked against
    theirs now, and each group of a layer whenever a read covers all of it.

    Args:
        path: The cube file's directory.

    Returns:
        A cube whose layers are ``ZarrArray`` objects.

    Raises:
        FileNotFoundError: If there is no store at ``path``.
        ValueError: If the store is not a cube file, has another format version, is
            corrupt, uses a Zarr feature this reader does not support, or holds
            tables, cell numbers or a normalization that do not match its stored
            digests.
        TypeError: If a stored value has a type the cube does not accept.
        RuntimeError: If the store was replaced while it was being read.
    """
    root = local_path(path)
    try:
        before = file_stamp(root / METADATA_FILE)
    except OSError:
        before = None  # read_group_attributes says what is wrong
    attributes = read_group_attributes(root)
    _check_root_attributes(attributes, root)
    samples, groups, features = (_read_table(root / axis, axis) for axis in _TABLE_AXES)
    n_cells = None
    if attributes["has_n_cells"]:
        cells = open_array(root / "n_cells")
        _check_dimensions(cells, ("samples", "groups"))
        if cells.dtype.kind not in "iu":
            raise ValueError(
                f"{cells.path}: n_cells must hold integers, not {cells.dtype}"
            )
        n_cells = np.asarray(cells)
    elif os.path.lexists(root / "n_cells"):
        raise ValueError(
            f"{root} has an n_cells array, but its attributes say has_n_cells is false"
        )
    read_group_attributes(root / "layers")
    layers = {}
    for name in attributes["layers"]:
        array = open_array(root / "layers" / name)
        _check_dimensions(array, AXES)
        layers[name] = array
    try:
        cube = Cube(
            layers,
            samples=samples,
            groups=groups,
            features=features,
            n_cells=n_cells,
            normalization=attributes["normalization"],
            provenance=attributes["provenance"],
        )
    except (TypeError, ValueError) as error:
        error.add_note(f"(while reading the cube file {root})")
        raise
    if "digests" in attributes:
        where = f"the 'digests' attribute of {root / METADATA_FILE}"
        _take_stored_digests(cube, attributes["digests"], where)
    # A store replaced while it was read could pair one cube's tables with another's
    # layers; the root metadata of a store written anew is a new file.
    try:
        after = file_stamp(root / METADATA_FILE)
    except OSError:
        after = None
    if before is None or after is None or not same_file(before, after):
        raise RuntimeError(
            f"the cube file at {root} was replaced or removed while it was being "
            "read; open it again"
        )
    return cube


# Writing.


def _check_target(target: Path, overwrite: bool) -> None:
    """Raise unless ``target`` is absent, or ``overwrite`` may replace it."""
    if not os.path.lexists(target):
        return
    if not overwrite:
        raise FileExistsError(
            f"{target} already exists; pass overwrite=True to replace it"
        )
    unchanged = (
        "overwrite=True replaces only a cube file or an empty directory, so nothing "
        "was changed"
    )
    if target.is_symlink() or not target.is_dir():
        kind = "a symbolic link" if target.is_symlink() else "a file"
        raise FileExistsError(f"{target} is {kind}; {unchanged}")
    try:
        working = Path(os.path.realpath(os.getcwd()))
    except FileNotFoundError:  # the working directory was deleted
        working = None
    real = Path(os.path.realpath(target))
    if working is not None and (real == working or real in working.parents):
        raise FileExistsError(
            f"{target} is the working directory or holds it; {unchanged}"
        )
    if any(target.iterdir()) and not _is_cube_file(target):
        raise FileExistsError(f"{target} is not a cube file; {unchanged}")


def _is_cube_file(path: Path) -> bool:
    """Return whether ``path`` holds a Zarr group whose format is a cube file's."""
    try:
        return read_group_attributes(path).get("format") == FORMAT_NAME
    except (OSError, ValueError):
        return False


def _move_into_place(staging: Path, target: Path, overwrite: bool) -> None:
    if not (overwrite and os.path.lexists(target)):
        try:
            os.rename(staging, target)
        except OSError as error:
            if os.path.lexists(target):
                raise FileExistsError(
                    f"{target} appeared while the cube was being written; nothing "
                    "was replaced"
                ) from error
            raise
        return
    _check_target(target, overwrite=True)  # it may have changed during the write
    backup = target.with_name(f".{target.name}.replaced-{secrets.token_hex(4)}")
    os.rename(target, backup)
    try:
        os.rename(staging, target)
    except BaseException as error:
        try:
            os.rename(backup, target)
        except OSError:
            error.add_note(f"(the previous store was left at {backup})")
        raise
    try:
        shutil.rmtree(backup)
    except OSError as error:
        warnings.warn(
            f"the cube was written to {target}, but the store it replaced could not "
            f"be removed from {backup}: {error}",
            RuntimeWarning,
            stacklevel=3,
        )


def _write_store(
    cube: Cube, root: Path, compression: str | None, level: int | None
) -> None:
    for axis, table in zip(
        _TABLE_AXES, (cube.samples, cube.groups, cube.features), strict=True
    ):
        _write_table(root / axis, table, axis, compression, level)
    n_cells = cube.n_cells
    if n_cells is not None:
        if n_cells.size == 0 or n_cells.max() <= np.iinfo(np.int32).max:
            n_cells = n_cells.astype(np.int32)
        rows = min(cube.n_samples, _MAX_CHUNK_ELEMENTS // max(cube.n_groups, 1))
        write_array(
            root / "n_cells",
            n_cells,
            chunks=(max(rows, 1), max(cube.n_groups, 1)),
            compression=compression,
            compression_level=level,
            dimension_names=["samples", "groups"],
        )
    write_group(root / "layers")
    digests = {}
    for name in cube.layer_names:
        groups = _GroupsOfLayer(cube, name)
        write_array(
            root / "layers" / name,
            groups,
            chunks=_layer_chunks(
                cube.n_samples, cube.n_features, groups.dtype.itemsize
            ),
            compression=compression,
            compression_level=level,
            dimension_names=list(AXES),
        )
        digests[name] = groups.digests()
    # The root metadata comes last: until it exists, the directory is not a group.
    write_group(
        root,
        {
            "format": FORMAT_NAME,
            "format_version": FORMAT_VERSION,
            "axes": list(AXES),
            "layers": list(cube.layer_names),
            "has_n_cells": cube.n_cells is not None,
            "normalization": cube.normalization,
            "provenance": cube.provenance,
            "digests": _digest_record_of(cube, digests),
        },
    )


def _write_table(
    directory: Path, table: Table, axis: str, compression: str | None, level: int | None
) -> None:
    names = list(table.columns)
    arrays = _array_names(names)
    write_group(directory, {"columns": names, "arrays": arrays})
    chunks = (max(min(len(table), _MAX_CHUNK_ELEMENTS), 1),)
    for name, array in zip(names, arrays, strict=True):
        column = table[name]
        attributes: dict[str, Any] = {}
        if column.dtype.kind == "O":  # text: str or None
            levels: dict[str, int] = {}
            codes = [
                -1 if value is None else levels.setdefault(value, len(levels))
                for value in column.tolist()
            ]
            column = np.array(codes, dtype=np.int32)
            attributes["levels"] = list(levels)
        elif column.dtype.kind in "mM":
            attributes["numpy_dtype"] = str(column.dtype)
            column = column.view(np.int64)
        try:
            write_array(
                directory / array,
                column,
                chunks=chunks,
                compression=compression,
                compression_level=level,
                attributes=attributes,
                dimension_names=[axis],
            )
        except (TypeError, ValueError) as error:
            error.add_note(f"(while writing the {axis} column {name!r})")
            raise


def _layer_chunks(
    n_samples: int, n_features: int, itemsize: int
) -> tuple[int, int, int]:
    """Return chunks of one group, every sample and up to ``FEATURES_PER_CHUNK``
    features, with fewer samples only where a chunk would exceed ``LAYER_CHUNK_BYTES``.
    """
    features = max(min(n_features, FEATURES_PER_CHUNK), 1)
    samples = min(max(n_samples, 1), LAYER_CHUNK_BYTES // (features * itemsize))
    return 1, samples, features


def _array_names(columns: Sequence[str]) -> list[str]:
    """Name each column's array: the column name if portable, else by position."""
    portable = [_is_portable(name) for name in columns]
    folded = Counter(
        name.lower() for name, ok in zip(columns, portable, strict=True) if ok
    )
    return [
        name if ok and folded[name.lower()] == 1 else f"column_{position}"
        for position, (name, ok) in enumerate(zip(columns, portable, strict=True))
    ]


def _is_portable(name: str) -> bool:
    return (
        _PORTABLE_NAME.fullmatch(name) is not None
        and not name.startswith("__")
        and not name.endswith(".")
        and name.lower() != METADATA_FILE
        and name.split(".")[0].upper() not in _WINDOWS_DEVICE_NAMES
        and _POSITIONAL_NAME.fullmatch(name) is None
    )


class _GroupsOfLayer:
    """One layer of a cube as ``write_array`` reads it: rows are groups.

    It keeps the digest of each group it gives, for the cube file's digests.
    """

    __slots__ = ("_cube", "_name", "_first", "_digests", "shape", "dtype")

    def __init__(self, cube: Cube, name: str) -> None:
        self._cube = cube
        self._name = name
        self.shape = (cube.n_groups, cube.n_samples, cube.n_features)
        self._digests: list[str | None] = [None] * cube.n_groups
        # Group 0 gives the dtype; a cube with an empty axis has no values to read.
        self._first = None if 0 in self.shape else cube.layer(name, 0)
        self.dtype = (cube.layer(name) if self._first is None else self._first).dtype

    def __getitem__(self, rows: slice) -> np.ndarray:
        blocks = []
        for group in range(*rows.indices(self.shape[0])):
            if group == 0 and self._first is not None:
                block, self._first = self._first, None
            else:
                block = self._cube.layer(self._name, group)
            self._digests[group] = _digest_of_read_group(
                self._cube, self._name, group, block
            )
            blocks.append(block)
        if len(blocks) == 1:
            return blocks[0][np.newaxis]
        return (
            np.stack(blocks) if blocks else np.empty((0, *self.shape[1:]), self.dtype)
        )

    def digests(self) -> list[str]:
        """Return the digest of each group, hashing those never read (empty ones)."""
        return [
            _digest_of_read_group(
                self._cube, self._name, group, self._cube.layer(self._name, group)
            )
            if digest is None
            else digest
            for group, digest in enumerate(self._digests)
        ]


# Reading.


def _check_root_attributes(attributes: dict[str, Any], root: Path) -> None:
    where = root / METADATA_FILE
    name = attributes.get("format")
    if name != FORMAT_NAME:
        if name is None:
            raise ValueError(
                f"{root} is a Zarr group but not a cube file: the attributes in its "
                f"{METADATA_FILE} have no 'format' (a cube file has 'format': "
                f"{FORMAT_NAME!r})"
            )
        raise ValueError(f"{root} holds the format {name!r}, not {FORMAT_NAME!r}")
    version = attributes.get("format_version")
    if type(version) is not int:
        raise ValueError(f"{where}: format_version must be an integer, got {version!r}")
    if version != FORMAT_VERSION:
        advice = "; update CytePool to read it" if version > FORMAT_VERSION else ""
        raise ValueError(
            f"{root} was written in cube format version {version}, but this version "
            f"of CytePool reads version {FORMAT_VERSION}{advice}"
        )
    for key in _ROOT_ATTRIBUTES:
        if key not in attributes:
            raise ValueError(f"{where} lacks the cube attribute {key!r}")
    if attributes["axes"] != list(AXES):
        raise ValueError(
            f"{where}: axes must be {list(AXES)}, got {attributes['axes']!r}"
        )
    layers = attributes["layers"]
    if (
        not isinstance(layers, list)
        or not all(isinstance(layer, str) and layer in LAYER_NAMES for layer in layers)
        or len(set(layers)) != len(layers)
    ):
        raise ValueError(
            f"{where}: layers must list distinct names among {', '.join(LAYER_NAMES)}, "
            f"got {layers!r}"
        )
    if type(attributes["has_n_cells"]) is not bool:
        raise ValueError(f"{where}: has_n_cells must be true or false")
    if not isinstance(attributes["normalization"], dict | None):
        raise ValueError(f"{where}: normalization must be an object or null")
    if not isinstance(attributes["provenance"], dict):
        raise ValueError(f"{where}: provenance must be an object")


def _read_table(directory: Path, axis: str) -> Table:
    attributes = read_group_attributes(directory)
    where = directory / METADATA_FILE
    columns, arrays = attributes.get("columns"), attributes.get("arrays")
    if (
        not isinstance(columns, list)
        or not isinstance(arrays, list)
        or len(columns) != len(arrays)
        or not all(isinstance(item, str) for item in columns + arrays)
    ):
        raise ValueError(
            f"{where}: the attributes 'columns' and 'arrays' must be lists of strings "
            "of the same length"
        )
    for listed, label in ((columns, "column"), (arrays, "array")):
        repeated = [item for item, count in Counter(listed).items() if count > 1]
        if repeated:
            raise ValueError(f"{where} lists the {label} {repeated[0]!r} twice")
    data = {}
    for name, node in zip(columns, arrays, strict=True):
        if (
            not node
            or set(node) == {"."}
            or node.startswith("__")
            or node == METADATA_FILE
            or any(character in node for character in "/\\\x00")
            or (directory / node).parent != directory  # a drive such as D: on Windows
        ):
            raise ValueError(f"{where}: {node!r} is not a valid array name")
        array = open_array(directory / node)
        _check_dimensions(array, (axis,))
        data[name] = _decode_column(array)
    try:
        return Table(data)
    except (TypeError, ValueError) as error:
        error.add_note(f"(while reading {directory})")
        raise


def _decode_column(array: ZarrArray) -> np.ndarray:
    """Read a table column, checking its metadata before its values."""
    where = array.path / METADATA_FILE
    attributes = array.attributes
    levels = attributes.get("levels")
    numpy_dtype = attributes.get("numpy_dtype")
    if "levels" in attributes and "numpy_dtype" in attributes:
        raise ValueError(f"{where} has both 'levels' and 'numpy_dtype'")
    if "levels" in attributes:
        if array.dtype != np.int32:
            raise ValueError(f"{where}: text codes must be int32, not {array.dtype}")
        if not isinstance(levels, list) or not all(
            isinstance(level, str) for level in levels
        ):
            raise ValueError(f"{where}: levels must be a list of strings")
        if len(set(levels)) != len(levels):
            raise ValueError(f"{where}: levels must not repeat a value")
        codes = np.asarray(array)
        bad = (codes < -1) | (codes >= len(levels))
        if bad.any():
            raise ValueError(
                f"{where} has the code {codes[bad][0]}, but codes run from -1 "
                f"(missing) to {len(levels) - 1}"
            )
        lookup = np.empty(len(levels) + 1, dtype=object)
        lookup[:-1] = levels
        text: np.ndarray = lookup[codes]  # the code -1 takes the trailing None
        return text
    if "numpy_dtype" in attributes:
        if array.dtype != np.int64:
            raise ValueError(
                f"{where}: times must be stored as int64, not {array.dtype}"
            )
        try:
            dtype = np.dtype(numpy_dtype) if isinstance(numpy_dtype, str) else None
        except (TypeError, ValueError):
            dtype = None
        if dtype is None or dtype.kind not in "mM" or str(dtype) != numpy_dtype:
            raise ValueError(
                f"{where}: numpy_dtype must name a datetime64 or timedelta64 dtype, "
                f"such as 'datetime64[s]', got {numpy_dtype!r}"
            )
        return np.asarray(array).view(dtype)
    return np.asarray(array)


def _check_dimensions(array: ZarrArray, expected: tuple[str, ...]) -> None:
    if array.dimension_names != expected:
        raise ValueError(
            f"{array.path / METADATA_FILE}: dimension_names must be {list(expected)}, "
            f"got {array.dimension_names!r}"
        )
