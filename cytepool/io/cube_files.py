"""Cube files: a cube saved as a Zarr v3 group and read as it is used.

A cube file is a directory, by convention named ``<name>.cube.zarr``, holding a Zarr v3
group, as the Zarr core specification (Miles et al. 2025) defines it, that any Zarr v3
reader can open. ``write_cube`` saves a cube. ``read_cube`` opens one: it reads the
metadata and the sample, group and feature tables, and leaves the layers on disk as
``ZarrArray`` objects that the cube reads one group at a time, and only the chunks
that hold the features it needs, so a cube far larger than memory can be opened and
analysed group by group, or for a few features of every group.

Format ``cytepool.cube``, version 1::

    <name>.cube.zarr/
      zarr.json      the root group; its attributes are listed below
      samples/       a group with one 1-D array per sample column
      groups/        a group with one 1-D array per group column
      features/      a group with one 1-D array per feature column
      n_cells        integers (n_samples, n_groups); absent from bulk cubes
      layers/        a group with one array per layer
        counts       (n_groups, n_samples, n_features)
        ...          detected, log_sum, log_sum_sq and linear_sum, when present

Root attributes (others are ignored, so a publisher may add, say, a title):

- ``format``: ``"cytepool.cube"``.
- ``format_version``: ``1``. A reader rejects other versions; any change to the layout
  increments the version.
- ``axes``: ``["groups", "samples", "features"]``, the axes of every layer, in order.
- ``layers``: the layer names, in the order ``LAYER_NAMES`` gives them.
- ``has_n_cells``: whether the ``n_cells`` array is present (false for bulk cubes,
  where cell numbers are unknown).
- ``normalization``: the cube's normalization mapping, or null.
- ``provenance``: the cube's provenance mapping (``cytepool.cube.provenance``).
- ``digests``: the digests of the cube's tables, cell numbers, normalization and of
  each group of each layer, and the fingerprint they make, as ``Cube.digests()``
  returns them (``cytepool.cube.fingerprints`` documents them). ``write_cube``
  computes them while it writes the layers. A file written before CytePool stored
  digests has none; it is read all the same, and its fingerprint is computed from
  its values when it is asked for.

Tables. Each table group holds one 1-D array per column, with ``dimension_names`` set
to the table's axis (``["samples"]``, ``["groups"]`` or ``["features"]``). Its
attributes give the columns in table order: ``columns`` lists the column names and
``arrays`` the name of each column's array. An array takes its column's name when that
name is portable: ASCII letters, digits, ``_``, ``.`` and ``-``, at most 64
characters, not starting with ``.``, ``-`` or ``__``, not ending with ``.``, not a
Windows device name (``con``, ``nul``, ``com1``, ...), not ``zarr.json`` in any case,
not of the form ``column_<n>``, and not equal to another column's name when case is
ignored.
Otherwise the array is named ``column_<position>``, counting from 0. Columns are stored
as:

- text: int32 codes with the attribute ``levels``, the distinct values in order of
  first appearance; code ``i`` stands for ``levels[i]`` and ``-1`` for a missing value;
- datetimes and timedeltas: int64 counts of their unit, with the attribute
  ``numpy_dtype`` (for example ``"datetime64[s]"``); NaT is ``-2**63``, as in numpy;
- booleans and numbers: their own data type.

Layers have ``dimension_names`` ``["groups", "samples", "features"]`` and chunks of one
group, every sample and up to 4,096 features, so reading one group reads only its own
chunks, and reading some features of it only the chunks that hold them. (A chunk holds
fewer samples only where it would otherwise exceed 256 MiB, ``LAYER_CHUNK_BYTES``:
beyond 8,192 samples of float64 values.) ``n_cells`` is int32
(int64 if a value does not fit), with ``dimension_names`` ``["samples", "groups"]``.

Every array has the default chunk key encoding with ``/``, a fill value of zero
(``false`` for bool) and the ``bytes`` codec, little endian, followed by ``gzip`` (the
default, level 4), ``zstd`` (level 3, with checksums; Python 3.14 or newer) or no
compression: the codecs of the Zarr specifications (Zarr developers 2023;
Maitin-Shepard 2025), which store gzip streams (RFC 1952; Deutsch 1996) and Zstandard
frames (RFC 8878; Collet and Kucherawy 2021). Chunks that hold only zeros are not
stored, as Zarr allows, and read as zeros.

Reading is strict: the format name and version, every attribute above, every node's
metadata (see ``cytepool.io.zarr``), dimension names, data types and text codes are
checked, and so are the values of every layer group when the cube reads it. The
tables, cell numbers and normalization are checked against their stored digests when
the file is opened, and each group of a layer whenever a read covers all of it, in
any order, so a file changed after it was written is reported, not read as if it
were the cube it held; ``Cube.verify_fingerprint()`` checks every group, also the
groups that a selection of some samples or features reads in part. Digests describe values, not their
encoding: a copy with other chunks, byte order or compression keeps them. A store
replaced while ``read_cube`` reads it raises an error instead of mixing the tables of
one cube with the layers of another, and so does a lazy layer whose file was replaced
after it was opened.

Writing is atomic: ``write_cube`` writes the store into a new hidden directory next to
the target, ``.<name>.partial-<random>``, writes the root ``zarr.json`` last, and
renames the directory to the target only when everything is written. A write that
fails or is interrupted removes the partial directory and leaves nothing at the
target. With ``overwrite=True``, which replaces only a cube file or an empty
directory, the old store is renamed aside to ``.<name>.replaced-<random>``, the new one
renamed into place and the old one deleted; if the new store cannot be moved into
place, the old one is put back. A process killed outright can leave either hidden
directory behind (between the two renames, the old store survives only under its
hidden name); a partial directory has no root ``zarr.json``, so it does not open as a
group. Files are not flushed to disk with fsync, so the guarantee covers failures of
the writing process, not of the operating system.

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
    level: int | None = None,
    overwrite: bool = False,
) -> None:
    """Save a cube as a cube file, a Zarr v3 group (see the module documentation).

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
        level: The compression level: 0 to 9 for gzip (default 4), -131072 to 22 for
            zstd (default 3); None without compression.
        overwrite: Replace what is at ``path``, the whole directory, if it is a cube
            file (a Zarr group whose ``format`` is ``"cytepool.cube"``) or an empty
            directory. Nothing else is ever replaced, nor the working directory or
            one that holds it.

    Raises:
        TypeError: If ``cube`` is not a Cube, or an argument has the wrong type.
        ValueError: If ``compression`` or ``level`` is invalid, ``path`` is a URL,
            or a lazy layer gives a value that breaks a constraint.
        OverflowError: If a sum of a derived lazy layer does not fit (see "Sums" in
            ``Cube``).
        FileExistsError: If ``path`` exists and ``overwrite`` is False, or it is not
            something ``overwrite`` replaces.
        FileNotFoundError: If the parent directory of ``path`` does not exist.
    """
    if not isinstance(cube, Cube):
        raise TypeError(f"write_cube() saves a Cube, got {type(cube).__name__}")
    if not isinstance(overwrite, bool):
        raise TypeError(f"overwrite must be True or False, got {overwrite!r}")
    level = check_compression(compression, level)
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
            level=level,
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
            level=level,
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
                level=level,
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
