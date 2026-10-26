"""Content fingerprints of cubes: what they cover and how they are encoded.

A cube's fingerprint identifies the values an analysis reads: the sample, group and
feature tables (ids included), the cells pooled into each entry (``n_cells``), the
normalization that defines the normalized layers, and every layer. It leaves out the
provenance, which records how the cube was made, so two cubes with the same values
have the same fingerprint however they were made, and a result can name the cube it
was computed from by its fingerprint. ``Cube.fingerprint()`` returns it and
``Cube.digests()`` the digests it is built from.

Parts and digests:
    The fingerprint is built from the digests of the cube's parts: each table, the
    cell numbers, the normalization, and each group of each layer. ``write_cube``
    computes the digests while it writes the cube and stores them with the cube file,
    so the fingerprint of a saved cube needs no pass over its data. ``read_cube``
    checks the tables, cell numbers and normalization against their digests when it
    opens the file, and every read that covers a whole group of a layer, in any
    order, against the group's digest. One group can thus be checked without
    reading the others, and an analysis that reads whole groups checks exactly what
    it read. (A read of some samples or features of a group cannot be checked on
    its own; ``Cube.verify_fingerprint()`` reads such groups whole.) A single digest
    of the whole cube would need every layer read to check any of it.

Encoding, version 1 (``FINGERPRINT_VERSION``):
    A digest is the SHA-256 hash (National Institute of Standards and Technology
    2015) of a byte string, written as ``sha256:`` and 64 lowercase hexadecimal
    digits. Numbers in byte strings are little endian: ``u64`` and ``i64`` are
    unsigned and signed 8-byte integers. Text is UTF-8 (Yergeau 2003), written as
    its length in bytes (``u64``) followed by its bytes; a lone surrogate, which a
    Python string can hold but a cube file cannot, is written in its 3-byte form, as
    Python's ``surrogatepass`` error handler writes it. Strings are taken as they
    are, without Unicode normalization.

    Tables (samples, groups and features): the number of rows and of columns
    (``u64`` each), then for each column, in table order, its name (text), its type
    (text) and its values:

    - text, type ``text``: the length in bytes of each value (``i64``, -1 for a
      missing value), then the bytes of the values present, one after another;
    - ``bool``: one byte per value, 0 or 1;
    - ``int8`` to ``int64`` and ``uint8`` to ``uint64``: the values in that type;
    - ``float16``, ``float32`` and ``float64``: the values in that type, the binary
      formats of IEEE 754 (IEEE 2019), with -0.0 written as 0.0 and every NaN as the
      quiet NaN (``0x7E00``, ``0x7FC00000`` or ``0x7FF8000000000000``);
    - ``complex64`` and ``complex128``: the real and imaginary parts of each value,
      as floats of half the size written as above; a value with a NaN part is
      missing, and both its parts are written as the quiet NaN;
    - datetimes and timedeltas, of type ``datetime64[unit]`` or ``timedelta64[unit]``
      (numpy's name, such as ``datetime64[ms]``): the values as ``i64``, NaT being
      -2**63.

    The type is numpy's name for the column's dtype. These are the column types
    cube files store, and the only ones a cube's tables accept. Two tables have the
    same digest exactly when ``Table.equals`` finds them equal: the same names in the
    same order, the same types, and equal values, with missing values equal.

    Cell numbers: the number of samples and of groups (``u64`` each), then
    ``n_cells`` as ``i64``, sample by sample.

    Normalization: the mapping as a JSON value (Bray 2017), each value written as a
    tag byte and a payload: ``n`` for null; ``f`` and ``t`` for false and true;
    ``i`` and the decimal digits of the number as text (with a leading ``-`` when
    negative) for a number with a whole value, such as 10000 or 10000.0; ``d`` and
    its 8 bytes as a float64 for any other number; ``s`` and the text for a string;
    ``a``, the number of items (``u64``) and the items for an array; ``o``, the
    number of members (``u64``) and, in the order of the UTF-8 bytes of their keys,
    each key (text) and value for an object. A number is thus taken by its value,
    however a JSON writer spells it.

    Layers: one digest for each group of each layer, of the group's samples x
    features block: the number of samples and of features (``u64`` each), then the
    values sample by sample, as ``i64`` for an integer layer and as float64, with
    -0.0 written as 0.0, for a float layer. An integer layer holds the same numbers
    whatever the width of its dtype (``Cube`` accepts integers from 0 to 2**63 - 1,
    and merging or pooling makes them int64), so a uint32 group and an int64 group
    with the same values have the same digest.

    Manifest: UTF-8 text with one line per part, each ending with a line feed
    (``\\n``)::

        cytepool cube fingerprint 1
        shape <groups> <samples> <features>
        samples <digest>
        groups <digest>
        features <digest>
        n_cells <digest, or none without cell numbers>
        normalization <digest, or none without normalization>
        layer <name> <int64 or float64> <group position> <digest>

    with a ``layer`` line for each group of each layer: layers in the order of
    ``LAYER_NAMES`` (counts, detected, log_sum, log_sum_sq, linear_sum), and groups in
    the cube's order, counted from 0. The fingerprint is the digest of the manifest.

Digest records:
    ``Cube.digests()`` returns, and a cube file stores as the root attribute
    ``digests``, a JSON object with everything the manifest lists (the order of its
    members does not matter)::

        {"version": 1, "shape": [<groups>, <samples>, <features>],
         "samples": <digest>, "groups": <digest>, "features": <digest>,
         "n_cells": <digest or null>, "normalization": <digest or null>,
         "layers": {<name>: {"type": "int64" or "float64",
                             "groups": [<digest of each group>]}},
         "fingerprint": <digest of the manifest>}

References:
    National Institute of Standards and Technology (2015). Secure Hash Standard (SHS).
    Federal Information Processing Standards Publication 180-4.
    doi:10.6028/NIST.FIPS.180-4

    Yergeau F (2003). UTF-8, a transformation format of ISO 10646. RFC 3629, RFC Editor.
    doi:10.17487/RFC3629

    Bray T (2017). The JavaScript Object Notation (JSON) Data Interchange Format.
    RFC 8259, RFC Editor. doi:10.17487/RFC8259

    IEEE (2019). IEEE Standard for Floating-Point Arithmetic. IEEE Std 754-2019.
    doi:10.1109/IEEESTD.2019.8766229
"""

import hashlib
import re
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..table import Table

__all__ = ["FINGERPRINT_VERSION"]

FINGERPRINT_VERSION = 1
"""The version of the encoding this module documents and computes."""

_PREFIX = "sha256:"
_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
_HEADER = f"cytepool cube fingerprint {FINGERPRINT_VERSION}"
_PARTS = ("samples", "groups", "features", "n_cells", "normalization")
# The order of layers in a manifest: that of LAYER_NAMES in cytepool.cube.cube.
_LAYER_ORDER = ("counts", "detected", "log_sum", "log_sum_sq", "linear_sum")
_RECORD_KEYS = frozenset({"version", "shape", "layers", "fingerprint", *_PARTS})
_PIECE = 1 << 16  # values converted and hashed at a time, which bounds the memory used
# For each float size: the unsigned type of its bits, the bits of -0.0, the
# exponent and mantissa masks, and the quiet NaN.
_FLOAT_BITS = {
    2: (np.uint16, 0x8000, 0x7C00, 0x03FF, 0x7E00),
    4: (np.uint32, 0x8000_0000, 0x7F80_0000, 0x007F_FFFF, 0x7FC0_0000),
    8: (
        np.uint64,
        0x8000_0000_0000_0000,
        0x7FF0_0000_0000_0000,
        0x000F_FFFF_FFFF_FFFF,
        0x7FF8_0000_0000_0000,
    ),
}
_FLOAT_TYPES = frozenset({"float16", "float32", "float64", "complex64", "complex128"})


def _table_digest(table: Table) -> str:
    """Return the digest of a table, as the module documentation describes."""
    hasher = hashlib.sha256(_u64(len(table)) + _u64(len(table.columns)))
    for name in table.columns:
        column = table[name]
        hasher.update(_text(name))
        if column.dtype.kind == "O":
            hasher.update(_text("text"))
            encoded = [
                None if value is None else value.encode("utf-8", "surrogatepass")
                for value in column.tolist()
            ]
            lengths = [-1 if value is None else len(value) for value in encoded]
            hasher.update(np.array(lengths, dtype="<i8").tobytes())
            hasher.update(b"".join(value for value in encoded if value is not None))
            continue
        hasher.update(_text(_column_type(column.dtype, f"column {name!r}")))
        for start in range(0, column.size, _PIECE):
            hasher.update(_column_bytes(column[start : start + _PIECE]))
    return _PREFIX + hasher.hexdigest()


def _column_type(dtype: np.dtype, where: str) -> str:
    """Return the type a table column is encoded as, checking that it has one.

    Args:
        dtype: The column's dtype, not that of a text column.
        where: Names the column in messages.

    Raises:
        TypeError: If the dtype is not one that cube files store, such as
            longdouble, whose bytes differ between platforms.
    """
    if dtype.kind in "biumM" or dtype.name in _FLOAT_TYPES:
        return dtype.name
    raise TypeError(
        f"{where} has dtype {dtype}, which a cube's tables do not hold: they hold "
        "text, booleans, integers, float16 to float64, complex64, complex128, "
        "datetimes and timedeltas, which are stored alike on every platform; "
        "convert it, for example to float64"
    )


def _cells_digest(n_cells: np.ndarray | None) -> str | None:
    """Return the digest of the cell numbers, or None for a cube without them."""
    if n_cells is None:
        return None
    rows, columns = n_cells.shape
    hasher = hashlib.sha256(_u64(rows) + _u64(columns))
    hasher.update(np.ascontiguousarray(n_cells, dtype="<i8"))
    return _PREFIX + hasher.hexdigest()


def _normalization_digest(normalization: Any) -> str | None:
    """Return the digest of a normalization mapping (plain JSON), or None for None."""
    if normalization is None:
        return None
    return _PREFIX + hashlib.sha256(_json_value(normalization)).hexdigest()


def _layer_type(dtype: np.dtype) -> str:
    """Return the type a layer is encoded as: int64 for integers, float64 otherwise."""
    return "int64" if dtype.kind in "iu" else "float64"


def _block_digest(
    block: np.ndarray,
    rows: np.ndarray | None = None,
    columns: np.ndarray | None = None,
) -> str:
    """Return the digest of one group of a layer, a samples x features block.

    Integer values must be from 0 to 2**63 - 1 and float values float64, as a cube's
    layers hold; they are converted a piece at a time, so the memory used besides
    the block stays small. ``block`` may hold the group's samples and features in
    another order: ``rows[i]`` and ``columns[j]`` then give where the group's
    sample ``i`` and feature ``j`` are in ``block`` (None where they are in place),
    and the digest is that of the group in its own order.
    """
    n_rows, n_columns = block.shape
    hasher = hashlib.sha256(_u64(n_rows) + _u64(n_columns))
    integer = block.dtype.kind in "iu"
    step = max(1, _PIECE // max(n_columns, 1))
    for start in range(0, n_rows, step):
        if rows is None:
            piece = block[start : start + step]
        else:
            piece = block[rows[start : start + step]]
        if columns is not None:
            piece = np.take(piece, columns, axis=1)
        if integer:
            hasher.update(np.ascontiguousarray(piece, dtype="<i8"))
        else:
            hasher.update(_float_bits(np.ascontiguousarray(piece), nan=False))
    return _PREFIX + hasher.hexdigest()


def _manifest(record: Mapping[str, Any]) -> str:
    """Return the manifest text of a digest record (without its fingerprint)."""
    lines = [_HEADER, "shape {} {} {}".format(*record["shape"])]
    for part in _PARTS:
        lines.append(f"{part} {record[part] or 'none'}")
    # In a fixed order, as JSON objects have none.
    for name in sorted(record["layers"], key=_LAYER_ORDER.index):
        layer = record["layers"][name]
        for position, digest in enumerate(layer["groups"]):
            lines.append(f"layer {name} {layer['type']} {position} {digest}")
    return "".join(line + "\n" for line in lines)


def _fingerprint_of(record: Mapping[str, Any]) -> str:
    """Return the fingerprint of a digest record: the digest of its manifest."""
    return _PREFIX + hashlib.sha256(_manifest(record).encode("utf-8")).hexdigest()


def _digest_record(
    shape: tuple[int, int, int],
    parts: Mapping[str, str | None],
    layers: Mapping[str, tuple[str, list[str]]],
) -> dict[str, Any]:
    """Return a digest record from the digests of a cube's parts.

    Args:
        shape: The numbers of groups, samples and features.
        parts: The digest of each table, of the cell numbers and of the
            normalization (None when the cube has none).
        layers: For each layer, in the order of ``LAYER_NAMES``, its type and the
            digest of each group.
    """
    record: dict[str, Any] = {"version": FINGERPRINT_VERSION, "shape": list(shape)}
    record |= {part: parts[part] for part in _PARTS}
    record["layers"] = {
        name: {"type": kind, "groups": list(digests)}
        for name, (kind, digests) in layers.items()
    }
    record["fingerprint"] = _fingerprint_of(record)
    return record


def _check_record(
    record: Any,
    shape: tuple[int, int, int],
    layer_types: Mapping[str, str],
    where: str,
) -> None:
    """Check a stored digest record's form against the cube it describes.

    Args:
        record: The record as read from JSON.
        shape: The cube's numbers of groups, samples and features.
        layer_types: The cube's layers, with the type each is encoded as.
        where: Names the record in messages.

    Raises:
        ValueError: If the record is malformed, has another version, describes
            another shape or other layers, or its fingerprint is not the digest of
            its manifest.
    """
    if not isinstance(record, dict) or set(record) != _RECORD_KEYS:
        keys = sorted(record) if isinstance(record, dict) else type(record).__name__
        raise ValueError(
            f"{where} must be an object with the keys {', '.join(sorted(_RECORD_KEYS))}"
            f"; got {keys}"
        )
    version = record["version"]
    if type(version) is not int or version != FINGERPRINT_VERSION:
        advice = (
            "; update CytePool to read it"
            if type(version) is int and version > FINGERPRINT_VERSION
            else ""
        )
        raise ValueError(
            f"{where} has fingerprint version {version!r}, but this version of "
            f"CytePool computes version {FINGERPRINT_VERSION}{advice}"
        )
    if record["shape"] != list(shape) or not all(
        type(size) is int for size in record["shape"]
    ):
        raise ValueError(
            f"{where} describes a cube of shape {record['shape']!r} (groups, samples, "
            f"features), but the cube has shape {list(shape)}"
        )
    for part in _PARTS:
        digest = record[part]
        optional = part in ("n_cells", "normalization")
        if not (_is_digest(digest) or (optional and digest is None)):
            raise ValueError(f"{where}: {part} must be a digest, got {digest!r}")
    layers = record["layers"]
    if not isinstance(layers, dict) or set(layers) != set(layer_types):
        found = list(layers) if isinstance(layers, dict) else layers
        raise ValueError(
            f"{where} lists the layers {found!r}, but the cube has {list(layer_types)}"
        )
    for name, kind in layer_types.items():
        layer = layers[name]
        if (
            not isinstance(layer, dict)
            or set(layer) != {"type", "groups"}
            or layer["type"] != kind
            or not isinstance(layer["groups"], list)
            or len(layer["groups"]) != shape[0]
            or not all(_is_digest(digest) for digest in layer["groups"])
        ):
            raise ValueError(
                f"{where}: layer {name!r} must be an object with the type {kind!r} "
                f"and one digest per group ({shape[0]})"
            )
    if record["fingerprint"] != _fingerprint_of(record):
        raise ValueError(
            f"{where}: the fingerprint {record['fingerprint']!r} is not the digest "
            "of the manifest of the digests stored with it"
        )


def _is_digest(value: Any) -> bool:
    return isinstance(value, str) and _DIGEST.fullmatch(value) is not None


def _u64(number: int) -> bytes:
    return number.to_bytes(8, "little")


def _text(value: str) -> bytes:
    data = value.encode("utf-8", "surrogatepass")
    return _u64(len(data)) + data


def _column_bytes(values: np.ndarray) -> np.ndarray:
    """Return a piece of a numeric column as the bytes of its encoding."""
    kind = values.dtype.kind
    if kind == "b":
        nonzero: np.ndarray = values.view(np.uint8) != 0
        return nonzero.view(np.uint8)
    if kind in "mM":
        return values.view(np.int64).astype("<i8", copy=False)
    if kind == "f":
        return _float_bits(values, nan=True)
    if kind == "c":
        part = np.dtype(f"f{values.dtype.itemsize // 2}")
        pairs = np.ascontiguousarray(values).view(part).reshape(-1, 2)
        missing = np.isnan(pairs).any(axis=1)
        if missing.any():
            pairs = pairs.copy()
            pairs[missing] = np.nan  # written as the quiet NaN below
        return _float_bits(pairs, nan=True)
    return values.astype(values.dtype.newbyteorder("<"), copy=False)


def _float_bits(values: np.ndarray, nan: bool) -> np.ndarray:
    """Return the bits of floats, little endian, with -0.0 as 0.0 and NaN as quiet.

    The bits are compared and set as integers, so the result does not depend on
    the floating-point environment (flushing subnormals, say). ``nan=False`` skips
    the search for NaN, which a cube's layers cannot hold.
    """
    unsigned, negative_zero, exponent, mantissa, quiet = _FLOAT_BITS[
        values.dtype.itemsize
    ]
    bits = values.view(unsigned)
    zero = bits == negative_zero
    is_nan = ((bits & exponent) == exponent) & ((bits & mantissa) != 0) if nan else None
    if zero.any() or (is_nan is not None and is_nan.any()):
        bits = bits.copy()
        bits[zero] = 0
        if is_nan is not None:
            bits[is_nan] = quiet
    return bits.astype(np.dtype(unsigned).newbyteorder("<"), copy=False)


def _json_value(value: Any) -> bytes:
    """Encode a plain JSON value (as json.loads returns it) as documented above."""
    if value is None:
        return b"n"
    if value is True or value is False:
        return b"t" if value else b"f"
    if isinstance(value, int):
        return b"i" + _text(str(value))
    if isinstance(value, float):
        if value.is_integer():
            return b"i" + _text(str(int(value)))
        return b"d" + struct.pack("<d", value)
    if isinstance(value, str):
        return b"s" + _text(value)
    if isinstance(value, list):
        return b"a" + _u64(len(value)) + b"".join(map(_json_value, value))
    if isinstance(value, dict):
        keys = sorted(value, key=lambda key: key.encode("utf-8", "surrogatepass"))
        members = b"".join(_text(key) + _json_value(value[key]) for key in keys)
        return b"o" + _u64(len(value)) + members
    raise TypeError(f"{type(value).__name__} is not a plain JSON value")
