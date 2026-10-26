"""Provenance records of cubes, checked and kept as JSON text.

A cube keeps its provenance record (described in ``cytepool.cube.provenance``) as JSON
texts: the record with an empty list in place of its operations, then each operation
record as a text of its own, so that an operation is added without reading the others
again. These are the parts of checking and storing records that cubes, pooling, the
AnnData import and the description of a record share:

- ``plain_json``: a value as plain JSON types, with an error that names a bad entry.
- ``provenance_as_texts``: a provenance record, checked, as texts.
- ``provenance_from_texts``: the record that ``provenance_as_texts`` stored.
- ``add_record``: the texts of a record with an operation record appended.
- ``source_record``: a caller's record of where the data came from, checked, as plain
  JSON.
"""

import json
import math
from collections.abc import Mapping
from typing import Any

import numpy as np

from .._masked_arrays import reject_masked


def plain_json(value: Any, path: str) -> Any:
    """Return value as plain JSON types, raising an error that names a bad entry."""
    if isinstance(value, str):
        return _valid_text(value, path)
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, int | np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        if not math.isfinite(value):
            raise ValueError(f"{path} is {value}; JSON has no NaN or infinity")
        return float(value)
    if isinstance(value, np.ndarray):
        return plain_json(value.tolist(), path)
    if isinstance(value, Mapping):
        plain = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} has the key {key!r}; JSON keys are strings")
            key = _valid_text(key, f"a key of {path}")
            plain[key] = plain_json(item, f"{path}[{key!r}]")
        return plain
    if isinstance(value, list | tuple):
        return [plain_json(item, f"{path}[{i}]") for i, item in enumerate(value)]
    raise TypeError(
        f"{path} has a value of type {type(value).__name__!r}, which is not "
        "JSON-serialisable"
    )


def _valid_text(text: str, path: str) -> str:
    """Return text as a str, refusing what UTF-8 cannot encode (lone surrogates)."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as error:
        raise ValueError(
            f"{path} holds the lone surrogate {text[error.start : error.end]!r}, which "
            "is not valid Unicode text; JSON metadata must be valid text"
        ) from None
    return str(text)


def provenance_as_texts(provenance: Mapping[str, Any] | None) -> tuple[str, ...]:
    """Return a provenance record as JSON texts, after checking it.

    The first text is the record with an empty list in place of its operations,
    and each operation record follows as a text of its own, so that an operation is
    added without reading the others again (``add_record``).
    """
    if provenance is None:
        return ("{}",)
    if not isinstance(provenance, Mapping):
        raise TypeError(
            f"provenance must be a mapping, got {type(provenance).__name__}"
        )
    reject_masked(provenance, "provenance", missing="None")
    plain = plain_json(provenance, "provenance")
    operations = plain.get("operations", [])
    if not isinstance(operations, list):
        raise ValueError("provenance['operations'] must be a list of operation records")
    if "operations" in plain:
        plain["operations"] = []
    return (json.dumps(plain), *map(json.dumps, operations))


def provenance_from_texts(texts: tuple[str, ...]) -> dict[str, Any]:
    """Return the provenance record that ``provenance_as_texts`` stored as texts."""
    record: dict[str, Any] = json.loads(texts[0])
    if len(texts) > 1 or "operations" in record:
        record["operations"] = [json.loads(text) for text in texts[1:]]
    return record


def source_record(source: Any) -> dict[str, Any] | None:
    """Return a caller's source record as plain JSON, after checking it."""
    if source is None:
        return None
    if not isinstance(source, Mapping):
        raise TypeError(
            "source must be a mapping (a JSON object) that records where the data "
            f"came from, such as a dataset identity and artifact references; got "
            f"{type(source).__name__}"
        )
    reject_masked(source, "source", missing="None")
    plain: dict[str, Any] = plain_json(source, "source")
    return plain


def add_record(provenance: tuple[str, ...], record: dict[str, Any]) -> tuple[str, ...]:
    """Return the texts of a provenance record with an operation record appended."""
    return (*provenance, json.dumps(record))
