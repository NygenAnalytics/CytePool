"""The axes of a cube: its tables of groups, samples and features, its cell numbers,
and the labels of cells, read and checked.

Each axis of a cube is described by a table whose id column holds unique, non-empty
text ids. These are the parts of reading them that cubes, pooling and the AnnData
import share:

- ``checked_axis_table``: an axis table checked, with the position of each id.
- ``checked_n_cells``: the number of cells of every sample in every group, from an
  integer array or a DataFrame labelled by the ids, as a read-only int64 array.
- ``axis_codes``: the position on an axis of every cell's label, with the axis table.
- ``count_dtype``: the dtype of a counts layer that holds pooled counts exactly.
- ``first_index``: the index of the first entry, in C order, where a mask is true.
- ``INT64_MAX``, ``UINT32_MAX`` and ``COUNT_LIMIT``: the largest int64, the largest
  uint32 (as a uint64), and 2**53, below which float64 holds every whole number.
"""

import math
from typing import Any

import numpy as np

from .._masked_arrays import reject_masked_table, unmasked_array
from ..table import Table
from .fingerprints import _column_type

# The largest int64. Integer layers and n_cells hold values from 0 to it, so every
# value converts to int64 exactly, and a sum is exact in int64 whenever it fits.
INT64_MAX = 2**63 - 1
# float64 holds every whole number below 2**53, so library sizes and integer sums of
# counts below it are exact; larger counts are rejected.
COUNT_LIMIT = 2.0**53
UINT32_MAX = np.uint64(2**32 - 1)
_OPTIONAL_TEXT_COLUMNS = {
    "groups": ("cell_ontology_id", "parent"),
    "features": ("symbol",),
}
_OPTIONAL_BOOLEAN_COLUMNS = {"features": ("measured",)}


def checked_axis_table(
    value: object, axis: str, id_column: str
) -> tuple[Table, dict[str, int]]:
    try:
        table = value if isinstance(value, Table) else Table(value)
    except (TypeError, ValueError) as error:
        error.add_note(f"(while reading the {axis} table)")
        raise
    if id_column not in table:
        raise ValueError(
            f"the {axis} table needs a {id_column!r} column; it has: "
            f"{', '.join(table.columns) or '(none)'}"
        )
    if len(table) == 0:
        # An empty column cannot hold a wrong value, but numpy reads [] as float64.
        text_columns = (id_column, *_OPTIONAL_TEXT_COLUMNS.get(axis, ()))
        empty: dict[str, np.ndarray] = {
            name: np.empty(0, dtype=object) for name in text_columns if name in table
        }
        for name in _OPTIONAL_BOOLEAN_COLUMNS.get(axis, ()):
            if name in table:
                empty[name] = np.empty(0, dtype=bool)
        table = table.with_columns(empty)
    ids = table[id_column]
    if ids.dtype != object:
        raise TypeError(
            f"{axis}[{id_column!r}] must hold text ids, got dtype {ids.dtype}; convert "
            "the ids to strings first"
        )
    index: dict[str, int] = {}
    for position, identifier in enumerate(ids.tolist()):
        if identifier is None or identifier == "":
            raise ValueError(
                f"{axis}[{id_column!r}] has a missing or empty id at row {position}"
            )
        if identifier in index:
            raise ValueError(
                f"{axis}[{id_column!r}] repeats the id {identifier!r}; ids must be "
                "unique"
            )
        index[identifier] = position
    for column in _OPTIONAL_TEXT_COLUMNS.get(axis, ()):
        if column in table and table[column].dtype != object:
            raise TypeError(
                f"{axis}[{column!r}] must hold text, got dtype {table[column].dtype}"
            )
    for column in _OPTIONAL_BOOLEAN_COLUMNS.get(axis, ()):
        if column in table and table[column].dtype != np.bool_:
            raise TypeError(
                f"{axis}[{column!r}] must be boolean, got dtype {table[column].dtype}"
            )
    for column in table.columns:  # the types a fingerprint and a cube file encode
        if table[column].dtype.kind != "O":
            _column_type(table[column].dtype, f"{axis}[{column!r}]")
    return table, index


def checked_n_cells(
    value: Any, ids: tuple[np.ndarray, np.ndarray, np.ndarray]
) -> np.ndarray:
    group_ids, sample_ids, _ = ids
    if all(hasattr(value, attribute) for attribute in ("index", "columns", "to_numpy")):
        value = _aligned_by_labels(value, sample_ids, group_ids)
    array = unmasked_array(value, "n_cells")
    if array.dtype.kind not in "iu":
        raise TypeError(f"n_cells must be an integer array, got dtype {array.dtype}")
    expected = (sample_ids.size, group_ids.size)
    if array.shape != expected:
        raise ValueError(
            f"n_cells must have shape (n_samples, n_groups) = {expected}, got "
            f"{array.shape}"
        )
    # Only uint64 (in either byte order) holds values above 2**63 - 1.
    if array.dtype.kind == "u" and array.dtype.itemsize == 8 and array.size:
        too_large = array > INT64_MAX
        if too_large.any():
            sample, group = first_index(too_large)
            raise ValueError(
                f"n_cells is above 2**63 - 1 ({array[sample, group]}) for sample "
                f"{sample_ids[sample]!r} and group {group_ids[group]!r}; cell numbers "
                "must be at most 2**63 - 1, the largest int64"
            )
    cells = array.astype(np.int64)
    negative = np.argwhere(cells < 0)
    if negative.size:
        sample, group = negative[0]
        raise ValueError(
            f"n_cells is negative ({cells[sample, group]}) for sample "
            f"{sample_ids[sample]!r} and group {group_ids[group]!r}"
        )
    cells.flags.writeable = False
    return cells


def _aligned_by_labels(
    frame: Any, sample_ids: np.ndarray, group_ids: np.ndarray
) -> np.ndarray:
    """Reorder a DataFrame of cell numbers to the cube's sample and group order."""
    positions = []
    for axis, labels, ids in (
        ("rows", list(frame.index), sample_ids.tolist()),
        ("columns", list(frame.columns), group_ids.tolist()),
    ):
        where = {label: i for i, label in enumerate(labels)}
        known = set(ids)
        missing = [identifier for identifier in ids if identifier not in where]
        extra = [label for label in labels if label not in known]
        if missing or extra or len(where) != len(labels):
            kind = "sample" if axis == "rows" else "group"
            raise ValueError(
                f"the {axis} of an n_cells DataFrame must be the {kind} ids, each once; "
                f"missing: {missing[:5]}, unexpected: {extra[:5]}; pass a numpy array "
                "to give cell numbers by position"
            )
        positions.append([where[identifier] for identifier in ids])
    values = np.asarray(frame.to_numpy())
    if values.dtype == object:  # for example pandas' nullable integer columns
        if hasattr(frame, "isna") and np.asarray(frame.isna().to_numpy()).any():
            raise ValueError(
                "an n_cells DataFrame has missing values; every entry needs a number "
                "of cells"
            )
        # Keep a conversion only when it is exact: 1.5 would become 1, and pandas
        # turns a UInt64 count of 2**63 into -2**63 in int64. uint64 keeps such a
        # count for the caller to report; otherwise the caller reports the dtype.
        overflowed = False
        for dtype in (np.int64, np.uint64):
            try:
                integers = np.asarray(frame.to_numpy(dtype=dtype))
            except OverflowError:
                overflowed = True
                continue
            except (TypeError, ValueError):
                continue
            if bool(np.all(integers == values)):
                values = integers
                break
        else:
            if overflowed:
                raise ValueError(
                    "an n_cells DataFrame holds a number of cells outside the int64 "
                    "range; cell numbers must be from 0 to 2**63 - 1"
                )
    aligned: np.ndarray = values[np.ix_(*positions)]
    return aligned


def first_index(mask: np.ndarray) -> tuple[int, ...]:
    """Return the index of the first entry, in C order, where a mask is true."""
    return tuple(int(position) for position in np.argwhere(mask)[0])


def axis_codes(
    labels: Any,
    name: str,
    n_cells: int,
    table: Any,
    table_name: str,
    id_column: str,
    unit: str = "cell",
) -> tuple[np.ndarray, Table, bool]:
    """Return the position of every cell's label on an axis, the axis table, and
    whether the labels were integers.

    ``unit`` names what holds the labels in messages: a cell, or a row of a table
    of profiles that already pools cells.
    """
    codes, distinct, integers = _factorize(labels, name, n_cells, unit)
    if table is None:
        ids = np.empty(len(distinct), dtype=object)
        ids[:] = distinct
        return codes, Table({id_column: ids}), integers
    reject_masked_table(table, table_name, id_column)
    axis_table, index = checked_axis_table(table, name, id_column)
    positions = np.empty(len(distinct), dtype=np.int64)
    for code, label in enumerate(distinct):
        position = index.get(label)
        if position is None:
            first = int(np.argmax(codes == code))
            raise ValueError(
                f"the label {label!r} of {unit} {first} in {name} is not in "
                f"{table_name}[{id_column!r}]; every label needs a row"
            )
        positions[code] = position
    return positions[codes], axis_table, integers


def _factorize(
    labels: Any, name: str, n_cells: int, unit: str = "cell"
) -> tuple[np.ndarray, list[str], bool]:
    """Return the code of every cell's label, the sorted distinct labels, and
    whether the labels are integers.

    Codes index the returned labels. Integer labels sort in numeric order and become
    their decimal text. ``unit`` names what holds the labels in messages.
    """
    categorical = _categorical_parts(labels)
    if categorical is not None:
        codes, values = categorical
    else:
        array = unmasked_array(labels, name)
        if array.ndim != 1:
            raise ValueError(
                f"{name} must be 1-D, one label per {unit}, got shape {array.shape}"
            )
        if array.dtype.kind == "U" and not isinstance(labels, np.ndarray):
            # numpy turns a sequence that mixes text with NaN or numbers into text
            # ("nan", "1"); check such a sequence value by value instead.
            items = list(labels)
            if not all(isinstance(item, str) for item in items):
                array = np.empty(len(items), dtype=object)
                array[:] = items
        if array.size == 0:  # numpy reads [] as float64
            codes, values = np.empty(0, dtype=np.int64), []
        elif array.dtype.kind in "iu":
            unique, codes = np.unique(array, return_inverse=True)
            values = unique.tolist()
        elif array.dtype.kind in "UTO":
            index: dict[Any, int] = {}
            try:
                codes = np.fromiter(
                    (index.setdefault(item, len(index)) for item in array.tolist()),
                    dtype=np.int64,
                    count=array.size,
                )
            except TypeError:
                raise TypeError(
                    f"{name} must hold text or integer labels, got values that "
                    "cannot be used as labels"
                ) from None
            values = list(index)
        else:
            raise TypeError(
                f"{name} must hold text or integer labels, got dtype {array.dtype}"
            )
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if codes.size != n_cells:
        raise ValueError(
            f"{name} must hold one label per {unit}: counts has {n_cells} rows, "
            f"{name} has {codes.size} labels"
        )
    if codes.size and codes.min() < 0:  # a categorical's code for a missing value
        raise ValueError(
            f"{name} has a missing label at {unit} {int(np.argmax(codes < 0))}"
        )
    used = np.unique(codes)
    if used.size < len(values):  # a categorical with unused categories
        values = [values[code] for code in used.tolist()]
        codes = np.searchsorted(used, codes)
    kinds: set[type] = set()
    for code, value in enumerate(values):
        if isinstance(value, str) and value:
            kinds.add(str)
            continue
        if isinstance(value, int | np.integer) and not isinstance(
            value, bool | np.bool_
        ):
            kinds.add(int)
            continue
        first = int(np.argmax(codes == code))
        if (
            value is None
            or type(value).__name__ in ("NAType", "NaTType")
            or (isinstance(value, float) and math.isnan(value))
        ):
            raise ValueError(f"{name} has a missing label at {unit} {first}")
        if isinstance(value, str):
            raise ValueError(f"{name} has an empty label at {unit} {first}")
        raise TypeError(
            f"{name} must hold text or integer labels; {unit} {first} has "
            f"{type(value).__name__} {value!r}"
        )
    if len(kinds) > 1:
        raise TypeError(f"{name} mixes text and integer labels; use one kind")
    order = sorted(range(len(values)), key=lambda code: values[code])
    rank = np.empty(len(values), dtype=np.int64)
    rank[order] = np.arange(len(values))
    return rank[codes], [str(values[code]) for code in order], kinds == {int}


def _categorical_parts(labels: Any) -> tuple[np.ndarray, list[Any]] | None:
    """Return the codes and categories of pandas categorical labels, else None."""
    if getattr(getattr(labels, "dtype", None), "name", None) != "category":
        return None
    parts = getattr(labels, "cat", labels)  # a Series keeps them in .cat
    try:
        codes = np.asarray(parts.codes)
        categories = np.asarray(parts.categories, dtype=object).tolist()
    except AttributeError:
        return None
    return codes, categories


def count_dtype(counts: np.ndarray, largest: Any) -> type:
    """Return the dtype of a counts layer that holds these counts exactly.

    ``counts`` are non-negative and finite, integers at most 2**63 - 1, and
    ``largest`` is their largest value (0 when there are none). The dtype is uint32
    when every count is a whole number up to 2**32 - 1, uint64 for larger whole
    numbers, and float64 when a count is not a whole number or, for float input,
    is at least 2**53, beyond which float64 does not hold every whole number.
    """
    whole = counts.dtype.kind in "iu" or bool(
        largest < COUNT_LIMIT and np.all(counts == np.floor(counts))
    )
    if not whole:
        return np.float64
    return np.uint32 if largest <= int(UINT32_MAX) else np.uint64
