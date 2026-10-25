"""An immutable table of named, equal-length columns.

CytePool keeps sample, group and feature metadata, and later tidy results, in a
``Table`` because pandas is not a dependency. A table holds 1-D numpy arrays of equal
length, one per column name. Columns are copied in and read-only, and every method
that changes something returns a new table.

Column types:

- Text columns are object arrays whose values are ``str`` or ``None`` (missing).
  Lists of strings, numpy ``str_`` arrays, numpy ``StringDType`` arrays and object
  arrays are converted to this form; in an object array, ``None``, NaN and pandas'
  ``NA`` and ``NaT`` mark missing values. Comparisons follow Python: a missing value
  equals another missing value and nothing else. (numpy's ``StringDType`` with
  ``None`` as its missing value treats a missing value as equal to ``""`` in array
  comparisons, which is why it is not used.) A text column that holds ``None``
  cannot be sorted, because Python cannot order ``None`` and ``str``.
- Numeric, boolean and datetime columns keep their dtype. NaN (or NaT) is the missing
  value of float (or datetime) columns. Integer and boolean columns have no missing
  value, so a sequence of numbers or booleans that contains ``None`` becomes float64
  with NaN.
- Bytes, structured and other object columns are rejected, and so are numpy masked
  arrays, whose mask numpy would drop: mark missing values with None or NaN instead.
"""

import importlib
import math
import numbers
from collections.abc import Iterator, Mapping
from typing import Any

import numpy as np

from ._masked_arrays import reject_masked, unmasked_array

_REPR_ROWS = 5  # rows shown at each end of a long table
_REPR_COLUMNS = 8  # columns shown as a grid; the rest are listed by name
_REPR_WIDTH = 24  # longest value shown in the grid


class Table:
    """An immutable store of named, equal-length 1-D numpy columns.

    Args:
        data: A mapping of column name to 1-D array-like, another ``Table``, or a
            pandas DataFrame. A DataFrame's index is kept as leading columns when it
            is named and dropped otherwise. ``None`` gives an empty table.

    Raises:
        TypeError: If ``data`` has an unsupported type, a column name is not a
            string, or a column holds unsupported values or is a numpy masked
            array.
        ValueError: If a column name is empty or repeated, a column is not 1-D, or
            the columns have unequal lengths.

    Example:
        >>> table = Table({"sample_id": ["S1", "S2"], "age": [34.0, None]})
        >>> table["age"]
        array([34., nan])
    """

    __slots__ = ("_columns", "_n_rows")

    _columns: dict[str, np.ndarray]
    _n_rows: int

    def __init__(self, data: object = None) -> None:
        if data is None:
            columns: dict[str, np.ndarray] = {}
        elif isinstance(data, Table):
            columns = dict(data._columns)
        elif isinstance(data, Mapping):
            reject_masked(data, "data", missing="None or NaN")
            columns = {}
            for name, values in data.items():
                _check_column_name(name)
                columns[name] = _as_column(name, values)
        elif _is_data_frame(data):
            columns = _columns_from_data_frame(data)
        else:
            raise TypeError(
                "Table data must be a mapping of column names to 1-D arrays, a Table "
                f"or a pandas DataFrame; got {type(data).__name__}"
            )
        _initialize(self, columns)

    # Immutability, copying and pickling.

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(
            "Table is immutable; with_columns(), rename(), select(), take() and "
            "filter() return new tables"
        )

    def __delattr__(self, name: str) -> None:
        raise AttributeError("Table is immutable")

    def __copy__(self) -> "Table":
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> "Table":
        return self

    def __reduce__(self) -> tuple[type["Table"], tuple[dict[str, np.ndarray]]]:
        return (Table, (self.to_dict(),))

    # Columns.

    @property
    def columns(self) -> tuple[str, ...]:
        """The column names, in order."""
        return tuple(self._columns)

    def __len__(self) -> int:
        """Return the number of rows."""
        return self._n_rows

    def __contains__(self, name: object) -> bool:
        """Return whether the table has a column with this name."""
        return name in self._columns

    def __iter__(self) -> Iterator[str]:
        """Iterate over the column names."""
        return iter(self._columns)

    def __getitem__(self, name: str) -> np.ndarray:
        """Return a column as a read-only array.

        Raises:
            TypeError: If ``name`` is not a string.
            KeyError: If there is no column with this name.
        """
        if not isinstance(name, str):
            raise TypeError(
                "columns are looked up by name (str); use take() or filter() to choose "
                f"rows and select() to choose several columns, not {type(name).__name__}"
            )
        try:
            return self._columns[name].view()
        except KeyError:
            raise KeyError(
                f"no column named {name!r}; columns are: {_names(self.columns)}"
            ) from None

    # Deriving tables.

    def take(self, indices: Any) -> "Table":
        """Return the rows at the given positions, in that order.

        Args:
            indices: 1-D integer positions; negative positions count from the end.

        Raises:
            TypeError: If ``indices`` are not integers, or are a numpy masked array.
            ValueError: If ``indices`` are not 1-D.
            IndexError: If a position is out of range.
        """
        positions = unmasked_array(indices, "indices")
        if positions.dtype.kind == "b":
            raise TypeError("take() needs integer positions; use filter() for a mask")
        if positions.size == 0:
            positions = positions.astype(np.intp)
        if positions.dtype.kind not in "iu":
            raise TypeError(
                f"take() needs integer positions, got dtype {positions.dtype}"
            )
        if positions.ndim != 1:
            raise ValueError(f"positions must be 1-D, got shape {positions.shape}")
        n = self._n_rows
        outside = (positions < -n) | (positions >= n)
        if outside.any():
            raise IndexError(
                f"position {positions[outside][0]} is out of range for a table with "
                f"{n} rows"
            )
        return self._with_new_arrays(
            {name: column[positions] for name, column in self._columns.items()},
            n_rows=positions.size,
        )

    def filter(self, mask: Any) -> "Table":
        """Return the rows where ``mask`` is true.

        Args:
            mask: A boolean array with one value per row.

        Raises:
            TypeError: If ``mask`` is not boolean, or is a numpy masked array.
            ValueError: If ``mask`` does not have one value per row.
        """
        keep = unmasked_array(mask, "mask")
        if keep.dtype != np.bool_:
            raise TypeError(
                f"filter() needs a boolean mask, got dtype {keep.dtype}; use take() for "
                "positions"
            )
        if keep.shape != (self._n_rows,):
            raise ValueError(
                f"mask must have one value per row ({self._n_rows}), got shape "
                f"{keep.shape}"
            )
        return self._with_new_arrays(
            {name: column[keep] for name, column in self._columns.items()},
            n_rows=int(np.count_nonzero(keep)),
        )

    def select(self, names: Any) -> "Table":
        """Return a table with the given columns, in the given order.

        Args:
            names: Column names; a single string selects one column.

        Raises:
            KeyError: If a name is not a column.
            TypeError: If ``names`` is a numpy masked array.
            ValueError: If a name is repeated.
        """
        reject_masked(names, "names")
        chosen = [names] if isinstance(names, str) else list(names)
        missing = [name for name in chosen if name not in self._columns]
        if missing:
            raise KeyError(
                f"no column named {missing[0]!r}; columns are: {_names(self.columns)}"
            )
        repeated = _repeated(chosen)
        if repeated:
            raise ValueError(f"column {repeated[0]!r} is selected more than once")
        return _from_columns(
            {name: self._columns[name] for name in chosen},
            self._n_rows if chosen else 0,
        )

    def with_columns(
        self, columns: Mapping[str, Any] | None = None, /, **named: Any
    ) -> "Table":
        """Return a table with columns added or replaced.

        A replaced column keeps its position; new columns are appended in the order
        given. Names that are not valid Python identifiers can be passed in the
        ``columns`` mapping.

        Args:
            columns: A mapping of column name to 1-D array-like.
            **named: More columns, given as keyword arguments.

        Raises:
            TypeError: If ``columns`` is not a mapping, or a name or column has an
                unsupported type or is a numpy masked array.
            ValueError: If a column is given twice, or a new column's length differs
                from the table's.
        """
        updates: dict[str, Any] = {}
        if columns is not None:
            if not isinstance(columns, Mapping):
                raise TypeError(
                    f"columns must be a mapping of name to values, got "
                    f"{type(columns).__name__}"
                )
            reject_masked(columns, "columns", missing="None or NaN")
            updates.update(columns)
        for name, values in named.items():
            reject_masked(values, name, missing="None or NaN")
            if name in updates:
                raise ValueError(f"column {name!r} is given twice")
        updates.update(named)
        converted = {}
        for name, values in updates.items():
            _check_column_name(name)
            converted[name] = _as_column(name, values)
        if self._columns:
            for name, column in converted.items():
                if len(column) != self._n_rows:
                    raise ValueError(
                        f"column {name!r} has {len(column)} values; the table has "
                        f"{self._n_rows} rows"
                    )
        result = dict(self._columns)
        result.update(converted)
        table = object.__new__(Table)
        _initialize(table, result)
        return table

    def rename(self, mapping: Mapping[str, str]) -> "Table":
        """Return a table with columns renamed.

        Args:
            mapping: Old column name to new column name.

        Raises:
            KeyError: If an old name is not a column.
            ValueError: If renaming would repeat a column name.
        """
        if not isinstance(mapping, Mapping):
            raise TypeError(
                f"rename() needs a mapping of old to new names, got "
                f"{type(mapping).__name__}"
            )
        for old, new in mapping.items():
            if old not in self._columns:
                raise KeyError(
                    f"no column named {old!r}; columns are: {_names(self.columns)}"
                )
            _check_column_name(new)
        names = [mapping.get(name, name) for name in self._columns]
        repeated = _repeated(names)
        if repeated:
            raise ValueError(f"renaming would give two columns named {repeated[0]!r}")
        return _from_columns(
            dict(zip(names, self._columns.values(), strict=True)), self._n_rows
        )

    # Conversion and comparison.

    def to_dict(self) -> dict[str, np.ndarray]:
        """Return a new dict of column name to read-only array."""
        return {name: column.view() for name, column in self._columns.items()}

    def to_pandas(self) -> Any:
        """Return a pandas DataFrame with the same columns.

        Text columns become object columns of ``str`` and ``None`` (pandas 3 stores
        them in its string dtype, with NaN as the missing value). Passing the
        DataFrame back to ``Table`` gives an equal table, except that pandas stores
        datetimes in seconds or finer units, so a coarser unit such as days comes
        back as seconds.

        Raises:
            ImportError: If pandas is not installed.
        """
        try:
            pandas = importlib.import_module("pandas")
        except ImportError as error:
            raise ImportError(
                "Table.to_pandas() needs pandas, which is not installed; install it "
                "with 'pip install pandas'"
            ) from error
        return pandas.DataFrame(self._columns, columns=list(self._columns), copy=True)

    def equals(self, other: object) -> bool:
        """Return whether ``other`` is a table with the same columns and values.

        Column names, their order, dtypes and values must match. Missing values
        (None, NaN, NaT) are equal to each other.
        """
        if not isinstance(other, Table):
            return False
        if self.columns != other.columns or self._n_rows != other._n_rows:
            return False
        for name, column in self._columns.items():
            theirs = other._columns[name]
            if column.dtype != theirs.dtype:
                return False
            if not _values_equal(column, theirs).all():
                return False
        return True

    def __repr__(self) -> str:
        names = self.columns
        header = (
            f"Table with {self._n_rows:,} row{'' if self._n_rows == 1 else 's'} and "
            f"{len(names)} column{'' if len(names) == 1 else 's'}"
        )
        if not names:
            return header
        shown = names[:_REPR_COLUMNS]
        if self._n_rows <= 2 * _REPR_ROWS:
            rows = list(range(self._n_rows))
        else:
            rows = list(range(_REPR_ROWS))
            rows += list(range(self._n_rows - _REPR_ROWS, self._n_rows))
        grid = [
            list(shown),
            [_dtype_label(self._columns[name].dtype) for name in shown],
        ]
        for row in rows:
            grid.append([_format_value(self._columns[name][row]) for name in shown])
        if self._n_rows > 2 * _REPR_ROWS:
            grid.insert(2 + _REPR_ROWS, ["..."] * len(shown))
        widths = [max(len(line[j]) for line in grid) for j in range(len(shown))]
        lines = [header]
        for line in grid:
            cells = (
                cell.ljust(width) for cell, width in zip(line, widths, strict=True)
            )
            lines.append(("  " + "  ".join(cells)).rstrip())
        hidden = names[_REPR_COLUMNS:]
        if hidden:
            listed = ", ".join(
                f"{name} ({_dtype_label(self._columns[name].dtype)})" for name in hidden
            )
            lines.append(f"  {len(hidden)} more columns: {listed}")
        return "\n".join(lines)

    def _with_new_arrays(self, columns: dict[str, np.ndarray], n_rows: int) -> "Table":
        for column in columns.values():
            column.flags.writeable = False
        return _from_columns(columns, n_rows if columns else 0)


def _initialize(table: Table, columns: dict[str, np.ndarray]) -> None:
    lengths = {name: len(column) for name, column in columns.items()}
    if len(set(lengths.values())) > 1:
        described = ", ".join(f"{name!r} has {n}" for name, n in lengths.items())
        raise ValueError(f"columns must have equal lengths; {described}")
    object.__setattr__(table, "_columns", columns)
    object.__setattr__(table, "_n_rows", next(iter(lengths.values()), 0))


def _from_columns(columns: dict[str, np.ndarray], n_rows: int) -> Table:
    """Build a table from read-only columns that are already validated."""
    table = object.__new__(Table)
    object.__setattr__(table, "_columns", columns)
    object.__setattr__(table, "_n_rows", n_rows)
    return table


def _check_column_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError(f"column names must be str, got {type(name).__name__} {name!r}")
    if not name:
        raise ValueError("column names must be non-empty strings")


def _as_column(name: str, values: Any, missing: np.ndarray | None = None) -> np.ndarray:
    """Convert values to a read-only 1-D column that the table owns.

    Args:
        name: The column name, for error messages.
        values: A 1-D array-like.
        missing: For object arrays, a mask of missing values that overrides the
            default rule (None or NaN).
    """
    try:
        array = unmasked_array(values, f"column {name!r}", missing="None or NaN")
    except ValueError as error:
        raise ValueError(
            f"column {name!r} could not be read as a 1-D array; columns must be flat "
            f"sequences, not nested or ragged ones ({error})"
        ) from error
    if array.ndim != 1:
        raise ValueError(f"column {name!r} must be 1-D, got shape {array.shape}")
    kind = array.dtype.kind
    if kind == "U" and not isinstance(values, np.ndarray):
        # numpy turns a sequence that mixes text with numbers or NaN into text
        # ("1", "nan"); read such a sequence value by value instead.
        items = list(values)
        if not all(isinstance(item, str) for item in items):
            array = np.empty(len(items), dtype=object)
            array[:] = items
            kind = "O"
    if kind in "biufcmM":
        column = array.astype(array.dtype.newbyteorder("="), copy=True)
    elif kind in "UTO":
        column = _column_from_objects(name, array.astype(object), missing)
    elif kind == "S":
        raise TypeError(
            f"column {name!r} holds bytes; decode it to str first, for example with "
            "numpy.char.decode"
        )
    else:
        raise TypeError(
            f"column {name!r} has unsupported dtype {array.dtype}; columns hold text, "
            "numbers, booleans or datetimes"
        )
    column.flags.writeable = False
    return column


def _column_from_objects(
    name: str, array: np.ndarray, missing: np.ndarray | None
) -> np.ndarray:
    values = array.tolist()
    if missing is None:
        is_missing = [_is_missing_object(value) for value in values]
    else:
        is_missing = np.asarray(missing, dtype=bool).tolist()
    present = [value for value, gap in zip(values, is_missing, strict=True) if not gap]
    if all(isinstance(value, str) for value in present):
        text = np.empty(len(values), dtype=object)
        text[:] = [
            None if gap else str(value)
            for value, gap in zip(values, is_missing, strict=True)
        ]
        return text
    any_missing = any(is_missing)
    if all(isinstance(value, bool | np.bool_) for value in present):
        if not any_missing:
            return np.array(present, dtype=bool)
        return _floats_with_missing(present, is_missing)
    if all(
        isinstance(value, numbers.Real) and not isinstance(value, bool | np.bool_)
        for value in present
    ):
        if any_missing:
            return _floats_with_missing(present, is_missing)
        numeric = np.asarray(present)
        if numeric.dtype.kind in "iuf":
            return numeric
    types = sorted({type(value).__name__ for value in present})
    found = (
        f"holds values of type {types[0]}"
        if len(types) == 1
        else f"mixes values of types {', '.join(types)}"
    )
    raise TypeError(
        f"column {name!r} {found}; a column holds text (str, with None for missing "
        "values), numbers, booleans or numpy datetimes"
    )


def _floats_with_missing(present: list[Any], is_missing: list[bool]) -> np.ndarray:
    gaps = np.array(is_missing, dtype=bool)
    column = np.full(gaps.size, np.nan)
    column[~gaps] = np.asarray(present, dtype=np.float64)
    return column


def _is_missing_object(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float | np.floating):
        return math.isnan(float(value))
    # pandas.NA and pandas.NaT, recognized without importing pandas.
    return type(value).__name__ in ("NAType", "NaTType")


def _is_data_frame(data: object) -> bool:
    return all(
        hasattr(data, attribute)
        for attribute in ("columns", "index", "iloc", "to_numpy", "isna")
    )


def _columns_from_data_frame(frame: Any) -> dict[str, np.ndarray]:
    index_names = [level for level in frame.index.names if level is not None]
    if index_names:
        clashing = [level for level in index_names if level in set(frame.columns)]
        if clashing:
            raise ValueError(
                f"the DataFrame index {clashing[0]!r} is also a column name; rename "
                "one of them or drop the index with reset_index(drop=True)"
            )
        frame = frame.reset_index()
    names = list(frame.columns)
    for name in names:
        _check_column_name(name)
    repeated = _repeated(names)
    if repeated:
        raise ValueError(
            f"the DataFrame has more than one column named {repeated[0]!r}"
        )
    columns = {}
    for position, name in enumerate(names):
        series = frame.iloc[:, position]
        values = series.to_numpy()
        missing = series.isna().to_numpy() if values.dtype == object else None
        columns[name] = _as_column(name, values, missing)
    return columns


def _missing_mask(column: np.ndarray) -> np.ndarray:
    """Return where a column holds missing values (None, NaN or NaT)."""
    kind = column.dtype.kind
    if kind == "O":
        return np.array([value is None for value in column.tolist()], dtype=bool)
    if kind in "fc":
        return np.isnan(column)
    if kind in "mM":
        return np.isnat(column)
    return np.zeros(column.shape, dtype=bool)


def _values_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compare two columns of one dtype elementwise; missing equals missing.

    Text compares with Python's ``==``, under which None equals only None.
    """
    equal = np.asarray(a == b, dtype=bool)
    if a.dtype.kind in "fcmM":
        equal |= _missing_mask(a) & _missing_mask(b)
    return equal


def _with_missing(column: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return a read-only copy of a column with missing values where mask is true.

    Integer and boolean columns have no missing value and become float64.
    """
    if not mask.any():
        return column
    kind = column.dtype.kind
    result = column.astype(np.float64) if kind in "biu" else column.copy()
    if kind == "O":
        result[mask] = None
    elif kind in "mM":
        result[mask] = np.array("NaT", dtype=result.dtype)
    else:
        result[mask] = np.nan
    result.flags.writeable = False
    return result


def _repeated(names: list[Any]) -> list[Any]:
    seen: set[Any] = set()
    repeated = []
    for name in names:
        if name in seen:
            repeated.append(name)
        seen.add(name)
    return repeated


def _names(names: tuple[str, ...]) -> str:
    return ", ".join(names) if names else "(none)"


def _dtype_label(dtype: np.dtype) -> str:
    return "str" if dtype.kind == "O" else dtype.name


def _format_value(value: object) -> str:
    text = "None" if value is None else str(value)
    if len(text) > _REPR_WIDTH:
        text = text[: _REPR_WIDTH - 3] + "..."
    return text
