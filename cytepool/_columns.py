"""Columns of tables: text columns for results, missing values and equality.

A ``Table`` and the cubes, designs and conversions that build on it read columns the
same way: None, NaN and NaT mark missing values, two columns are equal where their
values are equal or both missing, and a pandas DataFrame is recognised by the
attributes it has, without importing pandas.
"""

from collections.abc import Sequence

import numpy as np


def text_column(values: Sequence[str | None]) -> np.ndarray:
    """A text column for a ``Table``, also when it is empty; None marks a missing
    value.

    numpy reads an empty list as float64, which would make an empty column of ids or
    reasons a column of numbers; an object array stays a text column.
    """
    column = np.empty(len(values), dtype=object)
    column[:] = list(values)
    return column


def is_data_frame(data: object) -> bool:
    return all(
        hasattr(data, attribute)
        for attribute in ("columns", "index", "iloc", "to_numpy", "isna")
    )


def missing_mask(column: np.ndarray) -> np.ndarray:
    """Return where a column holds missing values (None, NaN or NaT)."""
    kind = column.dtype.kind
    if kind == "O":
        return np.array([value is None for value in column.tolist()], dtype=bool)
    if kind in "fc":
        return np.isnan(column)
    if kind in "mM":
        return np.isnat(column)
    return np.zeros(column.shape, dtype=bool)


def values_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compare two columns of one dtype elementwise; missing equals missing.

    Text compares with Python's ``==``, under which None equals only None.
    """
    equal = np.asarray(a == b, dtype=bool)
    if a.dtype.kind in "fcmM":
        equal |= missing_mask(a) & missing_mask(b)
    return equal


def with_missing(column: np.ndarray, mask: np.ndarray) -> np.ndarray:
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
