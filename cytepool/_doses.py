"""The reading of doses that dose axes, dose designs and simulated dose series share.

A dose column of a sample table holds real numbers, finite and at least 0, with 0 for
the vehicle. Distinct doses closer than 1e-8 of the larger are refused as one dose
that a conversion of units left as two, such as ``0.1 * 3`` and ``0.3``.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import Missing, unmasked_array
from ._messages import close_matches, listing
from .table import Table

FloatArray = npt.NDArray[np.float64]

SAME_DOSE = 1e-8
"""Distinct doses closer than this, relative to the larger, are one dose that a
conversion of units left as two."""


def real_vector(values: Any, name: str, *, missing: Missing = None) -> FloatArray:
    """A new float64 copy of a 1-D array of real numbers."""
    array = unmasked_array(values, name, missing=missing)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    return array.astype(np.float64)


def checked_doses(values: Any, name: str) -> FloatArray:
    """Doses as float64: a 1-D array of real numbers, finite and at least 0."""
    doses = real_vector(values, name)
    bad = ~np.isfinite(doses) | (doses < 0.0)
    if bad.any():
        raise ValueError(
            f"{name} must be finite and at least 0, with 0 for the vehicle; got "
            f"{listing([repr(float(dose)) for dose in doses[bad]])}"
        )
    return doses


def column_doses(table: Table, column: str) -> FloatArray:
    """The checked doses of a sample table's dose column.

    Raises:
        TypeError: If the column does not hold real numbers.
        ValueError: If the table has no such column, or a dose is not finite or is
            negative.
    """
    if column not in table:
        raise ValueError(
            f"the dose column {column!r} is not a column of the sample "
            f"table{close_matches(column, table.columns)}; columns are: "
            f"{listing([repr(name) for name in table.columns])}"
        )
    return checked_doses(table[column], f"samples[{column!r}]")


def distinct_doses(doses: FloatArray, unit: str) -> FloatArray:
    """The distinct doses, ascending, refusing two that differ only by rounding.

    Raises:
        ValueError: If two distinct doses differ by less than ``SAME_DOSE`` of the
            larger.
    """
    levels = np.unique(doses)
    close = np.flatnonzero(np.diff(levels) <= SAME_DOSE * levels[1:])
    if close.size:
        a, b = float(levels[close[0]]), float(levels[close[0] + 1])
        raise ValueError(
            f"the doses {a!r} and {b!r} {unit} differ by less than 1e-8 of the dose, "
            "as a conversion of units can leave one dose as two; round the dose "
            "column first, for example with numpy.round"
        )
    return levels
