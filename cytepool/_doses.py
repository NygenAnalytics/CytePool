"""The reading of doses that dose axes, dose designs, candidate shapes and simulated
dose series share.

A dose column of a sample table holds real numbers, finite and at least 0, with 0 for
the vehicle. Distinct doses closer than 1e-8 of the larger are refused as one dose
that a conversion of units left as two, such as ``0.1 * 3`` and ``0.3``. A dose axis
names the design columns it builds after the dose column: ``{column}_treated`` and
``{column}_s1``, ``{column}_s2``, ...; a model of the dose groups names the indicator
of each treated dose after the dose, ``{column}_0.03``, with 12 significant digits,
which separate any two doses that the 1e-8 rule keeps apart.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
import numpy.typing as npt

from ._columns import is_data_frame
from ._masked_arrays import Missing, reject_masked, unmasked_array
from ._messages import close_matches, listing
from .cube import Cube
from .table import Table

FloatArray = npt.NDArray[np.float64]

SAME_DOSE = 1e-8
"""Distinct doses closer than this, relative to the larger, are one dose that a
conversion of units left as two."""

LISTED_DOSES = 20
"""The most doses a message lists: dose series rarely have more."""

TREATED_SUFFIX, SPLINE_SUFFIX = "_treated", "_s"


def real_vector(values: Any, name: str, *, missing: Missing = None) -> FloatArray:
    """A new float64 copy of a 1-D array of real numbers."""
    array = unmasked_array(values, name, missing=missing)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    return array.astype(np.float64)


def checked_doses(values: Any, name: str) -> FloatArray:
    """Doses as float64: a 1-D array of real numbers, finite and at least 0, with a
    dose of -0.0 read as 0."""
    doses = real_vector(values, name)
    bad = ~np.isfinite(doses) | (doses < 0.0)
    if bad.any():
        raise ValueError(
            f"{name} must be finite and at least 0, with 0 for the vehicle; got "
            f"{listing([repr(float(dose)) for dose in doses[bad]])}"
        )
    # -0.0 + 0.0 is 0.0, so that a vehicle given as -0.0 prints as 0.
    unsigned: FloatArray = doses + 0.0
    return unsigned


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
            f"{listing([repr(name) for name in table.columns])}; name the column "
            "that holds the doses with DoseAxis(<column>, ...)"
        )
    return checked_doses(table[column], f"samples[{column!r}]")


def sample_doses(samples: Any, column: str) -> FloatArray:
    """The checked doses of the dose column of a sample table: a ``Table``, a mapping of
    column name to 1-D values, or a pandas DataFrame.

    Raises:
        TypeError: If ``samples`` is not a table or holds a numpy masked array, or the
            column does not hold real numbers.
        ValueError: If the table has no such column, or a dose is not finite or is
            negative.
    """
    reject_masked(samples, "samples", missing="None or NaN")
    if not (isinstance(samples, Table | Mapping) or is_data_frame(samples)):
        hint = "; pass cube.samples" if isinstance(samples, Cube) else ""
        raise TypeError(
            "samples must be a Table, a mapping of column names to 1-D values or "
            f"a pandas DataFrame, got {type(samples).__name__}{hint}"
        )
    table = samples if isinstance(samples, Table) else Table(samples)
    return column_doses(table, column)


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


def dose_text(doses: FloatArray, unit: str) -> str:
    """Doses for messages, such as "0.1, 1 and 10 ug/kg"."""
    return f"{listing([f'{dose:g}' for dose in doses], limit=LISTED_DOSES)} {unit}"


def treated_column(column: str) -> str:
    """The name of the design column that is 1 for a dose above 0."""
    return column + TREATED_SUFFIX


def spline_column(column: str, j: int) -> str:
    """The name of the j-th spline column of dose, counted from 1."""
    return f"{column}{SPLINE_SUFFIX}{j}"


def dose_label(dose: float) -> str:
    """A dose as names show it, with 12 significant digits, such as ``0.03``."""
    return f"{dose:.12g}"


def dose_group_column(column: str, dose: float) -> str:
    """The name of the design column that is 1 for the samples given a treated dose,
    such as ``dose_0.03``."""
    return f"{column}_{dose_label(dose)}"


def is_axis_column(name: str, column: str) -> bool:
    """Whether a variable is the dose column or a column a dose axis builds from it."""
    spline = column + SPLINE_SUFFIX
    return name in (column, treated_column(column)) or (
        name.startswith(spline) and name[len(spline) :].isdigit()
    )
