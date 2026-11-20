"""The reading of doses that dose axes and simulated dose series share.

A dose column of a sample table holds real numbers, finite and at least 0, with 0 for
the vehicle. Distinct doses closer than 1e-8 of the larger are refused as one dose
that a conversion of units left as two, such as ``0.1 * 3`` and ``0.3``.
"""

from typing import Any

import numpy as np
import numpy.typing as npt

from ._masked_arrays import Missing, unmasked_array
from ._messages import listing

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
