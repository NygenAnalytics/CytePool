"""Floating-point sums, means and quotients rounded once.

Several reference implementations accumulate in extended precision (R's ``sum``,
``mean`` and ``rowMeans``), and some decisions they make, such as which values tie,
depend on the last bit. These routines give the exactly rounded results with error-free
transformations: Knuth's two-sum (Knuth 1997), and Veltkamp's splitting and Dekker's
product (Dekker 1971), and with the cascaded summation of Ogita, Rump and Oishi (2005).

References:
    Knuth DE (1997). The Art of Computer Programming, Volume 2: Seminumerical
    Algorithms, 3rd edition. Addison-Wesley, Reading.

    Dekker TJ (1971). A floating-point technique for extending the available precision.
    Numerische Mathematik 18, 224-242. doi:10.1007/BF01397083

    Ogita T, Rump SM, Oishi S (2005). Accurate sum and dot product. SIAM Journal on
    Scientific Computing 26, 1955-1988. doi:10.1137/030601818
"""

import math

import numpy as np
import numpy.typing as npt

from ._jit import jit

FloatArray = npt.NDArray[np.float64]

# Below this magnitude the splitting in the exactly rounded division cannot
# overflow; above it, and for a sum that overflowed (NaN after two-sums), a mean
# is a plain division.
_EXACT_DIVISION_LIMIT = 2.0**995


@jit
def two_sum(a: float, b: float) -> tuple[float, float]:
    """a + b as the rounded sum and its exact rounding error (Knuth 1997)."""
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


@jit
def accurate_sum(values: FloatArray) -> tuple[float, float]:
    """Sum accurate to twice the working precision, as rounded value and remainder.

    This is the cascaded summation of Ogita, Rump and Oishi (2005); the rounded value
    is the sum rounded once.
    """
    high = 0.0
    low = 0.0
    for x in values:
        high, error = two_sum(high, x)
        low += error
    total, remainder = two_sum(high, low)
    return total, remainder


@jit
def accurate_mean(values: FloatArray) -> float:
    """Mean of a non-empty array, rounded once, as R's mean() gives it."""
    total, remainder = accurate_sum(values)
    mean: float = rounded_quotient(total, remainder, float(values.size))
    return mean


@jit
def rounded_quotient(total: float, remainder: float, n: float) -> float:
    """(total + remainder) / n, rounded once."""
    quotient = total / n
    # The exact product quotient * n as a rounded value and error (Dekker
    # 1971), so that the remainder of the division can correct the quotient.
    product = quotient * n
    q_high, q_low = split_double(quotient)
    n_high, n_low = split_double(n)
    error = (
        (q_high * n_high - product) + q_high * n_low + q_low * n_high
    ) + q_low * n_low
    result: float = quotient + (((total - product) - error) + remainder) / n
    return result


@jit
def split_double(a: float) -> tuple[float, float]:
    """Split a into two halves of 26 bits each, a = high + low (Veltkamp's splitting,
    Dekker 1971)."""
    c = 134217729.0 * a
    high = c - (c - a)
    return high, a - high


@jit
def accurate_sums(matrix: FloatArray, axis: int) -> FloatArray:
    """Sums over axis 0 (per column) or axis 1 (per row), each rounded once."""
    n_rows, n_columns = matrix.shape
    if axis == 1:
        out = np.empty(n_rows)
        for i in range(n_rows):
            out[i] = accurate_sum(matrix[i])[0]
        return out
    high = np.zeros(n_columns)
    low = np.zeros(n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            s, error = two_sum(high[j], matrix[i, j])
            high[j] = s
            low[j] += error
    return high + low


def observed_means(values: FloatArray) -> FloatArray:
    """Mean of the observed (non-NaN) values of each column; NaN for none.

    limma's ``Amean`` is R's ``rowMeans``, which accumulates in extended
    precision. Sums with error compensation, divided with a single rounding, give
    the exactly rounded mean, which is rowMeans' value to the last bit on
    log-expression data (found by probing; R rounds its extended sum, so values
    spanning many decades can differ in the last bit). voom needs them exactly:
    LOWESS treats tied means differently from means a rounding error apart.
    """
    means: FloatArray = _compensated_means(
        np.require(values, dtype=np.float64, requirements=["C", "W", "A"])
    )
    return means


@jit
def _compensated_means(values: FloatArray) -> FloatArray:
    """Column means of the non-NaN values, each rounded once (NaN for none).

    Near the largest doubles, where the exact division would overflow, the mean is
    the plain sum divided by the count, infinite when the sum overflows.
    """
    n_rows, n_columns = values.shape
    high = np.zeros(n_columns)
    low = np.zeros(n_columns)
    plain = np.zeros(n_columns)
    counts = np.zeros(n_columns)
    for i in range(n_rows):
        for j in range(n_columns):
            value = values[i, j]
            if math.isnan(value):
                continue
            total, error = two_sum(high[j], value)
            high[j] = total
            low[j] += error
            plain[j] += value
            counts[j] += 1.0
    means = np.empty(n_columns)
    for j in range(n_columns):
        if counts[j] == 0.0:
            means[j] = np.nan
            continue
        total, remainder = two_sum(high[j], low[j])
        if abs(total) < _EXACT_DIVISION_LIMIT:
            means[j] = rounded_quotient(total, remainder, counts[j])
        else:
            means[j] = plain[j] / counts[j]
    return means
