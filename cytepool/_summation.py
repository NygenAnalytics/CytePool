"""Floating-point sums, means and quotients rounded once.

Several reference implementations accumulate in extended precision (R's ``sum``,
``mean`` and ``rowMeans``), and some decisions they make, such as which values tie,
depend on the last bit. These routines give the exactly rounded results with error-free
transformations: Knuth's two-sum, Veltkamp's splitting and Dekker's product.
"""

import numpy as np
import numpy.typing as npt

from ._jit import jit

FloatArray = npt.NDArray[np.float64]


@jit
def two_sum(a: float, b: float) -> tuple[float, float]:
    """a + b as the rounded sum and its exact rounding error (Knuth)."""
    s = a + b
    t = s - a
    return s, (a - (s - t)) + (b - t)


@jit
def accurate_sum(values: FloatArray) -> tuple[float, float]:
    """Sum accurate to twice the working precision, as rounded value and remainder.

    This is the cascaded summation of Ogita, Rump and Oishi (2005, SIAM J Sci
    Comput 26:1955); the rounded value is the sum rounded once.
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
    """Split a into two halves of 26 bits each, a = high + low (Veltkamp)."""
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
