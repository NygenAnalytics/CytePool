"""The negative binomial unit deviance and the linear predictors of count models.

The unit deviance of a count y at mean mu with dispersion phi, accurate in every
regime, which the negative binomial fits, the quasi-likelihood and DESeq2 share, and
the linear predictor offset + x' beta of each observation, summed over the
coefficients in order (McCarthy, Chen and Smyth 2012).

References:
    McCarthy DJ, Chen Y, Smyth GK (2012). Differential expression analysis of
    multifactor RNA-Seq experiments with respect to biological variation. Nucleic Acids
    Research 40, 4288-4297. doi:10.1093/nar/gks042
"""

import math

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit

FloatArray = npt.NDArray[np.float64]

# The unit deviance is summed as a series when |y - mu| < this * mu.
SERIES_LIMIT = 0.125


@jit
def log1p_ratio(x: float) -> float:
    """log(1 + x) / x, which is 1 at x = 0."""
    if x == 0.0:
        return 1.0
    return math.log1p(x) / x


@jit
def unit_deviance(y: float, mu: float, phi: float) -> float:
    """The negative binomial unit deviance, accurate in every regime.

    With s = (y - mu) / mu and p = phi mu / (1 + phi mu), half the deviance is
    mu s^2 / (1 + phi mu) sum_n T_n / ((n + 1)(n + 2)), with T_0 = 1 and
    T_n = (-s)^n - p s T_{n-1}; for |s| < 1/8 the series is summed. Otherwise it is
    y log(1 + r) - (y - mu) log(1 + x) / x with r = (y - mu) / (mu (1 + phi y)) and
    x = phi (y - mu) / (1 + phi mu), each logarithm of a ratio taken directly where
    its argument is far from 1, or, where phi y or phi mu would overflow,
    -k log(y / mu) - (y + k) (log(1 + k / y) - log(1 + k / mu)) with k = 1 / phi.
    """
    if y == 0.0:
        if phi * mu < 1e300:
            zero: float = 2.0 * mu * log1p_ratio(phi * mu)
            return zero
        return 2.0 * (math.log(phi) + math.log(mu)) / phi
    if mu == 0.0:
        return math.inf
    difference = y - mu
    s = difference / mu
    if abs(s) < SERIES_LIMIT:
        product = phi * mu
        share = (
            product / (1.0 + product) if product <= 1.0 else 1.0 / (1.0 + 1.0 / product)
        )
        t = s * share
        total = 0.5
        term = 1.0
        power = 1.0
        n = 1.0
        while n < 64.0:
            power *= -s
            term = power - t * term
            updated = total + term / ((n + 1.0) * (n + 2.0))
            if updated == total:
                break
            total = updated
            n += 1.0
        return 2.0 * (s * s / (1.0 / mu + phi)) * total
    if phi * max(y, mu) > 1e300:
        k = 1.0 / phi
        gamma: float = -k * math.log(y / mu) - (y + k) * (
            math.log1p(k / y) - math.log1p(k / mu)
        )
        return 2.0 * gamma
    r = difference / (mu * (1.0 + phi * y))
    if -0.5 <= r <= 1e10:
        log_ratio = math.log1p(r)
    else:
        ratio = (y / mu) * ((1.0 + phi * mu) / (1.0 + phi * y))
        if 1e-300 < ratio < 1e300:
            log_ratio = math.log(ratio)
        else:
            log_ratio = (math.log(y) - math.log(mu)) - (
                math.log1p(phi * y) - math.log1p(phi * mu)
            )
    x = phi * difference / (1.0 + phi * mu)
    second: float
    if x >= -0.5:
        second = difference / (1.0 + phi * mu) * log1p_ratio(x)
    else:
        ratio = (1.0 + phi * y) / (1.0 + phi * mu)
        if ratio > 1e-300:
            second = math.log(ratio) / phi
        else:
            second = (math.log1p(phi * y) - math.log1p(phi * mu)) / phi
    return 2.0 * (y * log_ratio - second)


@jit
def unit_deviances(y: FloatArray, mu: FloatArray, phi: FloatArray) -> FloatArray:
    out = np.empty(y.size)
    for k in range(y.size):
        out[k] = unit_deviance(y[k], mu[k], phi[k])
    return out


@jit
def linear_predictors(
    x: FloatArray, beta: FloatArray, offsets: FloatArray, eta: FloatArray
) -> None:
    """eta = x @ beta + offsets: the linear predictors of a count model, each summed
    from its offset over the coefficients in order."""
    n, p = x.shape
    for i in range(n):
        total = offsets[i]
        for k in range(p):
            total += x[i, k] * beta[k]
        eta[i] = total
