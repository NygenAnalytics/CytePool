"""Special functions for statistics, compiled with numba.

The log-gamma, digamma and trigamma functions, the inverse of the trigamma function,
the log-beta function, and the regularized incomplete gamma and beta functions with
their complements and logarithms.

Every public function is vectorised: its arguments broadcast with numpy's rules and the
result is a float64 array of the broadcast shape, or a Python float when every argument
is a scalar. NaN arguments give NaN, and so do arguments outside a function's domain;
the functions never raise for numeric input. Arguments that are not real numbers
(strings, complex numbers, objects) raise ``TypeError``, and so do numpy masked arrays,
whose mask numpy would drop: NaN marks a missing value instead.

Each public function calls a compiled scalar routine named ``<function>_scalar``,
which other compiled code can call too.

Methods:
    - ``lgamma`` is the C library's ``lgamma``.
    - ``digamma``: a Taylor series about its positive root on [1, 2], which keeps the
      relative accuracy near the root; the recurrence psi(x + 1) = psi(x) + 1/x maps
      (0, 10) onto [1, 2]; the asymptotic expansion in Bernoulli numbers for x >= 10;
      reflection for x < 0 (DLMF 5.5, 5.11, 5.15).
    - ``trigamma``: recurrence up to x >= 12, then the asymptotic expansion; reflection
      for x < 0 (DLMF 5.15).
    - ``trigamma_inverse``: Newton's method on 1/trigamma(x), which is convex and close
      to linear (Smyth 2004). Both starting values lie to the right of the root, so the
      iteration decreases monotonically to it.
    - ``log_beta`` and the prefactors of the incomplete functions use Stirling's series
      with Loader's (2000) deviance form for large arguments, so that no large
      logarithms cancel.
    - Incomplete gamma: the power series for P (DLMF 8.7.1) and Legendre's continued
      fraction for Q (DLMF 8.9.2), each used where it converges fast and computing the
      smaller tail; the larger tail is its complement in log space. For shapes below 1
      and x <= 1.5, Q is computed directly from the alternating series of the lower
      function, since 1 - P would lose accuracy there. For shapes from 1e3 near the
      center (0.58 a < x < 1.59 a), Temme's uniform asymptotic expansion (DLMF 8.12)
      replaces the series and the fraction, whose step counts grow with the shape.
    - Incomplete beta: the continued fraction DLMF 8.17.22 in its even contraction,
      evaluated with the modified Lentz method (Lentz 1976; Thompson and Barnett 1986)
      for I_x(a, b) when x < (a + 1)/(a + b + 2), and for 1 - I_x(a, b) = I_{1-x}(b, a)
      otherwise. The contraction is written in lambda = a(1 - x) - b x, formed without
      cancellation, so that it stays accurate when x is close to 1 and a is large (the
      t and F distributions with many degrees of freedom). For a shape below 1 the tail
      given by the fraction can be close to 1; the other tail then comes from a
      term-by-term series of its own rather than from 1 minus it (the analogue of the
      incomplete gamma case). For two shapes from 1e10 within about one standard
      deviation of the center, where the fraction's steps grow like the cube root of
      the shapes, Temme's (1982) uniform expansion replaces it, as for the incomplete
      gamma function. Shapes from 2^500, and shapes below 1e-280, take forms of the
      fraction and its prefactor whose products and quotients stay in the double range;
      a shape below 1e-3 whose series exponent would be absorbed by the Stirling
      corrections takes it divided by the shape.

Limits:
    Accuracy is checked against 50-digit mpmath values (``validation/distributions.py``)
    for shapes from 1e-3 to 1e6, incomplete gamma shapes up to 1e18 above the center, and
    the t and F distributions with degrees of freedom from 0.5 to 1e12 and infinity:
    relative errors stay below 1e-12 down to probabilities of 1e-300. In deep tails they
    are of order eps |log p| (about 2e-13 at p = 1e-300), because a probability is
    evaluated through its logarithm. Larger shapes keep working: the uniform expansions
    serve the incomplete gamma function near the center for every shape from 1e3, and
    the incomplete beta function for two shapes from 1e10. The incomplete beta function
    is checked, too, at every combination of shapes and arguments from the smallest
    subnormal to the largest double (relative errors below 1e-12 there as well), and
    takes under a millisecond for any of them. A series or continued fraction that has
    not converged after 1e7 steps gives NaN rather than an inaccurate value.

References:
    Olver FWJ, Olde Daalhuis AB, Lozier DW, Schneider BI, Boisvert RF, Clark CW,
    Miller BR, Saunders BV, Cohl HS, McClain MA, eds (2026). NIST Digital Library of
    Mathematical Functions, release 1.2.8 of 2026-09-15. https://dlmf.nist.gov/

    Dekker TJ (1971). A floating-point technique for extending the available precision.
    Numerische Mathematik 18, 224-242. doi:10.1007/BF01397083

    Lentz WJ (1976). Generating Bessel functions in Mie scattering calculations using
    continued fractions. Applied Optics 15, 668-671. doi:10.1364/AO.15.000668

    Thompson IJ, Barnett AR (1986). Coulomb and Bessel functions of complex arguments
    and order. Journal of Computational Physics 64, 490-509.
    doi:10.1016/0021-9991(86)90046-X

    Loader C (2000). Fast and accurate computation of binomial probabilities. Technical
    note, Lucent Technologies, Murray Hill; the version of 5 February 2002 is available
    at https://www.r-project.org/doc/reports/CLoader-dbinom-2002.pdf

    Maechler M (2012). Accurately computing log(1 - exp(-|a|)) assessed by the Rmpfr
    package. Vignette of the R package Rmpfr, https://cran.r-project.org/package=Rmpfr

    Temme NM (1979). The asymptotic expansion of the incomplete gamma functions. SIAM
    Journal on Mathematical Analysis 10, 757-766. doi:10.1137/0510071

    Temme NM (1982). The uniform asymptotic expansion of a class of integrals related to
    cumulative distribution functions. SIAM Journal on Mathematical Analysis 13,
    239-253. doi:10.1137/0513017

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cytepool._jit import jit_numpy_errors
from cytepool.stats._scalar_special_functions import (
    EPS,
    INF,
    NAN,
    elementwise,
    log_beta_positive,
    log_betainc_pair_at,
    log_gammainc_pair,
)

__all__ = [
    "betainc",
    "betaincc",
    "digamma",
    "gammainc",
    "gammaincc",
    "lgamma",
    "log_beta",
    "log_betainc",
    "log_betaincc",
    "log_gammainc",
    "log_gammaincc",
    "trigamma",
    "trigamma_inverse",
]


_PI = math.pi
_PI_SQUARED = 9.869604401089358
_ZETA2 = 1.6449340668482264  # zeta(2) = pi^2 / 6

# The positive root of digamma, x0 = 1.46163214496836234126..., as a sum of two doubles.
_DIGAMMA_ROOT_HI = 1.4616321449683622
_DIGAMMA_ROOT_LO = 9.549995429965697e-17
# Taylor coefficients psi(x0 + h) = sum_{k>=1} c_k h^k with c_k = (-1)^(k+1) zeta(k+1, x0)
# (Hurwitz zeta), k = 1..40, computed with mpmath; 40 terms reach full accuracy on [1, 2].
_DIGAMMA_ROOT_TAYLOR = np.array(
    [
        0.9676722454476212,
        -0.4427631689835921,
        0.258499760955651,
        -0.16394270544240652,
        0.10782405069126237,
        -0.07219956125645471,
        0.04880428816414311,
        -0.03316112647484736,
        0.022597648232218104,
        -0.01542476590494896,
        0.010538791616612175,
        -0.007204534386356869,
        0.004926781395729853,
        -0.003369801655439328,
        0.002305126326734928,
        -0.0015769367714301972,
        0.0010788252019162967,
        -0.0007380709389960052,
        0.000504953265834602,
        -0.0003454680251063077,
        0.00023635601564027053,
        -0.00016170622091974803,
        0.0001106337276874741,
        -7.569179582195066e-05,
        5.178575795222081e-05,
        -3.5430070947659604e-05,
        2.424006611860132e-05,
        -1.6584242271854135e-05,
        1.134638458466385e-05,
        -7.762817668462094e-06,
        5.3110609208898636e-06,
        -3.6336507898010456e-06,
        2.486022733129538e-06,
        -1.7008538854332607e-06,
        1.1636675363548843e-06,
        -7.96142543124197e-07,
        5.446941930669446e-07,
        -3.7266161283438227e-07,
        2.549626552021554e-07,
        -1.7443695117727745e-07,
    ]
)


# ---------------------------------------------------------------------------------------
# Elementary helpers
# ---------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------
# Gamma function family
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def lgamma_scalar(x: float) -> float:
    """Return log|Gamma(x)|; +inf at the poles x = 0, -1, -2, ...; NaN at -inf."""
    if x == -INF:
        return NAN
    return math.lgamma(x)


@jit_numpy_errors
def _digamma_on_unit_interval(x: float) -> float:
    """Return psi(x) for 1 <= x <= 2 from the Taylor series about the positive root."""
    h = (x - _DIGAMMA_ROOT_HI) - _DIGAMMA_ROOT_LO
    c = _DIGAMMA_ROOT_TAYLOR
    s: float = c[c.size - 1]
    for k in range(c.size - 2, -1, -1):
        s = s * h + c[k]
    return s * h


@jit_numpy_errors
def _digamma_positive(x: float) -> float:
    """Return psi(x) for x > 0."""
    if x < 1.0:
        return _digamma_on_unit_interval(x + 1.0) - 1.0 / x
    if x <= 2.0:
        return _digamma_on_unit_interval(x)
    if x < 10.0:
        # psi(x) = psi(x - m) + sum_{j=1}^{m} 1/(x - j) with x - m in (1, 2]; the terms
        # are positive and added from the smallest.
        m = np.floor(x - 1.0)
        if x - m <= 1.0:
            m -= 1.0
        total = 0.0
        j = 1.0
        while j <= m:
            total += 1.0 / (x - j)
            j += 1.0
        return _digamma_on_unit_interval(x - m) + total
    # psi(x) = log x - 1/(2x) - sum_k B_2k / (2k x^2k) (DLMF 5.11.2)
    r2 = 1.0 / (x * x)
    tail = r2 * (
        1.0 / 12.0
        - r2
        * (
            1.0 / 120.0
            - r2
            * (
                1.0 / 252.0
                - r2
                * (
                    1.0 / 240.0
                    - r2
                    * (
                        1.0 / 132.0
                        - r2
                        * (691.0 / 32760.0 - r2 * (1.0 / 12.0 - r2 * (3617.0 / 8160.0)))
                    )
                )
            )
        )
    )
    return math.log(x) - 0.5 / x - tail


@jit_numpy_errors
def _sin_pi(x: float) -> float:
    """Return sin(pi r) with r = x - round(x), which has the same magnitude as sin(pi x)."""
    return math.sin(_PI * (x - np.floor(x + 0.5)))


@jit_numpy_errors
def digamma_scalar(x: float) -> float:
    """Return psi(x); NaN at the poles x = 0, -1, -2, ... and at -inf."""
    if math.isnan(x) or x == -INF:
        return NAN
    if x == INF:
        return INF
    if x <= 0.0:
        if x == np.floor(x):
            return NAN
        # psi(x) = psi(1 - x) - pi cot(pi x) (DLMF 5.5.4) with cot(pi x) = cot(pi r),
        # r = x - round(x) in [-1/2, 1/2]; cos(pi r) is formed as sin(pi (1/2 - |r|)) so
        # that it is exactly 0 at r = +-1/2.
        r = x - np.floor(x + 0.5)
        cot = math.sin(_PI * (0.5 - abs(r))) / math.sin(_PI * r)
        return _digamma_positive(1.0 - x) - _PI * cot
    return _digamma_positive(x)


@jit_numpy_errors
def _trigamma_positive(x: float) -> float:
    """Return psi'(x) for x > 0."""
    total = 0.0
    shift = 0.0
    while x + shift < 12.0:
        t = x + shift
        total += 1.0 / (t * t)
        shift += 1.0
    t = x + shift
    r = 1.0 / t
    r2 = r * r
    # psi'(x) = 1/x + 1/(2x^2) + sum_k B_2k / x^(2k+1) (DLMF 5.15.8)
    tail = (
        r
        * r2
        * (
            1.0 / 6.0
            - r2
            * (
                1.0 / 30.0
                - r2
                * (
                    1.0 / 42.0
                    - r2
                    * (
                        1.0 / 30.0
                        - r2
                        * (
                            5.0 / 66.0
                            - r2
                            * (
                                691.0 / 2730.0
                                - r2
                                * (
                                    7.0 / 6.0
                                    - r2 * (3617.0 / 510.0 - r2 * (43867.0 / 798.0))
                                )
                            )
                        )
                    )
                )
            )
        )
    )
    return total + (r + 0.5 * r2 + tail)


@jit_numpy_errors
def trigamma_scalar(x: float) -> float:
    """Return psi'(x); +inf at the poles x = 0, -1, -2, ...; NaN at -inf."""
    if math.isnan(x) or x == -INF:
        return NAN
    if x == INF:
        return 0.0
    if x <= 0.0:
        if x == np.floor(x):
            return INF
        # psi'(x) = pi^2 / sin^2(pi x) - psi'(1 - x) (DLMF 5.15.6)
        s = _sin_pi(x)
        return _PI_SQUARED / (s * s) - _trigamma_positive(1.0 - x)
    return _trigamma_positive(x)


@jit_numpy_errors
def _tetragamma_positive(x: float) -> float:
    """Return psi''(x) for x > 0."""
    total = 0.0
    shift = 0.0
    while x + shift < 12.0:
        t = x + shift
        total -= 2.0 / (t * t * t)
        shift += 1.0
    t = x + shift
    r = 1.0 / t
    r2 = r * r
    # psi''(x) = -1/x^2 - 1/x^3 - sum_k (2k + 1) B_2k / x^(2k+2)
    tail = (
        r2
        * r2
        * (
            0.5
            - r2
            * (
                1.0 / 6.0
                - r2
                * (
                    1.0 / 6.0
                    - r2 * (0.3 - r2 * (5.0 / 6.0 - r2 * (691.0 / 210.0 - r2 * 17.5)))
                )
            )
        )
    )
    return total - (r2 + r2 * r + tail)


@jit_numpy_errors
def trigamma_inverse_scalar(y: float) -> float:
    """Return the x > 0 with trigamma(x) = y, for y > 0; +inf at y = 0, 0 at y = inf."""
    if math.isnan(y) or y < 0.0:
        return NAN
    if y == 0.0:
        return INF
    if y == INF:
        return 0.0
    if y < 1e-4:
        # Inverting psi'(x) = 1/x + 1/(2x^2) + 1/(6x^3) - ... gives
        # x = 1/y + 1/2 - y/12 + 11 y^3/720 + ...; the omitted term is below 1e-18 x.
        return 1.0 / y + 0.5 - y / 12.0
    if y > 1e32:
        # psi'(x) = 1/x^2 + zeta(2) + O(x); the correction is below 1e-31 relative.
        return 1.0 / math.sqrt(y)
    # Newton's method on f(x) = 1/psi'(x) - 1/y (Smyth 2004). f is increasing and convex,
    # and both starting values have f > 0: psi'(x) < 1/(x - 1/2) gives f(1/2 + 1/y) > 0,
    # and psi'(x) < 1/x^2 + zeta(2) gives f((y - zeta(2))^(-1/2)) > 0. The iterates
    # therefore decrease monotonically to the root.
    if y > 10.0:
        x = 1.0 / math.sqrt(y - _ZETA2)
    else:
        x = 0.5 + 1.0 / y
    for _ in range(100):
        tri = _trigamma_positive(x)
        step = tri * (1.0 - tri / y) / _tetragamma_positive(x)
        x += step
        if -step <= 4.0 * EPS * x:
            break
    return x


@jit_numpy_errors
def log_beta_scalar(a: float, b: float) -> float:
    """Return log B(a, b) for a, b >= 0; +inf when a or b is 0."""
    if math.isnan(a) or math.isnan(b) or a < 0.0 or b < 0.0:
        return NAN
    if a == 0.0 or b == 0.0:
        return NAN if a == INF or b == INF else INF
    if a == INF or b == INF:
        return -INF
    return log_beta_positive(a, b)


# ---------------------------------------------------------------------------------------
# Regularized incomplete gamma function
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _log_gammainc_checked(a: float, x: float) -> tuple[float, float]:
    """Return log P(a, x) and log Q(a, x), or NaN for arguments outside the domain."""
    if math.isnan(a) or math.isnan(x) or not (0.0 < a < INF) or x < 0.0:
        return NAN, NAN
    return log_gammainc_pair(a, x, math.log(x), a - x)


@jit_numpy_errors
def gammainc_scalar(a: float, x: float) -> float:
    """Return the regularized lower incomplete gamma function P(a, x)."""
    return math.exp(_log_gammainc_checked(a, x)[0])


@jit_numpy_errors
def gammaincc_scalar(a: float, x: float) -> float:
    """Return the regularized upper incomplete gamma function Q(a, x)."""
    return math.exp(_log_gammainc_checked(a, x)[1])


@jit_numpy_errors
def log_gammainc_scalar(a: float, x: float) -> float:
    """Return log P(a, x)."""
    return _log_gammainc_checked(a, x)[0]


@jit_numpy_errors
def log_gammaincc_scalar(a: float, x: float) -> float:
    """Return log Q(a, x)."""
    return _log_gammainc_checked(a, x)[1]


# ---------------------------------------------------------------------------------------
# Regularized incomplete beta function
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _log_betainc_checked(a: float, b: float, x: float) -> tuple[float, float]:
    """Return log I_x(a, b) and its complement, or NaN for arguments outside the domain."""
    if (
        math.isnan(a)
        or math.isnan(b)
        or math.isnan(x)
        or not (0.0 < a < INF)
        or not (0.0 < b < INF)
        or not (0.0 <= x <= 1.0)
    ):
        return NAN, NAN
    return log_betainc_pair_at(a, b, x)


@jit_numpy_errors
def betainc_scalar(a: float, b: float, x: float) -> float:
    """Return the regularized incomplete beta function I_x(a, b)."""
    return math.exp(_log_betainc_checked(a, b, x)[0])


@jit_numpy_errors
def betaincc_scalar(a: float, b: float, x: float) -> float:
    """Return 1 - I_x(a, b)."""
    return math.exp(_log_betainc_checked(a, b, x)[1])


@jit_numpy_errors
def log_betainc_scalar(a: float, b: float, x: float) -> float:
    """Return log I_x(a, b)."""
    return _log_betainc_checked(a, b, x)[0]


@jit_numpy_errors
def log_betaincc_scalar(a: float, b: float, x: float) -> float:
    """Return log(1 - I_x(a, b))."""
    return _log_betainc_checked(a, b, x)[1]


# ---------------------------------------------------------------------------------------
# Array loops and the vectorised public functions
# ---------------------------------------------------------------------------------------

_LGAMMA, _DIGAMMA, _TRIGAMMA, _TRIGAMMA_INVERSE = 0, 1, 2, 3
_LOWER, _UPPER, _LOG_LOWER, _LOG_UPPER = 0, 1, 2, 3


@jit_numpy_errors
def _gamma_family_array(x: NDArray[np.float64], kind: int) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        if kind == _LGAMMA:
            out[i] = lgamma_scalar(x[i])
        elif kind == _DIGAMMA:
            out[i] = digamma_scalar(x[i])
        elif kind == _TRIGAMMA:
            out[i] = trigamma_scalar(x[i])
        else:
            out[i] = trigamma_inverse_scalar(x[i])
    return out


@jit_numpy_errors
def _log_beta_array(
    a: NDArray[np.float64], b: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(a.size)
    for i in range(a.size):
        out[i] = log_beta_scalar(a[i], b[i])
    return out


@jit_numpy_errors
def _gammainc_array(
    a: NDArray[np.float64], x: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(a.size)
    for i in range(a.size):
        log_p, log_q = _log_gammainc_checked(a[i], x[i])
        if kind == _LOWER:
            out[i] = math.exp(log_p)
        elif kind == _UPPER:
            out[i] = math.exp(log_q)
        elif kind == _LOG_LOWER:
            out[i] = log_p
        else:
            out[i] = log_q
    return out


@jit_numpy_errors
def _betainc_array(
    a: NDArray[np.float64], b: NDArray[np.float64], x: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(a.size)
    for i in range(a.size):
        log_i, log_c = _log_betainc_checked(a[i], b[i], x[i])
        if kind == _LOWER:
            out[i] = math.exp(log_i)
        elif kind == _UPPER:
            out[i] = math.exp(log_c)
        elif kind == _LOG_LOWER:
            out[i] = log_i
        else:
            out[i] = log_c
    return out


def lgamma(x: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the absolute value of the gamma function, log|Gamma(x)|.

    Args:
        x: Real argument. At the poles x = 0, -1, -2, ... the result is +inf; at
            x = -inf it is NaN.

    Returns:
        log|Gamma(x)|, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return elementwise(_gamma_family_array, _LGAMMA, x=x)


def digamma(x: ArrayLike) -> float | NDArray[np.float64]:
    """Digamma function psi(x) = d/dx log Gamma(x).

    Args:
        x: Real argument. The poles x = 0, -1, -2, ..., where psi has no limit, and
            x = -inf give NaN; x = +inf gives +inf.

    Returns:
        psi(x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return elementwise(_gamma_family_array, _DIGAMMA, x=x)


def trigamma(x: ArrayLike) -> float | NDArray[np.float64]:
    """Trigamma function psi'(x), the second derivative of log Gamma(x).

    Args:
        x: Real argument. The poles x = 0, -1, -2, ... give +inf (the limit from both
            sides), x = +inf gives 0 and x = -inf gives NaN.

    Returns:
        psi'(x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return elementwise(_gamma_family_array, _TRIGAMMA, x=x)


def trigamma_inverse(y: ArrayLike) -> float | NDArray[np.float64]:
    """Inverse of the trigamma function on x > 0: the x with psi'(x) = y.

    Solved by Newton's method to full double precision (Smyth 2004).

    Args:
        y: Value of the trigamma function, y > 0. y = 0 gives +inf, y = +inf gives 0
            and y < 0 gives NaN.

    Returns:
        x > 0 with trigamma(x) = y, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``y`` is not real-valued.
    """
    return elementwise(_gamma_family_array, _TRIGAMMA_INVERSE, y=y)


def log_beta(a: ArrayLike, b: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the beta function, log B(a, b).

    log B(a, b) = log Gamma(a) + log Gamma(b) - log Gamma(a + b), computed without the
    cancellation of the three log-gamma values for large arguments.

    Args:
        a: First parameter, a >= 0.
        b: Second parameter, b >= 0. If a or b is 0 the result is +inf (NaN if the other
            is +inf); if one is +inf and the other positive it is -inf. Negative values
            give NaN.

    Returns:
        log B(a, b), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_log_beta_array, 0, a=a, b=b)


def gammainc(a: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Regularized lower incomplete gamma function P(a, x) = gamma(a, x) / Gamma(a).

    Args:
        a: Shape, 0 < a < inf.
        x: Argument, x >= 0 (x = +inf gives 1).

    Returns:
        P(a, x), as a float or a float64 array of the broadcast shape. Arguments outside
        the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_gammainc_array, _LOWER, a=a, x=x)


def gammaincc(a: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).

    Computed directly, so it keeps full relative accuracy when it is small.

    Args:
        a: Shape, 0 < a < inf.
        x: Argument, x >= 0 (x = +inf gives 0).

    Returns:
        Q(a, x), as a float or a float64 array of the broadcast shape. Arguments outside
        the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_gammainc_array, _UPPER, a=a, x=x)


def log_gammainc(a: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the regularized lower incomplete gamma function, log P(a, x).

    Finite far beyond the point where P(a, x) underflows.

    Args:
        a: Shape, 0 < a < inf.
        x: Argument, x >= 0 (x = 0 gives -inf).

    Returns:
        log P(a, x), as a float or a float64 array of the broadcast shape. Arguments
        outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_gammainc_array, _LOG_LOWER, a=a, x=x)


def log_gammaincc(a: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the regularized upper incomplete gamma function, log Q(a, x).

    Finite far beyond the point where Q(a, x) underflows.

    Args:
        a: Shape, 0 < a < inf.
        x: Argument, x >= 0 (x = +inf gives -inf).

    Returns:
        log Q(a, x), as a float or a float64 array of the broadcast shape. Arguments
        outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_gammainc_array, _LOG_UPPER, a=a, x=x)


def betainc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Regularized incomplete beta function I_x(a, b).

    Args:
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.
        x: Argument, 0 <= x <= 1.

    Returns:
        I_x(a, b), as a float or a float64 array of the broadcast shape. Arguments
        outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_betainc_array, _LOWER, a=a, b=b, x=x)


def betaincc(a: ArrayLike, b: ArrayLike, x: ArrayLike) -> float | NDArray[np.float64]:
    """Complement of the regularized incomplete beta function, 1 - I_x(a, b).

    Equal to I_{1-x}(b, a) and computed directly, so it keeps full relative accuracy
    when it is small.

    Args:
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.
        x: Argument, 0 <= x <= 1.

    Returns:
        1 - I_x(a, b), as a float or a float64 array of the broadcast shape. Arguments
        outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_betainc_array, _UPPER, a=a, b=b, x=x)


def log_betainc(
    a: ArrayLike, b: ArrayLike, x: ArrayLike
) -> float | NDArray[np.float64]:
    """Logarithm of the regularized incomplete beta function, log I_x(a, b).

    Args:
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.
        x: Argument, 0 <= x <= 1 (x = 0 gives -inf).

    Returns:
        log I_x(a, b), as a float or a float64 array of the broadcast shape. Arguments
        outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_betainc_array, _LOG_LOWER, a=a, b=b, x=x)


def log_betaincc(
    a: ArrayLike, b: ArrayLike, x: ArrayLike
) -> float | NDArray[np.float64]:
    """Logarithm of the complement of the incomplete beta function, log(1 - I_x(a, b)).

    Args:
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.
        x: Argument, 0 <= x <= 1 (x = 1 gives -inf).

    Returns:
        log(1 - I_x(a, b)), as a float or a float64 array of the broadcast shape.
        Arguments outside the domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return elementwise(_betainc_array, _LOG_UPPER, a=a, b=b, x=x)
