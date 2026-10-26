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

The compiled scalar routines behind the public functions are named ``<function>_scalar``
and can be called from other compiled code.

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
      incomplete gamma case).

Limits:
    Accuracy is checked against 50-digit mpmath values (``validation/distributions.py``)
    for shapes from 1e-3 to 1e6, incomplete gamma shapes up to 1e18 above the center, and
    the t and F distributions with degrees of freedom from 0.5 to 1e12 and infinity:
    relative errors stay below 1e-12 down to probabilities of 1e-300. In deep tails they
    are of order eps |log p| (about 2e-13 at p = 1e-300), because a probability is
    evaluated through its logarithm. Larger shapes keep working: the incomplete beta
    fraction needs a few thousand steps at shapes of 1e12, and the uniform expansion
    serves the incomplete gamma function near the center for every shape from 1e3. A
    series or continued fraction that has not converged after 1e7 steps gives NaN rather
    than an inaccurate value.

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

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cytepool._jit import jit_numpy_errors
from cytepool._masked_arrays import unmasked_array

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


_INF = math.inf
_NAN = math.nan
_PI = math.pi
_PI_SQUARED = 9.869604401089358
_LN2 = 0.6931471805599453
_LN_SQRT_2PI = 0.9189385332046728  # log(sqrt(2 pi))
_EULER = 0.5772156649015329
_ZETA2 = 1.6449340668482264  # zeta(2) = pi^2 / 6
_EPS = 2.220446049250313e-16  # spacing of the doubles in [1, 2)
_SMALLEST_NORMAL = 2.2250738585072014e-308
# The modified Lentz method stops when an update changes the value by about one unit
# in the last place; this is a little above the spacing of the doubles just below 1.
_CF_TOLERANCE = 3e-16
_TINY = 1e-300  # replaces a zero denominator in the modified Lentz method
_MAX_ITERATIONS = 10_000_000

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

# log Gamma(1 + a) = -euler a + sum_{k>=2} c_k a^k with c_k = (-1)^k zeta(k) / k
# (DLMF 5.7.3), k = 2..30; used for |a| <= 1/4.
_LGAMMA1P_TAYLOR = np.array(
    [
        0.8224670334241132,
        -0.40068563438653143,
        0.27058080842778454,
        -0.20738555102867398,
        0.1695571769974082,
        -0.1440498967688461,
        0.12550966952474304,
        -0.11133426586956469,
        0.1000994575127818,
        -0.09095401714582904,
        0.083353840546109,
        -0.0769325164113522,
        0.07143294629536133,
        -0.06666870588242046,
        0.06250095514121304,
        -0.058823978658684585,
        0.055555767627403614,
        -0.05263167937961666,
        0.05000004769810169,
        -0.047619070330142226,
        0.04545455629320467,
        -0.04347826605304026,
        0.04166666915034121,
        -0.04000000119214014,
        0.03846153903467518,
        -0.037037037312989324,
        0.035714285847333355,
        -0.034482758684919304,
        0.03333333336437758,
    ]
)

# Stirling's error delta(n) = log Gamma(n) - (n - 1/2) log n + n - log sqrt(2 pi) at the
# integers n = 1..9 (index n; index 0 is unused), computed with mpmath.
_STIRLING_ERROR_AT_INTEGERS = np.array(
    [
        _NAN,
        0.08106146679532726,
        0.0413406959554093,
        0.02767792568499834,
        0.020790672103765093,
        0.016644691189821193,
        0.013876128823070748,
        0.01189670994589177,
        0.010411265261972096,
        0.009255462182712733,
    ]
)


# Temme's uniform expansion of the incomplete gamma function (DLMF 8.12.3, 8.12.8):
# Q(a, x) = erfc(eta sqrt(a/2)) / 2 + exp(-a eta^2 / 2) / sqrt(2 pi a) sum_k c_k(eta) a^-k,
# with eta^2 / 2 = lambda - 1 - log(lambda), lambda = x / a and eta of the sign of x - a.
# Rows are the Taylor coefficients of c_0 .. c_4 in eta, degrees 0..20, derived exactly
# with rational power series from c_0 = 1/(lambda - 1) - 1/eta and the recurrence
# c_k = c'_{k-1} / eta + (-1)^k gamma_k / (lambda - 1) (DLMF 8.12), gamma_k being the
# Stirling coefficients of Gamma*(a). For a >= 1e3 and |eta| <= 1/2 the omitted terms are
# below 1e-19 relative (checked against mpmath).
_UNIFORM_SHAPE = 1e3
_UNIFORM_COEFFICIENTS = np.array(
    [
        [
            -0.3333333333333333,
            0.08333333333333333,
            -0.014814814814814815,
            0.0011574074074074073,
            0.0003527336860670194,
            -0.0001787551440329218,
            3.919263178522438e-05,
            -2.185448510679992e-06,
            -1.85406221071516e-06,
            8.296711340953087e-07,
            -1.7665952736826078e-07,
            6.707853543401498e-09,
            1.0261809784240309e-08,
            -4.382036018453353e-09,
            9.14769958223679e-10,
            -2.5514193994946248e-11,
            -5.830772132550426e-11,
            2.4361948020667415e-11,
            -5.0276692801141755e-12,
            1.1004392031956135e-13,
            3.371763262400985e-13,
        ],
        [
            -0.001851851851851852,
            -0.003472222222222222,
            0.0026455026455026454,
            -0.0009902263374485596,
            0.00020576131687242798,
            -4.018775720164609e-07,
            -1.8098550334489977e-05,
            7.64916091608111e-06,
            -1.6120900894563446e-06,
            4.647127802807434e-09,
            1.378633446915721e-07,
            -5.752545603517705e-08,
            1.1951628599778148e-08,
            -1.7543241719747647e-11,
            -1.0091543710600413e-09,
            4.162792991842583e-10,
            -8.56390702649298e-11,
            6.067215101604758e-14,
            7.1624989648114856e-12,
            -2.933186643771437e-12,
            5.996696365683689e-13,
        ],
        [
            0.004133597883597883,
            -0.0026813271604938273,
            0.0007716049382716049,
            2.0093878600823047e-06,
            -0.0001073665322636516,
            5.2923448829120125e-05,
            -1.2760635188618728e-05,
            3.423578734096138e-08,
            1.3721957309062934e-06,
            -6.298992138380055e-07,
            1.4280614206064242e-07,
            -2.0477098421990866e-10,
            -1.409252991086752e-08,
            6.228974084922022e-09,
            -1.3670488396617114e-09,
            9.428356159014678e-13,
            1.2872252400089318e-10,
            -5.5645956134363323e-11,
            1.197593554636698e-11,
            -4.1689782251838634e-15,
            -1.0940640427884595e-12,
        ],
        [
            0.0006494341563786008,
            0.00022947209362139917,
            -0.0004691894943952557,
            0.00026772063206283885,
            -7.561801671883977e-05,
            -2.396505113867297e-07,
            1.1082654115347302e-05,
            -5.6749528269915965e-06,
            1.4230900732435883e-06,
            -2.7861080291528143e-11,
            -1.6958404091930278e-07,
            8.099464905388083e-08,
            -1.9111168485973655e-08,
            2.3928620439808118e-12,
            2.0620131815488797e-09,
            -9.460496661855133e-10,
            2.1541049775774907e-10,
            -1.388823336813903e-14,
            -2.1894761681963938e-11,
            9.790998951171684e-12,
            -2.178219188018096e-12,
        ],
        [
            -0.0008618882909167117,
            0.0007840392217200666,
            -0.0002990724803031902,
            -1.4638452578843418e-06,
            6.641498215465122e-05,
            -3.968365047179435e-05,
            1.1375726970678419e-05,
            2.507497226237533e-10,
            -1.6954149536558305e-06,
            8.907507532205309e-07,
            -2.292934834000805e-07,
            2.956794137544049e-11,
            2.8865829742708783e-08,
            -1.4189739437803219e-08,
            3.4463580499464896e-09,
            -2.3024517174528067e-13,
            -3.9409233028046403e-10,
            1.86023389685045e-10,
            -4.356323005056618e-11,
            1.278600101629623e-15,
            4.67927502665792e-12,
        ],
    ]
)
_SQRT_2_OVER_PI = 0.7978845608028654  # sqrt(2 / pi)
_SQRT2 = 1.4142135623730951
_INV_SQRT_2PI = 0.3989422804014327  # 1 / sqrt(2 pi)


# ---------------------------------------------------------------------------------------
# Elementary helpers
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _two_product(a: float, b: float) -> tuple[float, float]:
    """Return p = fl(a b) and e with a b = p + e exactly (Dekker 1971)."""
    p = a * b
    if not (abs(a) < 1e290 and abs(b) < 1e290):
        return p, 0.0
    split = 134217729.0  # 2^27 + 1
    t = split * a
    a_hi = t - (t - a)
    a_lo = a - a_hi
    t = split * b
    b_hi = t - (t - b)
    b_lo = b - b_hi
    return p, ((a_hi * b_hi - p) + a_hi * b_lo + a_lo * b_hi) + a_lo * b_lo


@jit_numpy_errors
def _log1mexp(a: float) -> float:
    """Return log(1 - exp(-a)) for a >= 0 without cancellation (Maechler 2012)."""
    if a <= _LN2:
        return math.log(-math.expm1(-a))
    return math.log1p(-math.exp(-a))


@jit_numpy_errors
def _log_product(c: float, v: float, log_v: float) -> float:
    """Return log(c v) for c > 0, v >= 0, given log_v = log(v) formed accurately.

    The product is used only when both it and v are normal doubles; a subnormal v has
    lost bits that log_v still carries.
    """
    p = c * v
    if v >= _SMALLEST_NORMAL and _SMALLEST_NORMAL <= p < _INF:
        return math.log(p)
    return math.log(c) + log_v


@jit_numpy_errors
def _erfcx(u: float) -> float:
    """Return exp(u^2) erfc(u) for u >= 0, with full relative accuracy."""
    if u < 26.0:
        # exp(u^2) with the square formed exactly (Dekker), times the C library's erfc
        p, e = _two_product(u, u)
        return math.exp(p) * (1.0 + e) * math.erfc(u)
    # exp(u^2) erfc(u) = sqrt(2/pi) / t with Laplace's continued fraction for Mills'
    # ratio, t = z + 1/(z + 2/(z + ...)) at z = u sqrt 2 (DLMF 7.9.2)
    z = u * _SQRT2
    t = z
    for k in range(40, 0, -1):
        t = z + k / t
    return _SQRT_2_OVER_PI / t


@jit_numpy_errors
def _stirling_error(z: float) -> float:
    """Return delta(z) = log Gamma(z) - (z - 1/2) log z + z - log sqrt(2 pi), z > 0."""
    if z >= 10.0:
        # delta(z) = sum_k B_2k / (2k (2k - 1) z^(2k - 1)) (DLMF 5.11.1), ten terms.
        r = 1.0 / z
        r2 = r * r
        return r * (
            1.0 / 12.0
            + r2
            * (
                -1.0 / 360.0
                + r2
                * (
                    1.0 / 1260.0
                    + r2
                    * (
                        -1.0 / 1680.0
                        + r2
                        * (
                            1.0 / 1188.0
                            + r2
                            * (
                                -691.0 / 360360.0
                                + r2
                                * (
                                    1.0 / 156.0
                                    + r2
                                    * (
                                        -3617.0 / 122400.0
                                        + r2
                                        * (
                                            43867.0 / 244188.0
                                            - r2 * (174611.0 / 125400.0)
                                        )
                                    )
                                )
                            )
                        )
                    )
                )
            )
        )
    if z == np.floor(z):
        return float(_STIRLING_ERROR_AT_INTEGERS[int(z)])
    return math.lgamma(z) - (z - 0.5) * math.log(z) + z - _LN_SQRT_2PI


@jit_numpy_errors
def _deviance_term(x: float, m: float, d: float) -> float:
    """Return Loader's bd0(x, m) = x log(x / m) + m - x, given d = x - m.

    The caller forms d accurately; near x = m the series in v = d / (x + m) avoids the
    cancellation of the direct form (Loader 2000).
    """
    if x == 0.0:
        return m
    s = x + m
    if abs(d) < 0.5 * s:
        v = d / s
        v2 = v * v
        total = d * v
        term = 2.0 * x * v
        j = 1.0
        while True:  # |v| < 1/2: the terms fall by a factor of four or more
            term *= v2
            updated = total + term / (2.0 * j + 1.0)
            if updated == total:
                return total
            total = updated
            j += 1.0
    return x * math.log(x / m) - d


@jit_numpy_errors
def _lgamma1p(a: float) -> float:
    """Return log Gamma(1 + a) for a > -1, with full relative accuracy near a = 0."""
    if abs(a) <= 0.25:
        c = _LGAMMA1P_TAYLOR
        s: float = c[c.size - 1]
        for k in range(c.size - 2, -1, -1):
            s = s * a + c[k]
        return a * (a * s - _EULER)
    return math.lgamma(1.0 + a)


@jit_numpy_errors
def _log_gamma_ratio(a: float, b: float) -> float:
    """Return log Gamma(a + b) - log Gamma(a) for a >= 10, b >= 0, without cancellation."""
    return (
        b * math.log(a)
        + (a + b - 0.5) * math.log1p(b / a)
        - b
        + _stirling_error(a + b)
        - _stirling_error(a)
    )


@jit_numpy_errors
def _log_gamma_increment(b: float, a: float) -> float:
    """Return log Gamma(b + a) - log Gamma(b) for b > 0 and a >= 0, accurate for small a.

    Below 10, b is shifted up with log Gamma(z + 1) = log Gamma(z) + log z, so the result
    is a sum of terms of one sign instead of a difference of two log-gamma values.
    """
    if b >= 10.0:
        return _log_gamma_ratio(b, a)
    total = 0.0
    shift = 0.0
    while b + shift < 10.0:
        total += math.log1p(a / (b + shift))
        shift += 1.0
    return _log_gamma_ratio(b + shift, a) - total


@jit_numpy_errors
def _log_beta_positive(a: float, b: float) -> float:
    """Return log B(a, b) for finite a, b > 0."""
    if a < b:
        a, b = b, a
    if b >= 10.0:
        # Stirling's series for the three gamma functions, with the logarithms combined
        # so that every term has the same sign.
        n = a + b  # may overflow; delta(inf) = 0 and a b / n is formed without it
        return (
            _LN_SQRT_2PI
            - 0.5 * math.log(b / (1.0 + b / a))
            - a * math.log1p(b / a)
            - b * math.log1p(a / b)
            + _stirling_error(a)
            + _stirling_error(b)
            - _stirling_error(n)
        )
    # log Gamma(b) - [log Gamma(a + b) - log Gamma(a)], with the bracket formed without
    # subtracting log-gamma values
    return math.lgamma(b) - _log_gamma_increment(a, b)


# ---------------------------------------------------------------------------------------
# Gamma function family
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def lgamma_scalar(x: float) -> float:
    """Return log|Gamma(x)|; +inf at the poles x = 0, -1, -2, ...; NaN at -inf."""
    if x == -_INF:
        return _NAN
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
    if math.isnan(x) or x == -_INF:
        return _NAN
    if x == _INF:
        return _INF
    if x <= 0.0:
        if x == np.floor(x):
            return _NAN
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
    if math.isnan(x) or x == -_INF:
        return _NAN
    if x == _INF:
        return 0.0
    if x <= 0.0:
        if x == np.floor(x):
            return _INF
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
        return _NAN
    if y == 0.0:
        return _INF
    if y == _INF:
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
        if -step <= 4.0 * _EPS * x:
            break
    return x


@jit_numpy_errors
def log_beta_scalar(a: float, b: float) -> float:
    """Return log B(a, b) for a, b >= 0; +inf when a or b is 0."""
    if math.isnan(a) or math.isnan(b) or a < 0.0 or b < 0.0:
        return _NAN
    if a == 0.0 or b == 0.0:
        return _NAN if a == _INF or b == _INF else _INF
    if a == _INF or b == _INF:
        return -_INF
    return _log_beta_positive(a, b)


# ---------------------------------------------------------------------------------------
# Regularized incomplete gamma function
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _log_gamma_prefactor(a: float, x: float, log_x: float, d: float) -> float:
    """Return log(x^a exp(-x) / Gamma(a + 1)), given log_x = log(x) and d = a - x."""
    if a >= 10.0:
        # -bd0(a, x) - delta(a) - log sqrt(2 pi a), with Loader's deviance term
        if abs(d) < 0.5 * (a + x):
            deviance = _deviance_term(a, x, d)
        else:
            deviance = a * (math.log(a) - log_x) - d
        return -deviance - _stirling_error(a) - _LN_SQRT_2PI - 0.5 * math.log(a)
    return a * log_x - x - _lgamma1p(a)


@jit_numpy_errors
def _gamma_series(a: float, x: float, d: float) -> float:
    """Return sum_{n>=0} x^n / ((a + 1) ... (a + n)), so P(a, x) = prefactor * sum."""
    total = 1.0
    term = 1.0
    n = 1.0
    while n < _MAX_ITERATIONS:
        term *= x / (a + n)
        total += term
        # The remaining terms fall at least geometrically with ratio x / (a + n + 1), so
        # their sum is below term * x / (a + n + 1 - x).
        gap = (n + 1.0) + d
        if gap > 0.0 and term * x <= 0.25 * _EPS * total * gap:
            return total
        n += 1.0
    return _NAN


@jit_numpy_errors
def _gamma_continued_fraction(a: float, d: float) -> float:
    """Return K with Gamma(a, x) = x^a exp(-x) / K, given d = a - x < -1.

    K = (x + 1 - a) - 1 (1 - a) / ((x + 3 - a) - 2 (2 - a) / ((x + 5 - a) - ...)) is
    Legendre's continued fraction (DLMF 8.9.2), evaluated by the modified Lentz method.
    """
    b0 = 1.0 - d  # x + 1 - a > 2
    f = b0
    c = b0
    e = 0.0
    j = 1.0
    while j < _MAX_ITERATIONS:
        num = j * (a - j)
        den = b0 + 2.0 * j
        e = den + num * e
        if e == 0.0:
            e = _TINY
        c = den + num / c
        if c == 0.0:
            c = _TINY
        e = 1.0 / e
        delta = c * e
        f *= delta
        if abs(delta - 1.0) < _CF_TOLERANCE:
            return f
        j += 1.0
    return _NAN


@jit_numpy_errors
def _gamma_upper_small_shape(a: float, x: float, log_x: float) -> float:
    """Return Q(a, x) for 0 < a < 1 and 0 < x <= 1.5 without forming 1 - P.

    From the alternating series of the lower function (DLMF 8.7.1),
    Q = -expm1(t) + exp(t) a sum_{n>=1} (-1)^(n+1) x^n / (n! (a + n)) with
    t = a log x - log Gamma(1 + a). For x <= 1.5 the two parts cancel by at most a
    factor of about ten.
    """
    t = a * log_x - _lgamma1p(a)
    total = 0.0
    power = 1.0  # x^n / n!
    sign = 1.0
    n = 1.0
    while n < 200.0:
        power *= x / n
        term = sign * power / (a + n)
        total += term
        if abs(term) <= 0.25 * _EPS * abs(total):
            break
        sign = -sign
        n += 1.0
    return -math.expm1(t) + math.exp(t) * a * total


@jit_numpy_errors
def _log_gammainc_uniform(a: float, d: float, deviance: float) -> tuple[float, float]:
    """Return log P(a, x) and log Q(a, x) for a >= 1e3 with |eta| <= 1/2, given d = a - x.

    Temme's uniform expansion (DLMF 8.12.3, 8.12.4, 8.12.8). With u = eta sqrt(a/2) and
    a eta^2 / 2 = bd0(a, x) = deviance, the tail on the side of x is
    exp(-deviance) [erfcx(|u|) / 2 + sign(eta) S / sqrt(2 pi a)], S = sum_k c_k(eta) a^-k,
    which avoids underflow; the two parts do not cancel there.
    """
    eta = math.sqrt(2.0 * deviance / a)
    if d > 0.0:  # x < a
        eta = -eta
    c = _UNIFORM_COEFFICIENTS
    inverse = 1.0 / a
    total = 0.0
    for k in range(c.shape[0] - 1, -1, -1):
        term: float = c[k, c.shape[1] - 1]
        for n in range(c.shape[1] - 2, -1, -1):
            term = term * eta + c[k, n]
        total = total * inverse + term
    scaled = _erfcx(math.sqrt(deviance))
    if d < 0.0:  # x > a: the upper tail is the smaller one
        log_q = -deviance + math.log(
            0.5 * scaled + total * _INV_SQRT_2PI / math.sqrt(a)
        )
        log_q = min(log_q, 0.0)
        return _log1mexp(-log_q), log_q
    log_p = -deviance + math.log(0.5 * scaled - total * _INV_SQRT_2PI / math.sqrt(a))
    log_p = min(log_p, 0.0)
    return log_p, _log1mexp(-log_p)


@jit_numpy_errors
def _log_gammainc_pair(
    a: float, x: float, log_x: float, d: float
) -> tuple[float, float]:
    """Return log P(a, x) and log Q(a, x) for finite a > 0 and x >= 0.

    The caller supplies log_x = log(x) and d = a - x, each formed accurately from its own
    inputs (for example without rounding a scaled argument first); d sets the accuracy of
    the prefactor for large a. x may underflow to 0 while log_x stays finite.
    """
    if log_x == -_INF:
        return -_INF, 0.0
    if x == _INF:
        return 0.0, -_INF
    if a >= _UNIFORM_SHAPE and abs(d) < 0.5 * (a + x):
        # Near the center the series and the continued fraction need many steps for a
        # large shape (about 8 sqrt(a) for the series); the uniform expansion does not.
        deviance = _deviance_term(a, x, d)
        if deviance <= 0.125 * a:  # |eta| <= 1/2
            return _log_gammainc_uniform(a, d, deviance)
    if a < 1.0 and x <= 1.5:
        # Both tails can be computed directly here; the logarithm of the larger one is
        # taken through the smaller one, log(1 - s) = log1p(-s).
        log_p = _log_gamma_prefactor(a, x, log_x, d) + math.log(_gamma_series(a, x, d))
        log_p = min(log_p, 0.0)
        if log_p <= -_LN2:
            return log_p, _log1mexp(-log_p)
        log_q = math.log(_gamma_upper_small_shape(a, x, log_x))
        return _log1mexp(-log_q), log_q
    if d > -1.0:  # x < a + 1: the series converges and P is the smaller tail
        log_p = _log_gamma_prefactor(a, x, log_x, d) + math.log(_gamma_series(a, x, d))
        log_p = min(log_p, 0.0)
        return log_p, _log1mexp(-log_p)
    log_q = (
        _log_gamma_prefactor(a, x, log_x, d)
        + math.log(a)
        - math.log(_gamma_continued_fraction(a, d))
    )
    log_q = min(log_q, 0.0)
    return _log1mexp(-log_q), log_q


@jit_numpy_errors
def _log_gammainc_checked(a: float, x: float) -> tuple[float, float]:
    """Return log P(a, x) and log Q(a, x), or NaN for arguments outside the domain."""
    if math.isnan(a) or math.isnan(x) or not (0.0 < a < _INF) or x < 0.0:
        return _NAN, _NAN
    return _log_gammainc_pair(a, x, math.log(x), a - x)


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
def _log_beta_prefactor(
    a: float, b: float, x: float, y: float, log_x: float, log_y: float, lam: float
) -> float:
    """Return log(x^a y^b / B(a, b)), where y = 1 - x and lam = a y - b x.

    A parameter counts as large from 10. With both large, Stirling's series and Loader's
    deviance terms bd0(a, (a + b) x) and bd0(b, (a + b) y) replace the logarithms, whose
    differences a - (a + b) x = lam and b - (a + b) y = -lam are known accurately. With one
    large parameter L and a small one, the term c log L of the gamma ratio is merged with
    c log v of the small parameter's power into c log(L v).
    """
    if a >= 10.0 and b >= 10.0:
        n = a + b
        m = n * x
        if abs(lam) < 0.5 * (a + m):
            dev_a = _deviance_term(a, m, lam)
        else:
            dev_a = a * (-math.log1p(b / a) - log_x) - lam
        m = n * y
        if abs(lam) < 0.5 * (b + m):
            dev_b = _deviance_term(b, m, -lam)
        else:
            dev_b = b * (-math.log1p(a / b) - log_y) + lam
        return (
            0.5 * math.log(a / n * b)
            - _LN_SQRT_2PI
            - dev_a
            - dev_b
            + _stirling_error(n)
            - _stirling_error(a)
            - _stirling_error(b)
        )
    if a >= 10.0:
        return (
            a * log_x
            + b * _log_product(a, y, log_y)
            + ((a + b - 0.5) * math.log1p(b / a) - b)
            + _stirling_error(a + b)
            - _stirling_error(a)
            - math.lgamma(b)
        )
    if b >= 10.0:
        return (
            b * log_y
            + a * _log_product(b, x, log_x)
            + ((a + b - 0.5) * math.log1p(a / b) - a)
            + _stirling_error(a + b)
            - _stirling_error(b)
            - math.lgamma(a)
        )
    return a * log_x + b * log_y - _log_beta_positive(a, b)


@jit_numpy_errors
def _beta_continued_fraction(
    a: float, b: float, x: float, y: float, lam: float
) -> float:
    """Return K with I_x(a, b) = x^a y^b / (B(a, b) K), for lam = a y - b x > x - y.

    DLMF 8.17.22 gives I_x(a, b) = x^a y^b / (a B(a, b)) / (1 + d_1/(1 + d_2/(1 + ...)))
    with d_{2m+1} = -(a + m)(a + b + m) x / ((a + 2m)(a + 2m + 1)) and
    d_{2m} = m (b - m) x / ((a + 2m - 1)(a + 2m)). Its even contraction has partial
    numerators A_k = -d_{2k-1} d_{2k} and denominators B_0 = 1 + d_1,
    B_k = 1 + d_{2k} + d_{2k+1}. In terms of lam and y, with s = a + 2k,
    B_0 = (1 + lam) / (a + 1) and
    B_k = (s [2k (a + k)(1 + y) + a - 1] + lam [a (s - 1) - 2k]) / ((s - 1) s (s + 1)),
    which do not cancel for lam > x - y. As x approaches 1 with a large, d_1 approaches -1
    and the plain fraction would lose accuracy in 1 + d_1. The fraction is scaled by a,
    and its k-th level by s, so that every element stays of moderate size for any a;
    K is a times the value of the contracted fraction. Evaluated by the modified Lentz
    method.
    """
    f = a * (1.0 + lam) / (a + 1.0)
    c = f
    e = 0.0
    k = 1.0
    while k < _MAX_ITERATIONS:
        s = a + 2.0 * k
        num = (
            ((a + k - 1.0) / (s - 1.0))
            * ((a + b + k - 1.0) * x / (s - 1.0))
            * (k * (b - k) * x)
        )
        den = (
            2.0 * k * (1.0 + y) * ((a + k) / (s - 1.0)) * (s / (s + 1.0))
            + ((a - 1.0) / (s - 1.0)) * (s / (s + 1.0))
            + lam * (a - 2.0 * k / (s - 1.0)) / (s + 1.0)
        )
        e = den + num * e
        if e == 0.0:
            e = _TINY
        c = den + num / c
        if c == 0.0:
            c = _TINY
        e = 1.0 / e
        delta = c * e
        f *= delta
        if abs(delta - 1.0) < _CF_TOLERANCE:
            return f
        k += 1.0
    return _NAN


@jit_numpy_errors
def _beta_upper_small_shape(a: float, b: float, x: float, log_x: float) -> float:
    """Return 1 - I_x(a, b) for 0 < a < 1 and b x < 2, without forming 1 - I.

    Integrating t^(a-1) [(1 - t)^(b-1) - 1] term by term gives
    1 - I_x(a, b) = -expm1(s) - exp(s) a sum_{n>=1} (1 - b)_n x^n / (n! (a + n)) with
    s = a log x - log(a B(a, b)). For a small, I_x(a, b) is close to 1 below the branch
    point of the continued fraction, where b x < a + 1, so 1 - I would lose accuracy;
    here the two parts cancel by at most a factor of about ten.
    """
    if b >= 10.0:
        # log Gamma(a + b) - log Gamma(b) = a log b + ..., merged with a log x into
        # a log(b x), as in the prefactor, so that no large logarithms cancel
        s = (
            a * _log_product(b, x, log_x)
            + ((a + b - 0.5) * math.log1p(a / b) - a)
            + _stirling_error(a + b)
            - _stirling_error(b)
            - _lgamma1p(a)
        )
    else:
        s = a * log_x - (_lgamma1p(a) - _log_gamma_increment(b, a))
    total = 0.0
    term = 1.0  # (1 - b)_n x^n / n!
    n = 1.0
    while n < 500.0:
        term *= (n - b) * x / n
        contribution = term / (a + n)
        total += contribution
        if abs(contribution) <= 0.25 * _EPS * abs(total):
            break
        n += 1.0
    return -math.expm1(s) - math.exp(s) * a * total


@jit_numpy_errors
def _log_betainc_pair(
    a: float, b: float, x: float, y: float, log_x: float, log_y: float, lam: float
) -> tuple[float, float]:
    """Return log I_x(a, b) and log(1 - I_x(a, b)) for finite a, b > 0.

    The caller supplies x, y = 1 - x, their logarithms and lam = a y - b x, each formed
    accurately from its own inputs (y and lam are never recomputed from a rounded x). x
    may underflow to 0 while log_x stays finite.
    """
    if log_x == -_INF:
        return -_INF, 0.0
    if log_y == -_INF:
        return 0.0, -_INF
    # The continued fraction gives the tail on its own side of the branch point; the
    # logarithm of the other tail is taken through it, log(1 - s) = log1p(-s), unless the
    # computed tail exceeds 1/2 with a shape below 1, where the other tail comes from the
    # small-shape series instead.
    if lam > x - y:  # x < (a + 1) / (a + b + 2)
        log_i = _log_beta_prefactor(a, b, x, y, log_x, log_y, lam) - math.log(
            _beta_continued_fraction(a, b, x, y, lam)
        )
        log_i = min(log_i, 0.0)
        if a < 1.0 and log_i > -_LN2:
            log_c = math.log(_beta_upper_small_shape(a, b, x, log_x))
            return _log1mexp(-log_c), log_c
        return log_i, _log1mexp(-log_i)
    log_c = _log_beta_prefactor(b, a, y, x, log_y, log_x, -lam) - math.log(
        _beta_continued_fraction(b, a, y, x, -lam)
    )
    log_c = min(log_c, 0.0)
    if b < 1.0 and log_c > -_LN2:
        log_i = math.log(_beta_upper_small_shape(b, a, y, log_y))
        return log_i, _log1mexp(-log_i)
    return _log1mexp(-log_c), log_c


@jit_numpy_errors
def _log_betainc_pair_at(a: float, b: float, x: float) -> tuple[float, float]:
    """Return log I_x(a, b) and log(1 - I_x(a, b)) for x in [0, 1], keeping 1 - x exact."""
    y = 1.0 - x
    if x < 0.5:
        y_low = -x - (y - 1.0)  # 1 - x = y + y_low exactly
        log_x = math.log(x)
        log_y = math.log1p(-x)
    else:
        y_low = 0.0  # 1 - x is exact
        log_x = math.log1p(-y)
        log_y = math.log(y)
    p, p_low = _two_product(a, y)
    q, q_low = _two_product(b, x)
    lam = (p - q) + ((p_low - q_low) + a * y_low)
    return _log_betainc_pair(a, b, x, y, log_x, log_y, lam)


@jit_numpy_errors
def _log_betainc_checked(a: float, b: float, x: float) -> tuple[float, float]:
    """Return log I_x(a, b) and its complement, or NaN for arguments outside the domain."""
    if (
        math.isnan(a)
        or math.isnan(b)
        or math.isnan(x)
        or not (0.0 < a < _INF)
        or not (0.0 < b < _INF)
        or not (0.0 <= x <= 1.0)
    ):
        return _NAN, _NAN
    return _log_betainc_pair_at(a, b, x)


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


def _apply(
    loop: Callable[..., Any], kind: int, **arguments: ArrayLike
) -> float | NDArray[np.float64]:
    """Broadcast the arguments to float64, run a compiled loop and shape its result.

    Every argument reaches the loop as a read-only, contiguous, one-dimensional float64
    array, so each loop is compiled for one signature only. Returns a Python float when
    every argument is a scalar or a 0-d array. A numpy masked array raises TypeError:
    NaN marks a missing value instead.
    """
    arrays = []
    for name, value in arguments.items():
        array = unmasked_array(value, name, missing="NaN")
        if array.dtype.kind not in "biuf":
            raise TypeError(
                f"{name} must contain real numbers (bool, integer or float); "
                f"got an array of dtype {array.dtype}"
            )
        arrays.append(array)
    try:
        shape = np.broadcast_shapes(*(array.shape for array in arrays))
    except ValueError:
        shapes = ", ".join(f"{n} {a.shape}" for n, a in zip(arguments, arrays))
        raise ValueError(
            f"arguments do not broadcast to a common shape: {shapes}"
        ) from None
    flat = []
    for array in arrays:
        values = np.ascontiguousarray(np.broadcast_to(array, shape), dtype=np.float64)
        values = values.reshape(-1)
        values.flags.writeable = False  # the view never writes through to the input
        flat.append(values)
    result: NDArray[np.float64] = loop(*flat, kind)
    if not shape:
        return float(result[0])
    return result.reshape(shape)


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
    return _apply(_gamma_family_array, _LGAMMA, x=x)


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
    return _apply(_gamma_family_array, _DIGAMMA, x=x)


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
    return _apply(_gamma_family_array, _TRIGAMMA, x=x)


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
    return _apply(_gamma_family_array, _TRIGAMMA_INVERSE, y=y)


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
    return _apply(_log_beta_array, 0, a=a, b=b)


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
    return _apply(_gammainc_array, _LOWER, a=a, x=x)


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
    return _apply(_gammainc_array, _UPPER, a=a, x=x)


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
    return _apply(_gammainc_array, _LOG_LOWER, a=a, x=x)


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
    return _apply(_gammainc_array, _LOG_UPPER, a=a, x=x)


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
    return _apply(_betainc_array, _LOWER, a=a, b=b, x=x)


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
    return _apply(_betainc_array, _UPPER, a=a, b=b, x=x)


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
    return _apply(_betainc_array, _LOG_LOWER, a=a, b=b, x=x)


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
    return _apply(_betainc_array, _LOG_UPPER, a=a, b=b, x=x)
