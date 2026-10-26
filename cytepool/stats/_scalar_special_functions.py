"""The special functions of scalars behind the vectorised ones, compiled with numba.

These routines compute, for one value at a time, the logarithms and differences of
gamma and beta functions and the regularized incomplete gamma and beta functions
with their complements, in log space. :mod:`cytepool.stats.special` and
:mod:`cytepool.stats.distributions` build their public functions on them, and the
count models call the logarithms of gamma ratios from compiled code. The methods are
those that :mod:`cytepool.stats.special` describes: Dekker's (1971) exact products,
Loader's (2000) deviance form of Stirling's series, Maechler's (2012) log(1 - exp(-a)),
the power series and continued fractions of DLMF 8.7, 8.9 and 8.17 evaluated by the
modified Lentz method (Lentz 1976; Thompson and Barnett 1986), and Temme's uniform
expansions of the incomplete gamma function (Temme 1979) and of the incomplete beta
function (Temme 1982).

``elementwise`` runs a compiled loop over broadcast arguments: every argument reaches
the loop as a read-only, contiguous, one-dimensional float64 array, so that each loop
compiles once.

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
"""

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cytepool._jit import jit_numpy_errors
from cytepool._masked_arrays import unmasked_array

INF = math.inf

NAN = math.nan

_LN2 = 0.6931471805599453

_LN_SQRT_2PI = 0.9189385332046728  # log(sqrt(2 pi))

_EULER = 0.5772156649015329

EPS = 2.220446049250313e-16  # spacing of the doubles in [1, 2)

_SMALLEST_NORMAL = 2.2250738585072014e-308

# The modified Lentz method stops when an update changes the value by about one unit
# in the last place; this is a little above the spacing of the doubles just below 1.
_CF_TOLERANCE = 3e-16

_TINY = 1e-300  # replaces a zero denominator in the modified Lentz method

_MAX_ITERATIONS = 10_000_000

# Arguments beyond the range of the usual forms. Deviance terms halve their arguments
# from this sum (2^1023). two_product splits factors below 1e290 only; a larger factor
# is scaled by 2^-600 first and its parts by 2^600 after. From shapes of 2^500, the
# products of the incomplete beta function's continued fraction and prefactor could
# overflow, so such shapes take forms of them ordered to stay in range.
_HALF_RANGE = 8.98846567431158e307
_SPLIT_LIMIT = 1e290
_SCALE_DOWN = 2.0**-600
_SCALE_UP = 2.0**600
_HUGE_SHAPE = 2.0**500
# Below this shape, the continued fraction's first elements, a times numbers of order
# one, could be subnormal; it is evaluated divided by a.
_TINY_SHAPE = 1e-280
# From this ratio of two shapes, the direct form of a deviance term subtracts
# logarithms that can be 40 times its own.
_FAR_SHAPES = 1e17
# Below this shape, and where the exponent s of the small-shape series is below the
# second number in absolute value, the series takes s / a instead of s, whose terms
# the Stirling corrections it adds and subtracts would absorb.
_SMALL_SERIES_SHAPE = 1e-3
_ABSORBED_EXPONENT = 1e-4

# Near its center, where x is within about one standard deviation of a / (a + b), the
# incomplete beta function of two shapes from 1e10 comes from Temme's uniform expansion
# instead of the continued fraction, whose number of steps there grows like the cube
# root of the shapes (about 11,600 at 1e10, found by counting them). The band is
# D = bd0(a, (a + b) x) + bd0(b, (a + b)(1 - x)) <= 1/2.
_UNIFORM_BETA_SHAPE = 1e10
_UNIFORM_BETA_BAND = 0.5

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
        NAN,
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

# The coefficients B_2k / (2k (2k - 1)) of Stirling's series for delta(z), k = 1..10
# (DLMF 5.11.1), which stirling_error sums for z >= 10.
_STIRLING_SERIES = np.array(
    [
        1.0 / 12.0,
        -1.0 / 360.0,
        1.0 / 1260.0,
        -1.0 / 1680.0,
        1.0 / 1188.0,
        -691.0 / 360360.0,
        1.0 / 156.0,
        -3617.0 / 122400.0,
        43867.0 / 244188.0,
        -174611.0 / 125400.0,
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


@jit_numpy_errors
def two_product(a: float, b: float) -> tuple[float, float]:
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
def log1mexp(a: float) -> float:
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
    if v >= _SMALLEST_NORMAL and _SMALLEST_NORMAL <= p < INF:
        return math.log(p)
    return math.log(c) + log_v


@jit_numpy_errors
def _erfcx(u: float) -> float:
    """Return exp(u^2) erfc(u) for u >= 0, with full relative accuracy."""
    if u < 26.0:
        # exp(u^2) with the square formed exactly (Dekker), times the C library's erfc
        p, e = two_product(u, u)
        return math.exp(p) * (1.0 + e) * math.erfc(u)
    # exp(u^2) erfc(u) = sqrt(2/pi) / t with Laplace's continued fraction for Mills'
    # ratio, t = z + 1/(z + 2/(z + ...)) at z = u sqrt 2 (DLMF 7.9.2)
    z = u * _SQRT2
    t = z
    for k in range(40, 0, -1):
        t = z + k / t
    return _SQRT_2_OVER_PI / t


@jit_numpy_errors
def stirling_error(z: float) -> float:
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
def deviance_term(x: float, m: float, d: float) -> float:
    """Return Loader's bd0(x, m) = x log(x / m) + m - x, given d = x - m.

    The caller forms d accurately; near x = m the series in v = d / (x + m) avoids the
    cancellation of the direct form (Loader 2000). bd0 is homogeneous of degree one:
    where x + m, or 2 x, would overflow, x, m and d are halved and the result doubled.
    """
    if x == 0.0:
        return m
    scale = 1.0
    s = x + m
    if not s < _HALF_RANGE:
        x, m, d, scale = 0.5 * x, 0.5 * m, 0.5 * d, 2.0
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
                return scale * total
            total = updated
            j += 1.0
    return scale * (x * math.log(x / m) - d)


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
def log_gamma_ratio(a: float, b: float) -> float:
    """Return log Gamma(a + b) - log Gamma(a) for a >= 10, b >= 0, without cancellation."""
    return (
        b * math.log(a)
        + (a + b - 0.5) * math.log1p(b / a)
        - b
        + stirling_error(a + b)
        - stirling_error(a)
    )


@jit_numpy_errors
def log_gamma_increment(b: float, a: float) -> float:
    """Return log Gamma(b + a) - log Gamma(b) for b > 0 and a >= 0, accurate for small a.

    Below 10, b is shifted up with log Gamma(z + 1) = log Gamma(z) + log z, so the result
    is a sum of terms of one sign instead of a difference of two log-gamma values.
    """
    if b >= 10.0:
        return log_gamma_ratio(b, a)
    total = 0.0
    shift = 0.0
    while b + shift < 10.0:
        total += math.log1p(a / (b + shift))
        shift += 1.0
    return log_gamma_ratio(b + shift, a) - total


@jit_numpy_errors
def log_beta_positive(a: float, b: float) -> float:
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
            + stirling_error(a)
            + stirling_error(b)
            - stirling_error(n)
        )
    # log Gamma(b) - [log Gamma(a + b) - log Gamma(a)], with the bracket formed without
    # subtracting log-gamma values
    return math.lgamma(b) - log_gamma_increment(a, b)


@jit_numpy_errors
def log_gamma_prefactor(a: float, x: float, log_x: float, d: float) -> float:
    """Return log(x^a exp(-x) / Gamma(a + 1)), given log_x = log(x) and d = a - x."""
    if a >= 10.0:
        # -bd0(a, x) - delta(a) - log sqrt(2 pi a), with Loader's deviance term
        if abs(d) < 0.5 * (a + x):
            deviance = deviance_term(a, x, d)
        else:
            deviance = a * (math.log(a) - log_x) - d
        return -deviance - stirling_error(a) - _LN_SQRT_2PI - 0.5 * math.log(a)
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
        if gap > 0.0 and term * x <= 0.25 * EPS * total * gap:
            return total
        n += 1.0
    return NAN


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
    return NAN


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
        if abs(term) <= 0.25 * EPS * abs(total):
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
        return log1mexp(-log_q), log_q
    log_p = -deviance + math.log(0.5 * scaled - total * _INV_SQRT_2PI / math.sqrt(a))
    log_p = min(log_p, 0.0)
    return log_p, log1mexp(-log_p)


@jit_numpy_errors
def log_gammainc_pair(
    a: float, x: float, log_x: float, d: float
) -> tuple[float, float]:
    """Return log P(a, x) and log Q(a, x) for finite a > 0 and x >= 0.

    The caller supplies log_x = log(x) and d = a - x, each formed accurately from its own
    inputs (for example without rounding a scaled argument first); d sets the accuracy of
    the prefactor for large a. x may underflow to 0 while log_x stays finite.
    """
    if log_x == -INF:
        return -INF, 0.0
    if x == INF:
        return 0.0, -INF
    if a >= _UNIFORM_SHAPE and abs(d) < 0.5 * (a + x):
        # Near the center the series and the continued fraction need many steps for a
        # large shape (about 8 sqrt(a) for the series); the uniform expansion does not.
        deviance = deviance_term(a, x, d)
        if deviance <= 0.125 * a:  # |eta| <= 1/2
            return _log_gammainc_uniform(a, d, deviance)
    if a < 1.0 and x <= 1.5:
        # Both tails can be computed directly here; the logarithm of the larger one is
        # taken through the smaller one, log(1 - s) = log1p(-s).
        log_p = log_gamma_prefactor(a, x, log_x, d) + math.log(_gamma_series(a, x, d))
        log_p = min(log_p, 0.0)
        if log_p <= -_LN2:
            return log_p, log1mexp(-log_p)
        log_q = math.log(_gamma_upper_small_shape(a, x, log_x))
        return log1mexp(-log_q), log_q
    if d > -1.0:  # x < a + 1: the series converges and P is the smaller tail
        log_p = log_gamma_prefactor(a, x, log_x, d) + math.log(_gamma_series(a, x, d))
        log_p = min(log_p, 0.0)
        return log_p, log1mexp(-log_p)
    log_q = (
        log_gamma_prefactor(a, x, log_x, d)
        + math.log(a)
        - math.log(_gamma_continued_fraction(a, d))
    )
    log_q = min(log_q, 0.0)
    return log1mexp(-log_q), log_q


@jit_numpy_errors
def log_beta_prefactor(
    a: float, b: float, x: float, y: float, log_x: float, log_y: float, lam: float
) -> float:
    """Return log(x^a y^b / B(a, b)), where y = 1 - x and lam = a y - b x.

    A parameter counts as large from 10. With both large, Stirling's series and Loader's
    deviance terms bd0(a, (a + b) x) and bd0(b, (a + b) y) replace the logarithms, whose
    differences a - (a + b) x = lam and b - (a + b) y = -lam are known accurately. With one
    large parameter L and a small one, the term c log L of the gamma ratio is merged with
    c log v of the small parameter's power into c log(L v). Where a + b overflows, a b /
    (a + b) is formed as b / (1 + b / a) and the deviance terms at half scale.
    """
    if a >= 10.0 and b >= 10.0:
        n = a + b
        if n < INF:
            dev_a = _shape_deviance(a, b, n * x, lam, log_x)
            dev_b = _shape_deviance(b, a, n * y, -lam, log_y)
            half_log = 0.5 * math.log(a / n * b)
        else:
            dev_a = _halved_shape_deviance(a, b, lam, log_x)
            dev_b = _halved_shape_deviance(b, a, -lam, log_y)
            half_log = 0.5 * math.log(b / (1.0 + b / a))
        return (
            half_log
            - _LN_SQRT_2PI
            - dev_a
            - dev_b
            + stirling_error(n)
            - stirling_error(a)
            - stirling_error(b)
        )
    if a >= 10.0:
        return (
            a * log_x
            + b * _log_product(a, y, log_y)
            + ((a + b - 0.5) * math.log1p(b / a) - b)
            + stirling_error(a + b)
            - stirling_error(a)
            - math.lgamma(b)
        )
    if b >= 10.0:
        return (
            b * log_y
            + a * _log_product(b, x, log_x)
            + ((a + b - 0.5) * math.log1p(a / b) - a)
            + stirling_error(a + b)
            - stirling_error(b)
            - math.lgamma(a)
        )
    return a * log_x + b * log_y - log_beta_positive(a, b)


@jit_numpy_errors
def _shape_deviance(a: float, b: float, m: float, lam: float, log_x: float) -> float:
    """Return bd0(a, m) for m = (a + b) x, lam = a - m and log_x = log x, a and b
    from 10, as log_beta_prefactor takes it.

    Near m = a, Loader's series. Further, a log(a / m) - lam, with a log(a / m) as
    a (-log1p(b / a) - log x), which needs no m; but where b exceeds 1e17 a and x is
    tiny the two logarithms cancel, and m, a product rounded once, gives the quotient
    better.
    """
    if abs(lam) < 0.5 * (a + m):
        return deviance_term(a, m, lam)
    if b > _FAR_SHAPES * a and m >= _SMALLEST_NORMAL and a / m < INF:
        return a * math.log(a / m) - lam
    return a * (-math.log1p(b / a) - log_x) - lam


@jit_numpy_errors
def _halved_shape_deviance(a: float, b: float, lam: float, log_x: float) -> float:
    """Return bd0(a, (a + b) x) for a + b beyond the double range, from bd0(a / 2,
    (a - lam) / 2) = bd0(a, a - lam) / 2, which stays in range."""
    half = 0.5 * a - 0.5 * lam  # (a + b) x / 2
    if abs(lam) < 0.5 * a + half:
        return 2.0 * deviance_term(0.5 * a, half, 0.5 * lam)
    return a * (-math.log1p(b / a) - log_x) - lam


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
    return NAN


@jit_numpy_errors
def _log_beta_continued_fraction_in_range(
    a: float, b: float, x: float, y: float, lam: float
) -> float:
    """Return log K, for the K of :func:`_beta_continued_fraction`, for shapes from 2^500,
    or for a below 1e-280.

    The same fraction, its products ordered so that none leaves the float64 range:
    a (1 + lam) / (a + 1) as (1 + lam) (a / (a + 1)); (a + b + k - 1) x / (s - 1) as
    x (a + k - 1) / (s - 1) + x b / (s - 1); lam (a - 2k / (s - 1)) / (s + 1) with the
    quotient first. The modified Lentz method keeps its second sequence as the
    denominators e_k = den + num / e_(k-1) instead of their reciprocals, which would be
    subnormal, and so inexact, for denominators beyond 2^1022; the value is scaled by
    2^-600 when it is that large. For a tiny a, whose products with the first two
    elements, both proportional to a, would be subnormal, it evaluates K / a: their
    quotients by a and the later elements as they are. A step whose terms are not finite
    ends it with NaN, instead of the remaining steps running on NaN.
    """
    tiny = a < _TINY_SHAPE
    if tiny:
        f = (1.0 + lam) / (a + 1.0)
        scale = math.log(a)
    else:
        f = (1.0 + lam) * (a / (a + 1.0))
        scale = 0.0
    c = f
    if f > _HUGE_SHAPE:
        f *= _SCALE_DOWN
        scale += 600.0 * _LN2
    e = INF
    k = 1.0
    while k < _MAX_ITERATIONS:
        s = a + 2.0 * k
        ratio = (a + k - 1.0) / (s - 1.0)
        first = 1.0 / (s - 1.0) if tiny and k == 1.0 else ratio
        num = first * (x * ratio + x * (b / (s - 1.0))) * (k * ((b - k) * x))
        den = (
            2.0 * k * (1.0 + y) * ((a + k) / (s - 1.0)) * (s / (s + 1.0))
            + ((a - 1.0) / (s - 1.0)) * (s / (s + 1.0))
            + lam * ((a - 2.0 * k / (s - 1.0)) / (s + 1.0))
        )
        if not (abs(num) < INF and abs(den) < INF):
            return NAN
        e = den + num / e
        if e == 0.0:
            e = _TINY
        c = den + num / c
        if c == 0.0:
            c = _TINY
        delta = c / e
        f *= delta
        if abs(delta - 1.0) < _CF_TOLERANCE:
            return math.log(f) + scale
        k += 1.0
    return NAN


@jit_numpy_errors
def _log_beta_fraction(a: float, b: float, x: float, y: float, lam: float) -> float:
    """Return log K, for the K of :func:`_beta_continued_fraction`, for any shapes."""
    if _TINY_SHAPE <= a < _HUGE_SHAPE and b < _HUGE_SHAPE:
        return math.log(_beta_continued_fraction(a, b, x, y, lam))
    return _log_beta_continued_fraction_in_range(a, b, x, y, lam)


@jit_numpy_errors
def _log_betainc_uniform(
    a: float, b: float, lam: float, deviance: float
) -> tuple[float, float]:
    """Return log I_x(a, b) and log(1 - I_x(a, b)) near the center, for a, b >= 1e10.

    Temme's (1982) uniform expansion, as the incomplete gamma function's above. With
    r = a + b, x0 = a / r and eta of the sign of x - x0 (so of -lam, lam = r (x0 - x))
    defined by r eta^2 / 2 = bd0(a, r x) + bd0(b, r (1 - x)) = deviance,

        I_x(a, b) = erfc(-eta sqrt(r / 2)) / 2 - G exp(-deviance) c0(eta) / sqrt(2 pi r)

    up to terms of relative order min(a, b)^(-3/2), below 1e-15 from 1e10; G is
    Gamma*(r) / (Gamma*(a) Gamma*(b)), the ratio of the Stirling series' corrections,
    and 1 - I has the opposite signs. The density in eta is
    sqrt(r / (2 pi)) G exp(-r eta^2 / 2) f(eta) with f(eta) = sqrt(x0 (1 - x0)) eta /
    (t - x0) at the t of eta, and c0(eta) = (f(eta) - 1) / eta. Its Taylor series,
    derived by reverting the series of eta in t - x0 (and checked against mpmath), is

        c0(eta) = (p - q) / (3 s) + (1 - s^2) / (12 s^2) eta
                  + (p - q)(2 + s^2) / (135 s^3) eta^2 + O(eta^3),

    with p = x0, q = 1 - x0 and s^2 = p q. In the band, r eta^2 / 2 <= 1/2, so that the
    omitted terms are below 1e-15 of c0, itself of order min(a, b)^(-1/2) of I. Every
    quantity is formed from p, q and a q = a b / r, so that r itself, which may
    overflow, is never needed.
    """
    if a <= b:
        ratio = a / b
        p = ratio / (1.0 + ratio)
        q = 1.0 / (1.0 + ratio)
    else:
        ratio = b / a
        p = 1.0 / (1.0 + ratio)
        q = ratio / (1.0 + ratio)
    pq = p * q
    pqr = a * q  # a b / r
    difference = (0.5 * a - 0.5 * b) / (0.5 * a + 0.5 * b)  # p - q
    root = math.sqrt(deviance)  # |eta| sqrt(r / 2)
    side = 1.0 if lam > 0.0 else -1.0  # x below the center: I is the smaller tail
    # c0(eta) / sqrt(r), with eta / sqrt(r) = -side sqrt(2 deviance) / r
    c0 = (
        difference / (3.0 * math.sqrt(pqr))
        - side * math.sqrt(2.0 * deviance) * (1.0 - pq) / (12.0 * pqr)
        + difference * (2.0 + pq) * 2.0 * deviance / (135.0 * pqr * math.sqrt(pqr))
    )
    gamma_ratio = math.exp(
        stirling_error(a + b) - stirling_error(a) - stirling_error(b)
    )
    correction = gamma_ratio * math.exp(-deviance) * c0 * _INV_SQRT_2PI
    lower = 0.5 * math.erfc(side * root) - correction
    upper = 0.5 * math.erfc(-side * root) + correction
    return math.log(lower), math.log(upper)


@jit_numpy_errors
def _beta_upper_small_shape(a: float, b: float, x: float, log_x: float) -> float:
    """Return 1 - I_x(a, b) for 0 < a < 1 and b x < 2, without forming 1 - I.

    Integrating t^(a-1) [(1 - t)^(b-1) - 1] term by term gives
    1 - I_x(a, b) = -expm1(s) - exp(s) a sum_{n>=1} (1 - b)_n x^n / (n! (a + n)) with
    s = a log x - log(a B(a, b)). For a small, I_x(a, b) is close to 1 below the branch
    point of the continued fraction, where b x < a + 1, so 1 - I would lose accuracy;
    here the two parts cancel by at most a factor of about ten.
    """
    s = _small_shape_exponent_sum(a, b, x, log_x)
    return -math.expm1(s) - math.exp(s) * a * _small_shape_series(a, b, x)


@jit_numpy_errors
def _small_shape_exponent_sum(a: float, b: float, x: float, log_x: float) -> float:
    """Return the s = a log x - log(a B(a, b)) of :func:`_beta_upper_small_shape`, as a
    sum of its terms."""
    if b >= 10.0:
        # log Gamma(a + b) - log Gamma(b) = a log b + ..., merged with a log x into
        # a log(b x), as in the prefactor, so that no large logarithms cancel
        return (
            a * _log_product(b, x, log_x)
            + ((a + b - 0.5) * math.log1p(a / b) - a)
            + stirling_error(a + b)
            - stirling_error(b)
            - _lgamma1p(a)
        )
    return a * log_x - (_lgamma1p(a) - log_gamma_increment(b, a))


@jit_numpy_errors
def _small_shape_series(a: float, b: float, x: float) -> float:
    """Return sum_{n>=1} (1 - b)_n x^n / (n! (a + n)), the series of
    :func:`_beta_upper_small_shape`."""
    total = 0.0
    term = 1.0  # (1 - b)_n x^n / n!
    n = 1.0
    while n < 500.0:
        term *= (n - b) * x / n
        contribution = term / (a + n)
        total += contribution
        if abs(contribution) <= 0.25 * EPS * abs(total):
            break
        n += 1.0
    return total


@jit_numpy_errors
def _log_beta_upper_small_shape(a: float, b: float, x: float, log_x: float) -> float:
    """Return log(1 - I_x(a, b)) for 0 < a < 1 and b x < 2.

    The logarithm of :func:`_beta_upper_small_shape`, unless a is below 1e-3 and s
    below 1e-4 in absolute value. The sum that forms s adds and subtracts Stirling
    corrections of up to 0.0084, whose rounding leaves an error of about 1e-18 in s:
    a relative error of 1e-14 or more there, all of s at shapes below about 1e-20,
    and a subnormal a rounds its terms too. There the same formula is taken as
    1 - I = a [-sigma expm1(a sigma) / (a sigma) - exp(a sigma) S], with sigma = s / a
    and S the series; sigma is log x - log Gamma(1 + a) / a
    + (log Gamma(b + a) - log Gamma(b)) / a, with every quotient by a taken from its
    series.
    """
    s = _small_shape_exponent_sum(a, b, x, log_x)
    sigma = NAN
    if a < _SMALL_SERIES_SHAPE and abs(s) < _ABSORBED_EXPONENT:
        sigma = _small_shape_exponent(a, b, x, log_x)
    if not abs(sigma) < INF:  # not needed, or beyond the range for subnormal shapes
        return math.log(-math.expm1(s) - math.exp(s) * a * _small_shape_series(a, b, x))
    t = a * sigma
    ratio = math.expm1(t) / t if t != 0.0 else 1.0
    bracket = -sigma * ratio - math.exp(t) * _small_shape_series(a, b, x)
    return math.log(a) + math.log(bracket)


@jit_numpy_errors
def _small_shape_exponent(a: float, b: float, x: float, log_x: float) -> float:
    """Return s / a = log x - log(a B(a, b)) / a for 0 < a < 1e-3 (any b > 0).

    log(a B(a, b)) = log Gamma(1 + a) - (log Gamma(b + a) - log Gamma(b)). The first
    quotient comes from the Taylor series of log Gamma(1 + a); the second, for b from
    10, from log Gamma(b + a) - log Gamma(b) = a log b + (b + a - 1/2) log1p(a / b) - a
    + delta(b + a) - delta(b), its log b merged with log x into log(b x) as in the
    prefactor; for b below 10, b is shifted up to 10 or more first, as
    :func:`log_gamma_increment` shifts it.
    """
    c = _LGAMMA1P_TAYLOR
    poly: float = c[c.size - 1]
    for k in range(c.size - 2, -1, -1):
        poly = poly * a + c[k]
    lgamma1p_over = a * poly - _EULER  # log Gamma(1 + a) / a
    if b >= 10.0:
        return _log_product(b, x, log_x) + _gamma_ratio_slope(b, a) - lgamma1p_over
    total = 0.0  # sum_j log1p(a / (b + j)) / a over the shifts
    z = b
    while z < 10.0:
        total += _log1p_ratio(a / z) / z
        z += 1.0
    return log_x + ((math.log(z) + _gamma_ratio_slope(z, a)) - total) - lgamma1p_over


@jit_numpy_errors
def _gamma_ratio_slope(z: float, a: float) -> float:
    """Return (log Gamma(z + a) - log Gamma(z)) / a - log z for z >= 10 and 0 < a < 1.

    That is (z + a - 1/2) log1p(u) / (u z) - 1 + (delta(z + a) - delta(z)) / a with
    u = a / z, each quotient from a form that stays accurate as u goes to 0, where
    their limits are -1/(2 z) and delta'(z).
    """
    u = a / z
    g = _log1p_ratio(u)
    # (z + a - 1/2) g / z - 1 = (g - 1) + g (a - 1/2) / z
    g_minus_one = -u * (0.5 - u / 3.0) if u < 1e-8 else g - 1.0
    # (delta(z + a) - delta(z)) / a = sum_k c_k z^-2k [(1 + u)^-(2k-1) - 1] / u
    slope = 0.0
    inverse_square = 1.0 / (z * z)
    power = inverse_square
    for k in range(_STIRLING_SERIES.size):
        m = 2.0 * k + 1.0
        if u < 1e-8:
            h = -m * (1.0 - 0.5 * (m + 1.0) * u)
        else:
            h = math.expm1(-m * math.log1p(u)) / u
        slope += _STIRLING_SERIES[k] * power * h
        power *= inverse_square
    return g_minus_one + g * (a - 0.5) / z + slope


@jit_numpy_errors
def _log1p_ratio(u: float) -> float:
    """Return log1p(u) / u for u >= 0, 1 at u = 0."""
    if u < 1e-8:
        return 1.0 - u * (0.5 - u / 3.0)
    return math.log1p(u) / u


@jit_numpy_errors
def log_betainc_pair(
    a: float, b: float, x: float, y: float, log_x: float, log_y: float, lam: float
) -> tuple[float, float]:
    """Return log I_x(a, b) and log(1 - I_x(a, b)) for finite a, b > 0.

    The caller supplies x, y = 1 - x, their logarithms and lam = a y - b x, each formed
    accurately from its own inputs (y and lam are never recomputed from a rounded x). x
    may underflow to 0 while log_x stays finite.
    """
    if log_x == -INF:
        return -INF, 0.0
    if log_y == -INF:
        return 0.0, -INF
    if a >= _UNIFORM_BETA_SHAPE and b >= _UNIFORM_BETA_SHAPE:
        # (a + b) x = a - lam and (a + b) y = b + lam
        deviance = deviance_term(a, a - lam, lam) + deviance_term(b, b + lam, -lam)
        if deviance <= _UNIFORM_BETA_BAND:
            return _log_betainc_uniform(a, b, lam, deviance)
    # The continued fraction gives the tail on its own side of the branch point; the
    # logarithm of the other tail is taken through it, log(1 - s) = log1p(-s), unless the
    # computed tail exceeds 1/2 with a shape below 1, where the other tail comes from the
    # small-shape series instead.
    if lam > x - y:  # x < (a + 1) / (a + b + 2)
        log_i = log_beta_prefactor(a, b, x, y, log_x, log_y, lam) - _log_beta_fraction(
            a, b, x, y, lam
        )
        log_i = min(log_i, 0.0)
        if a < 1.0 and log_i > -_LN2:
            log_c = _log_beta_upper_small_shape(a, b, x, log_x)
            return log1mexp(-log_c), log_c
        return log_i, log1mexp(-log_i)
    log_c = log_beta_prefactor(b, a, y, x, log_y, log_x, -lam) - _log_beta_fraction(
        b, a, y, x, -lam
    )
    log_c = min(log_c, 0.0)
    if b < 1.0 and log_c > -_LN2:
        log_i = _log_beta_upper_small_shape(b, a, y, log_y)
        return log_i, log1mexp(-log_i)
    return log1mexp(-log_c), log_c


@jit_numpy_errors
def log_betainc_pair_at(a: float, b: float, x: float) -> tuple[float, float]:
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
    if a < _SPLIT_LIMIT and b < _SPLIT_LIMIT:
        p, p_low = two_product(a, y)
        q, q_low = two_product(b, x)
    else:
        p, p_low = _scaled_two_product(a, y)
        q, q_low = _scaled_two_product(b, x)
    lam = (p - q) + ((p_low - q_low) + a * y_low)
    return log_betainc_pair(a, b, x, y, log_x, log_y, lam)


@jit_numpy_errors
def _scaled_two_product(c: float, v: float) -> tuple[float, float]:
    """Return two_product(c, v) for finite c >= 0 and 0 <= v <= 1, c up to the largest
    double: from 1e290, c is scaled by 2^-600 before Dekker's split, which would
    overflow, and both parts are scaled back, all exactly."""
    if c < _SPLIT_LIMIT:
        return two_product(c, v)
    p, e = two_product(c * _SCALE_DOWN, v)
    return p * _SCALE_UP, e * _SCALE_UP


def elementwise(
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
