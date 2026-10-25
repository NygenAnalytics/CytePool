"""Probability distributions: cumulative distribution functions, survival functions,
their logarithms and quantiles, compiled with numba.

Names and parametrisations follow ``scipy.stats``: ``norm``, ``t``, ``f``, ``chi2``,
``gamma`` (shape and scale), ``beta`` and ``hypergeom`` (population size ``M``, number of
successes ``n`` in the population, number of draws ``N``). The survival function ``sf`` is
P(X > x); ``logsf`` and ``logcdf`` stay finite far beyond the point where the plain
probabilities underflow, which p-values of strong effects routinely reach.

Every public function is vectorised: its arguments broadcast with numpy's rules and the
result is a float64 array of the broadcast shape, or a Python float when every argument
is a scalar. NaN arguments give NaN, and so do parameters outside their domain; the
functions never raise for numeric input. Arguments that are not real numbers raise
``TypeError``, and so do numpy masked arrays, whose mask numpy would drop: NaN marks a
missing value instead.

The compiled scalar routines behind the public functions are named
``<function>_scalar`` (for example ``t_sf_scalar``) and can be called from other compiled
code, such as per-gene loops.

Methods:
    - Normal: Q(x) = erfc(x / sqrt 2) / 2 from the C library's ``erfc``, with the
      rounding error of x / sqrt 2 corrected to first order (Dekker's exact product), so
      the tails keep full relative accuracy; beyond x = 37, log Q(x) from Laplace's
      continued fraction for Mills' ratio. Quantiles by Wichura's (1988) algorithm AS 241.
    - Student t: P(|T| > |x|) = I_w(df/2, 1/2) with w = df / (df + x^2); df may be
      non-integer and df = inf gives the normal distribution. Quantiles: closed forms for
      df = 1, 2 and inf; otherwise Newton's method in log t on log P(T > t), or on
      log P(|T| <= t) for p > 1/4 where the quantile is small, started from the better
      of the Cornish-Fisher expansion (Abramowitz and Stegun 26.7.5) and the power-law
      tail, and kept inside a bisection bracket.
    - F: P(F > x) = I_w(dfd/2, dfn/2) with w = dfd / (dfd + dfn x); dfd = inf gives
      dfn F ~ chi-squared(dfn), and dfn = inf gives dfd / F ~ chi-squared(dfd).
      Quantiles: Newton's method in log x on the logarithm of the tail that holds
      min(p, 1 - p), started from the better of Paulson's normal approximation to
      F^(1/3) and the power law of that tail, and kept inside a bisection bracket;
      dfn = inf uses 1 / F ~ F(dfd, dfn).
    - Chi-squared and gamma: the regularized incomplete gamma function; chi-squared
      quantiles are those of X / df ~ F(df, inf).
    - Beta: the regularized incomplete beta function.
    - Hypergeometric: each probability from Loader's (2000) saddle-point form of the
      binomial probabilities, P(X = k) = b(k; n, p) b(N - k; M - n, p) / b(N; M, p) with
      p = N / M; the tail on the far side of the mode is summed from its largest term
      by the ratio of successive probabilities, with compensated addition, until the
      rest of the tail (bounded through the log-concavity of the probabilities) is
      negligible, and the other tail is its complement in log space.

The incomplete gamma and beta functions are called with their arguments formed from the
distribution's own inputs (for example lambda = a w' - b w for the beta function is
formed as df (x^2 - 1) / (2 (df + x^2)) for the t distribution), so no accuracy is lost
to rounding a transformed argument; see ``cytepool.stats.special``.

References:
    - Abramowitz M, Stegun IA (1964). Handbook of Mathematical Functions. National Bureau
      of Standards, sections 26.4, 26.6, 26.7.
    - Dekker TJ (1971). A floating-point technique for extending the available
      precision. Numerische Mathematik 18:224-242.
    - Johnson NL, Kotz S, Balakrishnan N (1995). Continuous Univariate Distributions,
      volume 2, 2nd edition. Wiley, chapter 27.
    - Loader C (2000). Fast and accurate computation of binomial probabilities.
    - Paulson E (1942). An approximate normalization of the analysis of variance
      distribution. Annals of Mathematical Statistics 13:233-235.
    - Wichura MJ (1988). Algorithm AS 241: the percentage points of the normal
      distribution. Journal of the Royal Statistical Society C 37:477-484.
"""

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from cytepool.stats.special import (
    _apply,
    _compile,
    _deviance_term,
    _log1mexp,
    _log_beta_positive,
    _log_beta_prefactor,
    _log_betainc_pair,
    _log_betainc_pair_at,
    _log_gamma_prefactor,
    _log_gamma_ratio,
    _log_gammainc_pair,
    _stirling_error,
    _two_product,
)

__all__ = [
    "beta_cdf",
    "beta_sf",
    "chi2_cdf",
    "chi2_isf",
    "chi2_logsf",
    "chi2_ppf",
    "chi2_sf",
    "f_cdf",
    "f_isf",
    "f_logsf",
    "f_ppf",
    "f_sf",
    "gamma_cdf",
    "gamma_sf",
    "hypergeom_logsf",
    "hypergeom_sf",
    "norm_cdf",
    "norm_isf",
    "norm_logcdf",
    "norm_logsf",
    "norm_ppf",
    "norm_sf",
    "t_cdf",
    "t_isf",
    "t_logsf",
    "t_ppf",
    "t_sf",
]


_INF = math.inf
_NAN = math.nan
_PI = math.pi
_LN2 = 0.6931471805599453
_LN_SQRT_2PI = 0.9189385332046728  # log(sqrt(2 pi))
_INV_SQRT_PI = 0.5641895835477563  # 1 / sqrt(pi)
_SQRT_HALF_HI = 0.7071067811865476  # 1 / sqrt(2) as a sum of two doubles
_SQRT_HALF_LO = -4.833646656726457e-17
_EPS = 2.220446049250313e-16
_SMALLEST_NORMAL = 2.2250738585072014e-308
_LARGEST = 1.7976931348623157e308
_LOG_LARGEST = 709.782712893384  # log of the largest double
_EXACT_INTEGERS = 9007199254740992.0  # 2^53
_SMALLEST_POSITIVE = 5e-324  # the smallest subnormal double
_SUBNORMAL_SLACK = 2e-323  # four units of the smallest subnormal double
_MAX_TERMS = 10_000_000  # step limit of the hypergeometric tail sums

# Wichura (1988), algorithm AS 241 (PPND16): coefficients of the rational approximations,
# lowest degree first, for |p - 1/2| <= 0.425 (A / B), for r = sqrt(-log(min(p, 1 - p)))
# <= 5 (C / D) and for r > 5 (E / F).
_AS241_A = np.array(
    [
        3.3871328727963666080e0,
        1.3314166789178437745e2,
        1.9715909503065514427e3,
        1.3731693765509461125e4,
        4.5921953931549871457e4,
        6.7265770927008700853e4,
        3.3430575583588128105e4,
        2.5090809287301226727e3,
    ]
)
_AS241_B = np.array(
    [
        1.0,
        4.2313330701600911252e1,
        6.8718700749205790830e2,
        5.3941960214247511077e3,
        2.1213794301586595867e4,
        3.9307895800092710610e4,
        2.8729085735721942674e4,
        5.2264952788528545610e3,
    ]
)
_AS241_C = np.array(
    [
        1.42343711074968357734e0,
        4.63033784615654529590e0,
        5.76949722146069140550e0,
        3.64784832476320460504e0,
        1.27045825245236838258e0,
        2.41780725177450611770e-1,
        2.27238449892691845833e-2,
        7.74545014278341407640e-4,
    ]
)
_AS241_D = np.array(
    [
        1.0,
        2.05319162663775882187e0,
        1.67638483018380384940e0,
        6.89767334985100004550e-1,
        1.48103976427480074590e-1,
        1.51986665636164571966e-2,
        5.47593808499534494600e-4,
        1.05075007164441684324e-9,
    ]
)
_AS241_E = np.array(
    [
        6.65790464350110377720e0,
        5.46378491116411436990e0,
        1.78482653991729133580e0,
        2.96560571828504891230e-1,
        2.65321895265761230930e-2,
        1.24266094738807843860e-3,
        2.71155556874348757815e-5,
        2.01033439929228813265e-7,
    ]
)
_AS241_F = np.array(
    [
        1.0,
        5.99832206555887937690e-1,
        1.36929880922735805310e-1,
        1.48753612908506148525e-2,
        7.86869131145613259100e-4,
        1.84631831751005468180e-5,
        1.42151175831644588870e-7,
        2.04426310338993978564e-15,
    ]
)

_CDF, _SF, _LOGCDF, _LOGSF, _PPF, _ISF = 0, 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------------------
# Normal distribution
# ---------------------------------------------------------------------------------------


@_compile
def _normal_upper(x: float) -> float:
    """Return Q(x) = P(Z > x) for x >= 0."""
    if x < 0.5:
        return 0.5 * math.erfc(x * _SQRT_HALF_HI)
    if x > 40.0:
        return 0.0
    # Q(x) = erfc(u) / 2 with u = x / sqrt 2. Rounding u would cost a relative error of
    # about x^2 eps in the tail; Dekker's product gives the exact remainder du, which is
    # added to first order using d erfc(u) / du = -2 exp(-u^2) / sqrt(pi).
    u, u_low = _two_product(x, _SQRT_HALF_HI)
    du = u_low + x * _SQRT_HALF_LO
    return 0.5 * math.erfc(u) - _INV_SQRT_PI * math.exp(-u * u) * du


@_compile
def _normal_log_upper(x: float) -> float:
    """Return log Q(x) for x >= 0."""
    if x < 37.0:  # Q(37) = 5.7e-300 is still a normal double
        return math.log(_normal_upper(x))
    # Q(x) = phi(x) / t with Laplace's continued fraction for Mills' ratio,
    # t = x + 1/(x + 2/(x + 3/(x + ...))) (DLMF 7.9.2); forty levels are exact here.
    t = x
    for k in range(40, 0, -1):
        t = x + k / t
    return -0.5 * x * x - _LN_SQRT_2PI - math.log(t)


@_compile
def norm_cdf_scalar(x: float) -> float:
    """Return P(Z <= x) for a standard normal Z."""
    if math.isnan(x):
        return _NAN
    if x <= 0.0:
        return _normal_upper(-x)
    return 1.0 - _normal_upper(x)


@_compile
def norm_sf_scalar(x: float) -> float:
    """Return P(Z > x) for a standard normal Z."""
    return norm_cdf_scalar(-x)


@_compile
def norm_logcdf_scalar(x: float) -> float:
    """Return log P(Z <= x) for a standard normal Z."""
    if math.isnan(x):
        return _NAN
    if x <= 0.0:
        return _normal_log_upper(-x)
    return math.log1p(-_normal_upper(x))


@_compile
def norm_logsf_scalar(x: float) -> float:
    """Return log P(Z > x) for a standard normal Z."""
    return norm_logcdf_scalar(-x)


@_compile
def _polynomial7(c: NDArray[np.float64], r: float) -> float:
    s: float = c[7]
    for k in range(6, -1, -1):
        s = s * r + c[k]
    return s


@_compile
def norm_ppf_scalar(p: float) -> float:
    """Return the standard normal quantile z with P(Z <= z) = p (Wichura 1988, AS 241)."""
    if math.isnan(p) or p < 0.0 or p > 1.0:
        return _NAN
    if p == 0.0:
        return -_INF
    if p == 1.0:
        return _INF
    q = p - 0.5
    if abs(q) <= 0.425:
        r = 0.180625 - q * q
        return q * _polynomial7(_AS241_A, r) / _polynomial7(_AS241_B, r)
    r = math.sqrt(-math.log(p if q < 0.0 else 1.0 - p))
    if r <= 5.0:
        r -= 1.6
        z = _polynomial7(_AS241_C, r) / _polynomial7(_AS241_D, r)
    else:
        r -= 5.0
        z = _polynomial7(_AS241_E, r) / _polynomial7(_AS241_F, r)
    return -z if q < 0.0 else z


@_compile
def norm_isf_scalar(p: float) -> float:
    """Return the z with P(Z > z) = p."""
    return -norm_ppf_scalar(p)


# ---------------------------------------------------------------------------------------
# Student t distribution
# ---------------------------------------------------------------------------------------


@_compile
def _t_two_sided(x: float, df: float) -> tuple[float, float]:
    """Return log P(|T| > |x|) and log P(|T| <= |x|) for T ~ t(df), 0 < df < inf.

    P(|T| > |x|) = I_w(df/2, 1/2) with w = 1 / (1 + r), r = x^2 / df.
    """
    ax = abs(x)
    if ax == _INF:
        return -_INF, 0.0
    a = 0.5 * df
    if ax < 1e150:
        xx = ax * ax
        r = xx / df
    else:
        q = ax / math.sqrt(df)
        r = q * q
        xx = _INF
    if r == _INF:
        # w underflows but log w = -log r does not; lambda = a (1 - w) - w / 2 = a.
        log_r = 2.0 * math.log(ax) - math.log(df)
        return _log_betainc_pair(a, 0.5, 0.0, 1.0, -log_r, 0.0, a)
    w = 1.0 / (1.0 + r)
    v = r / (1.0 + r)  # 1 - w
    log_w = -math.log1p(r)
    if r >= 1.0:
        log_v = -math.log1p(1.0 / r)
    elif r >= _SMALLEST_NORMAL:
        log_v = math.log(r) + log_w
    else:  # a subnormal r has lost bits; its logarithm comes from the inputs
        log_v = 2.0 * math.log(ax) - math.log(df) + log_w
    if xx < _INF:
        lam = 0.5 * w * ((ax - 1.0) * (ax + 1.0))  # a v - w / 2 = w (x^2 - 1) / 2
    else:
        lam = 0.5 * (df * v - w)
    return _log_betainc_pair(a, 0.5, w, v, log_w, log_v, lam)


@_compile
def t_sf_scalar(x: float, df: float) -> float:
    """Return P(T > x) for T ~ t(df); df > 0, df = inf gives the normal distribution."""
    if math.isnan(x) or math.isnan(df) or df <= 0.0:
        return _NAN
    if df == _INF:
        return norm_sf_scalar(x)
    both = math.exp(_t_two_sided(x, df)[0])
    if x > 0.0:
        return 0.5 * both
    return 1.0 - 0.5 * both


@_compile
def t_cdf_scalar(x: float, df: float) -> float:
    """Return P(T <= x) for T ~ t(df)."""
    return t_sf_scalar(-x, df)


@_compile
def t_logsf_scalar(x: float, df: float) -> float:
    """Return log P(T > x) for T ~ t(df)."""
    if math.isnan(x) or math.isnan(df) or df <= 0.0:
        return _NAN
    if df == _INF:
        return norm_logsf_scalar(x)
    log_both = _t_two_sided(x, df)[0]
    if x > 0.0:
        return log_both - _LN2
    return math.log1p(-0.5 * math.exp(log_both))


@_compile
def _t_log_density(t: float, df: float, log_norm: float) -> float:
    """Return log f(t) of the t density for t >= 0, given its log normalising constant."""
    if t < 1e150:
        log_kernel = math.log1p(t * t / df)
    else:
        log_kernel = 2.0 * math.log(t) - math.log(df)
    return log_norm - 0.5 * (df + 1.0) * log_kernel


@_compile
def _t_quantile_residual(
    t: float, df: float, log_norm: float, central: bool, target: float
) -> tuple[float, float]:
    """Return h(t) = log G(t) - target and the Newton step for h in log t.

    G(t) is P(|T| <= t) when central, else P(T > t); d log G / dt is 2 f(t) / G(t) and
    -f(t) / G(t) respectively, with f the t density.
    """
    log_upper, log_central = _t_two_sided(t, df)
    log_density = _t_log_density(t, df, log_norm)
    if central:
        h = log_central - target
        return h, -h * math.exp(log_central - log_density) / (2.0 * t)
    log_s = log_upper - _LN2
    h = log_s - target
    return h, h * math.exp(log_s - log_density) / t


@_compile
def _t_upper_quantile(p: float, df: float) -> float:
    """Return the t > 0 with P(T > t) = p, for 0 < p < 1/2 and 0 < df < inf.

    Returns +inf when the quantile exceeds the largest double, and NaN if the iteration
    does not converge.
    """
    if df == 1.0:  # Cauchy: t = cot(pi p)
        if p < 0.25:
            return 1.0 / math.tan(_PI * p)
        return math.tan(_PI * (0.5 - p))
    if df == 2.0:  # P(T > t) = 1/2 - t / (2 sqrt(2 + t^2))
        return (1.0 - 2.0 * p) / math.sqrt(2.0 * p * (1.0 - p))
    a = 0.5 * df
    if a >= 10.0:
        log_norm = _log_gamma_ratio(a, 0.5) - 0.5 * math.log(df * _PI)
    else:
        log_norm = math.lgamma(a + 0.5) - math.lgamma(a) - 0.5 * math.log(df * _PI)
    # Near p = 1/2 the quantile is small and log P(T > t) is close to -log 2, so solving
    # for it would lose the relative precision of t. There the equation is solved for the
    # central probability P(|T| <= t) = 1 - 2p instead, exact for p >= 1/4.
    central = p > 0.25
    target = math.log(1.0 - 2.0 * p) if central else math.log(p)

    # Start from the better of the Cornish-Fisher expansion about the normal quantile
    # (Abramowitz and Stegun 26.7.5) and the power-law tail P(T > t) ~ C t^(-df), with
    # C = exp(log_norm) df^((df - 1) / 2) from integrating the density's leading term.
    z = -norm_ppf_scalar(p)
    z2 = z * z
    g1 = (z2 + 1.0) * z / 4.0
    g2 = ((5.0 * z2 + 16.0) * z2 + 3.0) * z / 96.0
    g3 = (((3.0 * z2 + 19.0) * z2 + 17.0) * z2 - 15.0) * z / 384.0
    g4 = (
        ((((79.0 * z2 + 776.0) * z2 + 1482.0) * z2 - 1920.0) * z2 - 945.0) * z / 92160.0
    )
    t = z + (g1 + (g2 + (g3 + g4 / df) / df) / df) / df
    if not (0.0 < t <= _LARGEST):
        t = z
    h, step = _t_quantile_residual(t, df, log_norm, central, target)
    log_tail = (log_norm + 0.5 * (df - 1.0) * math.log(df) - math.log(p)) / df
    t_tail = _LARGEST if log_tail >= _LOG_LARGEST else math.exp(log_tail)
    h_tail, step_tail = _t_quantile_residual(t_tail, df, log_norm, central, target)
    if abs(h_tail) < abs(h):
        t, h, step = t_tail, h_tail, step_tail

    # Newton's method on h in log t (for the upper tail additive once t <= 1), where log G
    # is close to linear, inside a bracket [low, high] that starts at the largest double;
    # a step that leaves the bracket is replaced by bisection, geometric when both ends
    # are positive.
    low = 0.0
    high = _LARGEST
    previous = _INF
    for _ in range(200):
        if h == 0.0:
            return t
        if (h > 0.0) == central:  # G(t) is above its target: t is too large
            high = t
        elif t >= _LARGEST:
            return _INF  # the quantile exceeds the largest double
        else:
            low = t
        if high - low <= 4.0 * _EPS * high:
            return t  # the bracket has closed to a few units in the last place
        if central or t > 1.0:
            t_new = t * math.exp(min(max(step, -20.0), 20.0))
        else:
            t_new = t + step * t
        change = abs(t_new - t)
        if change <= 4.0 * _EPS * t:
            return t_new  # the Newton step is at the level of rounding
        if low < t_new < high:
            # A Newton step that has stopped shrinking has reached the rounding noise
            # of log G(t).
            if change <= 1e-10 * t and change >= 0.5 * previous:
                return t_new
            previous = change
        else:
            t_new = math.sqrt(low) * math.sqrt(high) if low > 0.0 else 0.5 * high
            previous = _INF
        t = t_new
        h, step = _t_quantile_residual(t, df, log_norm, central, target)
    # Not converged: the quantile is beyond the largest double if G there is still on the
    # side of the target that calls for a larger t; anything else is a failure.
    h, _ = _t_quantile_residual(_LARGEST, df, log_norm, central, target)
    return _INF if (h > 0.0) != central else _NAN


@_compile
def t_isf_scalar(p: float, df: float) -> float:
    """Return the t with P(T > t) = p for T ~ t(df)."""
    if math.isnan(p) or math.isnan(df) or p < 0.0 or p > 1.0 or df <= 0.0:
        return _NAN
    if df == _INF:
        return norm_isf_scalar(p)
    if p == 0.0:
        return _INF
    if p == 1.0:
        return -_INF
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -_t_upper_quantile(1.0 - p, df)  # 1 - p is exact
    return _t_upper_quantile(p, df)


@_compile
def t_ppf_scalar(p: float, df: float) -> float:
    """Return the t with P(T <= t) = p for T ~ t(df)."""
    return -t_isf_scalar(p, df)


# ---------------------------------------------------------------------------------------
# F distribution
# ---------------------------------------------------------------------------------------


@_compile
def _f_tails(x: float, dfn: float, dfd: float) -> tuple[float, float]:
    """Return log P(F > x) and log P(F <= x) for F(dfn, dfd); NaN outside the domain."""
    if math.isnan(x) or math.isnan(dfn) or math.isnan(dfd) or dfn <= 0.0 or dfd <= 0.0:
        return _NAN, _NAN
    if x <= 0.0:
        return 0.0, -_INF
    if x == _INF:
        return -_INF, 0.0
    if dfd == _INF:
        if dfn == _INF:  # point mass at 1
            return (0.0, -_INF) if x < 1.0 else (-_INF, 0.0)
        # dfn F ~ chi-squared(dfn): P(F > x) = Q(dfn / 2, dfn x / 2)
        a = 0.5 * dfn
        z = a * x
        log_z = math.log(z) if z >= _SMALLEST_NORMAL else math.log(a) + math.log(x)
        log_p, log_q = _log_gammainc_pair(a, z, log_z, a * (1.0 - x))
        return log_q, log_p
    if dfn == _INF:
        # dfd / F ~ chi-squared(dfd): P(F > x) = P(dfd / 2, dfd / (2 x))
        a = 0.5 * dfd
        z = a / x
        log_z = math.log(z) if z >= _SMALLEST_NORMAL else math.log(a) - math.log(x)
        log_p, log_q = _log_gammainc_pair(a, z, log_z, a * (x - 1.0) / x)
        return log_p, log_q
    # P(F > x) = I_w(dfd / 2, dfn / 2), w = dfd / (dfd + dfn x) = 1 / (1 + r)
    r = dfn * x / dfd
    if not (r < _INF):
        r = x * (dfn / dfd)
    if r == _INF:
        log_r = math.log(dfn) + math.log(x) - math.log(dfd)
        return _log_betainc_pair(0.5 * dfd, 0.5 * dfn, 0.0, 1.0, -log_r, 0.0, 0.5 * dfd)
    w = 1.0 / (1.0 + r)
    v = r / (1.0 + r)
    log_w = -math.log1p(r)
    if r >= 1.0:
        log_v = -math.log1p(1.0 / r)
    elif r >= _SMALLEST_NORMAL:
        log_v = math.log(r) + log_w
    else:  # a subnormal r has lost bits; its logarithm comes from the inputs
        log_v = math.log(dfn) + math.log(x) - math.log(dfd) + log_w
    lam = 0.5 * dfn * w * (x - 1.0)  # (dfd / 2) v - (dfn / 2) w
    return _log_betainc_pair(0.5 * dfd, 0.5 * dfn, w, v, log_w, log_v, lam)


@_compile
def f_cdf_scalar(x: float, dfn: float, dfd: float) -> float:
    """Return P(F <= x) for F ~ F(dfn, dfd)."""
    return math.exp(_f_tails(x, dfn, dfd)[1])


@_compile
def f_sf_scalar(x: float, dfn: float, dfd: float) -> float:
    """Return P(F > x) for F ~ F(dfn, dfd); dfn or dfd may be inf."""
    return math.exp(_f_tails(x, dfn, dfd)[0])


@_compile
def f_logsf_scalar(x: float, dfn: float, dfd: float) -> float:
    """Return log P(F > x) for F ~ F(dfn, dfd)."""
    return _f_tails(x, dfn, dfd)[0]


@_compile
def _f_log_x_density(x: float, dfn: float, dfd: float) -> float:
    """Return log(x f(x)) for the density f of F(dfn, dfd), x > 0 and finite dfn.

    With a = dfn / 2, b = dfd / 2 and v = dfn x / (dfd + dfn x) = 1 - w,
    x f(x) = v^a w^b / B(a, b), formed with Loader's deviance terms; for dfd = inf,
    dfn F is chi-squared(dfn) and x f(x) = z^a exp(-z) / Gamma(a) with z = a x.
    """
    a = 0.5 * dfn
    if dfd == _INF:
        z = a * x
        log_z = math.log(z) if z >= _SMALLEST_NORMAL else math.log(a) + math.log(x)
        return math.log(a) + _log_gamma_prefactor(a, z, log_z, a * (1.0 - x))
    r = dfn * x / dfd
    if not (r < _INF):
        r = x * (dfn / dfd)
    if r == _INF:
        return -_INF
    w = 1.0 / (1.0 + r)
    v = r / (1.0 + r)
    log_w = -math.log1p(r)
    if r >= 1.0:
        log_v = -math.log1p(1.0 / r)
    elif r >= _SMALLEST_NORMAL:
        log_v = math.log(r) + log_w
    else:  # a subnormal r has lost bits; its logarithm comes from the inputs
        log_v = math.log(dfn) + math.log(x) - math.log(dfd) + log_w
    lam = 0.5 * dfn * w * (1.0 - x)  # a w - b v
    return _log_beta_prefactor(a, 0.5 * dfd, v, w, log_v, log_w, lam)


@_compile
def _f_quantile_residual(
    x: float, dfn: float, dfd: float, upper: bool, target: float
) -> tuple[float, float]:
    """Return h = log G(x) - target and the Newton step for h in log x.

    G(x) is P(F > x) when upper, else P(F <= x); d log G / d log x is -x f(x) / G(x)
    and x f(x) / G(x) respectively.
    """
    log_sf, log_cdf = _f_tails(x, dfn, dfd)
    log_tail = log_sf if upper else log_cdf
    h = log_tail - target
    ratio = math.exp(min(log_tail - _f_log_x_density(x, dfn, dfd), _LOG_LARGEST))
    return h, (h * ratio if upper else -h * ratio)


@_compile
def _f_quantile_finish(
    x: float, dfn: float, dfd: float, upper: bool, target: float, high: float
) -> float:
    """Return the converged x, or 0 or inf when the quantile is beyond the doubles.

    A subnormal x becomes 0 if the quantile is below the smallest positive double. An
    x near the largest double with no evaluated point above the quantile (``high``
    still the largest double) becomes inf if the quantile exceeds the largest double.
    """
    if 0.0 < x < _SMALLEST_NORMAL:
        h = _f_quantile_residual(_SMALLEST_POSITIVE, dfn, dfd, upper, target)[0]
        if (h > 0.0) != upper:  # the smallest positive double is above the quantile
            return 0.0
    elif high == _LARGEST and x >= 0.5 * _LARGEST:
        h = _f_quantile_residual(_LARGEST, dfn, dfd, upper, target)[0]
        if (h > 0.0) == upper:  # the largest double is below the quantile
            return _INF
    return x


@_compile
def _f_tail_quantile(log_p: float, upper: bool, dfn: float, dfd: float) -> float:
    """Return the x > 0 with log P(F > x) = log_p if upper, else log P(F <= x) = log_p.

    For log_p <= log(1/2), 0 < dfn < inf and 0 < dfd <= inf; the probability may be
    below the smallest double. Returns +inf when the quantile exceeds the largest
    double, 0 when it is below the smallest, and NaN if the iteration does not
    converge.
    """
    target = log_p
    a = 0.5 * dfn
    # Start from the better of Paulson's normal approximation to F^(1/3) and the power
    # law of the tail that holds p (Johnson, Kotz and Balakrishnan 1995, 27.5).
    z = norm_isf_scalar(math.exp(log_p))
    if not upper:
        z = -z
    c1 = 2.0 / (9.0 * dfn)
    c2 = 0.0 if dfd == _INF else 2.0 / (9.0 * dfd)
    quad_a = (1.0 - c2) * (1.0 - c2) - z * z * c2
    quad_b = -2.0 * (1.0 - c1) * (1.0 - c2)
    quad_c = (1.0 - c1) * (1.0 - c1) - z * z * c1
    discriminant = quad_b * quad_b - 4.0 * quad_a * quad_c
    x = 1.0
    if quad_a > 0.0 and discriminant >= 0.0:
        root = math.sqrt(discriminant)
        # the root with (1 - c2) y - (1 - c1) of the sign of z
        y = (
            (-quad_b + root) / (2.0 * quad_a)
            if z >= 0.0
            else (-quad_b - root) / (2.0 * quad_a)
        )
        if y > 0.0:
            x = y * y * y
    if not (0.0 < x <= _LARGEST):
        x = 1.0
    h, step = _f_quantile_residual(x, dfn, dfd, upper, target)
    if dfd == _INF:
        if upper:
            log_tail_start = _NAN
        else:  # P(X <= x) ~ (a x)^a / Gamma(a + 1) as x -> 0
            log_tail_start = (target + math.lgamma(a + 1.0)) / a - math.log(a)
    else:
        b = 0.5 * dfd
        log_beta = _log_beta_positive(a, b)
        if upper:  # P(F > x) ~ (dfd / (dfn x))^b / (b B(a, b)) as x -> inf
            log_tail_start = math.log(dfd / dfn) - (target + math.log(b) + log_beta) / b
        else:  # P(F <= x) ~ (dfn x / dfd)^a / (a B(a, b)) as x -> 0
            log_tail_start = math.log(dfd / dfn) + (target + math.log(a) + log_beta) / a
    if not math.isnan(log_tail_start):
        x_tail = (
            _LARGEST if log_tail_start >= _LOG_LARGEST else math.exp(log_tail_start)
        )
        if x_tail > 0.0:
            h_tail, step_tail = _f_quantile_residual(x_tail, dfn, dfd, upper, target)
            if abs(h_tail) < abs(h):
                x, h, step = x_tail, h_tail, step_tail

    # Newton's method in log x, where log G is close to linear in the tails, inside a
    # bracket [low, high]; a step that leaves the bracket is replaced by geometric
    # bisection (as for the t quantile).
    low = 0.0
    high = _LARGEST
    previous = _INF
    for _ in range(300):
        if h == 0.0:
            return x
        if (h > 0.0) == upper:  # G(x) is above its target on the side of small x
            if x >= _LARGEST:
                return _INF
            low = x
        else:
            high = x
        # Near and below the smallest normal double the spacing of the doubles is
        # absolute, so the tests allow a few units of the smallest subnormal.
        if high - low <= 4.0 * _EPS * high + _SUBNORMAL_SLACK:
            return _f_quantile_finish(x, dfn, dfd, upper, target, high)
        x_new = x * math.exp(min(max(step, -20.0), 20.0))
        change = abs(x_new - x)
        if change <= 4.0 * _EPS * x + _SUBNORMAL_SLACK:
            return _f_quantile_finish(x_new, dfn, dfd, upper, target, high)
        if low < x_new < high:
            # A Newton step that has stopped shrinking has reached the rounding noise
            # of log G(x).
            if change <= 1e-10 * x and change >= 0.5 * previous:
                return _f_quantile_finish(x_new, dfn, dfd, upper, target, high)
            previous = change
        else:
            x_new = math.sqrt(low) * math.sqrt(high) if low > 0.0 else 0.5 * high
            previous = _INF
            if x_new == 0.0:
                return 0.0  # the quantile is below the smallest positive double
        x = x_new
        h, step = _f_quantile_residual(x, dfn, dfd, upper, target)
    return _NAN


@_compile
def _f_quantile(p: float, upper: bool, dfn: float, dfd: float) -> float:
    """Return the x with P(F > x) = p if upper, else P(F <= x) = p, for 0 < p < 1.

    The tail that holds min(p, 1 - p) is solved; 1 - p is exact for p >= 1/2.
    """
    if dfn == _INF:
        if dfd == _INF:  # point mass at 1
            return 1.0
        # 1 / F ~ F(dfd, dfn): P(F > x) = P(1 / F < 1 / x)
        dfn, dfd = dfd, dfn
        upper = not upper
        if p <= 0.5:
            return 1.0 / _f_tail_quantile(math.log(p), upper, dfn, dfd)
        return 1.0 / _f_tail_quantile(math.log1p(-p), not upper, dfn, dfd)
    if p <= 0.5:
        return _f_tail_quantile(math.log(p), upper, dfn, dfd)
    return _f_tail_quantile(math.log1p(-p), not upper, dfn, dfd)


@_compile
def f_isf_scalar(p: float, dfn: float, dfd: float) -> float:
    """Return the x with P(F > x) = p for F ~ F(dfn, dfd); dfn or dfd may be inf."""
    if math.isnan(p) or math.isnan(dfn) or math.isnan(dfd):
        return _NAN
    if p < 0.0 or p > 1.0 or dfn <= 0.0 or dfd <= 0.0:
        return _NAN
    if p == 0.0:
        return _INF
    if p == 1.0:
        return 0.0
    return _f_quantile(p, True, dfn, dfd)


@_compile
def f_ppf_scalar(p: float, dfn: float, dfd: float) -> float:
    """Return the x with P(F <= x) = p for F ~ F(dfn, dfd); dfn or dfd may be inf."""
    if math.isnan(p) or math.isnan(dfn) or math.isnan(dfd):
        return _NAN
    if p < 0.0 or p > 1.0 or dfn <= 0.0 or dfd <= 0.0:
        return _NAN
    if p == 0.0:
        return 0.0
    if p == 1.0:
        return _INF
    return _f_quantile(p, False, dfn, dfd)


# ---------------------------------------------------------------------------------------
# Chi-squared and gamma distributions
# ---------------------------------------------------------------------------------------


@_compile
def _chi2_tails(x: float, df: float) -> tuple[float, float]:
    """Return log P(X <= x) and log P(X > x) for chi-squared(df); NaN outside the domain."""
    if math.isnan(x) or math.isnan(df) or not (0.0 < df < _INF):
        return _NAN, _NAN
    if x <= 0.0:
        return -_INF, 0.0
    z = 0.5 * x
    log_z = math.log(z) if z >= _SMALLEST_NORMAL else math.log(x) - _LN2
    return _log_gammainc_pair(0.5 * df, z, log_z, 0.5 * (df - x))


@_compile
def chi2_cdf_scalar(x: float, df: float) -> float:
    """Return P(X <= x) for X ~ chi-squared(df)."""
    return math.exp(_chi2_tails(x, df)[0])


@_compile
def chi2_sf_scalar(x: float, df: float) -> float:
    """Return P(X > x) for X ~ chi-squared(df)."""
    return math.exp(_chi2_tails(x, df)[1])


@_compile
def chi2_logsf_scalar(x: float, df: float) -> float:
    """Return log P(X > x) for X ~ chi-squared(df)."""
    return _chi2_tails(x, df)[1]


@_compile
def chi2_isf_scalar(p: float, df: float) -> float:
    """Return the x with P(X > x) = p for X ~ chi-squared(df), 0 < df < inf."""
    if math.isnan(df) or not (0.0 < df < _INF):
        return _NAN
    # X / df ~ F(df, inf)
    return df * f_isf_scalar(p, df, _INF)


@_compile
def chi2_ppf_scalar(p: float, df: float) -> float:
    """Return the x with P(X <= x) = p for X ~ chi-squared(df), 0 < df < inf."""
    if math.isnan(df) or not (0.0 < df < _INF):
        return _NAN
    return df * f_ppf_scalar(p, df, _INF)


@_compile
def _gamma_tails(x: float, shape: float, scale: float) -> tuple[float, float]:
    """Return log P(X <= x) and log P(X > x) for Gamma(shape, scale); NaN off the domain."""
    if (
        math.isnan(x)
        or math.isnan(shape)
        or math.isnan(scale)
        or not (0.0 < shape < _INF)
        or not (0.0 < scale < _INF)
    ):
        return _NAN, _NAN
    if x <= 0.0:
        return -_INF, 0.0
    z = x / scale
    log_z = math.log(z) if z >= _SMALLEST_NORMAL else math.log(x) - math.log(scale)
    # shape - x / scale = (shape scale - x) / scale, with shape scale formed exactly
    p, p_low = _two_product(shape, scale)
    d = ((p - x) + p_low) / scale if p < _INF else shape - z
    return _log_gammainc_pair(shape, z, log_z, d)


@_compile
def gamma_cdf_scalar(x: float, shape: float, scale: float) -> float:
    """Return P(X <= x) for X ~ Gamma(shape, scale)."""
    return math.exp(_gamma_tails(x, shape, scale)[0])


@_compile
def gamma_sf_scalar(x: float, shape: float, scale: float) -> float:
    """Return P(X > x) for X ~ Gamma(shape, scale)."""
    return math.exp(_gamma_tails(x, shape, scale)[1])


# ---------------------------------------------------------------------------------------
# Beta distribution
# ---------------------------------------------------------------------------------------


@_compile
def _beta_tails(x: float, a: float, b: float) -> tuple[float, float]:
    """Return log P(X <= x) and log P(X > x) for Beta(a, b); NaN outside the domain."""
    if (
        math.isnan(x)
        or math.isnan(a)
        or math.isnan(b)
        or not (0.0 < a < _INF)
        or not (0.0 < b < _INF)
    ):
        return _NAN, _NAN
    if x <= 0.0:
        return -_INF, 0.0
    if x >= 1.0:
        return 0.0, -_INF
    return _log_betainc_pair_at(a, b, x)


@_compile
def beta_cdf_scalar(x: float, a: float, b: float) -> float:
    """Return P(X <= x) for X ~ Beta(a, b)."""
    return math.exp(_beta_tails(x, a, b)[0])


@_compile
def beta_sf_scalar(x: float, a: float, b: float) -> float:
    """Return P(X > x) for X ~ Beta(a, b)."""
    return math.exp(_beta_tails(x, a, b)[1])


# ---------------------------------------------------------------------------------------
# Hypergeometric distribution
# ---------------------------------------------------------------------------------------


@_compile
def _log_binomial_probability(
    k: float, n: float, p: float, q: float, d: float
) -> float:
    """Return log[C(n, k) p^k q^(n - k)] for integers 0 <= k <= n (Loader 2000).

    q = 1 - p and d = k - n p are formed accurately by the caller.
    """
    if n == 0.0:
        return 0.0
    if k == 0.0:
        return n * (math.log1p(-p) if p < 0.5 else math.log(q))
    if k == n:
        return n * (math.log1p(-q) if q < 0.5 else math.log(p))
    return (
        _stirling_error(n)
        - _stirling_error(k)
        - _stirling_error(n - k)
        - _deviance_term(k, n * p, d)
        - _deviance_term(n - k, n * q, -d)
        + 0.5 * math.log(n / (k * (n - k)) / (2.0 * _PI))
    )


@_compile
def _hypergeom_log_probability(k: float, M: float, n: float, N: float) -> float:
    """Return log P(X = k) for k in the support, 0 < N < M (Loader 2000).

    P(X = k) = b(k; n, p) b(N - k; M - n, p) / b(N; M, p) for any p; with p = N / M the
    deviations from the binomial means are d = (k M - n N) / M and -d, and 0 for the
    denominator.
    """
    p = N / M
    q = (M - N) / M
    # k M - n N from the exact products (Dekker): in a large population the two products
    # exceed 2^53 and agree in their leading digits, so the difference of the rounded
    # products would leave d with a relative error of eps k / |d|, which the deviance
    # carries into log P(X = k) multiplied by the square of the number of standard
    # deviations between k and the mean.
    km, km_error = _two_product(k, M)
    nn, nn_error = _two_product(n, N)
    d = ((km - nn) + (km_error - nn_error)) / M
    return (
        _log_binomial_probability(k, n, p, q, d)
        + _log_binomial_probability(N - k, M - n, p, q, -d)
        - _log_binomial_probability(N, M, p, q, 0.0)
    )


@_compile
def _hypergeom_log_tails(k: float, M: float, n: float, N: float) -> tuple[float, float]:
    """Return log P(X > k) and log P(X <= k) for Hypergeom(M, n, N); NaN off the domain."""
    if math.isnan(k) or math.isnan(M) or math.isnan(n) or math.isnan(N):
        return _NAN, _NAN
    # Integers above 2^53 are not all doubles, so counts there are not exact.
    if not (0.0 <= M < _EXACT_INTEGERS and 0.0 <= n <= M and 0.0 <= N <= M):
        return _NAN, _NAN
    if M != np.floor(M) or n != np.floor(n) or N != np.floor(N):
        return _NAN, _NAN
    k = np.floor(k)
    low = max(0.0, N - (M - n))
    high = min(n, N)
    if k < low:
        return 0.0, -_INF
    if k >= high:
        return -_INF, 0.0
    mode = np.floor((N + 1.0) * (n + 1.0) / (M + 2.0))
    # Each tail is summed outwards from its largest term, relative to it. Near the mode
    # of a large population the ratio r of successive terms is close to 1, so the terms
    # fall slowly: the tail after a term is many times that term, and many terms fall
    # below the last digit of the running total. The probabilities are log-concave, so
    # r only falls on the way out and the rest of the tail is at most term r / (1 - r);
    # the sums stop when that bound is negligible, and the rounding error of every
    # addition is carried along (Fast2Sum; the total is never below the term).
    if k + 1.0 >= mode:
        # Upper tail: the probabilities fall from j = k + 1 on.
        j = k + 1.0
        log_first = _hypergeom_log_probability(j, M, n, N)
        total = 1.0
        carried = 0.0
        term = 1.0
        steps = 0
        while j < high:
            ratio = (n - j) * (N - j) / ((j + 1.0) * (M - n - N + j + 1.0))
            term *= ratio
            new_total = total + term
            carried += (total - new_total) + term
            total = new_total
            if term * ratio <= 0.25 * _EPS * total * (1.0 - ratio):
                break
            j += 1.0
            steps += 1
            if steps > _MAX_TERMS:
                return _NAN, _NAN
        log_sf = min(log_first + math.log(total + carried), 0.0)
        return log_sf, _log1mexp(-log_sf)
    # Lower tail: the probabilities fall from j = k down.
    j = k
    log_first = _hypergeom_log_probability(j, M, n, N)
    total = 1.0
    carried = 0.0
    term = 1.0
    steps = 0
    while j > low:
        ratio = j * (M - n - N + j) / ((n - j + 1.0) * (N - j + 1.0))
        term *= ratio
        new_total = total + term
        carried += (total - new_total) + term
        total = new_total
        if term * ratio <= 0.25 * _EPS * total * (1.0 - ratio):
            break
        j -= 1.0
        steps += 1
        if steps > _MAX_TERMS:
            return _NAN, _NAN
    log_cdf = min(log_first + math.log(total + carried), 0.0)
    return _log1mexp(-log_cdf), log_cdf


@_compile
def hypergeom_sf_scalar(k: float, M: float, n: float, N: float) -> float:
    """Return P(X > k) for X ~ Hypergeom(M, n, N)."""
    return math.exp(_hypergeom_log_tails(k, M, n, N)[0])


@_compile
def hypergeom_logsf_scalar(k: float, M: float, n: float, N: float) -> float:
    """Return log P(X > k) for X ~ Hypergeom(M, n, N)."""
    return _hypergeom_log_tails(k, M, n, N)[0]


# ---------------------------------------------------------------------------------------
# Array loops
# ---------------------------------------------------------------------------------------


@_compile
def _norm_array(x: NDArray[np.float64], kind: int) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        if kind == _CDF:
            out[i] = norm_cdf_scalar(x[i])
        elif kind == _SF:
            out[i] = norm_sf_scalar(x[i])
        elif kind == _LOGCDF:
            out[i] = norm_logcdf_scalar(x[i])
        elif kind == _LOGSF:
            out[i] = norm_logsf_scalar(x[i])
        elif kind == _PPF:
            out[i] = norm_ppf_scalar(x[i])
        else:
            out[i] = norm_isf_scalar(x[i])
    return out


@_compile
def _t_array(
    x: NDArray[np.float64], df: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        if kind == _CDF:
            out[i] = t_cdf_scalar(x[i], df[i])
        elif kind == _SF:
            out[i] = t_sf_scalar(x[i], df[i])
        elif kind == _LOGSF:
            out[i] = t_logsf_scalar(x[i], df[i])
        elif kind == _PPF:
            out[i] = t_ppf_scalar(x[i], df[i])
        else:
            out[i] = t_isf_scalar(x[i], df[i])
    return out


@_compile
def _f_array(
    x: NDArray[np.float64],
    dfn: NDArray[np.float64],
    dfd: NDArray[np.float64],
    kind: int,
) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        if kind == _PPF:
            out[i] = f_ppf_scalar(x[i], dfn[i], dfd[i])
        elif kind == _ISF:
            out[i] = f_isf_scalar(x[i], dfn[i], dfd[i])
        else:
            log_sf, log_cdf = _f_tails(x[i], dfn[i], dfd[i])
            if kind == _CDF:
                out[i] = math.exp(log_cdf)
            elif kind == _SF:
                out[i] = math.exp(log_sf)
            else:
                out[i] = log_sf
    return out


@_compile
def _chi2_array(
    x: NDArray[np.float64], df: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        if kind == _PPF:
            out[i] = chi2_ppf_scalar(x[i], df[i])
        elif kind == _ISF:
            out[i] = chi2_isf_scalar(x[i], df[i])
        else:
            log_cdf, log_sf = _chi2_tails(x[i], df[i])
            if kind == _CDF:
                out[i] = math.exp(log_cdf)
            elif kind == _SF:
                out[i] = math.exp(log_sf)
            else:
                out[i] = log_sf
    return out


@_compile
def _gamma_array(
    x: NDArray[np.float64],
    shape: NDArray[np.float64],
    scale: NDArray[np.float64],
    kind: int,
) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        log_cdf, log_sf = _gamma_tails(x[i], shape[i], scale[i])
        out[i] = math.exp(log_cdf if kind == _CDF else log_sf)
    return out


@_compile
def _beta_array(
    x: NDArray[np.float64], a: NDArray[np.float64], b: NDArray[np.float64], kind: int
) -> NDArray[np.float64]:
    out = np.empty(x.size)
    for i in range(x.size):
        log_cdf, log_sf = _beta_tails(x[i], a[i], b[i])
        out[i] = math.exp(log_cdf if kind == _CDF else log_sf)
    return out


@_compile
def _hypergeom_array(
    k: NDArray[np.float64],
    M: NDArray[np.float64],
    n: NDArray[np.float64],
    N: NDArray[np.float64],
    kind: int,
) -> NDArray[np.float64]:
    out = np.empty(k.size)
    for i in range(k.size):
        log_sf = _hypergeom_log_tails(k[i], M[i], n[i], N[i])[0]
        out[i] = math.exp(log_sf) if kind == _SF else log_sf
    return out


# ---------------------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------------------


def norm_cdf(x: ArrayLike) -> float | NDArray[np.float64]:
    """Cumulative distribution function of the standard normal distribution, P(Z <= x).

    Args:
        x: Quantile.

    Returns:
        P(Z <= x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return _apply(_norm_array, _CDF, x=x)


def norm_sf(x: ArrayLike) -> float | NDArray[np.float64]:
    """Survival function of the standard normal distribution, P(Z > x).

    Args:
        x: Quantile.

    Returns:
        P(Z > x), with full relative accuracy in the upper tail, as a float or a float64
        array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return _apply(_norm_array, _SF, x=x)


def norm_logcdf(x: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the standard normal distribution function, log P(Z <= x).

    Args:
        x: Quantile; finite down to x of about -1.9e154.

    Returns:
        log P(Z <= x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return _apply(_norm_array, _LOGCDF, x=x)


def norm_logsf(x: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the standard normal survival function, log P(Z > x).

    Args:
        x: Quantile; finite up to x of about 1.9e154.

    Returns:
        log P(Z > x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``x`` is not real-valued.
    """
    return _apply(_norm_array, _LOGSF, x=x)


def norm_ppf(p: ArrayLike) -> float | NDArray[np.float64]:
    """Quantile function of the standard normal distribution: the z with P(Z <= z) = p.

    Args:
        p: Probability in [0, 1]; 0 gives -inf, 1 gives +inf, values outside give NaN.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``p`` is not real-valued.
    """
    return _apply(_norm_array, _PPF, p=p)


def norm_isf(p: ArrayLike) -> float | NDArray[np.float64]:
    """Inverse survival function of the standard normal distribution, P(Z > z) = p.

    Args:
        p: Probability in [0, 1]; 0 gives +inf, 1 gives -inf, values outside give NaN.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If ``p`` is not real-valued.
    """
    return _apply(_norm_array, _ISF, p=p)


def t_cdf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Cumulative distribution function of Student's t distribution, P(T <= x).

    Args:
        x: Quantile.
        df: Degrees of freedom, df > 0, not necessarily an integer; df = inf gives the
            standard normal distribution. df <= 0 gives NaN.

    Returns:
        P(T <= x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_t_array, _CDF, x=x, df=df)


def t_sf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Survival function of Student's t distribution, P(T > x).

    Args:
        x: Quantile.
        df: Degrees of freedom, df > 0, not necessarily an integer; df = inf gives the
            standard normal distribution. df <= 0 gives NaN.

    Returns:
        P(T > x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_t_array, _SF, x=x, df=df)


def t_logsf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the survival function of Student's t distribution, log P(T > x).

    Args:
        x: Quantile.
        df: Degrees of freedom, df > 0, not necessarily an integer; df = inf gives the
            standard normal distribution. df <= 0 gives NaN.

    Returns:
        log P(T > x), as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_t_array, _LOGSF, x=x, df=df)


def t_ppf(p: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Quantile function of Student's t distribution: the t with P(T <= t) = p.

    With few degrees of freedom the quantile of a small p can be below minus the largest
    double; the result is then -inf.

    Args:
        p: Probability in [0, 1]; 0 gives -inf, 1 gives +inf, values outside give NaN.
        df: Degrees of freedom, df > 0, not necessarily an integer; df = inf gives the
            standard normal distribution. df <= 0 gives NaN.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_t_array, _PPF, p=p, df=df)


def t_isf(p: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Inverse survival function of Student's t distribution: the t with P(T > t) = p.

    Accurate for p down to the smallest doubles; use this rather than ``t_ppf(1 - p)``
    for upper-tail quantiles. With few degrees of freedom the quantile of a small p can
    exceed the largest double; the result is then +inf.

    Args:
        p: Probability in [0, 1]; 0 gives +inf, 1 gives -inf, values outside give NaN.
        df: Degrees of freedom, df > 0, not necessarily an integer; df = inf gives the
            standard normal distribution. df <= 0 gives NaN.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_t_array, _ISF, p=p, df=df)


def f_cdf(x: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> float | NDArray[np.float64]:
    """Cumulative distribution function of the F distribution, P(F <= x).

    Args:
        x: Quantile; x <= 0 gives 0.
        dfn: Numerator degrees of freedom, dfn > 0; inf gives dfd / F ~ chi-squared(dfd).
        dfd: Denominator degrees of freedom, dfd > 0; inf gives dfn F ~ chi-squared(dfn).

    Returns:
        P(F <= x), as a float or a float64 array of the broadcast shape. Degrees of
        freedom <= 0 give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_f_array, _CDF, x=x, dfn=dfn, dfd=dfd)


def f_sf(x: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> float | NDArray[np.float64]:
    """Survival function of the F distribution, P(F > x).

    Args:
        x: Quantile; x <= 0 gives 1.
        dfn: Numerator degrees of freedom, dfn > 0; inf gives dfd / F ~ chi-squared(dfd).
        dfd: Denominator degrees of freedom, dfd > 0; inf gives dfn F ~ chi-squared(dfn).

    Returns:
        P(F > x), as a float or a float64 array of the broadcast shape. Degrees of
        freedom <= 0 give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_f_array, _SF, x=x, dfn=dfn, dfd=dfd)


def f_logsf(
    x: ArrayLike, dfn: ArrayLike, dfd: ArrayLike
) -> float | NDArray[np.float64]:
    """Logarithm of the survival function of the F distribution, log P(F > x).

    Args:
        x: Quantile; x <= 0 gives 0.
        dfn: Numerator degrees of freedom, dfn > 0; inf gives dfd / F ~ chi-squared(dfd).
        dfd: Denominator degrees of freedom, dfd > 0; inf gives dfn F ~ chi-squared(dfn).

    Returns:
        log P(F > x), as a float or a float64 array of the broadcast shape. Degrees of
        freedom <= 0 give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_f_array, _LOGSF, x=x, dfn=dfn, dfd=dfd)


def f_ppf(p: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> float | NDArray[np.float64]:
    """Quantile function of the F distribution: the x with P(F <= x) = p.

    Accurate in both tails: for p above 1/2 the upper tail 1 - p, which is exact, is
    solved instead.

    Args:
        p: Probability in [0, 1]; 0 gives 0, 1 gives +inf, values outside give NaN.
        dfn: Numerator degrees of freedom, dfn > 0; inf gives dfd / F ~ chi-squared(dfd).
        dfd: Denominator degrees of freedom, dfd > 0; inf gives dfn F ~ chi-squared(dfn).

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape. Degrees of
        freedom <= 0 give NaN; a quantile beyond the largest double gives +inf.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_f_array, _PPF, p=p, dfn=dfn, dfd=dfd)


def f_isf(p: ArrayLike, dfn: ArrayLike, dfd: ArrayLike) -> float | NDArray[np.float64]:
    """Inverse survival function of the F distribution: the x with P(F > x) = p.

    Accurate for p down to the smallest doubles; use this rather than
    ``f_ppf(1 - p)`` for upper-tail quantiles.

    Args:
        p: Probability in [0, 1]; 0 gives +inf, 1 gives 0, values outside give NaN.
        dfn: Numerator degrees of freedom, dfn > 0; inf gives dfd / F ~ chi-squared(dfd).
        dfd: Denominator degrees of freedom, dfd > 0; inf gives dfn F ~ chi-squared(dfn).

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape. Degrees of
        freedom <= 0 give NaN; a quantile beyond the largest double gives +inf.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_f_array, _ISF, p=p, dfn=dfn, dfd=dfd)


def chi2_cdf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Cumulative distribution function of the chi-squared distribution, P(X <= x).

    Args:
        x: Quantile; x <= 0 gives 0.
        df: Degrees of freedom, 0 < df < inf, not necessarily an integer.

    Returns:
        P(X <= x), as a float or a float64 array of the broadcast shape. df outside
        (0, inf) gives NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_chi2_array, _CDF, x=x, df=df)


def chi2_sf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Survival function of the chi-squared distribution, P(X > x).

    Args:
        x: Quantile; x <= 0 gives 1.
        df: Degrees of freedom, 0 < df < inf, not necessarily an integer.

    Returns:
        P(X > x), as a float or a float64 array of the broadcast shape. df outside
        (0, inf) gives NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_chi2_array, _SF, x=x, df=df)


def chi2_logsf(x: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Logarithm of the survival function of the chi-squared distribution, log P(X > x).

    Args:
        x: Quantile; x <= 0 gives 0.
        df: Degrees of freedom, 0 < df < inf, not necessarily an integer.

    Returns:
        log P(X > x), as a float or a float64 array of the broadcast shape. df outside
        (0, inf) gives NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_chi2_array, _LOGSF, x=x, df=df)


def chi2_ppf(p: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Quantile function of the chi-squared distribution: the x with P(X <= x) = p.

    Args:
        p: Probability in [0, 1]; 0 gives 0, 1 gives +inf, values outside give NaN.
        df: Degrees of freedom, 0 < df < inf, not necessarily an integer.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape. df outside
        (0, inf) gives NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_chi2_array, _PPF, p=p, df=df)


def chi2_isf(p: ArrayLike, df: ArrayLike) -> float | NDArray[np.float64]:
    """Inverse survival function of the chi-squared distribution: P(X > x) = p.

    Accurate for p down to the smallest doubles.

    Args:
        p: Probability in [0, 1]; 0 gives +inf, 1 gives 0, values outside give NaN.
        df: Degrees of freedom, 0 < df < inf, not necessarily an integer.

    Returns:
        The quantile, as a float or a float64 array of the broadcast shape. df outside
        (0, inf) gives NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_chi2_array, _ISF, p=p, df=df)


def gamma_cdf(
    x: ArrayLike, shape: ArrayLike, scale: ArrayLike = 1.0
) -> float | NDArray[np.float64]:
    """Cumulative distribution function of the gamma distribution, P(X <= x).

    The density is x^(shape - 1) exp(-x / scale) / (Gamma(shape) scale^shape).

    Args:
        x: Quantile; x <= 0 gives 0.
        shape: Shape, 0 < shape < inf.
        scale: Scale, 0 < scale < inf.

    Returns:
        P(X <= x), as a float or a float64 array of the broadcast shape. Parameters
        outside their domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_gamma_array, _CDF, x=x, shape=shape, scale=scale)


def gamma_sf(
    x: ArrayLike, shape: ArrayLike, scale: ArrayLike = 1.0
) -> float | NDArray[np.float64]:
    """Survival function of the gamma distribution, P(X > x).

    Args:
        x: Quantile; x <= 0 gives 1.
        shape: Shape, 0 < shape < inf.
        scale: Scale, 0 < scale < inf.

    Returns:
        P(X > x), as a float or a float64 array of the broadcast shape. Parameters
        outside their domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_gamma_array, _SF, x=x, shape=shape, scale=scale)


def beta_cdf(x: ArrayLike, a: ArrayLike, b: ArrayLike) -> float | NDArray[np.float64]:
    """Cumulative distribution function of the beta distribution, P(X <= x).

    Args:
        x: Quantile; x <= 0 gives 0 and x >= 1 gives 1.
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.

    Returns:
        P(X <= x), as a float or a float64 array of the broadcast shape. Shapes outside
        their domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_beta_array, _CDF, x=x, a=a, b=b)


def beta_sf(x: ArrayLike, a: ArrayLike, b: ArrayLike) -> float | NDArray[np.float64]:
    """Survival function of the beta distribution, P(X > x).

    Args:
        x: Quantile; x <= 0 gives 1 and x >= 1 gives 0.
        a: First shape, 0 < a < inf.
        b: Second shape, 0 < b < inf.

    Returns:
        P(X > x), as a float or a float64 array of the broadcast shape. Shapes outside
        their domain give NaN.

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_beta_array, _SF, x=x, a=a, b=b)


def hypergeom_sf(
    k: ArrayLike, M: ArrayLike, n: ArrayLike, N: ArrayLike
) -> float | NDArray[np.float64]:
    """Survival function of the hypergeometric distribution, P(X > k).

    X counts the successes among N draws without replacement from a population of M
    items of which n are successes (scipy's convention).

    Args:
        k: Quantile; a non-integer k gives P(X > floor(k)).
        M: Population size, a non-negative integer below 2^53.
        n: Number of successes in the population, an integer with 0 <= n <= M.
        N: Number of draws, an integer with 0 <= N <= M.

    Returns:
        P(X > k), as a float or a float64 array of the broadcast shape. Parameters that
        are not integers in their ranges give NaN, and so does a tail that needs more
        than 1e7 terms (near the mode of populations above about 2e13).

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_hypergeom_array, _SF, k=k, M=M, n=n, N=N)


def hypergeom_logsf(
    k: ArrayLike, M: ArrayLike, n: ArrayLike, N: ArrayLike
) -> float | NDArray[np.float64]:
    """Logarithm of the survival function of the hypergeometric distribution, log P(X > k).

    Args:
        k: Quantile; a non-integer k gives log P(X > floor(k)).
        M: Population size, a non-negative integer below 2^53.
        n: Number of successes in the population, an integer with 0 <= n <= M.
        N: Number of draws, an integer with 0 <= N <= M.

    Returns:
        log P(X > k), as a float or a float64 array of the broadcast shape. Parameters
        that are not integers in their ranges give NaN, and so does a tail that needs
        more than 1e7 terms (near the mode of populations above about 2e13).

    Raises:
        TypeError: If an argument is not real-valued.
    """
    return _apply(_hypergeom_array, _LOGSF, k=k, M=M, n=n, N=N)
