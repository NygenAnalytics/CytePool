"""Probabilities and quantiles of the multivariate t distribution, each with its error.

A test that compares several correlated t-statistics at once, such as Dunnett's test of
each dose against the vehicle, refers the largest of them to the multivariate t
distribution. :func:`multivariate_t_cdf` gives the probability that every coordinate
of T stays below its limit, or within plus or minus its limit, and the complement,
the chance that some coordinate leaves its interval, which is the adjusted p-value of
a max-t test. :func:`multivariate_t_quantile` gives the equicoordinate quantile, the
critical value of such a test. Each value comes with its error, and the computation
stops when the error of the smaller of the probability and its complement is at most a
requested fraction of it, so that a p-value of 1e-10 is as accurate, relative to its
size, as one of 0.04.
https://docs.nygen.io/CytePool/statistics/multiple-testing#multivariate-t-probabilities
explains the method and its accuracy.

T = Z / S, with Z normal with mean 0 and covariance the correlation matrix R, and df S^2
chi-squared on df degrees of freedom, independent of Z; df = inf gives the
multivariate normal distribution. df need not be an integer.

Methods:
    - The complement of a box as a sum of first exceedances. With the coordinates in
      the order of decreasing marginal tails, the event that some T_i exceeds its limit
      b_i is the disjoint union of the events that coordinate r is the first to do so:
      1 - P(box) = sum_r P(T_r > b_r) h_r, with h_r = P(T_j in its interval for every
      j < r | T_r > b_r). The marginal tails are exact (``t_sf_scalar``), h_1 = 1, and
      each h_r lies between 0 and 1, so an absolute error e in each h_r gives the
      complement a relative error of at most k e, at any magnitude. Estimating each
      term by conditioning on its rare event is importance sampling for a union of
      rare events, whose relative error stays bounded however small the union (Owen,
      Maximov and Chertkov 2019). For a box symmetric about 0, the exceedances below
      the lower limits mirror those above the upper limits, and the sum is doubled.
    - Each h_r by conditioning on the exceedance. S is drawn from a proposal tilted
      towards the tail, V / (df + tau^2) with V chi-squared on df degrees of freedom
      and tau = max(b_r, 0), with the weight (df / (df + tau^2))^(df / 2)
      exp(tau^2 S^2 / 2) Q(b_r S) / P(T_r > b_r) that makes the draw exact, Q being the
      normal upper tail; the weight is bounded, since exp(x^2 / 2) Q(x) is at most 1/2
      for x >= 0. Z_r is drawn above b_r S by inversion, and the earlier coordinates,
      normal given Z_r with mean R[j, r] Z_r and covariance the Schur complement of R,
      by Genz's (1992) separation of variables, with a Cholesky factor pivoted on the
      largest residual variance. A coordinate whose residual variance is at most 1e-10
      is a linear function of the earlier random ones; its interval becomes limits of
      the last random coordinate it depends on, so that singular correlations, such as
      those of all pairwise contrasts, are integrated as accurately as others.
      Interval probabilities and draws use the normal tail on the side away from zero,
      which keeps them accurate far out.
    - Rare overlaps. When the earlier coordinates of a part exceed their limits with a
      total chance below 1e-3, given T_r at its threshold, that chance comes from
      draws deep in the tail of Z_r and near the ends of the earlier coordinates'
      intervals, which few points reach. The part then draws each earlier coordinate
      at the point (1 - cos(pi x)) / 2 of its interval, x the lattice coordinate,
      which brings draws near both ends, and Z_r at the point x^2 of its conditional
      distribution, which moves draws towards its threshold, with weights
      pi sin(pi x) / 2 and 2 x of mean 1, and estimates 1 - h_r as the mean of the
      weights times the chance that an earlier coordinate leaves its interval, whose
      variance is that of the overlap, not that of the weights (own construction from
      these standard changes of variable and control variates).
    - The probability itself, when it is below 1/2. The same estimator conditions on
      the coordinate with the smallest marginal probability of its interval, and
      integrates the others given it. For a one-sided limit below 0, the condition is
      the exceedance of -T_r above -b_r, and S is tilted as above; otherwise a small
      box's probability grows with S about as S^d, d the derivative of the box's
      log-probability in log S at S = 1 with its coordinates taken as independent, and
      the tilt t = -df d / (df + d) gives S^2 the mean (df + d) / df that it has given
      such a box. So the requested relative error applies to the smaller of the
      probability and its complement, and small boxes cost about what large ones do.
      A one-sided box whose limits lie far below 0 is the exception: its probability
      falls exponentially, and at 1% of it, one such box of 5 correlated contrasts
      took 24 s. A one-sided max-t p-value needs only the complement's accuracy, and
      with ``relative_to="complement"`` such a box is bounded instead: its
      probability is at most the smallest probability of its coordinates'
      intervals, exact from the t distribution, and at most that of its least
      correlated pair; when that bound is at most ``relative_error``, the box is not
      integrated: its probability is 0 and its complement 1, with the bound as their
      error, which bounds the difference from the true values. So a max-t p-value
      near 1 is 1, never below that of a more extreme contrast of the same test,
      whose box is larger. Every other box, two-sided ones included, is computed as
      without it, to the smaller side's target, which meets the complement's too.
    - Points and errors. Randomly shifted Kronecker points, x_n = frac(n alpha + Delta)
      with alpha_j the fractional part of the square root of the j-th prime, folded by
      the baker's transform 1 - |2 x - 1|; 16 random shifts are drawn once per call,
      so the problems of one call share them and a seed gives the same values whatever
      the order of the problems and the number of threads. The estimate is the mean of
      the 16 shift estimates and its standard error their standard deviation over 4
      (randomized quasi-Monte Carlo with independent shifts; Genz and Bretz 2002).
      Points per shift double from 4 until the standard error of the smaller of the
      probability and the complement is at most ``relative_error`` of it, or until
      2^16 points per shift (2^20 in all); the error then shows that the limit was
      reached. A problem with rare overlaps, or with a coordinate correlated beyond
      0.999 with the one it conditions on, takes at least 16 points per shift: both
      make features that 4 points per shift miss together, in a shift as in the
      others, so that their spread understates the error. Problems that share df share
      the chi-squared quantiles of the points.
    - Two coordinates without random numbers. A bivariate t vector is Z / S with Z = L
      u for a Cholesky factor L of R and u standard normal in the plane. In polar
      coordinates the angle of u is uniform and independent of its length, and
      (|u| / S)^2 / 2 follows the F distribution on 2 and df degrees of freedom, whose
      upper tail at r^2 / 2 is G(r) = (1 + r^2 / df)^(-df / 2) (exp(-r^2 / 2) for the
      normal). So the probability of a box is the average over the direction theta of
      G(rho_in) - G(rho_out), where rho_in and rho_out are the distances from the
      origin at which the ray in direction theta enters and leaves the box (own
      algebra from these two facts). The average is computed by adaptive Gauss-Kronrod
      quadrature (7 and 15 points; Piessens et al. 1983) between the directions where
      the face of the box that the ray meets changes, to a relative error of 1e-12 of
      the smaller of the probability and its complement, with the difference of the
      two rules as the error. One coordinate uses the t distribution directly.
    - Quantiles. The equicoordinate quantile solves for the limit at which the
      probability, or the complement for probabilities of at least 1/2, reaches its
      target. It lies between the univariate quantile at the probability and the one
      that Bonferroni's inequality gives; the root is found by bisection between these
      bounds with the shifts and points held fixed, which makes the estimate a smooth
      function of the limit, until the estimate changes across the bracket by less
      than 1% of its standard error; while the standard error exceeds its target,
      points double and the bisection starts again from the bounds. The quantile's
      error is the standard error of the probability over the slope of the estimate
      across the final bracket.

References:
    Genz A (1992). Numerical computation of multivariate normal probabilities. Journal
    of Computational and Graphical Statistics 1, 141-149.
    doi:10.1080/10618600.1992.10477010

    Genz A, Bretz F (2002). Comparison of methods for the computation of multivariate t
    probabilities. Journal of Computational and Graphical Statistics 11, 950-971.
    doi:10.1198/106186002394

    Owen AB, Maximov Y, Chertkov M (2019). Importance sampling the union of rare events
    with an application to power systems analysis. Electronic Journal of Statistics
    13, 231-254. doi:10.1214/18-EJS1527

    Piessens R, de Doncker-Kapenga E, Überhuber CW, Kahaner DK (1983). Quadpack.
    Springer, Berlin. doi:10.1007/978-3-642-61786-7
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import prange
from numpy.typing import ArrayLike, NDArray

from cytepool._jit import by_chunks, jit, jit_numpy_errors
from cytepool._masked_arrays import read_only, real_array, unmasked_array
from cytepool._options import (
    checked_choice,
    checked_flag,
    random_generator,
    real_number,
)
from cytepool.stats.distributions import (
    chi2_ppf_scalar,
    norm_cdf_scalar,
    norm_isf_scalar,
    norm_ppf_scalar,
    t_isf_scalar,
    t_sf_scalar,
)
from cytepool.stats.special import betainc_scalar

__all__ = [
    "MultivariateTProbability",
    "MultivariateTQuantile",
    "multivariate_t_cdf",
    "multivariate_t_quantile",
]

FloatArray = NDArray[np.float64]

_INF = math.inf
_NAN = math.nan
_SQRT_HALF = 0.7071067811865476
_EPS = 2.220446049250313e-16
_SHIFTS = 16
# Points per shift in the first round, and the most per shift; the limit is an argument
# of the compiled routines.
_FIRST_POINTS = 4
# The fewest points per shift before a problem may stop when it has rare overlaps or a
# coordinate nearly collinear with the one it conditions on: both make features of
# the integrand that a few points per shift miss together, so that the shifts'
# estimates agree while the error they suggest is too small.
_GUARDED_POINTS = 16
# A coordinate correlated with the conditioning one beyond this is nearly collinear.
_COLLINEAR = 0.999
_MOST_POINTS = 2**16
# A residual variance at or below this, relative to the unit variances of R, makes a
# coordinate a linear function of the earlier ones.
_PIVOT_TOLERANCE = 1e-10
# Tolerances of the checks of a correlation matrix: symmetry and unit diagonal
# (absolute), and the smallest eigenvalue relative to the largest.
_SYMMETRY_TOLERANCE = 1e-10
_EIGENVALUE_TOLERANCE = 1e-10
# The relative error of the two-dimensional quadrature, its most subintervals, and the
# rounding its error bound allows for, relative to the value.
_QUADRATURE_TOLERANCE = 1e-12
_MOST_INTERVALS = 4000
_ROUNDING = 1e-13
# Kinds of the event a part of the estimator conditions on.
_ABOVE, _WITHIN = 0, 1
# A part of the first-exceedance sum whose earlier coordinates exceed their limits
# with a total chance below this, given the conditioning coordinate at its threshold,
# has rare overlaps, and its draws are steered towards them.
_RARE_OVERLAP = 1e-3

# Gauss-Kronrod rule of 15 points with its embedded Gauss rule of 7, from QUADPACK
# (Piessens et al. 1983): the non-negative nodes of the Kronrod rule from the outside
# in, with the Kronrod weights, and the Gauss weights of the nodes 1, 3 and 5 and of
# the centre.
_KRONROD_NODES = np.array(
    [
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144845693013,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.0,
    ]
)
_KRONROD_WEIGHTS = np.array(
    [
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
    ]
)
_GAUSS_WEIGHTS = np.array(
    [
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
    ]
)


@dataclass(frozen=True, eq=False)
class MultivariateTProbability:
    """The probability of a box under a multivariate t distribution, with its error.

    Attributes:
        probability: P(T_i <= upper_i for every i), or P(|T_i| <= upper_i for every i)
            for a two-sided box.
        complement: 1 - probability, computed so that it keeps its relative accuracy
            when it is small: the chance that some T_i leaves its interval, the
            adjusted p-value of a max-t test.
        error: The estimated error of both: the standard error of the Monte Carlo
            estimate with three or more coordinates, the error bound of the quadrature
            with two, and 0 with one; with ``relative_to="complement"``, a one-sided
            box that a bound settles has probability 0 and that bound on its
            probability as its error, which bounds the difference from the true
            value.

    Each is a float for one problem, and a read-only array of the problems' shape
    otherwise.
    """

    probability: float | FloatArray
    complement: float | FloatArray
    error: float | FloatArray


@dataclass(frozen=True, eq=False)
class MultivariateTQuantile:
    """An equicoordinate quantile of a multivariate t distribution, with its error.

    Attributes:
        quantile: The q with P(T_i <= q for every i) = probability, or P(|T_i| <= q
            for every i) = probability for a two-sided quantile.
        error: The standard error of q.

    Each is a float for one problem, and a read-only array of the problems' shape
    otherwise.
    """

    quantile: float | FloatArray
    error: float | FloatArray


# ---------------------------------------------------------------------------------------
# Normal intervals and the t distribution of one coordinate
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _normal_interval(a: float, b: float, x: float, draw: bool) -> tuple[float, float]:
    """Return P(a < Z < b) and, when ``draw``, the z in (a, b) with P(a < Z < z) equal
    to x times it, for a standard normal Z; a <= b, either may be infinite.

    Both use the tail on the side away from zero, which keeps them accurate far out;
    an infinite end needs no tail of its own, as one-sided limits give.
    """
    if a == -_INF:
        if b <= 0.0:
            mass = norm_cdf_scalar(b)
            return mass, norm_ppf_scalar(x * mass) if draw else 0.0
        upper_b = norm_cdf_scalar(-b)
        mass = 1.0 - upper_b
        if not draw:
            return mass, 0.0
        below = x * mass
        if below <= 0.5:
            return mass, norm_ppf_scalar(below)
        return mass, norm_isf_scalar(upper_b + (mass - below))
    if b == _INF:
        if a >= 0.0:
            mass = norm_cdf_scalar(-a)
            return mass, norm_isf_scalar((1.0 - x) * mass) if draw else 0.0
        # Only the conditioning draws have an infinite upper end, and they draw.
        lower_a = norm_cdf_scalar(a)
        mass = 1.0 - lower_a
        below = x * mass
        if lower_a + below <= 0.5:
            return mass, norm_ppf_scalar(lower_a + below)
        return mass, norm_isf_scalar(mass - below)
    if a >= 0.0:
        upper_a = norm_cdf_scalar(-a)
        upper_b = norm_cdf_scalar(-b)
        mass = max(upper_a - upper_b, 0.0)
        if not draw:
            return mass, 0.0
        return mass, norm_isf_scalar(upper_a - x * mass)
    if b <= 0.0:
        lower_a = norm_cdf_scalar(a)
        lower_b = norm_cdf_scalar(b)
        mass = max(lower_b - lower_a, 0.0)
        if not draw:
            return mass, 0.0
        return mass, norm_ppf_scalar(lower_a + x * mass)
    lower_a = norm_cdf_scalar(a)
    upper_b = norm_cdf_scalar(-b)
    mass = max(1.0 - lower_a - upper_b, 0.0)
    if not draw:
        return mass, 0.0
    below = x * mass
    if lower_a + below <= 0.5:
        return mass, norm_ppf_scalar(lower_a + below)
    return mass, norm_isf_scalar(upper_b + (mass - below))


@jit_numpy_errors
def _t_within(b: float, df: float) -> float:
    """Return P(|T| <= b) for T ~ t(df), b finite and >= 0, accurate when it is
    small."""
    if df == _INF:
        return math.erf(b * _SQRT_HALF)
    return betainc_scalar(0.5, 0.5 * df, b * b / (df + b * b))


@jit_numpy_errors
def _one_coordinate(b: float, df: float, two_sided: bool) -> tuple[float, float]:
    """Return the probability and complement of one coordinate's interval."""
    if two_sided:
        return _t_within(b, df), 2.0 * t_sf_scalar(b, df)
    return t_sf_scalar(-b, df), t_sf_scalar(b, df)


# ---------------------------------------------------------------------------------------
# Two coordinates: the average over directions
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _radial_tail(r: float, df: float) -> float:
    """G(r) = P(|u| / S > r) for u standard normal in the plane."""
    if r == _INF:
        return 0.0
    if df == _INF:
        return math.exp(-0.5 * r * r)
    return math.exp(-0.5 * df * math.log1p(r * r / df))


@jit_numpy_errors
def _radial_head(r: float, df: float) -> float:
    """1 - G(r), accurate when it is small; 1 at r = inf."""
    if df == _INF:
        return -math.expm1(-0.5 * r * r)
    return -math.expm1(-0.5 * df * math.log1p(r * r / df))


@jit_numpy_errors
def _radial_between(r_in: float, r_out: float, df: float) -> float:
    """G(r_in) - G(r_out) for r_in < r_out, accurate when it is small."""
    if r_out == _INF:
        return _radial_tail(r_in, df)
    gap = (r_out - r_in) * (r_out + r_in)
    if df == _INF:
        return _radial_tail(r_in, df) * -math.expm1(-0.5 * gap)
    return _radial_tail(r_in, df) * -math.expm1(
        -0.5 * df * math.log1p(gap / (df + r_in * r_in))
    )


@jit_numpy_errors
def _ray(
    theta: float, loadings: FloatArray, lower: FloatArray, upper: FloatArray, df: float
) -> tuple[float, float]:
    """The probability that T = L u / S lies in the box, given the direction theta of
    u, and its complement: G(rho_in) - G(rho_out) and 1 minus it."""
    c, s = math.cos(theta), math.sin(theta)
    r_in, r_out = 0.0, _INF
    for i in range(loadings.shape[0]):
        slope = loadings[i, 0] * c + loadings[i, 1] * s
        if slope > 0.0:
            r_out = min(r_out, upper[i] / slope)
            r_in = max(r_in, lower[i] / slope)
        elif slope < 0.0:
            r_out = min(r_out, lower[i] / slope)
            r_in = max(r_in, upper[i] / slope)
        elif not (lower[i] <= 0.0 <= upper[i]):
            return 0.0, 1.0
    if r_in >= r_out:
        return 0.0, 1.0
    inside = _radial_between(r_in, r_out, df)
    return inside, _radial_head(r_in, df) + _radial_tail(r_out, df)


@jit_numpy_errors
def _kronrod(
    a: float,
    b: float,
    loadings: FloatArray,
    lower: FloatArray,
    upper: FloatArray,
    df: float,
) -> tuple[float, float, float, float]:
    """The 15-point Kronrod estimates of the integrals of the probability and of the
    complement over directions a to b, and |Kronrod - Gauss| of each, from its own
    integrand, so that the bound of a tiny integral is not lost to the rounding of
    the other, which is close to the width of the interval."""
    centre, half = 0.5 * (a + b), 0.5 * (b - a)
    inside_c, outside_c = _ray(centre, loadings, lower, upper, df)
    kronrod_in = _KRONROD_WEIGHTS[7] * inside_c
    kronrod_out = _KRONROD_WEIGHTS[7] * outside_c
    gauss_in = _GAUSS_WEIGHTS[3] * inside_c
    gauss_out = _GAUSS_WEIGHTS[3] * outside_c
    for j in range(7):
        step = half * _KRONROD_NODES[j]
        in_1, out_1 = _ray(centre - step, loadings, lower, upper, df)
        in_2, out_2 = _ray(centre + step, loadings, lower, upper, df)
        kronrod_in += _KRONROD_WEIGHTS[j] * (in_1 + in_2)
        kronrod_out += _KRONROD_WEIGHTS[j] * (out_1 + out_2)
        if j % 2 == 1:
            gauss_in += _GAUSS_WEIGHTS[j // 2] * (in_1 + in_2)
            gauss_out += _GAUSS_WEIGHTS[j // 2] * (out_1 + out_2)
    return (
        kronrod_in * half,
        kronrod_out * half,
        abs(kronrod_in - gauss_in) * half,
        abs(kronrod_out - gauss_out) * half,
    )


@jit_numpy_errors
def _directions(
    loadings: FloatArray, lower: FloatArray, upper: FloatArray
) -> FloatArray:
    """The directions between which the face of the box that a ray meets stays the
    same, with those of the largest slopes, sorted, from 0 to 2 pi."""
    k = loadings.shape[0]
    found = [0.0, 2.0 * math.pi]
    for i in range(k):
        x, y = loadings[i, 0], loadings[i, 1]
        phi = math.atan2(y, x)
        for turn in (0.0, 0.5, 1.0, 1.5):
            found.append(phi + turn * math.pi)
        for j in range(i + 1, k):
            for limit_i in (lower[i], upper[i]):
                for limit_j in (lower[j], upper[j]):
                    if not (math.isfinite(limit_i) and math.isfinite(limit_j)):
                        continue
                    # Equal distances limit_i / slope_i = limit_j / slope_j.
                    wx = limit_i * loadings[j, 0] - limit_j * x
                    wy = limit_i * loadings[j, 1] - limit_j * y
                    if wx == 0.0 and wy == 0.0:
                        continue
                    psi = math.atan2(wy, wx)
                    found.append(psi + 0.5 * math.pi)
                    found.append(psi + 1.5 * math.pi)
    angles = np.empty(len(found))
    for n in range(len(found)):
        angles[n] = found[n] % (2.0 * math.pi)
    angles[1] = 2.0 * math.pi
    angles.sort()
    kept = [angles[0]]
    for n in range(1, angles.size):
        if angles[n] - kept[-1] > 1e-12:
            kept.append(angles[n])
    kept[-1] = 2.0 * math.pi
    return np.array(kept)


@jit_numpy_errors
def _two_coordinates(
    loadings: FloatArray, lower: FloatArray, upper: FloatArray, df: float
) -> tuple[float, float, float]:
    """Return the probability of the box, its complement and the error bound, by
    adaptive Gauss-Kronrod quadrature of the average over directions."""
    edges = _directions(loadings, lower, upper)
    n = edges.size - 1
    start = np.empty(_MOST_INTERVALS)
    stop = np.empty(_MOST_INTERVALS)
    value_in = np.empty(_MOST_INTERVALS)
    value_out = np.empty(_MOST_INTERVALS)
    bound_in = np.empty(_MOST_INTERVALS)
    bound_out = np.empty(_MOST_INTERVALS)
    for i in range(n):
        start[i], stop[i] = edges[i], edges[i + 1]
        value_in[i], value_out[i], bound_in[i], bound_out[i] = _kronrod(
            start[i], stop[i], loadings, lower, upper, df
        )
    while True:
        total_in = value_in[:n].sum()
        total_out = value_out[:n].sum()
        # The smaller of the two integrals decides the error, and the interval to
        # split next.
        bound = bound_in if total_in <= total_out else bound_out
        error = bound[:n].sum()
        if error <= _QUADRATURE_TOLERANCE * min(total_in, total_out) or n + 1 > (
            _MOST_INTERVALS
        ):
            break
        worst = int(np.argmax(bound[:n]))
        middle = 0.5 * (start[worst] + stop[worst])
        if not start[worst] < middle < stop[worst]:
            break  # the interval cannot be split further
        start[n], stop[n] = middle, stop[worst]
        stop[worst] = middle
        value_in[worst], value_out[worst], bound_in[worst], bound_out[worst] = _kronrod(
            start[worst], stop[worst], loadings, lower, upper, df
        )
        value_in[n], value_out[n], bound_in[n], bound_out[n] = _kronrod(
            start[n], stop[n], loadings, lower, upper, df
        )
        n += 1
    scale = 0.5 / math.pi
    probability = min(max(total_in * scale, 0.0), 1.0)
    complement = min(max(total_out * scale, 0.0), 1.0)
    # The rules' difference bounds the error of the rule; rounding of the sums over
    # thousands of nodes adds up to about 1e-14 of the value.
    error = error * scale + _ROUNDING * min(probability, complement)
    return probability, complement, error


@jit_numpy_errors
def _plane_box(
    correlation: float, upper: FloatArray, df: float, two_sided: bool
) -> tuple[float, float, float]:
    """The probability, complement and error of a box of two coordinates with the
    given correlation."""
    loadings = np.zeros((2, 2))
    loadings[0, 0] = 1.0
    loadings[1, 0] = correlation
    loadings[1, 1] = math.sqrt(max(1.0 - correlation * correlation, 0.0))
    lower = -upper if two_sided else np.full(2, -_INF)
    return _two_coordinates(loadings, lower, upper, df)


# ---------------------------------------------------------------------------------------
# Three or more coordinates: conditioning on an exceedance
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _pivoted_cholesky(
    covariance: FloatArray,
) -> tuple[FloatArray, np.ndarray, int]:
    """Cholesky factor of a positive semidefinite matrix, pivoted on the largest
    residual variance: L, the order of the rows (L L' is the matrix in that order) and
    the number of random coordinates, those whose residual variance exceeds the
    tolerance; the rest follow them, with zero diagonal."""
    m = covariance.shape[0]
    work = covariance.copy()
    order = np.arange(m)
    factor = np.zeros((m, m))
    rank = m
    for p in range(m):
        best = p
        for q in range(p + 1, m):
            if work[q, q] > work[best, best]:
                best = q
        if best != p:
            for column in range(m):
                work[p, column], work[best, column] = (
                    work[best, column],
                    work[p, column],
                )
            for row in range(m):
                work[row, p], work[row, best] = work[row, best], work[row, p]
            for column in range(p):
                factor[p, column], factor[best, column] = (
                    factor[best, column],
                    factor[p, column],
                )
            order[p], order[best] = order[best], order[p]
        if work[p, p] <= _PIVOT_TOLERANCE:
            rank = p
            break
        pivot = math.sqrt(work[p, p])
        factor[p, p] = pivot
        for q in range(p + 1, m):
            factor[q, p] = work[q, p] / pivot
        for q in range(p + 1, m):
            for column in range(p + 1, q + 1):
                work[q, column] -= factor[q, p] * factor[column, p]
                work[column, q] = work[q, column]
    return factor, order, rank


@jit_numpy_errors
def _part(
    correlation: FloatArray,
    condition: int,
    sign: float,
    earlier: np.ndarray,
    lower: FloatArray,
    upper: FloatArray,
    factors: FloatArray,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
    part: int,
) -> int:
    """Store one part of the estimator: the coordinates ``earlier``, integrated given
    sign * T_condition, in the order of their pivoted Cholesky factor. Returns the
    number of random coordinates among them."""
    m = earlier.size
    covariance = np.empty((m, m))
    for a in range(m):
        for b in range(m):
            covariance[a, b] = (
                correlation[earlier[a], earlier[b]]
                - correlation[earlier[a], condition]
                * correlation[earlier[b], condition]
            )
    factor, order, rank = _pivoted_cholesky(covariance)
    for a in range(m):
        j = earlier[order[a]]
        loads[part, a] = sign * correlation[j, condition]
        lows[part, a] = lower[j]
        highs[part, a] = upper[j]
        for b in range(a + 1):
            factors[part, a, b] = factor[a, b]
    return rank


@jit_numpy_errors
def _chi2_points(
    table: FloatArray,
    table_df: float,
    table_points: int,
    df: float,
    points: int,
    alpha: FloatArray,
    shifts: FloatArray,
) -> tuple[FloatArray, float, int]:
    """The chi-squared quantiles of the second coordinate of the first ``points``
    points of every shift, kept from earlier problems with the same df."""
    if df != table_df:
        table_points = 0
    if table.shape[1] < points:
        grown = np.empty((_SHIFTS, max(points, 2 * table.shape[1])))
        grown[:, :table_points] = table[:, :table_points]
        table = grown
    for s in range(_SHIFTS):
        for n in range(table_points, points):
            x = (n + 1) * alpha[1] + shifts[s, 1]
            x = 1.0 - abs(2.0 * (x - math.floor(x)) - 1.0)
            table[s, n] = chi2_ppf_scalar(min(max(x, _EPS), 1.0 - _EPS), df)
    return table, df, max(table_points, points)


@jit_numpy_errors
def _accumulate(
    df: float,
    parts: tuple[np.ndarray, ...],
    overlaps: bool,
    chi2: FloatArray,
    alpha: FloatArray,
    shifts: FloatArray,
    first: int,
    last: int,
    totals: FloatArray,
) -> None:
    """Add the points first to last - 1 of every shift to the shifts' totals.

    ``parts`` holds, for each part, its kind, threshold, mass and tilt (from
    :func:`_complement_conditions` or :func:`_probability_condition`), and its number
    of earlier coordinates and of random ones, factors, loads, lows and highs (from
    :func:`_complement_parts` or :func:`_probability_parts`). Each part conditions on
    T_r above its threshold (kind ``_ABOVE``) or within plus or minus it
    (``_WITHIN``), whose probability is its mass, and integrates its earlier
    coordinates given T_r; a part with none counts its mass exactly.

    S is drawn as sqrt(V / (df + t)) for the part's tilt t, with V chi-squared on df
    degrees of freedom, and weighted by (df / (df + t))^(df / 2) exp(t S^2 / 2)
    P(Z_r in its interval given S) / P(T_r in its interval), which makes the draw
    exact.

    In a part of the first-exceedance sum (``overlaps``) whose overlaps are rare (see
    :func:`_rare_overlaps`), the chance that an earlier coordinate exceeds its limit
    as well comes from draws deep in the tail of Z_r and near the ends of the earlier
    coordinates' intervals, which few points reach, so that the shifts' estimates
    would agree while missing it. There each earlier coordinate is drawn at the point
    (1 - cos(pi x)) / 2 of its interval, x the lattice coordinate, with the weight
    pi sin(pi x) / 2, which brings draws near both ends, and Z_r at the point x^2 of
    its conditional distribution, with the weight 2 x, which moves draws towards its
    threshold. The weights have mean 1, so the part estimates 1 - h_r as the mean of
    the weights times the chance that an earlier coordinate leaves its interval,
    whose variance is that of the overlap, not that of the weights. Elsewhere the
    draws are plain, as these changes add variance where overlaps are common.

    A coordinate of zero residual variance is a linear function of the random ones;
    its interval limits the last random coordinate it depends on, so that the
    overlap is integrated, not sampled.
    """
    kinds, thresholds, masses, tilts, sizes, ranks, factors, loads, lows, highs = parts
    n_parts = kinds.size
    width = max(sizes.max(), 1)
    y = np.empty(width)
    rare = np.zeros(n_parts, dtype=np.bool_)
    if overlaps:
        for part in range(n_parts):
            rare[part] = _rare_overlaps(
                df, thresholds[part], sizes[part], loads[part], lows[part], highs[part]
            )
    # The last random coordinate that each coordinate of zero residual variance
    # depends on, or -1 for one that depends on none.
    targets = np.full((n_parts, width), -1, dtype=np.int64)
    for part in range(n_parts):
        for p in range(ranks[part], sizes[part]):
            for q in range(ranks[part] - 1, -1, -1):
                if abs(factors[part, p, q]) > _PIVOT_TOLERANCE:
                    targets[part, p] = q
                    break
    for s in range(_SHIFTS):
        total = 0.0
        for n in range(first, last):
            x = (n + 1) * alpha[0] + shifts[s, 0]
            x0 = 1.0 - abs(2.0 * (x - math.floor(x)) - 1.0)
            x0 = min(max(x0, _EPS), 1.0 - _EPS)
            v = chi2[s, n] if df != _INF else 1.0
            previous = _NAN
            previous_kind = -1
            previous_tilt = _NAN
            previous_steer = False
            weight = 0.0
            z = 0.0
            scale = 1.0
            usable = True
            point = 0.0
            for part in range(n_parts):
                mass = masses[part]
                m = sizes[part]
                if m == 0:
                    point += mass
                    continue
                if mass == 0.0:
                    continue
                c = thresholds[part]
                tilt = tilts[part]
                steer = rare[part]
                if (
                    c != previous
                    or kinds[part] != previous_kind
                    or tilt != previous_tilt
                    or steer != previous_steer
                ):
                    previous, previous_kind = c, kinds[part]
                    previous_tilt, previous_steer = tilt, steer
                    if df == _INF:
                        scale = 1.0
                    else:
                        scale = math.sqrt(v / (df + tilt))
                    if kinds[part] == _ABOVE:
                        tail = x0 * x0 if steer else x0
                        jacobian = 2.0 * x0 if steer else 1.0
                        within, z = _normal_interval(c * scale, _INF, tail, True)
                    else:
                        jacobian = 1.0
                        within, z = _normal_interval(-c * scale, c * scale, x0, True)
                    usable = within > 0.0 and math.isfinite(z)
                    if df == _INF:
                        weight = jacobian
                    elif usable:
                        weight = jacobian * math.exp(
                            0.5 * tilt * scale * scale
                            - 0.5 * df * math.log1p(tilt / df)
                            + math.log(within)
                            - math.log(mass)
                        )
                    else:
                        weight = 0.0
                if not usable:
                    # A point whose Z_r underflows carries no measure.
                    if steer:
                        point += mass
                    continue
                product = 1.0
                draw_weight = 1.0
                rank = ranks[part]
                # Coordinates that are linear functions of the conditioning one alone
                # count as indicators.
                for p in range(rank, m):
                    if targets[part, p] < 0:
                        shift = loads[part, p] * z
                        low = lows[part, p] * scale - shift
                        high = highs[part, p] * scale - shift
                        if not (low <= 0.0 <= high):
                            product = 0.0
                for p in range(rank):
                    if product == 0.0:
                        break
                    shift = loads[part, p] * z
                    for q in range(p):
                        shift += factors[part, p, q] * y[q]
                    pivot = factors[part, p, p]
                    low = (lows[part, p] * scale - shift) / pivot
                    high = (highs[part, p] * scale - shift) / pivot
                    # The constraints of the coordinates that depend on y[p] last, as
                    # limits of y[p].
                    for d in range(rank, m):
                        if targets[part, d] != p:
                            continue
                        rest = loads[part, d] * z
                        for q in range(p):
                            rest += factors[part, d, q] * y[q]
                        slope = factors[part, d, p]
                        from_low = (lows[part, d] * scale - rest) / slope
                        from_high = (highs[part, d] * scale - rest) / slope
                        if slope < 0.0:
                            from_low, from_high = from_high, from_low
                        low = max(low, from_low)
                        high = min(high, from_high)
                    if not low < high:
                        product = 0.0
                        break
                    xp = 0.5
                    draw = p < rank - 1
                    if draw:
                        x = (n + 1) * alpha[2 + p] + shifts[s, 2 + p]
                        xp = 1.0 - abs(2.0 * (x - math.floor(x)) - 1.0)
                        if steer:
                            ends = math.sin(math.pi * xp)
                            xp = 0.5 - 0.5 * math.cos(math.pi * xp)
                            draw_weight *= 0.5 * math.pi * ends
                        xp = min(max(xp, _EPS), 1.0 - _EPS)
                    within_p, y[p] = _normal_interval(low, high, xp, draw)
                    product *= within_p
                    if product < 1e-300 or (draw and not math.isfinite(y[p])):
                        product = 0.0
                if steer:
                    point += mass * (1.0 - weight * draw_weight * (1.0 - product))
                else:
                    point += mass * weight * product
            total += point
        totals[s] += total


@jit_numpy_errors
def _rare_overlaps(
    df: float,
    threshold: float,
    size: int,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
) -> bool:
    """Whether the earlier coordinates of a part exceed their limits with a total
    chance below ``_RARE_OVERLAP``, given T_r at its threshold: the sum over them of
    P(T_j outside its interval | Z_r = threshold S), at the scale S of the tilted
    proposal (1 for the normal), from the normal distribution of Z_j given Z_r, with
    mean rho_j Z_r and variance 1 - rho_j^2."""
    if size == 0 or threshold <= 0.0:
        return False
    scale = 1.0 if df == _INF else math.sqrt(df / (df + threshold * threshold))
    z = threshold * scale
    total = 0.0
    for p in range(size):
        rho = loads[p]
        spread = math.sqrt(max(1.0 - rho * rho, 0.0))
        low = lows[p] * scale - rho * z
        high = highs[p] * scale - rho * z
        if spread == 0.0:
            # A coordinate that moves with T_r is in or out by the position of Z_r
            # alone, which plain draws integrate: never a rare overlap.
            return False
        else:
            total += norm_cdf_scalar(-high / spread) + norm_cdf_scalar(low / spread)
    return total < _RARE_OVERLAP


@jit_numpy_errors
def _fewest_points(
    df: float,
    thresholds: FloatArray,
    sizes: np.ndarray,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
    overlaps: bool,
) -> int:
    """The fewest points per shift before a problem may stop: ``_GUARDED_POINTS`` when
    a part of the first-exceedance sum has rare overlaps, or when a coordinate is
    nearly collinear with the one its part conditions on, and ``_FIRST_POINTS``
    otherwise."""
    for part in range(sizes.size):
        for p in range(sizes[part]):
            if abs(loads[part, p]) > _COLLINEAR:
                return _GUARDED_POINTS
        if overlaps and _rare_overlaps(
            df, thresholds[part], sizes[part], loads[part], lows[part], highs[part]
        ):
            return _GUARDED_POINTS
    return _FIRST_POINTS


@jit_numpy_errors
def _shift_statistics(
    totals: FloatArray, points: int, factor: float
) -> tuple[float, float]:
    """The mean of the shift estimates and its standard error."""
    estimates = totals * (factor / points)
    mean = estimates.mean()
    spread = 0.0
    for s in range(_SHIFTS):
        spread += (estimates[s] - mean) ** 2
    return mean, math.sqrt(spread / (_SHIFTS - 1) / _SHIFTS)


@jit_numpy_errors
def _complement_parts(
    correlation: FloatArray,
    order: np.ndarray,
    lower: FloatArray,
    upper: FloatArray,
    factors: FloatArray,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    """The parts of the first-exceedance sum, with coordinates in ``order``: part r
    conditions on T_order[r] above its limit and integrates the coordinates before
    it. Returns the number of earlier coordinates of each part and of random ones."""
    k = order.size
    sizes = np.arange(k)
    ranks = np.zeros(k, dtype=np.int64)
    for r in range(1, k):
        ranks[r] = _part(
            correlation,
            order[r],
            1.0,
            order[:r],
            lower,
            upper,
            factors,
            loads,
            lows,
            highs,
            r,
        )
    return sizes, ranks


@jit_numpy_errors
def _complement_conditions(
    order: np.ndarray, upper: FloatArray, df: float
) -> tuple[np.ndarray, FloatArray, FloatArray, FloatArray]:
    """The kinds, thresholds, masses P(T_r > b_r) and tilts max(b_r, 0)^2 of the
    first-exceedance parts."""
    k = order.size
    kinds = np.full(k, _ABOVE)
    thresholds = np.empty(k)
    masses = np.empty(k)
    tilts = np.empty(k)
    for r in range(k):
        thresholds[r] = upper[order[r]]
        masses[r] = t_sf_scalar(thresholds[r], df)
        tilts[r] = max(thresholds[r], 0.0) ** 2
    return kinds, thresholds, masses, tilts


@jit_numpy_errors
def _box_slope(upper: FloatArray, two_sided: bool) -> float:
    """The derivative of log P(Z_i in its interval times s for every i) in log s at
    s = 1, summed over the coordinates as if they were independent: about the power
    of s by which a small box's probability grows with S. The limits are finite."""
    slope = 0.0
    for i in range(upper.size):
        b = upper[i]
        if two_sided:
            if b == 0.0:
                slope += 1.0
            else:
                density = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)
                slope += 2.0 * b * density / math.erf(b * _SQRT_HALF)
        else:
            density = math.exp(-0.5 * b * b) / math.sqrt(2.0 * math.pi)
            slope += b * density / norm_cdf_scalar(b)
    return slope


@jit_numpy_errors
def _probability_condition(
    condition: int, upper: FloatArray, df: float, two_sided: bool
) -> tuple[np.ndarray, FloatArray, FloatArray, FloatArray]:
    """The kind, threshold, mass and tilt of the part that estimates the probability
    of the box directly: T_condition within plus or minus its limit, or -T_condition
    above minus its one-sided limit.

    A one-sided limit below 0 makes the condition a tail event, and S is tilted
    towards the tail as for an exceedance. Otherwise the box's probability grows with
    S about as S^d, d the box's slope in log S, and the tilt t = -df d / (df + d) gives
    S^2 the mean (df + d) / df that it has given such a box, which keeps the weights of
    small boxes from varying with S^d.
    """
    b = upper[condition]
    kinds = np.empty(1, dtype=np.int64)
    thresholds = np.empty(1)
    masses = np.empty(1)
    tilts = np.zeros(1)
    if two_sided:
        kinds[0], thresholds[0], masses[0] = _WITHIN, b, _t_within(b, df)
    else:
        kinds[0], thresholds[0], masses[0] = _ABOVE, -b, t_sf_scalar(-b, df)
    if df == _INF:
        return kinds, thresholds, masses, tilts
    if not two_sided and b < 0.0:
        tilts[0] = b * b
    else:
        slope = min(max(_box_slope(upper, two_sided), 0.0), float(upper.size))
        tilts[0] = -df * slope / (df + slope)
    return kinds, thresholds, masses, tilts


@jit_numpy_errors
def _probability_parts(
    correlation: FloatArray,
    condition: int,
    lower: FloatArray,
    upper: FloatArray,
    two_sided: bool,
    factors: FloatArray,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    """The one part that estimates the probability of the box directly, integrating
    every other coordinate given T_condition (given -T_condition for a one-sided
    limit). Returns its number of coordinates and of random ones."""
    k = correlation.shape[0]
    earlier = np.empty(k - 1, dtype=np.int64)
    count = 0
    for j in range(k):
        if j != condition:
            earlier[count] = j
            count += 1
    sizes = np.full(1, k - 1)
    ranks = np.empty(1, dtype=np.int64)
    ranks[0] = _part(
        correlation,
        condition,
        1.0 if two_sided else -1.0,
        earlier,
        lower,
        upper,
        factors,
        loads,
        lows,
        highs,
        0,
    )
    return sizes, ranks


@jit_numpy_errors
def _sampled_box(
    correlation: FloatArray,
    upper: FloatArray,
    df: float,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    table: FloatArray,
    table_df: float,
    table_points: int,
    on_complement: bool,
) -> tuple[float, float, float, FloatArray, float, int]:
    """The probability, complement and standard error of a box of three or more
    coordinates, with the chi-squared table for the next problem; with
    ``on_complement``, a one-sided box whose probability is bounded within the
    complement's target is not integrated: it gets probability 0, complement 1 and
    the bound as their error."""
    k = correlation.shape[0]
    lower = -upper if two_sided else np.full(k, -_INF)
    factor = 2.0 if two_sided else 1.0
    tails = np.empty(k)
    for i in range(k):
        tails[i] = factor * t_sf_scalar(upper[i], df)
    if on_complement and not two_sided:
        # The box lies within each coordinate's interval and each pair's box, so
        # its probability is at most the smallest of their probabilities; when
        # that bound is within the target of a complement near 1, the probability
        # lies in [0, bound] whatever an integral gives. The complement is given as
        # 1, the end that a less extreme contrast's p-value can never fall below.
        bound = 1.0
        for i in range(k):
            bound = min(bound, t_sf_scalar(-upper[i], df))
        if relative_error < bound <= 0.5:
            # Of the pairs, the one of the smallest correlation, whose box is the
            # smallest when the limits are equal, as those of a max-t p-value are.
            # At a correlation of at least 0, a pair's box is at least the product
            # of its coordinates' interval probabilities (Slepian's inequality,
            # given S and then over S by Jensen's inequality), so at least the bound
            # squared: then it cannot settle the box when that square is beyond the
            # target.
            first, second = 0, 1
            for i in range(k):
                for j in range(i + 1, k):
                    if correlation[i, j] < correlation[first, second]:
                        first, second = i, j
            rho = correlation[first, second]
            if rho < 0.0 or bound * bound <= relative_error:
                pair = np.empty(2)
                pair[0] = upper[first]
                pair[1] = upper[second]
                inside, _, error = _plane_box(rho, pair, df, False)
                bound = min(bound, inside + error)
        if bound <= relative_error:
            return 0.0, 1.0, bound, table, table_df, table_points
    order = np.argsort(-tails, kind="mergesort")
    factors = np.zeros((k, k, k))
    loads = np.zeros((k, k))
    lows = np.zeros((k, k))
    highs = np.zeros((k, k))
    totals = np.zeros(_SHIFTS)
    points = 0
    side_is_probability = False
    if tails.sum() > 0.5:
        # The probability may be the smaller side: estimate it directly for one round,
        # conditioning on the coordinate with the smallest probability of its interval.
        sizes, ranks = _probability_parts(
            correlation,
            int(order[0]),
            lower,
            upper,
            two_sided,
            factors,
            loads,
            lows,
            highs,
        )
        kinds, thresholds, masses, tilts = _probability_condition(
            int(order[0]), upper, df, two_sided
        )
        if df != _INF:
            table, table_df, table_points = _chi2_points(
                table, table_df, table_points, df, _FIRST_POINTS, alpha, shifts
            )
        parts = (
            kinds,
            thresholds,
            masses,
            tilts,
            sizes,
            ranks,
            factors,
            loads,
            lows,
            highs,
        )
        _accumulate(df, parts, False, table, alpha, shifts, 0, _FIRST_POINTS, totals)
        points = _FIRST_POINTS
        estimate, _ = _shift_statistics(totals, points, 1.0)
        side_is_probability = estimate <= 0.5
        if not side_is_probability:
            totals[:] = 0.0
            points = 0
    if not side_is_probability:
        sizes, ranks = _complement_parts(
            correlation, order, lower, upper, factors, loads, lows, highs
        )
        kinds, thresholds, masses, tilts = _complement_conditions(order, upper, df)
    overlaps = not side_is_probability
    fewest = _fewest_points(df, thresholds, sizes, loads, lows, highs, overlaps)
    parts = (
        kinds,
        thresholds,
        masses,
        tilts,
        sizes,
        ranks,
        factors,
        loads,
        lows,
        highs,
    )
    side_factor = 1.0 if side_is_probability else factor
    while True:
        if points == 0:
            target = min(_FIRST_POINTS, most_points)
        else:
            target = min(2 * points, most_points)
        if target > points:
            if df != _INF:
                table, table_df, table_points = _chi2_points(
                    table, table_df, table_points, df, target, alpha, shifts
                )
            _accumulate(
                df, parts, overlaps, table, alpha, shifts, points, target, totals
            )
            points = target
        estimate, error = _shift_statistics(totals, points, side_factor)
        # The target is on the smaller of the probability and the complement, which
        # the side chosen after the first round is, unless both are near 1/2.
        smaller = min(estimate, 1.0 - estimate)
        if (
            error <= relative_error * smaller and points >= fewest
        ) or points >= most_points:
            break
    estimate = min(max(estimate, 0.0), 1.0)
    if side_is_probability:
        return estimate, 1.0 - estimate, error, table, table_df, table_points
    return 1.0 - estimate, estimate, error, table, table_df, table_points


@jit_numpy_errors
def _box(
    correlation: FloatArray,
    upper: FloatArray,
    df: float,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    table: FloatArray,
    table_df: float,
    table_points: int,
    on_complement: bool,
) -> tuple[float, float, float, FloatArray, float, int]:
    """One problem: the coordinates with an infinite limit are dropped, then one, two
    or more remain."""
    k = correlation.shape[0]
    if math.isnan(df):
        return _NAN, _NAN, _NAN, table, table_df, table_points
    for i in range(k):
        if math.isnan(upper[i]):
            return _NAN, _NAN, _NAN, table, table_df, table_points
        for j in range(k):
            if math.isnan(correlation[i, j]):
                return _NAN, _NAN, _NAN, table, table_df, table_points
        if upper[i] == -_INF:
            return 0.0, 1.0, 0.0, table, table_df, table_points
    kept = np.flatnonzero(upper < _INF)
    if kept.size == 0:
        return 1.0, 0.0, 0.0, table, table_df, table_points
    if kept.size == 1:
        probability, complement = _one_coordinate(upper[kept[0]], df, two_sided)
        return probability, complement, 0.0, table, table_df, table_points
    if kept.size == 2:
        probability, complement, error = _plane_box(
            correlation[kept[0], kept[1]], upper[kept], df, two_sided
        )
        return probability, complement, error, table, table_df, table_points
    if kept.size < k:
        correlation = np.ascontiguousarray(correlation[kept][:, kept])
        upper = upper[kept]
    return _sampled_box(
        correlation,
        upper,
        df,
        two_sided,
        relative_error,
        most_points,
        alpha,
        shifts,
        table,
        table_df,
        table_points,
        on_complement,
    )


@jit_numpy_errors
def _boxes(
    correlations: FloatArray,
    uppers: FloatArray,
    dfs: FloatArray,
    which_correlation: np.ndarray,
    which_upper: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    on_complement: bool,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    first: int,
    last: int,
    out: FloatArray,
) -> None:
    """Problems first to last - 1, into rows of ``out``: probability, complement,
    error."""
    table = np.empty((_SHIFTS, 0))
    table_df = _NAN
    table_points = 0
    for p in range(first, last):
        probability, complement, error, table, table_df, table_points = _box(
            correlations[which_correlation[p]],
            uppers[which_upper[p]],
            dfs[which_df[p]],
            two_sided,
            relative_error,
            most_points,
            alpha,
            shifts,
            table,
            table_df,
            table_points,
            on_complement,
        )
        out[p, 0] = probability
        out[p, 1] = complement
        out[p, 2] = error


@jit(parallel=True)
def _boxes_parallel(
    correlations: FloatArray,
    uppers: FloatArray,
    dfs: FloatArray,
    which_correlation: np.ndarray,
    which_upper: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    on_complement: bool,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    out: FloatArray,
    n_chunks: int,
) -> None:
    n = out.shape[0]
    for chunk in prange(n_chunks):
        _boxes(
            correlations,
            uppers,
            dfs,
            which_correlation,
            which_upper,
            which_df,
            two_sided,
            relative_error,
            on_complement,
            most_points,
            alpha,
            shifts,
            chunk * n // n_chunks,
            (chunk + 1) * n // n_chunks,
            out,
        )


@jit
def _boxes_serial(
    correlations: FloatArray,
    uppers: FloatArray,
    dfs: FloatArray,
    which_correlation: np.ndarray,
    which_upper: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    on_complement: bool,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    out: FloatArray,
) -> None:
    _boxes(
        correlations,
        uppers,
        dfs,
        which_correlation,
        which_upper,
        which_df,
        two_sided,
        relative_error,
        on_complement,
        most_points,
        alpha,
        shifts,
        0,
        out.shape[0],
        out,
    )


# ---------------------------------------------------------------------------------------
# Quantiles
# ---------------------------------------------------------------------------------------


@jit_numpy_errors
def _side_estimate(
    q: float,
    complement_side: bool,
    correlation: FloatArray,
    df: float,
    two_sided: bool,
    points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    table: FloatArray,
    order: np.ndarray,
    sizes: np.ndarray,
    ranks: np.ndarray,
    factors: FloatArray,
    loads: FloatArray,
    lows: FloatArray,
    highs: FloatArray,
) -> tuple[float, float]:
    """The estimate of the complement (or of the probability) of the equicoordinate
    box at q, and its error, with the parts' factors fixed and ``points`` per shift."""
    k = correlation.shape[0]
    upper = np.full(k, q)
    if k == 2:
        probability, complement, error = _plane_box(
            correlation[0, 1], upper, df, two_sided
        )
        return (complement if complement_side else probability), error
    scaled_lows = lows * q if two_sided else lows
    if complement_side:
        kinds, thresholds, masses, tilts = _complement_conditions(order, upper, df)
        factor = 2.0 if two_sided else 1.0
    else:
        kinds, thresholds, masses, tilts = _probability_condition(
            0, upper, df, two_sided
        )
        factor = 1.0
    parts = (
        kinds,
        thresholds,
        masses,
        tilts,
        sizes,
        ranks,
        factors,
        loads,
        scaled_lows,
        highs * q,
    )
    totals = np.zeros(_SHIFTS)
    _accumulate(df, parts, complement_side, table, alpha, shifts, 0, points, totals)
    return _shift_statistics(totals, points, factor)


@jit_numpy_errors
def _gap(
    q: float, target: float, complement_side: bool, problem: tuple[Any, ...]
) -> tuple[float, float]:
    """The estimate at q less the target on the complement side, or the target less
    the estimate on the probability side: a function that falls as q grows, whose
    root is the quantile; with the estimate's error. ``problem`` holds the other
    arguments of :func:`_side_estimate`."""
    estimate, error = _side_estimate(q, complement_side, *problem)
    if complement_side:
        return estimate - target, error
    return target - estimate, error


@jit_numpy_errors
def _quantile(
    probability: float,
    correlation: FloatArray,
    df: float,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    table: FloatArray,
    table_df: float,
    table_points: int,
) -> tuple[float, float, FloatArray, float, int]:
    """The equicoordinate quantile and its standard error, with the chi-squared table
    for the next problem."""
    k = correlation.shape[0]
    if math.isnan(probability) or math.isnan(df):
        return _NAN, _NAN, table, table_df, table_points
    for i in range(k):
        for j in range(k):
            if math.isnan(correlation[i, j]):
                return _NAN, _NAN, table, table_df, table_points
    # The quantile lies between the univariate one at the probability and the one
    # that Bonferroni's inequality gives.
    tail = 1.0 - probability
    if two_sided:
        lowest = t_isf_scalar(0.5 * tail, df)
        highest = t_isf_scalar(0.5 * tail / k, df)
    else:
        lowest = (
            -t_isf_scalar(probability, df)
            if probability < 0.5
            else t_isf_scalar(tail, df)
        )
        highest = t_isf_scalar(tail / k, df)
    if k == 1:
        return lowest, 0.0, table, table_df, table_points
    complement_side = probability >= 0.5
    target = tail if complement_side else probability
    # The parts depend on the limit only through their thresholds: with one limit for
    # every coordinate, the order and the factors stay the same.
    order = np.arange(k)
    factors = np.zeros((k, k, k))
    loads = np.zeros((k, k))
    lows = np.zeros((k, k))
    highs = np.zeros((k, k))
    unit_upper = np.ones(k)
    unit_lower = -unit_upper if two_sided else np.full(k, -_INF)
    sizes = np.zeros(1, dtype=np.int64)
    ranks = np.zeros(1, dtype=np.int64)
    if k > 2:
        if complement_side:
            sizes, ranks = _complement_parts(
                correlation, order, unit_lower, unit_upper, factors, loads, lows, highs
            )
        else:
            sizes, ranks = _probability_parts(
                correlation,
                0,
                unit_lower,
                unit_upper,
                two_sided,
                factors,
                loads,
                lows,
                highs,
            )
    points = _FIRST_POINTS
    while True:
        if k > 2 and df != _INF:
            table, table_df, table_points = _chi2_points(
                table, table_df, table_points, df, points, alpha, shifts
            )
        problem = (
            correlation,
            df,
            two_sided,
            points,
            alpha,
            shifts,
            table,
            order,
            sizes,
            ranks,
            factors,
            loads,
            lows,
            highs,
        )
        # Each round bisects between the bounds, which hold the root however the
        # estimate moves as the points double.
        low, high = lowest, highest
        f_low, _ = _gap(low, target, complement_side, problem)
        f_high, _ = _gap(high, target, complement_side, problem)
        error = 0.0
        for _ in range(200):
            middle = 0.5 * (low + high)
            f_middle, error = _gap(middle, target, complement_side, problem)
            if f_middle >= 0.0:
                low, f_low = middle, f_middle
            else:
                high, f_high = middle, f_middle
            if f_low - f_high <= 0.01 * error or high - low <= 4.0 * _EPS * max(
                abs(low), abs(high)
            ):
                break
        # The estimate is a smooth function of q with the points fixed: the final
        # bracket gives its root by interpolation, and its slope.
        drop = f_low - f_high
        slope = drop / (high - low) if drop > 0.0 else _NAN
        root = low + (high - low) * (f_low / drop if drop > 0.0 else 0.5)
        # The estimate's error at the bounds can carry the interpolation past them.
        root = min(max(root, lowest), highest)
        if k == 2 or points >= most_points:
            enough = True
        else:
            if complement_side:
                conditions = _complement_conditions(order, np.full(k, root), df)
            else:
                conditions = _probability_condition(0, np.full(k, root), df, two_sided)
            fewest = _fewest_points(
                df,
                conditions[1],
                sizes,
                loads,
                lows * root if two_sided else lows,
                highs * root,
                complement_side,
            )
            enough = error <= relative_error * target and points >= fewest
        if enough:
            spread = error / slope if slope > 0.0 else _NAN
            return root, spread, table, table_df, table_points
        points = min(2 * points, most_points)


@jit_numpy_errors
def _quantiles(
    probabilities: FloatArray,
    correlations: FloatArray,
    dfs: FloatArray,
    which_probability: np.ndarray,
    which_correlation: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    first: int,
    last: int,
    out: FloatArray,
) -> None:
    table = np.empty((_SHIFTS, 0))
    table_df = _NAN
    table_points = 0
    for p in range(first, last):
        q, error, table, table_df, table_points = _quantile(
            probabilities[which_probability[p]],
            correlations[which_correlation[p]],
            dfs[which_df[p]],
            two_sided,
            relative_error,
            most_points,
            alpha,
            shifts,
            table,
            table_df,
            table_points,
        )
        out[p, 0] = q
        out[p, 1] = error


@jit(parallel=True)
def _quantiles_parallel(
    probabilities: FloatArray,
    correlations: FloatArray,
    dfs: FloatArray,
    which_probability: np.ndarray,
    which_correlation: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    out: FloatArray,
    n_chunks: int,
) -> None:
    n = out.shape[0]
    for chunk in prange(n_chunks):
        _quantiles(
            probabilities,
            correlations,
            dfs,
            which_probability,
            which_correlation,
            which_df,
            two_sided,
            relative_error,
            most_points,
            alpha,
            shifts,
            chunk * n // n_chunks,
            (chunk + 1) * n // n_chunks,
            out,
        )


@jit
def _quantiles_serial(
    probabilities: FloatArray,
    correlations: FloatArray,
    dfs: FloatArray,
    which_probability: np.ndarray,
    which_correlation: np.ndarray,
    which_df: np.ndarray,
    two_sided: bool,
    relative_error: float,
    most_points: int,
    alpha: FloatArray,
    shifts: FloatArray,
    out: FloatArray,
) -> None:
    _quantiles(
        probabilities,
        correlations,
        dfs,
        which_probability,
        which_correlation,
        which_df,
        two_sided,
        relative_error,
        most_points,
        alpha,
        shifts,
        0,
        out.shape[0],
        out,
    )


# ---------------------------------------------------------------------------------------
# Arguments and the public functions
# ---------------------------------------------------------------------------------------


def _checked_correlation(values: Any) -> FloatArray:
    """The correlation matrices, (..., k, k), checked where they hold no NaN."""
    correlation = real_array(
        unmasked_array(values, "correlation", missing="NaN"),
        "correlation",
        objects=False,
    )
    if correlation.ndim < 2 or correlation.shape[-1] != correlation.shape[-2]:
        raise ValueError(
            "correlation must be a square matrix, or a stack of them with shape "
            f"(..., k, k); got shape {correlation.shape}"
        )
    k = correlation.shape[-1]
    if k == 0:
        raise ValueError("correlation must have at least one coordinate, got 0")
    if np.isinf(correlation).any():
        raise ValueError("correlation must be finite; NaN marks a missing matrix")
    stack = correlation.reshape(-1, k, k)
    complete = stack[~np.isnan(stack).any(axis=(1, 2))]
    if complete.size == 0:
        return correlation
    asymmetry = np.abs(complete - complete.transpose(0, 2, 1)).max()
    if asymmetry > _SYMMETRY_TOLERANCE:
        raise ValueError(
            f"correlation must be symmetric, to {_SYMMETRY_TOLERANCE:g}; entries "
            f"differ from their transposes by up to {asymmetry:.3g}"
        )
    diagonal = np.abs(np.diagonal(complete, axis1=1, axis2=2) - 1.0).max()
    if diagonal > _SYMMETRY_TOLERANCE:
        raise ValueError(
            "correlation must have a unit diagonal, to "
            f"{_SYMMETRY_TOLERANCE:g}: it is a correlation matrix, not a covariance "
            f"matrix; the diagonal differs from 1 by up to {diagonal:.3g}"
        )
    eigenvalues = np.linalg.eigvalsh(0.5 * (complete + complete.transpose(0, 2, 1)))
    smallest = eigenvalues[:, 0] / eigenvalues[:, -1]
    if smallest.min() < -_EIGENVALUE_TOLERANCE:
        worst = int(np.argmin(smallest))
        raise ValueError(
            "correlation must be positive semidefinite: its smallest eigenvalue must "
            f"be at least -{_EIGENVALUE_TOLERANCE:g} times its largest, but one "
            f"matrix has {eigenvalues[worst, 0]:.3g} against {eigenvalues[worst, -1]:.3g}"
        )
    return correlation


def _checked_df(values: Any) -> FloatArray:
    df = real_array(
        unmasked_array(values, "df", missing="NaN"), "df", boolean=False, objects=False
    )
    if (df <= 0.0).any():
        raise ValueError(
            f"df must be positive, or np.inf for the normal distribution; got "
            f"{df[df <= 0.0].flat[0]}"
        )
    return df


def _checked_relative_error(value: Any) -> float:
    relative_error = real_number(value, "relative_error", finite=True)
    if not 0.0 < relative_error < 1.0:
        raise ValueError(
            f"relative_error must be between 0 and 1, exclusive; got {relative_error}"
        )
    return relative_error


def _positions(shape: tuple[int, ...], batch: tuple[int, ...]) -> np.ndarray:
    """For each problem of the batch, the position of its argument among the
    arguments of ``shape``, broadcast."""
    count = math.prod(shape)
    return np.ascontiguousarray(
        np.broadcast_to(np.arange(count).reshape(shape), batch).reshape(-1)
    )


def _batch(*arguments: tuple[np.ndarray, int, str]) -> tuple[int, ...]:
    """The shape of the problems: the broadcast shape of the arguments' leading
    dimensions, those before the last ``trailing`` of each."""
    try:
        return np.broadcast_shapes(
            *(
                values.shape[: values.ndim - trailing]
                for values, trailing, _ in arguments
            )
        )
    except ValueError:
        listed = ", ".join(
            f"{name} of shape {values.shape}" for values, _, name in arguments
        )
        raise ValueError(
            "the arguments do not broadcast: the dimensions before the last one of "
            "upper and the last two of correlation must broadcast with those of the "
            f"other arguments; got {listed}"
        ) from None


def _lattice(
    k: int, rng: np.random.Generator | int | None
) -> tuple[FloatArray, FloatArray]:
    """The generators of the Kronecker points in k + 1 dimensions, and the shifts:
    none for one or two coordinates, which use no random numbers."""
    dimensions = k + 1
    primes: list[int] = []
    candidate = 2
    while len(primes) < dimensions:
        if all(candidate % p for p in primes if p * p <= candidate):
            primes.append(candidate)
        candidate += 1
    alpha = np.sqrt(np.array(primes, dtype=np.float64)) % 1.0
    generator = random_generator(rng)
    if k <= 2:
        return alpha, np.zeros((_SHIFTS, dimensions))
    return alpha, generator.random((_SHIFTS, dimensions))


def _shaped(values: FloatArray, batch: tuple[int, ...]) -> float | FloatArray:
    if not batch:
        return float(values[0])
    return read_only(np.ascontiguousarray(values.reshape(batch)))


def multivariate_t_cdf(
    upper: ArrayLike,
    correlation: ArrayLike,
    df: ArrayLike,
    *,
    two_sided: bool = False,
    relative_error: float = 0.01,
    relative_to: str = "smaller",
    rng: np.random.Generator | int | None = None,
) -> MultivariateTProbability:
    """Probability that every coordinate of a multivariate t vector stays within its
    limit, and its complement.

    T has standard t marginals on ``df`` degrees of freedom and correlation matrix
    ``correlation``: T = Z / S with Z normal, mean 0 and covariance ``correlation``,
    and df S^2 chi-squared on df degrees of freedom; ``df=np.inf`` gives the
    multivariate normal distribution, and df need not be an integer, as moderated
    tests give. The complement is computed directly, so that it keeps its relative
    accuracy when it is small: it is the adjusted p-value of a max-t test, such as
    P(max |T_i| >= c) for a two-sided box with every limit c. With three or more
    coordinates, the computation is randomized and stops when the standard error of
    the smaller of the probability and the complement is at most ``relative_error``
    times it; with two it is a deterministic quadrature, accurate to about 1e-12
    relative, and with one it is the t distribution itself.
    https://docs.nygen.io/CytePool/statistics/multiple-testing#multivariate-t-probabilities
    explains the method.

    The arguments broadcast over leading dimensions: the problems are those of the
    broadcast shape of ``upper[..., 0]``, ``correlation[..., 0, 0]`` and ``df``. NaN
    in a limit, a correlation or df gives NaN.

    Args:
        upper: ``(..., k)`` upper limits; +inf for none. With ``two_sided``, the box
            is |T_i| <= upper_i, and the limits must be at least 0.
        correlation: ``(..., k, k)`` correlation matrices: symmetric to 1e-10, with
            a unit diagonal to 1e-10, and positive semidefinite (smallest eigenvalue
            at least -1e-10 times the largest); singular matrices, such as those of
            all pairwise contrasts, are allowed.
        df: Degrees of freedom, positive; ``np.inf`` for the normal distribution.
        two_sided: Whether the box is two-sided, -upper_i <= T_i <= upper_i.
        relative_error: The largest standard error of the smaller of the probability
            and its complement, as a fraction of it, between 0 and 1; the default
            gives two significant digits. The computation stops at 2^20 points per
            problem whatever the error, which ``error`` then shows.
        relative_to: ``"smaller"`` (default) for the error of the smaller of the
            probability and its complement, or ``"complement"`` for the error of the
            complement alone, as a one-sided max-t p-value needs: a one-sided box of
            three or more coordinates whose probability is at most
            ``relative_error``, by the smallest probability of its coordinates'
            intervals or of its least correlated pair, is then not integrated, and
            gets probability 0 and complement 1, with that bound as their error.
            Such a box, whose limits lie far below 0, costs up to 2^20 points to
            reach its own relative error. Every other box is computed as with
            ``"smaller"``, whose target meets the complement's too.
        rng: A numpy random generator, a non-negative integer seed, or None. One
            set of 16 random shifts is drawn per call and serves every problem, so a
            seed gives the same values whatever the order of the problems and the
            number of threads. Problems of one or two coordinates use no random
            numbers.

    Returns:
        The probability, its complement and their error, as floats for one problem
        and as read-only arrays of the problems' shape otherwise.

    Raises:
        ValueError: If ``correlation`` is not square, symmetric, of unit diagonal or
            positive semidefinite, has no coordinates or an infinite entry, the
            shapes do not broadcast, a two-sided limit is negative, df is not
            positive, ``relative_error`` is not between 0 and 1, or ``relative_to``
            is neither ``"smaller"`` nor ``"complement"``.
        TypeError: If an argument has the wrong type or is a numpy masked array.

    Example:
        >>> import numpy as np
        >>> correlation = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
        >>> result = multivariate_t_cdf([2.5, 2.5, 2.5], correlation, df=10,
        ...                             two_sided=True, rng=1)
        >>> round(result.complement, 3)  # P(max |T_i| >= 2.5)
        0.076
    """
    correlation_values = _checked_correlation(correlation)
    k = correlation_values.shape[-1]
    upper_values = real_array(
        unmasked_array(upper, "upper", missing="NaN"), "upper", objects=False
    )
    if upper_values.ndim == 0 or upper_values.shape[-1] != k:
        raise ValueError(
            f"upper must have one limit per coordinate ({k}) in its last axis; got "
            f"shape {upper_values.shape}"
        )
    df_values = _checked_df(df)
    flag = checked_flag(two_sided, "two_sided")
    if flag and (upper_values < 0.0).any():
        raise ValueError(
            "a two-sided box needs limits of at least 0, as |T_i| <= upper_i; got "
            f"{upper_values[upper_values < 0.0].flat[0]}"
        )
    target = _checked_relative_error(relative_error)
    on_complement = (
        checked_choice(relative_to, "relative_to", ("smaller", "complement"))
        == "complement"
    )
    alpha, shifts = _lattice(k, rng)
    correlation_batch = correlation_values.shape[:-2]
    upper_batch = upper_values.shape[:-1]
    batch = _batch(
        (correlation_values, 2, "correlation"),
        (upper_values, 1, "upper"),
        (df_values, 0, "df"),
    )
    n = math.prod(batch)
    out = np.empty((n, 3))
    arguments = (
        np.ascontiguousarray(correlation_values.reshape(-1, k, k)),
        np.ascontiguousarray(upper_values.reshape(-1, k)),
        np.ascontiguousarray(df_values.reshape(-1)),
        _positions(correlation_batch, batch),
        _positions(upper_batch, batch),
        _positions(df_values.shape, batch),
        flag,
        target,
        on_complement,
        _MOST_POINTS,
        alpha,
        shifts,
        out,
    )
    by_chunks(_boxes_parallel, _boxes_serial, arguments, n)
    return MultivariateTProbability(
        probability=_shaped(out[:, 0], batch),
        complement=_shaped(out[:, 1], batch),
        error=_shaped(out[:, 2], batch),
    )


def multivariate_t_quantile(
    probability: ArrayLike,
    correlation: ArrayLike,
    df: ArrayLike,
    *,
    two_sided: bool = False,
    relative_error: float = 0.01,
    rng: np.random.Generator | int | None = None,
) -> MultivariateTQuantile:
    """Equicoordinate quantile of a multivariate t distribution: the critical value of
    a max-t test.

    The q with P(T_i <= q for every i) = probability, or, with ``two_sided``,
    P(|T_i| <= q for every i) = probability, as mvtnorm's ``qmvt`` with
    ``tail = "lower.tail"`` or ``"both.tails"``; T is as in
    :func:`multivariate_t_cdf`. A one-sided max-t test at level alpha rejects when
    the largest t-statistic exceeds the quantile at 1 - alpha, and a two-sided one
    when the largest absolute value does. The search stops when the standard error
    of the smaller of the probability and its complement at q is at most
    ``relative_error`` times it. https://docs.nygen.io/CytePool/statistics/multiple-testing#multivariate-t-probabilities
    explains the method.

    The arguments broadcast over leading dimensions, as in
    :func:`multivariate_t_cdf`. NaN in the probability, a correlation or df gives
    NaN.

    Args:
        probability: The probability of the box, between 0 and 1, exclusive.
        correlation: ``(..., k, k)`` correlation matrices, as in
            :func:`multivariate_t_cdf`.
        df: Degrees of freedom, positive; ``np.inf`` for the normal distribution.
        two_sided: Whether the box is two-sided, -q <= T_i <= q.
        relative_error: As in :func:`multivariate_t_cdf`.
        rng: As in :func:`multivariate_t_cdf`.

    Returns:
        The quantile and its standard error, which is 0 with one coordinate, as
        floats for one problem and as read-only arrays of the problems' shape
        otherwise.

    Raises:
        ValueError: If a probability is not between 0 and 1, ``correlation`` is not a
            valid correlation matrix, the shapes do not broadcast, df is not
            positive, or ``relative_error`` is not between 0 and 1.
        TypeError: If an argument has the wrong type or is a numpy masked array.

    Example:
        >>> import numpy as np
        >>> correlation = np.full((3, 3), 0.5) + 0.5 * np.eye(3)
        >>> critical = multivariate_t_quantile(0.95, correlation, df=10,
        ...                                    two_sided=True, rng=1)
        >>> round(critical.quantile, 2)
        2.76
    """
    probability_values = real_array(
        unmasked_array(probability, "probability", missing="NaN"),
        "probability",
        objects=False,
    )
    outside = (probability_values <= 0.0) | (probability_values >= 1.0)
    if outside.any():
        raise ValueError(
            "probability must be between 0 and 1, exclusive; got "
            f"{probability_values[outside].flat[0]}"
        )
    correlation_values = _checked_correlation(correlation)
    k = correlation_values.shape[-1]
    df_values = _checked_df(df)
    flag = checked_flag(two_sided, "two_sided")
    target = _checked_relative_error(relative_error)
    alpha, shifts = _lattice(k, rng)
    correlation_batch = correlation_values.shape[:-2]
    batch = _batch(
        (probability_values, 0, "probability"),
        (correlation_values, 2, "correlation"),
        (df_values, 0, "df"),
    )
    n = math.prod(batch)
    out = np.empty((n, 2))
    arguments = (
        np.ascontiguousarray(probability_values.reshape(-1)),
        np.ascontiguousarray(correlation_values.reshape(-1, k, k)),
        np.ascontiguousarray(df_values.reshape(-1)),
        _positions(probability_values.shape, batch),
        _positions(correlation_batch, batch),
        _positions(df_values.shape, batch),
        flag,
        target,
        _MOST_POINTS,
        alpha,
        shifts,
        out,
    )
    by_chunks(_quantiles_parallel, _quantiles_serial, arguments, n)
    return MultivariateTQuantile(
        quantile=_shaped(out[:, 0], batch), error=_shaped(out[:, 1], batch)
    )
