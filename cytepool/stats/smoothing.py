"""Scatterplot smoothing and natural cubic spline bases.

:func:`lowess` is Cleveland's (1979) robust locally weighted regression, and
matches R's ``stats::lowess``. The n points are sorted by x. The fit at x_i
uses the q = max(2, min(n, floor(span * n + 1e-7))) points nearest to x_i,
which lie within h_i, the distance from x_i to its q-th nearest neighbour
(counting x_i itself). Points at distance r get the tricube weight
(1 - (r / h_i)^3)^3, which is set to 1 for r <= 0.001 h_i and to 0 for
r > 0.999 h_i; when h_i is zero, every point tied with x_i gets weight 1. The
fitted value is that of a weighted least squares line, or of a weighted mean
when the weighted standard deviation of x in the neighbourhood is at most
0.001 times the range of x. A neighbourhood whose weights are all zero keeps
the observed value. Each robustness iteration refits with every weight
multiplied by Tukey's bisquare (1 - (e_k / s)^2)^2 of the residuals e_k of
the previous fit, with s six times the median absolute residual; the
bisquare is set to 1 for |e_k| <= 0.001 s and to 0 for |e_k| > 0.999 s. The
iterations stop early when s is below 1e-7 times the mean absolute residual.
To save time, points within ``delta`` of the last fitted point are not
fitted but interpolated linearly, and points tied with a fitted point share
its value.

:func:`choose_lowess_span` is limma's rule for the span of a LOWESS trend
through n points (limma's ``chooseLowessSpan``), with which
:func:`~cytepool.stats.voom.voom` chooses its span
(https://docs.nygen.io/CytePool/statistics/limma-voom#voom-span).

:func:`natural_spline_basis` returns a basis of the natural cubic splines
with given interior and boundary knots: cubic between the knots, twice
continuously differentiable at simple knots, and linear beyond the boundary
knots. Its column space matches that of R's ``splines::ns``. The basis is
built from the cubic B-splines on the knot sequence with each boundary knot
repeated four times (de Boor 1978), restricted to the subspace with zero
second derivative at both boundary knots.

References:
    Cleveland WS (1979). Robust locally weighted regression and smoothing scatterplots.
    Journal of the American Statistical Association 74, 829-836.
    doi:10.1080/01621459.1979.10481038

    Cleveland WS (1981). LOWESS: a program for smoothing scatterplots by robust locally
    weighted regression. The American Statistician 35, 54. doi:10.2307/2683591

    de Boor C (1978). A Practical Guide to Splines. Springer, New York.
    doi:10.1007/978-1-4612-6333-3

    Hastie TJ (1992). Generalized additive models. Chapter 7 of Statistical Models in S,
    eds Chambers JM and Hastie TJ. Wadsworth & Brooks/Cole, Pacific Grove.
"""

import math
import numbers
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._masked_arrays import Missing, reject_masked, unmasked_array
from cytepool._options import integral_number, real_number
from cytepool.stats._optimize import brent_minimum

__all__ = ["choose_lowess_span", "lowess", "natural_spline_basis"]

FloatArray = npt.NDArray[np.float64]

# The neighbourhood holds floor(span * n + _SPAN_FUZZ) points, so that a
# product span * n that rounding leaves just below an integer still counts as
# that integer (found by probing R).
_SPAN_FUZZ = 1e-7
# Relative cut-offs of the tricube and bisquare weights (found by probing R).
_NEAR = 0.001
_FAR = 0.999
# A local fit is a weighted mean when the weighted standard deviation of x is
# at most this fraction of the range of x (threshold found by probing R).
_FLAT_NEIGHBOURHOOD = 0.001
# Robustness iterations stop when six times the median absolute residual is
# below this fraction of the mean absolute residual (found by probing R).
_NEGLIGIBLE_SCALE = 1e-7
# limma's weightedLowess takes the prior-weighted mean of a window narrower than
# _NARROW_WINDOW, and a weighted mean instead of a line when the weighted sum of
# squares of x about its mean in a window is below _FLAT_WINDOW. Both are absolute,
# so the fit depends on the units of x and the scale of the weights (found by
# probing).
_NARROW_WINDOW = 1e-7
_FLAT_WINDOW = 1e-7
# An interior knot that a quantile places on a boundary knot is moved inside
# by this fraction of its distance to the nearest other knot, as R's ns()
# does (found by probing).
_KNOT_SHOVE = 0.125
_CUBIC_ORDER = 4
# R's smooth.spline(x, y, df = df), the smoother of qvalue's pi0est. Its help page
# gives the fit, (X'WX + lambda Sigma) c = X'Wy for the B-splines X of a knot at every
# distinct x scaled to [0, 1], lambda = r 256^(3 spar - 1), and the criterion of the
# search for spar, 3 + (df - tr S)^2; the rest was found by probing R 4.4.3:
# - Sigma integrates the products of the second derivatives of the B-splines, which
#   are linear between knots, with the weight 0.333 for the product of their slopes
#   where the exact integral has 1/3 (Sigma of keep.stuff = TRUE has these entries);
# - r is the ratio of the sums of the 3rd to the (nk - 3)th diagonal entries of
#   X'WX and Sigma, for nk B-splines;
# - spar is searched in [-1.5, 1.5] by Brent's method with tolerance 1e-4 and
#   relative precision 2e-8 (control.spar's trace shows the steps);
# - the fit is that of the last spar evaluated, within the search's tolerance of the
#   best one, which R reports as spar.
_SPAR_INTERVAL = (-1.5, 1.5)
_SPAR_TOLERANCE = 1e-4
_SPAR_RELATIVE = 2e-8
_SLOPE_PRODUCT = 0.333
_CRITERION_OFFSET = 3.0


def lowess(
    x: npt.ArrayLike,
    y: npt.ArrayLike,
    *,
    span: float = 2.0 / 3.0,
    iterations: int = 3,
    delta: float | None = None,
) -> FloatArray:
    """Smooth a scatterplot by robust locally weighted linear regression.

    Cleveland's LOWESS, with the conventions of R's ``lowess(x, y, f=span,
    iter=iterations, delta=delta)``; the module docstring gives the method.
    Unlike R, which returns the fitted values sorted by x, the values are
    returned in the order of the input points. Every point tied in x gets the
    same fitted value, so the result does not depend on the order of the
    input points (up to rounding when x has ties).

    Args:
        x: ``(n,)`` finite values of the predictor.
        y: ``(n,)`` finite values of the response.
        span: Fraction of the points in each local fit. A span of 1 or more
            uses all points, as in R; a span below 2 / n still uses two.
        iterations: Number of robustness iterations after the first fit.
        delta: Distance in x within which a point is interpolated from the
            neighbouring fits instead of being fitted. Defaults to 1% of the
            range of x, as in R; 0 fits every distinct x.

    Returns:
        ``(n,)`` fitted values, in the order of ``x``. A single point is
        returned unchanged.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array, or
            ``iterations`` is not an integer.
        ValueError: If ``x`` and ``y`` are not 1-D of the same non-zero
            length or are not finite, ``span`` is not positive and finite,
            ``iterations`` is negative, or ``delta`` is negative or not
            finite.
    """
    x_values = _real_vector(x, "x")
    y_values = _real_vector(y, "y")
    if x_values.shape != y_values.shape:
        raise ValueError(
            f"x and y must have the same length, got {x_values.size} and "
            f"{y_values.size}"
        )
    n = x_values.size
    if n == 0:
        raise ValueError("x and y must hold at least one point")
    if not np.isfinite(x_values).all():
        raise ValueError("x must be finite; it contains NaN or infinite values")
    if not np.isfinite(y_values).all():
        raise ValueError("y must be finite; it contains NaN or infinite values")
    fraction = real_number(span, "span")
    if not (math.isfinite(fraction) and fraction > 0.0):
        raise ValueError(f"span must be positive and finite, got {fraction}")
    robustness_iterations = _whole_number(iterations, "iterations")

    order = np.argsort(x_values, kind="stable")
    x_sorted = np.ascontiguousarray(x_values[order], dtype=np.float64)
    y_sorted = np.ascontiguousarray(y_values[order], dtype=np.float64)
    x_range = float(x_sorted[-1]) - float(x_sorted[0])
    if not math.isfinite(x_range):
        raise ValueError("the range of x must be finite; it overflows")
    if delta is None:
        skip = 0.01 * x_range
    else:
        skip = real_number(delta, "delta")
        if not (math.isfinite(skip) and skip >= 0.0):
            raise ValueError(f"delta must be non-negative and finite, got {skip}")
    if n == 1:
        return y_sorted.copy()
    if fraction >= 1.0:
        neighbours = n
    else:
        neighbours = max(2, min(n, math.floor(fraction * n + _SPAN_FUZZ)))
    fitted_sorted = _robust_lowess_sorted(
        x_sorted, y_sorted, int(neighbours), robustness_iterations, float(skip), x_range
    )
    fitted = np.empty(n)
    fitted[order] = fitted_sorted
    return fitted


def choose_lowess_span(
    n: int, *, small_n: int = 50, min_span: float = 0.3, power: float = 1 / 3
) -> float:
    """The span of a LOWESS trend through n points, by limma's rule.

    limma's ``chooseLowessSpan(n, small.n, min.span, power)``: the span is 1 for
    n up to ``small_n``, and

        min_span + (1 - min_span) (small_n / n)^power

    for larger n, which falls toward ``min_span`` as n grows and stays above it.
    With the defaults, limma's, the span is 1 up to 50 points, 0.65 at 400 and
    0.395 at 20,000. limma tunes the rule for mean-variance trends, which are
    usually monotonic and so suit large spans;
    :func:`~cytepool.stats.voom.voom` chooses its span with it from the number
    of features, as limma's ``voom`` does.

    limma computes its formula, capped at 1, for any arguments. This function
    refuses arguments outside the ranges below, where limma gives NaN for a
    negative n, and spans below 1 for fewer than ``small_n`` points for a
    negative ``power`` or a ``min_span`` above 1. At 0 points with a
    ``min_span`` of 1 or a ``small_n`` of 0, limma's formula gives NaN, and
    this function gives 1, as limma's documentation does.

    Args:
        n: Number of points to smooth, a non-negative integer.
        small_n: Number of points up to which the span is 1, a non-negative
            integer.
        min_span: Span that the rule approaches as n grows, in (0, 1].
        power: Rate at which the span falls with n, in [0, 1]; 0 gives a span
            of 1 for every n.

    Returns:
        The span, the fraction of the points in each local fit of
        :func:`lowess`: in [``min_span``, 1].

    Raises:
        TypeError: If ``n`` or ``small_n`` is not an integer, or ``min_span``
            or ``power`` is not a real number.
        ValueError: If ``n`` or ``small_n`` is negative, ``min_span`` is not in
            (0, 1], or ``power`` is not in [0, 1].
    """
    points = _whole_number(n, "n")
    full = _whole_number(small_n, "small_n")
    lower = real_number(min_span, "min_span")
    rate = real_number(power, "power")
    if not 0.0 < lower <= 1.0:
        raise ValueError(f"min_span must be in (0, 1], got {lower}")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"power must be in [0, 1], got {rate}")
    if points <= full:
        return 1.0
    # limma's operations in its order, which give its values to the last bit
    # (found by probing). limma also caps the span at 1 (found by probing); within
    # the ranges above, the formula stays at most 1 beyond small_n.
    shrink: float = (full / points) ** rate
    return lower + (1.0 - lower) * shrink


def natural_spline_basis(
    x: npt.ArrayLike,
    df: int | None = None,
    *,
    knots: npt.ArrayLike | None = None,
    intercept: bool = False,
    boundary_knots: npt.ArrayLike | None = None,
) -> FloatArray:
    """Basis of natural cubic splines evaluated at x.

    The columns span the same space as R's ``splines::ns(x, df=df,
    knots=knots, intercept=intercept, Boundary.knots=boundary_knots)``, so a
    least squares fit on either gives the same fitted values; the columns
    themselves differ from R's. Without an intercept the splines are those
    that vanish at the lower boundary knot, as in R.

    With ``df``, the ``df - 1 - intercept`` interior knots are placed at
    quantiles of x (R's default type 7) with probabilities k / (K + 1),
    k = 1, ..., K, over the values of x within the boundary knots. Ties in x
    can make knots coincide, which lowers the smoothness there as for
    repeated knots. A knot that falls on a boundary knot is moved inside, as
    R's ``ns`` does, by one eighth of its distance to the nearest other knot
    (first at the lower boundary, then at the upper one).

    NaN in x gives a row of NaN, as R's ``NA`` does; knots and boundary knots
    are computed from the other values. Where R's ``ns`` warns and increases a
    ``df`` below ``1 + intercept``, or accepts a knot on the lower boundary
    knot (returning a basis with a degenerate column), this function raises
    an error.

    Args:
        x: ``(n,)`` values at which to evaluate the basis; NaN marks a
            missing value.
        df: Number of columns. Give ``df`` or ``knots``; given both, they
            must agree.
        knots: Interior knots (one or a sequence), strictly inside the
            boundary knots; repeated knots are allowed. Defaults to none, or
            to quantiles when ``df`` is given.
        intercept: Whether the basis spans constant functions.
        boundary_knots: The two knots beyond which the splines are linear.
            Defaults to the range of the observed values of x; given in
            either order.

    Returns:
        ``(n, df)`` matrix, with ``df = len(knots) + 1 + intercept``.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array,
            ``df`` is not an integer or ``intercept`` is not a bool.
        ValueError: If ``x`` is not 1-D or has infinite values, ``df`` is
            too small, knots are not finite or not strictly inside the
            boundary knots, the boundary knots are not two distinct finite
            values, or there are too few values of x to place the knots.
    """
    # Searched before np.atleast_1d could turn numpy.ma.masked in a list into NaN.
    reject_masked(knots, "knots")
    reject_masked(boundary_knots, "boundary_knots")
    values = _real_vector(x, "x", missing="NaN")
    if np.isinf(values).any():
        raise ValueError(
            "x must not contain infinite values; NaN marks a missing value"
        )
    if not isinstance(intercept, (bool, np.bool_)):
        raise TypeError(
            f"intercept must be True or False, got {type(intercept).__name__}"
        )
    with_intercept = bool(intercept)
    if df is not None:
        if isinstance(df, (bool, np.bool_)) or not isinstance(df, numbers.Integral):
            raise TypeError(f"df must be an integer, got {type(df).__name__}")
        smallest = 1 + with_intercept
        if df < smallest:
            raise ValueError(
                f"df must be at least {smallest} with intercept={with_intercept}, "
                f"got {df}"
            )
    observed = ~np.isnan(values)
    lower, upper = _boundary_knots(boundary_knots, values[observed])
    if knots is not None:
        interior = np.sort(_real_vector(np.atleast_1d(knots), "knots"))
        if not np.isfinite(interior).all():
            raise ValueError("knots must be finite")
        if df is not None and df != interior.size + 1 + with_intercept:
            raise ValueError(
                f"df={df} disagrees with {interior.size} knots, which give "
                f"{interior.size + 1 + with_intercept} columns; give one of them"
            )
        outside = (interior <= lower) | (interior >= upper)
        if outside.any():
            raise ValueError(
                f"knots must lie strictly inside the boundary knots ({lower!r}, "
                f"{upper!r}); got {interior[outside].tolist()}"
            )
    elif df is not None:
        interior = _quantile_knots(
            values[observed], int(df) - 1 - with_intercept, lower, upper
        )
    else:
        interior = np.empty(0)

    knot_sequence = np.concatenate(
        [np.full(_CUBIC_ORDER, lower), interior, np.full(_CUBIC_ORDER, upper)]
    )
    bounds = np.array([lower, upper])
    splines = _bspline_design(values[observed], knot_sequence, lower, upper)
    constraints = _bspline_derivatives(bounds, knot_sequence, 2)
    if not with_intercept:
        # Every B-spline but the first vanishes at the lower boundary knot.
        splines = splines[:, 1:]
        constraints = constraints[:, 1:]
    # The basis of the natural splines is the B-spline basis times an
    # orthonormal basis of the null space of the two boundary conditions.
    scale = np.abs(constraints).max(axis=1, keepdims=True)
    q, _ = np.linalg.qr(
        (constraints / np.where(scale > 0.0, scale, 1.0)).T, mode="complete"
    )
    basis = np.full((values.size, splines.shape[1] - 2), np.nan)
    basis[observed] = splines @ q[:, 2:]
    return basis


def _robust_lowess_sorted(
    x: FloatArray,
    y: FloatArray,
    neighbours: int,
    iterations: int,
    delta: float,
    x_range: float,
) -> FloatArray:
    """LOWESS fitted values of at least two points sorted by x.

    Args:
        x: ``(n,)`` sorted predictor values.
        y: ``(n,)`` responses in the same order.
        neighbours: Points in each neighbourhood, from 2 to n.
        iterations: Robustness iterations after the first fit.
        delta: Distance within which points are interpolated.
        x_range: ``x[-1] - x[0]``.

    Returns:
        ``(n,)`` fitted values in the order of ``x``.
    """
    n = x.size
    fitted = np.empty(n)
    robustness = np.ones(n)
    weights = np.zeros(n)
    for iteration in range(iterations + 1):
        _lowess_pass(
            x, y, neighbours, delta, x_range, iteration > 0, robustness, weights, fitted
        )
        if iteration == iterations:
            break
        residuals = np.abs(y - fitted)
        ordered = np.sort(residuals)
        middle = n // 2
        if n % 2 == 0:
            scale = 3.0 * (ordered[middle] + ordered[n - middle - 1])
        else:
            scale = 6.0 * ordered[middle]
        if scale < _NEGLIGIBLE_SCALE * (residuals.sum() / n):
            break
        # Tukey's bisquare of the residuals; with every residual zero, every
        # weight is one.
        with np.errstate(divide="ignore", invalid="ignore"):
            u = residuals / scale
        v = 1.0 - u * u
        bisquare = np.where(residuals <= _FAR * scale, v * v, 0.0)
        robustness = np.where(residuals <= _NEAR * scale, 1.0, bisquare)
    return fitted


@jit
def _lowess_pass(
    x: FloatArray,
    y: FloatArray,
    neighbours: int,
    delta: float,
    x_range: float,
    robust: bool,
    robustness: FloatArray,
    weights: FloatArray,
    fitted: FloatArray,
) -> None:
    """One sweep of local fits along sorted x, writing every fitted value.

    The local fits weight points by the tricube of their distance, times
    ``robustness`` when ``robust`` is set; ``weights`` is workspace. The
    order of the floating point operations reproduces R's results to the
    last bit (found by probing).
    """
    n = x.shape[0]
    left = 0
    right = neighbours - 1
    last = -1
    i = 0
    while True:
        # The neighbourhood is the window of points left..right. It moves
        # right while the point after it is nearer to x[i] than its first
        # point; a tie leaves it in place.
        while right < n - 1 and x[i] - x[left] > x[right + 1] - x[i]:
            left += 1
            right += 1
        centre = x[i]
        radius = max(centre - x[left], x[right] - centre)
        near = _NEAR * radius
        far = _FAR * radius
        # Points tied with the right end of the window carry weight only
        # when the radius is zero; the scan goes on until x passes the radius.
        total = 0.0
        end = left
        for j in range(left, n):
            distance = abs(x[j] - centre)
            if distance <= far:
                if distance <= near:
                    weight = 1.0
                else:
                    t = distance / radius
                    u = 1.0 - t * t * t
                    weight = u * u * u
                if robust:
                    weight = weight * robustness[j]
                weights[j] = weight
                total += weight
            elif x[j] > centre:
                break
            else:
                weights[j] = 0.0
            end = j + 1

        if total <= 0.0:
            value = y[i]
        else:
            # Normalised weights; the local line, when the spread of x allows
            # one, is a sum of the y values with adjusted weights.
            mean = 0.0
            for j in range(left, end):
                weights[j] = weights[j] / total
                mean += weights[j] * x[j]
            spread = 0.0
            if radius > 0.0:
                for j in range(left, end):
                    difference = x[j] - mean
                    spread += weights[j] * (difference * difference)
            value = 0.0
            if radius > 0.0 and math.sqrt(spread) > _FLAT_NEIGHBOURHOOD * x_range:
                slope = (centre - mean) / spread
                for j in range(left, end):
                    value += weights[j] * (slope * (x[j] - mean) + 1.0) * y[j]
            else:
                for j in range(left, end):
                    value += weights[j] * y[j]
        fitted[i] = value

        if last < i - 1:
            # Points skipped since the last fit are interpolated linearly.
            width = x[i] - x[last]
            for j in range(last + 1, i):
                alpha = (x[j] - x[last]) / width
                fitted[j] = alpha * fitted[i] + (1.0 - alpha) * fitted[last]
        last = i
        # Points tied with the last fit take its value. The next fit is at
        # the last point within delta of it, or at the point after it.
        cut = x[last] + delta
        k = last + 1
        while k < n and x[k] <= cut:
            if x[k] == x[last]:
                fitted[k] = fitted[last]
                last = k
            k += 1
        if last >= n - 1:
            return
        i = max(last + 1, k - 1)


def _weighted_lowess(
    x: FloatArray, y: FloatArray, weights: FloatArray, span: float, npts: int = 200
) -> FloatArray:
    """One pass of LOWESS with prior weights, as limma's ``weightedLowess``.

    limma's ``weightedLowess(x, y, weights, span = span, iterations = 1)`` with its
    default ``npts`` and ``delta``, which ``loessFit`` calls for unequal weights
    (unless the points are too few, when it fits a straight line); the empirical
    Bayes trend of :mod:`cytepool.stats.empirical_bayes` uses it.
    limma's help page describes the method; the conventions it leaves open were
    found by probing limma 3.62.1:

    - The local fits are computed at anchor points, the other points are
      interpolated linearly between them, and points tied in x share a value.
      The first point is an anchor, each next anchor is the first point more
      than ``delta`` beyond the previous one, and the last point is an anchor.
      ``delta`` is 0 (every distinct x an anchor) when x has at most ``npts``
      distinct values; otherwise it solves sum_i min(g_i, delta) = npts delta
      over the gaps g_i between successive distinct values, which is the range
      over ``npts`` for equally spaced x and leaves out the large gaps.
    - The window of an anchor starts with the points tied with it and grows by
      the nearest distinct x-value on either side, with every point tied at that
      value, the right one when both are as near, until its weights sum to at
      least ``span`` times the total weight; h is the distance to the farthest
      value in it.
    - A window with h below 1e-7 gives the prior-weighted mean of its points.
      Otherwise a point at distance r gets the prior weight times the tricube
      (1 - (r / h)^3)^3, with no cut-off (unlike R's ``lowess``), and the fitted
      value is that of the weighted least squares line, or the weighted mean when
      the weighted sum of squares of x about its mean is below 1e-7. Both bounds
      are absolute, so the fit depends on the units of x and the scale of the
      weights.

    Args:
        x: ``(n,)`` finite predictor values, n >= 1.
        y: ``(n,)`` finite responses.
        weights: ``(n,)`` positive prior weights.
        span: Fraction of the total weight in each window, in (0, 1].
        npts: Approximate number of anchor points.

    Returns:
        ``(n,)`` fitted values in the order of ``x``.
    """
    order = np.argsort(x, kind="stable")
    x_sorted = np.ascontiguousarray(x[order], dtype=np.float64)
    fitted_sorted = _weighted_lowess_sorted(
        x_sorted,
        np.ascontiguousarray(y[order], dtype=np.float64),
        np.ascontiguousarray(weights[order], dtype=np.float64),
        float(span),
        _anchor_spacing(x_sorted, npts),
    )
    fitted = np.empty(x.size)
    fitted[order] = fitted_sorted
    return fitted


def _anchor_spacing(x_sorted: FloatArray, npts: int) -> float:
    """The d >= 0 with sum(min(gaps, d)) = npts d, or 0 for at most npts values.

    The gaps are those between successive distinct values. The function
    sum(min(g, d)) - npts d is concave and piecewise linear in d, with slope
    (number of gaps above d) - npts, so with more than npts gaps it has one
    positive root, found on the interval between two successive sorted gaps.
    With exactly npts gaps every d up to the smallest gap is a root, and the
    largest, the smallest gap, is taken (found by probing).
    """
    gaps = np.sort(np.diff(np.unique(x_sorted)))
    m = gaps.size
    if m < npts:
        return 0.0
    cumulative = np.concatenate(([0.0], np.cumsum(gaps)))
    for k in range(m - npts + 1, m + 1):
        # Below delta: the k smallest gaps; the other m - k count delta each.
        delta = float(cumulative[k]) / (npts - m + k)
        upper = float(gaps[k]) if k < m else math.inf
        if float(gaps[k - 1]) <= delta <= upper:
            return delta
    return float(cumulative[m]) / npts


@jit
def _weighted_lowess_sorted(
    x: FloatArray, y: FloatArray, weights: FloatArray, span: float, delta: float
) -> FloatArray:
    """Weighted LOWESS fitted values of points sorted by x (one pass)."""
    n = x.size
    fitted = np.empty(n)
    total = 0.0
    for j in range(n):
        total += weights[j]
    target = span * total
    window = np.empty(n)
    last = -1
    i = 0
    while True:
        fitted[i] = _weighted_local_fit(x, y, weights, i, target, window)
        if last >= 0:
            # Points between the anchors are interpolated; ties take a value.
            width = x[i] - x[last]
            for j in range(last + 1, i):
                if x[j] == x[last]:
                    fitted[j] = fitted[last]
                elif x[j] == x[i]:
                    fitted[j] = fitted[i]
                else:
                    alpha = (x[j] - x[last]) / width
                    fitted[j] = alpha * fitted[i] + (1.0 - alpha) * fitted[last]
        last = i
        if i == n - 1:
            break
        cut = x[i] + delta
        k = i + 1
        while k < n - 1 and x[k] <= cut:
            k += 1
        # The next anchor is the last of the points tied with it.
        while k < n - 1 and x[k + 1] == x[k]:
            k += 1
        i = k
    return fitted


@jit
def _weighted_local_fit(
    x: FloatArray,
    y: FloatArray,
    weights: FloatArray,
    anchor: int,
    target: float,
    window: FloatArray,
) -> float:
    """The weighted local linear fit at one anchor; ``window`` is workspace."""
    n = x.size
    x0 = x[anchor]
    left = anchor
    while left > 0 and x[left - 1] == x0:
        left -= 1
    right = anchor
    while right < n - 1 and x[right + 1] == x0:
        right += 1
    accumulated = 0.0
    for j in range(left, right + 1):
        accumulated += weights[j]
    # The window grows by whole groups of tied values, the nearer first and the
    # right one at equal distances (found by probing).
    while accumulated < target and (left > 0 or right < n - 1):
        to_left = x0 - x[left - 1] if left > 0 else np.inf
        to_right = x[right + 1] - x0 if right < n - 1 else np.inf
        if to_right <= to_left:
            value = x[right + 1]
            while right < n - 1 and x[right + 1] == value:
                right += 1
                accumulated += weights[right]
        else:
            value = x[left - 1]
            while left > 0 and x[left - 1] == value:
                left -= 1
                accumulated += weights[left]
    h = max(x0 - x[left], x[right] - x0)
    sum_w = 0.0
    sum_wy = 0.0
    if h < _NARROW_WINDOW:
        for j in range(left, right + 1):
            sum_w += weights[j]
            sum_wy += weights[j] * y[j]
        narrow: float = sum_wy / sum_w
        return narrow
    sum_wx = 0.0
    for j in range(left, right + 1):
        r = abs(x[j] - x0)
        if r < h:
            c = r / h
            c = 1.0 - c * c * c
            tricube = c * c * c
        else:
            tricube = 0.0
        w = weights[j] * tricube
        window[j] = w
        sum_w += w
        sum_wx += w * x[j]
        sum_wy += w * y[j]
    mean_x = sum_wx / sum_w
    mean_y = sum_wy / sum_w
    sxx = 0.0
    sxy = 0.0
    for j in range(left, right + 1):
        dx = x[j] - mean_x
        sxx += window[j] * dx * dx
        sxy += window[j] * dx * (y[j] - mean_y)
    if sxx < _FLAT_WINDOW:
        return mean_y
    fit: float = mean_y + sxy / sxx * (x0 - mean_x)
    return fit


def _boundary_knots(
    boundary_knots: npt.ArrayLike | None, observed: FloatArray
) -> tuple[float, float]:
    if boundary_knots is None:
        if observed.size == 0:
            raise ValueError(
                "x has no observed (non-NaN) values; give boundary_knots and knots"
            )
        lower, upper = float(observed.min()), float(observed.max())
        if not lower < upper:
            raise ValueError(
                "x must have at least two distinct values to set the boundary "
                f"knots, got only {lower!r}; give boundary_knots"
            )
        return lower, upper
    bounds = _real_vector(boundary_knots, "boundary_knots")
    if bounds.shape != (2,):
        raise ValueError(
            f"boundary_knots must hold two values, got shape {bounds.shape}"
        )
    if not np.isfinite(bounds).all():
        raise ValueError("boundary_knots must be finite")
    lower, upper = sorted(float(b) for b in bounds)
    if not lower < upper:
        raise ValueError(f"boundary_knots must be distinct, got {lower!r} twice")
    return lower, upper


def _quantile_knots(
    observed: FloatArray, count: int, lower: float, upper: float
) -> FloatArray:
    """Interior knots at quantiles of the values within the boundary knots."""
    if count == 0:
        return np.empty(0)
    inside = np.sort(observed[(observed >= lower) & (observed <= upper)])
    if inside.size == 0:
        raise ValueError(
            "no value of x lies within the boundary knots, so no knot can be "
            "placed at its quantiles; give knots"
        )
    # Probabilities k / (count + 1) and R's type 7 quantiles, computed in the
    # same floating point operations as R (found by probing).
    probabilities = np.arange(1, count + 1) * (1.0 / (count + 1))
    position = (inside.size - 1) * probabilities
    below = np.floor(position).astype(np.int64)
    above = np.ceil(position).astype(np.int64)
    fraction = position - below
    knots = inside[below].copy()
    blend = (position > below) & (inside[above] != knots)
    knots[blend] = (1.0 - fraction[blend]) * knots[blend] + fraction[blend] * inside[
        above[blend]
    ]
    at_lower = knots == lower
    if at_lower.any():
        if not (knots > lower).any():
            raise ValueError(
                "every interior knot falls on the lower boundary knot: x has too "
                "few distinct values within the boundary knots for this df; "
                "lower df or give knots"
            )
        knots[at_lower] = lower + _KNOT_SHOVE * (knots[knots > lower].min() - lower)
    at_upper = knots == upper
    if at_upper.any():
        if not (knots < upper).any():
            raise ValueError(
                "every interior knot falls on the upper boundary knot: x has too "
                "few distinct values within the boundary knots for this df; "
                "lower df or give knots"
            )
        knots[at_upper] = upper - _KNOT_SHOVE * (upper - knots[knots < upper].max())
    if ((knots <= lower) | (knots >= upper)).any():
        # Only possible when a knot lies within rounding error of a boundary
        # knot; R fails here too.
        raise ValueError(
            "an interior knot placed at a quantile of x lies on a boundary knot "
            "even after moving it inside; lower df or give knots"
        )
    return knots


def _bspline_design(
    points: FloatArray, knot_sequence: FloatArray, lower: float, upper: float
) -> FloatArray:
    """Cubic B-splines at points, continued linearly beyond the boundary knots."""
    design = np.empty((points.size, knot_sequence.size - _CUBIC_ORDER))
    inside = (points >= lower) & (points <= upper)
    design[inside] = _bspline_derivatives(points[inside], knot_sequence, 0)
    for bound, beyond in ((lower, points < lower), (upper, points > upper)):
        if beyond.any():
            at = np.array([bound])
            value = _bspline_derivatives(at, knot_sequence, 0)
            slope = _bspline_derivatives(at, knot_sequence, 1)
            design[beyond] = value + (points[beyond] - bound)[:, np.newaxis] * slope
    return design


def _bspline_derivatives(
    points: FloatArray, knot_sequence: FloatArray, derivative: int
) -> FloatArray:
    """Values or derivatives of the cubic B-splines on a knot sequence.

    Points must lie within the boundary knots, the fourth knot from each end.
    B-splines are right-continuous at interior knots; at the upper boundary
    knot they take their limits from the left. Values come from the Cox-de
    Boor recursion and derivatives from the difference of B-splines of lower
    order (de Boor 1978), with every term of a zero-length interval taken as
    zero.

    Returns:
        ``(n_points, len(knot_sequence) - 4)`` matrix.
    """
    t = knot_sequence
    n_intervals = t.size - 1
    nonempty = np.flatnonzero(t[:-1] < t[1:])
    interval = np.searchsorted(t, points, side="right") - 1
    # A point on the upper boundary knot belongs to the last nonempty interval.
    interval = np.minimum(interval, nonempty[-1])
    values = np.zeros((points.size, n_intervals))
    values[np.arange(points.size), interval] = 1.0
    for order in range(2, _CUBIC_ORDER - derivative + 1):
        values = _raise_order(values, points, t, order)
    for order in range(_CUBIC_ORDER - derivative + 1, _CUBIC_ORDER + 1):
        values = _differentiate(values, t, order)
    return values


def _raise_order(
    values: FloatArray, points: FloatArray, t: FloatArray, order: int
) -> FloatArray:
    """B-splines of the given order from those of the order below."""
    count = t.size - order
    start = t[:count]
    width_left = t[order - 1 : order - 1 + count] - start
    end = t[order : order + count]
    width_right = end - t[1 : 1 + count]
    left = _ratio(points[:, np.newaxis] - start, width_left) * values[:, :count]
    right = _ratio(end - points[:, np.newaxis], width_right) * values[:, 1 : count + 1]
    result: FloatArray = left + right
    return result


def _differentiate(values: FloatArray, t: FloatArray, order: int) -> FloatArray:
    """Derivatives of B-splines of the given order from the order below."""
    count = t.size - order
    width_left = t[order - 1 : order - 1 + count] - t[:count]
    width_right = t[order : order + count] - t[1 : 1 + count]
    result: FloatArray = (order - 1) * (
        _ratio(values[:, :count], width_left)
        - _ratio(values[:, 1 : count + 1], width_right)
    )
    return result


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """numerator / denominator, zero where the denominator is zero."""
    safe = np.where(denominator > 0.0, denominator, 1.0)
    result: FloatArray = np.where(denominator > 0.0, numerator / safe, 0.0)
    return result


def _smoothing_spline_fit(x: FloatArray, y: FloatArray, df: float) -> FloatArray:
    """Fitted values at x of R's ``smooth.spline(x, y, df = df)`` with unit weights.

    The cubic smoothing spline with a knot at every x whose equivalent degrees of
    freedom, the trace of its smoother matrix, are as near ``df`` as R's search for
    its smoothing parameter takes them; the constants above say how R computes it.
    x must hold at least four distinct values in increasing order, and ``df`` lie in
    (1, len(x)]: the callers check both. With more than 49 values R keeps a knot at
    some of them only, a regression spline; here every value is a knot.
    """
    scaled = (x - x[0]) / (x[-1] - x[0])
    ends = _CUBIC_ORDER - 1
    knots = np.concatenate([np.zeros(ends), scaled, np.ones(ends)])
    basis = np.ascontiguousarray(_bspline_derivatives(scaled, knots, 0))
    penalty = _spline_penalty(scaled, knots)
    gram, rhs = _normal_equations(basis, np.ascontiguousarray(y, dtype=np.float64))
    n_splines = basis.shape[1]
    inner = range(ends - 1, n_splines - ends)
    ratio = sum(gram[i, i] for i in inner) / sum(penalty[i, i] for i in inner)
    fitted = np.empty(x.size)
    last = [math.nan]

    def criterion(spar: float) -> float:
        last[0] = spar
        lam = ratio * 256.0 ** (3.0 * spar - 1.0)
        trace = _penalized_fit(basis, gram, penalty, rhs, lam, fitted)
        # A smoother whose system rounding makes singular is far from any df.
        return _CRITERION_OFFSET + (df - trace) ** 2 if trace == trace else math.inf

    low, high = _SPAR_INTERVAL
    brent_minimum(criterion, low, high, _SPAR_TOLERANCE, relative=_SPAR_RELATIVE)
    _penalized_fit(
        basis, gram, penalty, rhs, ratio * 256.0 ** (3.0 * last[0] - 1.0), fitted
    )
    return fitted


def _spline_penalty(scaled: FloatArray, knots: FloatArray) -> FloatArray:
    """Sigma, the integrals of the products of the second derivatives of the cubic
    B-splines on [0, 1], as R's smooth.spline weighs them (``_SLOPE_PRODUCT``)."""
    second = _bspline_derivatives(scaled, knots, 2)
    n_splines = second.shape[1]
    penalty = np.zeros((n_splines, n_splines))
    for i in range(scaled.size - 1):
        width = scaled[i + 1] - scaled[i]
        start = second[i]
        slope = second[i + 1] - start
        penalty += width * (
            np.outer(start, start)
            + 0.5 * (np.outer(slope, start) + np.outer(start, slope))
            + _SLOPE_PRODUCT * np.outer(slope, slope)
        )
    return penalty


@jit
def _normal_equations(
    basis: FloatArray, y: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """X'X and X'y for the B-splines X, with unit weights."""
    n, n_splines = basis.shape
    gram = np.zeros((n_splines, n_splines))
    rhs = np.zeros(n_splines)
    for i in range(n):
        for j in range(n_splines):
            b = basis[i, j]
            if b != 0.0:
                rhs[j] += b * y[i]
                for k in range(n_splines):
                    gram[j, k] += b * basis[i, k]
    return gram, rhs


@jit(error_model="numpy")
def _penalized_fit(
    basis: FloatArray,
    gram: FloatArray,
    penalty: FloatArray,
    rhs: FloatArray,
    lam: float,
    fitted: FloatArray,
) -> float:
    """Solve (X'X + lam Sigma) c = X'y by Cholesky's factorization, write the fitted
    values X c into ``fitted``, and return the trace of the smoother matrix,
    sum over i of x_i' (X'X + lam Sigma)^-1 x_i; NaN where the factorization fails."""
    n, n_splines = basis.shape
    factor = gram + lam * penalty
    for j in range(n_splines):
        pivot = factor[j, j]
        for k in range(j):
            pivot -= factor[j, k] * factor[j, k]
        if not pivot > 0.0:
            return math.nan
        root = math.sqrt(pivot)
        factor[j, j] = root
        for i in range(j + 1, n_splines):
            entry = factor[i, j]
            for k in range(j):
                entry -= factor[i, k] * factor[j, k]
            factor[i, j] = entry / root
    coefficients = np.empty(n_splines)
    for j in range(n_splines):
        entry = rhs[j]
        for k in range(j):
            entry -= factor[j, k] * coefficients[k]
        coefficients[j] = entry / factor[j, j]
    for j in range(n_splines - 1, -1, -1):
        entry = coefficients[j]
        for k in range(j + 1, n_splines):
            entry -= factor[k, j] * coefficients[k]
        coefficients[j] = entry / factor[j, j]
    solved = np.empty(n_splines)
    trace = 0.0
    for i in range(n):
        value = 0.0
        for j in range(n_splines):
            value += basis[i, j] * coefficients[j]
            entry = basis[i, j]
            for k in range(j):
                entry -= factor[j, k] * solved[k]
            solved[j] = entry / factor[j, j]
            trace += solved[j] * solved[j]
        fitted[i] = value
    return trace


def _real_vector(values: Any, name: str, *, missing: Missing = None) -> FloatArray:
    array = unmasked_array(values, name, missing=missing)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _whole_number(value: Any, name: str) -> int:
    number = integral_number(value, name)
    if number < 0:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return number
