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

:func:`natural_spline_basis` returns a basis of the natural cubic splines
with given interior and boundary knots: cubic between the knots, twice
continuously differentiable at simple knots, and linear beyond the boundary
knots. Its column space matches that of R's ``splines::ns``. The basis is
built from the cubic B-splines on the knot sequence with each boundary knot
repeated four times (de Boor 1978), restricted to the subspace with zero
second derivative at both boundary knots.

References:
    Cleveland WS (1979). Robust locally weighted regression and smoothing
    scatterplots. Journal of the American Statistical Association 74,
    829-836.

    Cleveland WS (1981). LOWESS: a program for smoothing scatterplots by
    robust locally weighted regression. The American Statistician 35, 54.

    de Boor C (1978). A Practical Guide to Splines. Springer, New York.

    Hastie TJ (1992). Generalized additive models. Chapter 7 of Statistical
    Models in S, eds Chambers JM and Hastie TJ. Wadsworth & Brooks/Cole.
"""

import math
import numbers
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit

__all__ = ["lowess", "natural_spline_basis"]

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
# An interior knot that a quantile places on a boundary knot is moved inside
# by this fraction of its distance to the nearest other knot, as R's ns()
# does (found by probing).
_KNOT_SHOVE = 0.125
_CUBIC_ORDER = 4


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
        TypeError: If an input is not real-valued, or ``iterations`` is not
            an integer.
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
    fraction = _real_scalar(span, "span")
    if not (math.isfinite(fraction) and fraction > 0.0):
        raise ValueError(f"span must be positive and finite, got {fraction}")
    if isinstance(iterations, (bool, np.bool_)) or not isinstance(
        iterations, numbers.Integral
    ):
        raise TypeError(
            f"iterations must be an integer, got {type(iterations).__name__}"
        )
    if iterations < 0:
        raise ValueError(f"iterations must be non-negative, got {iterations}")

    order = np.argsort(x_values, kind="stable")
    x_sorted = np.ascontiguousarray(x_values[order], dtype=np.float64)
    y_sorted = np.ascontiguousarray(y_values[order], dtype=np.float64)
    x_range = float(x_sorted[-1]) - float(x_sorted[0])
    if not math.isfinite(x_range):
        raise ValueError("the range of x must be finite; it overflows")
    if delta is None:
        skip = 0.01 * x_range
    else:
        skip = _real_scalar(delta, "delta")
        if not (math.isfinite(skip) and skip >= 0.0):
            raise ValueError(f"delta must be non-negative and finite, got {skip}")
    if n == 1:
        return y_sorted.copy()
    if fraction >= 1.0:
        neighbours = n
    else:
        neighbours = max(2, min(n, math.floor(fraction * n + _SPAN_FUZZ)))
    fitted_sorted = _robust_lowess_sorted(
        x_sorted, y_sorted, int(neighbours), int(iterations), float(skip), x_range
    )
    fitted = np.empty(n)
    fitted[order] = fitted_sorted
    return fitted


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
        TypeError: If an input is not real-valued, ``df`` is not an integer
            or ``intercept`` is not a bool.
        ValueError: If ``x`` is not 1-D or has infinite values, ``df`` is
            too small, knots are not finite or not strictly inside the
            boundary knots, the boundary knots are not two distinct finite
            values, or there are too few values of x to place the knots.
    """
    values = _real_vector(x, "x")
    if np.isinf(values).any():
        raise ValueError(
            "x must not contain infinite values; NaN marks a missing value"
        )
    if not isinstance(intercept, (bool, np.bool_)):
        raise TypeError(f"intercept must be a bool, got {type(intercept).__name__}")
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


def _real_vector(values: Any, name: str) -> FloatArray:
    array = np.asarray(values)
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must hold real numbers, got dtype {array.dtype}")
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {array.shape}")
    return array.astype(np.float64, copy=False)


def _real_scalar(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, got {type(value).__name__}")
    return float(value)
