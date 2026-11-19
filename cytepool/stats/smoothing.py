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

:class:`NaturalSplineBasis` is a basis of the natural cubic splines with given
knots: cubic between the knots, twice continuously differentiable at simple knots,
and linear beyond the two boundary knots. It keeps its knots, gives its columns at
any points and records itself as plain values, in one of two forms that span the
same curves, so that fits and the F-tests of all columns agree (Durrleman and Simon
1989; Harrell 2015):

- ``"b_spline"``, the columns of R's ``splines::ns`` up to a change of basis: the
  cubic B-splines on the knot sequence with each boundary knot repeated four times
  (de Boor 1978), continued linearly beyond the boundary knots, times an
  orthonormal basis of the splines with zero second derivative at both boundary
  knots.
- ``"restricted_cubic"``, Harrell's restricted cubic spline: with all the knots
  t_1 < ... < t_k, the boundary knots included, and u_+ = max(u, 0), the columns
  x - t_1 and, for j = 1, ..., k - 2,
  [(x - t_j)_+^3 - (x - t_(k-1))_+^3 (t_k - t_j) / (t_k - t_(k-1))
  + (x - t_k)_+^3 (t_(k-1) - t_j) / (t_k - t_(k-1))] / (t_k - t_1)^2,
  as Hmisc's ``rcspline.eval`` gives them with its default normalization (Harrell
  2026), except that its first column is x itself. The first column is the linear
  part, so that the F-test of the other columns tests departure from a straight
  line (Desquilbet and Mariotti 2010).

Without an intercept every column vanishes at the lower boundary knot, in both
forms. The interior knots are given, or placed by one of three rules: R's rule for
``ns``, the quantiles of x with probabilities k / (K + 1) for K interior knots; the
same rule on the distinct values of x, each counted once; or Harrell's percentiles,
as ``rcspline.eval`` places all k knots: probabilities from 0.10 to 0.90 for 3
knots, 0.05 to 0.95 for 4 to 6 and 0.025 to 0.975 for more, equally spaced, with
the 5th smallest and 5th largest values as the outer knots when x has fewer than
100 values. Quantiles are R's type 7, computed in R's floating-point operations, so
that the knots equal R's to the last bit (found by probing). Where Hmisc places
fewer knots than asked or switches to another method, which its help page
describes only in outline, Harrell's rule here raises an error instead.
:func:`natural_spline_basis` gives the columns of the B-spline form at x.

References:
    Cleveland WS (1979). Robust locally weighted regression and smoothing scatterplots.
    Journal of the American Statistical Association 74, 829-836.
    doi:10.1080/01621459.1979.10481038

    Cleveland WS (1981). LOWESS: a program for smoothing scatterplots by robust locally
    weighted regression. The American Statistician 35, 54. doi:10.2307/2683591

    de Boor C (1978). A Practical Guide to Splines. Springer, New York.
    doi:10.1007/978-1-4612-6333-3

    Desquilbet L, Mariotti F (2010). Dose-response analyses using restricted cubic
    spline functions in public health research. Statistics in Medicine 29,
    1037-1057. doi:10.1002/sim.3841

    Durrleman S, Simon R (1989). Flexible regression models with cubic splines.
    Statistics in Medicine 8, 551-561. doi:10.1002/sim.4780080504

    Harrell FE (2015). Regression Modeling Strategies, 2nd edition. Springer, Cham.
    doi:10.1007/978-3-319-19425-7

    Harrell FE (2026). Hmisc: Harrell Miscellaneous. R package version 5.3-1, help
    page of rcspline.eval. https://cran.r-project.org/package=Hmisc

    Hastie TJ (1992). Generalized additive models. Chapter 7 of Statistical Models in S,
    eds Chambers JM and Hastie TJ. Wadsworth & Brooks/Cole, Pacific Grove.
"""

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._jit import jit
from cytepool._masked_arrays import Missing, reject_masked, unmasked_array
from cytepool._options import (
    checked_choice,
    checked_flag,
    integral_number,
    real_number,
)
from cytepool.stats._optimize import brent_minimum

__all__ = [
    "NaturalSplineBasis",
    "choose_lowess_span",
    "lowess",
    "natural_spline_basis",
]

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
_FORMS = ("b_spline", "restricted_cubic")
# The rules that place interior knots, and how messages name them.
_RULES = {
    "quantiles": "R's quantile rule",
    "levels": "R's quantile rule on the distinct values",
    "harrell": "Harrell's rule",
}
# Harrell's rule as the help page of Hmisc's rcspline.eval gives it: at least 3 knots;
# the outer knots of samples smaller than _SMALL_SAMPLE are the _FIFTH smallest and
# largest values; and when _TIED_SHARE of the values or more tie at an extreme, and no
# value between the extremes ties as often, Hmisc switches to another method.
_FEWEST_HARRELL_KNOTS = 3
_SMALL_SAMPLE = 100
_FIFTH = 5
_TIED_SHARE = 0.05
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


@dataclass(frozen=True, init=False)
class NaturalSplineBasis:
    """A basis of natural cubic splines that keeps its knots.

    The columns are cubic between the knots and linear beyond the two boundary knots.
    The basis is built from values of x, which place the knots, and gives its columns
    at any points with :meth:`matrix`: at the samples of a fit, or at new doses where
    the fitted curve is drawn. In a model with an intercept the F-test of all its
    columns tests any change along x; in the restricted cubic form, the F-test of the
    columns after the first tests departure from a straight line
    (https://docs.nygen.io/CytePool/statistics/linear-models#spline-bases).

    Both forms span the same curves for the same knots: ``"b_spline"`` gives the
    columns of R's ``splines::ns`` up to a change of basis, and
    ``"restricted_cubic"`` those of Hmisc's ``rcspline.eval`` and rms's ``rcs``,
    whose first column is x minus the lower boundary knot instead of x. The module
    docstring gives the formulas. Without an intercept every column vanishes at the
    lower boundary knot. :func:`natural_spline_basis` returns
    ``NaturalSplineBasis(x, df, knots=..., ...).matrix(x)`` in the B-spline form.

    The interior knots are given, or placed by a rule on the observed values of x
    (NaN marks a missing value) within the boundary knots:

    - ``"quantiles"``, R's rule for ``ns``: ``df - 1 - intercept`` knots at the
      quantiles of x with probabilities k / (K + 1); ties in x can make knots
      coincide, and a knot that falls on a boundary knot is moved inside by one
      eighth of its distance to the nearest other knot, as R does.
      ``attr(ns(x, df), "knots")`` gives the same knots.
    - ``"levels"``, the same rule on the distinct values of x, each counted once,
      so that a dose counts once whatever its number of replicates:
      ``attr(ns(unique(x), df), "knots")``. These knots never coincide.
    - ``"harrell"``, Harrell's percentiles, as ``rcspline.eval(x, nk = df + 1 -
      intercept)`` and rms's ``rcs`` place all the knots, the outer ones included,
      which become the boundary knots. It raises an error where Hmisc places fewer
      knots than asked or switches to the method it uses when 5% of the values or
      more tie at an extreme.

    With a rule and no ``df``, there are no interior knots, as in R's ``ns(x)``.

    Args:
        x: ``(n,)`` real values that place the knots; NaN marks a missing value.
            Needed by the rules, and for the boundary knots when they are not given.
        df: Number of columns, ``len(knots) + 1 + intercept``, as in R's ``ns``;
            needed by ``"harrell"``. With knots given, it must agree with them.
        knots: A rule, ``"quantiles"``, ``"levels"`` or ``"harrell"``, or the
            interior knots, strictly inside the boundary knots. The B-spline form
            accepts repeated knots, which lower the smoothness there; the restricted
            cubic form needs distinct knots.
        intercept: Whether the columns span constant functions.
        boundary_knots: The two knots beyond which the curves are linear, in either
            order. Defaults to the range of the observed values of x; refused with
            ``"harrell"``, which places them itself.
        form: ``"b_spline"`` or ``"restricted_cubic"``.

    Attributes:
        knots: The interior knots, ascending, as a tuple of floats.
        boundary_knots: The lower and upper boundary knots.
        intercept: Whether the columns span constant functions.
        form: ``"b_spline"`` or ``"restricted_cubic"``.

    The basis is immutable and can be pickled. Two bases are equal when their knots,
    boundary knots, intercept and form are, and ``NaturalSplineBasis(**basis.to_dict())``
    rebuilds a basis whose columns are identical to the last bit.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array, ``df``
            is not an integer, ``intercept`` is not a bool, or ``knots`` is None.
        ValueError: If ``x`` is not 1-D or has infinite values, ``df`` is too small,
            ``knots`` names no rule, the rule or the boundary knots need ``x``, given
            knots are not finite, disagree with ``df`` or are not strictly inside the
            boundary knots, the boundary knots are not two distinct finite values,
            the restricted cubic form gets repeated knots, or the values of x are too
            few, or too tied, to place the knots by the rule.

    Example:
        >>> import numpy as np
        >>> doses = np.log10(np.repeat([0.01, 0.1, 1.0, 10.0], 3))
        >>> basis = NaturalSplineBasis(
        ...     doses, df=2, knots="levels", form="restricted_cubic"
        ... )
        >>> basis.knots, basis.boundary_knots
        ((-0.5,), (-2.0, 1.0))
        >>> basis.matrix([-2.0, 1.0, 2.0]).round(4).tolist()
        [[0.0, 0.0], [3.0, 2.25], [4.0, 3.75]]
    """

    knots: tuple[float, ...]
    boundary_knots: tuple[float, float]
    intercept: bool
    form: str

    def __init__(
        self,
        x: npt.ArrayLike | None = None,
        df: int | None = None,
        *,
        knots: str | npt.ArrayLike = "quantiles",
        intercept: bool = False,
        boundary_knots: npt.ArrayLike | None = None,
        form: str = "b_spline",
    ) -> None:
        # Searched before np.atleast_1d could turn numpy.ma.masked in a list into NaN.
        reject_masked(knots, "knots")
        reject_masked(boundary_knots, "boundary_knots")
        chosen_form = checked_choice(form, "form", _FORMS)
        with_intercept = checked_flag(intercept, "intercept")
        columns = None if df is None else integral_number(df, "df")
        smallest = 1 + with_intercept
        if columns is not None and columns < smallest:
            raise ValueError(
                f"df must be at least {smallest} with intercept={with_intercept}, "
                f"got {columns}"
            )
        if knots is None:
            raise TypeError(
                f"knots must be a rule, one of {', '.join(map(repr, _RULES))}, or "
                "the interior knots, got None"
            )
        if isinstance(knots, str) and knots not in _RULES:
            raise ValueError(
                f"knots must be a rule, one of {', '.join(map(repr, _RULES))}, or the "
                f"interior knots, got {knots!r}"
            )
        rule = knots if isinstance(knots, str) else None
        observed = None
        if x is not None:
            values = _real_vector(x, "x", missing="NaN")
            if np.isinf(values).any():
                raise ValueError(
                    "x must not contain infinite values; NaN marks a missing value"
                )
            observed = values[~np.isnan(values)]

        if rule == "harrell":
            if boundary_knots is not None:
                raise ValueError(
                    "Harrell's rule places the boundary knots itself, at percentiles "
                    "of x; give the knots and boundary_knots without the rule instead"
                )
            if columns is None or observed is None:
                raise ValueError(
                    "knots='harrell' needs x, whose percentiles place the knots, and "
                    "df, the number of columns, of which it places df + 1 - intercept "
                    "knots"
                )
            count = columns + 1 - with_intercept
            if count < _FEWEST_HARRELL_KNOTS:
                raise ValueError(
                    f"Harrell's rule places at least {_FEWEST_HARRELL_KNOTS} knots, so "
                    f"df must be at least {_FEWEST_HARRELL_KNOTS - 1 + with_intercept} "
                    f"with intercept={with_intercept}, got {columns}"
                )
            placed = _harrell_knots(observed, count)
            interior = placed[1:-1]
            lower, upper = float(placed[0]), float(placed[-1])
        else:
            lower, upper = _boundary_knots(boundary_knots, observed)
            if rule is None:
                interior = _given_knots(knots, columns, with_intercept, lower, upper)
            elif columns is None:
                interior = np.empty(0)
            elif observed is None:
                raise ValueError(
                    f"{_RULES[rule]} places the knots on the values of x; give x"
                )
            else:
                values = observed if rule == "quantiles" else np.unique(observed)
                interior = _quantile_knots(
                    values, columns - 1 - with_intercept, lower, upper
                )
        if chosen_form == "restricted_cubic":
            repeated = interior[1:][np.diff(interior) == 0.0]
            if repeated.size:
                source = "the knots given" if rule is None else _RULES[rule]
                raise ValueError(
                    f"the restricted cubic form needs distinct knots, and {source} "
                    f"repeats {float(repeated[0])!r}; use form='b_spline', which "
                    "accepts repeated knots, or knots='levels', or give distinct knots"
                )
        object.__setattr__(self, "knots", tuple(float(knot) for knot in interior))
        object.__setattr__(self, "boundary_knots", (lower, upper))
        object.__setattr__(self, "intercept", with_intercept)
        object.__setattr__(self, "form", chosen_form)

    @property
    def df(self) -> int:
        """The number of columns, ``len(knots) + 1 + intercept``."""
        return len(self.knots) + 1 + self.intercept

    def matrix(self, x: npt.ArrayLike) -> FloatArray:
        """The columns at x.

        Args:
            x: ``(n,)`` real values; NaN marks a missing value. Values beyond the
                boundary knots lie on the straight lines that continue the curves.

        Returns:
            ``(n, df)`` float64 matrix, with a row of NaN for each NaN in ``x``.

        Raises:
            TypeError: If ``x`` is not real-valued or is a numpy masked array.
            ValueError: If ``x`` is not 1-D or has infinite values.
        """
        points = _real_vector(x, "x", missing="NaN")
        if np.isinf(points).any():
            raise ValueError(
                "x must not contain infinite values; NaN marks a missing value"
            )
        observed = ~np.isnan(points)
        lower, upper = self.boundary_knots
        interior = np.array(self.knots, dtype=np.float64)
        columns = np.full((points.size, self.df), np.nan)
        if self.form == "b_spline":
            columns[observed] = _natural_bspline_columns(
                points[observed], interior, lower, upper, self.intercept
            )
        else:
            all_knots = np.concatenate([[lower], interior, [upper]])
            columns[observed] = _restricted_cubic_columns(
                points[observed], all_knots, self.intercept
            )
        return columns

    def to_dict(self) -> dict[str, Any]:
        """The basis as plain values that JSON stores: ``knots``, ``boundary_knots``
        (lists of floats), ``intercept`` and ``form``.

        ``NaturalSplineBasis(**basis.to_dict())`` rebuilds the basis.
        """
        return {
            "knots": list(self.knots),
            "boundary_knots": list(self.boundary_knots),
            "intercept": self.intercept,
            "form": self.form,
        }


def natural_spline_basis(
    x: npt.ArrayLike,
    df: int | None = None,
    *,
    knots: npt.ArrayLike | None = None,
    intercept: bool = False,
    boundary_knots: npt.ArrayLike | None = None,
) -> FloatArray:
    """Basis of natural cubic splines evaluated at x.

    ``NaturalSplineBasis(x, df, knots="quantiles" if knots is None else knots,
    intercept=intercept, boundary_knots=boundary_knots).matrix(x)``: the B-spline form
    of :class:`NaturalSplineBasis`, whose knots it places and keeps, and which gives
    the columns at other points. The columns span the same space as R's
    ``splines::ns(x, df=df, knots=knots, intercept=intercept,
    Boundary.knots=boundary_knots)``, so a least squares fit on either gives the same
    fitted values; the columns themselves differ from R's. Without an intercept the
    splines are those that vanish at the lower boundary knot, as in R.

    With ``df``, the ``df - 1 - intercept`` interior knots are placed at
    quantiles of x (R's default type 7) with probabilities k / (K + 1),
    k = 1, ..., K, over the values of x within the boundary knots, equal to R's to
    the last bit. Ties in x can make knots coincide, which lowers the smoothness
    there as for repeated knots. A knot that falls on a boundary knot is moved
    inside, as R's ``ns`` does, by one eighth of its distance to the nearest other
    knot (first at the lower boundary, then at the upper one).

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
            to quantiles when ``df`` is given. The other rules of
            :class:`NaturalSplineBasis` are not taken here.
        intercept: Whether the basis spans constant functions.
        boundary_knots: The two knots beyond which the splines are linear.
            Defaults to the range of the observed values of x; given in
            either order.

    Returns:
        ``(n, df)`` matrix, with ``df = len(knots) + 1 + intercept``.

    Raises:
        TypeError: If an input is not real-valued or is a numpy masked array,
            ``knots`` names a rule, ``df`` is not an integer or ``intercept`` is not
            a bool.
        ValueError: If ``x`` is not 1-D or has infinite values, ``df`` is
            too small, knots are not finite or not strictly inside the
            boundary knots, the boundary knots are not two distinct finite
            values, or there are too few values of x to place the knots.
    """
    if isinstance(knots, str):
        raise TypeError(
            f"knots must be the interior knots, got the text {knots!r}; "
            f"NaturalSplineBasis(x, df, knots={knots!r}) places knots by a rule and "
            "keeps them, and its matrix(x) gives the columns"
        )
    basis = NaturalSplineBasis(
        x,
        df,
        knots="quantiles" if knots is None else knots,
        intercept=intercept,
        boundary_knots=boundary_knots,
    )
    return basis.matrix(x)


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
    boundary_knots: npt.ArrayLike | None, observed: FloatArray | None
) -> tuple[float, float]:
    """The boundary knots given, or else the range of the observed values of x."""
    if boundary_knots is None:
        if observed is None:
            raise ValueError(
                "the boundary knots are the range of x by default; give x or "
                "boundary_knots"
            )
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


def _given_knots(
    knots: npt.ArrayLike, df: int | None, intercept: bool, lower: float, upper: float
) -> FloatArray:
    """Interior knots given as values, sorted, checked against df and the boundary
    knots."""
    interior = np.sort(_real_vector(np.atleast_1d(knots), "knots"))
    if not np.isfinite(interior).all():
        raise ValueError("knots must be finite")
    if df is not None and df != interior.size + 1 + intercept:
        raise ValueError(
            f"df={df} disagrees with {interior.size} knots, which give "
            f"{interior.size + 1 + intercept} columns; give one of them"
        )
    outside = (interior <= lower) | (interior >= upper)
    if outside.any():
        raise ValueError(
            f"knots must lie strictly inside the boundary knots ({lower!r}, "
            f"{upper!r}); got {interior[outside].tolist()}"
        )
    return interior


def _type7_quantiles(
    ordered: FloatArray, probabilities: npt.NDArray[np.floating[Any]]
) -> FloatArray:
    """R's type 7 quantiles of sorted values, in R's floating-point operations.

    The quantile at p lies at position i = 1 + (n - 1) p, counted from 1, with whole
    part j and fraction h: it is the j-th value where h is 0 or the j-th and (j + 1)-th
    values are equal, and (1 - h) x_j + h x_(j+1) otherwise, as R's help page writes
    it (the order of the operations was found by probing; the position counted from 0
    differs from R's in the last bit for some n and p).
    """
    position = 1.0 + (ordered.size - 1) * probabilities
    whole = np.floor(position)
    fraction = position - whole
    below = whole.astype(np.int64) - 1
    above = np.minimum(below + 1, ordered.size - 1)
    quantiles = ordered[below].copy()
    blend = (fraction > 0.0) & (ordered[above] != quantiles)
    quantiles[blend] = (1.0 - fraction[blend]) * quantiles[blend] + fraction[
        blend
    ] * ordered[above[blend]]
    return quantiles


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
    # Probabilities k / (count + 1), computed as R computes them (found by probing).
    probabilities = np.arange(1, count + 1) * (1.0 / (count + 1))
    knots = _type7_quantiles(inside, probabilities)
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


def _harrell_knots(observed: FloatArray, count: int) -> FloatArray:
    """All ``count`` knots by Harrell's rule, as Hmisc's rcspline.eval places them.

    The rule of its help page, with the conventions it leaves open found by probing
    Hmisc 5.3-1: with exactly m - 2 knots for m distinct values, the distinct values
    but the smallest and the largest; otherwise R's type 7 quantiles at probabilities
    from ``lo`` to ``1 - lo``, computed as R's ``seq(lo, 1 - lo, length.out =
    count)`` computes them, the outer two replaced by the 5th smallest and 5th
    largest values in samples of fewer than 100, and sorted. Where Hmisc departs
    from this rule, by placing fewer knots, by switching to the method it uses for
    values tied at an extreme, or by dropping a repeated knot, this raises an error.
    """
    ordered = np.sort(observed)
    distinct, counts = np.unique(ordered, return_counts=True)
    m = distinct.size
    if count > m - 2:
        raise ValueError(
            f"Harrell's rule places {count} knots strictly inside the range of x "
            f"only among at least {count + 2} distinct values, and x has {m}; Hmisc "
            f"would place {max(m - 2, 0)} knots, fewer than asked. Lower df, or use "
            "knots='levels' or 'quantiles'"
        )
    if count == m - 2:
        result: FloatArray = distinct[1:-1].astype(np.float64)
        return result
    shares = counts / ordered.size
    if (shares[0] >= _TIED_SHARE or shares[-1] >= _TIED_SHARE) and not (
        shares[1:-1] >= _TIED_SHARE
    ).any():
        end = 0 if shares[0] >= _TIED_SHARE else m - 1
        raise ValueError(
            f"{shares[end]:.1%} of the values of x tie at "
            f"{'the smallest' if end == 0 else 'the largest'} value, "
            f"{float(distinct[end])!r}, and no value between the extremes ties as "
            f"often; Hmisc then places the knots by a method that its help page "
            "describes only in outline. Use knots='levels' or 'quantiles', or give "
            "the knots"
        )
    if count == _FEWEST_HARRELL_KNOTS:
        outer = 0.1
    elif count <= 6:
        outer = 0.05
    else:
        outer = 0.025
    probabilities = outer + np.arange(count) * (((1.0 - outer) - outer) / (count - 1))
    knots = _type7_quantiles(ordered, probabilities)
    if ordered.size < _SMALL_SAMPLE:
        knots[0] = ordered[_FIFTH - 1]
        knots[-1] = ordered[ordered.size - _FIFTH]
    knots.sort()
    repeated = knots[1:][np.diff(knots) == 0.0]
    if repeated.size:
        raise ValueError(
            f"Harrell's rule places two knots at {float(repeated[0])!r}, where Hmisc "
            "would drop one and place fewer knots than asked. Lower df, or use "
            "knots='levels', or give the knots"
        )
    return knots


def _natural_bspline_columns(
    points: FloatArray,
    interior: FloatArray,
    lower: float,
    upper: float,
    intercept: bool,
) -> FloatArray:
    """The B-spline form: natural cubic splines at points, linear beyond the
    boundary knots, as the module docstring describes them."""
    knot_sequence = np.concatenate(
        [np.full(_CUBIC_ORDER, lower), interior, np.full(_CUBIC_ORDER, upper)]
    )
    bounds = np.array([lower, upper])
    splines = _bspline_design(points, knot_sequence, lower, upper)
    constraints = _bspline_derivatives(bounds, knot_sequence, 2)
    if not intercept:
        # Every B-spline but the first vanishes at the lower boundary knot.
        splines = splines[:, 1:]
        constraints = constraints[:, 1:]
    # The basis of the natural splines is the B-spline basis times an
    # orthonormal basis of the null space of the two boundary conditions.
    scale = np.abs(constraints).max(axis=1, keepdims=True)
    q, _ = np.linalg.qr(
        (constraints / np.where(scale > 0.0, scale, 1.0)).T, mode="complete"
    )
    columns: FloatArray = splines @ q[:, 2:]
    return columns


def _restricted_cubic_columns(
    points: FloatArray, knots: FloatArray, intercept: bool
) -> FloatArray:
    """The restricted cubic form at points, for all the knots t_1 < ... < t_k.

    x - t_1, and the terms of the module docstring computed from truncated cubes, as
    Hmisc's rcspline.eval gives them with its default normalization by (t_k - t_1)^2;
    a column of ones first with an intercept.
    """
    first, second_last, last = knots[0], knots[-2], knots[-1]
    spread = (last - first) ** 2
    gap = last - second_last
    columns = [np.ones(points.size)] if intercept else []
    columns.append(points - first)
    for knot in knots[:-2]:
        term = (
            _truncated_cube(points - knot)
            - _truncated_cube(points - second_last) * (last - knot) / gap
            + _truncated_cube(points - last) * (second_last - knot) / gap
        )
        columns.append(term / spread)
    return np.column_stack(columns)


def _truncated_cube(u: FloatArray) -> FloatArray:
    """max(u, 0)^3."""
    cube: FloatArray = np.maximum(u, 0.0) ** 3
    return cube


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
