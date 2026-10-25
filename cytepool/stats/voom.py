"""Precision weights for log counts from their mean-variance trend, as in limma's voom.

For counts r_gi of feature g in sample i and effective library sizes R_i (library
sizes times normalization factors), voom (Law, Chen, Shi and Smyth 2014) works with
the log2 counts per million

    y_gi = log2((r_gi + 0.5) / (R_i + 1) * 1e6).

The design is fitted to every feature by ordinary least squares, which gives fitted
values mu_gi and residual standard deviations s_g. The square roots s_g^(1/2) are
smoothed by LOWESS against the average log2 count of each feature,

    x_g = mean_i(y_gi) + mean_i(log2(R_i + 1)) - log2(1e6),

and the smoothed curve lo() is interpolated linearly between its points and held
constant beyond them. Each observation is then weighted by the inverse of its
predicted variance, w_gi = lo(lambda_gi)^(-4), where

    lambda_gi = log2(2^mu_gi (R_i + 1) / 1e6)

is its fitted log2 count. A weighted least squares fit of the log2 counts per million
with these weights, followed by empirical Bayes moderation, is limma-voom.

Results match limma 3.62.1, called with ``adaptive.span = TRUE`` where the span is
chosen. Conventions that the paper leaves open were settled from limma's
documentation (its help page and NEWS) or by running limma as a black box; comments
mark the latter "found by probing":

- Features with no counts in any sample are left out of the trend (found by probing);
  their weights are read from it at their fitted log2 counts, as for any other
  feature.
- The LOWESS is R's ``lowess`` with its defaults other than the span: three
  robustness iterations, and interpolation within 1% of the range of x (found by
  probing).
- By default the span is chosen from the number of features n by limma's rule,
  :func:`~cytepool.stats.smoothing.choose_lowess_span`: 1 up to 50 features, and
  0.3 + 0.7 (50 / n)^(1/3) beyond. limma's voom chooses it so with
  ``adaptive.span = TRUE``, an option since limma 3.60 and the default since 3.66
  (limma's NEWS); its default span was 0.5 before. n counts every feature, also those
  without counts that the trend leaves out, as limma's ``nrow(counts)`` does (its help
  page; confirmed by probing). A span given as a number is used as it is, as with
  ``adaptive.span = FALSE``.
- The interpolation averages the curve over tied values of x (limma's NEWS); LOWESS
  gives tied points the same value, so each distinct x is one point of the curve.
  The x are computed as R computes them, to the last bit, so that their ties are
  limma's: LOWESS treats tied points differently from points a rounding error apart.
- A design that leaves no residual degrees of freedom gives every observation weight
  one, with a warning (limma's NEWS).
- There must be at least two features, as limma requires, and at least two with counts
  to fit a trend.

Deliberate deviations, where limma returns values without meaning or fails:

- A sample whose library size is zero is an error. limma gives it the log2 counts per
  million of a count of 0.5 in a library of one read.
- A trend that is not positive where weights are read from it is an error. limma takes
  its inverse fourth power anyway, which is positive for a negative trend.
- When every feature in the trend has the same x, the curve is constant; limma fails.
- A design that is not of full column rank is rejected by
  :func:`~cytepool.stats.linear_models.fit_linear_model`; limma drops the aliased
  columns.
- ``span`` must lie in (0, 1], as limma documents; limma accepts larger values.

References:
    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear
    model analysis tools for RNA-seq read counts. Genome Biology 15, R29.
"""

import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._masked_arrays import reject_masked
from cytepool._summation import accurate_mean
from cytepool.normalization import _as_counts, _effective_library_sizes
from cytepool.stats.linear_models import _design_matrix, fit_linear_model
from cytepool.stats.smoothing import choose_lowess_span, lowess

__all__ = ["VoomResult", "voom"]

FloatArray = npt.NDArray[np.float64]

# Robustness iterations of the LOWESS trend: R's default, which limma keeps (found
# by probing).
_TREND_ITERATIONS = 3
_LOG2_MILLION = math.log2(1e6)


@dataclass(frozen=True, eq=False, repr=False)
class VoomResult:
    """Log2 counts per million and their precision weights, from :func:`voom`.

    Fit a linear model with ``fit_linear_model(result.log_cpm, design,
    weights=result.weights)``. Arrays with a sample axis have one row per sample;
    the others have one value per feature, in the order of the columns of the
    counts. The arrays are read-only views; the constructor does not copy them.

    Attributes:
        log_cpm: ``(n_samples, n_features)`` log2 counts per million,
            ``log2((r + 0.5) / (R + 1) * 1e6)`` with R the effective library size.
        weights: ``(n_samples, n_features)`` precision weights: the inverse of each
            observation's variance predicted by the trend.
        effective_library_sizes: ``(n_samples,)`` library sizes times
            normalization factors. Pass them back as ``library_sizes`` without
            normalization factors to repeat the computation.
        mean_log_count: ``(n_features,)`` average log2 count of each feature, the
            x of the mean-variance trend. NaN for features with no counts, which
            are left out of the trend.
        sqrt_sd: ``(n_features,)`` square root of each feature's residual standard
            deviation, the y of the trend. NaN where ``mean_log_count`` is, and for
            every feature when the design leaves no residual degrees of freedom.
        sqrt_sd_trend: ``(n_features,)`` the LOWESS trend of ``sqrt_sd`` at
            ``mean_log_count``; NaN where ``sqrt_sd`` is. Sort by
            ``mean_log_count`` to draw it as a curve.
        span: The span of the LOWESS trend, the fraction of the features in
            the trend that each local fit uses, as chosen or given. NaN when
            the design leaves no residual degrees of freedom, so that no trend
            is fitted.

    Raises:
        TypeError: If an array is a numpy masked array, or ``span`` is not a
            real number.
        ValueError: If the shapes of the arrays disagree, or ``span`` is
            neither in (0, 1] nor NaN.
    """

    log_cpm: FloatArray
    weights: FloatArray
    effective_library_sizes: FloatArray
    mean_log_count: FloatArray
    sqrt_sd: FloatArray
    sqrt_sd_trend: FloatArray
    span: float

    def __post_init__(self) -> None:
        for name in ("log_cpm", "weights", "effective_library_sizes"):
            reject_masked(getattr(self, name), name)
        # NaN marks the features that the trend leaves out.
        for name in ("mean_log_count", "sqrt_sd", "sqrt_sd_trend"):
            reject_masked(getattr(self, name), name, missing="NaN")
        span = self.span
        if isinstance(span, (bool, np.bool_)) or not isinstance(span, numbers.Real):
            raise TypeError(f"span must be a real number, got {type(span).__name__}")
        if not (math.isnan(span) or 0.0 < span <= 1.0):
            raise ValueError(
                f"span must be in (0, 1], or NaN when no trend is fitted, got {span}"
            )
        object.__setattr__(self, "span", float(span))
        log_cpm = np.asarray(self.log_cpm, dtype=np.float64)
        if log_cpm.ndim != 2:
            raise ValueError(
                f"log_cpm must be a 2-D array (samples x features), got shape "
                f"{log_cpm.shape}"
            )
        n_samples, n_features = log_cpm.shape
        shapes = {
            "log_cpm": log_cpm.shape,
            "weights": log_cpm.shape,
            "effective_library_sizes": (n_samples,),
            "mean_log_count": (n_features,),
            "sqrt_sd": (n_features,),
            "sqrt_sd_trend": (n_features,),
        }
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            view = array.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    def __repr__(self) -> str:
        n_samples, n_features = self.log_cpm.shape
        in_trend = int(np.count_nonzero(~np.isnan(self.sqrt_sd)))
        return (
            f"VoomResult(n_samples={n_samples}, n_features={n_features}, "
            f"features_in_trend={in_trend}, span={self.span:.3g})"
        )


def voom(
    counts: npt.ArrayLike,
    design: Any,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    span: float | None = None,
) -> VoomResult:
    """Log2 counts per million with precision weights from the mean-variance trend.

    limma's ``voom`` (Law et al. 2014); the module docstring gives the method. The
    design is fitted by :func:`~cytepool.stats.linear_models.fit_linear_model`,
    which checks it. Filter out features with very low counts first, for example
    with :func:`~cytepool.normalization.filter_by_expression`, as limma advises.

    Continue with ``fit_linear_model(result.log_cpm, design,
    weights=result.weights)`` and
    :func:`~cytepool.stats.empirical_bayes.moderated_statistics`. With observation
    weights, limma's ``contrasts.fit`` approximates the standard errors of contrasts
    of correlated coefficients; ``contrasts_fit(method="limma")`` reproduces it, and
    the default computes them exactly.

    Args:
        counts: ``(n_samples, n_features)`` counts: finite and non-negative, integer
            or not (estimated counts are fine). Do not pass values already scaled
            by library size, such as CPM or TPM: voom needs the library sizes.
        design: ``(n_samples, n_coefficients)`` design matrix of full column rank,
            or anything else that ``fit_linear_model`` accepts, such as a
            :class:`~cytepool.design.Design`.
        library_sizes: Library size of each sample; defaults to the total counts.
            Must be positive.
        norm_factors: Normalization factor of each sample, such as TMM factors from
            :func:`~cytepool.normalization.tmm_factors`, multiplied into the
            library sizes; defaults to ones. Must be positive.
        span: Fraction of the features in the trend (those with counts) that each
            local fit of the LOWESS uses, in (0, 1]. By default it is chosen
            from the number of features, those without counts included, by
            :func:`~cytepool.stats.smoothing.choose_lowess_span`: 1 up to 50
            features, 0.65 at 400 and 0.395 at 20,000, as limma's
            ``voom`` chooses it by default since limma 3.66
            (``adaptive.span = TRUE``). A number fixes it, as
            ``adaptive.span = FALSE, span = <number>`` does in limma, whose
            default before 3.66 was 0.5.

    Returns:
        The log2 counts per million, their weights (positive and finite), the
        effective library sizes, and the mean-variance trend with its span.

    Warns:
        RuntimeWarning: If the design leaves no residual degrees of freedom, so
            that no trend can be fitted; every weight is then one.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``counts`` is not a 2-D array of finite, non-negative values
            with at least two features, the design does not have one row per
            sample or is not of full column rank, a library size or normalization
            factor is not positive, ``span`` is not in (0, 1], fewer than two
            features have counts (with residual degrees of freedom), or the trend
            is not positive where the weights are read from it.
    """
    y = _as_counts(counts)
    n_samples, n_features = y.shape
    if n_features < 2:
        raise ValueError(
            f"a mean-variance trend needs at least two features, got {n_features}"
        )
    n_design_rows = _design_matrix(design, None)[0].shape[0]
    if n_design_rows != n_samples:
        raise ValueError(
            f"counts have {n_samples} rows (samples) but the design has "
            f"{n_design_rows}; both need one row per sample"
        )
    # A copy, so that the result does not change with the caller's array.
    sizes = np.array(_effective_library_sizes(y, library_sizes, norm_factors))
    fraction = _trend_span(span, n_features)

    # The operations of limma, which give its values to the last bit (found by
    # probing).
    log_cpm = np.log2((y + 0.5) / (sizes + 1.0)[:, np.newaxis] * 1e6)
    fit = fit_linear_model(log_cpm, design)
    # fit.mean_expression is R's rowMeans to the last bit on log2 CPM, and the sum
    # is formed in limma's order, so that x and its ties are limma's (found by
    # probing).
    log_size = np.log2(sizes + 1.0)
    mean_log_count = (fit.mean_expression + accurate_mean(log_size)) - _LOG2_MILLION
    # Features without counts are left out of the trend (found by probing).
    in_trend = (y > 0.0).any(axis=0)
    mean_log_count[~in_trend] = np.nan
    sqrt_sd = np.full(n_features, np.nan)
    sqrt_sd_trend = np.full(n_features, np.nan)

    if fit.df_residual[0] == 0.0:
        warnings.warn(
            "the design leaves no residual degrees of freedom (no replication), so "
            "no mean-variance trend can be fitted; every weight is set to 1",
            RuntimeWarning,
            stacklevel=2,
        )
        # limma returns no span then (found by probing).
        return VoomResult(
            log_cpm=log_cpm,
            weights=np.ones_like(log_cpm),
            effective_library_sizes=sizes,
            mean_log_count=mean_log_count,
            sqrt_sd=sqrt_sd,
            sqrt_sd_trend=sqrt_sd_trend,
            span=math.nan,
        )
    n_in_trend = int(np.count_nonzero(in_trend))
    if n_in_trend < 2:
        raise ValueError(
            "a mean-variance trend needs at least two features with counts, got "
            f"{n_in_trend}"
        )

    trend_x = mean_log_count[in_trend]
    trend_y = np.sqrt(fit.sigma[in_trend])
    # R's default delta, 1% of the range of x (found by probing).
    trend = lowess(trend_x, trend_y, span=fraction, iterations=_TREND_ITERATIONS)
    sqrt_sd[in_trend] = trend_y
    sqrt_sd_trend[in_trend] = trend
    # limma averages the curve over tied x (its NEWS); tied points share their
    # fitted value, so each distinct x is one knot with that value.
    knots_x, first = np.unique(trend_x, return_index=True)
    knots_y = trend[first]

    fitted = fit.design @ fit.coefficients.T
    # The fitted log2 counts, formed on the count scale as limma forms them (found
    # by probing).
    fitted_log_count = np.log2(
        np.power(2.0, fitted) * (sizes + 1.0)[:, np.newaxis] * 1e-6
    )
    predicted = _interpolate(knots_x, knots_y, fitted_log_count)
    with np.errstate(divide="ignore", over="ignore"):
        weights = 1.0 / np.power(predicted, 4.0)
    if not np.all((predicted > 0.0) & np.isfinite(weights)):
        raise ValueError(
            "the mean-variance trend falls to "
            f"{float(np.min(predicted)):.3g} where weights are read from it, but "
            "the weights, its inverse fourth power, need it positive and large "
            "enough to invert. Few features, or features that the design fits "
            "exactly, can cause this: filter out features with low counts, or use "
            "more features"
        )
    return VoomResult(
        log_cpm=log_cpm,
        weights=weights,
        effective_library_sizes=sizes,
        mean_log_count=mean_log_count,
        sqrt_sd=sqrt_sd,
        sqrt_sd_trend=sqrt_sd_trend,
        span=fraction,
    )


def _interpolate(
    knots_x: FloatArray, knots_y: FloatArray, at: FloatArray
) -> FloatArray:
    """Linear interpolation through sorted distinct knots, constant beyond them.

    The value between knots j and j + 1 is y_j + (y_j+1 - y_j) ((v - x_j) /
    (x_j+1 - x_j)), the operations of R's ``approx`` (found by probing); at a knot
    it is the knot's value.
    """
    if knots_x.size == 1:
        return np.full(at.shape, knots_y[0])
    j = np.searchsorted(knots_x, at, side="right") - 1
    np.clip(j, 0, knots_x.size - 2, out=j)
    x0, x1 = knots_x[j], knots_x[j + 1]
    y0, y1 = knots_y[j], knots_y[j + 1]
    values: FloatArray = y0 + (y1 - y0) * ((at - x0) / (x1 - x0))
    values[at <= knots_x[0]] = knots_y[0]
    values[at >= knots_x[-1]] = knots_y[-1]
    return values


def _trend_span(span: Any, n_features: int) -> float:
    """The span of the trend: as given, or chosen from the number of features."""
    if span is None:
        # limma's voom passes nrow(counts), which counts the features that the
        # trend leaves out too (its help page; confirmed by probing).
        return choose_lowess_span(n_features)
    if isinstance(span, (bool, np.bool_)) or not isinstance(span, numbers.Real):
        raise TypeError(
            "span must be a real number, or None to choose it from the number of "
            f"features, got {type(span).__name__}"
        )
    fraction = float(span)
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            f"span must be a fraction of the features in (0, 1], got {fraction}"
        )
    return fraction
