"""The mean-variance trend of log counts and the precision weights read from it.

voom (Law, Chen, Shi and Smyth 2014) smooths the square roots of the residual standard
deviations of the features' log2 counts per million against their average log2 counts
by LOWESS, and weights every observation by the inverse fourth power of the curve at
its fitted log2 count. These are the parts of it that voom, voom with sample quality
weights, dream's weights over mixed-model fits (Hoffman and Roussos 2021) and the
workflows on cubes share:

- ``voom_inputs``: the counts, effective library sizes and span of a trend, checked;
  ``trend_span`` reads the span, or chooses it from the number of features.
- ``voom_log_cpm``: voom's log2 counts per million, log2((r + 0.5) / (R + 1) * 1e6).
- ``voom_mean_log_count``: the x of the trend, each feature's average log2 count.
- ``trend_weights``: the trend of the features given and every observation's weight,
  as a ``Trend``.
- ``scaled_prior_weights``: prior weights with each feature's scaled to mean one, as
  dream takes them.

Arrays are samples x features. Each function states the order of its operations where
that order decides the last bit of a value that limma reproduces.

References:
    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687
"""

import math
import numbers
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from cytepool._counts import as_counts, effective_library_sizes
from cytepool._masked_arrays import real_array, unmasked_array
from cytepool._summation import accurate_mean
from cytepool.design._design_matrices import design_matrix, refuse_random_effects
from cytepool.stats.smoothing import choose_lowess_span, lowess

FloatArray = npt.NDArray[np.float64]

# Robustness iterations of the LOWESS trend: R's default, which limma keeps (found
# by probing).
_TREND_ITERATIONS = 3
_LOG2_MILLION = math.log2(1e6)


@dataclass(frozen=True)
class Trend:
    """voom's quantities before prior weights are multiplied in."""

    log_cpm: FloatArray
    weights: FloatArray  # the trend's: NaN only where a prior weight is zero
    mean_log_count: FloatArray
    sqrt_sd: FloatArray
    sqrt_sd_trend: FloatArray
    span: float


def voom_inputs(
    counts: npt.ArrayLike,
    design: Any,
    library_sizes: npt.ArrayLike | None,
    norm_factors: npt.ArrayLike | None,
    span: Any,
    function: str,
) -> tuple[FloatArray, FloatArray, float]:
    """The counts, effective library sizes and span, checked for ``function``."""
    y = as_counts(counts)
    n_samples, n_features = y.shape
    if n_features < 2:
        raise ValueError(
            f"a mean-variance trend needs at least two features, got {n_features}"
        )
    n_design_rows = design_matrix(design, None)[0].shape[0]
    refuse_random_effects(design, function)
    if n_design_rows != n_samples:
        raise ValueError(
            f"counts have {n_samples} rows (samples) but the design has "
            f"{n_design_rows}; both need one row per sample"
        )
    # A copy, so that the result does not change with the caller's array.
    sizes = np.array(effective_library_sizes(y, library_sizes, norm_factors))
    return y, sizes, trend_span(span, n_features)


def trend_span(span: Any, n_features: int) -> float:
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


def voom_log_cpm(y: FloatArray, sizes: FloatArray) -> FloatArray:
    """voom's log2 counts per million, log2((r + 0.5) / (R + 1) * 1e6)."""
    # The operations of limma, which give its values to the last bit (found by
    # probing).
    log_cpm: FloatArray = np.log2((y + 0.5) / (sizes + 1.0)[:, np.newaxis] * 1e6)
    return log_cpm


def voom_mean_log_count(
    mean_expression: FloatArray, sizes: FloatArray, y: FloatArray
) -> FloatArray:
    """The x of the trend: average log2 counts, NaN for features without counts.

    ``mean_expression`` is R's rowMeans to the last bit on log2 CPM, and the sum is
    formed in limma's order, so that x and its ties are limma's (found by probing).
    """
    log_size = np.log2(sizes + 1.0)
    mean_log_count: FloatArray = (
        mean_expression + accurate_mean(log_size)
    ) - _LOG2_MILLION
    # Features without counts are left out of the trend (found by probing).
    mean_log_count[~(y > 0.0).any(axis=0)] = np.nan
    return mean_log_count


def trend_weights(
    log_cpm: FloatArray,
    mean_log_count: FloatArray,
    sigma: FloatArray,
    in_trend: npt.NDArray[np.bool_],
    fitted: FloatArray,
    sizes: FloatArray,
    fraction: float,
    weighted: npt.NDArray[np.bool_],
    condition: str,
) -> Trend:
    """The mean-variance trend of the features ``in_trend`` and its weights.

    The square roots of the residual standard deviations ``sigma`` are smoothed by
    LOWESS against ``mean_log_count``, and every observation is weighted by the
    inverse fourth power of the trend at its fitted log2 count, from ``fitted``.

    Args:
        log_cpm: The log2 counts per million.
        mean_log_count: The x of the trend.
        sigma: The residual standard deviation of each feature.
        in_trend: The features whose ``sigma`` enter the trend.
        fitted: ``(n_samples, n_features)`` fitted log2 CPM.
        sizes: The effective library sizes.
        fraction: The span.
        weighted: The observations whose weights must be usable.
        condition: What the features in the trend need besides counts, for the
            message when there are fewer than two.

    Raises:
        ValueError: If fewer than two features are in the trend, or the trend is not
            positive where weights are read from it.
    """
    n_features = log_cpm.shape[1]
    sqrt_sd = np.full(n_features, np.nan)
    sqrt_sd_trend = np.full(n_features, np.nan)
    n_in_trend = int(np.count_nonzero(in_trend))
    if n_in_trend < 2:
        raise ValueError(
            f"a mean-variance trend needs at least two features with counts"
            f"{condition}, got {n_in_trend}"
        )
    trend_x = mean_log_count[in_trend]
    trend_y = np.sqrt(sigma[in_trend])
    # R's default delta, 1% of the range of x (found by probing).
    trend = lowess(trend_x, trend_y, span=fraction, iterations=_TREND_ITERATIONS)
    sqrt_sd[in_trend] = trend_y
    sqrt_sd_trend[in_trend] = trend
    # limma averages the curve over tied x (its NEWS); tied points share their
    # fitted value, so each distinct x is one knot with that value.
    knots_x, first = np.unique(trend_x, return_index=True)
    knots_y = trend[first]

    # The fitted log2 counts, formed on the count scale as limma forms them (found
    # by probing).
    fitted_log_count = np.log2(
        np.power(2.0, fitted) * (sizes + 1.0)[:, np.newaxis] * 1e-6
    )
    predicted = _interpolate(knots_x, knots_y, fitted_log_count)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        weights = 1.0 / np.power(predicted, 4.0)
    usable = (predicted > 0.0) & np.isfinite(weights)
    if not np.all(usable[weighted]):
        lowest = float(np.nanmin(np.where(weighted, predicted, np.inf)))
        raise ValueError(
            "the mean-variance trend falls to "
            f"{lowest:.3g} where weights are read from it, but "
            "the weights, its inverse fourth power, need it positive and large "
            "enough to invert. Few features, or features that the design fits "
            "exactly, can cause this: filter out features with low counts, or use "
            "more features"
        )
    return Trend(log_cpm, weights, mean_log_count, sqrt_sd, sqrt_sd_trend, fraction)


def _interpolate(
    knots_x: FloatArray, knots_y: FloatArray, at: FloatArray
) -> FloatArray:
    """Linear interpolation through sorted distinct knots, constant beyond them.

    The value between knots j and j + 1 is y_j + (y_j+1 - y_j) ((v - x_j) /
    (x_j+1 - x_j)), the operations of R's ``approx`` (found by probing); at a knot
    it is the knot's value. NaN where ``at`` is NaN.
    """
    if knots_x.size == 1:
        constant: FloatArray = np.where(np.isnan(at), np.nan, knots_y[0])
        return constant
    j = np.searchsorted(knots_x, at, side="right") - 1
    np.clip(j, 0, knots_x.size - 2, out=j)
    x0, x1 = knots_x[j], knots_x[j + 1]
    y0, y1 = knots_y[j], knots_y[j + 1]
    values: FloatArray = y0 + (y1 - y0) * ((at - x0) / (x1 - x0))
    values[at <= knots_x[0]] = knots_y[0]
    values[at >= knots_x[-1]] = knots_y[-1]
    return values


def scaled_prior_weights(
    weights: npt.ArrayLike | None, shape: tuple[int, int]
) -> FloatArray | None:
    """Prior weights with each feature's divided by their mean, as variancePartition.

    Sample weights become a vector of mean one, observation weights a matrix whose
    columns (features) have mean one (found by probing variancePartition).
    """
    if weights is None:
        return None
    w = real_array(unmasked_array(weights, "weights"), "weights")
    if w.shape not in ((shape[0],), shape):
        raise ValueError(
            f"weights must have shape ({shape[0]},) for sample weights or {shape} "
            f"for observation weights, got {w.shape}"
        )
    with np.errstate(invalid="ignore"):
        invalid = int(np.count_nonzero(~(np.isfinite(w) & (w > 0.0))))
    if invalid:
        raise ValueError(
            f"prior weights must be finite and positive; {invalid} are zero, negative "
            "or not finite. The mixed model of a feature is fitted to all its samples"
        )
    scaled: FloatArray = w / np.mean(w, axis=0)
    return scaled
