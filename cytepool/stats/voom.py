"""Precision weights for log counts from their mean-variance trend, as in limma's voom.

:func:`voom` gives the log2 counts per million of a samples x features matrix of
counts with a precision weight for each observation, from the trend of their variance
against their mean (Law, Chen, Shi and Smyth 2014), in a :class:`VoomResult`.
:func:`voom_with_quality_weights` combines those weights with sample quality weights
(Liu et al. 2015), as limma's ``voomWithQualityWeights`` does. A weighted least squares
fit of the log2 counts per million with these weights, followed by empirical Bayes
moderation, is limma-voom; https://docs.nygen.io/CytePool/statistics/limma-voom#voom
gives the method.

Matching limma:
    The results match limma 3.62.1, called with ``adaptive.span = TRUE`` where the
    span is chosen. limma's documentation (its help page and NEWS) gives some of the
    conventions below, and running limma on chosen inputs gave the others.

    - Features with no counts in any sample are left out of the trend; their weights
      are read from it at their fitted log2 counts, as for any other feature.
    - The LOWESS is R's ``lowess`` with its defaults other than the span: three
      robustness iterations, and interpolation within 1% of the range of x.
    - The span that :func:`voom` chooses by default follows the number of features,
      which counts every feature, also those without counts that the trend leaves
      out, as limma's ``nrow(counts)`` does. limma's ``adaptive.span = TRUE``, which
      chooses it, is an option since limma 3.60 and the default since 3.66 (limma's
      NEWS); a span given as a number is used as it is, as with
      ``adaptive.span = FALSE``.
    - The trend is interpolated linearly between its points and held constant beyond
      them. The interpolation averages the curve over tied values of x (limma's
      NEWS); LOWESS gives tied points the same value, so each distinct x is one point
      of the curve. The x are computed as R computes them, to the last bit, so that
      their ties are limma's: LOWESS treats tied points differently from points a
      rounding error apart.
    - x averages every log2 count per million of a feature, those of zero prior
      weight included, as limma's ``Amean`` does.
    - A design that leaves no residual degrees of freedom gives every observation
      weight one, times its prior weight, with a warning (limma's NEWS).
    - There must be at least two features, as limma requires, and at least two with
      counts (and residual degrees of freedom under the prior weights) to fit a
      trend.

Differences from limma:
    - A sample whose library size is zero is an error. limma gives it the log2 counts
      per million of a count of 0.5 in a library of one read.
    - A trend that is not positive where weights are read from it is an error. limma
      takes its inverse fourth power anyway, which is positive for a negative trend.
    - When every feature in the trend has the same x, the curve is constant; limma
      fails.
    - A design that is not of full column rank is rejected by
      :func:`~cytepool.stats.linear_models.fit_linear_model`; limma drops the aliased
      columns.
    - ``span`` must lie in (0, 1], as limma documents; limma accepts larger values.
    - A feature whose prior weights leave no residual degrees of freedom is left out of
      the trend. limma passes its missing standard deviation to R's ``lowess``, which
      then returns a curve without meaning for every feature.
    - Where zero prior weights leave coefficients of a feature that its other samples
      cannot estimate, its fitted values are those of the least squares fit over the
      samples of positive weight, and its observations of zero prior weight get weight
      zero. limma returns missing weights for every observation of such a feature.

References:
    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Liu R, Holik AZ, Su S, Jansz N, Chen K, Leong HS, Blewitt ME, Asselin-Labat ML,
    Smyth GK, Ritchie ME (2015). Why weight? Modelling sample and observational level
    variability improves power in RNA-seq analyses. Nucleic Acids Research 43, e97.
    doi:10.1093/nar/gkv412
"""

import math
import numbers
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cytepool._masked_arrays import reject_masked
from cytepool.design._design_matrices import design_matrix
from cytepool.stats._mean_variance_trend import (
    Trend,
    trend_weights,
    voom_inputs,
    voom_log_cpm,
    voom_mean_log_count,
)
from cytepool.stats.linear_models import LinearModelFit, fit_linear_model
from cytepool.stats.sample_weights import (
    _estimate,
    _prior_weights,
    _too_few_df_message,
    _variance_model,
    _VarianceModel,
    _weights_of,
)

__all__ = ["VoomResult", "voom", "voom_with_quality_weights"]

FloatArray = npt.NDArray[np.float64]

_NO_REPLICATION = (
    "the design leaves no residual degrees of freedom (no replication), so no "
    "mean-variance trend can be fitted; every weight is set to 1"
)
_NO_RESIDUAL_DF = (
    "the design, without the observations of zero prior weight, leaves no residual "
    "degrees of freedom, so no mean-variance trend can be fitted; every weight is "
    "set to its prior weight"
)


@dataclass(frozen=True, eq=False, repr=False)
class VoomResult:
    """Log2 counts per million and their precision weights, from :func:`voom`.

    Fit a linear model with ``fit_linear_model(result.log2_cpm, design,
    weights=result.weights)``, or, for the weights of
    :func:`~cytepool.stats.dream.voom_with_dream_weights`, a mixed model with
    ``fit_mixed_model``. Arrays with a sample axis have one row per sample;
    the others have one value per feature, in the order of the columns of the
    counts. The arrays are read-only views; the constructor does not copy them.

    Attributes:
        log2_cpm: ``(n_samples, n_features)`` log2 counts per million,
            ``log2((r + 0.5) / (R + 1) * 1e6)`` with R the effective library size.
        weights: ``(n_samples, n_features)`` precision weights: the inverse of each
            observation's variance predicted by the trend, times the prior weights
            given to :func:`voom` (each feature's divided by their mean, for
            ``voom_with_dream_weights``) or the ``sample_weights``.
        effective_library_sizes: ``(n_samples,)`` library sizes times
            normalization factors. Pass them back as ``library_sizes`` without
            normalization factors to repeat the computation.
        mean_log_count: ``(n_features,)`` average log2 count of each feature, the
            x of the mean-variance trend. NaN for features with no counts, which
            are left out of the trend.
        sqrt_sd: ``(n_features,)`` square root of each feature's residual standard
            deviation, the y of the trend. NaN for the features left out of the
            trend: those with no counts, those whose prior weights leave no residual
            degrees of freedom, and every feature when the design leaves none.
        sqrt_sd_trend: ``(n_features,)`` the LOWESS trend of ``sqrt_sd`` at
            ``mean_log_count``; NaN where ``sqrt_sd`` is. Sort by
            ``mean_log_count`` to draw it as a curve.
        span: The span of the LOWESS trend, the fraction of the features in
            the trend that each local fit uses, as chosen or given. NaN when
            the design leaves no residual degrees of freedom, so that no trend
            is fitted.
        sample_weights: ``(n_samples,)`` sample quality weights that ``weights``
            include, from :func:`voom_with_quality_weights`; None otherwise.

    Raises:
        TypeError: If an array is a numpy masked array, or ``span`` is not a
            real number.
        ValueError: If the shapes of the arrays disagree, ``span`` is
            neither in (0, 1] nor NaN, or a sample weight is not positive and
            finite.
    """

    log2_cpm: FloatArray
    weights: FloatArray
    effective_library_sizes: FloatArray
    mean_log_count: FloatArray
    sqrt_sd: FloatArray
    sqrt_sd_trend: FloatArray
    span: float
    sample_weights: FloatArray | None = None

    def __post_init__(self) -> None:
        for name in (
            "log2_cpm",
            "weights",
            "effective_library_sizes",
            "sample_weights",
        ):
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
        log2_cpm = np.asarray(self.log2_cpm, dtype=np.float64)
        if log2_cpm.ndim != 2:
            raise ValueError(
                f"log2_cpm must be a 2-D array (samples x features), got shape "
                f"{log2_cpm.shape}"
            )
        n_samples, n_features = log2_cpm.shape
        shapes = {
            "log2_cpm": log2_cpm.shape,
            "weights": log2_cpm.shape,
            "effective_library_sizes": (n_samples,),
            "mean_log_count": (n_features,),
            "sqrt_sd": (n_features,),
            "sqrt_sd_trend": (n_features,),
        }
        if self.sample_weights is not None:
            shapes["sample_weights"] = (n_samples,)
        for name, shape in shapes.items():
            array = np.asarray(getattr(self, name), dtype=np.float64)
            if array.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
            if name == "sample_weights" and not np.all(
                np.isfinite(array) & (array > 0.0)
            ):
                raise ValueError("sample_weights must be positive and finite")
            view = array.view()
            view.flags.writeable = False
            object.__setattr__(self, name, view)

    def __repr__(self) -> str:
        n_samples, n_features = self.log2_cpm.shape
        in_trend = int(np.count_nonzero(~np.isnan(self.sqrt_sd)))
        quality = "" if self.sample_weights is None else ", with sample weights"
        return (
            f"VoomResult(n_samples={n_samples}, n_features={n_features}, "
            f"features_in_trend={in_trend}, span={self.span:.3g}{quality})"
        )


def voom(
    counts: npt.ArrayLike,
    design: Any,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    span: float | None = None,
    weights: npt.ArrayLike | None = None,
) -> VoomResult:
    """Log2 counts per million with precision weights from the mean-variance trend.

    limma's ``voom`` (Law et al. 2014). The design is fitted to every feature by
    :func:`~cytepool.stats.linear_models.fit_linear_model`, which checks it; the
    square roots of the residual standard deviations are smoothed by LOWESS against
    each feature's average log2 count; and each observation is weighted by the
    inverse of the variance that this trend predicts at its fitted log2 count, times
    its prior weight. https://docs.nygen.io/CytePool/statistics/limma-voom#voom gives
    the equations, and the module documentation the conventions that match limma.
    Filter out features with very low counts first, for example with
    :func:`~cytepool.normalization.filter_by_expression`, as limma advises.

    Continue with ``fit_linear_model(result.log2_cpm, design,
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
            :class:`~cytepool.Design`.
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
        weights: Prior weights, as limma's ``voom(weights = ...)`` takes them:
            ``(n_samples,)`` sample weights, such as sample quality weights,
            ``(n_samples, n_features)`` observation weights, or ``(1, n_features)``
            feature weights. Finite and non-negative. They weight the least
            squares fits from which the trend is estimated, so that a zero weight
            leaves an observation out of them and gives it weight zero, and the
            returned weights include them, the weights a fit needs (limma's
            ``voom`` returns the trend's alone). A vector is always sample weights,
            also when there are as many features as samples, where limma takes it
            for feature weights.

    Returns:
        The log2 counts per million, their weights (finite and non-negative;
        positive where the prior weights are), the effective library sizes, and the
        mean-variance trend with its span.

    Warns:
        RuntimeWarning: If the design, without the observations of zero prior
            weight, leaves no residual degrees of freedom, so that no trend can be
            fitted; every weight is then its prior weight, or one.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``counts`` is not a 2-D array of finite, non-negative values
            with at least two features, the design does not have one row per
            sample or is not of full column rank, a library size or normalization
            factor is not positive, ``span`` is not in (0, 1], the weights have
            another shape or a negative or non-finite value, fewer than two
            features have counts (with residual degrees of freedom), or the trend
            is not positive where the weights are read from it.
    """
    y, sizes, fraction = voom_inputs(
        counts, design, library_sizes, norm_factors, span, "voom"
    )
    prior = _prior_weights(weights, y.shape, np.ones(y.shape, dtype=bool))
    trend, fitted = _voom_trend(y, design, sizes, fraction, prior)
    if not fitted:
        message = _NO_REPLICATION if prior is None else _NO_RESIDUAL_DF
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return VoomResult(
        log2_cpm=trend.log2_cpm,
        weights=_times_prior(trend.weights, prior),
        effective_library_sizes=sizes,
        mean_log_count=trend.mean_log_count,
        sqrt_sd=trend.sqrt_sd,
        sqrt_sd_trend=trend.sqrt_sd_trend,
        span=trend.span,
    )


def voom_with_quality_weights(
    counts: npt.ArrayLike,
    design: Any,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    span: float | None = None,
    variance_design: Any = None,
    variance_groups: npt.ArrayLike | None = None,
    method: Literal["gene_by_gene", "reml"] = "gene_by_gene",
    prior_n: float = 10.0,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> VoomResult:
    """voom's precision weights combined with sample quality weights.

    limma's ``voomWithQualityWeights`` (Liu et al. 2015), in its steps: :func:`voom`;
    sample quality weights a from its log2 counts per million and weights, by
    :func:`~cytepool.stats.sample_weights.sample_quality_weights`; :func:`voom`
    again, with a as prior weights; sample quality weights b from the new trend's
    weights w'; and the weights w'_gi b_i. Samples that follow the model less
    closely, such as pseudobulk samples pooled from few cells or from an outlying
    donor, get lower weights instead of being removed
    (https://docs.nygen.io/CytePool/statistics/limma-voom#sample-quality-weights).
    Continue as after :func:`voom`, with ``fit_linear_model(result.log2_cpm, design,
    weights=result.weights)``.

    Args:
        counts: ``(n_samples, n_features)`` counts, as for :func:`voom`.
        design: The design, as for :func:`voom`.
        library_sizes: Library size of each sample, as for :func:`voom`.
        norm_factors: Normalization factor of each sample, as for :func:`voom`.
        span: The span of the LOWESS trend, as for :func:`voom`: by default chosen
            from the number of features, as limma's ``voomWithQualityWeights``
            does with ``adaptive.span = TRUE``.
        variance_design: Design of the variance model, as for
            ``sample_quality_weights``; defaults to a weight for each sample.
        variance_groups: One label per sample, for a weight per group of samples;
            give at most one of the two.
        method: ``"gene_by_gene"``, limma's default here, or ``"reml"``.
        prior_n: Prior number of genes at equal weights, as for
            ``sample_quality_weights``; limma's ``voomWithQualityWeights`` uses 10.
        max_iterations: The most REML iterations for each estimate of the weights.
        tolerance: Convergence tolerance of REML.

    Returns:
        The log2 counts per million, their weights (the second trend's times the
        sample weights), the effective library sizes, the second trend, its span,
        and the ``sample_weights``.

    Warns:
        RuntimeWarning: If the design leaves no residual degrees of freedom (every
            weight is one), fewer than two (every sample weight is one), or the
            sample weights warn as ``sample_quality_weights`` does.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If an argument is invalid as for :func:`voom` or
            ``sample_quality_weights``, or the trend is not positive where weights
            are read from it.
    """
    y, sizes, fraction = voom_inputs(
        counts, design, library_sizes, norm_factors, span, "voom_with_quality_weights"
    )
    model = _variance_model(
        variance_design,
        variance_groups,
        y.shape[0],
        prior_n=prior_n,
        method=method,
        max_iterations=max_iterations,
        tolerance=tolerance,
        methods=("gene_by_gene", "reml"),
    )
    result, notes = _quality_weighted_voom(y, design, sizes, fraction, model)
    for note in notes:
        warnings.warn(note, RuntimeWarning, stacklevel=2)
    return result


def _quality_weighted_voom(
    y: FloatArray,
    design: Any,
    sizes: FloatArray,
    fraction: float,
    model: _VarianceModel,
) -> tuple[VoomResult, list[str]]:
    """The steps of :func:`voom_with_quality_weights` on checked inputs.

    Returns the result and the warnings for the caller to give, so that a caller
    can also record them.

    Args:
        y: The counts, checked.
        design: The design, checked by ``fit_linear_model``.
        sizes: The effective library sizes.
        fraction: The span.
        model: The variance model and the settings of its estimation.
    """
    x, _ = design_matrix(design, None)
    first, fitted = _voom_trend(y, design, sizes, fraction, None)
    notes = []
    if not fitted:
        notes.append(_NO_REPLICATION)
        second, quality = first, np.ones(y.shape[0])
    elif y.shape[0] - x.shape[1] < 2:
        # The sample weights need two residual degrees of freedom; voom does not.
        notes.append(_too_few_df_message(y.shape[0] - x.shape[1]))
        second, quality = first, np.ones(y.shape[0])
    else:
        log_variances, first_notes = _estimate(first.log2_cpm, x, first.weights, model)
        second, _ = _voom_trend(y, design, sizes, fraction, _weights_of(log_variances))
        log_variances, second_notes = _estimate(
            second.log2_cpm, x, second.weights, model
        )
        notes += list(dict.fromkeys(first_notes + second_notes))
        quality = _weights_of(log_variances)
    result = VoomResult(
        log2_cpm=second.log2_cpm,
        weights=second.weights * quality[:, np.newaxis],
        effective_library_sizes=sizes,
        mean_log_count=second.mean_log_count,
        sqrt_sd=second.sqrt_sd,
        sqrt_sd_trend=second.sqrt_sd_trend,
        span=second.span,
        sample_weights=quality,
    )
    return result, notes


def _voom_trend(
    y: FloatArray,
    design: Any,
    sizes: FloatArray,
    fraction: float,
    prior: FloatArray | None,
) -> tuple[Trend, bool]:
    """voom's log2 CPM and the trend's weights; False when no trend can be fitted.

    Args:
        y: The counts, checked.
        design: The design, checked by ``fit_linear_model``.
        sizes: The effective library sizes.
        fraction: The span.
        prior: Prior weights from ``_prior_weights``, or None.
    """
    log2_cpm = voom_log_cpm(y, sizes)
    fit = fit_linear_model(log2_cpm, design, weights=prior)
    mean_log_count = voom_mean_log_count(fit.mean_expression, sizes, y)
    if not np.any(fit.df_residual > 0.0):
        # limma returns no span then (found by probing).
        nan = np.full(y.shape[1], np.nan)
        unit = Trend(
            log2_cpm, np.ones_like(log2_cpm), mean_log_count, nan, nan.copy(), math.nan
        )
        return unit, False
    in_trend = (y > 0.0).any(axis=0) & (fit.df_residual > 0.0)
    weighted = (
        np.ones(y.shape, dtype=bool) if prior is None else _positive(prior, y.shape)
    )
    trend = trend_weights(
        log2_cpm,
        mean_log_count,
        fit.sigma,
        in_trend,
        _fitted_values(fit, log2_cpm, prior),
        sizes,
        fraction,
        weighted,
        "" if prior is None else " and residual degrees of freedom",
    )
    return trend, True


def _fitted_values(
    fit: LinearModelFit, log2_cpm: FloatArray, prior: FloatArray | None
) -> FloatArray:
    """Fitted log2 CPM of every observation, NaN where it is not determined.

    A feature whose coefficients are all estimable has fitted values design @
    coefficients. Zero prior weights can leave some coefficients of a feature
    inestimable; its observations of positive weight then get the values of the
    least squares fit over them, which determines them, and the others NaN.
    """
    fitted: FloatArray = fit.design @ fit.coefficients.T
    inestimable = np.flatnonzero(np.isnan(fit.coefficients).any(axis=1))
    if prior is None or inestimable.size == 0:
        return fitted
    for g in inestimable:
        weights = prior if prior.ndim == 1 else prior[:, g]
        rows = weights > 0.0
        fitted[:, g] = np.nan
        if rows.any():
            root = np.sqrt(weights[rows])
            design = fit.design[rows]
            solution = np.linalg.lstsq(
                design * root[:, np.newaxis], log2_cpm[rows, g] * root, rcond=None
            )[0]
            fitted[rows, g] = design @ solution
    return fitted


def _positive(prior: FloatArray, shape: tuple[int, int]) -> npt.NDArray[np.bool_]:
    positive = prior > 0.0
    if prior.ndim == 1:
        return np.repeat(positive[:, np.newaxis], shape[1], axis=1)
    return positive


def _times_prior(weights: FloatArray, prior: FloatArray | None) -> FloatArray:
    """The trend's weights times the prior weights; zero where a prior weight is."""
    if prior is None:
        return weights
    full = prior[:, np.newaxis] if prior.ndim == 1 else prior
    with np.errstate(invalid="ignore"):
        combined: FloatArray = np.where(full > 0.0, weights * full, 0.0)
    return combined
