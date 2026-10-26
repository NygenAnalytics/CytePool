"""dream: voom weights from mixed models, and moderated tests of their fixed effects.

dream (Hoffman and Roussos 2021) is limma-voom with a linear mixed model in place of
the linear model, for designs with repeated measures: several samples of a donor,
tissues of one patient, time courses. The donor (or another grouping factor) gets a
random intercept, whose variance is estimated for every gene, and contrasts of the
fixed effects are tested with Satterthwaite's (1946) degrees of freedom. dreamlet
(Hoffman et al. 2026) runs it on the pseudobulk profiles of every cell type. The
steps, as variancePartition runs them (``voomWithDreamWeights``, ``dream``,
``eBayes`` and ``topTable``):

1. Precision weights, :func:`voom_with_dream_weights`. voom's (Law et al. 2014): the
   log2 counts per million are fitted, the square roots of the residual standard
   deviations are smoothed by LOWESS against the average log2 counts, and each
   observation gets the inverse fourth power of the trend at its fitted value
   (:mod:`cytepool.stats.voom`). The fit is the mixed model of every feature, by
   maximum likelihood, and the fitted values include the predicted random effects.
2. Fits and tests: :func:`~cytepool.stats.mixed_models.fit_mixed_model` by REML with
   those weights, and :func:`~cytepool.stats.mixed_models.fixed_effect_tests`, whose
   t-statistics have Satterthwaite's degrees of freedom.
3. Moderation, :func:`moderated_fixed_effect_tests`. The residual variance of every
   feature, that of an observation of its average weight, is shrunk towards a prior
   by limma's empirical Bayes (Smyth 2004) on the approximate residual degrees of
   freedom of the mixed model, tr((I - H)' (I - H)) for its hat matrix H
   (``MixedModelFit.residual_df``). The moderated t-statistic divides an estimate by
   its standard error with the posterior variance in place of the residual
   variance, and has Satterthwaite's degrees of freedom plus the prior's.

Conventions that the documentation leaves open were found by running
variancePartition 1.36.2, with limma 3.62.1 and lme4 1.1-38, as a black box:

- ``voomWithDreamWeights`` fits the mixed models by maximum likelihood (its weights
  agree with those of maximum likelihood fits to lme4's tolerance, and differ from
  those of REML fits by up to 12% in an example), and reads the trend at lme4's
  ``fitted`` values, which include the predicted random effects. Without random
  effects it is limma's ``voom``.
- It divides each feature's prior weights by their mean before the fits, for sample
  weights and observation weights alike. By default it returns the trend's weights
  alone (``rescaleWeightsAfter = FALSE``), as limma's ``voom`` does; with
  ``rescaleWeightsAfter = TRUE``, the trend's weights times the scaled prior
  weights, which are the weights a fit needs and the ones returned here. They do
  not change when the prior weights are multiplied by a constant.
- ``dream`` also fits each feature with its weights divided by their mean, so its
  ``sigma`` is the residual standard deviation of an observation of the feature's
  average weight: ``residual_variance / mean_weights`` of a fit to the weights as
  given. With voom's weights this is about the variance that the trend predicts at
  the feature's expression, so the variances that ``eBayes`` moderates keep the
  mean-variance trend that the weights removed; ``trend=True`` lets the prior follow
  it. Its ``sigma`` therefore differs from limma's with the same weights; with
  every variance component at zero, the fit is limma's weighted fit, and the
  moderated tests are limma's for the weights of each feature divided by their mean.
- ``eBayes`` moderates those variances on the approximate residual degrees of
  freedom ``rdf``, with limma's ``squeezeVar`` and its default estimator: the
  likelihood estimator (Chen et al. 2025) when the degrees of freedom differ between
  features, as they usually do, and the method of moments otherwise
  (``legacy = NULL``). The moderated t-statistics and the B-statistics are those of
  limma's ``eBayes`` for these variances on these degrees of freedom; the p-values
  use Satterthwaite's degrees of freedom plus the prior's, which, unlike limma's
  total degrees of freedom, are not capped at the sum of the residual degrees of
  freedom.
- ``topTable`` adds ``z.std``, sign(t) times the standard normal quantile of half
  the two-sided p-value: the z-statistic with the p-value of the t-statistic, so
  that tests on different degrees of freedom can be compared. It computes it from
  the p-value floored at 1e-300.

Deliberate differences from variancePartition, reported by ``validation/dream.py``:

- The LOWESS span is chosen from the number of features by default, as
  :func:`~cytepool.stats.voom.voom` and limma's ``voom`` (since limma 3.66) choose it.
  ``voomWithDreamWeights`` uses 0.5, which ``span=0.5`` reproduces.
- Beyond a p-value of 1e-300, the z-statistic is computed from the logarithm of the
  tail probability, so it stays exact where ``topTable`` stops at 37.07.
- A feature whose mixed model has no minimum, which the fixed and random effects
  reproduce, is not fitted (see :class:`~cytepool.stats.mixed_models.MixedModelFit`):
  it is left out of the trend, its weights are read at its values, which the effects
  reproduce, and its tests are NaN. lme4 fits it with a residual standard deviation
  near zero, which enters the trend, and ``dream`` tests it.

Kenward-Roger's degrees of freedom (``dream``'s default up to 20 samples), random
slopes, F-tests of several contrasts and variance partitioning are not implemented
yet.

References:
    Hoffman GE, Roussos P (2021). Dream: powerful differential expression analysis for
    repeated measures designs. Bioinformatics 37, 192-201.
    doi:10.1093/bioinformatics/btaa687

    Hoffman GE, Lee D, Bendl J, Prashant NM, Hong A, Casey C, Alvia M, Shao Z,
    Argyriou S, Therrien K, Venkatesh S, Voloudakis G, Haroutunian V, Fullard JF,
    Roussos P (2026). Efficient differential expression analysis of large-scale
    single-cell transcriptomics data using Dreamlet. Nature Communications 17, 9597.
    doi:10.1038/s41467-026-75680-8

    Law CW, Chen Y, Shi W, Smyth GK (2014). voom: precision weights unlock linear model
    analysis tools for RNA-seq read counts. Genome Biology 15, R29.
    doi:10.1186/gb-2014-15-2-r29

    Smyth GK (2004). Linear models and empirical Bayes methods for assessing
    differential expression in microarray experiments. Statistical Applications in
    Genetics and Molecular Biology 3, article 3. doi:10.2202/1544-6115.1027

    Chen Y, Chen L, Lun ATL, Baldoni PL, Smyth GK (2025). edgeR v4: powerful
    differential analysis of sequencing data with expanded functionality and improved
    support for small counts and larger datasets. Nucleic Acids Research 53, gkaf018.
    doi:10.1093/nar/gkaf018

    Satterthwaite FE (1946). An approximate distribution of estimates of variance
    components. Biometrics Bulletin 2, 110-114. doi:10.2307/3002019
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

from cytepool._jit import thread_limit
from cytepool._masked_arrays import unmasked_array
from cytepool.design.design import Design
from cytepool._counts import as_counts, effective_library_sizes
from cytepool.stats.distributions import norm_isf, norm_logsf, t_logsf, t_sf
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import LinearModelFit, _real_array
from cytepool.stats.mixed_models import (
    FixedEffectTests,
    _fit,
    _maximum_likelihood,
    _mixed_design,
    _MixedDesign,
)
from cytepool.stats.voom import (
    VoomResult,
    _log_cpm,
    _mean_log_count,
    _trend_span,
    _trend_weights,
)

__all__ = [
    "ModeratedFixedEffectTests",
    "moderated_fixed_effect_tests",
    "voom_with_dream_weights",
]

FloatArray = npt.NDArray[np.float64]

# topTable floors the p-values at this value to compute z.std (found by probing);
# below it, the z-statistics here come from the logarithm of the tail probability.
_SMALLEST_P_VALUE = 1e-300


@dataclass(frozen=True, eq=False, repr=False)
class ModeratedFixedEffectTests:
    """dream's moderated t-tests of contrasts of the fixed effects of mixed models.

    Rows of the per-feature arrays are features and columns are contrasts, as in
    ``tests``. The arrays are read-only.

    Attributes:
        tests: The tests moderated, with their fit, contrasts and Satterthwaite's
            degrees of freedom.
        residual_variance_at_mean_weight: ``(n_features,)`` the variances
            moderated: the residual variance of an observation of each feature's
            average weight, ``fit.residual_variance / fit.mean_weights`` (dream's
            ``sigma^2``).
        residual_df: ``(n_features,)`` their degrees of freedom, the fit's
            approximate residual degrees of freedom (dream's ``rdf``).
        prior_variance: The prior variance (``s2.prior``): a float, or one value
            per feature with a trend.
        prior_df: The prior degrees of freedom (``df.prior``): a float, possibly
            infinite, or one value per feature from a robust fit that finds
            outliers.
        posterior_variance: ``(n_features,)`` posterior variances (``s2.post``).
        t: ``(n_features, n_contrasts)`` moderated t-statistics: the estimates over
            their standard errors times (residual_variance_at_mean_weight /
            posterior_variance)^(1/2).
        df_total: ``(n_features, n_contrasts)`` their degrees of freedom:
            Satterthwaite's plus the prior's (``df.total``).
        p_value: Two-sided p-values of ``t`` on ``df_total`` degrees of freedom.
        lods: Log-odds that each contrast is non-zero (the B-statistic), as limma's
            ``eBayes`` gives them for these variances on ``residual_df``.
        z: Signed z-statistics with the same p-values, sign(t) times the standard
            normal quantile of half the p-value (``topTable``'s ``z.std``).
        coefficient_prior_variance: ``(n_contrasts,)`` unscaled prior variance of
            a non-zero contrast (``var.prior``).
        proportion: The assumed proportion of non-zero contrasts behind ``lods``.
        prior_estimator: The estimator of the prior: ``"moments"`` or
            ``"likelihood"``.
    """

    tests: FixedEffectTests
    residual_variance_at_mean_weight: FloatArray
    residual_df: FloatArray
    prior_variance: float | FloatArray
    prior_df: float | FloatArray
    posterior_variance: FloatArray
    t: FloatArray
    df_total: FloatArray
    p_value: FloatArray
    lods: FloatArray
    z: FloatArray
    coefficient_prior_variance: FloatArray
    proportion: float
    prior_estimator: str

    def __repr__(self) -> str:
        return (
            f"ModeratedFixedEffectTests(n_features={self.t.shape[0]}, "
            f"contrast_names={self.tests.contrast_names!r}, "
            f"prior_estimator={self.prior_estimator!r})"
        )


def voom_with_dream_weights(
    counts: npt.ArrayLike,
    design: Design,
    *,
    library_sizes: npt.ArrayLike | None = None,
    norm_factors: npt.ArrayLike | None = None,
    span: float | None = None,
    weights: npt.ArrayLike | None = None,
    method: Literal["ml", "reml"] = "ml",
    n_threads: int | None = None,
) -> VoomResult:
    """voom's precision weights from the mean-variance trend of mixed model fits.

    variancePartition's ``voomWithDreamWeights``: voom (see
    :func:`~cytepool.stats.voom.voom`), with the linear model replaced by the mixed
    model of ``design``, fitted to every feature by maximum likelihood, and the trend
    read at the fitted values with the predicted random effects; the module
    docstring gives the conventions. Continue with ``fit_mixed_model(result.log_cpm,
    design, weights=result.weights)``, ``fixed_effect_tests`` and
    :func:`moderated_fixed_effect_tests`.

    Args:
        counts: ``(n_samples, n_features)`` counts, as for ``voom``: finite and
            non-negative, at least two features.
        design: A ``Design`` with random intercepts, as
            :func:`~cytepool.stats.mixed_models.fit_mixed_model` takes; for a design
            without random effects use ``voom``.
        library_sizes: Library size of each sample; defaults to the total counts.
        norm_factors: Normalization factor of each sample, such as TMM factors;
            defaults to ones.
        span: The span of the LOWESS trend, in (0, 1]; by default chosen from the
            number of features, as ``voom`` chooses it.
            ``voomWithDreamWeights`` uses 0.5.
        weights: Prior weights: ``(n_samples,)`` sample weights, such as each
            pseudobulk sample's number of cells, or ``(n_samples, n_features)``
            observation weights; finite and positive. Each feature's weights are
            divided by their mean, then weight the fits behind the trend, and
            multiply the weights returned.
        method: ``"ml"``, maximum likelihood, as ``voomWithDreamWeights`` fits, or
            ``"reml"``.
        n_threads: The most threads for the fits, as for ``fit_mixed_model``.

    Returns:
        The log2 counts per million, their weights (the trend's times the scaled
        prior weights), the effective library sizes, and the trend with its span.

    Raises:
        TypeError: If an argument has the wrong type, or is or holds a numpy masked
            array.
        ValueError: If ``counts`` is not a 2-D array of finite non-negative values
            with at least two features, the design has no random effects or cannot
            be fitted (as for ``fit_mixed_model``), its samples are not the rows of
            ``counts``, a library size or normalization factor is not positive,
            ``span`` is not in (0, 1], a prior weight is not finite and positive,
            fewer than two features with counts can be fitted, or the trend is not
            positive where weights are read from it.
    """
    y = as_counts(counts)
    if y.shape[1] < 2:
        raise ValueError(
            f"a mean-variance trend needs at least two features, got {y.shape[1]}"
        )
    if isinstance(design, Design) and not design.random:
        raise ValueError(
            "the design has no random effects, so voom_with_dream_weights has no mixed "
            "model to fit; use voom, which fits the linear model of such a design, or "
            "build the design with random effects such as random='(1 | donor)'"
        )
    ml = _maximum_likelihood(method)
    parts = _mixed_design(design)
    if parts.x.shape[0] != y.shape[0]:
        raise ValueError(
            f"counts have {y.shape[0]} rows (samples) but the design has "
            f"{parts.x.shape[0]}; both need one row per sample"
        )
    # A copy, so that the result does not change with the caller's array.
    sizes = np.array(effective_library_sizes(y, library_sizes, norm_factors))
    fraction = _trend_span(span, y.shape[1])
    prior = _scaled_prior_weights(weights, y.shape)
    with thread_limit(n_threads):
        return _dream_weights(y, sizes, fraction, parts, prior, ml)


def moderated_fixed_effect_tests(
    tests: FixedEffectTests,
    *,
    proportion: float = 0.01,
    stdev_coef_lim: Sequence[float] = (0.1, 4.0),
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments", "likelihood"] | None = None,
) -> ModeratedFixedEffectTests:
    """Empirical Bayes moderated t-tests of mixed model contrasts, as dream's eBayes.

    variancePartition's ``eBayes`` for ``dream`` fits, with ``topTable``'s z-statistic:
    the residual variance of an observation of each feature's average weight is
    shrunk towards a prior on the approximate residual degrees of freedom by limma's
    empirical Bayes (:func:`~cytepool.stats.empirical_bayes.moderated_statistics`,
    with its estimators and options), each contrast's t-statistic uses the posterior
    variance, and its degrees of freedom are Satterthwaite's plus the prior's. See
    the module docstring.

    Args:
        tests: Tests from :func:`~cytepool.stats.mixed_models.fixed_effect_tests`,
            of a REML fit as ``dream`` makes them by default.
        proportion: Assumed proportion of non-zero contrasts, in (0, 1).
        stdev_coef_lim: Limits for the standard deviation of a non-zero contrast,
            as for ``moderated_statistics``.
        trend: Whether the prior variance follows the mean expression of the
            features (``eBayes(trend = TRUE)``), or an ``(n_features,)`` covariate.
        robust: Whether to estimate the prior robustly (``robust = TRUE``).
        winsor_tail_p: Winsorized tail proportions of the robust moment estimator.
        prior_estimator: ``"moments"``, ``"likelihood"`` or None for limma's default
            (``legacy = NULL``), as for ``moderated_statistics``.

    Returns:
        The moderated tests.

    Raises:
        TypeError: If ``tests`` is not a ``FixedEffectTests``, or an option has the
            wrong type or is a numpy masked array.
        ValueError: If ``tests`` has no contrasts, an option is invalid, or no
            feature has a fit.
    """
    if not isinstance(tests, FixedEffectTests):
        raise TypeError(
            f"tests must be a FixedEffectTests, from fixed_effect_tests, got "
            f"{type(tests).__name__}"
        )
    fit = tests.fit
    matrix = tests.contrasts
    if matrix.shape[1] == 0:
        raise ValueError(
            "tests has no contrasts to moderate: the design has no fixed effects, so "
            "fixed_effect_tests tested none; add fixed effects to the design, such as "
            "an intercept"
        )
    # dream's sigma^2: the residual variance of an observation of average weight.
    variance = fit.residual_variance / fit.mean_weights
    fitted = np.isfinite(variance) & np.isfinite(fit.residual_df)
    if not fitted.any():
        raise ValueError(
            "no feature has a mixed model fit, so the prior cannot be estimated; "
            "fit_mixed_model leaves unfitted the features that the fixed and random "
            "effects reproduce"
        )
    sigma = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        stdev_unscaled = tests.standard_error / sigma[:, np.newaxis]
        per_feature = (
            np.einsum("jc,gjk,kd->gcd", matrix, fit.cov_coefficients, matrix)
            / variance[:, np.newaxis, np.newaxis]
        )
    cross = fit.design.T @ fit.design
    linear = LinearModelFit(
        coefficients=tests.estimate,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        # A feature without a fit has no information about the prior.
        df_residual=np.where(fitted, fit.residual_df, 0.0),
        cov_coefficients=matrix.T @ np.linalg.solve(cross, matrix),
        mean_expression=fit.mean_expression,
        coefficient_names=tests.contrast_names,
        design=fit.design,
        cov_coefficients_per_feature=per_feature,
        contrasts=matrix,
    )
    moderated = moderated_statistics(
        linear,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        prior_estimator=prior_estimator,
    )
    prior_df = np.broadcast_to(moderated.prior_df, variance.shape)
    df_total = tests.df + prior_df[:, np.newaxis]
    t = moderated.t
    p_value = 2.0 * np.asarray(t_sf(np.abs(t), df_total), dtype=np.float64)
    return ModeratedFixedEffectTests(
        tests=tests,
        residual_variance_at_mean_weight=_frozen(variance),
        residual_df=_frozen(fit.residual_df),
        prior_variance=moderated.prior_variance,
        prior_df=moderated.prior_df,
        posterior_variance=moderated.posterior_variance,
        t=t,
        df_total=_frozen(df_total),
        p_value=_frozen(p_value),
        lods=moderated.lods,
        z=_frozen(_signed_z(t, df_total, p_value)),
        coefficient_prior_variance=moderated.coefficient_prior_variance,
        proportion=moderated.proportion,
        prior_estimator=moderated.prior_estimator,
    )


def _dream_weights(
    y: FloatArray,
    sizes: FloatArray,
    fraction: float,
    parts: _MixedDesign,
    prior: FloatArray | None,
    ml: bool,
) -> VoomResult:
    """The steps of :func:`voom_with_dream_weights` on checked inputs.

    Call it within ``thread_limit``, which sets the threads of the fits.

    Args:
        y: The counts, checked.
        sizes: The effective library sizes.
        fraction: The span.
        parts: The checked parts of the mixed model's design.
        prior: Prior weights, each feature's scaled to mean one, or None.
        ml: Whether to fit by maximum likelihood.
    """
    log_cpm = _log_cpm(y, sizes)
    fit = _fit(log_cpm, prior, ml, parts)
    mean_log_count = _mean_log_count(fit.mean_expression, sizes, y)
    sigma = np.sqrt(fit.residual_variance)
    in_trend = (y > 0.0).any(axis=0) & np.isfinite(sigma)
    # A feature that the effects reproduce has no fit; its fitted values are its
    # values, the limit of the fits as the residual variance goes to zero.
    fitted = np.where(np.isnan(fit.fitted_values), log_cpm, fit.fitted_values)
    trend = _trend_weights(
        log_cpm,
        mean_log_count,
        sigma,
        in_trend,
        fitted,
        sizes,
        fraction,
        np.ones(y.shape, dtype=bool),
        " and a mixed model fit",
    )
    weights = trend.weights
    if prior is not None:
        weights = weights * (prior[:, np.newaxis] if prior.ndim == 1 else prior)
    return VoomResult(
        log_cpm=log_cpm,
        weights=weights,
        effective_library_sizes=sizes,
        mean_log_count=mean_log_count,
        sqrt_sd=trend.sqrt_sd,
        sqrt_sd_trend=trend.sqrt_sd_trend,
        span=fraction,
    )


def _scaled_prior_weights(
    weights: npt.ArrayLike | None, shape: tuple[int, int]
) -> FloatArray | None:
    """Prior weights with each feature's divided by their mean, as variancePartition.

    Sample weights become a vector of mean one, observation weights a matrix whose
    columns (features) have mean one (found by probing variancePartition).
    """
    if weights is None:
        return None
    w = _real_array(unmasked_array(weights, "weights"), "weights")
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


def _signed_z(t: FloatArray, df: FloatArray, p_value: FloatArray) -> FloatArray:
    """sign(t) times the standard normal quantile of half the two-sided p-value.

    As topTable's z.std computes it from the p-value; where the p-value is below
    1e-300, from the logarithm of the tail probability of ``t``, which stays exact
    where the p-value underflows.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        z = np.asarray(norm_isf(p_value / 2.0), dtype=np.float64)
        tiny = p_value < _SMALLEST_P_VALUE
    if tiny.any():
        log_tail = np.asarray(t_logsf(np.abs(t[tiny]), df[tiny]), dtype=np.float64)
        z[tiny] = [_normal_upper_quantile_of_log(value) for value in log_tail]
    signed: FloatArray = np.sign(t) * z
    return signed


def _normal_upper_quantile_of_log(log_tail: float) -> float:
    """The z > 0 whose standard normal upper tail probability is exp(log_tail).

    Newton's method on log P(Z > z) = log_tail, for tails far below the smallest
    double, from the leading terms of the asymptotic expansion of Mills' ratio.
    """
    if not math.isfinite(log_tail):
        return math.inf if log_tail == -math.inf else math.nan
    z = math.sqrt(-2.0 * log_tail - math.log(-4.0 * math.pi * log_tail))
    for _ in range(50):
        log_upper = float(norm_logsf(z))
        log_density = -0.5 * z * z - 0.5 * math.log(2.0 * math.pi)
        step = (log_upper - log_tail) / math.exp(log_density - log_upper)
        z += step
        if abs(step) <= 1e-15 * z:
            break
    return z


def _frozen(values: Any) -> Any:
    array: npt.NDArray[Any] = np.array(values, dtype=np.float64)
    array.flags.writeable = False
    return array
