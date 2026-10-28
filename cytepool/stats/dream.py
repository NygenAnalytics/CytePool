"""dream: voom weights from mixed models, and moderated tests of their fixed effects.

:func:`voom_with_dream_weights` gives voom's precision weights from the mean-variance
trend of mixed model fits, as variancePartition's ``voomWithDreamWeights`` does, in a
:class:`~cytepool.stats.voom.VoomResult`, and :func:`moderated_fixed_effect_tests`
moderates the tests of :func:`~cytepool.stats.mixed_models.fixed_effect_tests` by
empirical Bayes, as its ``eBayes`` and ``topTable`` do for ``dream`` fits, in a
:class:`ModeratedFixedEffectTests`. dream (Hoffman and Roussos 2021) is limma-voom (Law
et al. 2014) on linear mixed models; :func:`~cytepool.dream_voom` runs it on the cell
types of a cube, as dreamlet does (Hoffman et al. 2026).
https://docs.nygen.io/CytePool/statistics/mixed-models#dream gives the steps.

Matching variancePartition:
    The results match variancePartition 1.36.2, with limma 3.62.1 and lme4 1.1-38: its
    documentation gives the method, and running it on chosen inputs settled the
    conventions it leaves open.

    - ``voomWithDreamWeights`` fits the mixed models by maximum likelihood, and reads
      the trend at lme4's ``fitted`` values, which include the predicted random effects.
      Without random effects it is limma's ``voom``.
    - It divides each feature's prior weights by their mean before the fits, for sample
      weights and observation weights alike. By default it returns the trend's weights
      alone (``rescaleWeightsAfter = FALSE``), as limma's ``voom`` does; with
      ``rescaleWeightsAfter = TRUE``, the trend's weights times the scaled prior
      weights, which are the weights a fit needs and the ones returned here. They do not
      change when the prior weights are multiplied by a constant.
    - ``dream`` also fits each feature with its weights divided by their mean, so its
      ``sigma`` is the residual standard deviation of an observation of the feature's
      average weight: ``residual_variance / mean_weights`` of a fit to the weights as
      given (https://docs.nygen.io/CytePool/statistics/mixed-models#dream says what
      that means for the trend). With every variance component at zero, the fit is
      limma's weighted fit, and the moderated tests are limma's for the weights of each
      feature divided by their mean.
    - ``eBayes`` moderates those variances on the approximate residual degrees of
      freedom ``rdf``, with limma's ``squeezeVar`` and its default estimator: the
      likelihood estimator (Chen et al. 2025) when the degrees of freedom differ between
      features, as they usually do, and the method of moments otherwise
      (``legacy = NULL``). The moderated t-statistics and the B-statistics are those of
      limma's ``eBayes`` (Smyth 2004) for these variances on these degrees of freedom;
      the p-values use the degrees of freedom of the tests, Satterthwaite's or
      Kenward-Roger's, plus the prior's, which, unlike limma's total degrees of freedom,
      are not capped at the sum of the residual degrees of freedom.
    - ``topTable`` adds ``z.std``, sign(t) times the standard normal quantile of half
      the two-sided p-value: the z-statistic with the p-value of the t-statistic, so
      that tests on different degrees of freedom can be compared. It computes it from
      the p-value floored at 1e-300.
    - ``dream(ddf = "adaptive")``, the default, uses Kenward-Roger's method (Kenward
      and Roger 1997) for 19 samples and Satterthwaite's (1946) for 20; its help page
      says that it "uses KR for <= 20 samples". With ``ddf = "Kenward-Roger"`` it
      tests with lmerTest's ``contest``, which calls pbkrtest: Kenward-Roger's
      adjustment is computed with every observation at the average weight (see
      :mod:`cytepool.stats.mixed_models`), the standard errors are the adjusted ones,
      and ``eBayes`` moderates the tests as it does Satterthwaite's. A gene whose
      adjusted covariance matrix of the coefficients is not positive definite or has a
      condition number above 1e6, by R's estimate ``kappa``, is tested by
      Satterthwaite's method instead, the rule that variancePartition's NEWS for
      1.31.16 states. The condition number depends on the units of continuous
      covariates.
    - ``topTable`` with several coefficients gives the coefficients, ``AveExpr``,
      ``F``, ``P.Value``, ``adj.P.Val`` and ``F.std``: F is limma's moderated
      F-statistic, from the moderated t-statistics and each gene's covariance matrix of
      the coefficients, Kenward-Roger's adjusted one under their method, and
      ``F.std`` the F-statistic on infinite denominator degrees of freedom with the same
      p-value.

Differences from variancePartition:
    https://docs.nygen.io/CytePool/statistics/mixed-models#mixed-models-limits lists
    them; the docstrings of :func:`voom_with_dream_weights`,
    :class:`ModeratedFixedEffectTests` and :func:`~cytepool.dream_voom` give those that
    change what they return.

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

    Kenward MG, Roger JH (1997). Small sample inference for fixed effects from
    restricted maximum likelihood. Biometrics 53, 983-997. doi:10.2307/2533558

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

from cytepool._counts import as_counts, effective_library_sizes
from cytepool._jit import thread_limit
from cytepool.design.design import Design
from cytepool.stats._mean_variance_trend import (
    scaled_prior_weights,
    trend_span,
    trend_weights,
    voom_log_cpm,
    voom_mean_log_count,
)
from cytepool.stats._fixed_effect_inference import (
    InferenceParts,
    combined_df,
    contrast_rank,
    correlation_directions,
    kenward_roger_parts,
    satterthwaite_parts,
)
from cytepool.stats._mixed_model_designs import (
    MixedDesign,
    maximum_likelihood,
    mixed_design,
)
from cytepool.stats.distributions import (
    f_isf,
    f_sf,
    norm_isf,
    norm_logsf,
    t_logsf,
    t_sf,
)
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import LinearModelFit
from cytepool.stats.mixed_models import FixedEffectTests, _fit
from cytepool.stats.voom import VoomResult

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
        tests: The tests moderated, with their fit, contrasts and degrees of
            freedom, Satterthwaite's or Kenward-Roger's.
        residual_variance_at_mean_weight: ``(n_features,)`` the variances
            moderated: the residual variance of an observation of each feature's
            average weight, ``fit.residual_variance / fit.mean_weights`` (dream's
            ``sigma^2``).
        df_residual: ``(n_features,)`` their degrees of freedom, the fit's
            approximate residual degrees of freedom (dream's ``rdf``).
        prior_variance: The prior variance (``s2.prior``): a float, or one value
            per feature with a trend.
        df_prior: The prior degrees of freedom (``df.prior``): a float, possibly
            infinite, or one value per feature from a robust fit that finds
            outliers.
        posterior_variance: ``(n_features,)`` posterior variances (``s2.post``).
        t: ``(n_features, n_contrasts)`` moderated t-statistics: the estimates over
            their standard errors times (residual_variance_at_mean_weight /
            posterior_variance)^(1/2).
        df_total: ``(n_features, n_contrasts)`` their degrees of freedom: those
            of the tests plus the prior's (``df.total``).
        p_value: Two-sided p-values of ``t`` on ``df_total`` degrees of freedom.
        log_odds: Log-odds that each contrast is non-zero (limma's B-statistic,
            ``lods``), as limma's ``eBayes`` gives them for these variances on
            ``df_residual``.
        z: Signed z-statistics with the same p-values, sign(t) times the standard
            normal quantile of half the p-value (``topTable``'s ``z.std``). Below
            p-values of 1e-300 they come from the log of the tail probability, so
            they keep growing where ``z.std`` stops at 37.07.
        coefficient_prior_variance: ``(n_contrasts,)`` unscaled prior variance of
            a non-zero contrast (``var.prior``).
        proportion: The assumed proportion of non-zero contrasts that
            ``log_odds`` takes.
        prior_estimator: The estimator of the prior: ``"moments"`` or
            ``"likelihood"``.
        f_statistic: ``(n_features,)`` moderated F-statistics testing every
            contrast of ``tests`` at once: t' R^-1 t / ``f_df1``, for the moderated
            t-statistics t and the correlation matrix R of their estimates
            (``topTable``'s ``F`` for these contrasts).
        f_df1: The numerator degrees of freedom of the F-tests, the number of
            linearly independent contrasts.
        f_df2: ``(n_features,)`` their denominator degrees of freedom, those of
            the r uncorrelated moderated t-statistics combined as 2 E / (E - r),
            with E = sum nu_m / (nu_m - 2) over their degrees of freedom nu_m, as
            lmerTest combines those of its F-tests: each feature's own, where
            ``topTable`` uses the first gene's for every gene. With one contrast,
            the F-test is the moderated t-test.
        f_p_value: ``(n_features,)`` p-values of ``f_statistic`` on ``f_df1`` and
            ``f_df2`` degrees of freedom.
        standardized_f_statistic: ``(n_features,)`` the F-statistic on ``f_df1``
            and infinite degrees of freedom with the same p-value (``topTable``'s
            ``F.std``, which takes the number of coefficients where it differs from
            their rank), so that tests on different degrees of freedom compare.
    """

    tests: FixedEffectTests
    residual_variance_at_mean_weight: FloatArray
    df_residual: FloatArray
    prior_variance: float | FloatArray
    df_prior: float | FloatArray
    posterior_variance: FloatArray
    t: FloatArray
    df_total: FloatArray
    p_value: FloatArray
    log_odds: FloatArray
    z: FloatArray
    coefficient_prior_variance: FloatArray
    proportion: float
    prior_estimator: str
    f_statistic: FloatArray
    f_df1: int
    f_df2: FloatArray
    f_p_value: FloatArray
    standardized_f_statistic: FloatArray

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
    read at the fitted values with the predicted random effects. A feature whose
    mixed model has no minimum, as one that the fixed and random effects reproduce
    (see :class:`~cytepool.stats.mixed_models.MixedModelFit`), is left out of the
    trend, and its weights are read from the trend at its own values. Continue with
    ``fit_mixed_model(result.log2_cpm, design, weights=result.weights)``,
    ``fixed_effect_tests`` and :func:`moderated_fixed_effect_tests`.
    https://docs.nygen.io/CytePool/statistics/mixed-models#dream explains the steps.

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
            divided by their mean, then weight the fits from which the trend is
            estimated, and multiply the weights returned.
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
    ml = maximum_likelihood(method)
    parts = mixed_design(design)
    if parts.x.shape[0] != y.shape[0]:
        raise ValueError(
            f"counts have {y.shape[0]} rows (samples) but the design has "
            f"{parts.x.shape[0]}; both need one row per sample"
        )
    # A copy, so that the result does not change with the caller's array.
    sizes = np.array(effective_library_sizes(y, library_sizes, norm_factors))
    fraction = trend_span(span, y.shape[1])
    # voomWithDreamWeights divides each feature's prior weights by their mean, and with
    # rescaleWeightsAfter = TRUE returns the trend's weights times them (found by
    # probing).
    prior = scaled_prior_weights(weights, y.shape)
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
    variance, and its degrees of freedom are those of the tests (Satterthwaite's or
    Kenward-Roger's) plus the prior's. The moderated F-statistic tests every
    contrast of ``tests`` at once, as ``topTable`` does for several coefficients,
    with degrees of freedom of its own for every feature.
    https://docs.nygen.io/CytePool/statistics/mixed-models#dream explains the steps.

    Args:
        tests: Tests from :func:`~cytepool.stats.mixed_models.fixed_effect_tests`,
            of a REML fit as ``dream`` makes them by default. To F-test some of the
            contrasts, test those alone.
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
    if tests.df_method == "kenward_roger":
        parts = kenward_roger_parts(tests.fit, tests.kenward_roger_weights == "average")
    else:
        parts = satterthwaite_parts(tests.fit)
    return _moderated_tests(
        tests,
        parts,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        prior_estimator=prior_estimator,
    )


def _moderated_tests(
    tests: FixedEffectTests,
    parts: InferenceParts,
    *,
    proportion: float = 0.01,
    stdev_coef_lim: Sequence[float] = (0.1, 4.0),
    trend: bool | npt.ArrayLike = False,
    robust: bool = False,
    winsor_tail_p: float | Sequence[float] = (0.05, 0.1),
    prior_estimator: Literal["moments", "likelihood"] | None = None,
) -> ModeratedFixedEffectTests:
    """The steps of :func:`moderated_fixed_effect_tests`, with the parts of the tests.

    ``parts`` are those from which ``tests`` were computed, feature by feature: the
    degrees of freedom of the directions of the F-test come from them.
    """
    fit = tests.fit
    matrix = tests.contrasts
    if matrix.shape[1] == 0:
        raise ValueError(
            "tests has no contrasts to moderate: the design has no fixed effects, so "
            "fixed_effect_tests tested none; add fixed effects to the design, such as "
            "an intercept"
        )
    # dream's sigma^2: the residual variance of an observation of average weight, as
    # dream fits each feature with its weights divided by their mean (found by probing).
    variance = fit.residual_variance / fit.mean_weights
    fitted = np.isfinite(variance) & np.isfinite(fit.df_residual)
    if not fitted.any():
        raise ValueError(
            "no feature has a mixed model fit, so the prior cannot be estimated; "
            "fit_mixed_model leaves unfitted the features that the fixed and random "
            "effects reproduce"
        )
    sigma = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        stdev_unscaled = tests.standard_error / sigma[:, np.newaxis]
        # The covariance matrix that the tests use: dream's cov.coefficients.list,
        # Kenward-Roger's adjusted one under their method (found by probing).
        per_feature = tests.covariance / variance[:, np.newaxis, np.newaxis]
    cross = fit.design.T @ fit.design
    linear = LinearModelFit(
        coefficients=tests.estimate,
        stdev_unscaled=stdev_unscaled,
        sigma=sigma,
        # A feature without a fit has no information about the prior.
        df_residual=np.where(fitted, fit.df_residual, 0.0),
        cov_coefficients=matrix.T @ np.linalg.solve(cross, matrix),
        mean_expression=fit.mean_expression,
        coefficient_names=tests.contrast_names,
        design=fit.design,
        cov_coefficients_per_feature=per_feature,
        contrasts=matrix,
    )
    # eBayes moderates on the approximate residual degrees of freedom with limma's
    # default estimator, and its p-values take the tests' degrees of freedom plus the
    # prior's, without limma's cap (found by probing).
    moderated = moderated_statistics(
        linear,
        proportion=proportion,
        stdev_coef_lim=stdev_coef_lim,
        trend=trend,
        robust=robust,
        winsor_tail_p=winsor_tail_p,
        prior_estimator=prior_estimator,
    )
    df_prior = np.broadcast_to(moderated.df_prior, variance.shape)
    df_total = tests.df + df_prior[:, np.newaxis]
    t = moderated.t
    p_value = 2.0 * np.asarray(t_sf(np.abs(t), df_total), dtype=np.float64)
    f_value, f_df1, f_df2 = _moderated_f(t, tests, parts, df_prior)
    with np.errstate(invalid="ignore"):
        f_p_value = np.asarray(f_sf(f_value, float(f_df1), f_df2), dtype=np.float64)
        standardized = np.asarray(
            f_isf(f_p_value, float(f_df1), np.inf), dtype=np.float64
        )
    return ModeratedFixedEffectTests(
        tests=tests,
        residual_variance_at_mean_weight=_frozen(variance),
        df_residual=_frozen(fit.df_residual),
        prior_variance=moderated.prior_variance,
        df_prior=moderated.df_prior,
        posterior_variance=moderated.posterior_variance,
        t=t,
        df_total=_frozen(df_total),
        p_value=_frozen(p_value),
        log_odds=moderated.log_odds,
        z=_frozen(_signed_z(t, df_total, p_value)),
        coefficient_prior_variance=moderated.coefficient_prior_variance,
        proportion=moderated.proportion,
        prior_estimator=moderated.prior_estimator,
        f_statistic=_frozen(f_value),
        f_df1=f_df1,
        f_df2=_frozen(f_df2),
        f_p_value=_frozen(f_p_value),
        standardized_f_statistic=_frozen(standardized),
    )


def _moderated_f(
    t: FloatArray, tests: FixedEffectTests, parts: InferenceParts, df_prior: FloatArray
) -> tuple[FloatArray, int, FloatArray]:
    """dream's moderated F-statistic of every contrast, and its degrees of freedom.

    F = t' R^-1 t / r, for the moderated t-statistics and the correlation matrix R of
    the estimates, is the mean of the squares of r uncorrelated moderated
    t-statistics, those of the eigenvectors of R (``correlation_directions``). Each
    has the degrees of freedom of its contrast plus the prior's, as a moderated
    t-statistic of dream does, and the denominator degrees of freedom combine them
    as lmerTest combines those of an F-test (``combined_df``).

    Returns:
        The F-statistics, the numerator and the denominator degrees of freedom.
    """
    rank = contrast_rank(tests.contrasts)
    n_features = t.shape[0]
    if rank == 0:
        return np.full(n_features, np.nan), 0, np.full(n_features, np.nan)
    values, vectors, df = correlation_directions(
        parts, tests.contrasts, tests.covariance, rank
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.einsum("gc,gcm->gm", t, vectors) / np.sqrt(values)
        f_value: FloatArray = np.sum(scores**2, axis=1) / rank
    f_df2 = combined_df(df + df_prior[:, np.newaxis])
    return f_value, rank, f_df2


def _dream_weights(
    y: FloatArray,
    sizes: FloatArray,
    fraction: float,
    parts: MixedDesign,
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
    log2_cpm = voom_log_cpm(y, sizes)
    # voomWithDreamWeights fits by maximum likelihood, and reads the trend at the
    # fitted values with the predicted random effects (found by probing).
    fit = _fit(log2_cpm, prior, ml, parts)
    mean_log_count = voom_mean_log_count(fit.mean_expression, sizes, y)
    sigma = np.sqrt(fit.residual_variance)
    in_trend = (y > 0.0).any(axis=0) & np.isfinite(sigma)
    # A feature that the effects reproduce has no fit; its fitted values are its
    # values, the limit of the fits as the residual variance goes to zero.
    fitted = np.where(np.isnan(fit.fitted_values), log2_cpm, fit.fitted_values)
    trend = trend_weights(
        log2_cpm,
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
        log2_cpm=log2_cpm,
        weights=weights,
        effective_library_sizes=sizes,
        mean_log_count=mean_log_count,
        sqrt_sd=trend.sqrt_sd,
        sqrt_sd_trend=trend.sqrt_sd_trend,
        span=fraction,
    )


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
