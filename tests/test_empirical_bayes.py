"""Tests for empirical Bayes moderation of feature-wise variances.

Reference values in ``tests/data/empirical_bayes`` come from limma's ``eBayes``,
``treat``, ``squeezeVar``, ``fitFDist``, ``fitFDistRobustly`` and ``fitFDistUnequalDF1``,
with ``legacy`` TRUE, FALSE or NULL, and from its ``weightedLowess``, written by
``validation/empirical_bayes.py``. The other tests check properties that a wrong
implementation would break: limits, invariances and closed forms.
"""

import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.stats import distributions
from cytepool.stats.empirical_bayes import (
    ModeratedFit,
    fit_f_distribution,
    fit_f_distribution_likelihood,
    fit_f_distribution_robust,
    moderated_statistics,
    squeeze_variances,
    treat,
)
from cytepool.stats.linear_models import (
    LinearModelFit,
    contrasts_fit,
    fit_linear_model,
)
from cytepool.stats.smoothing import _weighted_lowess
from cytepool.stats.special import digamma

DATA = Path(__file__).parent / "data" / "empirical_bayes"
EXACT = 1e-10
# limma solves the robust estimator's equation for the prior df only to a
# tolerance, which leaves its df.prior within about 1e-7 of the exact root, and
# everything downstream of it as close.
CLOSE = 1e-6
LFC = float(np.log2(1.2))
OPTIONS = {
    "constant": (False, False),
    "trend": (True, False),
    "robust": (False, True),
    "trend_robust": (True, True),
}
# Option sets of the model cases: trend, robust and the prior estimator, which is
# limma's legacy = TRUE ("moments"), FALSE ("likelihood") or NULL (None).
RUNS: dict[str, tuple[bool, bool, str | None]] = {
    name: (*options, "moments") for name, options in OPTIONS.items()
}
for _suffix, _estimator in (("default", None), ("likelihood", "likelihood")):
    RUNS |= {f"{n}_{_suffix}": (*o, _estimator) for n, o in OPTIONS.items()}
# Option sets of the variance cases: robust, the estimator and the Winsorized tails.
VARIANCE_RUNS: dict[str, tuple[bool, str, tuple[float, float]]] = {
    "moments": (False, "moments", (0.05, 0.1)),
    "robust": (True, "moments", (0.05, 0.1)),
    "robust_wide": (True, "moments", (0.1, 0.2)),
    "likelihood": (False, "likelihood", (0.05, 0.1)),
    "likelihood_robust": (True, "likelihood", (0.05, 0.1)),
}
# Runs where CytePool deliberately differs from limma: robust likelihood fits leave
# out variances that are not usable, and keep prior weights in their second fit.
DIFFERS = {
    ("no_residual_df", "robust_default"),
    ("no_residual_df", "trend_robust_default"),
    ("prior_weights", "likelihood_robust"),
    ("robust_unusable", "likelihood_robust"),
}
MODEL_CASES = (
    "two_groups",
    "contrasts",
    "forty_samples",
    "infinite_prior",
    "missing_values",
    "observation_weights",
    "zero_contrast",
    "2_genes",
    "3_genes",
    "5_genes",
    "25_genes",
    "1_gene_without_variation",
    "2_genes_one_without_variation",
    "2_genes_without_variation",
    "missing_values_all_df",
    "zero_weights",
    "no_residual_df",
)
VARIANCE_CASES = (
    "sizes_2",
    "sizes_3",
    "sizes_5",
    "sizes_6",
    "sizes_12",
    "sizes_29",
    "sizes_30",
    "sizes_200",
    "sizes_2_constant",
    "sizes_29_constant",
    "sizes_200_constant",
    "2_covariate_levels",
    "3_covariate_levels",
    "infinite_covariate",
    "zero_variances",
    "mostly_zero",
    "unusable",
    "outliers",
    "outliers_constant",
    "outliers_unequal_df",
    "outliers_unequal_df_constant",
    "robust_infinite_prior",
    "robust_missing",
    "robust_nine",
    "robust_moments_infinite",
    "robust_infinite_unequal_df",
    "robust_infinite_trend",
    "fractional_df",
    "prior_weights",
    "robust_unusable",
    "large_trend",
    "flat_likelihood",
    "heavy_tails",
    "few_trended_3",
    "few_trended_4",
    "few_trended_5",
    "few_trended_missing",
    "two_positive",
    "flat_covariate",
    "narrow_covariate",
)
WEIGHTED_LOWESS_CASES = (
    "ties",
    "narrow_windows",
    "flat_windows",
    "two_clusters",
    "trend_weights",
)
# p-values are compared relative to max(p, 1e-300) max(1, |log p|), about the error
# in log p relative to |log p|: an error in the prior df moves a p-value in the far
# tail in proportion to |log p|, and doubles below 1e-300 carry no relative
# precision. Statistics are compared relative to max(|value|, 1).
PROBABILITIES = {"p.value", "F.p.value", "treat.p", "means.F.p.value"}
PROBABILITIES |= {"tail.p.value"}
STATISTICS = {"t", "lods", "F", "treat.t", "means.F", "prob.outlier"}


@cache
def reference(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz") as data:
        return {key: data[key] for key in data.files}


def runs(case: str) -> list[str]:
    return sorted({key.split(" ")[1] for key in reference(case) if " " in key})


def limma(case: str, run: str, key: str, shape: tuple[int, ...]) -> np.ndarray:
    """A limma output, as an array of the given shape (R writes column by column)."""
    values = reference(case)[f"limma {run} {key}"]
    if len(shape) == 2:
        return values.reshape(shape[1], shape[0]).T
    return np.broadcast_to(values, shape) if values.size == 1 else values


def assert_agrees(
    ours: Any,
    desired: Any,
    tolerance: float,
    floor: float = 0.0,
    log_scaled: bool = False,
) -> None:
    """NaN and infinities in the same places, |ours - desired| <= tolerance * scale.

    The scale is max(|desired|, floor), times max(1, |log scale|) if ``log_scaled``.
    """
    ours, desired = np.broadcast_arrays(
        np.asarray(ours, dtype=float), np.asarray(desired, dtype=float)
    )
    np.testing.assert_array_equal(np.isnan(ours), np.isnan(desired))
    infinite = np.isinf(desired)
    np.testing.assert_array_equal(ours[infinite], desired[infinite])
    finite = np.isfinite(desired)
    scale = np.maximum(np.abs(desired[finite]), floor)
    if log_scaled:
        scale *= np.maximum(1.0, np.abs(np.log(scale)))
    difference = np.abs(ours[finite] - desired[finite])
    assert np.all(difference <= tolerance * scale), np.max(difference / scale)


def check(case: str, run: str, key: str, ours: Any, tolerance: float) -> None:
    floor = 1.0 if key in STATISTICS else 1e-300 if key in PROBABILITIES else 0.0
    shape = np.shape(ours) if np.ndim(ours) else (1,)
    if not key.endswith("df2.shrunk"):  # always one per feature here
        # One value where limma reports one, one per feature where it reports more.
        single = reference(case)[f"limma {run} {key}"].size == 1
        assert single == (np.size(ours) == 1), (key, np.shape(ours))
    desired = limma(case, run, key, shape)
    assert_agrees(ours, desired, tolerance, floor, log_scaled=key in PROBABILITIES)


@cache
def model_fit(case: str) -> LinearModelFit:
    data = reference(case)
    weights = data["w"].T if "w" in data else None
    fit = fit_linear_model(data["y"].T, data["X"], weights=weights)
    if "C" in data:
        fit = contrasts_fit(fit, data["C"], method="limma")
    return fit


@cache
def moderated(case: str, run: str, f_statistic: str = "limma") -> ModeratedFit:
    trend, robust, estimator = RUNS[run]
    return moderated_statistics(
        model_fit(case),
        trend=trend,
        robust=robust,
        f_statistic=f_statistic,
        prior_estimator=estimator,  # type: ignore[arg-type]
    )


def tolerance_of(run: str) -> float:
    # The robust moment estimator and the likelihood estimator are iterative.
    _, robust, estimator = RUNS[run]
    return CLOSE if robust or estimator != "moments" else EXACT


MODEL_RUNS = [
    (case, run)
    for case in MODEL_CASES
    for run in runs(case)
    if (case, run) not in DIFFERS
]
HYPERPARAMETER_RUNS = [
    (case, run)
    for case in VARIANCE_CASES
    for run in runs(case)
    if (case, run) not in DIFFERS
]


@pytest.mark.parametrize(("case", "run"), MODEL_RUNS)
def test_moderated_statistics_match_limma(case: str, run: str) -> None:
    eb = moderated(case, run)
    tolerance = tolerance_of(run)
    for key, ours in (
        ("s2.prior", eb.prior_variance),
        ("df.prior", eb.prior_df),
        ("s2.post", eb.posterior_variance),
        ("df.total", eb.df_total),
        ("t", eb.t),
        ("p.value", eb.p_value),
        ("lods", eb.lods),
        ("var.prior", eb.coefficient_prior_variance),
        ("F", eb.f_statistic),
        ("F.p.value", eb.f_p_value),
    ):
        check(case, run, key, ours, tolerance)


@pytest.mark.parametrize(("case", "run"), MODEL_RUNS)
def test_treat_matches_limma(case: str, run: str) -> None:
    trend, robust, estimator = RUNS[run]
    fit = model_fit(case)
    result = treat(
        fit,
        lfc=LFC,
        trend=trend,
        robust=robust,
        prior_estimator=estimator,  # type: ignore[arg-type]
    )
    tolerance = tolerance_of(run)
    check(case, run, "treat.p", result.p_value, tolerance)
    # limma reports t = 0 where a coefficient is NaN; it is NaN here
    nan = np.isnan(fit.coefficients)
    check(case, run, "treat.t", np.where(nan, 0.0, result.t), tolerance)
    np.testing.assert_array_equal(np.isnan(result.t), nan)
    assert result.lfc == pytest.approx(LFC, rel=1e-15)


@pytest.mark.parametrize(
    ("case", "run"),
    [(c, r) for c, r in MODEL_RUNS if c in ("missing_values", "observation_weights")],
)
def test_exact_f_statistics_match_limma_on_group_means(case: str, run: str) -> None:
    # The F-test of all coefficients does not depend on how the groups are
    # coded, and with group means as coefficients limma's correlation matrix is
    # every gene's own; limma's F of that coding is the exact F-statistic.
    exact = moderated(case, run, "exact")
    tolerance = tolerance_of(run)
    check(case, run, "means.F", exact.f_statistic, tolerance)
    check(case, run, "means.F.p.value", exact.f_p_value, tolerance)
    approximate = moderated(case, run)
    assert np.nanmax(np.abs(exact.f_statistic / approximate.f_statistic - 1)) > 1e-3


@pytest.mark.parametrize(("case", "run"), HYPERPARAMETER_RUNS)
def test_hyperparameters_match_limma(case: str, run: str) -> None:
    data = reference(case)
    robust, estimator, tails = VARIANCE_RUNS[run]
    covariate = data.get("covariate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        squeezed = squeeze_variances(
            data["v"],
            data["d"],
            covariate=covariate,
            robust=robust,
            winsor_tail_p=tails,
            prior_estimator=estimator,  # type: ignore[arg-type]
        )
        if estimator == "likelihood":
            fit = fit_f_distribution_likelihood(
                data["v"],
                data["d"],
                covariate=covariate,
                prior_weights=data.get("pw"),
                robust=robust,
            )
            parts = dict(zip(("scale", "df2", "df2.shrunk", "df2.outlier"), fit))
        elif robust:
            fit = fit_f_distribution_robust(
                data["v"], data["d"], covariate=covariate, winsor_tail_p=tails
            )
            parts = {
                "scale": fit.scale,
                "df2": fit.df2,
                "df2.shrunk": fit.df2_shrunk,
                "tail.p.value": fit.tail_p_value,
                "prob.outlier": fit.prob_outlier,
                "df2.outlier": fit.df2_outlier,
            }
        else:
            moments = fit_f_distribution(data["v"], data["d"], covariate=covariate)
            parts = {"scale": moments.scale, "df2": moments.df2}
    tolerance = CLOSE if robust or estimator == "likelihood" else EXACT
    check(case, run, "var.post", squeezed.posterior_variance, tolerance)
    check(case, run, "var.prior", squeezed.prior_variance, tolerance)
    check(case, run, "df.prior", squeezed.prior_df, tolerance)
    for key, ours in parts.items():
        if reference(case)[f"limma {run} {key}"].size:  # limma omits some parts
            check(case, run, key, ours, tolerance)


def test_infinite_prior_df_gives_ordinary_t_on_the_prior_variance() -> None:
    eb = moderated("infinite_prior", "constant")
    fit = model_fit("infinite_prior")
    assert eb.prior_df == np.inf
    np.testing.assert_array_equal(eb.posterior_variance, eb.prior_variance)
    ordinary = fit.coefficients / (fit.stdev_unscaled * np.sqrt(eb.prior_variance))
    np.testing.assert_allclose(eb.t, ordinary, rtol=1e-14)
    # the t-tests have the residual df pooled over all genes
    np.testing.assert_array_equal(eb.df_total, fit.df_residual.sum())
    np.testing.assert_allclose(
        eb.p_value,
        2.0 * distributions.t_sf(np.abs(ordinary), eb.df_total[0]),
        rtol=1e-13,
    )

    # With the variance known to be s0^2, the log-odds are the prior log-odds plus
    # the log Bayes factor of N(0, s0^2 (v + v0)) against N(0, s0^2 v) for the
    # estimated coefficient, with v its squared unscaled standard error.
    def log_normal_density(x: np.ndarray, variance: np.ndarray) -> np.ndarray:
        return -0.5 * (np.log(2.0 * np.pi * variance) + x * x / variance)

    v = eb.prior_variance * fit.stdev_unscaled**2
    v0 = eb.prior_variance * eb.coefficient_prior_variance
    bayes_factor = log_normal_density(fit.coefficients, v + v0)
    bayes_factor -= log_normal_density(fit.coefficients, v)
    np.testing.assert_allclose(
        eb.lods, np.log(0.01 / 0.99) + bayes_factor, rtol=1e-12, atol=1e-12
    )


def test_shrinkage_moves_every_variance_towards_the_prior() -> None:
    for case, run in (("two_groups", "constant"), ("two_groups", "trend_robust")):
        eb = moderated(case, run)
        s2 = model_fit(case).sigma ** 2
        prior = np.broadcast_to(eb.prior_variance, s2.shape)
        low, high = np.minimum(s2, prior), np.maximum(s2, prior)
        post = eb.posterior_variance
        assert np.all((post >= low * (1 - 1e-14)) & (post <= high * (1 + 1e-14)))
        assert np.all(np.abs(post - prior) < np.abs(s2 - prior) + 1e-15)


def test_robust_estimation_downweights_planted_outliers() -> None:
    # 1,000 variances from 0.2 F(4, 10), of which 20 are multiplied by 100
    rng = np.random.default_rng(11)
    clean = 0.2 * rng.f(4.0, 10.0, 1000)
    planted = np.arange(20)
    v = clean.copy()
    v[planted] *= 100.0
    # Without outliers the robust fit is the moment fit (Phipson et al. 2016).
    reference = fit_f_distribution(clean, 4.0)
    assert fit_f_distribution_robust(clean, 4.0).df2 == reference.df2
    robust = fit_f_distribution_robust(v, 4.0)
    moments = fit_f_distribution(v, 4.0)
    # The outliers halve the moment estimate of df2; the robust one stays near the
    # estimate from the clean data.
    assert moments.df2 < 0.5 * reference.df2
    assert abs(robust.df2 - reference.df2) < 0.5 * abs(moments.df2 - reference.df2)
    assert np.all(robust.df2_shrunk[planted] < 0.2 * robust.df2)
    assert np.all(robust.prob_outlier[planted] > 0.8)
    others = np.setdiff1d(np.arange(v.size), planted)
    assert np.mean(robust.df2_shrunk[others] == robust.df2) > 0.9
    # prior df never exceed df2 and fall with the tail probability
    assert np.all(robust.df2_shrunk <= robust.df2)
    order = np.argsort(robust.tail_p_value)
    assert np.all(np.diff(robust.df2_shrunk[order]) >= -1e-12 * robust.df2)


def test_treat_p_values_are_at_least_the_moderated_p_values() -> None:
    fit = model_fit("two_groups")
    eb = moderated("two_groups", "constant")
    zero = treat(fit, lfc=0.0)
    np.testing.assert_allclose(zero.p_value, eb.p_value, rtol=1e-14)
    np.testing.assert_array_equal(zero.t, eb.t)
    previous = zero.p_value
    for lfc in (0.1, 0.5, 1.0):
        result = treat(fit, lfc=lfc)
        assert np.all(result.p_value >= previous * (1 - 1e-14))
        assert np.all(result.p_value <= 1.0)
        assert np.all(np.abs(result.t) <= np.abs(eb.t))
        previous = result.p_value
    np.testing.assert_allclose(
        treat(fit, fc=1.5).p_value, treat(fit, lfc=np.log2(1.5)).p_value, rtol=1e-13
    )
    np.testing.assert_allclose(
        treat(fit, fc=0.8).p_value, treat(fit, fc=1.25).p_value, rtol=1e-13
    )
    np.testing.assert_array_equal(treat(fit).p_value, treat(fit, fc=1.2).p_value)
    # An all-zero contrast is null for any positive threshold (t = 0 and p = 1, as
    # in limma); at a zero threshold its t is 0 / 0, NaN (limma reports 0).
    contrast = model_fit("zero_contrast")
    positive = treat(contrast, lfc=0.5)
    assert np.all(positive.t[:, 1] == 0.0) and np.all(positive.p_value[:, 1] == 1.0)
    assert np.isnan(treat(contrast, lfc=0.0).t[:, 1]).all()


def test_zero_coefficient_variance_gives_the_prior_log_odds() -> None:
    # stdev_coef_lim = (0, 0) forces v0 = 0: a non-zero coefficient is then
    # indistinguishable from zero, and every log-odds is the prior log-odds.
    eb = moderated_statistics(
        model_fit("two_groups"), proportion=0.05, stdev_coef_lim=(0.0, 0.0)
    )
    np.testing.assert_array_equal(eb.coefficient_prior_variance, 0.0)
    np.testing.assert_allclose(eb.lods, np.log(0.05 / 0.95), rtol=1e-14)


def test_exact_f_statistic_is_invariant_to_coding_and_matches_group_means() -> None:
    # Treatment coding with missing values gives correlated coefficients that
    # differ between genes; group-mean coding gives uncorrelated ones, whose exact
    # F-statistic is the mean of the squared t-statistics.
    data = reference("missing_values")
    treatment = moderated_statistics(fit_linear_model(data["y"].T, data["X"]))
    means = moderated_statistics(fit_linear_model(data["y"].T, data["Xm"]))
    expected = np.mean(means.t**2, axis=1)
    np.testing.assert_allclose(treatment.f_statistic, expected, rtol=1e-12)
    np.testing.assert_allclose(means.f_statistic, expected, rtol=1e-12)
    assert treatment.f_df1 == means.f_df1 == 3
    complete = fit_linear_model(
        reference("two_groups")["y"].T, reference("two_groups")["X"]
    )
    exact = moderated_statistics(complete, f_statistic="exact")
    limma_f = moderated_statistics(complete, f_statistic="limma")
    np.testing.assert_allclose(exact.f_statistic, limma_f.f_statistic, rtol=1e-13)


# Each option set with each estimator named.
EXPLICIT_RUNS = [run for run in RUNS if not run.endswith("_default")]


def options(run: str) -> dict[str, Any]:
    trend, robust, estimator = RUNS[run]
    return {"trend": trend, "robust": robust, "prior_estimator": estimator}


@pytest.mark.parametrize("run", EXPLICIT_RUNS)
def test_results_do_not_depend_on_the_order_of_features(run: str) -> None:
    data = reference("two_groups")
    order = np.random.default_rng(5).permutation(data["y"].shape[0])
    permuted = moderated_statistics(
        fit_linear_model(data["y"][order].T, data["X"]), **options(run)
    )
    eb = moderated_statistics(fit_linear_model(data["y"].T, data["X"]), **options(run))
    for name in ("posterior_variance", "df_total", "t", "p_value", "lods"):
        np.testing.assert_allclose(
            getattr(permuted, name), getattr(eb, name)[order], rtol=1e-10
        )
    np.testing.assert_allclose(
        permuted.coefficient_prior_variance, eb.coefficient_prior_variance, rtol=1e-12
    )


@pytest.mark.parametrize("run", EXPLICIT_RUNS)
def test_rescaling_the_data_rescales_the_variances(run: str) -> None:
    # Multiplying expression by c multiplies variances by c^2 (and the mean
    # expression, the trend covariate, by c); the tests do not change.
    data = reference("two_groups")
    eb = moderated_statistics(fit_linear_model(data["y"].T, data["X"]), **options(run))
    # stdev_coef_lim bounds standard deviations in the units of the data
    scaled = moderated_statistics(
        fit_linear_model(4.0 * data["y"].T, data["X"]),
        stdev_coef_lim=(0.4, 16.0),
        **options(run),
    )
    np.testing.assert_allclose(
        scaled.posterior_variance, 16.0 * eb.posterior_variance, rtol=1e-10
    )
    np.testing.assert_allclose(scaled.prior_df, eb.prior_df, rtol=1e-10)
    np.testing.assert_allclose(scaled.t, eb.t, rtol=1e-10)
    np.testing.assert_allclose(scaled.lods, eb.lods, rtol=1e-10, atol=1e-10)


def test_features_without_observations_do_not_change_the_others() -> None:
    # limma's eBayes(trend=TRUE) stops on the NaN mean of such a feature.
    data = reference("two_groups")
    y = np.vstack([data["y"], np.full((1, data["y"].shape[1]), np.nan)])
    for run in EXPLICIT_RUNS:
        eb = moderated_statistics(fit_linear_model(y.T, data["X"]), **options(run))
        base = moderated_statistics(
            fit_linear_model(data["y"].T, data["X"]), **options(run)
        )
        assert np.isnan(eb.t[-1]).all() and np.isnan(eb.f_statistic[-1])
        # the fit with a missing feature takes another linear model code path
        np.testing.assert_allclose(eb.t[:-1], base.t, rtol=1e-11)
        np.testing.assert_allclose(eb.df_total[:-1], base.df_total, rtol=1e-12)


def test_variance_with_zero_df_gets_the_prior() -> None:
    v = np.array([0.05, 0.9, np.nan, 0.8, 0.3, 2.5, 0.01, 0.25])
    d = np.array([4.0, 4.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0])
    squeezed = squeeze_variances(v, d)
    assert np.isfinite(squeezed.prior_df)
    assert squeezed.posterior_variance[2] == squeezed.prior_variance
    assert squeezed.posterior_variance[3] == squeezed.prior_variance
    usable = d > 0
    alone = squeeze_variances(v[usable], d[usable])
    assert alone.prior_df == squeezed.prior_df
    np.testing.assert_array_equal(
        alone.posterior_variance, squeezed.posterior_variance[usable]
    )
    # a missing variance with df stays missing
    nan = squeeze_variances(np.r_[v, np.nan], np.r_[d, 4.0])
    assert np.isnan(nan.posterior_variance[-1])


def test_equal_variances_give_an_infinite_prior_df() -> None:
    v = np.full(50, 0.3)
    fit = fit_f_distribution(v, 4.0)
    assert fit.df2 == np.inf and fit.scale == pytest.approx(0.3, rel=1e-15)
    squeezed = squeeze_variances(v, 4.0)
    np.testing.assert_allclose(squeezed.posterior_variance, 0.3, rtol=1e-15)
    # With a trend, the scale for an infinite df2 is exp of the fitted
    # log s^2 - digamma(d / 2) + log(d / 2), as in limma, not the mean variance.
    trended = fit_f_distribution(v, 4.0, covariate=np.linspace(0, 1, 50))
    assert trended.df2 == np.inf
    np.testing.assert_allclose(
        trended.scale, 0.3 * np.exp(np.log(2.0) - digamma(2.0)), rtol=1e-13
    )


def test_at_most_two_variances_are_not_moderated() -> None:
    for v in (np.array([0.5]), np.array([0.2, 3.0])):
        squeezed = squeeze_variances(v, 4.0, robust=True)
        assert squeezed.prior_df == 0.0
        np.testing.assert_array_equal(squeezed.posterior_variance, v)
        np.testing.assert_array_equal(squeezed.prior_variance, v)


def test_one_or_two_features_with_zero_variance_give_nan_not_an_error() -> None:
    # One or two features are not moderated, so an all-zero feature keeps a zero
    # prior variance: its t-statistic is 0 / 0. Without a finite t-statistic the
    # coefficients' prior variance is limma's default, the reciprocal of the first
    # prior variance: 1 / 0, infinite. limma 3.62.1 gives these values; the
    # *_without_variation fixtures check every output against it.
    one = fit_linear_model(np.zeros((3, 1)), np.ones((3, 1)))
    y = np.array([[0.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    two = fit_linear_model(y, np.ones((3, 1)))
    for trend in (False, True):
        for robust in (False, True):
            eb = moderated_statistics(one, trend=trend, robust=robust)
            assert np.isnan(eb.t).all() and np.isnan(eb.lods).all()
            np.testing.assert_array_equal(eb.prior_variance, 0.0)
            assert eb.prior_df == 0.0
            np.testing.assert_array_equal(eb.coefficient_prior_variance, np.inf)
            eb = moderated_statistics(two, trend=trend, robust=robust)
            # The second feature keeps its own variance, 1 on 2 df: t = sqrt(3).
            assert np.isnan(eb.t[0, 0])
            assert eb.t[1, 0] == pytest.approx(np.sqrt(3.0), rel=1e-14)
            assert np.isfinite(eb.coefficient_prior_variance).all()
            assert treat(one, trend=trend, robust=robust).p_value[0, 0] == 1.0


def test_one_usable_variance_gives_zero_prior_df() -> None:
    fit = fit_f_distribution(np.array([0.4, np.nan, 0.7]), np.array([3.0, 3.0, 0.0]))
    assert fit.df2 == 0.0 and fit.scale == 0.4


def test_mostly_zero_variances_warn_and_stop_the_robust_estimator() -> None:
    v = np.r_[np.zeros(6), 0.3, 0.5, 0.2]
    with pytest.warns(RuntimeWarning, match="more than half of the variances"):
        fit = fit_f_distribution(v, 4.0)
    # zeros count as 1e-5 when the median is zero, as in limma
    assert np.isfinite(fit.df2) and fit.scale > 0.0
    # the robust estimator is undefined; limma stops too
    with pytest.raises(ValueError, match="6 of the 9 usable variances are zero"):
        fit_f_distribution_robust(v, 4.0)
    with pytest.raises(ValueError, match="robust=False"):
        squeeze_variances(v, 4.0, robust=True)
    # with at most two usable variances the robust fit is the moment fit, whose
    # zeros count as 1e-5 (limma returns the same)
    with pytest.warns(RuntimeWarning, match="more than half of the variances"):
        few = fit_f_distribution_robust([0.0, 0.0, np.nan], 4.0)
    assert few.df2 == np.inf and few.scale == pytest.approx(1e-5, rel=1e-12)


def test_nearly_constant_variances_have_no_outliers() -> None:
    # Variances far less spread than chi-squared sampling gives: df2 is infinite and
    # every prior df stays infinite, as in limma. The probabilities of not being an
    # outlier that fall below one belong to the smallest variances, and the
    # cumulative maximum over increasing tail probabilities lifts them to one.
    v = np.array([0.30, 0.25, 0.35, 0.28, 0.33, 0.27, 0.31, 0.29, 0.26, 0.4, 0.32, 0.3])
    robust = fit_f_distribution_robust(v, 4.0)
    assert robust.df2 == np.inf
    assert np.all(robust.prob_outlier[np.argsort(v)[:3]] > 0.0)
    np.testing.assert_array_equal(robust.df2_shrunk, np.inf)
    squeezed = squeeze_variances(v, 4.0, robust=True)
    np.testing.assert_array_equal(squeezed.posterior_variance, squeezed.prior_variance)


def test_few_distinct_covariate_values_give_group_means() -> None:
    # With as many spline df as distinct covariate values, the trend is the mean
    # of the log-variance terms in each group.
    rng = np.random.default_rng(7)
    z = np.repeat([0.0, 1.0, 5.0], 40)
    v = np.exp(z / 2) * rng.f(4.0, 6.0, z.size)
    fit = fit_f_distribution(v, 4.0, covariate=z)
    assert np.isfinite(fit.df2)
    log_scale = np.log(fit.scale)
    for level in (0.0, 1.0, 5.0):
        assert np.ptp(log_scale[z == level]) < 1e-12
    group = [np.log(v[z == level]).mean() for level in (0.0, 1.0, 5.0)]
    np.testing.assert_allclose(
        np.diff(log_scale[[0, 40, 80]]), np.diff(group), rtol=1e-12
    )


def test_infinite_covariates_move_one_unit_beyond_the_range() -> None:
    rng = np.random.default_rng(8)
    z = rng.uniform(0.0, 10.0, 60)
    v = (0.1 + 0.05 * z) * rng.f(4.0, 6.0, 60)
    moved = z.copy()
    moved[[0, 1]] = [z.max() + 1.0, z.min() - 1.0]
    infinite = z.copy()
    infinite[[0, 1]] = [np.inf, -np.inf]
    a = fit_f_distribution(v, 4.0, covariate=infinite)
    b = fit_f_distribution(v, 4.0, covariate=moved)
    np.testing.assert_array_equal(a.scale, b.scale)
    assert a.df2 == b.df2


def test_robust_fit_of_few_features_is_the_moment_fit() -> None:
    rng = np.random.default_rng(9)
    v = 0.3 * rng.f(4.0, 6.0, 9)
    v[0] *= 30.0
    robust = fit_f_distribution_robust(v, 4.0)
    moments = fit_f_distribution(v, 4.0)
    assert robust.df2 == moments.df2 and robust.scale == moments.scale
    np.testing.assert_array_equal(robust.df2_shrunk, moments.df2)
    assert np.isnan(robust.tail_p_value).all() and np.isnan(robust.df2_outlier)
    # 9 features times a tail proportion of 0.2 reach one: the robust fit runs
    wide = fit_f_distribution_robust(v, 4.0, winsor_tail_p=(0.1, 0.2))
    assert np.isfinite(wide.tail_p_value).all()


def test_default_estimator_follows_limmas_rule_for_fits() -> None:
    # Equal residual df: the method of moments, bit for bit, as limma's eBayes with
    # its default legacy = NULL; unequal residual df: the likelihood estimator.
    fit = model_fit("two_groups")
    default = moderated_statistics(fit, trend=True)
    moments = moderated_statistics(fit, trend=True, prior_estimator="moments")
    assert default.prior_estimator == moments.prior_estimator == "moments"
    for name in ("prior_variance", "prior_df", "posterior_variance", "t", "lods"):
        np.testing.assert_array_equal(getattr(default, name), getattr(moments, name))
    unequal = model_fit("missing_values_all_df")
    assert np.unique(unequal.df_residual).size > 1
    default = moderated_statistics(unequal, robust=True)
    likelihood = moderated_statistics(
        unequal, robust=True, prior_estimator="likelihood"
    )
    assert default.prior_estimator == "likelihood"
    np.testing.assert_array_equal(default.t, likelihood.t)
    assert treat(unequal).prior_estimator == "likelihood"
    assert treat(unequal, prior_estimator="moments").prior_estimator == "moments"
    assert "prior_estimator='likelihood'" in repr(default)


def test_default_estimator_compares_the_positive_df_exactly() -> None:
    # limma's legacy = NULL (found by probing): df of zero do not count, the df of a
    # missing variance does, and any difference in the df counts.
    v = 0.4 * np.random.default_rng(12).f(4.0, 8.0, 40)

    def estimator_of(variances: np.ndarray, df: np.ndarray) -> str:
        chosen = squeeze_variances(variances, df).prior_df
        moments = squeeze_variances(variances, df, prior_estimator="moments")
        likelihood = squeeze_variances(variances, df, prior_estimator="likelihood")
        assert moments.prior_df != likelihood.prior_df
        assert chosen in (moments.prior_df, likelihood.prior_df)
        return "moments" if chosen == moments.prior_df else "likelihood"

    d = np.full(40, 4.0)
    assert estimator_of(v, d) == "moments"
    assert estimator_of(v, np.r_[np.zeros(3), d[3:]]) == "moments"
    assert estimator_of(np.r_[np.nan, v[1:]], np.r_[3.0, d[1:]]) == "likelihood"
    assert estimator_of(v, np.r_[4.0 + 1e-12, d[1:]]) == "likelihood"
    # A df below 1e-15 counts before it is set to zero for the fit.
    assert estimator_of(v, np.r_[1e-16, d[1:]]) == "likelihood"


def test_likelihood_search_ends_where_limmas_does() -> None:
    # limma searches d0 by R's optimize over v = d0 / (d0 + 2) in [1/2, 0.9998]
    # (found by probing). When the likelihood keeps rising towards an end, the golden
    # section steps stop where limma 3.62.1 stops for every such input it was given:
    # at d0 = 2.0003664572957871 and d0 = 8134.8447803826621.
    rng = np.random.default_rng(0)
    spread = 0.5 * rng.f(20.0, 0.3, 1000)
    assert fit_f_distribution_likelihood(spread, 20.0).df2 == 2.0003664572957871
    narrow = 0.5 * np.exp(rng.normal(0.0, 0.01, 1000))
    assert fit_f_distribution_likelihood(narrow, 4.0).df2 == 8134.8447803826621


def test_variances_that_are_not_usable_do_not_change_the_likelihood_fit() -> None:
    # Missing variances, df of zero or below 0.01, and zero prior weights take no
    # part in the fit, robust or not, nor in a trend when their covariate is NaN.
    rng = np.random.default_rng(4)
    d = rng.choice([2.0, 3.0, 6.0], 300)
    v = 0.5 * rng.f(d, 6.0)
    v[:6] *= 30.0
    more_v = np.r_[v, np.nan, 0.7, 0.2, 0.9]
    more_d = np.r_[d, 4.0, 0.0, 0.005, 3.0]
    weights = np.r_[np.ones(300), 1.0, 1.0, 1.0, 0.0]
    for robust in (False, True):
        base = fit_f_distribution_likelihood(v, d, robust=robust)
        more = fit_f_distribution_likelihood(
            more_v, more_d, prior_weights=weights, robust=robust
        )
        assert more.scale == base.scale and more.df2 == base.df2
        np.testing.assert_array_equal(more.df2_shrunk[:300], base.df2_shrunk)
        np.testing.assert_array_equal(more.df2_shrunk[300:], base.df2)
        np.testing.assert_array_equal(more.df2_outlier, base.df2_outlier)
    assert np.isfinite(base.df2_outlier)  # the robust fit found outliers
    z = rng.uniform(0.0, 10.0, 300)
    base = fit_f_distribution_likelihood(v, d, covariate=z, robust=True)
    more = fit_f_distribution_likelihood(
        np.r_[v, np.nan, 0.7, 0.2],
        np.r_[d, 4.0, 0.005, 3.0],
        covariate=np.r_[z, np.nan, np.nan, np.nan],
        prior_weights=np.r_[np.ones(302), 0.0],
        robust=True,
    )
    assert more.df2 == base.df2
    np.testing.assert_array_equal(more.scale[:300], base.scale)
    assert np.isnan(more.scale[300:]).all()


def test_robust_likelihood_fit_leaves_out_variances_that_are_not_usable() -> None:
    # limma 3.62.1 enters them as zero variances; its fit of the usable variances
    # alone is CytePool's fit of all of them.
    data = reference("robust_unusable")
    fit = fit_f_distribution_likelihood(data["v"], data["d"], robust=True)
    keep = ~np.isnan(data["v"]) & (data["d"] >= 0.01)
    assert not keep.all()
    run = "likelihood_robust"
    check("robust_unusable", run, "usable scale", fit.scale, CLOSE)
    check("robust_unusable", run, "usable df2", fit.df2, CLOSE)
    check("robust_unusable", run, "usable df2.shrunk", fit.df2_shrunk[keep], CLOSE)
    check("robust_unusable", run, "usable df2.outlier", fit.df2_outlier, CLOSE)
    np.testing.assert_array_equal(fit.df2_shrunk[~keep], fit.df2)
    assert abs(fit.df2 / limma("robust_unusable", run, "df2", (1,))[0] - 1) > 1e-3


def test_robust_likelihood_fit_without_outliers_is_the_first_fit() -> None:
    v = 0.5 * distributions.f_ppf((np.arange(200) + 0.5) / 200, 4.0, 10.0)
    plain = fit_f_distribution_likelihood(v, 4.0)
    robust = fit_f_distribution_likelihood(v, 4.0, robust=True)
    assert (robust.scale, robust.df2) == (plain.scale, plain.df2)
    assert np.isnan(robust.df2_outlier) and np.isnan(plain.df2_outlier)
    np.testing.assert_array_equal(robust.df2_shrunk, plain.df2)


def test_robust_fit_without_genes_lacking_residual_df_is_limmas_of_the_others() -> None:
    # The deliberate difference: genes without residual df take no part in the
    # robust likelihood estimator's test for outliers, where limma counts them. With
    # a constant prior, CytePool's robust moderation of all genes is then limma's of
    # the other genes alone, and differs from limma's of all genes.
    case, run = "no_residual_df", "robust_default"
    eb = moderated(case, run)
    keep = reference(case)["keep"] == 1
    assert not keep.all() and eb.prior_estimator == "likelihood"
    prior_df = eb.prior_df[keep] if np.ndim(eb.prior_df) else eb.prior_df
    for key, ours in (
        ("s2.prior", eb.prior_variance),
        ("df.prior", prior_df),
        ("s2.post", eb.posterior_variance[keep]),
        ("df.total", eb.df_total[keep]),
        ("t", eb.t[keep]),
        ("p.value", eb.p_value[keep]),
    ):
        check(case, run, f"usable {key}", ours, CLOSE)
    everything = reference(case)[f"limma {run} df.prior"]
    everything = everything[keep] if everything.size > 1 else everything
    assert np.max(np.abs(np.asarray(prior_df) / everything - 1.0)) > 1e-3


@pytest.mark.parametrize("case", WEIGHTED_LOWESS_CASES)
def test_trend_smoother_matches_limmas_weighted_lowess(case: str) -> None:
    # Ties, windows narrower than 1e-7, windows whose x barely vary, two distant
    # clusters, and weights from 1e-8 to 100 as in the trend. The local fits are
    # closed forms, compared relative to max(|value|, 1).
    data = reference(f"weighted_lowess_{case}")
    x, y, w = data["lowess_x"], data["lowess_y"], data["lowess_w"]
    fitted = _weighted_lowess(x, y, w, float(data["span"][0]))
    assert_agrees(fitted, data["limma weighted_lowess fitted"], EXACT, floor=1.0)


def test_two_variances_above_zero_give_a_constant_fit_without_robust_stage() -> None:
    # As limma's fitFDistUnequalDF1 (found by probing; the two_positive fixture
    # checks it against limma): with two usable variances above zero, the others
    # zero, the location is constant whatever the covariate, and a robust fit is the
    # first fit. Variances that are not usable change neither, where limma returns
    # NaN.
    v = np.r_[0.5, 0.9, np.zeros(10)]
    d = np.r_[4.0, 2.0, np.full(10, 3.0)]
    z = np.arange(12.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # mostly zero variances
        plain = fit_f_distribution_likelihood(v, d)
        for robust in (False, True):
            fit = fit_f_distribution_likelihood(v, d, covariate=z, robust=robust)
            np.testing.assert_array_equal(fit.scale, plain.scale)
            assert fit.df2 == plain.df2 and np.isnan(fit.df2_outlier)
        more = fit_f_distribution_likelihood(
            np.r_[v, np.nan, 0.3], np.r_[d, 3.0, 0.004], covariate=np.r_[z, np.nan, 5.0]
        )
    np.testing.assert_array_equal(more.scale, plain.scale)
    assert more.df2 == plain.df2


def test_fit_object_fields_are_read_only() -> None:
    eb = moderated("two_groups", "trend")
    with pytest.raises(ValueError, match="read-only"):
        eb.t[0, 0] = 1.0
    assert "n_features=100" in repr(eb) and "prior_df=" in repr(eb)
    assert "prior_estimator='moments'" in repr(eb)
    assert "lfc=" in repr(treat(model_fit("two_groups")))


def test_exact_f_statistics_need_exact_standard_errors() -> None:
    data = reference("missing_values")
    fit = contrasts_fit(
        fit_linear_model(data["y"].T, data["X"]), np.eye(3)[:, 1:], method="limma"
    )
    assert not fit.exact_standard_errors
    with pytest.raises(ValueError, match="limma's approximation"):
        moderated_statistics(fit, f_statistic="exact")
    default = moderated_statistics(fit)
    np.testing.assert_array_equal(
        default.f_statistic, moderated_statistics(fit, f_statistic="limma").f_statistic
    )


FIT_DATA = reference("two_groups")


@pytest.mark.parametrize(
    ("function", "arguments", "keywords", "error", "message"),
    [
        (
            fit_f_distribution,
            ([0.1, -0.2], 4.0),
            {},
            ValueError,
            "non-negative and finite",
        ),
        (
            fit_f_distribution,
            ([0.1, np.inf], 4.0),
            {},
            ValueError,
            "negative or infinite",
        ),
        (fit_f_distribution, ([], 4.0), {}, ValueError, "at least one value"),
        (fit_f_distribution, ([[0.1]], 4.0), {}, ValueError, "1-D array"),
        (fit_f_distribution, (["a"], 4.0), {}, TypeError, "real numbers"),
        (
            fit_f_distribution,
            ([0.1, 0.2], [4.0, 4.0, 4.0]),
            {},
            ValueError,
            "one value per",
        ),
        (
            fit_f_distribution,
            ([0.1, 0.2], -1.0),
            {},
            ValueError,
            "finite and non-negative",
        ),
        (
            fit_f_distribution,
            ([np.nan, 0.2], [4.0, 0.0]),
            {},
            ValueError,
            "no variance is usable",
        ),
        (
            fit_f_distribution,
            ([0.1, 0.2, 0.3], 4.0),
            {"covariate": [1.0, np.nan, 2.0]},
            ValueError,
            "covariate is NaN for 1",
        ),
        (
            fit_f_distribution,
            ([0.1, 0.2, 0.3], 4.0),
            {"covariate": [1.0, 2.0]},
            ValueError,
            "one value per variance",
        ),
        (
            fit_f_distribution_robust,
            ([0.1, 0.2, 0.3], 4.0),
            {"winsor_tail_p": (0.05, 0.6)},
            ValueError,
            "strictly between 0 and 0.5",
        ),
        (
            fit_f_distribution_robust,
            ([0.1, 0.2, 0.3], 4.0),
            {"winsor_tail_p": (0.05, 0.1, 0.2)},
            ValueError,
            "one or two proportions",
        ),
        (
            squeeze_variances,
            ([0.1, 0.2, 0.3], 4.0),
            {"robust": 1},
            TypeError,
            "robust must be",
        ),
        (
            squeeze_variances,
            ([0.1, 0.2, 0.3], 4.0),
            {"prior_estimator": "unequal"},
            ValueError,
            "prior_estimator must be 'moments'.*'likelihood'.*or None",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], 4.0),
            {"prior_weights": [1.0, -1.0, 1.0]},
            ValueError,
            "prior_weights must be finite and non-negative",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], 4.0),
            {"prior_weights": [1.0, np.nan, 1.0]},
            ValueError,
            "prior_weights must be finite",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], 4.0),
            {"prior_weights": [1.0, 1.0]},
            ValueError,
            "one value per variance",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], 4.0),
            {"robust": "yes"},
            TypeError,
            "robust must be a bool",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, np.nan, 0.3], [4.0, 4.0, 0.005]),
            {},
            ValueError,
            "at least two usable variances above zero .* got 1",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], 4.0),
            {"prior_weights": [0.0, 0.0, 1.0]},
            ValueError,
            "at least two usable variances above zero .* got 1",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.5, 0.0, 0.0, 0.0], 4.0),
            {},
            ValueError,
            "at least two usable variances above zero .* got 1",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.0, 0.0, 0.0], 4.0),
            {},
            ValueError,
            "at least two usable variances above zero .* got 0",
        ),
        (
            fit_f_distribution_likelihood,
            ([0.1, 0.2, 0.3], [4.0, 4.0, 0.005]),
            {"covariate": [1.0, np.nan, 2.0]},
            ValueError,
            "NaN only where the variance is NaN, has fewer than 0.01 degrees",
        ),
        (
            squeeze_variances,
            ([0.1, np.nan, np.nan, 0.2], [4.0, 4.0, 3.0, 0.0]),
            {},
            ValueError,
            "squeezeVar stops here",
        ),
        (
            squeeze_variances,
            ([0.5, 0.0, 0.0, 0.0], [4.0, 4.0, 5.0, 3.0]),
            {},
            ValueError,
            "above zero .* got 1; limma's fitFDistUnequalDF1 returns NA",
        ),
    ],
)
@pytest.mark.filterwarnings("ignore:more than half of the variances:RuntimeWarning")
def test_hyperparameter_functions_reject_invalid_input(
    function: Any,
    arguments: tuple[Any, ...],
    keywords: dict[str, Any],
    error: type,
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        function(*arguments, **keywords)


@pytest.mark.parametrize(
    ("function", "keywords", "error", "message"),
    [
        (
            moderated_statistics,
            {"proportion": 0.0},
            ValueError,
            "strictly between 0 and 1",
        ),
        (moderated_statistics, {"proportion": True}, TypeError, "real number"),
        (
            moderated_statistics,
            {"stdev_coef_lim": (4.0, 0.1)},
            ValueError,
            "lower <= upper",
        ),
        (moderated_statistics, {"stdev_coef_lim": (0.1,)}, ValueError, "two values"),
        (
            moderated_statistics,
            {"stdev_coef_lim": (-0.1, 4.0)},
            ValueError,
            "0 <= lower",
        ),
        (
            moderated_statistics,
            {"trend": np.ones(3)},
            ValueError,
            "one covariate value per",
        ),
        (
            moderated_statistics,
            {"f_statistic": "approximate"},
            ValueError,
            "'exact', 'limma'",
        ),
        (treat, {"lfc": 1.0, "fc": 2.0}, ValueError, "not both"),
        (treat, {"fc": -1.0}, ValueError, "positive and finite"),
        (treat, {"lfc": np.nan}, ValueError, "finite"),
        (treat, {"robust": "yes"}, TypeError, "robust must be a bool"),
        (treat, {"prior_estimator": "fitFDist"}, ValueError, "prior_estimator must"),
        (
            moderated_statistics,
            {"prior_estimator": True},
            ValueError,
            "prior_estimator must be 'moments'",
        ),
    ],
)
def test_statistics_reject_invalid_input(
    function: Any, keywords: dict[str, Any], error: type, message: str
) -> None:
    with pytest.raises(error, match=message):
        function(model_fit("two_groups"), **keywords)


def test_statistics_need_a_fit_with_residual_df() -> None:
    with pytest.raises(TypeError, match="LinearModelFit"):
        moderated_statistics({"sigma": 1.0})  # type: ignore[arg-type]
    saturated = fit_linear_model(np.ones((2, 3)) + np.eye(2, 3), np.eye(2))
    with pytest.raises(ValueError, match="no feature has residual degrees of freedom"):
        moderated_statistics(saturated)
