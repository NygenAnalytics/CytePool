"""Tests for empirical Bayes moderation of feature-wise variances.

Reference values in ``tests/data/empirical_bayes`` come from limma's ``eBayes``,
``treat``, ``squeezeVar``, ``fitFDist`` and ``fitFDistRobustly`` (``legacy=TRUE``),
written by ``validation/empirical_bayes.py``. The other tests check properties that a
wrong implementation would break: limits, invariances and closed forms.
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
from cytepool.stats.special import digamma

DATA = Path(__file__).parent / "data" / "empirical_bayes"
EXACT = 1e-10
# limma solves the robust estimator's equation for the prior df only to a
# tolerance, which leaves its df.prior within about 1e-7 of the exact root, and
# everything downstream of it as close.
CLOSE = 1e-6
LFC = float(np.log2(1.2))
RUNS = {
    "constant": (False, False),
    "trend": (True, False),
    "robust": (False, True),
    "trend_robust": (True, True),
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
    trend, robust = RUNS[run]
    return moderated_statistics(
        model_fit(case), trend=trend, robust=robust, f_statistic=f_statistic
    )


MODEL_RUNS = [(case, run) for case in MODEL_CASES for run in runs(case)]
VARIANCE_RUNS = [(case, run) for case in VARIANCE_CASES for run in runs(case)]


@pytest.mark.parametrize(("case", "run"), MODEL_RUNS)
def test_moderated_statistics_match_limma(case: str, run: str) -> None:
    eb = moderated(case, run)
    tolerance = CLOSE if RUNS[run][1] else EXACT
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
    trend, robust = RUNS[run]
    fit = model_fit(case)
    result = treat(fit, lfc=LFC, trend=trend, robust=robust)
    tolerance = CLOSE if robust else EXACT
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
    tolerance = CLOSE if RUNS[run][1] else EXACT
    check(case, run, "means.F", exact.f_statistic, tolerance)
    check(case, run, "means.F.p.value", exact.f_p_value, tolerance)
    approximate = moderated(case, run)
    assert np.nanmax(np.abs(exact.f_statistic / approximate.f_statistic - 1)) > 1e-3


@pytest.mark.parametrize(("case", "run"), VARIANCE_RUNS)
def test_hyperparameters_match_limma(case: str, run: str) -> None:
    data = reference(case)
    tails = (0.1, 0.2) if run == "robust_wide" else (0.05, 0.1)
    robust = run != "moments"
    covariate = data.get("covariate")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        squeezed = squeeze_variances(
            data["v"],
            data["d"],
            covariate=covariate,
            robust=robust,
            winsor_tail_p=tails,
        )
        if robust:
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
    tolerance = CLOSE if robust else EXACT
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


@pytest.mark.parametrize("run", list(RUNS))
def test_results_do_not_depend_on_the_order_of_features(run: str) -> None:
    data = reference("two_groups")
    trend, robust = RUNS[run]
    order = np.random.default_rng(5).permutation(data["y"].shape[0])
    permuted = moderated_statistics(
        fit_linear_model(data["y"][order].T, data["X"]), trend=trend, robust=robust
    )
    eb = moderated_statistics(
        fit_linear_model(data["y"].T, data["X"]), trend=trend, robust=robust
    )
    for name in ("posterior_variance", "df_total", "t", "p_value", "lods"):
        np.testing.assert_allclose(
            getattr(permuted, name), getattr(eb, name)[order], rtol=1e-10
        )
    np.testing.assert_allclose(
        permuted.coefficient_prior_variance, eb.coefficient_prior_variance, rtol=1e-12
    )


@pytest.mark.parametrize("run", list(RUNS))
def test_rescaling_the_data_rescales_the_variances(run: str) -> None:
    # Multiplying expression by c multiplies variances by c^2 (and the mean
    # expression, the trend covariate, by c); the tests do not change.
    data = reference("two_groups")
    trend, robust = RUNS[run]
    eb = moderated_statistics(
        fit_linear_model(data["y"].T, data["X"]), trend=trend, robust=robust
    )
    # stdev_coef_lim bounds standard deviations in the units of the data
    scaled = moderated_statistics(
        fit_linear_model(4.0 * data["y"].T, data["X"]),
        trend=trend,
        robust=robust,
        stdev_coef_lim=(0.4, 16.0),
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
    for trend, robust in RUNS.values():
        eb = moderated_statistics(
            fit_linear_model(y.T, data["X"]), trend=trend, robust=robust
        )
        base = moderated_statistics(
            fit_linear_model(data["y"].T, data["X"]), trend=trend, robust=robust
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


def test_fit_object_fields_are_read_only() -> None:
    eb = moderated("two_groups", "trend")
    with pytest.raises(ValueError, match="read-only"):
        eb.t[0, 0] = 1.0
    assert "n_features=100" in repr(eb) and "prior_df=" in repr(eb)
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
            "fitFDistUnequalDF1",
        ),
    ],
)
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
