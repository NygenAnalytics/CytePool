"""Tests for linear mixed models with random intercepts and their Satterthwaite tests.

Reference values in ``tests/data/mixed_models`` come from lme4's ``lmer`` (REML) and
lmerTest's ``summary`` and ``contest1D``, written by ``validation/mixed_models.py``.
Balanced designs have closed-form REML estimates, those of the analysis of variance,
and closed-form Satterthwaite degrees of freedom; a fit whose variance components are
all zero is the weighted least squares fit of ``fit_linear_model``.
"""

import math
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.design import Design, make_contrasts
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.mixed_models import (
    FixedEffectTests,
    MixedModelFit,
    fit_mixed_model,
    fixed_effect_tests,
)

DATA = Path(__file__).parent / "data" / "mixed_models"
CASES = (
    "one_intercept",
    "unbalanced",
    "crossed",
    "nested",
    "boundary",
    "observation_weights",
    "weights_crossed",
    "large_ratios",
)
# Closed-form results.
EXACT = 1e-10
# lme4's optimiser converges to about 1e-8 with the tolerances the validation script
# sets: the close tier for iterative fits.
CLOSE = 1e-6
# lmerTest differentiates the deviance numerically in theta, on which it depends
# through theta^2 only, so its numerical Hessian loses relative precision like
# 1 / theta^2 for a small variance (validation/mixed_models.py, SATTERTHWAITE).
SATTERTHWAITE = 1e-5
# Where theta is large, lme4 places it to about 1e-6 only: the criterion is nearly
# flat in it (validation/mixed_models.py, LARGE_RATIOS).
OPTIMUM = {"large_ratios": 1e-5}


@cache
def reference(case: str) -> dict[str, Any]:
    with np.load(DATA / f"{case}.npz") as data:
        return {key: data[key] for key in data.files}


def reference_design(data: dict[str, Any]) -> Design:
    samples = {
        key.removeprefix("sample_"): value
        for key, value in data.items()
        if key.startswith("sample_")
    }
    return Design(samples, str(data["formula"]), random=str(data["random"]))


@cache
def fitted(case: str) -> MixedModelFit:
    data = reference(case)
    return fit_mixed_model(
        data["y"], reference_design(data), weights=data.get("weights")
    )


def assert_relative(
    actual: Any, desired: Any, tolerance: float, floor: Any = 0.0
) -> None:
    """|actual - desired| <= tolerance * max(|desired|, floor), NaN where desired is."""
    actual, desired = np.asarray(actual), np.asarray(desired)
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(desired))
    scale = np.fmax(np.abs(desired), floor)
    finite = ~np.isnan(desired)
    excess = np.abs(actual - desired)[finite] - tolerance * scale[finite]
    assert np.all(excess <= 0.0), float(np.max(excess))


def average_residual(data: dict[str, Any]) -> np.ndarray:
    """lme4's residual variance over the mean weight: that of an average observation,
    which does not depend on the scale of the weights."""
    weights = data.get("weights")
    mean_weight = 1.0 if weights is None else weights.mean(axis=0)
    return (data["sigma2"] / mean_weight)[:, None]


def group_codes(design: Design) -> list[np.ndarray]:
    """The level of each sample for every grouping factor, from the sample table."""
    codes = []
    for effect in design.random:
        labels = [design.samples[name].astype(str) for name in effect.grouping]
        joined = np.array(["\t".join(parts) for parts in zip(*labels)])
        codes.append(np.unique(joined, return_inverse=True)[1])
    return codes


def p_value_scale(p: np.ndarray) -> np.ndarray:
    """max(p, 1e-300) max(1, |log p|): in a tail, log p moves by about t^2 times the
    relative change of t, so p-values are compared on the scale of log p."""
    bounded = np.fmax(p, 1e-300)
    return bounded * np.fmax(1.0, np.abs(np.log(bounded)))


@pytest.mark.parametrize("case", CASES)
def test_the_optimum_matches_lme4(case: str) -> None:
    data = reference(case)
    fit = fitted(case)
    assert fit.converged.all()
    # lme4 from its own start, with tightened tolerances: the criterion at the
    # minimum is flat to first order, so it agrees to the exact tier, and no fit stops
    # above lme4's minimum.
    assert_relative(fit.reml_criterion, data["criterion"], EXACT)
    criterion = data["criterion"]
    assert np.all(fit.reml_criterion <= criterion + EXACT * np.abs(criterion))
    close = OPTIMUM.get(case, CLOSE)
    assert_relative(fit.coefficients, data["coefficients"], close, data["se"])
    assert_relative(fixed_effect_tests(fit).standard_error, data["se"], close)
    # The optimisers locate theta to an absolute precision, so a small component has
    # little relative precision; components are compared relative to the larger of
    # themselves and the residual variance of an observation of average weight.
    assert_relative(
        fit.variance_components, data["components"], close, average_residual(data)
    )
    assert_relative(fit.residual_variance, data["sigma2"], close)
    # lme4 stops near a theta of zero rather than at it; where it does reach zero, so
    # does CytePool.
    assert fit.boundary[data["theta"] == 0.0].all()


@pytest.mark.parametrize("case", CASES)
def test_the_tests_match_lmertest_at_the_same_estimates(case: str) -> None:
    data = reference(case)
    fit = fitted(case)
    tests = fixed_effect_tests(fit)
    # lme4 evaluated at CytePool's estimates: closed forms there.
    assert_relative(fit.reml_criterion, data["given_criterion"], EXACT)
    assert_relative(
        fit.coefficients, data["given_coefficients"], EXACT, data["given_se"]
    )
    assert_relative(tests.standard_error, data["given_se"], EXACT)
    assert_relative(tests.t, data["t"], EXACT, 1.0)
    # lmerTest differentiates numerically.
    assert_relative(tests.df, data["df"], SATTERTHWAITE)
    assert_relative(tests.p_value, data["p"], SATTERTHWAITE, p_value_scale(data["p"]))
    contrast = fixed_effect_tests(fit, data["contrast_weights"])
    expected = data["contrast"]
    assert_relative(contrast.estimate[:, 0], expected[:, 0], EXACT, expected[:, 1])
    assert_relative(contrast.standard_error[:, 0], expected[:, 1], EXACT)
    assert_relative(contrast.t[:, 0], expected[:, 3], EXACT, 1.0)
    assert_relative(contrast.df[:, 0], expected[:, 2], SATTERTHWAITE)
    assert_relative(
        contrast.p_value[:, 0],
        expected[:, 4],
        SATTERTHWAITE,
        p_value_scale(expected[:, 4]),
    )


def test_the_reference_cases_cover_the_boundary_and_both_kinds_of_grouping() -> None:
    # Without fits at the boundary, or with only one grouping factor, the comparison
    # with lme4 would leave the boundary rule of the degrees of freedom untested.
    assert np.count_nonzero(fitted("boundary").boundary) >= 4
    for case in ("crossed", "nested", "weights_crossed"):
        fit = fitted(case)
        assert len(fit.component_names) == 2
        partly = fit.boundary.any(axis=1) & ~fit.boundary.all(axis=1)
        assert partly.any(), case
    assert fitted("nested").component_names == ("donor", "sample:donor")


# Closed forms.


def one_way(n_groups: int, n_replicates: int, seed: int) -> tuple[Design, np.ndarray]:
    """A balanced one-way layout and values with group effects of several sizes.

    The last feature has the same mean in every group, so its fit lies on the
    boundary.
    """
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"G{i}" for i in range(n_groups)], n_replicates)
    design = Design({"group": groups}, "~ 1", random="(1 | group)")
    codes = np.repeat(np.arange(n_groups), n_replicates)
    spread = np.array([0.0, 0.3, 1.0, 3.0])
    y = (
        5.0
        + rng.normal(size=(n_groups, spread.size))[codes] * spread
        + rng.normal(size=(codes.size, spread.size))
    )
    flat = y[:, -1] - y[:, -1].reshape(n_groups, n_replicates).mean(axis=1)[codes]
    return design, np.column_stack([y, flat + 5.0])


def test_a_balanced_one_way_design_has_the_estimates_of_the_analysis_of_variance() -> (
    None
):
    q, m = 6, 4
    n = q * m
    design, y = one_way(q, m, seed=11)
    fit = fit_mixed_model(y, design)
    tests = fixed_effect_tests(fit)
    means = y.reshape(q, m, -1).mean(axis=1)
    grand = y.mean(axis=0)
    between = m * np.sum((means - grand) ** 2, axis=0) / (q - 1)
    within = np.sum((y.reshape(q, m, -1) - means[:, None]) ** 2, axis=(0, 1)) / (
        q * (m - 1)
    )
    interior = between > within
    assert interior.sum() >= 2 and not interior[-1]
    # Inside: sigma^2 = MSW, sigma_b^2 = (MSB - MSW) / m, var(mean) = MSB / n on q - 1
    # degrees of freedom; at the boundary, one variance SST / (n - 1) on n - 1.
    pooled = (between * (q - 1) + within * q * (m - 1)) / (n - 1)
    residual = np.where(interior, within, pooled)
    component = np.where(interior, (between - within) / m, 0.0)
    np.testing.assert_allclose(fit.residual_variance, residual, rtol=EXACT)
    assert_relative(fit.variance_components[:, 0], component, EXACT, residual)
    np.testing.assert_array_equal(fit.boundary[:, 0], ~interior)
    np.testing.assert_allclose(fit.coefficients[:, 0], grand, rtol=EXACT)
    variance = np.where(interior, between, pooled) / n
    np.testing.assert_allclose(tests.standard_error[:, 0] ** 2, variance, rtol=EXACT)
    np.testing.assert_allclose(
        tests.df[:, 0], np.where(interior, q - 1, n - 1), rtol=EXACT
    )
    # The REML criterion: -2 log L of the group means and of the contrasts within
    # groups, with (q - 1) MSB and q (m - 1) MSW their sums of squares.
    criterion = (
        np.where(
            interior,
            (q - 1) * np.log(between) + q * (m - 1) * np.log(within),
            (n - 1) * np.log(pooled),
        )
        + (n - 1) * (1.0 + math.log(2.0 * math.pi))
        + math.log(n)
    )
    np.testing.assert_allclose(fit.reml_criterion, criterion, rtol=EXACT)


def test_a_randomised_block_design_has_the_classical_satterthwaite_df() -> None:
    # Donors as random blocks and each treatment once per donor: the mean of a
    # treatment combines the mean squares of donors (MSD, q - 1 df) and of the
    # residual (MSE, (q - 1)(t - 1) df), the classical case of Satterthwaite (1946).
    q, t = 7, 3
    rng = np.random.default_rng(5)
    donors = np.repeat([f"D{i}" for i in range(q)], t)
    treatment = np.tile(["a", "b", "c"], q)
    design = Design(
        {"donor": donors, "treatment": treatment},
        "~ 0 + treatment",
        random="(1 | donor)",
    )
    effects = np.array([4.0, 5.0, 4.5])
    y = (
        effects[np.tile(np.arange(t), q)][:, None]
        + rng.normal(0.0, 2.0, (q, 5))[np.repeat(np.arange(q), t)]
        + rng.normal(0.0, 0.5, (q * t, 5))
    )
    fit = fit_mixed_model(y, design)
    assert not fit.boundary.any()
    cells = y.reshape(q, t, -1)
    donor_means = cells.mean(axis=1)
    treatment_means = cells.mean(axis=0)
    grand = cells.mean(axis=(0, 1))
    msd = t * np.sum((donor_means - grand) ** 2, axis=0) / (q - 1)
    interaction = cells - donor_means[:, None] - treatment_means[None] + grand
    mse = np.sum(interaction**2, axis=(0, 1)) / ((q - 1) * (t - 1))
    np.testing.assert_allclose(fit.residual_variance, mse, rtol=EXACT)
    np.testing.assert_allclose(
        fit.variance_components[:, 0], (msd - mse) / t, rtol=EXACT
    )
    np.testing.assert_allclose(fit.coefficients, treatment_means.T, rtol=EXACT)
    tests = fixed_effect_tests(fit, {"a": [1.0, 0, 0], "b_vs_a": [-1.0, 1.0, 0]})
    combined = msd + (t - 1) * mse
    np.testing.assert_allclose(
        tests.standard_error[:, 0] ** 2, combined / (t * q), rtol=EXACT
    )
    expected = combined**2 / (
        msd**2 / (q - 1) + ((t - 1) * mse) ** 2 / ((q - 1) * (t - 1))
    )
    np.testing.assert_allclose(tests.df[:, 0], expected, rtol=EXACT)
    # A difference of treatments lies within donors: only MSE.
    np.testing.assert_allclose(tests.standard_error[:, 1] ** 2, 2 * mse / q, rtol=EXACT)
    np.testing.assert_allclose(tests.df[:, 1], (q - 1) * (t - 1), rtol=EXACT)


def test_large_variance_ratios_keep_the_estimates_of_the_analysis_of_variance() -> None:
    # Donor standard deviations 1e2 to 1e4 times the residual one. The cross-products
    # of the absorbed factor are taken about its levels' means, so the estimates keep
    # the closed form of the balanced design to rounding.
    q, m = 8, 3
    codes = np.repeat(np.arange(q), m)
    design = Design({"group": codes.astype(str)}, "~ 1", random="(1 | group)")
    rng = np.random.default_rng(17)
    spread = np.array([1e2, 1e3, 1e4])
    y = 3.0 + rng.normal(size=(q, 3))[codes] * spread + rng.normal(size=(q * m, 3))
    fit = fit_mixed_model(y, design)
    assert fit.converged.all()
    means = y.reshape(q, m, -1).mean(axis=1)
    between = m * np.sum((means - y.mean(axis=0)) ** 2, axis=0) / (q - 1)
    within = np.sum((y.reshape(q, m, -1) - means[:, None]) ** 2, axis=(0, 1)) / (
        q * (m - 1)
    )
    # Means of values 1e4 times their spread within groups lose four digits.
    np.testing.assert_allclose(fit.residual_variance, within, rtol=1e-9)
    np.testing.assert_allclose(
        fit.variance_components[:, 0], (between - within) / m, rtol=EXACT
    )


def test_a_design_without_fixed_effects_maximises_the_likelihood() -> None:
    # With no fixed effects REML is maximum likelihood. In a balanced one-way layout
    # the group means and the deviations from them are independent, with variances
    # lambda = sigma^2 + m sigma_b^2 and sigma^2: lambda = (sum of squared group means
    # times m) / q and sigma^2 = MSW at the optimum.
    q, m = 6, 4
    n = q * m
    design, y = one_way(q, m, seed=23)
    design = Design(design.samples, "~ 0", random="(1 | group)")
    fit = fit_mixed_model(y[:, 2:4], design)
    values = y[:, 2:4]
    means = values.reshape(q, m, -1).mean(axis=1)
    lam = m * np.sum(means**2, axis=0) / q
    within = np.sum((values.reshape(q, m, -1) - means[:, None]) ** 2, axis=(0, 1)) / (
        q * (m - 1)
    )
    assert fit.coefficients.shape == (2, 0) and not fit.boundary.any()
    np.testing.assert_allclose(fit.residual_variance, within, rtol=EXACT)
    np.testing.assert_allclose(
        fit.variance_components[:, 0], (lam - within) / m, rtol=EXACT
    )
    criterion = (
        n * (1.0 + math.log(2.0 * math.pi))
        + q * np.log(lam)
        + q * (m - 1) * np.log(within)
    )
    np.testing.assert_allclose(fit.reml_criterion, criterion, rtol=EXACT)


def boundary_values(
    design: Design, weights: np.ndarray, seed: int, n_features: int = 3
) -> np.ndarray:
    """Values whose residuals are orthogonal, in the weights' inner product, to the
    fixed effects and to every grouping: at zero variance components the gradient of
    the criterion points outward, so the REML estimates are zero."""
    rng = np.random.default_rng(seed)
    x = design.matrix
    indicators = [np.eye(codes.max() + 1)[codes] for codes in group_codes(design)]
    spanned = np.column_stack([x, *indicators])
    noise = rng.normal(size=(x.shape[0], n_features))
    projection = (
        spanned
        @ np.linalg.lstsq(
            spanned * weights[:, None] ** 0.5,
            noise * weights[:, None] ** 0.5,
            rcond=None,
        )[0]
    )
    return x @ rng.normal(size=(x.shape[1], n_features)) + noise - projection


def test_variance_components_at_zero_give_the_weighted_least_squares_fit() -> None:
    rng = np.random.default_rng(3)
    n = 30
    samples = {
        "donor": np.repeat([f"D{i}" for i in range(6)], 5),
        "batch": np.tile(["B1", "B2", "B3"], 10),
        "condition": np.tile(["ctrl", "stim"], 15),
        "x": rng.normal(size=n),
    }
    design = Design(samples, "~ condition + x", random="(1 | donor) + (1 | batch)")
    weights = np.exp(rng.normal(0.0, 0.5, n))
    y = boundary_values(design, weights, seed=4)
    fit = fit_mixed_model(y, design, weights=weights)
    assert fit.boundary.all() and fit.converged.all()
    np.testing.assert_array_equal(fit.variance_components, 0.0)
    # fit_linear_model is validated against limma separately.
    least_squares = fit_linear_model(y, design.model_matrix, weights=weights)
    np.testing.assert_allclose(fit.coefficients, least_squares.coefficients, rtol=EXACT)
    np.testing.assert_allclose(
        fit.residual_variance, least_squares.sigma**2, rtol=EXACT
    )
    standard_error = least_squares.stdev_unscaled * least_squares.sigma[:, None]
    tests = fixed_effect_tests(fit)
    np.testing.assert_allclose(tests.standard_error, standard_error, rtol=EXACT)
    # With every component at zero the degrees of freedom are those of least
    # squares, n - p, as in lmerTest.
    np.testing.assert_allclose(tests.df, n - 3, rtol=EXACT)
    np.testing.assert_array_equal(fit.variance_parameter_covariance[:, :2], 0.0)


# Invariants.


@cache
def crossed_fit() -> tuple[Design, np.ndarray, np.ndarray, MixedModelFit]:
    data = reference("weights_crossed")
    design = reference_design(data)
    return design, data["y"], data["weights"], fitted("weights_crossed")


def test_reparametrising_the_fixed_effects_changes_only_the_coefficients() -> None:
    data = reference("one_intercept")
    design = reference_design(data)
    fit = fitted("one_intercept")
    # Group means instead of a reference level, and age in months: x_other = x t.
    samples = dict(design.samples.to_dict(), age_months=12.0 * data["sample_age"])
    other = Design(samples, "~ 0 + condition + age_months", random="(1 | donor)")
    refit = fit_mixed_model(data["y"], other)
    # Two fits of the same model that converge to rounding by different arithmetic.
    np.testing.assert_allclose(
        refit.variance_components, fit.variance_components, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(
        refit.residual_variance, fit.residual_variance, rtol=1e-9
    )
    np.testing.assert_array_equal(refit.boundary, fit.boundary)
    # The criterion moves by 2 log |det t| = 2 log 12.
    t = np.linalg.lstsq(design.matrix, other.matrix, rcond=None)[0]
    np.testing.assert_allclose(abs(np.linalg.det(t)), 12.0, rtol=EXACT)
    np.testing.assert_allclose(
        refit.reml_criterion, fit.reml_criterion + 2.0 * math.log(12.0), rtol=1e-9
    )
    first = fixed_effect_tests(fit, {"stim": [0.0, 1.0, 0.0], "age": [0.0, 0.0, 1.0]})
    second = fixed_effect_tests(
        refit, {"stim": [-1.0, 1.0, 0.0], "age": [0.0, 0.0, 12.0]}
    )
    for name in ("estimate", "standard_error", "t", "df", "p_value"):
        np.testing.assert_allclose(
            getattr(second, name), getattr(first, name), rtol=1e-8, err_msg=name
        )


def test_relabelling_groups_and_reordering_samples_change_nothing() -> None:
    design, y, weights, fit = crossed_fit()
    order = np.random.default_rng(2).permutation(y.shape[0])
    samples = {
        "donor": np.array(["donor " + str(d)[::-1] for d in design.samples["donor"]]),
        "batch": np.array(["x" + str(b) for b in design.samples["batch"]]),
        "condition": design.samples["condition"],
    }
    shuffled = Design(
        {name: np.asarray(values)[order] for name, values in samples.items()},
        "~ condition",
        random="(1 | donor) + (1 | batch)",
    )
    refit = fit_mixed_model(y[order], shuffled, weights=weights[order])
    for name in (
        "coefficients",
        "cov_coefficients",
        "variance_components",
        "residual_variance",
        "reml_criterion",
        "variance_parameter_covariance",
        "cov_coefficients_derivatives",
    ):
        # Sums taken in another order differ in rounding.
        np.testing.assert_allclose(
            getattr(refit, name),
            getattr(fit, name),
            rtol=1e-9,
            atol=1e-13,
            err_msg=name,
        )
    np.testing.assert_array_equal(refit.boundary, fit.boundary)


def test_scaling_the_weights_scales_only_the_residual_variance() -> None:
    design, y, weights, fit = crossed_fit()
    refit = fit_mixed_model(y, design, weights=7.0 * weights)
    np.testing.assert_allclose(
        refit.residual_variance, 7.0 * fit.residual_variance, rtol=1e-9
    )
    for name in (
        "coefficients",
        "cov_coefficients",
        "variance_components",
        "reml_criterion",
    ):
        np.testing.assert_allclose(
            getattr(refit, name),
            getattr(fit, name),
            rtol=1e-9,
            atol=1e-13,
            err_msg=name,
        )
    np.testing.assert_allclose(
        fixed_effect_tests(refit).df, fixed_effect_tests(fit).df, rtol=1e-8
    )


def test_an_affine_change_of_the_values_carries_through() -> None:
    # y' = a y + x b: variances scale by a^2, coefficients by a and shift by b, the
    # criterion moves by (n - p) log a^2, and the t-statistics and df stay.
    design, y, weights, fit = crossed_fit()
    a = 2.5
    b = np.array([3.0, 0.0])
    refit = fit_mixed_model(
        a * y + (design.matrix @ b)[:, None], design, weights=weights
    )
    np.testing.assert_allclose(
        refit.variance_components, a**2 * fit.variance_components, rtol=1e-9, atol=1e-12
    )
    np.testing.assert_allclose(refit.coefficients, a * fit.coefficients + b, rtol=1e-9)
    n, p = design.matrix.shape
    np.testing.assert_allclose(
        refit.reml_criterion, fit.reml_criterion + (n - p) * math.log(a**2), rtol=1e-9
    )
    first, second = fixed_effect_tests(fit), fixed_effect_tests(refit)
    np.testing.assert_allclose(second.df, first.df, rtol=1e-8)
    np.testing.assert_allclose(second.t[:, 1], first.t[:, 1], rtol=1e-8)


def test_results_depend_neither_on_threads_nor_on_the_other_features() -> None:
    design, y, weights, fit = crossed_fit()
    serial = fit_mixed_model(y, design, weights=weights, n_threads=1)
    threaded = fit_mixed_model(y, design, weights=weights, n_threads=3)
    alone = fit_mixed_model(y[:, 3:5], design, weights=weights[:, 3:5])
    for name in ("coefficients", "variance_components", "reml_criterion"):
        np.testing.assert_array_equal(getattr(threaded, name), getattr(serial, name))
        # The projection onto the fixed effects of fewer features may round
        # differently in BLAS.
        np.testing.assert_allclose(
            getattr(alone, name), getattr(fit, name)[3:5], rtol=1e-12, atol=1e-15
        )
    np.testing.assert_array_equal(threaded.iterations, serial.iterations)


# Contrasts and the fit's record.


def test_named_contrasts_of_a_design_are_matched_by_name() -> None:
    data = reference("one_intercept")
    design = reference_design(data).with_contrasts(
        {"stim_vs_ctrl": "conditionstim", "stim_plus_age": "conditionstim + age"}
    )
    fit = fit_mixed_model(data["y"], design)
    tests = fixed_effect_tests(fit, design.contrasts)
    assert tests.contrast_names == ("stim_vs_ctrl", "stim_plus_age")
    by_coefficient = fixed_effect_tests(fit)
    np.testing.assert_allclose(tests.t[:, 0], by_coefficient.t[:, 1], rtol=EXACT)
    np.testing.assert_allclose(tests.df[:, 0], by_coefficient.df[:, 1], rtol=EXACT)
    # Contrasts made over the columns in another order are matched by name.
    reordered = make_contrasts(
        {"stim_vs_ctrl": "conditionstim", "stim_plus_age": "age + conditionstim"},
        fit.coefficient_names[::-1],
    )
    again = fixed_effect_tests(fit, reordered)
    np.testing.assert_allclose(again.estimate, tests.estimate, rtol=EXACT)
    np.testing.assert_allclose(again.df, tests.df, rtol=EXACT)
    np.testing.assert_allclose(
        tests.estimate[:, 1],
        fit.coefficients[:, 1] + fit.coefficients[:, 2],
        rtol=EXACT,
    )


def test_the_fit_and_its_tests_are_read_only_records() -> None:
    fit = fitted("crossed")
    assert fit.component_names == ("donor", "batch")
    assert fit.coefficient_names == ("(Intercept)", "conditionstim")
    assert fit.cov_coefficients_derivatives.shape == (
        fit.coefficients.shape[0],
        3,
        2,
        2,
    )
    assert repr(fit).startswith("MixedModelFit(n_features=12, n_samples=40")
    tests = fixed_effect_tests(fit)
    assert isinstance(tests, FixedEffectTests) and tests.fit is fit
    assert repr(tests).endswith("df_method='satterthwaite')")
    for array in (fit.coefficients, fit.boundary, tests.df, tests.contrasts):
        with pytest.raises(ValueError, match="read-only"):
            array[0] = 0
    copy = pickle.loads(pickle.dumps(tests))
    np.testing.assert_array_equal(copy.p_value, tests.p_value)


def test_the_derivatives_of_the_covariance_are_those_of_its_definition() -> None:
    # C(phi) = (x' S^-1 x)^-1 with S = sum of sigma_k^2 Z_k Z_k' + sigma^2 W^-1,
    # differentiated by central differences, relative step 1e-5: the truncation error
    # is about 1e-10 and the rounding error about 1e-11.
    design, y, weights, fit = crossed_fit()
    x = design.matrix
    indicators = [np.eye(codes.max() + 1)[codes] for codes in group_codes(design)]
    gene = int(np.flatnonzero(~fit.boundary.any(axis=1))[0])
    parameters = np.append(fit.variance_components[gene], fit.residual_variance[gene])
    pieces = [z @ z.T for z in indicators] + [np.diag(1.0 / weights[:, gene])]

    def covariance(values: np.ndarray) -> np.ndarray:
        s = sum(value * piece for value, piece in zip(values, pieces))
        result: np.ndarray = np.linalg.inv(x.T @ np.linalg.solve(s, x))
        return result

    np.testing.assert_allclose(
        covariance(parameters), fit.cov_coefficients[gene], rtol=1e-10
    )
    for j, value in enumerate(parameters):
        step = 1e-5 * value
        upper, lower = parameters.copy(), parameters.copy()
        upper[j] += step
        lower[j] -= step
        derivative = (covariance(upper) - covariance(lower)) / (2.0 * step)
        np.testing.assert_allclose(
            fit.cov_coefficients_derivatives[gene, j],
            derivative,
            rtol=1e-7,
            atol=1e-7 * np.abs(derivative).max(),
            err_msg=str(j),
        )


# Edge cases and errors.


def test_a_feature_without_residual_variation_is_not_fitted() -> None:
    data = reference("one_intercept")
    design = reference_design(data)
    exact = design.matrix @ np.array([3.0, 1.0, 0.01])
    y = np.column_stack([data["y"][:, 0], np.full(design.n_samples, 4.0), exact])
    fit = fit_mixed_model(y, design)
    np.testing.assert_array_equal(fit.converged, [True, False, False])
    np.testing.assert_array_equal(fit.iterations, [fit.iterations[0], 0, 0])
    assert (
        np.isnan(fit.coefficients[1:]).all() and np.isnan(fit.reml_criterion[1:]).all()
    )
    assert not fit.boundary[1:].any()
    tests = fixed_effect_tests(fit)
    assert np.isnan(tests.p_value[1:]).all() and np.isfinite(tests.p_value[0]).all()
    np.testing.assert_allclose(
        fit.coefficients[0], fitted("one_intercept").coefficients[0], rtol=1e-12
    )


def test_a_feature_the_effects_reproduce_is_not_fitted() -> None:
    # Constant within donors, or the sum of a donor and a batch effect: the fixed and
    # random effects reproduce the values, and the REML criterion decreases without
    # bound as the residual variance goes to zero. lme4 fails to converge on such
    # features. Noise of 1e-3 within donors is a feature to fit.
    rng = np.random.default_rng(29)
    donor = np.repeat(np.arange(8), 4)
    batch = np.tile(np.arange(4), 8)
    samples = {"donor": donor.astype(str), "batch": batch.astype(str)}
    one = Design(samples, "~ 1", random="(1 | donor)")
    two = Design(samples, "~ 1", random="(1 | donor) + (1 | batch)")
    donor_effect = rng.normal(5.0, 1.0, 8)[donor]
    batch_effect = rng.normal(0.0, 1.0, 4)[batch]
    noise = rng.normal(size=32)
    fit = fit_mixed_model(
        np.column_stack([donor_effect, donor_effect + 1e-9 * noise]), one
    )
    np.testing.assert_array_equal(fit.converged, False)
    assert np.isnan(fit.variance_components).all()
    fit = fit_mixed_model(
        np.column_stack(
            [donor_effect + batch_effect, donor_effect + batch_effect + 1e-3 * noise]
        ),
        two,
    )
    np.testing.assert_array_equal(fit.converged, [False, True])
    assert np.isnan(fit.reml_criterion[0])
    np.testing.assert_allclose(np.sqrt(fit.residual_variance[1]), 1e-3, rtol=0.5)


def samples_for_errors() -> dict[str, Any]:
    return {
        "donor": np.repeat(["D1", "D2", "D3", "D4"], 3),
        "batch": np.tile(["B1", "B2", "B3"], 4),
        "condition": np.tile(["ctrl", "stim", "stim"], 4),
        "x": np.arange(12.0),
        "one": ["S"] * 12,
        "sample": [f"S{i}" for i in range(12)],
    }


@pytest.mark.parametrize(
    ("formula", "random", "error", "message"),
    [
        ("~ condition", None, ValueError, r"the design has no random effects"),
        (
            "~ condition",
            "(x | donor)",
            ValueError,
            r"fits random intercepts, such as \(1 \| donor\), but the random effect "
            r"\(x \| donor\) has random slopes",
        ),
        (
            "~ condition",
            "(1 | one)",
            ValueError,
            r"random_single_group|has a single group",
        ),
        (
            "~ condition",
            "(1 | sample)",
            ValueError,
            r"cannot be estimated: every group of the random effect \(1 \| sample\)",
        ),
        (
            "~ condition + donor",
            "(1 | donor)",
            ValueError,
            r"cannot be estimated: the fixed effects already separate every group",
        ),
        (
            "~ condition + x + batch",
            "(1 | donor)",
            ValueError,
            r"the fixed effects are not of full column rank: column\(s\) 'batchB2', "
            r"'batchB3' are linear combinations",
        ),
    ],
    ids=[
        "fixed_only",
        "slope",
        "one_group",
        "one_sample_each",
        "confounded",
        "aliased",
    ],
)
def test_designs_that_cannot_be_fitted_are_refused(
    formula: str, random: str | None, error: type, message: str
) -> None:
    samples = samples_for_errors()
    if formula == "~ condition + x + batch":
        samples["x"] = (np.asarray(samples["batch"]) == "B3").astype(float)
    design = Design(samples, formula, random=random)
    with pytest.raises(error, match=message):
        fit_mixed_model(np.zeros((12, 2)), design)


def test_a_design_whose_effects_span_every_sample_is_refused() -> None:
    # Each random effect passes the design checks on its own, but with a covariate
    # the fixed effects and both groupings span every sample, so they reproduce every
    # feature exactly. lme4 fits such a design; fit_mixed_model refuses it.
    samples = {
        "donor": ["D1", "D1", "D2", "D2", "D3", "D3"],
        "batch": ["B1", "B2", "B2", "B3", "B3", "B1"],
        "x": [0.3, 1.2, -0.5, 2.0, 0.1, 0.8],
    }
    design = Design(samples, "~ x", random="(1 | donor) + (1 | batch)")
    assert design.check().ok
    with pytest.raises(
        ValueError,
        match=r"together span every sample \(6 dimensions for 6 samples\), so they "
        r"reproduce every feature exactly; fit_mixed_model does not fit such designs",
    ):
        fit_mixed_model(np.ones((6, 1)), design)


def test_invalid_values_and_weights_are_refused() -> None:
    data = reference("one_intercept")
    design = reference_design(data)
    y = data["y"]
    with pytest.raises(TypeError, match="design must be a Design with random effects"):
        fit_mixed_model(y, design.matrix)
    with pytest.raises(
        ValueError, match=r"y has 23 rows but the design has 24 samples"
    ):
        fit_mixed_model(y[1:], design)
    with pytest.raises(ValueError, match=r"y must be a 2-D array"):
        fit_mixed_model(y[:, 0], design)
    missing = y.copy()
    missing[3, 1] = np.nan
    with pytest.raises(ValueError, match=r"it has 1 missing \(NaN\) and 0 infinite"):
        fit_mixed_model(missing, design)
    with pytest.raises(TypeError, match="y must hold real numbers"):
        fit_mixed_model(y > 5.0, design)
    weights = np.ones_like(y)
    weights[0, 0] = 0.0
    with pytest.raises(
        ValueError, match=r"observation weights must be finite and positive; 1"
    ):
        fit_mixed_model(y, design, weights=weights)
    with pytest.raises(ValueError, match=r"sample weights must be finite and positive"):
        fit_mixed_model(y, design, weights=-np.ones(24))
    with pytest.raises(ValueError, match=r"weights must have shape \(24,\)"):
        fit_mixed_model(y, design, weights=np.ones(3))
    with pytest.raises(ValueError, match="n_threads must be at least 1"):
        fit_mixed_model(y, design, n_threads=0)


def test_invalid_tests_are_refused() -> None:
    fit = fitted("one_intercept")
    with pytest.raises(TypeError, match="fit must be a MixedModelFit"):
        fixed_effect_tests(
            fit_linear_model(reference("one_intercept")["y"], fit.design)
        )
    with pytest.raises(
        ValueError, match="df_method must be one of 'satterthwaite', 'kenward_roger'"
    ):
        fixed_effect_tests(fit, df_method="containment")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="kenward_roger_weights must be one of"):
        fixed_effect_tests(fit, kenward_roger_weights=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="names are given only with contrasts"):
        fixed_effect_tests(fit, names=["a"])
    with pytest.raises(ValueError, match=r"must weigh exactly the fit's coefficients"):
        fixed_effect_tests(fit, make_contrasts(["groupB"], ["(Intercept)", "groupB"]))
    with pytest.raises(ValueError, match="one row per coefficient"):
        fixed_effect_tests(fit, np.ones(2))
