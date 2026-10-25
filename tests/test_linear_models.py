"""Tests for linear models fitted to many features at once.

Reference values in ``tests/data/linear_models`` come from limma's ``lmFit``
and ``contrasts.fit``, written by ``validation/linear_models.py``.
"""

import pickle
import re
from fractions import Fraction
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numba
import numpy as np
import pytest

from cytepool._jit import thread_limit
from cytepool.cube import from_bulk
from cytepool.design import Design
from cytepool.stats import linear_models
from cytepool.stats.linear_models import LinearModelFit, contrasts_fit, fit_linear_model

DATA = Path(__file__).parent / "data" / "linear_models"
CASES = (
    "two_groups",
    "groups_and_covariate",
    "large_scale_covariates",
    "sample_weights",
    "observation_weights",
    "group_means_observation_weights",
    "missing_values",
    "missing_values_sample_weights",
    "block_correlation",
    "block_correlation_sample_weights",
    "block_correlation_observation_weights",
    "block_correlation_missing_values",
    "zero_observation_weights",
    "zero_sample_weight",
    "large_scale_covariates_weights_missing",
)
# Cases in which some genes lose rank because of missing values or zero weights.
LOSING_RANK = (
    "missing_values",
    "missing_values_sample_weights",
    "block_correlation_missing_values",
    "zero_observation_weights",
    "large_scale_covariates_weights_missing",
)
# The tier for closed-form results.
EXACT = 1e-10


@cache
def reference(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz") as data:
        return {key: data[key] for key in data.files}


def options(data: dict[str, np.ndarray]) -> dict[str, Any]:
    keywords: dict[str, Any] = {"coefficient_names": tuple(data["coefficient_names"])}
    if "weights" in data:
        keywords["weights"] = data["weights"]
    if "block" in data:
        keywords["block"] = data["block"]
        keywords["correlation"] = float(data["correlation"])
    return keywords


@cache
def fitted(case: str) -> LinearModelFit:
    data = reference(case)
    return fit_linear_model(data["y"], data["design"], **options(data))


def not_estimable(case: str) -> np.ndarray:
    """Coefficients limma leaves NA or whose value depends on the column order.

    Estimable coefficients have one least squares estimate, so limma returns
    the same value when the columns of the design are reversed.
    """
    data = reference(case)
    values, reversed_values = data["coefficients"], data["reversed_coefficients"]
    unknown = np.isnan(values) | np.isnan(reversed_values)
    known = ~unknown
    spread = np.abs(values[known] - reversed_values[known])
    unknown[known] = spread > 1e-8 * np.maximum(np.abs(values[known]), 1.0)
    return unknown


def expected_nan(case: str) -> np.ndarray:
    if case in LOSING_RANK:
        return not_estimable(case)
    return np.isnan(reference(case)["coefficients"])


def undefined_contrasts(case: str) -> np.ndarray:
    """Contrasts that put weight on a coefficient expected to be NaN."""
    return expected_nan(case).astype(float) @ (reference(case)["contrasts"] != 0) > 0


def assert_matches(
    actual: np.ndarray,
    desired: np.ndarray,
    *,
    nan: np.ndarray | None = None,
    floor: np.ndarray | None = None,
) -> None:
    """Assert NaN at ``nan`` (default: where desired is NaN) and agreement elsewhere.

    Differences are relative to ``|desired|``, or to ``floor`` where that is
    larger: a coefficient far below its standard error has no relative
    precision in any implementation.
    """
    nan = np.isnan(desired) if nan is None else nan
    np.testing.assert_array_equal(np.isnan(actual), nan)
    scale = np.abs(desired) if floor is None else np.fmax(np.abs(desired), floor)
    finite = ~nan
    assert not np.isnan(desired[finite]).any()
    difference = np.abs(actual[finite] - desired[finite])
    assert np.all(difference <= EXACT * scale[finite]), np.max(
        difference / scale[finite]
    )


def assert_same_covariance(
    actual: np.ndarray, desired: np.ndarray, tolerance: float = EXACT
) -> None:
    """Assert agreement relative to the product of the standard deviations."""
    sd = np.sqrt(np.diag(desired))
    np.testing.assert_array_less(
        np.abs(actual - desired), tolerance * np.outer(sd, sd) + 1e-300
    )


@pytest.mark.parametrize("case", CASES)
def test_fit_matches_limma(case: str) -> None:
    data, fit = reference(case), fitted(case)
    standard_error = data["stdev_unscaled"] * data["sigma"][:, np.newaxis]
    nan = expected_nan(case)
    assert_matches(
        fit.coefficients, data["coefficients"], nan=nan, floor=standard_error
    )
    assert_matches(fit.stdev_unscaled, data["stdev_unscaled"], nan=nan)
    assert_matches(fit.sigma, data["sigma"])
    np.testing.assert_array_equal(fit.df_residual, data["df_residual"])
    assert_matches(fit.mean_expression, data["mean_expression"])
    assert_same_covariance(fit.cov_coefficients, data["cov_coefficients"])
    assert fit.coefficient_names == tuple(data["coefficient_names"])
    assert all(type(name) is str for name in fit.coefficient_names)


@pytest.mark.parametrize("case", CASES)
def test_limma_method_matches_contrasts_fit(case: str) -> None:
    data = reference(case)
    contrast = contrasts_fit(
        fitted(case),
        data["contrasts"],
        names=tuple(data["contrast_names"]),
        method="limma",
    )
    nan = undefined_contrasts(case)
    standard_error = data["contrast_stdev_unscaled"] * data["sigma"][:, np.newaxis]
    assert_matches(
        contrast.coefficients,
        data["contrast_coefficients"],
        nan=nan,
        floor=standard_error,
    )
    assert_matches(contrast.stdev_unscaled, data["contrast_stdev_unscaled"], nan=nan)
    assert_same_covariance(contrast.cov_coefficients, data["contrast_cov_coefficients"])
    assert contrast.cov_coefficients_per_feature is None
    # The approximation applies exactly when each feature has its own fit.
    shared = fitted(case).cov_coefficients_per_feature is None
    assert contrast.exact_standard_errors == shared


@pytest.mark.parametrize("case", CASES)
def test_exact_method_matches_limma_refit_on_reparametrised_design(case: str) -> None:
    data = reference(case)
    contrast = contrasts_fit(fitted(case), data["contrasts"], method="exact")
    nan = undefined_contrasts(case)
    standard_error = data["refit_stdev_unscaled"] * data["sigma"][:, np.newaxis]
    assert_matches(
        contrast.coefficients, data["refit_coefficients"], nan=nan, floor=standard_error
    )
    assert_matches(contrast.stdev_unscaled, data["refit_stdev_unscaled"], nan=nan)
    np.testing.assert_array_equal(contrast.sigma, fitted(case).sigma)
    np.testing.assert_array_equal(contrast.df_residual, fitted(case).df_residual)
    assert contrast.exact_standard_errors


@pytest.mark.parametrize("case", ["missing_values", "zero_observation_weights"])
def test_exact_contrasts_leave_undefined_covariances_nan(case: str) -> None:
    contrast = contrasts_fit(fitted(case), reference(case)["contrasts"])
    covariances = contrast.cov_coefficients_per_feature
    assert covariances is not None
    undefined = undefined_contrasts(case)
    assert undefined.any() and not undefined.all()
    expected = undefined[:, :, np.newaxis] | undefined[:, np.newaxis, :]
    np.testing.assert_array_equal(np.isnan(covariances), expected)
    diagonal = np.sqrt(np.diagonal(covariances, axis1=1, axis2=2))
    np.testing.assert_array_equal(diagonal, contrast.stdev_unscaled)


def test_methods_agree_for_uncorrelated_coefficients() -> None:
    # Group means without blocks are uncorrelated whatever the weights and
    # missing values, so limma's combination of standard errors is exact.
    rng = np.random.default_rng(23)
    groups = np.repeat([0, 1, 2], 4)
    design = (groups[:, np.newaxis] == np.arange(3)).astype(float)
    y = rng.normal(5.0, 1.0, (12, 20))
    y[rng.random(y.shape) < 0.1] = np.nan
    weights = rng.uniform(0.2, 3.0, y.shape)
    fit = fit_linear_model(y, design, weights=weights)
    contrast = np.array([[-1.0, 1.0, 0.0], [0.0, -1.0, 1.0], [0.0, 0.5, 0.5]]).T
    exact = contrasts_fit(fit, contrast, method="exact")
    limma = contrasts_fit(fit, contrast, method="limma")
    assert np.ptp(exact.stdev_unscaled[:, 0]) > 0.1  # errors differ between features
    np.testing.assert_allclose(limma.stdev_unscaled, exact.stdev_unscaled, rtol=1e-13)
    np.testing.assert_allclose(limma.coefficients, exact.coefficients, rtol=1e-13)


def test_methods_differ_for_observation_weights_on_a_non_orthogonal_design() -> None:
    data = reference("observation_weights")
    fit = fitted("observation_weights")
    exact = contrasts_fit(fit, data["contrasts"], method="exact")
    limma = contrasts_fit(fit, data["contrasts"], method="limma")
    ratio = limma.stdev_unscaled / exact.stdev_unscaled
    assert np.abs(ratio - 1.0).max() > 0.1
    np.testing.assert_allclose(
        limma.coefficients, exact.coefficients, rtol=1e-13, atol=1e-13
    )


@pytest.mark.parametrize(
    "case",
    [
        "groups_and_covariate",
        "missing_values_sample_weights",
        "block_correlation_missing_values",
        "zero_observation_weights",
    ],
)
def test_results_do_not_depend_on_the_order_of_samples(case: str) -> None:
    data = reference(case)
    keywords = options(data)
    order = np.random.default_rng(3).permutation(data["y"].shape[0])
    for key in ("weights", "block"):
        if key in keywords:
            keywords[key] = keywords[key][order]
    permuted = fit_linear_model(data["y"][order], data["design"][order], **keywords)
    fit = fitted(case)
    for name in ("coefficients", "stdev_unscaled", "sigma", "cov_coefficients"):
        np.testing.assert_allclose(
            getattr(permuted, name), getattr(fit, name), rtol=1e-10, atol=1e-12
        )
    np.testing.assert_array_equal(permuted.df_residual, fit.df_residual)
    np.testing.assert_allclose(
        permuted.mean_expression, fit.mean_expression, rtol=1e-14
    )
    if fit.cov_coefficients_per_feature is not None:
        np.testing.assert_allclose(
            permuted.cov_coefficients_per_feature,
            fit.cov_coefficients_per_feature,
            rtol=1e-10,
            atol=1e-12,
        )


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_estimability_does_not_depend_on_the_scale_of_a_column(scale: float) -> None:
    data = reference("missing_values")
    design = data["design"] * np.array([1.0, 1.0, 1.0, scale])
    fit = fit_linear_model(data["y"], design)
    original = fitted("missing_values")
    nan = np.isnan(original.coefficients)
    scaling = np.array([1.0, 1.0, 1.0, 1.0 / scale])
    standard_error = original.stdev_unscaled * original.sigma[:, np.newaxis] * scaling
    assert_matches(
        fit.coefficients,
        original.coefficients * scaling,
        nan=nan,
        floor=standard_error,
    )
    assert_matches(fit.stdev_unscaled, original.stdev_unscaled * scaling, nan=nan)
    np.testing.assert_array_equal(fit.df_residual, original.df_residual)
    assert_matches(fit.sigma, original.sigma)


def test_large_column_aliased_with_small_ones_is_not_estimable() -> None:
    # Over the samples of gene 0, size = 2e7 + 3e7 * group exactly, so size,
    # group and the intercept are confounded although size comes first.
    groups = np.repeat([0.0, 1.0], 5)
    offsets = np.array([0.0, 0.0, 0.0, 0.3, -0.2, 0.0, 0.0, 0.0, 0.25, -0.4])
    size = 1e7 * (2.0 + 3.0 * groups + offsets)
    design = np.column_stack([size, groups, np.ones(10)])
    y = np.random.default_rng(29).normal(5.0, 1.0, (10, 3))
    y[[3, 4, 8, 9], 0] = np.nan
    fit = fit_linear_model(y, design)
    assert np.isnan(fit.coefficients[0]).all()
    assert np.isfinite(fit.coefficients[1:]).all()
    observed = ~np.isnan(y[:, 0])
    squares = 0.0
    for g in (0.0, 1.0):
        values = y[observed & (groups == g), 0]
        squares += np.sum((values - values.mean()) ** 2)
    assert fit.df_residual[0] == 4
    np.testing.assert_allclose(fit.sigma[0], np.sqrt(squares / 4), rtol=1e-12)


def nearly_collinear_design(relative_residual: float) -> np.ndarray:
    """Columns 1, x and z with z - x tiny over samples 0 to 7 only.

    Over the first eight samples the part of z orthogonal to 1 and x has the
    given norm relative to z; the last two samples make the design well
    conditioned.
    """
    rng = np.random.default_rng(31)
    x = rng.normal(0.0, 1.0, 10)
    basis = np.column_stack([np.ones(8), x[:8]])
    direction = rng.normal(0.0, 1.0, 8)
    direction -= basis @ np.linalg.lstsq(basis, direction, rcond=None)[0]
    direction /= np.linalg.norm(direction)
    z = x.copy()
    z[:8] += relative_residual * np.linalg.norm(x[:8]) * direction
    z[8:] += [1.5, -2.0]
    return np.column_stack([np.ones(10), x, z])


@pytest.mark.parametrize(
    ("relative_residual", "aliased"), [(1e-9, True), (1e-6, False)]
)
def test_rank_tolerance_of_each_feature_matches_r(
    relative_residual: float, aliased: bool
) -> None:
    # R's qr() drops a column whose remaining norm is at most 1e-7 of its own.
    y = np.random.default_rng(37).normal(5.0, 1.0, (10, 2))
    y[8:, 0] = np.nan
    fit = fit_linear_model(y, nearly_collinear_design(relative_residual))
    assert fit.df_residual[0] == (6 if aliased else 5)
    np.testing.assert_array_equal(
        np.isnan(fit.coefficients[0]), [False, aliased, aliased]
    )
    assert np.isfinite(fit.coefficients[1]).all()


def test_rank_tolerance_of_the_design_matches_r() -> None:
    with pytest.raises(ValueError, match=r"column\(s\) 'x3'"):
        fit_linear_model(np.ones((8, 1)), nearly_collinear_design(1e-9)[:8])
    fit_linear_model(np.ones((8, 1)), nearly_collinear_design(1e-6)[:8])


def test_columns_that_combine_exactly_over_a_features_samples_are_not_estimable() -> (
    None
):
    # Over samples 0 to 7, a = 0.95 x + 0.05 z exactly, so z = 20 a - 19 x and
    # none of x, a and z is estimable, although a alone is within the rank
    # tolerance of x and z is not. The intercept stays estimable.
    design = nearly_collinear_design(1e-6)
    a = design[:, 1].copy()
    a[:8] += 0.05 * (design[:8, 2] - design[:8, 1])
    a[8:] += [0.7, 0.4]
    design = np.column_stack([design[:, :2], a, design[:, 2]])
    y = np.random.default_rng(41).normal(5.0, 1.0, (10, 2))
    y[8:, 0] = np.nan
    fit = fit_linear_model(y, design)
    np.testing.assert_array_equal(
        np.isnan(fit.coefficients[0]), [False, True, True, True]
    )
    assert np.isfinite(fit.coefficients[1]).all()
    # The columns span the same space as 1, x and z over those samples.
    reduced = fit_linear_model(y[:8, :1], design[:8][:, [0, 1, 3]])
    np.testing.assert_allclose(
        fit.coefficients[0, 0], reduced.coefficients[0, 0], rtol=1e-6
    )
    np.testing.assert_allclose(fit.sigma[0], reduced.sigma[0], rtol=1e-9)
    assert fit.df_residual[0] == reduced.df_residual[0] == 5


@pytest.fixture(scope="module")
def one_way() -> SimpleNamespace:
    """A one-way layout with three groups, a covariate-free treatment design."""
    rng = np.random.default_rng(11)
    groups = np.repeat([0, 1, 2], 4)
    design = np.column_stack([np.ones(12), groups == 1, groups == 2]).astype(float)
    y = (
        rng.normal(5.0, 1.0, (12, 6))
        + np.array([0.0, 1.0, -1.0])[groups][:, np.newaxis]
    )
    return SimpleNamespace(
        groups=groups, design=design, y=y, blocks=np.repeat(np.arange(6), 2)
    )


@pytest.mark.parametrize(
    "keywords",
    [
        {},
        {"weights": np.linspace(0.5, 2.0, 12)},
        {"weights": "observation"},
        {"block": np.repeat(np.arange(6), 2), "correlation": 0.4},
        {"missing": True},
    ],
    ids=[
        "ordinary",
        "sample_weights",
        "observation_weights",
        "block",
        "missing_values",
    ],
)
def test_constant_feature_has_zero_residual_variance(
    one_way: SimpleNamespace, keywords: dict[str, Any]
) -> None:
    y = one_way.y.copy()
    y[:, 2] = 7.25
    keywords = dict(keywords)
    if keywords.pop("missing", False):
        y[0, 2] = np.nan
        y[5, 0] = np.nan
    if isinstance(keywords.get("weights"), str):
        keywords["weights"] = np.random.default_rng(5).uniform(0.5, 2.0, y.shape)
    fit = fit_linear_model(y, one_way.design, **keywords)
    assert fit.sigma[2] <= 1e-14 * 7.25
    np.testing.assert_allclose(
        fit.coefficients[2], [7.25, 0.0, 0.0], rtol=1e-14, atol=1e-13
    )
    assert np.all(fit.sigma[[0, 1, 3, 4, 5]] > 0.1)


def test_one_way_layout_matches_group_means_and_pooled_variance(
    one_way: SimpleNamespace,
) -> None:
    y, groups = one_way.y.copy(), one_way.groups
    y[groups == 2, 1] = np.nan  # gene 1: group C missing
    y[groups == 0, 2] = np.nan  # gene 2: the reference group missing
    y[[0, 5], 3] = np.nan  # gene 3: unbalanced groups
    fit = fit_linear_model(y, one_way.design, coefficient_names=["A", "BvsA", "CvsA"])

    for gene in range(y.shape[1]):
        observed = ~np.isnan(y[:, gene])
        present = [g for g in range(3) if np.any(observed & (groups == g))]
        counts = {g: np.sum(observed & (groups == g)) for g in present}
        means = {g: np.mean(y[observed & (groups == g), gene]) for g in present}
        squares = sum(
            np.sum((y[observed & (groups == g), gene] - means[g]) ** 2) for g in present
        )
        df = observed.sum() - len(present)
        assert fit.df_residual[gene] == df
        np.testing.assert_allclose(fit.sigma[gene], np.sqrt(squares / df), rtol=1e-13)
        if 0 not in present:
            # Without the reference group, no coefficient is estimable.
            assert np.isnan(fit.coefficients[gene]).all()
            assert np.isnan(fit.stdev_unscaled[gene]).all()
            continue
        np.testing.assert_allclose(fit.coefficients[gene, 0], means[0], rtol=1e-13)
        np.testing.assert_allclose(
            fit.stdev_unscaled[gene, 0], np.sqrt(1 / counts[0]), rtol=1e-13
        )
        for g in (1, 2):
            if g in present:
                np.testing.assert_allclose(
                    fit.coefficients[gene, g], means[g] - means[0], rtol=1e-12
                )
                np.testing.assert_allclose(
                    fit.stdev_unscaled[gene, g],
                    np.sqrt(1 / counts[0] + 1 / counts[g]),
                    rtol=1e-13,
                )
            else:
                assert np.isnan(fit.coefficients[gene, g])
                assert np.isnan(fit.cov_coefficients_per_feature[gene, g]).all()

    contrast = contrasts_fit(fit, {"CvsB": [0.0, -1.0, 1.0], "BvsA": [0.0, 1.0, 0.0]})
    assert np.isnan(contrast.coefficients[1, 0])  # uses the missing group C
    np.testing.assert_allclose(
        contrast.coefficients[1, 1],
        np.nanmean(y[groups == 1, 1]) - np.nanmean(y[groups == 0, 1]),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        contrast.stdev_unscaled[1, 1], np.sqrt(1 / 4 + 1 / 4), rtol=1e-13
    )


def test_missing_values_fit_each_feature_on_its_own_samples() -> None:
    data = reference("block_correlation_missing_values")
    keywords = options(data)
    fit = fitted("block_correlation_missing_values")
    # Gene 0 lacks a whole block, gene 1 keeps one sample of a block, and the
    # first later gene with a missing value lost it at random.
    later = next(
        g for g in range(3, data["y"].shape[1]) if np.isnan(data["y"][:, g]).any()
    )
    for gene in (0, 1, later):
        observed = ~np.isnan(data["y"][:, gene])
        alone = fit_linear_model(
            data["y"][observed, gene : gene + 1],
            data["design"][observed],
            block=keywords["block"][observed],
            correlation=keywords["correlation"],
        )
        assert alone.cov_coefficients_per_feature is None  # a fit with a shared design
        np.testing.assert_allclose(
            alone.coefficients[0], fit.coefficients[gene], rtol=1e-11
        )
        np.testing.assert_allclose(
            alone.stdev_unscaled[0], fit.stdev_unscaled[gene], rtol=1e-12
        )
        np.testing.assert_allclose(alone.sigma[0], fit.sigma[gene], rtol=1e-12)
        assert alone.df_residual[0] == fit.df_residual[gene]
        assert_same_covariance(
            alone.cov_coefficients, fit.cov_coefficients_per_feature[gene], 1e-12
        )


def test_features_without_residual_degrees_of_freedom() -> None:
    fit = fitted("missing_values")
    y = reference("missing_values")["y"]
    # Gene 3 has exactly as many observations as coefficients, gene 4 none.
    assert fit.df_residual[3] == 0 and np.isnan(fit.sigma[3])
    assert np.isfinite(fit.coefficients[3]).all()
    assert fit.df_residual[4] == 0 and np.isnan(fit.sigma[4])
    assert np.isnan(fit.coefficients[4]).all() and np.isnan(fit.mean_expression[4])
    np.testing.assert_allclose(fit.mean_expression[3], np.nanmean(y[:, 3]), rtol=1e-15)


@pytest.mark.parametrize("missing", [False, True])
def test_mean_expression_is_the_exact_mean_rounded_once(missing: bool) -> None:
    """As R's rowMeans, which accumulates in extended precision, gives limma's Amean.

    The reference is exact rational arithmetic, rounded once to the nearest double.
    voom depends on it: the ties of its trend are limma's only when these means are
    R's to the last bit.
    """
    rng = np.random.default_rng(17)
    y = rng.normal(0.0, 1.0, (7, 400)) * 10.0 ** rng.uniform(-3.0, 3.0, 400)
    if missing:
        y[rng.random(y.shape) < 0.2] = np.nan
        y[:, 0] = np.nan
    fit = fit_linear_model(y, np.ones((7, 1)))
    exact = np.full(y.shape[1], np.nan)
    for j, column in enumerate(y.T):
        observed = column[~np.isnan(column)]
        if observed.size:
            exact[j] = float(sum(map(Fraction, observed)) / observed.size)
    np.testing.assert_array_equal(fit.mean_expression, exact)


def test_mean_expression_near_the_largest_doubles() -> None:
    y = np.array([[2e300, 1e308], [2.18e300, 1e308], [2.09e300, 1e308]])
    with np.errstate(over="ignore", invalid="ignore"):
        fit = fit_linear_model(y, np.ones((3, 1)))
    exact = float(sum(map(Fraction, y[:, 0])) / 3)
    np.testing.assert_allclose(fit.mean_expression[0], exact, rtol=1e-15)
    # The sum overflows: the mean is infinite, as numpy's is.
    assert fit.mean_expression[1] == np.inf


def test_zero_weights_leave_values_out_of_the_fit_but_not_the_mean(
    one_way: SimpleNamespace,
) -> None:
    weights = np.random.default_rng(13).uniform(0.5, 2.0, one_way.y.shape)
    weights[3, 1] = 0.0
    zero = fit_linear_model(one_way.y, one_way.design, weights=weights)
    missing = one_way.y.copy()
    missing[3, 1] = np.nan
    left_out = fit_linear_model(missing, one_way.design, weights=weights)
    for name in ("coefficients", "stdev_unscaled", "sigma", "df_residual"):
        np.testing.assert_array_equal(getattr(zero, name), getattr(left_out, name))
    np.testing.assert_allclose(zero.mean_expression, one_way.y.mean(axis=0), rtol=1e-15)
    assert zero.df_residual[1] == 8 and zero.df_residual[0] == 9

    # A zero sample weight leaves the sample out of every feature's fit.
    sample_weights = np.linspace(0.5, 2.0, 12)
    sample_weights[7] = 0.0
    zero_sample = fit_linear_model(one_way.y, one_way.design, weights=sample_weights)
    keep = sample_weights > 0
    without = fit_linear_model(
        one_way.y[keep], one_way.design[keep], weights=sample_weights[keep]
    )
    np.testing.assert_allclose(
        zero_sample.coefficients, without.coefficients, rtol=1e-12
    )
    np.testing.assert_allclose(zero_sample.sigma, without.sigma, rtol=1e-12)
    np.testing.assert_array_equal(zero_sample.df_residual, without.df_residual)
    np.testing.assert_allclose(
        zero_sample.cov_coefficients_per_feature[0],
        without.cov_coefficients,
        rtol=1e-12,
    )


def test_equal_weights_rescale_the_fit(one_way: SimpleNamespace) -> None:
    ordinary = fit_linear_model(one_way.y, one_way.design)
    for weights in (np.full(12, 4.0), np.full(one_way.y.shape, 4.0)):
        weighted = fit_linear_model(one_way.y, one_way.design, weights=weights)
        np.testing.assert_allclose(
            weighted.coefficients, ordinary.coefficients, rtol=1e-13
        )
        np.testing.assert_allclose(
            weighted.stdev_unscaled, ordinary.stdev_unscaled / 2.0, rtol=1e-13
        )
        np.testing.assert_allclose(weighted.sigma, ordinary.sigma * 2.0, rtol=1e-13)


def test_sample_weights_give_a_shared_covariance_and_observation_weights_do_not(
    one_way: SimpleNamespace,
) -> None:
    weights = np.linspace(0.5, 2.0, 12)
    shared = fit_linear_model(one_way.y, one_way.design, weights=weights)
    expanded = np.repeat(weights[:, np.newaxis], one_way.y.shape[1], axis=1)
    per_feature = fit_linear_model(one_way.y, one_way.design, weights=expanded)
    assert shared.cov_coefficients_per_feature is None
    assert per_feature.cov_coefficients_per_feature is not None
    np.testing.assert_allclose(
        per_feature.coefficients, shared.coefficients, rtol=1e-12
    )
    np.testing.assert_allclose(
        per_feature.stdev_unscaled, shared.stdev_unscaled, rtol=1e-12
    )
    np.testing.assert_allclose(per_feature.sigma, shared.sigma, rtol=1e-12)
    for covariance in per_feature.cov_coefficients_per_feature:
        np.testing.assert_allclose(covariance, shared.cov_coefficients, rtol=1e-12)
    # As in limma, a fit with feature-specific weights reports the covariance
    # matrix of the unweighted fit.
    unweighted = fit_linear_model(one_way.y, one_way.design).cov_coefficients
    np.testing.assert_allclose(per_feature.cov_coefficients, unweighted, rtol=1e-14)


def test_block_correlation_without_effect_matches_ordinary_least_squares(
    one_way: SimpleNamespace,
) -> None:
    ordinary = fit_linear_model(one_way.y, one_way.design)
    for block, correlation in ((one_way.blocks, 0.0), (np.arange(12), 0.7)):
        fit = fit_linear_model(
            one_way.y, one_way.design, block=block, correlation=correlation
        )
        np.testing.assert_allclose(fit.coefficients, ordinary.coefficients, rtol=1e-13)
        np.testing.assert_allclose(
            fit.stdev_unscaled, ordinary.stdev_unscaled, rtol=1e-13
        )
        np.testing.assert_allclose(fit.sigma, ordinary.sigma, rtol=1e-13)


def test_design_names_and_input_types(one_way: SimpleNamespace) -> None:
    design = SimpleNamespace(matrix=one_way.design, columns=["Intercept", "B", "C"])
    named = fit_linear_model(one_way.y, design)
    assert named.coefficient_names == ("Intercept", "B", "C")
    plain = fit_linear_model(one_way.y, one_way.design)
    assert plain.coefficient_names == ("x1", "x2", "x3")
    renamed = fit_linear_model(one_way.y, design, coefficient_names=["a", "b", "c"])
    assert renamed.coefficient_names == ("a", "b", "c")
    generated = fit_linear_model(
        one_way.y, one_way.design, coefficient_names=(n for n in "abc")
    )
    assert generated.coefficient_names == ("a", "b", "c")
    contrast = contrasts_fit(generated, np.eye(3)[:, 1:], names=(n for n in "BC"))
    assert contrast.coefficient_names == ("B", "C")

    rounded = one_way.y.round()
    integers = fit_linear_model(rounded.astype(int), one_way.design.astype(int))
    floats = fit_linear_model(rounded, one_way.design)
    np.testing.assert_array_equal(integers.coefficients, floats.coefficients)


class Table:
    """A table whose mixed boolean and float columns give an object array."""

    def __init__(self, values: np.ndarray, columns: list[str]) -> None:
        self.values, self.columns = values, columns

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        return self.values


def test_tables_with_mixed_column_types_are_accepted(one_way: SimpleNamespace) -> None:
    values = np.empty((12, 3), dtype=object)
    values[:, 0] = 1.0
    values[:, 1] = [bool(g == 1) for g in one_way.groups]
    values[:, 2] = [bool(g == 2) for g in one_way.groups]
    fit = fit_linear_model(one_way.y, Table(values, ["Intercept", "B", "C"]))
    assert fit.coefficient_names == ("Intercept", "B", "C")
    expected = fit_linear_model(one_way.y, one_way.design)
    np.testing.assert_array_equal(fit.coefficients, expected.coefficients)
    words = np.full((12, 3), "one", dtype=object)
    with pytest.raises(TypeError, match="design must hold real numbers"):
        fit_linear_model(one_way.y, Table(words, ["a", "b", "c"]))


def assert_fits_like_limma(fit: LinearModelFit, data: dict[str, np.ndarray]) -> None:
    standard_error = data["stdev_unscaled"] * data["sigma"][:, np.newaxis]
    assert_matches(fit.coefficients, data["coefficients"], floor=standard_error)
    assert_matches(fit.stdev_unscaled, data["stdev_unscaled"])
    assert_matches(fit.sigma, data["sigma"])
    np.testing.assert_array_equal(fit.df_residual, data["df_residual"])
    assert_same_covariance(fit.cov_coefficients, data["cov_coefficients"])


def groups_and_covariate_samples() -> dict[str, Any]:
    """The samples of that fixture, with four donors who each give one per group."""
    design = reference("groups_and_covariate")["design"]
    group = np.where(design[:, 1] == 1, "B", np.where(design[:, 2] == 1, "C", "A"))
    donor = np.tile(["D1", "D2", "D3", "D4"], 3)
    return {"group": group, "age": design[:, 3], "donor": donor}


def test_a_design_fits_as_limma_does_and_one_with_random_effects_is_refused() -> None:
    data = reference("groups_and_covariate")
    samples = groups_and_covariate_samples()
    design = Design(samples, "~ group + age")
    np.testing.assert_array_equal(design.matrix, data["design"])
    fit = fit_linear_model(data["y"], design)
    assert fit.coefficient_names == ("(Intercept)", "groupB", "groupC", "age")
    assert_fits_like_limma(fit, data)

    # A random donor intercept would be dropped by a fit of the fixed effects.
    mixed = Design(samples, "~ group + age", random="(1 | donor)")
    with pytest.raises(
        ValueError,
        match=r"^fit_linear_model fits fixed effects and, through block= and "
        r"correlation=, one random intercept with a known correlation, so it would "
        r"drop the random effect \(1 \| donor\) of this design\. fit_mixed_model "
        r"estimates it by REML, and fixed_effect_tests tests contrasts of the fixed "
        r"effects with Satterthwaite's degrees of freedom\. Other analyses: a "
        r"fixed-effects analysis, with the Design built without random= and donor "
        r"in the formula when the conditions compared vary within each donor "
        r"\(left out, the samples of a donor count as independent\); ",
    ):
        fit_linear_model(data["y"], mixed)
    # Copies and derived designs keep the random effect, and are refused too.
    for derived in (
        pickle.loads(pickle.dumps(mixed)),
        mixed.with_contrasts({"b": "groupB"}),
    ):
        with pytest.raises(ValueError, match=r"drop the random effect \(1 \| donor\)"):
            fit_linear_model(data["y"], derived)
    # Its fixed effects alone are a design without random effects.
    fixed_only = fit_linear_model(data["y"], mixed.model_matrix)
    assert fixed_only.coefficient_names == fit.coefficient_names
    assert_fits_like_limma(fixed_only, data)


def test_a_random_intercept_with_a_known_correlation_fits_as_limma_does() -> None:
    # limma's lmFit with block and correlation fits a random intercept for each
    # block with the correlation known. A design whose only random effect is that
    # intercept is fitted when block groups the samples as the term does, whatever
    # the labels, and refused when it groups them otherwise.
    data = reference("block_correlation")
    donors = [f"D{int(label)}" for label in data["block"]]
    samples = {
        "treated": data["design"][:, 1],
        "donor_age": data["design"][:, 2],
        "donor": donors,
    }
    design = Design(samples, "~ treated + donor_age", random="(1 | donor)")
    np.testing.assert_array_equal(design.matrix, data["design"])
    correlation = float(data["correlation"])
    # The donor labels, the fixture's integers, and other names for the same pairs.
    for block in (design.samples["donor"], data["block"], donors[::-1]):
        fit = fit_linear_model(data["y"], design, block=block, correlation=correlation)
        assert fit.coefficient_names == ("(Intercept)", "treated", "donor_age")
        assert_fits_like_limma(fit, data)
    pairs = data["block"] // 2  # two donors to a block: another grouping
    with pytest.raises(
        ValueError,
        match=r"; block= does not group the samples as \(1 \| donor\) does\.",
    ):
        fit_linear_model(data["y"], design, block=pairs, correlation=correlation)
    with pytest.raises(ValueError, match=r"given block=design\.samples\['donor'\] and"):
        fit_linear_model(data["y"], design)


@pytest.mark.parametrize(
    ("random", "block", "present", "absent"),
    [
        (
            "(1 | donor)",
            None,
            [
                r"the random effect \(1 \| donor\) of this design\.",
                r"; this random intercept with a known correlation between the samples "
                r"of each donor, given block=design\.samples\['donor'\] and "
                r"correlation= \(fit_linear_model does not estimate the "
                r"correlation\);",
                r"; or, when the conditions compared differ between levels of donor, "
                r"not within them, a fit of one profile per donor, pooled first with "
                r"Cube\.collapse_samples\('donor'\)\.$",
            ],
            [],
        ),
        (
            "(1 || donor)",
            None,
            [r"effect \(1 \|\| donor\)", r"given block=design\.samples\['donor'\]"],
            [],
        ),
        (
            "(age | donor)",
            None,
            [r"effect \(age \| donor\)", r"Cube\.collapse_samples\('donor'\)\.$"],
            ["given block="],
        ),
        (
            "(age | donor)",
            "donor",
            [r"; block= can stand only for a single random intercept\."],
            ["given block="],
        ),
        (
            "(1 | donor) + (0 + age | donor)",
            None,
            [
                r"effects \(1 \| donor\) \+ \(0 \+ age \| donor\)",
                r"donor in the formula",
            ],
            ["given block="],
        ),
        (
            "(1 | donor) + (1 | plate)",
            None,
            [r"effects \(1 \| donor\) \+ \(1 \| plate\)", r"the grouping variables in"],
            ["given block=", "collapse_samples"],
        ),
        (
            ["(1 | donor)", "(1 | plate)"],
            None,
            [r"effects \(1 \| donor\) \+ \(1 \| plate\)"],
            ["given block=", "collapse_samples"],
        ),
        (
            "(1 | donor/plate)",
            None,
            [r"effects \(1 \| donor\) \+ \(1 \| plate:donor\)"],
            ["given block=", "collapse_samples"],
        ),
        (
            "(1 | donor:plate)",
            None,
            [
                r"given block= \(the donor:plate of each sample\)",
                r"Cube\.collapse_samples, by a column holding the donor:plate of each "
                r"sample\.$",
            ],
            [],
        ),
        (
            "(1 | donor)",
            "plate",
            [r"; block= does not group the samples as \(1 \| donor\) does\."],
            [],
        ),
        (
            "(1 | `cell type`)",
            None,
            [
                r"and `cell type` in the formula when",
                r"given block=design\.samples\[.cell type.\] and",
                r"Cube\.collapse_samples\(.cell type.\)\.$",
            ],
            [],
        ),
    ],
    ids=[
        "intercept",
        "uncorrelated",
        "slope",
        "slope_with_block",
        "two_by_donor",
        "two_groupings",
        "list",
        "nested",
        "combined",
        "other_block",
        "quoted_name",
    ],
)
def test_the_refusal_names_the_random_effects_and_what_can_be_fitted(
    random: str | list[str], block: str | None, present: list[str], absent: list[str]
) -> None:
    samples = groups_and_covariate_samples()
    samples["plate"] = np.repeat(["P1", "P2"], 6)
    samples["cell type"] = samples["donor"]
    design = Design(samples, "~ group + age", random=random)
    keywords: dict[str, Any] = {}
    if block is not None:
        keywords = {"block": design.samples[block], "correlation": 0.3}
    with pytest.raises(ValueError) as raised:
        fit_linear_model(reference("groups_and_covariate")["y"], design, **keywords)
    message = str(raised.value)
    assert message.startswith("fit_linear_model fits fixed effects and")
    # Random intercepts are estimated by fit_mixed_model; random slopes by nothing.
    if any(effect.effects.terms for effect in design.random):
        assert (
            "Random slopes are not available yet: fit_mixed_model fits random "
            "intercepts only. What can be fitted now: a fixed-effects analysis"
        ) in message
    else:
        assert "fit_mixed_model estimates" in message
        assert "Other analyses: a fixed-effects analysis" in message
    for pattern in present:
        assert re.search(pattern, message), (pattern, message)
    for text in absent:
        assert text not in message, (text, message)


def test_a_block_that_would_drop_a_random_effect_is_refused() -> None:
    # block= stands for one random intercept that groups the samples as it does,
    # with a correlation of at least zero; anything else would drop a random
    # effect, and an invalid block is reported as such.
    samples = groups_and_covariate_samples()
    samples["plate"] = np.repeat(["P1", "P2"], 6)
    data = reference("groups_and_covariate")
    intercept = Design(samples, "~ group + age", random="(1 | donor)")
    two = Design(samples, "~ group + age", random="(1 | donor) + (1 | plate)")
    donors = intercept.samples["donor"]
    for design, block, correlation, reason in (
        (two, donors, 0.3, r"; block= can stand only for a single random intercept\."),
        (
            intercept,
            np.arange(12),  # one block per sample: finer than the donors
            0.3,
            r"; block= does not group the samples as \(1 \| donor\) does\.",
        ),
        (
            intercept,
            donors,
            -0.3,
            r"; a random intercept gives the samples of a group a correlation of at "
            r"least zero, so correlation=-0\.3 does not stand for \(1 \| donor\)\.",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            fit_linear_model(data["y"], design, block=block, correlation=correlation)
    # A correlation of zero is a random intercept without variance: limma's fit.
    zero = fit_linear_model(data["y"], intercept, block=donors, correlation=0.0)
    assert_fits_like_limma(zero, data)
    with pytest.raises(
        ValueError, match=r"one label per sample \(12\), got shape \(\)"
    ):
        fit_linear_model(data["y"], intercept, block="donor", correlation=0.3)


def test_pooling_the_samples_of_each_donor_gives_a_fit_between_donors() -> None:
    # The pooling the refusal names, end to end: when the conditions differ between
    # donors, one profile per donor is the unit of replication. The reference is
    # the per-donor sums and the group means of a one-way layout, by hand.
    counts = np.random.default_rng(43).poisson(50.0, (12, 5)).astype(float)
    samples = {
        "sample_id": [f"S{i}" for i in range(12)],
        "donor": np.repeat([f"D{i}" for i in range(6)], 2),
        "condition": np.repeat(["ctrl", "stim"], 6),
    }
    mixed = Design(samples, "~ condition", random="(1 | donor)")
    with pytest.raises(ValueError, match=r"Cube\.collapse_samples\('donor'\)\.$"):
        fit_linear_model(np.log2(counts + 1.0), mixed)
    pooled = from_bulk(counts, samples=samples).collapse_samples("donor")
    y = np.log2(pooled.layer("counts", "bulk") + 1.0)
    fit = fit_linear_model(y, Design(pooled.samples, "~ condition"))
    by_hand = np.log2(counts.reshape(6, 2, 5).sum(axis=1) + 1.0)
    np.testing.assert_array_equal(y, by_hand)
    ctrl, stim = by_hand[:3].mean(axis=0), by_hand[3:].mean(axis=0)
    np.testing.assert_allclose(fit.coefficients[:, 0], ctrl, rtol=1e-13)
    # A difference of means near zero has absolute, not relative, precision.
    np.testing.assert_allclose(fit.coefficients[:, 1], stim - ctrl, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fit.df_residual, 4.0)


def test_only_a_design_carries_random_effects() -> None:
    # An object with a matrix and columns is a matrix with names; an attribute that
    # happens to be called random is not a random effect.
    data = reference("groups_and_covariate")
    names = list(data["coefficient_names"])
    for random in ("seed", [0.1, 0.2], ""):
        laid_out = SimpleNamespace(matrix=data["design"], columns=names, random=random)
        assert_fits_like_limma(fit_linear_model(data["y"], laid_out), data)


def test_dataframe_designs_are_read_as_tables_whatever_their_column_names() -> None:
    # pandas shows columns as attributes; columns named random or matrix must not
    # be taken for a design's random effects or its matrix.
    pandas = pytest.importorskip("pandas")
    data = reference("groups_and_covariate")
    names = ["random", "matrix", "columns", "age"]
    frame = pandas.DataFrame(data["design"], columns=names)
    fit = fit_linear_model(data["y"], frame)
    assert fit.coefficient_names == tuple(names)
    assert_fits_like_limma(fit, data)


def test_fit_does_not_share_memory_with_its_inputs(one_way: SimpleNamespace) -> None:
    design = one_way.design.copy()
    fit = fit_linear_model(one_way.y, design)
    design[:, 1] = 0.0
    np.testing.assert_array_equal(fit.design, one_way.design)
    contrast = np.array([0.0, 1.0, 0.0])
    contrasted = contrasts_fit(fit, contrast)
    contrast[1] = 99.0
    np.testing.assert_array_equal(contrasted.contrasts, [[0.0], [1.0], [0.0]])


def test_per_feature_fits_compile_one_signature(one_way: SimpleNamespace) -> None:
    # In parallel, or on the calling thread alone with one thread.
    routines = (
        linear_models._least_squares_per_feature,
        linear_models._least_squares_in_order,
    )
    if not all(hasattr(routine, "signatures") for routine in routines):
        pytest.skip("compilation is disabled")
    y = one_way.y.copy()
    y[0, 0] = np.nan
    read_only = np.asfortranarray(y)
    read_only.flags.writeable = False
    weights = np.asfortranarray(np.ones(y.shape, dtype=np.float32))
    for threads in (1, numba.config.NUMBA_NUM_THREADS):
        with thread_limit(threads):
            fit = fit_linear_model(y, one_way.design)
            fit_linear_model(read_only, fit.design, weights=weights)
            fit_linear_model(
                read_only[:, ::2], fit.design, block=one_way.blocks, correlation=0.3
            )
    parallel, serial = (len(routine.signatures) for routine in routines)
    assert serial == 1
    # With a single thread available, only other tests call the parallel routine.
    assert parallel == 1 or (parallel == 0 and numba.config.NUMBA_NUM_THREADS == 1)


def test_per_feature_fits_do_not_depend_on_threads_or_chunks() -> None:
    # Each feature is fitted by the same operations in whichever run of features,
    # and on whichever thread, it falls: the results are identical, not just close.
    rng = np.random.default_rng(11)
    design = np.column_stack([np.ones(12), rng.normal(size=12), np.tile([0, 1], 6)])
    y = rng.normal(size=(12, 37))
    y[rng.random(y.shape) < 0.1] = np.nan
    weights = rng.uniform(0.2, 2.0, y.shape)
    blocks = np.repeat(np.arange(4), 3)
    arguments = (
        np.ascontiguousarray(y.T),
        np.ascontiguousarray(weights.T),
        design,
        blocks.astype(np.int64),
        4,
        0.3,
        1e-7,
    )
    serial = linear_models._least_squares_in_order(*arguments)
    for n_chunks in (1, 2, 5, 37):
        chunked = linear_models._least_squares_per_feature(*arguments, n_chunks)
        for first, second in zip(serial, chunked, strict=True):
            assert np.array_equal(first, second, equal_nan=True)
    fits = [
        fit_linear_model(
            y, design, weights=weights, block=blocks, correlation=0.3, n_threads=threads
        )
        for threads in (1, numba.config.NUMBA_NUM_THREADS)
    ]
    for name in (
        "coefficients",
        "stdev_unscaled",
        "sigma",
        "cov_coefficients_per_feature",
    ):
        assert np.array_equal(
            getattr(fits[0], name), getattr(fits[1], name), equal_nan=True
        )


@pytest.mark.parametrize(
    ("n_threads", "error", "message"),
    [
        (0, ValueError, "n_threads must be at least 1, got 0"),
        (1.5, TypeError, "n_threads must be a positive integer or None, got float"),
        (True, TypeError, "n_threads must be a positive integer or None, not bool"),
    ],
)
def test_thread_numbers_are_checked_before_any_fit(
    one_way: SimpleNamespace, n_threads: Any, error: type[Exception], message: str
) -> None:
    # Also where one factorisation serves every feature and no thread would run.
    with pytest.raises(error, match=message):
        fit_linear_model(one_way.y, one_way.design, n_threads=n_threads)


def test_contrast_forms_names_and_composition(one_way: SimpleNamespace) -> None:
    fit = fit_linear_model(one_way.y, one_way.design, weights=np.ones(one_way.y.shape))
    matrix = np.array([[0.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
    by_matrix = contrasts_fit(fit, matrix)
    assert by_matrix.coefficient_names == ("contrast1", "contrast2")
    by_mapping = contrasts_fit(fit, {"B": matrix[:, 0], "CvsB": matrix[:, 1]})
    assert by_mapping.coefficient_names == ("B", "CvsB")
    np.testing.assert_array_equal(by_mapping.coefficients, by_matrix.coefficients)
    single = contrasts_fit(fit, matrix[:, 1], names=["CvsB"])
    np.testing.assert_allclose(
        single.coefficients[:, 0], by_matrix.coefficients[:, 1], rtol=1e-14
    )
    np.testing.assert_array_equal(by_matrix.contrasts, matrix)

    # Contrasts of contrasts equal the composed contrast, with exact covariances.
    second = np.array([[1.0], [2.0]])
    chained = contrasts_fit(by_matrix, second, names=["combined"])
    direct = contrasts_fit(fit, matrix @ second, names=["combined"])
    np.testing.assert_allclose(chained.coefficients, direct.coefficients, rtol=1e-13)
    np.testing.assert_allclose(
        chained.stdev_unscaled, direct.stdev_unscaled, rtol=1e-13
    )
    np.testing.assert_allclose(chained.contrasts, matrix @ second)


def test_exact_method_refuses_approximate_standard_errors() -> None:
    data = reference("observation_weights")
    limma = contrasts_fit(
        fitted("observation_weights"), data["contrasts"], method="limma"
    )
    assert not limma.exact_standard_errors
    with pytest.raises(ValueError, match="limma's approximation"):
        contrasts_fit(limma, np.eye(3), method="exact")
    # With one covariance matrix for all features, the limma method is exact.
    shared = contrasts_fit(fitted("two_groups"), [0.0, 1.0], method="limma")
    assert shared.exact_standard_errors
    twice = contrasts_fit(shared, [2.0], method="exact")
    np.testing.assert_allclose(
        twice.stdev_unscaled, 2.0 * shared.stdev_unscaled, rtol=1e-15
    )


def test_contrast_with_zero_variance_keeps_limma_method_finite() -> None:
    data = reference("observation_weights")
    matrix = np.column_stack([data["contrasts"][:, 1], np.zeros(4)])
    first = contrasts_fit(fitted("observation_weights"), matrix, method="limma")
    np.testing.assert_array_equal(first.stdev_unscaled[:, 1], 0.0)
    again = contrasts_fit(first, [1.0, 0.0], method="limma")
    np.testing.assert_allclose(
        again.stdev_unscaled[:, 0], first.stdev_unscaled[:, 0], rtol=1e-14
    )
    np.testing.assert_array_equal(again.coefficients[:, 0], first.coefficients[:, 0])


def test_fit_arrays_are_read_only() -> None:
    fit = fitted("missing_values")
    with pytest.raises(ValueError, match="read-only"):
        fit.coefficients[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        fit.cov_coefficients_per_feature[0, 0, 0] = 1.0
    assert "n_features=50" in repr(fit)


def test_weights_are_not_checked_where_values_are_missing(
    one_way: SimpleNamespace,
) -> None:
    y = one_way.y.copy()
    y[3, 1] = np.nan
    weights = np.ones(y.shape)
    weights[3, 1] = -1.0
    flagged = fit_linear_model(y, one_way.design, weights=weights)
    weights[3, 1] = 1.0
    np.testing.assert_array_equal(
        flagged.coefficients,
        fit_linear_model(y, one_way.design, weights=weights).coefficients,
    )


DESIGN = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6]]).astype(
    float
)
Y = np.arange(12.0).reshape(6, 2) ** 1.5


@pytest.mark.parametrize(
    ("arguments", "keywords", "error", "message"),
    [
        ((np.ones(6), DESIGN), {}, ValueError, "y must be a 2-D array"),
        ((np.ones((5, 2)), DESIGN), {}, ValueError, "y has 5 rows but design has 6"),
        ((np.ones((6, 0)), DESIGN), {}, ValueError, "at least one feature"),
        ((np.full((6, 2), "a"), DESIGN), {}, TypeError, "y must hold real numbers"),
        ((np.ones((6, 2), dtype=bool), DESIGN), {}, TypeError, "got dtype bool"),
        (
            (np.ones((6, 2), dtype=bool).astype(object), DESIGN),
            {},
            TypeError,
            "y must hold real numbers",
        ),
        (
            (Y, DESIGN.astype(int).astype(str).astype(object)),
            {},
            TypeError,
            "design must hold real numbers",
        ),
        (
            (np.where(np.eye(6, 2) > 0, np.inf, Y), DESIGN),
            {},
            ValueError,
            "infinite values",
        ),
        ((Y, DESIGN[:, 0]), {}, ValueError, "design must be a 2-D array"),
        (
            (Y, np.where(np.eye(6, 3) > 0, np.nan, DESIGN)),
            {},
            ValueError,
            "design must be finite",
        ),
        (
            (Y, np.column_stack([DESIGN, 2 * DESIGN[:, 2] - DESIGN[:, 1]])),
            {"coefficient_names": ["a", "b", "c", "d"]},
            ValueError,
            r"column\(s\) 'd' are linear combinations",
        ),
        ((Y, DESIGN), {"weights": np.ones(5)}, ValueError, "weights must have shape"),
        (
            (Y, DESIGN),
            {"weights": -np.ones(6)},
            ValueError,
            "sample weights must be finite and non-negative",
        ),
        (
            (Y, DESIGN),
            {"weights": np.where(np.eye(6, 2) > 0, -1.0, 1.0)},
            ValueError,
            "observation weights must be finite and non-negative wherever y is "
            "observed; 2 are",
        ),
        (
            (Y, DESIGN),
            {"weights": np.where(np.eye(6, 2) > 0, np.inf, 1.0)},
            ValueError,
            "observation weights must be finite",
        ),
        ((Y, DESIGN), {"block": np.arange(6)}, ValueError, "given together"),
        ((Y, DESIGN), {"correlation": 0.5}, ValueError, "given together"),
        (
            (Y, DESIGN),
            {"block": np.arange(5), "correlation": 0.5},
            ValueError,
            "one label per sample",
        ),
        (
            (Y, DESIGN),
            {"block": np.arange(6), "correlation": 1.0},
            ValueError,
            r"in \(-1, 1\)",
        ),
        (
            (Y, DESIGN),
            {"block": np.arange(6), "correlation": np.nan},
            ValueError,
            r"in \(-1, 1\)",
        ),
        (
            (Y, DESIGN),
            {"block": np.arange(6), "correlation": "0.5"},
            TypeError,
            "real number",
        ),
        (
            (Y, DESIGN),
            {"block": [0, 0, 1, 1, 1, 2], "correlation": -0.6},
            ValueError,
            "too negative for a block of 3 samples",
        ),
        (
            (Y, DESIGN),
            {"block": [0.0, 0.0, np.nan, 1.0, 1.0, 1.0], "correlation": 0.2},
            ValueError,
            "must not be NaN",
        ),
        (
            (Y, DESIGN),
            {"coefficient_names": ["a", "b"]},
            ValueError,
            "must have 3 entries",
        ),
        (
            (Y, DESIGN),
            {"coefficient_names": ["a", "a", "b"]},
            ValueError,
            "must be unique",
        ),
        ((Y, DESIGN), {"coefficient_names": "abc"}, TypeError, "not a single string"),
    ],
)
def test_fit_linear_model_rejects_invalid_input(
    arguments: tuple[Any, ...],
    keywords: dict[str, Any],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        fit_linear_model(*arguments, **keywords)


def test_negative_correlation_is_allowed_above_the_bound_of_the_largest_block() -> None:
    fit = fit_linear_model(Y, DESIGN, block=[0, 0, 1, 1, 1, 2], correlation=-0.45)
    assert np.isfinite(fit.coefficients).all()


@pytest.mark.parametrize(
    ("contrasts", "keywords", "error", "message"),
    [
        (np.ones((2, 1)), {}, ValueError, "one row per coefficient of the fit"),
        (np.ones((3, 0)), {}, ValueError, "at least one contrast"),
        (np.array([0.0, np.nan, 1.0]), {}, ValueError, "contrasts must be finite"),
        (np.ones((3, 2)), {"names": ["a"]}, ValueError, "names must have 2 entries"),
        (
            {"a": [0.0, 1.0, 0.0]},
            {"names": ["a"]},
            ValueError,
            "names must not be given",
        ),
        (
            {"a": [0.0, 1.0]},
            {},
            ValueError,
            "contrast 'a' must have one value per coefficient",
        ),
        ({}, {}, ValueError, "at least one contrast"),
        (
            np.ones(3),
            {"method": "approximate"},
            ValueError,
            "method must be 'exact' or 'limma'",
        ),
    ],
)
def test_contrasts_fit_rejects_invalid_input(
    contrasts: Any, keywords: dict[str, Any], error: type[Exception], message: str
) -> None:
    fit = fit_linear_model(Y, DESIGN)
    with pytest.raises(error, match=message):
        contrasts_fit(fit, contrasts, **keywords)


def test_contrasts_fit_needs_a_fit() -> None:
    with pytest.raises(TypeError, match="LinearModelFit"):
        contrasts_fit({"coefficients": Y}, np.ones(3))


@pytest.mark.parametrize("small", [0.0, 1e-20])
def test_sample_weights_that_leave_one_sample_make_the_design_rank_deficient(
    small: float,
) -> None:
    weights = np.array([1.0, small, small, small, small, small])
    with pytest.raises(ValueError, match=r"once the sample weights are applied"):
        fit_linear_model(Y, DESIGN[:, [0, 2]], weights=weights)
    y = Y.copy()
    y[0, 0] = np.nan  # the same with a fit for each feature
    with pytest.raises(ValueError, match=r"column\(s\) 'x2'"):
        fit_linear_model(y, DESIGN[:, [0, 2]], weights=weights)


def test_block_correlation_near_one_can_make_the_design_rank_deficient() -> None:
    # Whitening scales differences within a block by (1 - r)^(-1/2) = 1e7, so
    # the small between-block part of the second column becomes negligible
    # next to its within-block part, which the third column repeats.
    between = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    within = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    design = np.column_stack([np.ones(6), 0.01 * between + within, within])
    fit_linear_model(Y, design, block=between, correlation=0.5)
    with pytest.raises(ValueError, match="numerically rank deficient once the weights"):
        fit_linear_model(Y, design, block=between, correlation=1 - 1e-14)


def test_fit_object_checks_its_fields() -> None:
    fit = fitted("two_groups")
    fields = {name: getattr(fit, name) for name in fit.__dataclass_fields__}
    with pytest.raises(ValueError, match="stdev_unscaled must have shape"):
        LinearModelFit(**{**fields, "stdev_unscaled": fit.stdev_unscaled[:, :1]})
    with pytest.raises(ValueError, match="coefficient_names must be 2 strings"):
        LinearModelFit(**{**fields, "coefficient_names": ("a",)})
    with pytest.raises(TypeError, match="sequence of strings"):
        LinearModelFit(**{**fields, "coefficient_names": "ab"})
    with pytest.raises(
        ValueError, match="design has 2 columns but there are 1 coefficients"
    ):
        LinearModelFit(
            **{
                **fields,
                "coefficients": fit.coefficients[:, :1],
                "stdev_unscaled": fit.stdev_unscaled[:, :1],
                "cov_coefficients": fit.cov_coefficients[:1, :1],
                "coefficient_names": ("a",),
            }
        )
    # Standard errors that cov_coefficients does not imply need per-feature
    # covariance matrices, or the mark of an approximation.
    factors = np.linspace(1.0, 2.0, fit.stdev_unscaled.shape[0])[:, np.newaxis]
    varied = fit.stdev_unscaled * factors
    with pytest.raises(ValueError, match="stdev_unscaled differ"):
        LinearModelFit(**{**fields, "stdev_unscaled": varied})
    approximate = LinearModelFit(
        **{**fields, "stdev_unscaled": varied, "exact_standard_errors": False}
    )
    assert not approximate.exact_standard_errors
    per_feature = fitted("missing_values")
    stored = {
        name: getattr(per_feature, name) for name in per_feature.__dataclass_fields__
    }
    with pytest.raises(ValueError, match="no per-feature covariance matrices"):
        LinearModelFit(**{**stored, "exact_standard_errors": False})
    with pytest.raises(TypeError, match="must be a bool"):
        LinearModelFit(**{**fields, "exact_standard_errors": 1})
    flagged = LinearModelFit(**{**fields, "exact_standard_errors": np.True_})
    assert flagged.exact_standard_errors is True
