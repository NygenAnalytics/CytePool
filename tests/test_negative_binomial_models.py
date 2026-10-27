"""Tests for negative binomial fits and deviances.

Reference values in ``tests/data/negative_binomial_fits`` come from edgeR's ``glmFit``,
``mglmLevenberg(tol = 1e-14)`` (on edgeR's ``addPriorCount`` output for the shrunk
coefficients) and ``nbinomDeviance``, from the maximum that Newton's method in R reaches
from edgeR's fit, and from mpmath for the unit deviance; ``validation/
negative_binomial_fits.py`` writes them. The other tests check closed forms and
properties that a wrong fit would break: equivariance under scaling of the library
sizes, reparametrization of the design, weights against replicated samples, the two
fitting methods against each other, and the documented edge cases.
"""

import dataclasses
import decimal
from functools import cache
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.optimize import brentq

from cytepool import Design
from cytepool.stats import negative_binomial_models
from cytepool.stats.negative_binomial_models import (
    NegativeBinomialFit,
    fit_negative_binomial_model,
    negative_binomial_deviance,
    negative_binomial_unit_deviance,
)

DATA = Path(__file__).parent / "data" / "negative_binomial_fits"
ONE_WAY = (
    "two_groups",
    "group_means_tmm",
    "intercept_only",
    "one_coefficient_per_sample",
    "sample_weights",
    "groups_per_observation",
)
LEVENBERG = (
    "groups_covariate",
    "paired",
    "per_observation",
    "covariate_sample_weights",
    "poisson",
    "sparse_counts",
    "date_covariate",
    "large_counts",
    "dispersion_below_1e-4",
)
# edgeR's Levenberg fits stop up to 2e-3 from the maximum with dispersions and weights
# per observation, 2e-6 with sparse counts, 1e-5 with a covariate such as a date or
# with large counts, and 2e-3 with a dispersion of 2e-5, so those cases are compared
# with the maximum only.
EDGER_ACCURATE = ("groups_covariate", "paired", "covariate_sample_weights", "poisson")
# edgeR's deviance is accurate in every case but the dispersion of 2e-5.
EDGER_DEVIANCE = ONE_WAY + LEVENBERG[:-1]
TWO_GROUPS = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])


@cache
def fixture(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def fit_fixture(case: str, **options: Any) -> NegativeBinomialFit:
    data = fixture(case)
    samples = {
        k.removeprefix("samples "): v
        for k, v in data.items()
        if k.startswith("samples ")
    }
    keys = ("dispersion", "weights", "offsets", "library_sizes", "norm_factors")
    inputs = {key: data[key] for key in keys if key in data}
    design = Design(samples, str(data["formula"]))
    return fit_negative_binomial_model(data["counts"], design, **inputs, **options)


def edger(case: str, key: str) -> np.ndarray:
    """An edgeR matrix, written by R column by column, as genes x columns."""
    n_genes = fixture(case)["counts"].shape[1]
    return fixture(case)[f"edgeR {key}"].reshape(-1, n_genes).T


def scaled(actual: np.ndarray, desired: np.ndarray, floor: float) -> float:
    """Largest |actual - desired| / max(|desired|, floor) over finite values."""
    keep = np.isfinite(desired)
    scale = np.maximum(np.abs(desired[keep]), floor)
    return float(np.max(np.abs(actual[keep] - desired[keep]) / scale, initial=0.0))


def simulated(n_samples: int, n_genes: int, dispersion: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    mean = rng.uniform(5.0, 500.0, n_genes) * rng.uniform(0.5, 2.0, (n_samples, 1))
    if dispersion == 0.0:
        return rng.poisson(mean).astype(float)
    size = 1.0 / dispersion
    return rng.negative_binomial(size, size / (size + mean)).astype(float)


# Against edgeR and the maximum.


@pytest.mark.parametrize("case", ONE_WAY)
def test_one_way_layouts_match_glmfit(case: str) -> None:
    fit = fit_fixture(case)
    n_samples = fixture(case)["counts"].shape[0]

    assert str(fixture(case)["method"]) == "oneway"
    # edgeR fits each group to steps of 1e-10 (mglmOneGroup's tol); both agree to
    # about 1e-11, also for groups without counts, whose log mean both set to -1e8.
    assert scaled(fit.coefficients, edger(case, "coefficients"), 1.0) < 1e-9
    assert scaled(fit.shrunk_coefficients, edger(case, "shrunk"), 1.0) < 1e-9
    assert scaled(fit.fitted_values.T, edger(case, "fitted"), 1e-300) < 1e-9
    # edgeR adds 1e-8 to counts and means in its deviance, which moves the unit
    # deviance of a zero count by up to about 5e-7.
    deviance = fixture(case)["edgeR deviance"]
    assert scaled(fit.deviance, deviance, n_samples) < 1e-6
    assert_array_equal(fit.df_residual, fixture(case)["edgeR df"])
    assert fit.converged.all()


@pytest.mark.parametrize("case", LEVENBERG)
def test_other_designs_reach_the_maximum(case: str) -> None:
    fit = fit_fixture(case)
    newton = edger(case, "newton")  # NaN for genes whose maximum is at infinity
    found = np.isfinite(newton).all(axis=1)
    fitted = np.exp(fit.design @ newton[found].T + fit.offsets[:, found])

    assert str(fixture(case)["method"]) == "levenberg"
    assert found.sum() >= 10
    # Both converge to the last steps of 1e-10 on the log scale.
    assert scaled(fit.coefficients[found], newton[found], 1.0) < 1e-8
    assert scaled(fit.fitted_values[:, found], fitted, 1e-300) < 1e-8
    assert fit.converged.all()


@pytest.mark.parametrize("case", EDGER_ACCURATE)
def test_other_designs_match_edgers_converged_fits(case: str) -> None:
    fit = fit_fixture(case)
    counts = fixture(case)["counts"]
    tight = edger(case, "tight_fitted").T
    # Genes whose fitted values tend to zero are compared below.
    vanishing = (counts == 0) & (tight < 1e-6)
    interior = ~vanishing.any(axis=0)
    coefficients = edger(case, "tight_coefficients")

    # mglmLevenberg(tol = 1e-14) stops up to about 2e-7 from the maximum.
    assert scaled(fit.coefficients[interior], coefficients[interior], 1.0) < 1e-6
    assert scaled(fit.fitted_values[:, interior], tight[:, interior], 1e-300) < 1e-6
    shrunk = edger(case, "tight_shrunk")
    assert scaled(fit.shrunk_coefficients, shrunk, 1.0) < 1e-6
    deviance = fixture(case)["edgeR deviance"]
    assert scaled(fit.deviance, deviance, counts.shape[0]) < 1e-6
    # Where a maximum lies at infinity, the fitted values that tend to zero are below
    # 1e-6 in both programs, and the others agree.
    assert np.all(fit.fitted_values[vanishing] < 1e-6)
    assert scaled(fit.fitted_values[~vanishing], tight[~vanishing], 1e-300) < 1e-6


@pytest.mark.parametrize("case", EDGER_DEVIANCE)
def test_deviance_at_edgers_fitted_values_matches_nbinomdeviance(case: str) -> None:
    data = fixture(case)
    weights = {"weights": data["weights"]} if "weights" in data else {}
    deviance = negative_binomial_deviance(
        data["counts"], edger(case, "fitted").T, data["dispersion"], **weights
    )
    # edgeR's offset of 1e-8, as above.
    assert (
        scaled(deviance, data["edgeR nbinom_deviance"], data["counts"].shape[0]) < 1e-6
    )


def test_genes_without_counts_have_the_constant_coefficients() -> None:
    # Groups and a covariate, with an intercept, and four genes without counts.
    data = fixture("per_observation")
    empty = ~data["counts"].any(axis=0)
    fit = fit_fixture("per_observation")

    assert empty.sum() == 4
    # edgeR's Levenberg fit returns NA for them; its one-way fit -1e8 and 0.
    assert np.isnan(edger("per_observation", "coefficients")[empty]).all()
    assert np.all(fit.coefficients[empty, 0] == -1e8)
    assert np.all(fit.coefficients[empty, 1:] == 0.0)
    assert not fit.fitted_values[:, empty].any()
    assert not fit.deviance[empty].any()
    assert np.all(fit.iterations[empty] == 0)


def test_unit_deviance_matches_mpmath() -> None:
    data = fixture("unit_deviance")
    y, mu, phi = data["points"].T
    exact = data["mpmath"]
    ours = negative_binomial_unit_deviance(y, mu, phi)
    # Accurate to about 1e-14 over counts and means from 1e-30 to 1e30 and dispersions
    # up to 1e12; zero and infinite values are exact.
    special = (exact == 0) | np.isinf(exact)

    assert special.any()
    assert_array_equal(ours[special], exact[special])
    assert_allclose(ours[~special], exact[~special], rtol=1e-13, atol=0)


# Closed forms.


def test_one_group_poisson_fit_is_the_ratio_of_totals() -> None:
    counts = simulated(6, 5, 0.0, 1)
    sizes = np.array([1.1e6, 2.3e6, 0.7e6, 1.9e6, 3.0e6, 1.4e6])
    weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0, 3.0])
    fit = fit_negative_binomial_model(
        counts, np.ones((6, 1)), dispersion=0.0, library_sizes=sizes, weights=weights
    )
    expected = np.log((weights @ counts) / (weights @ sizes))

    assert_allclose(fit.coefficients[:, 0], expected, rtol=1e-14)


def test_one_group_with_equal_libraries_fits_the_weighted_mean() -> None:
    # With equal offsets and dispersions the score sum w (y - mu) / (1 + phi mu) is
    # zero at the weighted mean, whatever the dispersion.
    counts = simulated(5, 4, 0.3, 2)
    weights = np.array([1.0, 0.5, 2.0, 1.0, 1.5])
    fit = fit_negative_binomial_model(
        counts, np.ones((5, 1)), dispersion=0.3, offsets=np.log(1e5), weights=weights
    )

    assert_allclose(
        fit.fitted_values[0], (weights @ counts) / weights.sum(), rtol=1e-13
    )


def test_poisson_group_means_are_ratios_of_group_totals() -> None:
    counts = simulated(6, 4, 0.0, 3)
    sizes = np.array([1.0e6, 1.5e6, 2.0e6, 0.8e6, 1.2e6, 2.5e6])
    fit = fit_negative_binomial_model(
        counts, TWO_GROUPS, dispersion=0.0, library_sizes=sizes, prior_count=0
    )
    first = counts[:3].sum(axis=0) / sizes[:3].sum()
    second = counts[3:].sum(axis=0) / sizes[3:].sum()

    assert_allclose(fit.coefficients[:, 0], np.log(first), rtol=1e-14)
    assert_allclose(
        fit.coefficients[:, 1], np.log(second / first), rtol=1e-12, atol=1e-14
    )


def test_shrunk_poisson_fold_changes_have_their_closed_form() -> None:
    # edgeR's addPriorCount gives sample i the prior c N_i / mean(N) and adds twice
    # that to N_i, so a Poisson group mean becomes (Y + c S / mean(N)) / (S + 2 c S /
    # mean(N)) for the group's total count Y and total library size S.
    counts = simulated(6, 6, 0.0, 4)
    counts[3:, 0] = 0.0
    sizes = np.array([1.0e3, 1.5e3, 2.0e3, 0.8e3, 1.2e3, 2.5e3])
    fit = fit_negative_binomial_model(
        counts, TWO_GROUPS, dispersion=0.0, library_sizes=sizes, prior_count=2.0
    )
    a = 2.0 / sizes.mean()
    first = counts[:3].sum(axis=0) / sizes[:3].sum()
    second = counts[3:].sum(axis=0) / sizes[3:].sum()

    assert_allclose(
        fit.shrunk_coefficients[:, 1], np.log((second + a) / (first + a)), rtol=1e-12
    )
    # Shrunk towards zero, and finite for the group without counts.
    assert np.all(
        np.abs(fit.shrunk_coefficients[1:, 1]) < np.abs(fit.coefficients[1:, 1])
    )
    assert fit.coefficients[0, 1] < -1e7 < fit.shrunk_coefficients[0, 1]


def test_a_coefficient_per_sample_fits_every_count() -> None:
    counts = simulated(4, 5, 0.2, 5)
    counts[2, 1] = 0.0
    sizes = np.array([2e5, 3e5, 4e5, 5e5])
    fit = fit_negative_binomial_model(
        counts, np.eye(4), dispersion=0.2, library_sizes=sizes, prior_count=0
    )

    assert_allclose(fit.fitted_values, counts, rtol=1e-13)
    assert_allclose(fit.deviance, 0.0, atol=1e-12)
    assert_array_equal(fit.df_residual, 0.0)
    # A sample without counts: its log mean is edgeR's stand-in for minus infinity.
    assert fit.coefficients[1, 2] == -1e8
    assert fit.fitted_values[2, 1] == 0.0


def test_a_gene_without_counts_and_no_constant_has_a_finite_maximum() -> None:
    # With y = 0 the likelihood falls with each mean; x = (-1, 1) cannot make both
    # small, and by symmetry the maximum has equal means, exp((o1 + o2) / 2).
    offsets = np.array([3.0, 5.0])
    fit = fit_negative_binomial_model(
        np.zeros((2, 1)), np.array([[-1.0], [1.0]]), dispersion=0.5, offsets=offsets
    )

    assert_allclose(fit.coefficients[0, 0], (offsets[0] - offsets[1]) / 2, atol=1e-10)
    assert_allclose(fit.fitted_values[:, 0], np.exp(4.0), rtol=1e-10)
    assert_allclose(fit.deviance, 2 * 2 * np.log1p(0.5 * np.exp(4.0)) / 0.5, rtol=1e-12)
    assert fit.converged.all()


def test_unit_deviance_closed_forms() -> None:
    y = np.array([0.0, 0.0, 3.0, 7.0, 40.0, 1e9])
    mu = np.array([2.5, 1e-9, 3.0, 2.0, 55.0, 1.2e9])

    # Zero counts: 2 log(1 + phi mu) / phi, and 2 mu in the Poisson limit.
    assert_allclose(
        negative_binomial_unit_deviance(0.0, mu, 0.4),
        2 * np.log1p(0.4 * mu) / 0.4,
        rtol=1e-14,
    )
    assert_allclose(negative_binomial_unit_deviance(0.0, mu, 0.0), 2 * mu, rtol=1e-15)
    # Poisson: 2 (y log(y / mu) - (y - mu)), where the direct form is accurate.
    far = (y > 0) & (np.abs(y - mu) > 0.5 * mu)
    poisson = 2 * (y[far] * np.log(y[far] / mu[far]) - (y[far] - mu[far]))
    assert_allclose(
        negative_binomial_unit_deviance(y[far], mu[far], 0.0), poisson, rtol=1e-13
    )
    # Zero at the mean; infinite for a count above a mean of zero.
    assert negative_binomial_unit_deviance(3.0, 3.0, 0.7) == 0.0
    assert negative_binomial_unit_deviance(1.0, 0.0, 0.7) == np.inf
    assert negative_binomial_unit_deviance(0.0, 0.0, 0.7) == 0.0


def test_unit_deviance_has_the_gamma_limit_and_the_poisson_limit() -> None:
    y, mu = 30.0, 45.0
    gamma = 2 * (y / mu - 1 - np.log(y / mu))
    poisson = 2 * (y * np.log(y / mu) - (y - mu))
    # phi d tends to the gamma deviance as phi grows, with a relative error of about
    # (1 / y - 1 / mu) / (phi gamma) = 0.15 / phi here, and d to the Poisson deviance
    # as phi falls, with one of about phi (y - mu)^2 / poisson = 40 phi.
    for phi in (1e3, 1e5, 1e7):
        assert (
            abs(phi * negative_binomial_unit_deviance(y, mu, phi) / gamma - 1)
            < 0.2 / phi
        )
    for phi in (1e-3, 1e-6, 1e-9):
        assert abs(negative_binomial_unit_deviance(y, mu, phi) / poisson - 1) < 50 * phi


def test_unit_deviance_is_continuous_where_its_method_changes() -> None:
    # The series is summed for |y - mu| < mu / 8; the direct form beyond.
    mu = 1000.0
    for phi in (0.0, 1e-3, 0.5, 1e4):
        edges = mu * (1 + np.array([-0.125, 0.125]))
        for edge in edges:
            near = np.array([np.nextafter(edge, 0), edge, np.nextafter(edge, np.inf)])
            values = negative_binomial_unit_deviance(near, mu, phi)
            assert np.ptp(values) <= 1e-13 * values.max()


def test_unit_deviance_broadcasts() -> None:
    counts = np.arange(6.0).reshape(2, 3)
    per_gene = negative_binomial_unit_deviance(counts, 2.0, np.array([0.1, 0.2, 0.3]))

    assert per_gene.shape == (2, 3)
    assert per_gene[1, 2] == negative_binomial_unit_deviance(5.0, 2.0, 0.3)
    assert isinstance(negative_binomial_unit_deviance(5.0, 2.0, 0.3), float)


def test_deviance_weights_observations() -> None:
    counts = simulated(4, 3, 0.1, 6)
    means = counts + 1.5
    unit = negative_binomial_unit_deviance(counts, means, 0.1)
    weights = np.array([1.0, 0.0, 2.0, 0.5])

    assert_allclose(negative_binomial_deviance(counts, means, 0.1), unit.sum(axis=0))
    weighted = negative_binomial_deviance(counts, means, 0.1, weights=weights)
    assert_allclose(weighted, weights @ unit, rtol=1e-14)
    # A zero weight drops an observation, even one whose unit deviance is infinite.
    means[1, 0] = 0.0
    counts[1, 0] = 4.0
    assert np.isfinite(
        negative_binomial_deviance(counts, means, 0.1, weights=weights)
    ).all()


# Invariants.


@pytest.mark.parametrize(
    "design",
    [TWO_GROUPS, np.column_stack([TWO_GROUPS, [0.3, -1.1, 0.4, 1.6, -0.2, 0.9]])],
    ids=["one-way", "levenberg"],
)
def test_scaling_library_sizes_shifts_only_the_intercept(design: np.ndarray) -> None:
    counts = simulated(6, 8, 0.1, 7)
    counts[:3, 0] = 0.0  # a group without counts
    sizes = np.array([1.0e6, 1.5e6, 2.0e6, 0.8e6, 1.2e6, 2.5e6])
    fit = fit_negative_binomial_model(
        counts, design, dispersion=0.1, library_sizes=sizes
    )
    scaled_fit = fit_negative_binomial_model(
        counts, design, dispersion=0.1, library_sizes=sizes * 1000.0
    )
    usual = slice(1, None)

    assert_allclose(scaled_fit.fitted_values, fit.fitted_values, rtol=1e-9, atol=1e-9)
    assert_allclose(scaled_fit.deviance, fit.deviance, rtol=1e-9, atol=1e-12)
    shift = scaled_fit.coefficients[usual, 0] - fit.coefficients[usual, 0]
    assert_allclose(shift, -np.log(1000.0), rtol=1e-9)
    assert_allclose(
        scaled_fit.coefficients[usual, 1:], fit.coefficients[usual, 1:], atol=1e-9
    )


def test_reparametrising_the_design_transforms_the_coefficients() -> None:
    counts = simulated(8, 10, 0.2, 8)
    x = np.column_stack([np.ones(8), np.repeat([0.0, 1.0], 4), np.linspace(-1, 1, 8)])
    transform = np.array([[1.0, 0.5, -0.2], [0.0, 2.0, 0.3], [0.0, 0.0, -1.5]])
    fit = fit_negative_binomial_model(counts, x, dispersion=0.2)
    other = fit_negative_binomial_model(counts, x @ transform, dispersion=0.2)

    assert_allclose(other.fitted_values, fit.fitted_values, rtol=1e-9)
    assert_allclose(other.deviance, fit.deviance, rtol=1e-9)
    expected = np.linalg.solve(transform, fit.coefficients.T).T
    assert_allclose(other.coefficients, expected, rtol=1e-9, atol=1e-10)
    assert_allclose(
        other.shrunk_coefficients,
        np.linalg.solve(transform, fit.shrunk_coefficients.T).T,
        rtol=1e-9,
        atol=1e-10,
    )


@pytest.mark.parametrize("one_way", [True, False], ids=["one-way", "levenberg"])
def test_integer_weights_equal_replicated_samples(one_way: bool) -> None:
    counts = simulated(5, 6, 0.15, 9)
    x = np.column_stack([np.ones(5), [0.0, 0.0, 1.0, 1.0, 1.0]])
    if not one_way:
        x = np.column_stack([x, [0.2, -0.5, 1.0, 0.1, -0.9]])
    sizes = np.array([1e5, 2e5, 1.5e5, 3e5, 2.5e5])
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0])
    replicated = np.repeat(np.arange(5), weights.astype(int))
    fit = fit_negative_binomial_model(
        counts, x, dispersion=0.15, library_sizes=sizes, weights=weights, prior_count=0
    )
    copies = fit_negative_binomial_model(
        counts[replicated],
        x[replicated],
        dispersion=0.15,
        library_sizes=sizes[replicated],
        prior_count=0,
    )

    assert_allclose(fit.coefficients, copies.coefficients, rtol=1e-9, atol=1e-10)
    assert_allclose(fit.deviance, copies.deviance, rtol=1e-9)


def test_group_fits_agree_with_newton_iterations_on_the_same_layout() -> None:
    # The one-way layout fitted group by group, and by Newton's iteration on an
    # orthonormal basis as any other design is: two methods, one maximum.
    counts = simulated(6, 12, 0.1, 10)
    counts[:3, 3] = 0.0
    x = np.column_stack([np.ones(6), np.repeat([0.0, 1.0], 3)])
    groups = negative_binomial_models._layout(x)
    basis, triangular = np.linalg.qr(x)
    iterative = dataclasses.replace(
        groups,
        one_way=False,
        members=np.arange(6),
        group_starts=np.array([0, 6]),
        group_inverse=np.empty((0, 0)),
        basis=np.ascontiguousarray(basis),
        triangular=triangular,
        start=basis.T @ np.ones(6),
    )
    by_gene = np.ascontiguousarray(counts.T)
    offsets = np.log(counts.sum(axis=1)).reshape(1, 6)
    arguments = (by_gene, offsets, np.full((1, 1), 0.1), np.ones((1, 1)))
    by_group = negative_binomial_models._fit(*arguments, groups, 200)
    by_newton = negative_binomial_models._fit(*arguments, iterative, 200)
    usual = np.arange(12) != 3

    assert groups.one_way
    assert_allclose(by_newton[0][usual], by_group[0][usual], rtol=1e-9, atol=1e-10)
    assert_allclose(by_newton[1], by_group[1], rtol=1e-9, atol=1e-9)
    assert_allclose(by_newton[2], by_group[2], rtol=1e-9, atol=1e-9)
    # A group without counts: fitted values 0 in one, below 1e-9 in the other.
    assert not by_group[1][3, :3].any()
    assert np.all(by_newton[1][3, :3] < 1e-9)


def test_fits_satisfy_the_score_equations() -> None:
    # At the maximum, X' s = 0 for s = w (y - mu) / (1 + phi mu): to rounding,
    # relative to the size of its terms.
    rng = np.random.default_rng(17)
    x = np.column_stack([np.ones(10), np.repeat([0.0, 1.0], 5), rng.normal(0, 1, 10)])
    for scale, phi in ((1.0, 0.2), (1e8, 0.002), (1.0, 0.0)):
        counts = simulated(10, 20, phi, 18) * scale
        weights = rng.uniform(0.5, 2.0, counts.shape)
        dispersion = phi * rng.uniform(0.5, 2.0, counts.shape)
        fit = fit_negative_binomial_model(
            counts, x, dispersion=dispersion, weights=weights, prior_count=0
        )
        mu = fit.fitted_values
        score = weights * (counts - mu) / (1 + dispersion * mu)
        size = np.abs(x).T @ (weights * (counts + mu) / (1 + dispersion * mu))
        assert np.all(np.abs(x.T @ score) <= 1e-9 * size)


def test_one_group_fit_solves_its_score_equation() -> None:
    # Unequal offsets, dispersions and weights: no closed form, so the root of
    # sum w (y - mu) / (1 + phi mu) is found by bisection-based root finding.
    counts = np.array([[3.0], [17.0], [0.0], [42.0], [8.0]])
    offsets = np.log([[2e4], [5e4], [1e4], [9e4], [3e4]])
    dispersion = np.array([[0.05], [0.5], [1.0], [0.2], [2.0]])
    weights = np.array([1.0, 0.3, 2.0, 1.5, 0.7])
    fit = fit_negative_binomial_model(
        counts, np.ones((5, 1)), dispersion=dispersion, offsets=offsets,
        weights=weights, prior_count=0,
    )  # fmt: skip

    def score(m: float) -> float:
        mu = np.exp(m + offsets[:, 0])
        return float(weights @ ((counts[:, 0] - mu) / (1 + dispersion[:, 0] * mu)))

    root = brentq(score, -20.0, 0.0, xtol=1e-15, rtol=1e-15)
    assert_allclose(fit.coefficients[0, 0], root, rtol=1e-14)


@pytest.mark.parametrize("shift", [0.0, 2.024e7], ids=["scaled", "shifted"])
def test_scaled_or_shifted_columns_give_the_same_fit(shift: float) -> None:
    # The iteration works on an orthonormal basis of the column space, so neither
    # the scale of a covariate nor its distance from zero changes the fit.
    rng = np.random.default_rng(19)
    days = rng.integers(0, 300, 12).astype(float)
    group = np.repeat([0.0, 1.0, 2.0], 4)
    centred = np.column_stack([np.ones(12), group == 1, group == 2, days - days.mean()])
    raw = centred.copy()
    raw[:, 3] = (days + shift) * 2.0**20 if shift == 0.0 else days + shift
    counts = simulated(12, 30, 0.1, 20)
    reference = fit_negative_binomial_model(counts, centred, dispersion=0.1)
    other = fit_negative_binomial_model(counts, raw, dispersion=0.1)

    assert other.converged.all()
    # The covariate shifted by 2e7 is stored to 4e-9 of its range of 300.
    assert_allclose(other.fitted_values, reference.fitted_values, rtol=1e-7)
    assert_allclose(other.deviance, reference.deviance, rtol=1e-7, atol=1e-9)


def test_scaling_every_weight_changes_only_the_deviance() -> None:
    counts = simulated(8, 10, 0.2, 21)
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8), np.tile([0.0, 1.0], 4)])
    fit = fit_negative_binomial_model(counts, x, dispersion=0.2, prior_count=0)
    for factor in (1e-25, 3.0, 1e20):
        scaled_fit = fit_negative_binomial_model(
            counts, x, dispersion=0.2, weights=factor, prior_count=0
        )
        assert_allclose(scaled_fit.coefficients, fit.coefficients, rtol=1e-12)
        assert_allclose(scaled_fit.deviance, factor * fit.deviance, rtol=1e-12)


def test_sparse_genes_with_a_covariate_tend_to_their_limit() -> None:
    # One count, at one end of a covariate: the maximum is at infinity, where the
    # other fitted values are zero and the count is fitted exactly (deviance 0). The
    # fitted values fall at rates that grow with the distance from the count.
    for n_samples in (8, 16, 24):
        x = np.column_stack([np.ones(n_samples), np.linspace(-1, 1, n_samples)])
        counts = np.zeros((n_samples, 2))
        counts[-1, 0] = 5.0
        counts[[-1, -2], 1] = [3.0, 1.0]
        fit = fit_negative_binomial_model(
            counts, x, dispersion=0.1, offsets=np.zeros(n_samples), prior_count=0
        )
        assert fit.converged.all()
        assert fit.fitted_values[-1, 0] == pytest.approx(5.0, rel=1e-6)
        assert np.all(fit.fitted_values[:-1, 0] < 1e-6)
        assert fit.deviance[0] < 1e-6
        # Two counts determine both coefficients, so that maximum is finite: the
        # fitted values stay positive, and the score equations hold.
        mu = fit.fitted_values[:, 1]
        score = x.T @ ((counts[:, 1] - mu) / (1 + 0.1 * mu))
        assert np.all(mu > 0)
        assert np.all(np.abs(score) <= 1e-9 * (np.abs(x).T @ (counts[:, 1] + mu)))


def test_a_single_count_next_to_a_nearly_equal_covariate_reaches_its_limit() -> None:
    # The count is at the oldest sample of its group, 0.1 years older than the next:
    # the maximum is at infinity, where the count is fitted exactly and every other
    # fitted value is zero, and the slope that separates the two samples grows
    # without bound. The fitted values of the zeros fall at very different rates,
    # through subnormal numbers.
    age = np.array([28.0, 27.1, 35.6, 53.5, 24.1, 47.4, 36.0, 47.3, 59.6, 52.0, 63.7])
    age = np.append(age, 21.8)
    group = np.repeat([0.0, 1.0, 2.0], 4)
    x = np.column_stack([np.ones(12), group == 1, group == 2, (age - 45) / 15])
    sizes = [1569, 1911, 1364, 1746, 2885, 1359, 2243, 3214, 1206, 2166, 2839, 3346]
    counts = np.zeros((12, 1))
    counts[5] = 1.0
    fit = fit_negative_binomial_model(
        counts, x, dispersion=0.1, library_sizes=np.array(sizes, float), prior_count=0
    )

    assert fit.converged[0]
    assert fit.deviance[0] < 1e-8
    assert fit.fitted_values[5, 0] == pytest.approx(1.0, rel=1e-8)
    assert np.all(np.delete(fit.fitted_values[:, 0], 5) < 1e-8)


def test_newton_system_is_finite_for_extreme_fitted_values() -> None:
    # Subnormal and huge fitted values: the square roots of the information and the
    # scores, against 50-digit decimal arithmetic. Zero counts whose fitted values are
    # below the floor are left out; positive counts are not, and one whose fitted
    # value is zero keeps its score w y, without information.
    y = np.array([0.0, 3.0, 0.0, 1e6, 2.0, 0.0, 4.0, 0.0])
    mu = np.array([5e-324, 1e-310, 1e300, 1e-200, 2.0, 1e-301, 0.0, 1e-299])
    phi = np.array([0.1, 0.1, 0.1, 0.0, 1e12, 0.1, 0.1, 0.1])
    w = np.array([1.0, 0.5, 1.0, 0.25, 1.0, 1.0, 0.5, 1.0])
    n, floor = y.size, 1e-300
    # One column per sample, so that score j is the score of sample j.
    work, scale, score, size = np.empty((n, n)), np.empty(n), np.empty(n), np.empty(n)
    negative_binomial_models._newton_system(
        y,
        mu,
        phi,
        w,
        np.eye(n),
        np.zeros(n),
        np.zeros(n),
        floor,
        work,
        scale,
        score,
        size,
    )
    with decimal.localcontext(prec=50):
        for i in range(n):
            yi, mi, pi, wi = (decimal.Decimal(float(v[i])) for v in (y, mu, phi, w))
            if mi == 0 or (yi == 0 and mi < decimal.Decimal(floor)):
                root, derivative = decimal.Decimal(0), wi * yi
            else:
                root = (wi * mi * (1 + pi * yi)).sqrt() / (1 + pi * mi)
                derivative = wi * (yi - mi) / (1 + pi * mi)
            assert work[i, i] == pytest.approx(float(root), rel=1e-14, abs=0)
            assert score[i] == pytest.approx(float(derivative), rel=1e-14, abs=0)


def test_dispersion_zero_is_the_limit_of_small_dispersions() -> None:
    counts = simulated(8, 6, 0.0, 11)
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    poisson = fit_negative_binomial_model(counts, x, dispersion=0.0)
    nearly = fit_negative_binomial_model(counts, x, dispersion=1e-12)

    assert_allclose(nearly.coefficients, poisson.coefficients, rtol=1e-8, atol=1e-9)
    assert_allclose(nearly.deviance, poisson.deviance, rtol=1e-8)


def test_threads_and_runs_of_genes_do_not_change_the_result() -> None:
    # Each gene is fitted by the same operations in whichever run of genes, and on
    # whichever thread, it falls: the results are identical, not just close.
    counts = simulated(8, 37, 0.1, 12)
    counts[:4, 5] = 0.0
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    layout = negative_binomial_models._layout(x)
    arguments = (
        np.ascontiguousarray(counts.T),
        np.log(counts.sum(axis=1)).reshape(1, 8),
        np.full((1, 1), 0.1),
        np.ones((1, 1)),
        layout.basis,
        layout.members,
        layout.group_starts,
        layout.group_inverse,
        layout.start,
        layout.has_constant,
        layout.one_way,
        200,
    )
    serial = negative_binomial_models._fit_in_order(*arguments)
    for n_chunks in (1, 2, 5, 37):
        chunked = negative_binomial_models._fit_in_parallel(*arguments, n_chunks)
        for first, second in zip(serial, chunked, strict=True):
            assert_array_equal(first, second)
    fits = [
        fit_negative_binomial_model(counts, x, dispersion=0.1, n_threads=threads)
        for threads in (1, numba.config.NUMBA_NUM_THREADS)
    ]
    assert_array_equal(fits[0].coefficients, fits[1].coefficients)
    assert_array_equal(fits[0].fitted_values, fits[1].fitted_values)


# Edges.


def test_large_counts_are_fitted() -> None:
    counts = simulated(6, 5, 0.0, 13) * 1e9
    sizes = np.full(6, 1e14)
    fit = fit_negative_binomial_model(
        counts, TWO_GROUPS, dispersion=0.0, library_sizes=sizes
    )
    first = counts[:3].sum(axis=0) / sizes[:3].sum()

    assert_allclose(fit.coefficients[:, 0], np.log(first), rtol=1e-14)
    covariate = np.column_stack([TWO_GROUPS, np.linspace(-1, 1, 6)])
    other = fit_negative_binomial_model(
        counts, covariate, dispersion=0.01, library_sizes=sizes, prior_count=0
    )
    assert other.converged.all()
    mu = other.fitted_values
    score = covariate.T @ ((counts - mu) / (1 + 0.01 * mu))
    assert np.all(np.abs(score) <= 1e-12 * (np.abs(covariate).T @ (counts + mu)))


def test_a_fit_whose_deviance_rounds_coarsely_still_converges() -> None:
    # Counts near 1e9 and a dispersion of 1e-6: the deviance cannot be evaluated to
    # 1e-12 of itself, so that the last Newton steps must be taken within its rounding
    # error. Two nearly collinear columns.
    counts = np.array([4379602577.0, 1130538500.0, 242665086.0, 2374975503.0])
    counts = np.append(counts, [6035990939.0, 486914257.0])
    covariate = np.array([-0.354129, 0.724185, 0.893589, 0.338096, -0.472852, 1.306493])
    nearly = np.array([-0.354051, 0.724382, 0.893752, 0.338109, -0.472954, 1.306227])
    x = np.column_stack([np.ones(6), covariate, nearly])
    offsets = np.array([0.295, 0.618, -0.657, 0.76, 0.428, 0.683])
    fit = fit_negative_binomial_model(
        counts[:, None], x, dispersion=1e-6, offsets=offsets, prior_count=0
    )
    mu = fit.fitted_values[:, 0]
    score = x.T @ ((counts - mu) / (1 + 1e-6 * mu))

    assert fit.converged[0]
    assert np.all(np.abs(score) <= 1e-12 * (np.abs(x).T @ (counts + mu)))


@pytest.mark.parametrize("factor", [1e-15, 1e15])
def test_scaling_poisson_counts_and_libraries_together_keeps_the_fit(
    factor: float,
) -> None:
    # The Poisson score equations are linear in the counts and fitted values, so that
    # counts and library sizes scaled together leave the coefficients unchanged,
    # whatever the scale of the deviance.
    counts = simulated(12, 20, 0.0, 22)
    x = np.column_stack([np.ones(12), np.tile([0.0, 1.0], 6), np.linspace(-1, 1, 12)])
    sizes = counts.sum(axis=1)
    fit = fit_negative_binomial_model(
        counts, x, dispersion=0.0, library_sizes=sizes, prior_count=0
    )
    scaled_fit = fit_negative_binomial_model(
        counts * factor, x, dispersion=0.0, library_sizes=sizes * factor, prior_count=0
    )

    assert scaled_fit.converged.all()
    assert_allclose(scaled_fit.coefficients, fit.coefficients, rtol=1e-12, atol=1e-12)
    assert_allclose(scaled_fit.fitted_values, fit.fitted_values * factor, rtol=1e-12)


def test_zero_counts_with_a_large_dispersion_times_mean_reach_their_limit() -> None:
    # A group without counts next to counts near 1e8, with a dispersion of 100: the
    # deviance of those zero counts grows only with the log of their fitted values,
    # where a Newton step would lower them by the dispersion times the mean. In the
    # limit they are zero, and the other samples have the fit of the model without
    # that group.
    group = np.repeat([0.0, 1.0, 2.0], 4)
    covariate = np.linspace(-1, 1, 12)
    x = np.column_stack([np.ones(12), group == 1, group == 2, covariate])
    rng = np.random.default_rng(23)
    counts = np.round(1e8 * np.exp(rng.normal(0, 0.3, 12) + 0.3 * (group == 1)))
    counts[group == 2] = 0.0
    offsets = rng.normal(0, 0.3, 12)
    fit = fit_negative_binomial_model(
        counts[:, None], x, dispersion=100.0, offsets=offsets, prior_count=0
    )
    rest = group != 2
    without = fit_negative_binomial_model(
        counts[rest, None],
        x[rest][:, [0, 1, 3]],
        dispersion=100.0,
        offsets=offsets[rest],
        prior_count=0,
    )

    assert fit.converged[0] and without.converged[0]
    assert_allclose(fit.fitted_values[rest], without.fitted_values, rtol=1e-12)
    assert np.all(fit.fitted_values[~rest] < 1e-20)
    assert np.all(np.abs(fit.coefficients) < 1e3)


def test_an_infinite_starting_deviance_is_not_reported_as_converged() -> None:
    # An offset 760 below the others makes the fitted value of a positive count zero
    # at the start, where its information is zero too.
    counts = np.array([[5.0], [7], [3], [9], [4], [6], [8], [2]])
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8), np.tile([0.0, 1.0], 4)])
    offsets = np.zeros(8)
    offsets[3] = -760.0
    fit = fit_negative_binomial_model(
        counts, x, dispersion=0.1, offsets=offsets, prior_count=0
    )

    assert not fit.converged[0]
    assert np.isfinite(fit.coefficients).all()


def test_a_fit_that_needs_more_iterations_is_reported() -> None:
    counts = simulated(8, 6, 0.1, 14)
    x = np.column_stack([np.ones(8), np.linspace(-1, 1, 8)])
    fit = fit_negative_binomial_model(counts, x, dispersion=0.1, max_iterations=1)

    assert not fit.converged.any()
    assert np.all(fit.iterations == 1)
    assert np.all(np.isfinite(fit.coefficients))
    assert "converged=0 of 6" in repr(fit)


def test_fits_are_read_only_and_keep_their_inputs() -> None:
    counts = simulated(6, 4, 0.1, 15)
    design = Design({"group": np.repeat(["a", "b"], 3)}, "~ group")
    dispersion = np.array([0.1, 0.2, 0.3, 0.4])
    fit = fit_negative_binomial_model(counts, design, dispersion=dispersion)

    assert fit.coefficient_names == ("(Intercept)", "groupb")
    assert fit.prior_count == 0.125
    assert fit.dispersion.shape == fit.offsets.shape == fit.weights.shape == (6, 4)
    assert_array_equal(fit.dispersion[2], dispersion)
    assert_allclose(fit.offsets[:, 0], np.log(counts.sum(axis=1)))
    assert_array_equal(fit.weights, 1.0)
    for name in ("coefficients", "fitted_values", "deviance", "converged", "offsets"):
        with pytest.raises(ValueError, match="read-only"):
            getattr(fit, name)[0] = 0
    dispersion[2] = 5.0  # the fit keeps its own copy
    assert fit.dispersion[0, 2] == 0.3


def test_prior_count_zero_gives_the_maximum_as_shrunk_coefficients() -> None:
    counts = simulated(6, 4, 0.1, 16)
    fit = fit_negative_binomial_model(counts, TWO_GROUPS, dispersion=0.1, prior_count=0)

    assert_array_equal(fit.shrunk_coefficients, fit.coefficients)


@pytest.mark.parametrize(
    ("change", "error", "message"),
    [
        ({"counts": -np.ones((6, 2))}, ValueError, "finite and non-negative"),
        ({"design": np.ones((5, 2))}, ValueError, "one row per sample"),
        (
            {"design": np.column_stack([TWO_GROUPS, TWO_GROUPS[:, 1]])},
            ValueError,
            "'x3' are linear",
        ),
        (
            {"dispersion": -0.1},
            ValueError,
            "dispersion must be finite and non-negative",
        ),
        ({"dispersion": np.inf}, ValueError, "dispersion must be finite"),
        ({"dispersion": np.ones(5)}, ValueError, "one per feature"),
        ({"dispersion": True}, TypeError, "dispersion must hold real numbers"),
        (
            {"offsets": np.array([np.inf] + [1.0] * 5)},
            ValueError,
            "offsets must be finite",
        ),
        ({"offsets": np.ones(4)}, ValueError, "one per sample"),
        ({"offsets": np.ones(6), "library_sizes": np.ones(6)}, ValueError, "not both"),
        ({"weights": np.r_[0.0, np.ones(5)]}, ValueError, "positive"),
        ({"weights": np.ones((2, 6))}, ValueError, "weights must be one value"),
        ({"library_sizes": np.r_[0.0, np.ones(5)]}, ValueError, "finite and positive"),
        ({"prior_count": -1.0}, ValueError, "prior_count must be non-negative"),
        ({"max_iterations": 0}, ValueError, "at least 1"),
        ({"max_iterations": 2.5}, TypeError, "positive integer"),
        ({"coefficient_names": ["a"]}, ValueError, "2 entries"),
    ],
)
def test_invalid_arguments_are_refused(
    change: dict[str, Any], error: type, message: str
) -> None:
    arguments = {"counts": np.ones((6, 2)), "design": TWO_GROUPS, "dispersion": 0.1}
    with pytest.raises(error, match=message):
        fit_negative_binomial_model(**(arguments | change))


def test_a_sample_without_counts_has_no_library_size() -> None:
    counts = np.ones((6, 2))
    counts[2] = 0.0
    with pytest.raises(ValueError, match=r"sample\(s\) \[2\] have no counts"):
        fit_negative_binomial_model(counts, TWO_GROUPS, dispersion=0.1)


def test_random_effects_are_refused() -> None:
    samples = {"group": np.repeat(["a", "b"], 3), "donor": ["d1", "d2", "d3"] * 2}
    design = Design(samples, "~ group", random="(1 | donor)")
    with pytest.raises(ValueError, match="fits fixed effects only"):
        fit_negative_binomial_model(np.ones((6, 2)), design, dispersion=0.1)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((np.ones(3), np.ones(2), 0.1), "do not broadcast"),
        ((np.ones(3), -np.ones(3), 0.1), "means must be finite and non-negative"),
        ((np.ones(3), np.ones(3), np.nan), "dispersion must be finite"),
    ],
)
def test_unit_deviance_refuses_invalid_arguments(
    arguments: tuple[Any, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        negative_binomial_unit_deviance(*arguments)


@pytest.mark.parametrize(
    ("keywords", "message"),
    [
        ({"means": np.ones((3, 2))}, "shape of counts"),
        ({"dispersion": np.ones(3)}, "one per feature"),
        ({"weights": np.ones(2)}, "one per sample"),
        ({"counts": np.ones(4)}, "2-D array"),
    ],
)
def test_deviance_refuses_invalid_shapes(
    keywords: dict[str, Any], message: str
) -> None:
    arguments = {"counts": np.ones((4, 2)), "means": np.ones((4, 2)), "dispersion": 0.1}
    with pytest.raises(ValueError, match=message):
        negative_binomial_deviance(**(arguments | keywords))
