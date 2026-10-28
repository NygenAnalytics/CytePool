"""Tests of edgeR's quasi-likelihood pipeline in cytepool.stats.quasi_likelihood.

The references are independent of the code under test: edgeR 4.4.0 and limma 3.62.1
outputs and 30-digit mpmath sums written by validation/quasi_likelihood.py, closed
forms (the gamma and Poisson limits of the moments, the pooled CPM), and invariants
that a wrong implementation would break (sample and gene order, library-size scale,
parametrisation of the design, equivalent contrasts).

Tolerances against edgeR follow the tiers of the validation script, which explains
them, at about three times the largest difference these fixtures show: edgeR
approximates the scale of the unit deviance to about 1e-3 and its df to about 2e-4, so
the quasi-dispersions agree to 5e-3; its search for the dispersion stops within 1e-5;
its Levenberg fits stop short of the maximum.
"""

import math
import warnings
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import Design
from cytepool.design import make_contrasts
from cytepool.stats.negative_binomial_models import fit_negative_binomial_model
from cytepool.stats.quasi_likelihood import (
    QuasiLikelihoodFit,
    QuasiLikelihoodTest,
    UnitDevianceMoments,
    _chi_squared_match,
    _leverages,
    average_log2_cpm,
    common_dispersion,
    fit_quasi_likelihood,
    negative_binomial_unit_deviance_moments,
    quasi_likelihood_f_test,
)
from cytepool.stats.special import digamma, trigamma

DATA = Path(__file__).parent / "data" / "quasi_likelihood"
CASES = (
    "two_groups",
    "groups_covariate",
    "paired",
    "prior_weights",
    "dispersion_given_low",
    "robust",
    "small_counts",
    "trended_dispersion",
    "no_trend",
    "covariate_trend",
    "offsets_per_observation",
)
MOMENTS = 5e-3  # edgeR's scale of the unit deviance is off by about 1e-3
DF = 5e-4  # and its df by about 2e-4
AVERAGE = 5e-3  # the average quasi-dispersion compounds them over two passes
PRIOR_DF = 1e-3  # the prior df maximizes a likelihood that is flat at large values
PROBABILITY = 5e-3  # of max(1, |log10 p|)
# The fits where the working dispersions differ (an average quasi-dispersion above 1)
# or where edgeR's Levenberg fits stop short of the maximum (not a one-way layout).
FITS = 2e-4


def load(name):
    return dict(np.load(DATA / f"{name}.npz"))


def contrast_or_coefficients(case, test):
    if test.startswith("coef"):
        return {"coefficients": [int(word) - 1 for word in test.split()[1:]]}
    return {"contrasts": case[f"contrast_{test}"]}


def fit_case(case):
    keywords = {
        key: case[key]
        for key in ("library_sizes", "offsets", "weights", "dispersion")
        if key in case
    }
    trend = case["covariate"] if "covariate" in case else not case["no_trend"]
    return fit_quasi_likelihood(
        case["counts"],
        case["design"],
        trend=trend,
        robust=bool(case["robust"]),
        **keywords,
    )


@pytest.fixture(scope="module")
def fitted():
    results = {}
    for name in CASES:
        case = load(name)
        fit = fit_case(case)
        tests = [
            quasi_likelihood_f_test(fit, **contrast_or_coefficients(case, str(test)))
            for test in case["tests"]
        ]
        results[name] = (case, fit, tests)
    return results


def one_way(design):
    return np.unique(design, axis=0).shape[0] == design.shape[1]


def same_working_dispersion(case, fit):
    """Whether the average quasi-dispersion is 1 in both, so that the fits are at the
    NB dispersion (up to edgeR's search tolerance where both estimate it)."""
    return case["edger_average.ql.dispersion"][0] == 1.0 == fit.average_quasi_dispersion


def compared_genes(case):
    """All genes but, with offsets per observation, the first: edgeR's aveLogCPM
    ignores that gene's offsets (minus infinity for it, which has no counts)."""
    return np.arange(case["counts"].shape[1]) >= (1 if "offsets" in case else 0)


def scaled(a, b, floor=1.0):
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), floor)))


def log10_ratio(a, b):
    la, lb = np.log10(np.asarray(a, float)), np.log10(np.asarray(b, float))
    return float(np.max(np.abs(la - lb) / np.maximum(1.0, np.abs(lb))))


def gamma_limit(phi):
    """Mean and variance of the gamma unit deviance with shape 1 / phi."""
    r = 1.0 / phi
    return (
        2.0 * r * (math.log(r) - float(digamma(r))),
        4.0 * r * r * (float(trigamma(r)) - 1.0 / r),
    )


# ---------------------------------------------------------------------------------------
# Against edgeR
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", CASES)
def test_the_dispersion_and_its_genes_are_edgers(fitted, name):
    case, fit, _ = fitted[name]
    if "edger_top.proportion" not in case:
        assert fit.top_proportion is None
        assert_array_equal(fit.dispersion, case["dispersion"])
        return
    assert fit.top_proportion == case["edger_top.proportion"][0]
    # edgeR's own search stops within its tolerance; with a tolerance of 1e-12 it
    # finds nearly the same maximum. The likelihood of the 24 top genes of these
    # small cases is flat enough that edgeR's deviance, which adds 1e-8 to counts and
    # means, and its Levenberg fits move its maximum by up to about 1e-5 (with 1,000
    # genes, the validation finds 1e-6).
    assert_allclose(fit.dispersion, case["edger_dispersion"][0], rtol=1e-4)
    assert_allclose(fit.dispersion, case["edger_tight_dispersion"][0], rtol=3e-5)


@pytest.mark.parametrize("name", CASES)
def test_the_quasi_dispersions_are_edgers(fitted, name):
    case, fit, _ = fitted[name]
    n_genes = case["counts"].shape[1]
    genes = compared_genes(case)
    assert_allclose(
        fit.average_quasi_dispersion,
        case["edger_average.ql.dispersion"][0],
        rtol=AVERAGE,
    )
    # Residual df below 0.01 never enter the prior; they are compared on that scale.
    assert scaled(fit.deviance_adjusted, case["edger_deviance.adj"], 1e-2) < MOMENTS
    assert scaled(fit.df_residual_adjusted, case["edger_df.residual.adj"], 1e-2) < DF
    assert_allclose(
        np.broadcast_to(fit.prior_quasi_dispersion, n_genes)[genes],
        np.broadcast_to(case["edger_s2.prior"], n_genes)[genes],
        rtol=MOMENTS,
    )
    assert_allclose(
        fit.posterior_quasi_dispersion[genes],
        case["edger_s2.post"][genes],
        rtol=MOMENTS,
    )
    # One value where edgeR reports one, and one per gene where it reports more.
    assert np.ndim(fit.df_prior) == (case["edger_df.prior"].size > 1)
    assert np.ndim(fit.prior_quasi_dispersion) == (case["edger_s2.prior"].size > 1)
    assert_allclose(
        np.broadcast_to(fit.df_prior, n_genes),
        np.broadcast_to(case["edger_df.prior"], n_genes),
        rtol=PRIOR_DF,
    )


@pytest.mark.parametrize("name", CASES)
def test_the_fits_leverages_and_abundance_are_edgers(fitted, name):
    case, fit, _ = fitted[name]
    n, n_genes = case["counts"].shape
    exact = one_way(case["design"]) and same_working_dispersion(case, fit)
    tolerance = 1e-6 if exact else FITS
    # Observations whose fitted values tend to zero (to about 1e-8 in edgeR, far less
    # here) get the leverages of the limit in both, and the rest those of the fits.
    leverage = case["edger_leverage"].reshape(n, n_genes)
    assert np.max(np.abs(fit.leverage - leverage)) < tolerance
    expected = case["edger_fitted.values"].reshape(n, n_genes)
    living = expected >= 1e-3
    fitted_values = np.asarray(fit.fit.fitted_values)[living]
    assert_allclose(fitted_values, expected[living], rtol=tolerance)
    if fit.covariate is not None and "covariate" not in case:
        # The trends follow aveLogCPM at the given dispersion, or at 0.05.
        assert_allclose(fit.covariate, case["edger_abundance"], atol=1e-10)


@pytest.mark.parametrize("name", CASES)
def test_the_test_columns_are_edgers(fitted, name):
    case, fit, tests = fitted[name]
    n_genes = case["counts"].shape[1]
    genes = compared_genes(case)
    estimated = "edger_top.proportion" in case
    exact = one_way(case["design"]) and same_working_dispersion(case, fit)
    for i, test in enumerate(tests, start=1):
        e = {
            key[len(f"edger_test{i}_") :]: value
            for key, value in case.items()
            if key.startswith(f"edger_test{i}_")
        }
        # The log CPM is at the NB dispersion, which differs by up to 1e-5 when both
        # estimate it.
        assert_allclose(
            test.average_log2_cpm[genes],
            e["logCPM"][genes],
            atol=1e-6 if estimated else 1e-8,
        )
        lfc = e["logFC"].reshape(-1, n_genes).T
        assert_allclose(test.log2_fold_change, lfc, atol=1e-6 if exact else FITS)
        assert scaled(test.f_statistic[genes], e["F"][genes]) < MOMENTS
        assert log10_ratio(test.p_value[genes], e["PValue"][genes]) < PROBABILITY
        assert log10_ratio(test.adjusted_p_value[genes], e["FDR"][genes]) < (
            PROBABILITY
        )
        assert_allclose(test.df_total, e["df.total"], rtol=PRIOR_DF)


@pytest.mark.parametrize("name", CASES)
def test_the_table_holds_the_columns_of_edgers_toptags(fitted, name):
    case, fit, tests = fitted[name]
    n_genes = case["counts"].shape[1]
    genes = compared_genes(case)
    estimated = "edger_top.proportion" in case
    exact = one_way(case["design"]) and same_working_dispersion(case, fit)
    for i, test in enumerate(tests, start=1):
        e = {
            key[len(f"edger_test{i}_") :]: value
            for key, value in case.items()
            if key.startswith(f"edger_test{i}_")
        }
        table = test.table()
        lfc = e["logFC"].reshape(-1, n_genes).T
        changes = (
            ["log2_fold_change"]
            if lfc.shape[1] == 1
            else [f"log2_fold_change_{tested}" for tested in test.names]
        )
        assert table.columns == (
            "feature_id",
            *changes,
            "average_log2_cpm",
            "f_statistic",
            "p_value",
            "adjusted_p_value",
        )
        assert table["feature_id"].tolist() == [f"feature_{j}" for j in range(n_genes)]
        for j, column in enumerate(changes):
            assert_allclose(table[column], lfc[:, j], atol=1e-6 if exact else FITS)
        assert_allclose(
            table["average_log2_cpm"][genes],
            e["logCPM"][genes],
            atol=1e-6 if estimated else 1e-8,
        )
        assert scaled(table["f_statistic"][genes], e["F"][genes]) < MOMENTS
        assert log10_ratio(table["p_value"][genes], e["PValue"][genes]) < PROBABILITY
        assert log10_ratio(table["adjusted_p_value"][genes], e["FDR"][genes]) < (
            PROBABILITY
        )
        ids = [f"g{j}" for j in range(n_genes)]
        assert test.table(feature_ids=ids)["feature_id"].tolist() == ids


# The fixtures where edgeR's average quasi-dispersion is 1.
AVERAGE_ONE = [
    name for name in CASES if load(name)["edger_average.ql.dispersion"][0] == 1.0
]


@pytest.mark.parametrize("name", AVERAGE_ONE)
def test_the_likelihood_ratio_is_edgers_at_the_same_dispersion(fitted, name):
    # Where the average quasi-dispersion is 1 in both, the fits are at dispersions
    # that differ by at most edgeR's search tolerance, so the likelihood ratios agree
    # to it and to edgeR's Levenberg fits, without the moments.
    case, fit, tests = fitted[name]
    assert fit.average_quasi_dispersion == 1.0
    for i, test in enumerate(tests, start=1):
        reference = case[f"edger_test{i}_F"] * case["edger_s2.post"] * test.df_test
        assert scaled(test.likelihood_ratio, reference) < 1e-4


# ---------------------------------------------------------------------------------------
# Moments of the unit deviance
# ---------------------------------------------------------------------------------------


def test_the_moments_match_30_digit_sums():
    reference = load("moments")
    mean, variance = negative_binomial_unit_deviance_moments(
        reference["means"], reference["dispersions"]
    )
    assert_allclose(mean, reference["mean"], rtol=1e-12)
    assert_allclose(variance, reference["variance"], rtol=1e-12)


@pytest.mark.parametrize("mu", [1e14, 1e100])
@pytest.mark.parametrize("phi", [0.01, 0.2, 1.0])
def test_large_means_reach_the_gamma_limit(phi, mu):
    # y / mu tends to a gamma variable with shape 1 / phi; the corrections are about
    # 1 / mu.
    mean, variance = negative_binomial_unit_deviance_moments(mu, phi)
    assert_allclose((mean, variance), gamma_limit(phi), rtol=1e-11)


@pytest.mark.parametrize("mu", [1e6, 1e9, 1e12, 1e100])
def test_poisson_moments_have_their_large_mean_expansion(mu):
    # E d = 1 + 1/(6 mu) + 1/(6 mu^2) + O(mu^-3) and var d = 2 + 2/(3 mu) + O(mu^-2),
    # with the O(mu^-2) term of the variance below 2 / mu^2. A count near the mean,
    # rounded to a float, is off by up to 1e-16 mu, a share 1e-16 sqrt(mu) of the
    # spread of the counts: a deviance taken from it would be off by about that.
    mean, variance = negative_binomial_unit_deviance_moments(mu, 0.0)
    assert abs(mean - (1.0 + 1.0 / (6 * mu) + 1.0 / (6 * mu * mu))) < 1e-14
    assert abs(variance - (2.0 + 2.0 / (3 * mu))) < 2.0 / mu**2 + 2e-14


def test_a_small_dispersion_tends_to_the_poisson_moments():
    means = np.array([1e-6, 0.3, 4.0, 250.0])
    poisson = negative_binomial_unit_deviance_moments(means, 0.0)
    nearly = negative_binomial_unit_deviance_moments(means, 1e-10)
    assert_allclose(nearly.mean, poisson.mean, rtol=1e-8)
    assert_allclose(nearly.variance, poisson.variance, rtol=1e-8)


def test_a_mean_of_zero_has_zero_moments_and_small_means_lose_their_df():
    means = np.array([0.0, 1e-300, 1e-20])
    mean, variance = negative_binomial_unit_deviance_moments(means, 0.2)
    assert mean[0] == variance[0] == 0.0
    # k = 2 E^2 / V tends to zero with the mean, as 2 mu (1 + O(1 / log mu)): the
    # observation carries no degrees of freedom, and its scale grows as log(1 / mu).
    scale, df = _chi_squared_match(means, np.array([0.2]))
    assert df[0] == 0.0 and scale[0] == np.inf
    assert_allclose(df[1:] / (2 * means[1:]), 1.0, rtol=0.05)
    assert_allclose(scale[1:] / np.log(1 / means[1:]), 1.0, rtol=0.05)


@pytest.mark.parametrize("phi", [0.0, 1e-4, 0.15, 3.0])
def test_the_interpolated_table_matches_the_sums(phi):
    # A dispersion shared by every observation uses Chebyshev interpolation; one per
    # observation sums each.
    means = np.exp(np.random.default_rng(3).uniform(-25.0, 16.0, 400))
    table = _chi_squared_match(means, np.array([phi]))
    each = _chi_squared_match(means, np.full(means.size, phi))
    assert_allclose(table[0], each[0], rtol=1e-12)
    assert_allclose(table[1], each[1], rtol=1e-12)


def test_moments_broadcast_and_scalars_give_floats():
    result = negative_binomial_unit_deviance_moments(5.0, 0.1)
    assert isinstance(result, UnitDevianceMoments)
    assert isinstance(result.mean, float) and isinstance(result.variance, float)
    grid = negative_binomial_unit_deviance_moments([[1.0], [10.0]], [0.0, 0.5, 2.0])
    assert grid.mean.shape == (2, 3)
    single = negative_binomial_unit_deviance_moments(10.0, 0.5)
    assert grid.mean[1, 1] == single.mean


# ---------------------------------------------------------------------------------------
# Average log CPM and the common dispersion
# ---------------------------------------------------------------------------------------


def test_average_log2_cpm_without_dispersion_or_prior_pools_the_counts():
    # A Poisson fit of one group: the mean per unit of library size is the ratio of
    # the totals.
    counts = np.array([[3.0, 0.0, 50.0], [7.0, 1.0, 20.0], [1.0, 0.0, 90.0]])
    sizes = np.array([1e6, 2e6, 3e6])
    expected = np.log2(counts.sum(axis=0) / sizes.sum() * 1e6)
    result = average_log2_cpm(
        counts, library_sizes=sizes, prior_count=0.0, dispersion=0.0
    )
    assert_allclose(result[[0, 2]], expected[[0, 2]], rtol=1e-12)
    assert_allclose(result[1], expected[1], rtol=1e-12)


def test_average_log2_cpm_is_the_cpm_of_a_single_sample():
    counts = np.array([[12.0, 0.0, 5.0]])
    result = average_log2_cpm(counts, library_sizes=[2e6], prior_count=0.5)
    prior = 0.5  # one sample: its library size is the mean
    expected = np.log2((counts[0] + prior) / (2e6 + 2 * prior) * 1e6)
    assert_allclose(result, expected, rtol=1e-12)


def test_the_common_dispersion_does_not_depend_on_order_or_library_scale():
    case = load("two_groups")
    counts, design, sizes = case["counts"], case["design"], case["library_sizes"]
    reference = common_dispersion(counts, design, library_sizes=sizes)
    order = np.random.default_rng(1).permutation(counts.shape[0])
    genes = np.random.default_rng(2).permutation(counts.shape[1])
    permuted = common_dispersion(
        counts[order][:, genes], design[order], library_sizes=sizes[order]
    )
    rescaled = common_dispersion(counts, design, library_sizes=sizes * 1e3)
    # The search locates the maximum to about 1e-7, as the rounding of the likelihood
    # allows.
    assert_allclose(permuted, reference, rtol=1e-6)
    assert_allclose(rescaled, reference, rtol=1e-6)


def test_the_dispersion_search_warns_at_the_lower_end():
    # At a Poisson-like maximum the search reaches the lower end and says so.
    rng = np.random.default_rng(4)
    counts = rng.poisson(np.linspace(20.0, 400.0, 60), (6, 60)).astype(float)
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    with pytest.warns(RuntimeWarning, match="lower end of its search interval"):
        phi = common_dispersion(counts, design, interval=(0.05, 1.0))
    assert phi == pytest.approx(0.05, abs=1e-8)


# ---------------------------------------------------------------------------------------
# Invariants of the fit and the tests
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def two_groups():
    case = load("two_groups")
    fit = fit_case(case)
    return case, fit, quasi_likelihood_f_test(fit, coefficients=1)


def test_sample_order_does_not_matter(two_groups):
    case, fit, test = two_groups
    order = np.array([5, 0, 7, 2, 1, 6, 3, 4])
    other = fit_quasi_likelihood(
        case["counts"][order],
        case["design"][order],
        library_sizes=case["library_sizes"][order],
    )
    result = quasi_likelihood_f_test(other, coefficients=1)
    # The search locates the maximum to about 1e-7, as the rounding of the likelihood
    # allows.
    assert_allclose(other.dispersion, fit.dispersion, rtol=1e-6)
    assert_allclose(other.average_quasi_dispersion, fit.average_quasi_dispersion)
    assert_allclose(other.leverage, np.asarray(fit.leverage)[order], atol=1e-6)
    assert_allclose(result.f_statistic, test.f_statistic, rtol=1e-5)
    assert_allclose(result.p_value, test.p_value, rtol=1e-4)


def test_library_sizes_scale_only_the_log2_cpm(two_groups):
    case, fit, test = two_groups
    other = fit_quasi_likelihood(
        case["counts"], case["design"], library_sizes=case["library_sizes"] * 10.0
    )
    result = quasi_likelihood_f_test(other, coefficients=1)
    assert_allclose(other.dispersion, fit.dispersion, rtol=1e-6)
    assert_allclose(
        other.posterior_quasi_dispersion, fit.posterior_quasi_dispersion, rtol=1e-6
    )
    assert_allclose(result.f_statistic, test.f_statistic, rtol=1e-5)
    assert_allclose(result.log2_fold_change, test.log2_fold_change, atol=1e-7)
    # Up to the prior counts, which do not grow with the libraries: about c / N.
    shifted = test.average_log2_cpm - math.log2(10.0)
    assert_allclose(result.average_log2_cpm, shifted, atol=1e-5)


def test_a_gene_counted_in_one_condition_keeps_the_invariances(fitted):
    # One gene of the paired case is counted in the first condition only: its fitted
    # values in the second tend to zero, and the fit reproduces its counted samples.
    # Relabelling the pairs, reordering the samples and scaling the libraries leave
    # the model as it was. The rounding of the exact fits once moved the average
    # quasi-dispersion by up to 6% under each.
    case, fit, _ = fitted["paired"]
    counts, design = case["counts"], case["design"]
    second = design[:, 4] == 1
    alone = np.flatnonzero((counts[second] == 0).all(axis=0))
    assert alone.size == 1 and (counts[~second, alone[0]] > 0).all()
    reference = quasi_likelihood_f_test(fit, coefficients=4)
    sizes = counts.sum(axis=1)
    pairs = np.array([2, 3, 0, 1, 6, 7, 4, 5])
    order = np.array([5, 0, 7, 2, 1, 6, 3, 4])
    for other in (
        fit_quasi_likelihood(counts[pairs], design, library_sizes=sizes[pairs]),
        fit_quasi_likelihood(counts[order], design[order], library_sizes=sizes[order]),
        fit_quasi_likelihood(counts, design, library_sizes=sizes * 10.0),
    ):
        result = quasi_likelihood_f_test(other, coefficients=4)
        # The dispersion search locates its maximum to about 1e-7.
        assert_allclose(
            other.average_quasi_dispersion, fit.average_quasi_dispersion, rtol=1e-6
        )
        assert_allclose(
            other.posterior_quasi_dispersion, fit.posterior_quasi_dispersion, rtol=1e-6
        )
        assert_allclose(result.f_statistic, reference.f_statistic, rtol=1e-5)


def test_a_reparametrised_design_gives_the_same_tests(two_groups):
    case, fit, test = two_groups
    means = np.column_stack([1 - case["design"][:, 1], case["design"][:, 1]])
    other = fit_quasi_likelihood(
        case["counts"], means, library_sizes=case["library_sizes"]
    )
    result = quasi_likelihood_f_test(other, contrasts=[-1.0, 1.0])
    assert_allclose(
        other.posterior_quasi_dispersion, fit.posterior_quasi_dispersion, rtol=1e-8
    )
    assert_allclose(result.f_statistic, test.f_statistic, rtol=1e-8)
    assert_allclose(result.log2_fold_change, test.log2_fold_change, atol=1e-9)


def test_equivalent_contrasts_give_the_same_f_test(fitted):
    case, fit, tests = fitted["groups_covariate"]
    joint = case["contrast_joint"]
    # The same null hypothesis: scaled columns, and their sum next to one of them.
    combined = np.column_stack([2.0 * joint[:, 0], joint[:, 0] + joint[:, 1]])
    reference = quasi_likelihood_f_test(fit, contrasts=joint)
    other = quasi_likelihood_f_test(fit, contrasts=combined)
    assert reference.df_test == other.df_test == 2
    assert_allclose(other.f_statistic, reference.f_statistic, rtol=1e-6)
    # A coefficient test is the test of its unit contrast.
    unit = quasi_likelihood_f_test(fit, contrasts=[0.0, 0.0, 0.0, 1.0])
    assert_allclose(unit.f_statistic, tests[0].f_statistic, rtol=1e-6)
    assert_allclose(unit.log2_fold_change, tests[0].log2_fold_change)
    # A contrast that repeats one is no more test.
    repeated = quasi_likelihood_f_test(fit, contrasts=np.column_stack([joint, joint]))
    assert repeated.df_test == 2
    assert_allclose(repeated.f_statistic, reference.f_statistic, rtol=1e-6)


def test_named_contrasts_of_a_design_are_matched_by_name():
    rng = np.random.default_rng(6)
    samples = {"group": np.repeat(["A", "B", "C"], 3)}
    design = Design(samples, "~ 0 + group")
    counts = rng.negative_binomial(5, 5 / (5 + np.linspace(5, 300, 30)), (9, 30))
    fit = fit_quasi_likelihood(counts.astype(float), design, dispersion=0.2)
    contrasts = make_contrasts({"b_vs_a": "groupB - groupA"}, design)
    test = quasi_likelihood_f_test(fit, contrasts=contrasts)
    by_weights = quasi_likelihood_f_test(fit, contrasts=[-1.0, 1.0, 0.0])
    assert test.names == ("b_vs_a",)
    assert_allclose(test.f_statistic, by_weights.f_statistic)
    by_name = quasi_likelihood_f_test(fit, coefficients=["groupB", "groupC"])
    assert by_name.names == ("groupB", "groupC")


def test_a_gene_without_counts_is_not_tested():
    rng = np.random.default_rng(7)
    counts = rng.negative_binomial(5, 5 / (5 + np.linspace(5, 300, 30)), (6, 30))
    counts = counts.astype(float)
    counts[:, 3] = 0.0
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    fit = fit_quasi_likelihood(counts, design, dispersion=0.2)
    test = quasi_likelihood_f_test(fit)
    assert fit.df_residual_adjusted[3] == 0.0
    assert math.isnan(fit.quasi_dispersion[3])
    assert fit.posterior_quasi_dispersion[3] == pytest.approx(
        np.broadcast_to(fit.prior_quasi_dispersion, 30)[3]
    )
    assert test.f_statistic[3] == 0.0 and test.p_value[3] == 1.0
    assert np.isfinite(test.log2_fold_change[3, 0])


def test_weights_per_sample_are_the_same_weights_per_observation(two_groups):
    case, _, _ = two_groups
    counts, design, sizes = case["counts"], case["design"], case["library_sizes"]
    w = np.array([0.5, 1.0, 2.0, 1.5, 1.0, 0.8, 1.2, 2.5])
    by_sample = fit_quasi_likelihood(counts, design, library_sizes=sizes, weights=w)
    tiled = np.tile(w[:, None], (1, counts.shape[1]))
    by_observation = fit_quasi_likelihood(
        counts, design, library_sizes=sizes, weights=tiled
    )
    assert by_observation.dispersion == pytest.approx(by_sample.dispersion, rel=1e-12)
    assert_allclose(by_observation.leverage, by_sample.leverage, atol=1e-12)
    assert_allclose(
        by_observation.posterior_quasi_dispersion,
        by_sample.posterior_quasi_dispersion,
        rtol=1e-10,
    )


def test_offsets_are_log_library_sizes(two_groups):
    case, fit, test = two_groups
    counts, design, sizes = case["counts"], case["design"], case["library_sizes"]
    tiled = np.tile(np.log(sizes)[:, None], (1, counts.shape[1]))
    for offsets in (np.log(sizes), tiled):
        other = fit_quasi_likelihood(counts, design, offsets=offsets)
        result = quasi_likelihood_f_test(other, coefficients=1)
        assert other.dispersion == pytest.approx(fit.dispersion, rel=1e-10)
        assert_allclose(result.f_statistic, test.f_statistic, rtol=1e-9)
        assert_allclose(result.average_log2_cpm, test.average_log2_cpm, atol=1e-10)


def test_a_condition_without_counts_gets_the_leverages_of_the_limit():
    # A gene without counts in one condition of a paired design: its fitted values
    # there tend to zero in proportion to w_d = s_0d y_1d / s_1d (library sizes s),
    # since each donor's other sample is fitted exactly, and their leverages tend to
    # w_d / sum(w), which sum to the one coefficient that they alone determine.
    # Rounding loses them where that condition is the baseline and the fitted values
    # are below about 1e-30 of the others, as here.
    rng = np.random.default_rng(10)
    condition = np.tile([0.0, 1.0], 4)
    donors = [np.repeat(np.arange(4), 2) == d for d in range(1, 4)]
    sizes = rng.uniform(1e6, 3e6, 8)
    mean = np.linspace(20.0, 400.0, 30)
    counts = rng.negative_binomial(5, 5 / (5 + mean), (8, 30)).astype(float)
    counts[:, 0] = [0, 1200, 0, 3000, 0, 700, 0, 2100]
    level = sizes[condition == 0] * counts[condition == 1, 0] / sizes[condition == 1]
    for coded in (condition, 1.0 - condition):
        design = np.column_stack([np.ones(8), *donors, coded])
        fit = fit_quasi_likelihood(counts, design, dispersion=0.3, library_sizes=sizes)
        assert_allclose(fit.leverage[condition == 0, 0], level / level.sum(), rtol=1e-6)
        assert_allclose(fit.leverage[condition == 1, 0], 1.0, rtol=1e-9)
        assert fit.unit_df_adjusted[condition == 0, 0].max() < 1e-12
        # The same limit for weights given directly, whatever their size below 1e-12
        # of the others; weights of exactly zero carry no leverage.
        base = rng.uniform(1.0, 3.0, 8)
        for scale in (1e-20, 1e-60, 0.0):
            weights = np.where(condition == 0, scale * base, base)[None, :]
            leverage = _leverages(np.ascontiguousarray(design), weights)[0]
            share = base[condition == 0] / base[condition == 0].sum()
            assert_allclose(leverage[condition == 0], share * (scale > 0), atol=1e-12)
            assert_allclose(leverage[condition == 1], 1.0, rtol=1e-12)


def test_the_fit_keeps_its_own_counts(small):
    counts, design, _ = small
    mine = counts.copy()
    fit = fit_quasi_likelihood(mine, design, dispersion=0.1)
    before = quasi_likelihood_f_test(fit).f_statistic
    mine *= 2.0
    assert_array_equal(quasi_likelihood_f_test(fit).f_statistic, before)


def test_threads_do_not_change_the_result(two_groups):
    case, fit, _ = two_groups
    one = fit_quasi_likelihood(
        case["counts"],
        case["design"],
        library_sizes=case["library_sizes"],
        n_threads=1,
    )
    assert one.dispersion == fit.dispersion
    assert_array_equal(one.posterior_quasi_dispersion, fit.posterior_quasi_dispersion)


def test_the_options_of_the_trend_shape_the_prior(two_groups):
    case, _, _ = two_groups
    counts, design = case["counts"], case["design"]
    flat = fit_quasi_likelihood(counts, design, dispersion=0.1, trend=False)
    assert isinstance(flat.prior_quasi_dispersion, float)
    assert flat.covariate is None and flat.top_proportion is None
    # The default trend is the trend in its covariate, given as one.
    trended = fit_quasi_likelihood(counts, design, dispersion=0.1)
    given = fit_quasi_likelihood(
        counts, design, dispersion=0.1, trend=trended.covariate
    )
    assert_array_equal(given.prior_quasi_dispersion, trended.prior_quasi_dispersion)
    assert_array_equal(
        given.posterior_quasi_dispersion, trended.posterior_quasi_dispersion
    )


def test_results_are_read_only_and_described(two_groups):
    _, fit, test = two_groups
    assert isinstance(fit, QuasiLikelihoodFit) and isinstance(test, QuasiLikelihoodTest)
    for values in (fit.deviance_adjusted, fit.leverage, test.f_statistic, test.p_value):
        with pytest.raises(ValueError, match="read-only"):
            values[0] = 0.0
    assert "QuasiLikelihoodFit(n_features=40" in repr(fit)
    assert "QuasiLikelihoodTest(n_features=40" in repr(test)
    assert fit.coefficient_names == fit.fit.coefficient_names


# ---------------------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small():
    rng = np.random.default_rng(8)
    counts = rng.poisson(np.linspace(10.0, 200.0, 20), (6, 20)).astype(float)
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    return counts, design, fit_quasi_likelihood(counts, design, dispersion=0.1)


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        ({"dispersion": 0.1, "top_proportion": 0.5}, ValueError, "not both"),
        ({"top_proportion": 1.5}, ValueError, r"top_proportion must be in \(0, 1\]"),
        ({"robust": 1}, TypeError, "robust must be True or False"),
        ({"trend": [1.0, 2.0]}, ValueError, "one covariate value per feature"),
        ({"trend": ["a"] * 20}, TypeError, "trend must be True, False or real"),
        ({"dispersion": -0.1}, ValueError, "dispersion must be finite"),
        ({"dispersion": 2e4}, ValueError, "dispersion must be at most 10000"),
        ({"prior_count": -1.0}, ValueError, "prior_count must be non-negative"),
    ],
)
def test_invalid_fits_are_refused(small, keywords, error, match):
    counts, design, _ = small
    with pytest.raises(error, match=match):
        fit_quasi_likelihood(counts, design, **keywords)


def test_quasi_dispersions_without_df_cannot_be_moderated(small):
    # Weights this small leave every observation a df of about 1e-8 or less, below
    # the 0.01 that limma's prior needs.
    counts, design, _ = small
    with pytest.raises(ValueError, match="cannot be moderated"):
        fit_quasi_likelihood(counts, design, dispersion=0.1, weights=1e-8)


def test_a_design_without_residual_df_is_refused(small):
    counts, _, _ = small
    with pytest.raises(ValueError, match="no residual degrees of freedom"):
        fit_quasi_likelihood(counts, np.eye(6), dispersion=0.1)


@pytest.mark.parametrize(
    ("keywords", "error", "match"),
    [
        ({"coefficients": "x3"}, ValueError, "no coefficient named 'x3'"),
        ({"coefficients": 5}, ValueError, "out of range"),
        ({"coefficients": [1, 1]}, ValueError, "more than once"),
        ({"coefficients": 1.0}, TypeError, "a name, an integer position"),
        ({"coefficients": 1, "contrasts": [0, 1]}, ValueError, "not both"),
        ({"contrasts": [0.0, 0.0]}, ValueError, "test nothing"),
        ({"contrasts": [0.0, np.inf]}, ValueError, "finite"),
        ({"coefficients": [0, 1]}, ValueError, "without coefficients"),
        ({"names": ["a"]}, ValueError, "not used with coefficients"),
    ],
)
def test_invalid_tests_are_refused(small, keywords, error, match):
    _, _, fit = small
    with pytest.raises(error, match=match):
        quasi_likelihood_f_test(fit, **keywords)


def test_a_fit_of_another_kind_is_refused(small):
    counts, design, _ = small
    other = fit_negative_binomial_model(counts, design, dispersion=0.1)
    with pytest.raises(TypeError, match="fit must be a QuasiLikelihoodFit"):
        quasi_likelihood_f_test(other)


@pytest.mark.parametrize(
    ("keywords", "match"),
    [
        ({"interval": (1.0, 0.5)}, "0 <= low < high"),
        ({"interval": (0.0, np.inf)}, "0 <= low < high"),
        ({"subset": 0}, "subset must be at least 1"),
        ({"min_row_sum": 1e9}, "no gene has at least"),
    ],
)
def test_invalid_dispersion_searches_are_refused(small, keywords, match):
    counts, design, _ = small
    with pytest.raises(ValueError, match=match):
        common_dispersion(counts, design, **keywords)


@pytest.mark.parametrize(
    ("means", "dispersion", "match"),
    [
        (-1.0, 0.1, "means must be finite and non-negative"),
        (1e101, 0.1, "means must be at most 1e[+]100"),
        (1.0, 2e4, "dispersion must be at most 10000"),
        ([1.0, 2.0], [0.1, 0.2, 0.3], "do not broadcast"),
    ],
)
def test_invalid_moments_are_refused(means, dispersion, match):
    with pytest.raises(ValueError, match=match):
        negative_binomial_unit_deviance_moments(means, dispersion)


def test_the_dispersion_search_warns_at_the_upper_end():
    rng = np.random.default_rng(9)
    mean = np.linspace(50.0, 500.0, 40)
    counts = rng.negative_binomial(0.25, 0.25 / (0.25 + mean), (6, 40)).astype(float)
    design = np.column_stack([np.ones(6), [0, 0, 0, 1, 1, 1]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        phi = common_dispersion(counts, design, interval=(0.0, 0.5))
    assert phi == pytest.approx(0.5, rel=1e-6)
    assert any("upper end" in str(item.message) for item in caught)
