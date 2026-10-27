"""Tests for DESeq2's dispersion estimates, fits, and Wald and likelihood-ratio tests.

Reference values in ``tests/data/deseq2_tests`` come from DESeq2 1.46.0, run by
``validation/deseq2_tests.py`` on 150 simulated genes per case: its estimates at every
stage; its fits at its final dispersions with ``betaTol = 1e-14``, and the same fits
polished by Newton's method in R; its tests at those fits, with other options of
``results()`` and with twenty filter statistics; its Cook's distances and flags at given
dispersions under four designs; the maxima of the Cox-Reid adjusted log-likelihood
found in R by a grid and the root of its derivative; the maximum likelihood estimate of
the prior's variance by integration on a grid; and mpmath derivatives of a count's
log-likelihood. Every stage is given DESeq2's own inputs to it. The other tests check
closed forms and properties that a wrong implementation would break: the likelihood
ratio test of one coefficient against its Wald test as the samples grow, invariance
under reparametrization and the order of the samples, the independence of a gene's
estimate from the other genes, the rules found by probing, and the errors.
"""

import json
import warnings
from functools import cache
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool import Design
from cytepool.stats import deseq2
from cytepool.stats.deseq2 import (
    DESeq2Dispersions,
    DESeq2Fit,
    DESeq2Results,
    _count_score,
    _independent_filtering,
    deseq2_dispersions,
    deseq2_likelihood_ratio_test,
    deseq2_wald_test,
    fit_deseq2,
)

DATA = Path(__file__).parent / "data" / "deseq2_tests"
CASES = (
    "two_groups",
    "several_groups",
    "covariate",
    "paired",
    "small_counts",
    "two_df",
    "outlier_sample",
    "rising_dispersion",
    "near_poisson",
)
TESTS = [
    (case, i)
    for case in CASES
    for i in range(len(json.loads(str(np.load(DATA / f"{case}.npz")["tests"]))))
]
RULES = [
    (case, i)
    for case in ("cooks_two_groups", "cooks_unbalanced")
    for i in range(len(json.loads(str(np.load(DATA / f"{case}.npz")["formulas"]))))
]
# The degrees of freedom of each likelihood ratio test, by hand: four groups against
# an intercept, and one coefficient (a group, a condition) left out.
DF = {("several_groups", "~ 1"): 3, ("covariate", "~ group"): 1}
DF[("paired", "~ donor")] = 1


@cache
def fixture(case: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{case}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def reference(case: str, key: str) -> np.ndarray:
    return fixture(case)[f"R {key}"]


def counts(case: str) -> np.ndarray:
    return fixture(case)["counts"].astype(float)


def samples(case: str) -> dict[str, np.ndarray]:
    data = fixture(case)
    return {k[8:]: v for k, v in data.items() if k.startswith("samples ")}


def design(case: str, formula: str | None = None) -> Design:
    data = fixture(case)
    levels = {k[7:]: list(v) for k, v in data.items() if k.startswith("levels ")}
    return Design(samples(case), formula or str(data["formula"]), levels=levels)


def per_gene(case: str, key: str, columns: int) -> np.ndarray:
    """An R matrix of genes x columns, written column by column."""
    return reference(case, key).reshape(-1, columns, order="F")


def trend(case: str) -> str:
    return str(fixture(case)["fit_type"])


@cache
def estimates(case: str) -> DESeq2Dispersions:
    return deseq2_dispersions(
        counts(case),
        design(case),
        size_factors=reference(case, "size_factors"),
        trend=trend(case),
    )


@cache
def staged(case: str) -> DESeq2Dispersions:
    """The maximum a posteriori stage, given DESeq2's estimates before it."""
    return deseq2_dispersions(
        counts(case),
        design(case),
        size_factors=reference(case, "size_factors"),
        genewise_dispersions=reference(case, "dispGeneEst"),
        trended_dispersions=reference(case, "dispFit"),
        prior_variance=float(reference(case, "prior_var")[0]),
    )


@cache
def fitted(case: str) -> DESeq2Fit:
    """The fits at DESeq2's final dispersions and size factors."""
    return fit_deseq2(
        counts(case),
        design(case),
        size_factors=reference(case, "size_factors"),
        dispersion=reference(case, "dispersion"),
    )


def wald_contrast(case: str) -> np.ndarray:
    tests = json.loads(str(fixture(case)["tests"]))
    return np.array(next(value for kind, value in tests if kind == "wald"))


def relative(actual: np.ndarray, desired: np.ndarray, floor: float = 0.0) -> float:
    """Largest |actual - desired| / max(|desired|, floor) where desired is finite."""
    keep = np.isfinite(desired)
    assert np.isfinite(actual[keep]).all()
    scale = np.maximum(np.abs(desired[keep]), max(floor, 1e-300))
    return float(np.max(np.abs(actual[keep] - desired[keep]) / scale, initial=0.0))


def log_relative(actual: np.ndarray, desired: np.ndarray) -> float:
    """Largest |log actual - log desired| / max(|log desired|, 1) where desired is
    finite, for p-values, which below 1e-300 count as 1e-300: an error e in a statistic
    z moves log p by about z^2 e."""
    keep = np.isfinite(desired)
    assert np.isfinite(actual[keep]).all()
    a, d = (np.log(np.maximum(v[keep], 1e-300)) for v in (actual, desired))
    return float(np.max(np.abs(a - d) / np.maximum(np.abs(d), 1.0), initial=0.0))


def simulated(
    n_per_group: int, n_genes: int, seed: int, effect: float = 0.0
) -> tuple[np.ndarray, Design, np.ndarray]:
    """Two groups of negative binomial counts with known dispersions."""
    rng = np.random.default_rng(seed)
    group = np.repeat(["A", "B"], n_per_group)
    mean = rng.uniform(30.0, 300.0, n_genes)
    alpha = rng.uniform(0.05, 0.2, n_genes)
    log_mean = np.log(mean) + effect * (group == "B")[:, None]
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + np.exp(log_mean))).astype(float)
    return y, Design({"group": group}, "~ group"), alpha


# Against DESeq2.


@pytest.mark.parametrize("case", CASES)
def test_means_of_the_likelihood_match_deseq2(case: str) -> None:
    # DESeq2's own for one-way layouts; for the covariate and paired designs its fit at
    # the rough dispersions, polished by Newton's method: DESeq2 stops that fit early.
    means = per_gene(case, "mu_ref", counts(case).shape[0]).T

    assert relative(estimates(case).means, means) < 1e-10


@pytest.mark.parametrize("case", CASES)
def test_genewise_dispersions_are_the_maxima(case: str) -> None:
    d = estimates(case)
    maxima = reference(case, "genewise_ref")
    held = reference(case, "rough") <= 1e-8
    # Below 1e-6 the adjusted likelihood is flat in the log dispersion.
    curved = ~held & (maxima >= 1e-6)

    assert curved.sum() >= 10
    # Both found to brackets of about 1e-12 on the log scale.
    assert relative(d.genewise[curved], maxima[curved]) < 1e-9
    assert np.all(d.genewise[~held & ~curved & ~d.all_zero] < 1e-6)
    # A gene whose rough dispersion is at the lower bound keeps it, as in DESeq2, even
    # where the maximum is far above it.
    assert held.sum() >= 10 and np.any(maxima[held] > 1e-4)
    assert np.all(d.genewise[held] == 1e-8)
    assert np.all(reference(case, "dispGeneEst")[held] == 1e-8)


@pytest.mark.parametrize("case", CASES)
def test_trend_and_prior_match_deseq2_from_its_genewise_estimates(case: str) -> None:
    y, x = counts(case), design(case)
    sizes, genewise = reference(case, "size_factors"), reference(case, "dispGeneEst")
    if trend(case) == "parametric":
        d = deseq2_dispersions(y, x, size_factors=sizes, genewise_dispersions=genewise)
        # DESeq2's gamma fits, glm()'s iteration from the coefficients before.
        coefficients = [d.asymptotic_dispersion, d.extra_poisson]
        assert_allclose(coefficients, reference(case, "trend"), rtol=1e-12)
        assert d.trend == "parametric"
        assert_allclose(d.trended, reference(case, "dispFit"), rtol=1e-12)
    else:
        # The parametric trend fails, as DESeq2's does; DESeq2 then substitutes a local
        # regression, CytePool the mean.
        with pytest.warns(RuntimeWarning, match="parametric dispersion trend"):
            d = deseq2_dispersions(
                y, x, size_factors=sizes, genewise_dispersions=genewise
            )
        assert d.trend == "mean"
        assert d.extra_poisson == 0.0
        assert_allclose(d.trended, reference(case, "dispFit"), rtol=1e-14)
    given = deseq2_dispersions(
        y,
        x,
        size_factors=sizes,
        genewise_dispersions=genewise,
        trended_dispersions=reference(case, "dispFit"),
    )
    assert given.trend == "given"
    assert_allclose(given.log_residual_variance, reference(case, "var_log"), rtol=1e-12)
    if given.df_residual > 3:
        assert_allclose(given.prior_variance, reference(case, "prior_var"), rtol=1e-12)
    else:
        # DESeq2 simulates; CytePool maximizes the likelihood of the same model, which
        # the grid integration reproduces to about 1e-8.
        mle = float(fixture(case)["mle_prior_variance"])
        assert_allclose(given.prior_variance, mle, rtol=1e-6)
        assert given.prior_variance != pytest.approx(reference(case, "prior_var")[0])


def test_the_trend_is_fitted_to_the_genes_above_1e_6() -> None:
    # Most nearly Poisson genes keep the lower bound. DESeq2 leaves every gene at or
    # below 1e-6 out of the fit (found by probing): moving them up to 1e-6 changes
    # nothing, and moving them above it changes the trend or makes it fail.
    y, x = counts("near_poisson"), design("near_poisson")
    sizes = reference("near_poisson", "size_factors")
    genewise = reference("near_poisson", "dispGeneEst")
    low = genewise <= 1e-6
    assert low.sum() >= 50

    def trend_of(values: np.ndarray) -> DESeq2Dispersions:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return deseq2_dispersions(
                y, x, size_factors=sizes, genewise_dispersions=values
            )

    found, at, above = (
        trend_of(np.where(low, v, genewise)) for v in (1e-8, 1e-6, 2e-6)
    )
    assert found.trend == at.trend == "parametric"
    coefficients = [found.asymptotic_dispersion, found.extra_poisson]
    assert_allclose(coefficients, reference("near_poisson", "trend"), rtol=1e-12)
    assert at.asymptotic_dispersion == found.asymptotic_dispersion
    assert at.extra_poisson == found.extra_poisson
    assert above.trend != "parametric" or not np.isclose(
        above.asymptotic_dispersion, found.asymptotic_dispersion, rtol=1e-3
    )


@pytest.mark.parametrize("case", CASES)
def test_map_dispersions_are_the_maxima(case: str) -> None:
    d = staged(case)
    counted = ~d.all_zero
    flags = reference(case, "dispOutlier") == 1.0

    assert relative(d.map_estimate, reference(case, "map_ref")) < 1e-9
    # R gives NA for genes without counts.
    assert_array_equal(d.outlier[counted], flags[counted])
    assert not d.outlier[~counted].any()
    # Outliers keep DESeq2's gene-wise estimate, the others take the maxima found in R.
    final = np.where(flags, reference(case, "dispGeneEst"), reference(case, "map_ref"))
    assert relative(d.dispersion, final) < 1e-9


@pytest.mark.parametrize("case", CASES)
def test_fits_match_deseq2(case: str) -> None:
    fit = fitted(case)
    p = fit.design.shape[1]
    n = fit.counts.shape[0]

    # DESeq2's fits with betaTol = 1e-14, polished by Newton's method in R.
    assert relative(fit.coefficients, per_gene(case, "polished_coef", p), 1.0) < 1e-10
    assert relative(fit.standard_errors, per_gene(case, "polished_se", p)) < 1e-10
    # The deviance is stationary at the fixed point: DESeq2's tight fits, which stop
    # short of it, agree to about 1e-11.
    deviance = reference(case, "tight_deviance")
    assert relative(fit.minus_twice_log_likelihood, deviance) < 1e-9
    # Their coefficients, up to about 2e-6 from the fixed point in these genes, move
    # what is computed from them as far.
    cooks = per_gene(case, "tight_cooks", n).T
    assert relative(fit.cooks_distances, cooks, 1.0) < 1e-5
    max_cooks = reference(case, "tight_maxCooks")
    assert_array_equal(np.isnan(fit.max_cooks), np.isnan(max_cooks))
    if np.isfinite(max_cooks).any():
        assert relative(fit.max_cooks, max_cooks, 1.0) < 1e-5
    assert fit.converged.all()


@pytest.mark.parametrize(("case", "index"), TESTS)
def test_tests_match_deseq2(case: str, index: int) -> None:
    kind, value = json.loads(str(fixture(case)["tests"]))[index]
    fit = fitted(case)
    if kind == "wald":
        result = deseq2_wald_test(fit, np.array(value))
        assert result.test == "Wald" and result.df is None
        assert result.reduced_converged is None
        # Statistics as the fits, within DESeq2's stopping error.
        tolerance = 1e-5
    else:
        result = deseq2_likelihood_ratio_test(fit, design(case, value))
        assert result.test == "likelihood ratio" and result.df == DF[(case, value)]
        assert result.reduced_converged is not None and result.reduced_converged.all()
        # Differences of deviances, which are stationary at the fixed points.
        tolerance = 1e-9
    assert result.converged.all()
    tight = {
        k: reference(case, f"tight{index + 1}_{k}")
        for k in ("log2FoldChange", "lfcSE", "stat", "pvalue", "padj")
    }

    assert relative(result.log2_fold_change, tight["log2FoldChange"], 1.0) < 1e-5
    assert relative(result.standard_error, tight["lfcSE"]) < 1e-5
    assert relative(result.statistic, tight["stat"], 1.0) < tolerance
    assert log_relative(result.p_value, tight["pvalue"]) < tolerance
    assert log_relative(result.adjusted_p_value, tight["padj"]) < tolerance
    # Cook's distances flag and independent filtering removes the same genes.
    assert_array_equal(np.isnan(result.p_value), np.isnan(tight["pvalue"]))
    assert_array_equal(np.isnan(result.adjusted_p_value), np.isnan(tight["padj"]))
    assert result.filter_quantile == reference(case, f"tight{index + 1}_theta")[0]
    assert_allclose(
        result.base_mean, reference(case, f"tight{index + 1}_baseMean"), rtol=1e-13
    )


def test_the_table_holds_the_columns_of_deseq2s_results() -> None:
    case = CASES[0]
    kind, value = json.loads(str(fixture(case)["tests"]))[0]
    assert kind == "wald"
    result = deseq2_wald_test(fitted(case), np.array(value))
    n_genes = result.p_value.size
    ids = [f"gene{j}" for j in range(n_genes)]
    table = result.table(feature_ids=ids)
    assert table.columns == (
        "feature_id",
        "base_mean",
        "log2_fold_change",
        "standard_error",
        "statistic",
        "p_value",
        "adjusted_p_value",
    )
    assert table["feature_id"].tolist() == ids
    # Each column is DESeq2's, in the order of the genes, within its stopping error.
    tight = {
        k: reference(case, f"tight1_{k}")
        for k in ("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")
    }
    assert_allclose(table["base_mean"], tight["baseMean"], rtol=1e-13)
    assert relative(table["log2_fold_change"], tight["log2FoldChange"], 1.0) < 1e-5
    assert relative(table["standard_error"], tight["lfcSE"]) < 1e-5
    assert relative(table["statistic"], tight["stat"], 1.0) < 1e-5
    assert log_relative(table["p_value"], tight["pvalue"]) < 1e-5
    assert log_relative(table["adjusted_p_value"], tight["padj"]) < 1e-5
    assert_array_equal(np.isnan(table["adjusted_p_value"]), np.isnan(tight["padj"]))
    default = result.table()
    assert default["feature_id"].tolist() == [f"feature_{j}" for j in range(n_genes)]
    with pytest.raises(ValueError, match=rf"one id per feature \({n_genes}\), got"):
        result.table(feature_ids=["a", "b"])
    with pytest.raises(TypeError, match="feature_ids must hold text ids, got int 0"):
        result.table(feature_ids=list(range(n_genes)))


@pytest.mark.parametrize("case", CASES)
def test_results_options_match_deseq2(case: str) -> None:
    fit, contrast = fitted(case), wald_contrast(case)
    cut = deseq2_wald_test(fit, contrast, cooks_cutoff=0.5)
    unfiltered = deseq2_wald_test(fit, contrast, independent_filtering=False)
    pvalue, padj = reference(case, "cutoff_pvalue"), reference(case, "cutoff_padj")

    assert cut.cooks_cutoff == 0.5
    assert_array_equal(np.isnan(cut.p_value), np.isnan(pvalue))
    assert_array_equal(np.isnan(cut.adjusted_p_value), np.isnan(padj))
    assert log_relative(cut.adjusted_p_value, padj) < 1e-5
    desired = reference(case, "unfiltered_padj")
    assert_array_equal(np.isnan(unfiltered.adjusted_p_value), np.isnan(desired))
    assert log_relative(unfiltered.adjusted_p_value, desired) < 1e-5
    assert np.isnan(unfiltered.filter_quantile) and unfiltered.rejections.size == 0


def test_a_cooks_cutoff_of_false_or_infinity_removes_no_p_value() -> None:
    fit = fitted("outlier_sample")
    default = deseq2_wald_test(fit)
    none = deseq2_wald_test(fit, cooks_cutoff=False)
    infinite = deseq2_wald_test(fit, cooks_cutoff=float("inf"))

    assert default.cooks_outlier.sum() >= 3
    assert none.cooks_cutoff == np.inf and not none.cooks_outlier.any()
    assert_array_equal(none.p_value, infinite.p_value)
    flagged = default.cooks_outlier
    assert_array_equal(none.p_value[~flagged], default.p_value[~flagged])
    assert np.all(np.isfinite(none.p_value[flagged]))


@pytest.mark.parametrize(("case", "index"), RULES)
def test_cooks_distance_flags_match_deseq2(case: str, index: int) -> None:
    # Outliers, some with other samples at or next to their count, at given dispersions:
    # in a comparison of two groups (a factor of two levels, with or without an
    # intercept) DESeq2 spares a gene when three or more samples have larger counts
    # than the sample whose Cook's distance is largest, over every sample; not for a
    # numeric covariate.
    formula, weights = json.loads(str(fixture(case)["formulas"]))[index]
    data = fixture(case)
    x = Design(samples(case), formula, levels={"group": ["A", "B"]})
    y = counts(case)
    fit = fit_deseq2(
        y,
        x,
        size_factors=data["size_factors"],
        dispersion=np.full(y.shape[1], 0.05),
    )
    result = deseq2_wald_test(fit, np.array(weights))
    desired = reference(case, f"pvalue{index + 1}")
    above = fit.max_cooks > result.cooks_cutoff

    assert fit.two_groups == (formula != "~ treated")
    assert np.isnan(desired).sum() >= 5
    assert_array_equal(np.isnan(result.p_value), np.isnan(desired))
    assert_array_equal(result.cooks_outlier, np.isnan(desired))
    assert relative(fit.max_cooks, reference(case, f"maxCooks{index + 1}"), 1.0) < 1e-9
    if fit.two_groups:
        assert np.count_nonzero(above & ~result.cooks_outlier) >= 1
    else:
        assert_array_equal(result.cooks_outlier, above)


def test_the_default_cooks_cutoff_is_the_99_percent_point_of_f() -> None:
    # qf(0.99, 2, 6) and qf(0.99, 2, 5) in R.
    cutoffs = {
        "cooks_two_groups": 10.924766500838333,
        "cooks_unbalanced": 13.273933612004827,
    }
    for case, value in cutoffs.items():
        x = Design(samples(case), "~ group", levels={"group": ["A", "B"]})
        y = counts(case)
        fit = fit_deseq2(y, x, dispersion=np.full(y.shape[1], 0.05))
        assert_allclose(deseq2_wald_test(fit).cooks_cutoff, value, rtol=1e-12)


def test_independent_filtering_matches_deseq2_on_twenty_filters() -> None:
    # Statistics whose rejection curves take every shape, among them flat ones where
    # the quantile comes from the fallbacks to 90% and 80% of the fitted maximum.
    p_value = reference("two_groups", "tight1_pvalue")
    filters = fixture("two_groups")["filters"]
    for k in range(filters.shape[1]):
        adjusted, _, theta, _, _, _ = _independent_filtering(
            p_value, filters[:, k], 0.1
        )
        desired = reference("two_groups", f"filter{k + 1}_padj")
        assert theta == reference("two_groups", f"filter{k + 1}_theta")[0]
        assert_array_equal(np.isnan(adjusted), np.isnan(desired))
        assert relative(adjusted, desired) < 1e-14


def test_count_score_matches_mpmath() -> None:
    data = np.load(DATA / "count_score.npz")
    points, exact = data["points"], data["mpmath"]
    ours = np.array([_count_score(*point) for point in points])
    y, mu, alpha = points.T
    # Relative to the larger of the derivative and the terms it is a difference of:
    # alpha ((y - mu)^2 + y) up to a dispersion of 0.1, below which the terms of order
    # y and mu are cancelled analytically, and 1 above.
    terms = np.where(alpha <= 0.1, alpha * ((y - mu) ** 2 + y), 1.0)

    assert np.max(np.abs(ours - exact) / np.maximum(np.abs(exact), terms)) < 1e-13


# Properties.


def test_likelihood_ratio_test_of_one_coefficient_approaches_its_wald_test() -> None:
    # Under the null hypothesis both statistics tend to the same chi-squared variable,
    # their difference falling as n^(-1/2); at the true dispersions.
    differences = []
    for n_per_group in (4, 100):
        y, x, alpha = simulated(n_per_group, 40, seed=n_per_group)
        fit = fit_deseq2(y, x, size_factors=np.ones(2 * n_per_group), dispersion=alpha)
        wald = deseq2_wald_test(fit, "groupB", cooks_cutoff=False)
        reduced = Design({"group": np.repeat(["A", "B"], n_per_group)}, "~ 1")
        lrt = deseq2_likelihood_ratio_test(fit, reduced, cooks_cutoff=False)
        differences.append(np.abs(lrt.statistic - wald.statistic**2))

    small, large = (float(np.median(d)) for d in differences)
    assert large < 0.2 * small
    assert np.max(differences[1]) < 0.05


@pytest.mark.parametrize(
    "ridge", [0.0, deseq2._NATURAL_RIDGE], ids=["without the ridge", "with the ridge"]
)
def test_results_do_not_depend_on_the_parametrization(
    ridge: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(deseq2, "_NATURAL_RIDGE", ridge)
    y = counts("two_groups")
    reference_level = fit_deseq2(y, design("two_groups"))
    means = fit_deseq2(y, design("two_groups", "~ 0 + group"))
    first = deseq2_wald_test(reference_level, "groupB")
    second = deseq2_wald_test(means, "groupB - groupA")
    # Without the ridge penalty the fits depend on the column space alone, to the
    # steps of 1e-10 at which they stop. The penalty, 1e-6 per squared log2
    # coefficient, depends on the parametrization: it moves a coefficient by about 2e-6
    # times the coefficient over its information, up to 2e-5 for a group whose counts
    # are all zero, with a coefficient near -8 and an information near 1.
    tolerance = 1e-9 if ridge == 0.0 else 5e-5

    # The dispersions are invariant: the means, and the Cox-Reid term up to a constant.
    assert_allclose(means.dispersion, reference_level.dispersion, rtol=1e-12)
    assert relative(means.fitted_values, reference_level.fitted_values) < tolerance
    assert relative(second.log2_fold_change, first.log2_fold_change, 1.0) < tolerance
    assert relative(second.standard_error, first.standard_error) < tolerance
    assert relative(second.statistic, first.statistic, 1.0) < tolerance


def test_results_do_not_depend_on_the_order_of_the_samples() -> None:
    y, x = counts("covariate"), design("covariate")
    order = np.random.default_rng(3).permutation(y.shape[0])
    shuffled_samples = {k: np.asarray(v)[order] for k, v in x.samples.to_dict().items()}
    shuffled = Design(shuffled_samples, x.formula, levels=x.levels)
    first = fit_deseq2(y, x)
    second = fit_deseq2(y[order], shuffled)
    a, b = first.dispersion_estimates, second.dispersion_estimates
    assert a is not None and b is not None

    for name in ("genewise", "trended", "map_estimate", "dispersion"):
        assert relative(getattr(b, name), getattr(a, name)) < 1e-9
    assert relative(second.coefficients, first.coefficients, 1.0) < 1e-9
    assert relative(second.cooks_distances, first.cooks_distances[order], 1.0) < 1e-9
    for test in (deseq2_wald_test, lambda fit: deseq2_wald_test(fit, "age")):
        one, other = test(first), test(second)
        assert relative(other.p_value, one.p_value) < 1e-8
        assert relative(other.adjusted_p_value, one.adjusted_p_value) < 1e-8


def test_genewise_estimates_do_not_depend_on_other_genes() -> None:
    y, x = counts("covariate"), design("covariate")
    sizes = reference("covariate", "size_factors")
    every = deseq2_dispersions(y, x, size_factors=sizes)
    subset = np.arange(0, y.shape[1], 3)
    some = deseq2_dispersions(y[:, subset], x, size_factors=sizes)

    assert relative(some.genewise, every.genewise[subset]) < 1e-12
    assert relative(some.means, every.means[:, subset]) < 1e-12


@pytest.mark.skipif(
    numba.config.NUMBA_NUM_THREADS < 2, reason="needs two threads to run in parallel"
)
def test_threads_do_not_change_the_results() -> None:
    y, x = counts("covariate"), design("covariate")
    one = fit_deseq2(y, x, n_threads=1)
    two = fit_deseq2(y, x, n_threads=2)

    assert_array_equal(one.dispersion, two.dispersion)
    assert_array_equal(one.coefficients, two.coefficients)
    assert_array_equal(one.cooks_distances, two.cooks_distances)


def test_size_factors_default_to_deseq2s_and_travel_with_the_estimates() -> None:
    y, x = counts("paired"), design("paired")
    default = fit_deseq2(y, x)
    assert_allclose(
        default.size_factors, reference("paired", "size_factors"), rtol=1e-13
    )
    # Estimates made with other size factors bring them to the fit.
    sizes = np.linspace(0.5, 2.0, y.shape[0])
    d = deseq2_dispersions(y, x, size_factors=sizes)
    again = fit_deseq2(y, x, dispersion=d)
    assert_array_equal(again.size_factors, sizes)
    assert again.dispersion_estimates is d
    given = fit_deseq2(y, x, size_factors=sizes, dispersion=d.dispersion)
    assert_array_equal(again.coefficients, given.coefficients)
    assert relative(again.coefficients, default.coefficients, 1.0) > 1e-3


# Edge cases and rules found by probing.


def test_genes_without_counts_get_nan() -> None:
    y = counts("two_groups").copy()
    y[:, 5] = 0.0
    fit = fit_deseq2(y, design("two_groups"), dispersion=np.full(y.shape[1], 0.1))
    result = deseq2_wald_test(fit)

    assert fit.base_mean[5] == 0.0
    assert np.isnan(fit.coefficients[5]).all() and np.isnan(fit.dispersion[5])
    for values in (result.log2_fold_change, result.standard_error, result.p_value):
        assert np.isnan(values[5])
    assert np.isnan(result.adjusted_p_value[5]) and result.converged[5]
    d = deseq2_dispersions(y, design("two_groups"))
    assert d.all_zero[5] and np.isnan(d.genewise[5]) and not d.outlier[5]


def test_a_given_prior_variance_needs_no_gene_to_estimate_it_from() -> None:
    # With every gene-wise estimate at the lower bound the spread of the log residuals
    # is undefined: no gene is an outlier, and each takes its maximum a posteriori
    # estimate under the given prior.
    y, x = counts("two_groups"), design("two_groups")
    genewise = np.full(y.shape[1], 1e-8)
    trended = np.full(y.shape[1], 0.05)
    options = dict(genewise_dispersions=genewise, trended_dispersions=trended)
    with pytest.raises(ValueError, match="pass prior_variance"):
        deseq2_dispersions(y, x, **options)
    d = deseq2_dispersions(y, x, **options, prior_variance=0.5)

    assert np.isnan(d.log_residual_variance) and not d.outlier.any()
    counted = ~d.all_zero
    assert_array_equal(d.dispersion[counted], d.map_estimate[counted])
    assert np.all(d.map_estimate[counted] > 1e-8)


def test_a_group_without_counts_gets_the_log_mean_where_the_continuation_is_flat() -> (
    None
):
    # Below a fitted value of 0.5 DESeq2's fits use the quadratic expansion of the
    # log-likelihood there; for counts of zero its slope is -c and its curvature c, so
    # the group's log mean m solves sum_j (1 + m + log s_j - log 0.5) = 0.
    y = np.array(
        [
            [30.0, 40.0],
            [25.0, 60.0],
            [33.0, 50.0],
            [0.0, 45.0],
            [0.0, 52.0],
            [0.0, 38.0],
        ]
    )
    s = np.array([0.8, 1.1, 1.0, 0.9, 1.2, 1.05])
    x = Design({"group": np.repeat(["A", "B"], 3)}, "~ 0 + group")
    fit = fit_deseq2(y, x, size_factors=s, dispersion=np.array([0.1, 0.1]))
    expected = (np.log(0.5) - 1.0 - np.mean(np.log(s[3:]))) / np.log(2.0)

    # The ridge penalty moves it by about 2e-6 times the coefficient over 1.4.
    assert_allclose(fit.coefficients[0, 1], expected, rtol=1e-5)
    assert fit.fitted_values[3:, 0].max() < 0.5


def test_the_contrast_of_two_groups_without_counts_is_zero() -> None:
    # The genes with no counts in groups C and D: DESeq2 reports 0, its p-value 1.
    fit = fitted("several_groups")
    result = deseq2_wald_test(fit, "groupD - groupC")
    zero = ~(fit.counts[6:] > 0).any(axis=0) & (fit.base_mean > 0)

    assert zero.sum() >= 3
    assert np.all(result.log2_fold_change[zero] == 0.0)
    assert np.all(result.statistic[zero] == 0.0)
    assert np.all(result.p_value[zero] == 1.0)
    assert np.all(result.standard_error[zero] > 0.0)
    # One-sided contrasts and coefficients do not get the rule.
    single = deseq2_wald_test(fit, "groupD")
    assert np.all(single.log2_fold_change[zero] != 0.0)
    assert_array_equal(
        result.log2_fold_change[~zero],
        deseq2_wald_test(fit, np.array([0.0, 0.0, -1.0, 1.0])).log2_fold_change[~zero],
    )


def test_contrasts_can_be_given_in_every_form() -> None:
    fit = fitted("several_groups")
    x = design("several_groups")
    expected = deseq2_wald_test(fit, np.array([0.0, -1.0, 0.0, 1.0]))
    named = x.with_contrasts({"d_vs_b": "groupD - groupB"}).contrasts
    for contrast in ("groupD - groupB", {"d_vs_b": "groupD - groupB"}, named):
        result = deseq2_wald_test(fit, contrast)
        assert_array_equal(result.statistic, expected.statistic)
        assert_array_equal(result.contrast, expected.contrast)
    assert deseq2_wald_test(fit, named).name == "d_vs_b"
    assert deseq2_wald_test(fit).name == "groupD"


def test_an_intercept_alone_is_fitted_by_the_log_mean_count() -> None:
    y = counts("two_groups")
    s = reference("two_groups", "size_factors")
    x = Design({"group": np.repeat(["A", "B"], 3)}, "~ 1")
    fit = fit_deseq2(y, x, size_factors=s, dispersion=np.full(y.shape[1], 0.2))
    counted = fit.base_mean > 0
    means = np.mean(y[:, counted] / s[:, None], axis=0)

    assert_allclose(fit.coefficients[counted, 0], np.log2(means), rtol=1e-14)


def test_few_residual_degrees_of_freedom_floor_the_prior_variance() -> None:
    # Gene-wise estimates that scatter less than the chi-squared sampling variance.
    y, x = counts("two_df"), design("two_df")
    trended = np.full(y.shape[1], 0.1)
    genewise = trended * np.exp(np.random.default_rng(2).normal(0.0, 0.1, y.shape[1]))
    d = deseq2_dispersions(
        y, x, genewise_dispersions=genewise, trended_dispersions=trended
    )

    assert d.df_residual == 2
    assert d.prior_variance == 0.25


def test_reprs_name_what_was_computed() -> None:
    fit = fitted("two_groups")
    assert "converged=" in repr(fit)
    assert "prior_variance=" in repr(estimates("two_groups"))
    assert repr(deseq2_wald_test(fit)).startswith("DESeq2Results(test='Wald'")
    assert isinstance(deseq2_wald_test(fit), DESeq2Results)


# Errors.

Y = np.array([[10.0, 0.0, 4.0], [12.0, 1.0, 5.0], [8.0, 0.0, 9.0], [20.0, 3.0, 7.0]])
X = np.column_stack([np.ones(4), [0.0, 0.0, 1.0, 1.0]])


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"counts": Y + 0.5}, ValueError, "whole numbers"),
        ({"counts": -Y}, ValueError, "non-negative"),
        ({"design": np.column_stack([X, X[:, 1]])}, ValueError, "full column rank"),
        ({"design": np.eye(4)}, ValueError, "no residual degrees of freedom"),
        ({"design": X[:3]}, ValueError, "one row per sample"),
        ({"size_factors": [1.0, 1.0]}, ValueError, "one value per sample"),
        ({"size_factors": [1.0, 0.0, 1.0, 1.0]}, ValueError, "positive and finite"),
        ({"trend": "local"}, ValueError, 'trend must be "parametric" or "mean"'),
        ({"min_mean": 0.0}, ValueError, "min_mean must be positive"),
        ({"prior_variance": -1.0}, ValueError, "prior_variance must be positive"),
        ({"genewise_dispersions": [0.1, 0.1]}, ValueError, "one value per gene"),
        ({"trended_dispersions": [0.1, -1.0, 0.1]}, ValueError, "positive and finite"),
        ({"trend": 3}, TypeError, "trend must be a string"),
    ],
)
def test_deseq2_dispersions_refuses_invalid_input(
    arguments: dict[str, Any], error: type[Exception], match: str
) -> None:
    call = {"counts": Y, "design": X} | arguments
    with pytest.raises(error, match=match):
        deseq2_dispersions(**call)


def test_size_factors_cannot_be_estimated_when_every_gene_has_a_zero() -> None:
    y = Y.copy()
    y[0, [0, 2]] = 0.0
    with pytest.raises(ValueError, match="poscounts"):
        fit_deseq2(y, X, dispersion=[0.1, 0.1, 0.1])


@pytest.mark.parametrize(
    ("arguments", "error", "match"),
    [
        ({"contrast": "nothing"}, ValueError, "nothing"),
        (
            {"contrast": np.array([1.0, 0.0, 0.0])},
            ValueError,
            "one row per coefficient",
        ),
        ({"contrast": np.eye(2)}, ValueError, "one contrast"),
        ({"contrast": np.zeros(2)}, ValueError, "so it compares nothing"),
        ({"contrast": "x2 - x2"}, ValueError, "so it compares nothing"),
        ({"alpha": 1.5}, ValueError, r"alpha must be in \(0, 1\)"),
        ({"independent_filtering": 1}, TypeError, "True or False"),
        ({"cooks_cutoff": True}, TypeError, "cooks_cutoff"),
        ({"cooks_cutoff": -1.0}, ValueError, "cooks_cutoff must be positive"),
    ],
)
def test_deseq2_wald_test_refuses_invalid_input(
    arguments: dict[str, Any], error: type[Exception], match: str
) -> None:
    fit = fit_deseq2(Y, X, dispersion=[0.1, 0.1, 0.1], size_factors=np.ones(4))
    with pytest.raises(error, match=match):
        deseq2_wald_test(fit, **arguments)


@pytest.mark.parametrize(
    ("reduced", "error", "match"),
    [
        (X[:3, :1], ValueError, "one row per sample"),
        (np.column_stack([np.ones(4), [1.0, 2.0, 3.0, 5.0]]), ValueError, "nested"),
        (X, ValueError, "fewer columns"),
        (np.column_stack([np.ones(4), np.ones(4)]), ValueError, "full column rank"),
        ("~ 1", TypeError, r"Design\(samples, '~ 1'\)"),
    ],
)
def test_deseq2_likelihood_ratio_test_refuses_invalid_reduced_designs(
    reduced: Any, error: type[Exception], match: str
) -> None:
    fit = fit_deseq2(Y, X, dispersion=[0.1, 0.1, 0.1], size_factors=np.ones(4))
    with pytest.raises(error, match=match):
        deseq2_likelihood_ratio_test(fit, reduced)
    with pytest.raises(ValueError, match="coefficient must be one of"):
        deseq2_likelihood_ratio_test(fit, X[:, :1], coefficient="x9")


def test_tests_need_a_fit() -> None:
    with pytest.raises(TypeError, match="fit must be a DESeq2Fit"):
        deseq2_wald_test(np.ones((2, 2)))  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="fit must be a DESeq2Fit"):
        deseq2_likelihood_ratio_test("fit", X)  # type: ignore[arg-type]


def test_given_dispersions_must_be_positive_for_genes_with_counts() -> None:
    with pytest.raises(ValueError, match="positive and finite for genes with counts"):
        fit_deseq2(Y, X, dispersion=[0.1, 0.0, 0.1], size_factors=np.ones(4))
    y = Y.copy()
    y[:, 1] = 0.0
    fit = fit_deseq2(y, X, dispersion=[0.1, np.nan, 0.1], size_factors=np.ones(4))
    assert np.isnan(fit.dispersion[1])


def test_designs_with_random_effects_are_refused() -> None:
    x = Design(
        {"group": ["A", "A", "B", "B"], "donor": ["1", "2", "1", "2"]},
        "~ group",
        random="(1 | donor)",
    )
    with pytest.raises(ValueError, match="random effect"):
        fit_deseq2(Y, x, dispersion=[0.1, 0.1, 0.1])
