"""Tests for dream: voom weights from mixed models, moderated tests, dream_voom.

Reference values in ``tests/data/dream`` come from variancePartition 1.36.2 (limma
3.62.1, lme4 1.1-38), as ``validation/dream.py`` ran it: ``voomWithDreamWeights``
(``span = 0.5``; ``rescaleWeightsAfter = TRUE`` with prior weights), ``dream(ddf =
"Satterthwaite")``, ``eBayes`` (also with ``robust`` and ``trend``) and ``topTable``,
on counts with a random donor intercept (``one_intercept``), crossed donor and batch
intercepts with sample prior weights (``crossed``), unbalanced donors with prior
weights for every observation (``unbalanced``), and three cell types of a cube
(``cube``); and lme4 and lmerTest evaluated at CytePool's maximum likelihood and REML
estimates of the first 20 genes of ``one_intercept`` and ``crossed``. The other tests check closed forms of the maximum likelihood fits, their
fitted values and approximate residual degrees of freedom; that a donor variance of
zero gives limma's moderated tests; and invariants: the order of samples, the labels
of donors, the scale of prior weights.
"""

import json
import math
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool import Cube, Design, dream_voom, from_bulk, limma_voom
from cytepool.stats.dream import (
    ModeratedFixedEffectTests,
    _signed_z,
    moderated_fixed_effect_tests,
    voom_with_dream_weights,
)
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.mixed_models import fit_mixed_model, fixed_effect_tests
from cytepool.stats.multiple_testing import adjust_p_values

DATA = Path(__file__).parent / "data" / "dream"
# variancePartition's fits converge to about 1e-8 of the optimum (see the validation
# script); lmerTest differentiates the deviance numerically for Satterthwaite's df.
CLOSE = 1e-6
SATTERTHWAITE = 1e-5
# Closed forms at given estimates.
EXACT = 1e-10
N_GIVEN = 20
VARIANTS = {"default": {}, "robust": {"robust": True}, "trend": {"trend": True}}
CONTRASTS = {"stim_minus_age": "conditionstim - age"}


@cache
def fixture(name: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    return arrays, json.loads((DATA / f"{name}.json").read_text())


def case_design(name: str) -> Design:
    samples = fixture(name)[1]
    random = "(1 | donor) + (1 | batch)" if "batch" in samples else "(1 | donor)"
    return Design(samples, "~ condition + age", random=random)


def contrast_weights(design: Design, names: np.ndarray) -> np.ndarray:
    """R's coefficient names (and the contrast) as weights on the design's columns."""
    matrix = np.zeros((len(design.columns), len(names)))
    for j, name in enumerate(names.tolist()):
        for sign, term in zip((1.0, -1.0), CONTRASTS.get(name, name).split(" - ")):
            matrix[design.columns.index(term), j] = sign
    return matrix


@cache
def fixture_tests(name: str) -> Any:
    data, _ = fixture(name)
    design = case_design(name)
    voom = voom_with_dream_weights(
        data["counts"], design, span=0.5, weights=data.get("prior_weights")
    )
    fit = fit_mixed_model(voom.log2_cpm, design, weights=voom.weights)
    names = data["names"]
    return voom, fixed_effect_tests(fit, contrast_weights(design, names), names=names)


def scaled(ours: Any, reference: Any, floor: float = 1.0) -> float:
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    return float(np.max(np.abs(ours - reference) / np.fmax(np.abs(reference), floor)))


def p_value_difference(ours: Any, reference: Any) -> float:
    """|ours - p| over max(p, 1e-300) max(1, |log p|): the error relative to log p."""
    p = np.fmax(np.asarray(reference, float), 1e-300)
    return float(
        np.max(np.abs(np.asarray(ours) - p) / (p * np.fmax(1.0, np.abs(np.log(p)))))
    )


# Against variancePartition.


@pytest.mark.parametrize("name", ["one_intercept", "crossed", "unbalanced"])
def test_the_weights_and_tests_match_variance_partition(name: str) -> None:
    data, _ = fixture(name)
    voom, tests = fixture_tests(name)
    np.testing.assert_allclose(voom.weights, data["weights"], rtol=CLOSE, atol=0.0)
    np.testing.assert_allclose(voom.sqrt_sd, data["sqrt_sd"], rtol=CLOSE, atol=0.0)
    se, coefficients = data["se"], data["coefficients"]
    difference = np.abs(tests.estimate - coefficients) / np.fmax(
        np.abs(coefficients), se
    )
    assert np.max(difference) <= CLOSE
    np.testing.assert_allclose(tests.standard_error, se, rtol=CLOSE, atol=0.0)
    np.testing.assert_allclose(tests.df, data["df"], rtol=SATTERTHWAITE, atol=0.0)


@pytest.mark.parametrize(
    "name, variant",
    [
        ("one_intercept", "default"),
        ("one_intercept", "robust"),
        ("one_intercept", "trend"),
        ("crossed", "default"),
        ("unbalanced", "default"),
    ],
)
def test_the_moderation_matches_variance_partitions_ebayes(
    name: str, variant: str
) -> None:
    data, _ = fixture(name)
    _, tests = fixture_tests(name)
    moderated = moderated_fixed_effect_tests(tests, **VARIANTS[variant])
    reference = {
        key.split(" ", 1)[1]: value
        for key, value in data.items()
        if key.startswith(f"{variant} ")
    }
    n = tests.df.shape[0]
    np.testing.assert_allclose(
        np.sqrt(moderated.residual_variance_at_mean_weight), data["sigma"], rtol=CLOSE
    )
    np.testing.assert_allclose(moderated.df_residual, data["rdf"], rtol=CLOSE)
    for ours, key in (
        (moderated.df_prior, "df_prior"),
        (moderated.prior_variance, "s2_prior"),
    ):
        np.testing.assert_allclose(
            np.broadcast_to(ours, n), np.broadcast_to(reference[key], n), rtol=CLOSE
        )
    np.testing.assert_allclose(
        moderated.posterior_variance, reference["s2_post"], rtol=CLOSE
    )
    assert scaled(moderated.t, reference["t"]) <= CLOSE
    assert scaled(moderated.log_odds, reference["lods"]) <= CLOSE
    np.testing.assert_allclose(
        moderated.df_total, reference["df_total"], rtol=SATTERTHWAITE
    )
    assert p_value_difference(moderated.p_value, reference["p"]) <= SATTERTHWAITE
    assert scaled(moderated.z, reference["z"]) <= SATTERTHWAITE
    if variant == "robust":  # the fixture's outliers get smaller prior df
        assert np.ptp(moderated.df_prior) > 1.0


@cache
def fixture_cube() -> tuple[Cube, Design]:
    data, samples = fixture("cube")
    cube = Cube(
        {"counts": data["counts"]},
        samples=samples,
        groups={"group_id": ["B", "NK", "rare"]},
        features={"feature_id": [f"gene{i}" for i in range(data["counts"].shape[2])]},
        n_cells=data["n_cells"],
    )
    design = Design(
        cube.samples,
        "~ condition",
        random="(1 | donor)",
        levels={"condition": ["ctrl", "stim"]},
        contrasts={"stim_vs_ctrl": "conditionstim"},
    )
    return cube, design


@pytest.mark.parametrize("variant", ["default", "trend"])
def test_dream_voom_matches_variance_partition_on_every_cell_type(variant: str) -> None:
    data, _ = fixture("cube")
    cube, design = fixture_cube()
    # The fixture is dream(ddf = "Satterthwaite"); the default, "adaptive", would
    # use Kenward-Roger's method in these groups of fewer than 20 samples.
    result = dream_voom(
        cube,
        design,
        prior_weights="n_cells",
        span=0.5,
        trend=variant == "trend",
        df_method="satterthwaite",
    )
    assert result.method == "dream_voom"
    assert result.summary["df_method"].tolist() == ["satterthwaite"] * 2 + [""]
    for group in ("B", "NK"):
        table = result.table(group, "stim_vs_ctrl")
        reference = {
            key.split(" ")[-1]: value
            for key, value in data.items()
            if key.startswith(f"{group} {variant} ")
        }
        assert scaled(table["log2_fold_change"], data[f"{group} logFC"]) <= CLOSE
        assert scaled(table["average_log2_cpm"], reference["average"]) <= CLOSE
        assert scaled(table["t"], reference["t"]) <= CLOSE
        assert scaled(table["log_odds"], reference["lods"]) <= CLOSE
        np.testing.assert_allclose(
            table["df"], reference["df_total"], rtol=SATTERTHWAITE
        )
        assert p_value_difference(table["p_value"], reference["p"]) <= SATTERTHWAITE
        assert (
            p_value_difference(table["adjusted_p_value"], reference["adjusted"])
            <= SATTERTHWAITE
        )
        assert scaled(table["z_statistic"], reference["z"]) <= SATTERTHWAITE
    # NK: donor D1's two samples and one of D3's have fewer than 10 cells.
    nk = result.samples["group_id"] == "NK"
    left_out = result.samples["sample_id"][nk & ~result.samples["used"]].tolist()
    assert left_out == ["D1_ctrl", "D1_stim", "D3_ctrl"]
    # rare: one sample per donor is left, which cannot separate the donor variance.
    summary = {key: result.summary[key][2] for key in ("group_id", "status", "reason")}
    assert summary["status"] == "random_effects_not_estimable"
    assert (
        "every group of the random effect (1 | donor) has a single sample"
        in summary["reason"]
    )
    with pytest.raises(KeyError, match="random_effects_not_estimable"):
        result.table("rare", "stim_vs_ctrl")


# The mixed models behind the weights: against lme4, and closed forms.


@pytest.mark.parametrize("name", ["one_intercept", "crossed"])
@pytest.mark.parametrize("method", ["ml", "reml"])
def test_the_fits_match_lme4_and_lmertest_at_the_same_estimates(
    name: str, method: str
) -> None:
    """The fits of the trend (maximum likelihood, prior weights each divided by their
    mean) and of the tests (REML, the precision weights), against lme4's logLik,
    sigma and fitted, variancePartition's rdf.merMod and lmerTest's Satterthwaite df
    evaluated at CytePool's estimates of the first 20 genes."""
    data, _ = fixture(name)
    voom, _ = fixture_tests(name)
    n = voom.log2_cpm.shape[0]
    if method == "ml":
        prior = data.get("prior_weights", np.ones(n))
        weights = np.repeat((prior / prior.mean())[:, None], N_GIVEN, axis=1)
    else:
        weights = voom.weights[:, :N_GIVEN]
    fit = fit_mixed_model(
        voom.log2_cpm[:, :N_GIVEN],
        case_design(name),
        weights=weights,
        method=method,  # type: ignore[arg-type]
    )
    reference = {
        key.split(" ")[-1]: value
        for key, value in data.items()
        if key.startswith(f"given_{method} ")
    }
    np.testing.assert_allclose(fit.log_likelihood, reference["loglik"], rtol=EXACT)
    np.testing.assert_allclose(fit.residual_variance, reference["sigma2"], rtol=EXACT)
    np.testing.assert_allclose(fit.df_residual, reference["rdf"], rtol=EXACT)
    largest = np.abs(reference["fitted"]).max()
    np.testing.assert_allclose(
        fit.fitted_values, reference["fitted"].T, rtol=0.0, atol=EXACT * largest
    )
    np.testing.assert_allclose(
        fixed_effect_tests(fit).df, reference["df"], rtol=SATTERTHWAITE
    )


def balanced_one_way(seed: int, q: int = 6, m: int = 4) -> tuple[np.ndarray, Design]:
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"G{i}" for i in range(q)], m)
    y = rng.normal(size=(q * m, 3)) + np.repeat(
        rng.normal(size=(q, 3)) * [1.5, 3.0, 1.0], m, axis=0
    )
    return y, Design({"group": groups}, "~ 1", random="(1 | group)")


def test_maximum_likelihood_of_a_balanced_one_way_design_has_closed_forms() -> None:
    q, m = 6, 4
    y, design = balanced_one_way(1, q, m)
    fit = fit_mixed_model(y, design, method="ml")
    means = y.reshape(q, m, -1).mean(axis=1)
    within = ((y.reshape(q, m, -1) - means[:, None]) ** 2).sum(axis=(0, 1)) / (
        q * (m - 1)
    )
    between = m * ((means - y.mean(axis=0)) ** 2).sum(axis=0)
    # sigma^2 = MSW and sigma^2 + m sigma_a^2 = SSB / q (REML has q - 1 there).
    assert np.all(between / q > within)  # the fixture's estimates are inside
    np.testing.assert_allclose(fit.residual_variance, within, rtol=1e-9)
    np.testing.assert_allclose(
        fit.variance_components[:, 0], (between / q - within) / m, rtol=1e-8
    )
    n = q * m
    log_likelihood = -0.5 * (
        n * np.log(2 * np.pi)
        + q * (m - 1) * np.log(within)
        + q * np.log(between / q)
        + n
    )
    np.testing.assert_allclose(fit.log_likelihood, log_likelihood, rtol=1e-12)
    assert fit.method == "ml" and np.isnan(fit.reml_criterion).all()
    reml = fit_mixed_model(y, design)
    np.testing.assert_allclose(
        reml.log_likelihood, -0.5 * reml.reml_criterion, rtol=1e-14
    )


@pytest.mark.parametrize("method", ["reml", "ml"])
def test_fitted_values_and_residual_df_of_a_balanced_design_have_closed_forms(
    method: str,
) -> None:
    q, m = 6, 4
    y, design = balanced_one_way(2, q, m)
    fit = fit_mixed_model(y, design, method=method)  # type: ignore[arg-type]
    gamma = fit.variance_components[:, 0] / fit.residual_variance
    means = y.reshape(q, m, -1).mean(axis=1)
    # The predictions shrink each group's mean towards the grand mean by
    # m gamma / (1 + m gamma).
    shrinkage = m * gamma / (1.0 + m * gamma)
    fitted = y.mean(axis=0) + shrinkage * (means - y.mean(axis=0))
    np.testing.assert_allclose(
        fit.fitted_values, np.repeat(fitted, m, axis=0), rtol=1e-10
    )
    # I - H is the identity on the q (m - 1) contrasts within groups, 1 / (1 + m
    # gamma) on the q - 1 between them and zero on the grand mean.
    rdf = q * (m - 1) + (q - 1) / (1.0 + m * gamma) ** 2
    np.testing.assert_allclose(fit.df_residual, rdf, rtol=1e-10)
    np.testing.assert_allclose(fit.mean_expression, y.mean(axis=0), rtol=1e-15)
    np.testing.assert_array_equal(fit.mean_weights, np.ones(3))


@pytest.mark.parametrize("shape", ["observations", "samples"])
def test_the_variance_at_the_mean_weight_is_that_of_weights_of_mean_one(
    shape: str,
) -> None:
    """Dream's variance, residual_variance / mean_weights, does not depend on the
    scale of the weights, and is the residual variance of a fit to the weights divided
    by their mean, whose mean weight is one."""
    y, design = balanced_one_way(3)
    weights = np.random.default_rng(3).uniform(0.5, 4.0, size=y.shape)
    if shape == "samples":
        weights = weights[:, 0]
    fit = fit_mixed_model(y, design, weights=weights)
    larger = fit_mixed_model(y, design, weights=7.0 * weights)
    unit = fit_mixed_model(y, design, weights=weights / weights.mean(axis=0))
    np.testing.assert_allclose(unit.mean_weights, 1.0, rtol=1e-14)
    variance = fit.residual_variance / fit.mean_weights
    np.testing.assert_allclose(variance, unit.residual_variance, rtol=1e-9)
    np.testing.assert_allclose(
        larger.residual_variance / larger.mean_weights, variance, rtol=1e-9
    )
    np.testing.assert_allclose(
        larger.residual_variance, 7.0 * fit.residual_variance, rtol=1e-9
    )


# Invariants.


def test_sample_order_donor_labels_and_the_scale_of_prior_weights_change_nothing() -> (
    None
):
    data, samples = fixture("crossed")
    design = case_design("crossed")
    reference = voom_with_dream_weights(
        data["counts"], design, span=0.5, weights=data["prior_weights"]
    )
    order = np.random.default_rng(4).permutation(data["counts"].shape[0])
    relabelled = {
        key: np.asarray(value)[order].tolist() for key, value in samples.items()
    }
    relabelled["donor"] = [f"donor_{label[::-1]}" for label in relabelled["donor"]]
    other = Design(relabelled, "~ condition + age", random="(1 | donor) + (1 | batch)")
    permuted = voom_with_dream_weights(
        data["counts"][order],
        other,
        span=0.5,
        weights=7.0 * data["prior_weights"][order],
    )
    np.testing.assert_allclose(permuted.weights, reference.weights[order], rtol=1e-8)
    first = fixed_effect_tests(
        fit_mixed_model(reference.log2_cpm, design, weights=reference.weights)
    )
    second = fixed_effect_tests(
        fit_mixed_model(permuted.log2_cpm, other, weights=permuted.weights)
    )
    a, b = moderated_fixed_effect_tests(first), moderated_fixed_effect_tests(second)
    np.testing.assert_allclose(b.t, a.t, rtol=1e-7)
    np.testing.assert_allclose(b.p_value, a.p_value, rtol=1e-6)


def test_a_donor_variance_of_zero_gives_limmas_moderated_tests() -> None:
    """At zero donor variance dream's fit is limma's weighted fit, its df are n - p,
    and its variances are those of each feature's weights divided by their mean."""
    rng = np.random.default_rng(5)
    n_donors, per_donor, n_features = 8, 3, 40
    donor = np.repeat([f"D{i}" for i in range(n_donors)], per_donor)
    samples = {"donor": donor, "condition": np.tile(["ctrl", "stim", "stim"], n_donors)}
    design = Design(samples, "~ condition", random="(1 | donor)")
    x = design.matrix
    z = (donor[:, None] == np.unique(donor)[None, :]).astype(float)
    weights = rng.uniform(0.5, 5.0, size=(donor.size, n_features))
    y = np.empty_like(weights)
    for g in range(n_features):
        # Residuals orthogonal, in the weighted inner product, to the fixed effects
        # and to every donor, so that the REML gradient points out of the boundary.
        span = np.column_stack([x, z])
        w = weights[:, g]
        noise = rng.normal(size=donor.size) * rng.uniform(0.3, 2.0)
        coefficients = np.linalg.lstsq(
            span * np.sqrt(w)[:, None], noise * np.sqrt(w), rcond=None
        )[0]
        y[:, g] = x @ [5.0, rng.normal()] + noise - span @ coefficients
    fit = fit_mixed_model(y, design, weights=weights)
    assert np.all(fit.boundary)
    moderated = moderated_fixed_effect_tests(fixed_effect_tests(fit))
    np.testing.assert_allclose(moderated.df_residual, donor.size - 2, rtol=1e-12)
    limma = fit_linear_model(
        y, x, weights=weights / weights.mean(axis=0), coefficient_names=design.columns
    )
    reference = moderated_statistics(limma)
    np.testing.assert_allclose(
        moderated.posterior_variance, reference.posterior_variance, rtol=1e-9
    )
    np.testing.assert_allclose(moderated.t, reference.t, rtol=1e-9)
    np.testing.assert_allclose(moderated.log_odds, reference.log_odds, rtol=1e-9)
    # dream adds the prior df to n - p without limma's cap, which these data leave
    # unreached.
    total = np.broadcast_to(reference.df_total[:, None], moderated.df_total.shape)
    np.testing.assert_allclose(moderated.df_total, total, rtol=1e-9)
    np.testing.assert_allclose(moderated.p_value, reference.p_value, rtol=1e-8)


def test_z_statistics_on_infinite_df_are_the_t_statistics_even_beyond_underflow() -> (
    None
):
    t = np.array([[0.0, -1.5, 8.0], [-30.0, 38.0, 41.0], [-60.0, 200.0, 1e4]])
    df = np.full(t.shape, np.inf)
    # Two-sided normal p-values from the complementary error function, which
    # underflows to zero beyond |t| of about 38.5.
    p_value = np.vectorize(lambda value: math.erfc(abs(value) / math.sqrt(2.0)))(t)
    assert p_value[1, 2] == 0.0 and p_value[1, 1] < 1e-300
    z = _signed_z(t, df, p_value)
    np.testing.assert_allclose(z, t, rtol=1e-13, atol=1e-15)


# Edges and errors.


def reproduced_counts() -> tuple[np.ndarray, Design]:
    """Counts where gene 0 is constant within donors, with equal library sizes there."""
    rng = np.random.default_rng(6)
    counts = rng.negative_binomial(10, 0.1, size=(12, 20))
    counts[1::2, 1:] = np.roll(counts[0::2, 1:], 1, axis=1)
    counts[:, 0] = np.repeat(np.arange(6) * 7 + 20, 2)
    samples = {
        "donor": np.repeat([f"D{i}" for i in range(6)], 2),
        "condition": ["a", "b"] * 6,
    }
    return counts, Design(samples, "~ condition", random="(1 | donor)")


def test_a_feature_the_effects_reproduce_is_left_out_of_the_trend_and_untested() -> (
    None
):
    counts, design = reproduced_counts()
    voom = voom_with_dream_weights(counts, design)
    assert np.isnan(voom.sqrt_sd[0]) and np.isfinite(voom.sqrt_sd[1:]).all()
    assert np.isfinite(voom.weights).all() and (voom.weights > 0.0).all()
    moderated = moderated_fixed_effect_tests(
        fixed_effect_tests(fit_mixed_model(voom.log2_cpm, design, weights=voom.weights))
    )
    assert np.isnan(moderated.t[0]).all() and np.isfinite(moderated.t[1:]).all()
    assert np.isnan(moderated.p_value[0]).all() and np.isnan(moderated.z[0]).all()


def test_dream_voom_records_its_settings_and_the_features_not_fitted() -> None:
    counts, design = reproduced_counts()
    samples = design.samples.to_dict()
    samples["sample_id"] = [f"S{i}" for i in range(12)]
    cube = from_bulk(counts, samples=samples)
    named = Design(
        samples, "~ condition", random="(1 | donor)", contrasts={"b_vs_a": "conditionb"}
    )
    # Library sizes alone: equal within each donor, so gene 0 stays reproduced.
    result = dream_voom(
        cube,
        named,
        expression_filter=False,
        normalization=None,
        adjust_across_groups=True,
    )
    table = result.table("bulk", "b_vs_a")
    assert table.columns == (
        "feature_id",
        "log2_fold_change",
        "average_log2_cpm",
        "t",
        "df",
        "p_value",
        "adjusted_p_value",
        "adjusted_p_value_across_groups",
        "log_odds",
        "z_statistic",
    )
    # The feature without a fit is untested, and left out of both adjustments.
    assert np.isnan(table["t"][0]) and np.isnan(table["adjusted_p_value"][0])
    for column in ("adjusted_p_value", "adjusted_p_value_across_groups"):
        np.testing.assert_allclose(
            table[column][1:], adjust_p_values(table["p_value"][1:], "bh"), rtol=1e-15
        )
    record = result.provenance
    assert record["method"] == "dream_voom"
    assert {
        "weights_fit": "ml",
        "fit": "reml",
        "df_method": "adaptive",
        "kenward_roger_weights": "average",
        "f_tests": None,
        "trend": False,
    }.items() <= record["settings"].items()
    assert "quality_weights" not in record["settings"]
    group = record["groups"][0]
    assert group["features_not_fitted"] == 1 and group["features_not_converged"] == 0
    assert group["voom_span"] == 1.0
    # 12 samples: dream's adaptive choice is Kenward-Roger's method.
    assert group["df_method"] == "kenward_roger"
    assert group["features_satterthwaite_fallback"] == 0 and group["f_test_df1"] == {}
    assert pickle.loads(pickle.dumps(result)).provenance == record
    # The design's rows in another order give the same tables.
    reversed_samples = {key: value[::-1] for key, value in samples.items()}
    reversed_design = Design(
        reversed_samples,
        "~ condition",
        random="(1 | donor)",
        contrasts={"b_vs_a": "conditionb"},
    )
    again = dream_voom(
        cube, reversed_design, expression_filter=False, normalization=None
    )
    np.testing.assert_array_equal(again.table("bulk", "b_vs_a")["t"], table["t"])


def test_invalid_inputs_are_refused_with_what_to_do() -> None:
    counts, design = reproduced_counts()
    fixed = Design(design.samples, "~ condition")
    with pytest.raises(ValueError, match="no random effects.*use voom"):
        voom_with_dream_weights(counts, fixed)
    slopes = Design(
        {**design.samples.to_dict(), "age": np.arange(12.0)},
        "~ condition",
        random="(age | donor)",
    )
    with pytest.raises(ValueError, match="random slopes"):
        voom_with_dream_weights(counts, slopes)
    with pytest.raises(ValueError, match="finite and positive; 1 are zero"):
        voom_with_dream_weights(counts, design, weights=np.r_[0.0, np.ones(11)])
    with pytest.raises(ValueError, match=r"shape \(12,\) for sample weights"):
        voom_with_dream_weights(counts, design, weights=np.ones(5))
    with pytest.raises(ValueError, match="method must be 'reml'"):
        voom_with_dream_weights(counts, design, method="lsq")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="method must be 'reml'"):
        fit_mixed_model(np.ones((12, 2)), design, method=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least two features"):
        voom_with_dream_weights(counts[:, :1], design)
    with pytest.raises(ValueError, match="counts have 6 rows"):
        voom_with_dream_weights(counts[:6], design)
    with pytest.raises(TypeError, match="must be a FixedEffectTests"):
        moderated_fixed_effect_tests(fit_mixed_model(np.log1p(counts), design))  # type: ignore[arg-type]
    no_fixed = Design(design.samples, "~ 0", random="(1 | donor)")
    with pytest.raises(ValueError, match="no contrasts to moderate.*no fixed effects"):
        moderated_fixed_effect_tests(
            fixed_effect_tests(fit_mixed_model(np.log1p(counts), no_fixed))
        )
    constant = fit_mixed_model(
        np.repeat(np.arange(6.0), 2)[:, None] * np.ones((1, 3)), design
    )
    with pytest.raises(ValueError, match="no feature has a mixed model fit"):
        moderated_fixed_effect_tests(fixed_effect_tests(constant))
    cube = from_bulk(
        counts,
        samples={"sample_id": [f"S{i}" for i in range(12)], **design.samples.to_dict()},
    )
    plain = Design(cube.samples, "~ condition", contrasts={"b": "conditionb"})
    with pytest.raises(ValueError, match="no random effects; .*limma_voom"):
        dream_voom(cube, plain)
    mixed = Design(
        cube.samples, "~ condition", random="(1 | donor)", contrasts={"b": "conditionb"}
    )
    with pytest.raises(
        ValueError, match="drop the random effect.*dream_voom fits these"
    ):
        limma_voom(cube, mixed)
    with pytest.raises(ValueError, match="does not record"):
        dream_voom(
            cube,
            Design(
                cube.samples,
                "~ condition",
                random="(1 | donor)",
                contrasts={"b": "conditionb"},
            ),
            prior_weights="n_cells",
        )
    moderated = moderated_fixed_effect_tests(
        fixed_effect_tests(fit_mixed_model(np.log1p(counts), design))
    )
    assert isinstance(moderated, ModeratedFixedEffectTests)
    assert repr(moderated).startswith("ModeratedFixedEffectTests(n_features=20")
