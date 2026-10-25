"""Tests for the meta-analysis of many features across studies.

Reference values in ``tests/data/meta_analysis/metafor.npz`` come from metafor's
``rma.uni``, run feature by feature on the effects and standard errors stored with
them: the fixed-effect model and the DerSimonian-Laird, REML and Paule-Mandel
estimators, each with z-tests and the Knapp-Hartung adjustment. ``cohorts.npz`` holds
the pseudobulk counts of three simulated cohorts and metafor's meta-analyses of their
``limma_voom`` tables, merged by feature id in R. ``validation/meta_analysis.py``
writes both files. The other tests compare with closed forms derived by hand (equal
standard errors, two studies, one study) and check what a wrong implementation would
break: the order of the studies, the scale and location of the effects, a study that
did not measure a feature, the maximum of the restricted likelihood, the estimating
equation of Paule and Mandel, and the matching of features by id across studies.
"""

import dataclasses
import math
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import (
    DifferentialExpressionResult,
    limma_voom,
)
from cytepool.meta_analysis import (
    MetaAnalysisFit,
    MetaAnalysisResult,
    differential_expression_meta_analysis,
    meta_analysis,
)
from cytepool.stats.distributions import norm_isf, norm_sf, t_isf, t_sf
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "meta_analysis"
EXACT = 1e-10
# REML and Paule-Mandel are iterative; metafor converged to 1e-13 on tau^2.
CLOSE = 1e-6
TIERS = {"fixed": EXACT, "dl": EXACT, "reml": CLOSE, "pm": CLOSE}
METHODS = ("fixed", "dl", "reml", "pm")
TESTS = ("z", "knapp_hartung")
# metafor's outputs, in order, and the attributes they match.
FIELDS = (
    "estimate",
    "standard_error",
    "statistic",
    "p_value",
    "ci_lower",
    "ci_upper",
    "tau_squared",
    "q_statistic",
    "q_p_value",
    "i_squared",
    "h_squared",
    "n_studies",
)
Z_975 = float(norm_isf(0.025))


@cache
def fixture(name: str) -> dict[str, np.ndarray]:
    with np.load(DATA / f"{name}.npz", allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def scaled_difference(ours: Any, reference: Any, field: str) -> float:
    """|ours - reference| over max(|reference|, 1), or for p-values over
    max(p, 1e-300) max(1, |log p|); equal values, missing ones included, differ by 0."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    same = (ours == reference) | (np.isnan(ours) & np.isnan(reference))
    with np.errstate(invalid="ignore"):  # NaN where a feature has no study
        if field.endswith("p_value"):
            scale = np.maximum(reference, 1e-300)
            scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
        else:
            scale = np.maximum(np.abs(reference), 1.0)
        scaled = np.where(same, 0.0, np.abs(ours - reference) / scale)
    return float(np.max(np.where(np.isnan(scaled), np.inf, scaled), initial=0.0))


def restricted_log_likelihood(y: np.ndarray, v: np.ndarray, tau2: Any) -> np.ndarray:
    """The restricted log-likelihood of the random-effects model, without its
    constant, at each tau2: the function REML maximizes, by its definition."""
    t = np.asarray(tau2, float)[..., np.newaxis]
    w = 1.0 / (v + t)
    mu = np.sum(w * y, axis=-1, keepdims=True) / np.sum(w, axis=-1, keepdims=True)
    return -0.5 * (
        np.sum(np.log(v + t), axis=-1)
        + np.log(np.sum(w, axis=-1))
        + np.sum(w * (y - mu) ** 2, axis=-1)
    )


def fit_values(fit: MetaAnalysisFit) -> list[np.ndarray]:
    """A fit's values in metafor's order, with I^2 in percent as metafor gives it."""
    values = [np.asarray(getattr(fit, field), float) for field in FIELDS]
    values[FIELDS.index("i_squared")] = values[FIELDS.index("i_squared")] * 100.0
    return values


# Against metafor.


@pytest.mark.parametrize("case", range(9))
def test_meta_analysis_matches_metafor(case: int) -> None:
    data = fixture("metafor")
    y, se = data[f"case{case} y"], data[f"case{case} se"]
    higher = data[f"case{case} reml higher"]
    for method in METHODS:
        for test in TESTS:
            fit = meta_analysis(y, se, method=method, test=test)
            theirs = data[f"case{case} {method} {test}"]
            # Where the restricted likelihood has another maximum, REML's is the
            # higher one; metafor's is the other, compared below.
            compared = ~higher if method == "reml" else np.ones_like(higher)
            for field, ours, reference in zip(
                FIELDS, fit_values(fit), theirs, strict=True
            ):
                difference = scaled_difference(
                    ours[compared], reference[compared], field
                )
                assert difference <= TIERS[method], (method, test, field, difference)
            assert fit.converged.all()


def test_reml_takes_the_higher_maximum_where_metafor_takes_the_other() -> None:
    data = fixture("metafor")
    case = [str(data[f"case{c} name"]) for c in range(9)].index(
        "4 studies, one imprecise and outlying"
    )
    higher = data[f"case{case} reml higher"]
    y, se = data[f"case{case} y"][:, higher], data[f"case{case} se"][:, higher]
    theirs = data[f"case{case} reml z"][FIELDS.index("tau_squared"), higher]
    ours = meta_analysis(y, se).tau_squared

    assert higher.any()
    gain = restricted_log_likelihood(y.T, se.T**2, ours) - restricted_log_likelihood(
        y.T, se.T**2, theirs
    )
    assert np.all(gain > 1e-6)
    assert np.all(np.abs(ours - theirs) > 1e-3)


def test_reml_estimates_maximize_the_restricted_likelihood() -> None:
    rng = np.random.default_rng(3)
    k, n = 4, 200
    se = np.exp(rng.normal(-1.5, 1.0, (k, n)))
    y = rng.normal(0.0, se) + rng.normal(0.0, 0.3, n)
    y[0] += rng.normal(0.0, 3.0, n)  # an imprecise outlying study: two maxima at times
    se[0] *= 5.0
    fit = meta_analysis(y, se)
    grid = np.concatenate([[0.0], np.geomspace(1e-9, 1e3, 1500)])
    on_grid = restricted_log_likelihood(
        y.T[:, np.newaxis], se.T[:, np.newaxis] ** 2, grid
    )
    at_estimate = restricted_log_likelihood(y.T, se.T**2, fit.tau_squared)

    # No point of a fine grid is higher, beyond rounding.
    assert np.all(at_estimate >= on_grid.max(axis=1) - 1e-12)
    # At zero, the derivative of the likelihood is not positive (a maximum there).
    zero = fit.tau_squared == 0.0
    assert zero.any() and (~zero).any()
    step = restricted_log_likelihood(y.T[zero], se.T[zero] ** 2, 1e-9)
    assert np.all(step <= at_estimate[zero] + 1e-12)


def test_paule_mandel_solves_its_estimating_equation() -> None:
    rng = np.random.default_rng(4)
    k, n = 6, 200
    se = rng.uniform(0.1, 0.5, (k, n))
    y = rng.normal(0.0, 1.0, n) + rng.normal(0.0, rng.uniform(0, 0.6, n), (k, n))
    y = y + rng.normal(0.0, se)
    fit = meta_analysis(y, se, method="pm")
    adjusted = meta_analysis(y, se, method="pm", test="knapp_hartung")
    w = 1.0 / (se**2 + fit.tau_squared)
    mu = np.sum(w * y, axis=0) / np.sum(w, axis=0)
    generalized_q = np.sum(w * (y - mu) ** 2, axis=0)
    positive = fit.tau_squared > 0.0

    assert positive.any() and (~positive).any()
    # Q(tau^2) = k - 1 where tau^2 > 0, and Cochran's Q <= k - 1 where it is zero.
    assert_allclose(generalized_q[positive], k - 1.0, rtol=1e-12)
    assert np.all(fit.q_statistic[~positive] <= k - 1.0)
    assert_allclose(fit.estimate, mu, rtol=1e-12)
    # So Knapp and Hartung's factor q is 1 there: only the distribution changes.
    assert_allclose(
        adjusted.standard_error[positive], fit.standard_error[positive], rtol=1e-12
    )


# Closed forms.


def test_equal_standard_errors_give_the_closed_forms_of_every_estimator() -> None:
    # With v_i = s^2 for every study, every estimator of tau^2 is
    # max(0, sum (y - y_bar)^2 / (k - 1) - s^2), the weights are equal, and
    # Knapp and Hartung's q is 1 wherever tau^2 > 0.
    y = np.array([[0.0, 1.0], [1.0, 1.1], [2.0, 0.9], [5.0, 1.0]])
    s = 0.5
    se = np.full(y.shape, s)
    k = 4
    mean = y.mean(axis=0)
    squares = np.sum((y - mean) ** 2, axis=0)  # 14 and 0.02
    tau2 = np.maximum(0.0, squares / (k - 1) - s**2)
    assert tau2[0] == pytest.approx(14 / 3 - 0.25) and tau2[1] == 0.0
    for method in ("dl", "reml", "pm"):
        fit = meta_analysis(y, se, method=method)
        adjusted = meta_analysis(y, se, method=method, test="knapp_hartung")
        error = np.sqrt((s**2 + tau2) / k)
        assert_allclose(fit.tau_squared, tau2, rtol=1e-13, atol=1e-15)
        assert_allclose(fit.estimate, mean, rtol=1e-14)
        assert_allclose(fit.standard_error, error, rtol=1e-13)
        assert_allclose(fit.q_statistic, squares / s**2, rtol=1e-14)
        assert_allclose(fit.i_squared, tau2 / (tau2 + s**2), rtol=1e-13, atol=1e-15)
        assert_allclose(fit.h_squared, (tau2 + s**2) / s**2, rtol=1e-13)
        assert_allclose(fit.ci_upper - fit.estimate, Z_975 * error, rtol=1e-13)
        # q = 1 with tau^2 > 0; with tau^2 = 0 it is Q / (k - 1).
        q = np.where(tau2 > 0, 1.0, squares / s**2 / (k - 1))
        assert_allclose(adjusted.standard_error, error * np.sqrt(q), rtol=1e-12)
        statistic = mean / (error * np.sqrt(q))
        assert_allclose(
            adjusted.p_value, 2 * t_sf(np.abs(statistic), k - 1), rtol=1e-12
        )
        assert_allclose(
            adjusted.ci_lower,
            mean - float(t_isf(0.025, k - 1)) * error * np.sqrt(q),
            rtol=1e-12,
        )


def test_two_studies_give_the_closed_form_of_every_estimator() -> None:
    # For two studies DerSimonian-Laird, REML and Paule-Mandel all give
    # max(0, ((y1 - y2)^2 - v1 - v2) / 2): with u = v1 + v2 + 2 tau^2, Q(tau^2) =
    # (y1 - y2)^2 / u and the restricted likelihood is -log(u) / 2 - (y1 - y2)^2 / 2u
    # plus a constant.
    y = np.array([[0.5, 0.5], [-0.3, 0.4]])
    se = np.array([[0.2, 0.2], [0.1, 0.1]])
    expected = np.maximum(0.0, ((y[0] - y[1]) ** 2 - se[0] ** 2 - se[1] ** 2) / 2)
    assert expected[0] == pytest.approx(0.295) and expected[1] == 0.0
    for method in ("dl", "reml", "pm"):
        fit = meta_analysis(y, se, method=method)
        w = 1.0 / (se**2 + expected)
        assert_allclose(fit.tau_squared, expected, rtol=1e-13, atol=1e-16)
        assert_allclose(fit.estimate, np.sum(w * y, axis=0) / w.sum(axis=0), rtol=1e-13)
        assert_array_equal(fit.n_studies, [2, 2])


def test_one_study_is_its_own_estimate_with_metafors_conventions() -> None:
    for method in METHODS:
        for test in TESTS:
            fit = meta_analysis([0.5], [0.2], method=method, test=test)
            assert fit.estimate == 0.5 and fit.standard_error == 0.2
            # The z-test, also with Knapp-Hartung, which needs two studies.
            assert fit.statistic == pytest.approx(2.5, rel=1e-15)
            assert fit.p_value == pytest.approx(2 * float(norm_sf(2.5)), rel=1e-15)
            assert fit.ci_lower == pytest.approx(0.5 - Z_975 * 0.2, rel=1e-15)
            assert (fit.tau_squared, fit.q_statistic, fit.q_p_value) == (0.0, 0.0, 1.0)
            assert (fit.i_squared, fit.h_squared, fit.n_studies) == (0.0, 1.0, 1)
            # A vector of studies is one feature: 0-d results.
            assert fit.estimate.shape == ()


def test_fixed_effect_is_the_inverse_variance_weighted_mean() -> None:
    y = np.array([1.0, 2.0, 4.0])
    v = np.array([1.0, 0.5, 0.25])  # weights 1, 2 and 4
    fit = meta_analysis(y, np.sqrt(v), method="fixed")
    mean = (1 + 4 + 16) / 7  # 3
    q = (1 - 3) ** 2 + 2 * (2 - 3) ** 2 + 4 * (4 - 3) ** 2  # 10

    assert fit.estimate == pytest.approx(mean, rel=1e-15)
    assert fit.standard_error == pytest.approx(1 / math.sqrt(7), rel=1e-15)
    assert fit.q_statistic == pytest.approx(q, rel=1e-15)
    assert fit.q_p_value == pytest.approx(math.exp(-q / 2), rel=1e-13)  # chi2 on 2 df
    assert fit.i_squared == pytest.approx((q - 2) / q, rel=1e-15)
    assert fit.h_squared == pytest.approx(q / 2, rel=1e-15)
    assert fit.tau_squared == 0.0
    adjusted = meta_analysis(y, np.sqrt(v), method="fixed", test="knapp_hartung")
    assert adjusted.standard_error == pytest.approx(math.sqrt(q / 2 / 7), rel=1e-15)


def test_level_sets_the_quantile_of_the_intervals() -> None:
    y, se = np.array([0.2, 0.5, 0.9]), np.array([0.1, 0.2, 0.3])
    for level in (0.5, 0.99):
        fit = meta_analysis(y, se, method="dl", level=level)
        adjusted = meta_analysis(y, se, method="dl", test="knapp_hartung", level=level)
        alpha = (1 - level) / 2
        assert fit.ci_upper - fit.estimate == pytest.approx(
            float(norm_isf(alpha)) * fit.standard_error, rel=1e-14
        )
        assert adjusted.estimate - adjusted.ci_lower == pytest.approx(
            float(t_isf(alpha, 2)) * adjusted.standard_error, rel=1e-14
        )
        assert fit.level == level


# Invariants.


def random_studies(seed: int, k: int = 6, n: int = 50) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    se = rng.uniform(0.05, 0.6, (k, n))
    y = rng.normal(0.0, 1.0, n) + rng.normal(0.0, rng.uniform(0, 0.5, n), (k, n))
    return y + rng.normal(0.0, se), se


def assert_fits_equal(
    first: MetaAnalysisFit, second: MetaAnalysisFit, rtol: float
) -> None:
    for field in FIELDS:
        assert_allclose(
            getattr(first, field), getattr(second, field), rtol=rtol, atol=0
        )


@pytest.mark.parametrize("method", METHODS)
def test_the_order_of_the_studies_does_not_matter(method: str) -> None:
    y, se = random_studies(5)
    order = np.random.default_rng(6).permutation(y.shape[0])
    for test in TESTS:
        assert_fits_equal(
            meta_analysis(y, se, method=method, test=test),
            meta_analysis(y[order], se[order], method=method, test=test),
            rtol=1e-11,
        )


@pytest.mark.parametrize("method", METHODS)
def test_scaling_and_shifting_the_effects(method: str) -> None:
    y, se = random_studies(7)
    fit = meta_analysis(y, se, method=method, test="knapp_hartung")
    unchanged = ("statistic", "p_value", "q_statistic", "q_p_value", "i_squared")
    unchanged += ("h_squared", "n_studies")
    # A power of two scales every number exactly.
    scaled = meta_analysis(1024 * y, 1024 * se, method=method, test="knapp_hartung")
    for field in ("estimate", "standard_error", "ci_lower", "ci_upper"):
        assert_array_equal(getattr(scaled, field), 1024 * getattr(fit, field))
    assert_array_equal(scaled.tau_squared, 1024**2 * fit.tau_squared)
    for field in unchanged:
        assert_array_equal(getattr(scaled, field), getattr(fit, field))
    # Another factor, to rounding.
    scaled = meta_analysis(-3.7 * y, 3.7 * se, method=method, test="knapp_hartung")
    assert_allclose(scaled.estimate, -3.7 * fit.estimate, rtol=1e-11)
    assert_allclose(
        scaled.tau_squared, 3.7**2 * fit.tau_squared, rtol=1e-11, atol=1e-15
    )
    assert_allclose(scaled.statistic, -fit.statistic, rtol=1e-11)
    # A shift moves the estimate and its interval only.
    shifted = meta_analysis(y + 2.5, se, method=method, test="knapp_hartung")
    for field in ("estimate", "ci_lower", "ci_upper"):
        assert_allclose(getattr(shifted, field), getattr(fit, field) + 2.5, rtol=1e-12)
    for field in ("standard_error", "q_statistic", "i_squared", "h_squared"):
        assert_allclose(getattr(shifted, field), getattr(fit, field), rtol=1e-11)
    assert_allclose(shifted.tau_squared, fit.tau_squared, rtol=1e-10, atol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_a_study_that_did_not_measure_a_feature_is_left_out(method: str) -> None:
    y, se = random_studies(8, k=5, n=4)
    y[1, 0] = np.nan  # an effect missing
    se[3, 1] = np.nan  # a standard error missing
    y[[0, 2], 2] = np.nan
    se[[0, 2], 2] = np.nan
    fit = meta_analysis(y, se, method=method)
    for feature, studies in ((0, [0, 2, 3, 4]), (1, [0, 1, 2, 4]), (2, [1, 3, 4])):
        alone = meta_analysis(y[studies, feature], se[studies, feature], method=method)
        for field in FIELDS:
            assert getattr(fit, field)[feature] == getattr(alone, field), field
    assert_array_equal(fit.n_studies, [4, 4, 3, 5])
    # Each feature is analysed on its own.
    single = meta_analysis(y[:, 3:], se[:, 3:], method=method)
    for field in FIELDS:
        assert_array_equal(getattr(single, field), getattr(fit, field)[3:])


def test_features_of_no_study_and_identical_effects() -> None:
    y = np.array([[np.nan, 0.5], [np.nan, 0.5], [np.nan, 0.5]])
    se = np.array([[0.1, 0.1], [0.2, 0.2], [np.nan, 0.25]])
    for method in METHODS:
        fit = meta_analysis(y, se, method=method)
        assert fit.n_studies[0] == 0
        assert all(np.isnan(getattr(fit, field)[0]) for field in FIELDS[:-1])
        # Identical effects: no heterogeneity at all.
        assert (fit.estimate[1], fit.tau_squared[1], fit.q_statistic[1]) == (0.5, 0, 0)
        assert fit.q_p_value[1] == 1.0 and fit.i_squared[1] == 0.0
        # H^2 = Q / (k - 1) for the fixed effect, as metafor gives it; 1 otherwise.
        assert fit.h_squared[1] == (0.0 if method == "fixed" else 1.0)


def test_a_far_more_precise_study_dominates_without_cancellation() -> None:
    # sum w - sum w^2 / sum w cancels when one weight dominates; formed from the
    # weights of the other studies it does not, so DerSimonian-Laird keeps its digits.
    y = np.array([0.0, 1.0, 3.0])
    v = np.array([1e-10, 1.0, 1.0])
    fit = meta_analysis(y, np.sqrt(v), method="dl")
    w = 1.0 / v
    total = w.sum()
    centre = np.sum(w * y) / total
    q = np.sum(w * (y - centre) ** 2)
    # sum over i != j of w_i w_j / sum w, without the dominant term's square.
    c = 2 * (w[0] * w[1] + w[0] * w[2] + w[1] * w[2]) / total
    assert fit.tau_squared == pytest.approx((q - 2) / c, rel=1e-13)


def test_standard_errors_far_below_the_spread_of_the_effects() -> None:
    # As the sampling variances vanish, the weights 1 / (v + tau^2) become equal, and
    # REML and Paule-Mandel tend to the unweighted variance of the effects,
    # sum (y - y_bar)^2 / (k - 1): the error is of the order of the variances.
    rng = np.random.default_rng(10)
    y = rng.normal(0.0, 1.0, (5, 300))
    for scale in (3e-8, 1e-9, 1e-12):
        se = scale * rng.uniform(0.5, 1.5, y.shape)
        variance = np.var(y, axis=0, ddof=1)
        for method in ("reml", "pm"):
            fit = meta_analysis(y, se, method=method)
            assert fit.converged.all()
            assert_allclose(fit.tau_squared, variance, rtol=1e-12)
    # One such feature, where REML once found no maximum: metafor gives 1.70706.
    y = [0.5811181041963531, -1.127317413330213, -0.3517947554042179]
    y += [2.347654639873423, 0.7329250507982262]
    se = np.array(
        [1.1163182354, 0.8131728389, 1.0605225626, 0.5383073261, 0.6153736126]
    )
    fit = meta_analysis(y, 1e-8 * se)
    assert fit.tau_squared == pytest.approx(np.var(y, ddof=1), rel=1e-12)
    assert fit.p_value == pytest.approx(0.455020, abs=1e-6)
    # Effects 1e100 times their standard errors: no square overflows or underflows.
    for method in ("reml", "pm", "dl"):
        extreme = meta_analysis([1e100, -1e100, 0.0], [1.0, 1.0, 1.0], method=method)
        assert extreme.tau_squared == pytest.approx(1e200, rel=1e-12)


# Arguments.


@pytest.mark.parametrize(
    ("effects", "errors", "options", "error", "match"),
    [
        ([0.1, 0.2], [0.1], {}, ValueError, "same shape"),
        (np.zeros((2, 2, 2)), np.ones((2, 2, 2)), {}, ValueError, "got 3 dimensions"),
        (np.zeros((0, 3)), np.ones((0, 3)), {}, ValueError, "no studies"),
        ([0.1, np.inf], [0.1, 0.2], {}, ValueError, "effects must be finite"),
        (
            [0.1, 0.2],
            [0.1, 0.0],
            {},
            ValueError,
            "positive and finite.*0.0 at \\(1,\\)",
        ),
        ([0.1, 0.2], [-0.1, 0.2], {}, ValueError, "standard_errors must be positive"),
        ([0.1, 0.2], [0.1, np.inf], {}, ValueError, "standard_errors must be positive"),
        ([True, False], [0.1, 0.2], {}, TypeError, "effects must hold real numbers"),
        ([0.1, 0.2], ["a", "b"], {}, TypeError, "standard_errors must hold real"),
        ([0.1], [0.1], {"method": "REML"}, ValueError, "method must be one of"),
        ([0.1], [0.1], {"test": "knha"}, ValueError, "test must be one of"),
        ([0.1], [0.1], {"level": 95}, ValueError, "such as 0.95"),
        ([0.1], [0.1], {"level": 0.0}, ValueError, "level must be in \\(0, 1\\)"),
        ([0.1], [0.1], {"level": True}, TypeError, "level must be a real number"),
    ],
)
def test_invalid_arguments_are_refused(
    effects: Any, errors: Any, options: dict[str, Any], error: type, match: str
) -> None:
    with pytest.raises(error, match=match):
        meta_analysis(effects, errors, **options)


def test_features_and_their_shapes() -> None:
    empty = meta_analysis(np.zeros((3, 0)), np.ones((3, 0)))
    assert empty.estimate.shape == (0,) and empty.n_studies.dtype == np.int64
    y, se = random_studies(9, n=3)
    fit = meta_analysis(y, se)
    assert fit.estimate.shape == (3,) and not fit.estimate.flags.writeable
    assert fit.method == "reml" and fit.test == "z" and fit.level == 0.95
    assert repr(fit).startswith("MetaAnalysisFit(n_features=3, method='reml'")
    # Integer effects are read as numbers.
    assert meta_analysis([1, 2, 4], [1, 1, 1], method="fixed").estimate == 7 / 3


# Differential expression of several cohorts.


@cache
def cohort_results() -> dict[str, DifferentialExpressionResult]:
    """limma_voom on each cohort of the fixture, as the validation script ran it."""
    data = fixture("cohorts")
    results = {}
    for name in ("A", "B", "C"):
        samples = {c: data[f"{name} {c}"].tolist() for c in ("sample_id", "donor")}
        samples["condition"] = data[f"{name} condition"].tolist()
        cube = Cube(
            {"counts": data[f"{name} counts"]},
            samples=samples,
            groups={"group_id": data[f"{name} group_id"].tolist()},
            features={"feature_id": data[f"{name} feature_id"].tolist()},
            n_cells=data[f"{name} n_cells"],
        )
        design = Design(
            cube.samples,
            "~ donor + condition",
            levels={"condition": ["ctrl", "stim"]},
            contrasts={"stim_vs_ctrl": "conditionstim"},
        )
        results[name] = limma_voom(cube, design)
    return results


COLUMNS = FIELDS[1:] + ("adjusted_p_value",)


@pytest.mark.parametrize(
    ("method", "test"), [("reml", "z"), ("dl", "knapp_hartung")], ids=["reml", "dl"]
)
def test_meta_analysis_of_cohorts_matches_metafor(method: str, test: str) -> None:
    data = fixture("cohorts")
    meta = differential_expression_meta_analysis(
        cohort_results(), method=method, test=test
    )
    assert set(meta.tables) == {("T", "stim_vs_ctrl"), ("B", "stim_vs_ctrl")}
    for (group, contrast), table in meta.tables.items():
        prefix = f"{method} {test} 2 {group} {contrast}"
        ids = data[f"{prefix} feature_id"].tolist()
        theirs = data[f"{prefix} values"]
        # Where REML's estimate is another, higher maximum than metafor's, which the
        # validation script checks by metafor's own likelihood (none here).
        compared = ~data[f"{prefix} higher"]
        assert sorted(table["feature_id"].tolist()) == sorted(ids)
        rows = [table["feature_id"].tolist().index(feature) for feature in ids]
        columns = ("log2_fold_change", *COLUMNS)
        for column, reference in zip(columns, theirs, strict=True):
            ours = np.asarray(table[column], float)[rows]
            if column == "i_squared":
                ours = ours * 100.0
            difference = scaled_difference(ours[compared], reference[compared], column)
            assert difference <= TIERS[method], (group, column, difference)


def test_features_are_matched_by_id_across_studies() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(results, method="dl")
    table = meta.table("T", "stim_vs_ctrl")
    # Gather each feature's values by id, study by study, and meta-analyse them.
    by_id = {
        name: dict(
            zip(
                result.table("T", "stim_vs_ctrl")["feature_id"].tolist(),
                zip(
                    result.table("T", "stim_vs_ctrl")["log2_fold_change"].tolist(),
                    result.table("T", "stim_vs_ctrl")["standard_error"].tolist(),
                    strict=True,
                ),
                strict=True,
            )
        )
        for name, result in results.items()
    }
    for row in (0, 7, len(table) - 1):
        feature = table["feature_id"][row]
        values = [found[feature] for found in by_id.values() if feature in found]
        y, se = np.array(values).T
        alone = meta_analysis(y, se, method="dl")
        assert table["n_studies"][row] == len(values) >= 2
        assert table["log2_fold_change"][row] == pytest.approx(
            alone.estimate, rel=1e-14
        )
        assert table["tau_squared"][row] == pytest.approx(alone.tau_squared, rel=1e-13)
    # The adjusted p-values are Benjamini-Hochberg over the table's features.
    assert_array_equal(table["adjusted_p_value"], adjust_p_values(table["p_value"]))
    # The order of first appearance: study A lists every shared gene in order.
    assert table["feature_id"][:3].tolist() == ["g0000", "g0001", "g0002"]


def test_the_order_of_the_studies_changes_no_number() -> None:
    results = cohort_results()
    forward = differential_expression_meta_analysis(results)
    backward = differential_expression_meta_analysis(dict(reversed(results.items())))
    for key, table in forward.tables.items():
        other = backward.tables[key]
        rows = [other["feature_id"].tolist().index(f) for f in table["feature_id"]]
        assert sorted(rows) == list(range(len(other)))
        for column in ("log2_fold_change", *COLUMNS):
            assert_allclose(
                np.asarray(other[column], float)[rows],
                table[column],
                rtol=1e-11,
                atol=1e-300,
            )


def test_studies_left_out_and_features_tested_by_too_few() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(results)
    summary = meta.summary
    nk = summary.filter(summary["group_id"] == "NK")
    assert nk["status"].tolist() == ["too_few_studies"]
    assert nk["reason"][0] == "1 of the 3 studies tested it, fewer than min_studies (2)"
    with pytest.raises(KeyError, match="were not meta-analysed: 1 of the 3 studies"):
        meta.table("NK", "stim_vs_ctrl")
    studies = meta.studies.filter(meta.studies["group_id"] == "NK")
    assert studies["study"].tolist() == ["A", "B", "C"]
    assert studies["tested"].tolist() == [True, False, False]
    # B fitted no NK cells, and C has none: the reasons are those of their results.
    assert "was not fitted (no_samples)" in studies["reason"][1]
    assert studies["reason"][2].startswith("no group 'NK' was analysed")
    # Genes that only cohort B measured are in one study: left out of the table.
    t_row = summary.filter(summary["group_id"] == "T")
    table = meta.table("T", "stim_vs_ctrl")
    assert not any(f.startswith("b") for f in table["feature_id"].tolist())
    union = set().union(
        *(r.table("T", "stim_vs_ctrl")["feature_id"].tolist() for r in results.values())
    )
    assert t_row["n_features"][0] == len(table) == np.sum(table["n_studies"] >= 2)
    assert t_row["n_features"][0] + t_row["n_features_left_out"][0] == len(union)


def test_one_study_per_feature_gives_its_own_estimates() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(results, min_studies=1)
    table = meta.table("NK", "stim_vs_ctrl")
    own = results["A"].table("NK", "stim_vs_ctrl")
    assert table["feature_id"].tolist() == own["feature_id"].tolist()
    assert_array_equal(table["log2_fold_change"], own["log2_fold_change"])
    assert_array_equal(table["standard_error"], own["standard_error"])
    assert_array_equal(table["n_studies"], 1)
    assert_array_equal(table["tau_squared"], 0.0)
    # The z-test of the study's estimate with its standard error taken as known, also
    # with Knapp-Hartung, as metafor gives it for one study; not the study's t-test.
    z = own["log2_fold_change"] / own["standard_error"]
    assert_allclose(table["p_value"], 2 * norm_sf(np.abs(z)), rtol=1e-13)
    adjusted = differential_expression_meta_analysis(
        results, min_studies=1, test="knapp_hartung"
    ).table("NK", "stim_vs_ctrl")
    assert_array_equal(adjusted["p_value"], table["p_value"])
    assert np.all(table["p_value"] <= own["p_value"])
    # min_studies=1 also keeps the genes of cohort B alone.
    assert any(f.startswith("b") for f in meta.table("T", "stim_vs_ctrl")["feature_id"])


def test_groups_and_contrasts_can_be_chosen() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(
        results, groups="B", contrasts=["stim_vs_ctrl"], method="fixed", level=0.9
    )
    assert list(meta.tables) == [("B", "stim_vs_ctrl")]
    assert meta.summary["group_id"].tolist() == ["B"]
    table = meta.table("B", "stim_vs_ctrl")
    assert_allclose(
        table["ci_upper"] - table["log2_fold_change"],
        float(norm_isf(0.05)) * table["standard_error"],
        rtol=1e-12,
    )
    with pytest.raises(KeyError, match="no group 'T' was analysed; groups are: B"):
        meta.table("T", "stim_vs_ctrl")
    with pytest.raises(KeyError, match="no contrast 'other' was analysed"):
        meta.table("B", "other")


def test_the_result_records_its_studies_and_settings() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(
        results, method="pm", test="knapp_hartung"
    )
    record = meta.provenance
    assert record["analysis"] == "differential_expression_meta_analysis"
    assert meta.settings == {
        "method": "pm",
        "test": "knapp_hartung",
        "level": 0.95,
        "min_studies": 2,
        "p_value_adjustment": "BH",
    }
    # Each study's own record, which names its cube by fingerprint.
    assert record["studies"] == {name: r.provenance for name, r in results.items()}
    analyses = {(a["group_id"], a["contrast"]): a for a in record["analyses"]}
    assert analyses["T", "stim_vs_ctrl"]["studies"] == ["A", "B", "C"]
    assert set(analyses["NK", "stim_vs_ctrl"]["left_out"]) == {"B", "C"}
    # Immutable, and pickled whole.
    with pytest.raises(AttributeError, match="immutable"):
        meta.summary = meta.studies  # type: ignore[misc]
    copy = pickle.loads(pickle.dumps(meta))
    assert copy.provenance == record
    assert all(copy.tables[key].equals(table) for key, table in meta.tables.items())
    assert repr(meta).splitlines()[0] == (
        "MetaAnalysisResult (pm, test knapp_hartung): 3 studies; 3 group and contrast "
        "pairs, 2 analysed"
    )


def renamed(result: DifferentialExpressionResult, prefix: str, **changes: Any) -> Any:
    """The result with every feature id prefixed, and some values changed."""
    tables = {}
    for key, table in result.tables.items():
        ids = [prefix + feature for feature in table["feature_id"].tolist()]
        tables[key] = table.with_columns({"feature_id": ids, **changes})
    return DifferentialExpressionResult(
        result.method,
        result.contrasts,
        result.summary,
        result.samples,
        tables,
        result.provenance,
    )


def test_groups_and_contrasts_without_shared_features() -> None:
    results = cohort_results()
    meta = differential_expression_meta_analysis(
        {"A": results["A"], "other": renamed(results["B"], "x")}, groups="T"
    )
    assert meta.tables == {}
    assert meta.summary["status"].tolist() == ["no_features"]
    n = len(results["A"].table("T", "stim_vs_ctrl")) + len(
        results["B"].table("T", "stim_vs_ctrl")
    )
    assert meta.summary["n_features_left_out"][0] == n
    with pytest.raises(KeyError, match=f"none of the {n} features that the 2 studies"):
        meta.table("T", "stim_vs_ctrl")


def test_a_value_the_meta_analysis_cannot_use_names_its_feature() -> None:
    results = cohort_results()
    table = results["B"].table("T", "stim_vs_ctrl")
    errors = np.array(table["standard_error"])
    errors[5] = 0.0
    bad = renamed(results["B"], "", standard_error=errors)
    feature = table["feature_id"][5]
    with pytest.raises(
        ValueError,
        match=(
            f"the table of study 'B' for group 'T' and contrast 'stim_vs_ctrl' has the "
            f"standard_error 0.0 for feature '{feature}'; it must be positive and finite"
        ),
    ):
        differential_expression_meta_analysis({"A": results["A"], "B": bad})


def test_estimates_that_did_not_converge_are_reported(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The iterations converge on every input the other tests use; this checks how
    # the wrapper reports a feature whose estimate does not.
    import cytepool.meta_analysis as module

    original = module.meta_analysis

    def first_fails(*arguments: Any, **options: Any) -> MetaAnalysisFit:
        fit = original(*arguments, **options)
        flags = np.array(fit.converged)
        flags[0] = False
        return dataclasses.replace(fit, converged=flags)

    monkeypatch.setattr(module, "meta_analysis", first_fails)
    with pytest.warns(RuntimeWarning, match="did not converge for 1 feature, 'g0000'"):
        meta = differential_expression_meta_analysis(cohort_results(), groups="T")
    record = meta.provenance["analyses"][0]
    assert record["not_converged"] == ["g0000"]


def test_invalid_studies_and_options_are_refused() -> None:
    results = cohort_results()
    a = results["A"]
    with pytest.raises(TypeError, match="results must be a mapping of study names"):
        differential_expression_meta_analysis([a, a])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="keyed by study names"):
        differential_expression_meta_analysis({1: a, 2: a})  # type: ignore[dict-item]
    with pytest.raises(TypeError, match=r"results\['x'\] must be a Differential"):
        differential_expression_meta_analysis({"a": a, "x": a.summary})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="at least two studies, got 1"):
        differential_expression_meta_analysis({"a": a})
    with pytest.raises(KeyError, match="no study analysed the group 'DC'"):
        differential_expression_meta_analysis(results, groups=["T", "DC"])
    with pytest.raises(ValueError, match="groups lists 'T' more than once"):
        differential_expression_meta_analysis(results, groups=["T", "T"])
    with pytest.raises(ValueError, match="contrasts is empty"):
        differential_expression_meta_analysis(results, contrasts=[])
    with pytest.raises(TypeError, match="contrasts must be contrast names"):
        differential_expression_meta_analysis(results, contrasts=[1])
    with pytest.raises(ValueError, match="min_studies must be at least 1, got 0"):
        differential_expression_meta_analysis(results, min_studies=0)
    with pytest.raises(
        ValueError, match=r"min_studies \(4\) is larger than the number"
    ):
        differential_expression_meta_analysis(results, min_studies=4)
    with pytest.raises(TypeError, match="min_studies must be a whole number"):
        differential_expression_meta_analysis(results, min_studies=True)
    with pytest.raises(ValueError, match="method must be one of"):
        differential_expression_meta_analysis(results, method="ML")
    # A result whose tables lack the standard errors.
    table = a.table("T", "stim_vs_ctrl")
    bare = DifferentialExpressionResult(
        a.method,
        a.contrasts,
        a.summary,
        a.samples,
        {("T", "stim_vs_ctrl"): table.select(["feature_id", "log2_fold_change"])},
        a.provenance,
    )
    with pytest.raises(ValueError, match="has no column 'standard_error'"):
        differential_expression_meta_analysis({"a": a, "bare": bare})
    with pytest.raises(TypeError, match="tables must be keyed by"):
        MetaAnalysisResult(Table(), Table(), {"T": table}, {})  # type: ignore[dict-item]
