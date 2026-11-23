"""Tests for the curves of named dose-response models, fit_dose_curves and DoseCurveFits.

Reference values in ``tests/data/dose_curves/references.json`` come from pybmds 25.2,
drc 4.0-0, DRomics 2.6.3 and a 200-start scipy 1.18.1 search, written by
``validation/dose_curves.py``, which runs each reference as a program of its own on the
doses and values, with each model's mean and bounds written from the documentation.
The other tests check what the references cannot: the means written here from the
documented formulas, the analytic gradients against central differences, least
squares and likelihoods derived by hand, the rules of declines and turns, and
invariants of units, order and threads.
"""

import functools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from cytepool.stats import _dose_curve_models as models
from cytepool.stats.dose_curves import (
    _constant_variance_data,
    _CurveData,
    DoseCurveFits,
    fit_dose_curves,
    _fit_means,
)
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "dose_curves"
HALF_LOG = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
SETS = ("epa", "ll4", "dromics")


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Phi(z) from erfc, apart from CytePool's."""
    return 0.5 * np.vectorize(math.erfc)(-np.asarray(z, dtype=float) / math.sqrt(2.0))


def documented(model: str, theta: Any, x: np.ndarray) -> np.ndarray:
    """Each model's mean as the chapter writes it, on reported parameters."""
    t = [float(v) for v in theta]
    x = np.asarray(x, dtype=float)
    with np.errstate(all="ignore"):
        if model == "linear_epa":
            return t[0] + t[1] * x
        if model == "linear_dromics":
            return t[0] + t[1] * x  # d + b x
        if model == "polynomial_2":
            return t[0] + t[1] * x + t[2] * x * x
        if model == "power":
            return t[0] + t[1] * x ** t[2]
        if model == "hill":
            g, v, k, n = t
            return g + v * np.where(x > 0, x**n / (k**n + x**n), 0.0)
        if model == "exponential_2":
            return t[0] * np.exp(t[1] * x)
        if model == "exponential_3":
            a, b, d = t
            return a * np.exp(np.sign(b) * (abs(b) * x) ** d)
        if model == "exponential_4":
            a, b, c = t
            return a * (c - (c - 1) * np.exp(-b * x))
        if model == "exponential_5":
            a, b, c, d = t
            return a * (c - (c - 1) * np.exp(-((b * x) ** d)))
        if model == "ll4":
            b, c, d, e = t
            return c + (d - c) / (1 + (x / e) ** b)
        if model == "exponential":
            b, d, e = t
            return d + b * (np.exp(x / e) - 1)
        if model.endswith("symmetric"):
            b, d, e, f = t
            c = d
        elif model.endswith("monotone"):
            b, c, d, e = t
            f = 0.0
        else:
            b, c, d, e, f = t
        if model.startswith("log"):
            z = np.where(
                x > 0, (np.log(np.where(x > 0, x, 1.0)) - math.log(e)) / b, -np.inf
            )
        else:
            z = (x - e) / b
        return f * np.exp(-z * z / 2) + d + (c - d) * normal_cdf(z)


# Every model with parameters inside its bounds at the half-log ladder, identified by
# its nine means.
TRUE_CURVES = {
    ("epa", "linear"): [5.0, 0.1],
    ("epa", "polynomial_2"): [5.0, 0.05, 0.002],
    ("epa", "power"): [5.0, 0.3, 1.6],
    ("epa", "hill"): [6.0, 2.0, 0.8, 2.2],
    ("epa", "exponential_2"): [5.0, 0.02],
    ("epa", "exponential_3"): [5.0, -0.05, 1.8],
    ("epa", "exponential_4"): [5.0, 0.4, 1.6],
    ("epa", "exponential_5"): [5.0, 0.3, 0.5, 2.0],
    ("ll4", "ll4"): [1.5, 8.0, 6.0, 0.5],
    ("dromics", "linear"): [5.0, 0.1],
    ("dromics", "exponential"): [0.5, 5.0, 10.0],
    ("dromics", "ll4"): [1.5, 6.0, 8.0, 0.5],
    ("dromics", "gauss_probit"): [8.0, 7.0, 6.0, 10.0, 1.0],
    ("dromics", "gauss_probit_monotone"): [8.0, 7.5, 6.0, 12.0],
    ("dromics", "gauss_probit_symmetric"): [8.0, 6.0, 12.0, 1.5],
    ("dromics", "log_gauss_probit"): [1.2, 7.0, 6.0, 0.5, 1.0],
    ("dromics", "log_gauss_probit_monotone"): [1.5, 7.5, 6.0, 0.5],
    ("dromics", "log_gauss_probit_symmetric"): [1.2, 6.0, 0.5, 1.5],
}


def formula_name(model_set: str, model: str) -> str:
    return f"linear_{model_set}" if model == "linear" else model


def parameters_of(fits: DoseCurveFits, model: str) -> np.ndarray:
    """A model's parameters for every gene, in the model's own order."""
    spec = models.model_of(fits.model_set, model)
    m = fits.models.index(model)
    columns = [fits.parameter_names.index(name) for name in spec.parameters]
    return np.asarray(fits.parameters[:, m][:, columns])


def replicated(
    doses: np.ndarray, means: np.ndarray, n: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """Samples of the means, ``n`` per dose: doses and values, samples x genes."""
    return np.repeat(doses, n), np.repeat(np.atleast_2d(means).T, n, axis=0)


# ---------------------------------------------------------------------------------------
# The models
# ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("model_set", "model"), list(TRUE_CURVES))
def test_noise_free_curves_are_recovered(model_set: str, model: str) -> None:
    truth = np.array(TRUE_CURVES[(model_set, model)])
    curve = documented(formula_name(model_set, model), truth, HALF_LOG)
    noise = np.where(np.arange(HALF_LOG.size) % 2, 1e-3, -1e-3)
    # A gene with the means alone and one with a little noise around them, whose
    # within-group variation keeps the likelihood finite.
    doses, values = replicated(HALF_LOG, np.vstack([curve, curve + noise]))
    values[::3, 1] += 1e-4
    fits = fit_dose_curves(doses, values, models=model_set)
    m = fits.models.index(model)
    fitted = documented(
        formula_name(model_set, model), parameters_of(fits, model)[0], HALF_LOG
    )
    assert np.max(np.abs(fitted - curve)) <= 1e-9 * np.ptp(curve)
    assert_allclose(parameters_of(fits, model)[0], truth, rtol=1e-5)
    assert fits.not_fitted_reason[0, m] == ""
    # The fitted curve of the noisy gene passes near its means, at the maximum.
    noisy = documented(
        formula_name(model_set, model), parameters_of(fits, model)[1], HALF_LOG
    )
    assert np.max(np.abs(noisy - curve)) < 0.01


@pytest.mark.parametrize("kind", range(16))
def test_gradients_are_the_means_derivatives(kind: int) -> None:
    """The analytic gradient of every kind against central differences of its mean,
    at interior points and at the limits that IEEE arithmetic leaves undefined."""
    scale = models.DoseScale.of(HALF_LOG)
    lower, upper = models.bounds(kind, scale)
    rng = np.random.default_rng(kind)
    points = []
    for _ in range(4):
        theta = np.empty(lower.size)
        for j in range(lower.size):
            lo = lower[j] if np.isfinite(lower[j]) else -3.0
            hi = upper[j] if np.isfinite(upper[j]) else 3.0
            lo, hi = lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo)
            theta[j] = rng.uniform(lo, hi)
        if kind in models.POSITIVE_MEANS:
            theta[0] = abs(theta[0]) + 1.0
        points.append(theta)
    gradient = np.zeros(lower.size)
    for theta in points:
        for x in (0.0, 0.01, 0.3, 7.0, 30.0):
            if not np.isfinite(models.mean_value(kind, theta, x)):
                continue  # an exponential beyond the largest double
            models.mean_gradient(kind, theta, x, gradient)
            for j in range(theta.size):
                h = 1e-6 * max(abs(theta[j]), 1e-2)
                up, down = theta.copy(), theta.copy()
                up[j] += h
                down[j] -= h
                numeric = (
                    models.mean_value(kind, up, x) - models.mean_value(kind, down, x)
                ) / (2 * h)
                # Truncation of order h^2, and rounding of the mean of order eps |f| / h.
                rounding = 1e-13 * abs(models.mean_value(kind, theta, x)) / h
                tolerance = 1e-5 * abs(numeric) + 1e-6 + rounding
                assert abs(gradient[j] - numeric) <= tolerance, (j, x)
    # The limits: x^n log x and x^n / (k^n + x^n) at dose 0, a location of 0, a rate
    # of 0, z at minus infinity on ln dose.
    theta = points[0].copy()
    models.mean_gradient(kind, theta, 0.0, gradient)
    assert np.isfinite(gradient).all()
    if kind == models.HILL:
        for n, expected in ((1.0, -2.0 / 0.5), (2.5, 0.0)):
            theta = np.array([1.0, 2.0, 0.0, n])
            models.mean_gradient(kind, theta, 0.5, gradient)
            assert gradient[2] == pytest.approx(expected)
            assert models.mean_value(kind, theta, 0.5) == 3.0
            assert models.mean_value(kind, theta, 0.0) == 1.0
    if kind in (models.EXP3, models.EXP5):
        theta = np.array(
            [2.0, 0.0, 1.0] if kind == models.EXP3 else [2.0, 0.0, 3.0, 1.0]
        )
        models.mean_gradient(kind, theta, 0.5, gradient)
        assert np.isfinite(gradient).all()
    if kind == models.DROMICS_EXPONENTIAL:
        # At a rate of 0, (exp(s x) - 1) / s takes its limit x, and its derivative in
        # s its limit x^2 / 2.
        theta = np.array([2.0, 5.0, 0.0])
        assert models.mean_value(kind, theta, 0.5) == 6.0
        models.mean_gradient(kind, theta, 0.5, gradient)
        assert gradient[0] == 0.5 and gradient[2] == pytest.approx(2.0 * 0.125)


@pytest.mark.parametrize(("model_set", "model"), list(TRUE_CURVES))
def test_means_follow_the_documented_formulas(model_set: str, model: str) -> None:
    spec = models.model_of(model_set, model)
    truth = np.array([TRUE_CURVES[(model_set, model)]])
    theta = np.ascontiguousarray(models.internal(spec.kind, truth))
    x = np.array([0.0, 0.004, 0.02, 0.5, 4.0, 30.0, 60.0])
    assert_allclose(
        models.mean_values(spec.kind, theta, x)[0],
        documented(formula_name(model_set, model), truth[0], x),
        rtol=1e-13,
        atol=1e-13,
    )


def test_linear_and_polynomial_are_least_squares() -> None:
    rng = np.random.default_rng(1)
    doses = np.repeat(HALF_LOG, [3, 2, 3, 3, 1, 3, 3, 2, 3])
    rising = 5.0 + 0.02 * doses + 0.003 * doses**2
    # A curve that rises and then falls, which the restriction to one sign cannot follow.
    turning = 5.0 + 0.3 * doses - 0.01 * doses**2
    values = np.column_stack([rising, turning]) + rng.normal(0, 0.1, (doses.size, 2))
    fits = fit_dose_curves(doses, values, models="epa")
    x = np.column_stack([np.ones(doses.size), doses, doses**2])
    for g in range(2):
        line, *_ = np.linalg.lstsq(x[:, :2], values[:, g], rcond=None)
        assert_allclose(parameters_of(fits, "linear")[g], line, rtol=1e-10)
        rss = np.sum((values[:, g] - x[:, :2] @ line) ** 2)
        n = doses.size
        density = -0.5 * n * np.log(2 * np.pi * rss / n) - rss / (2 * rss / n)
        assert fits.log_likelihood[g, fits.models.index("linear")] == pytest.approx(
            density, rel=1e-12
        )
    free, *_ = np.linalg.lstsq(x, values[:, 0], rcond=None)
    assert free[1] > 0 and free[2] > 0
    assert_allclose(parameters_of(fits, "polynomial_2")[0], free, rtol=1e-9)
    # With b1 > 0 and b2 < 0 the restriction binds: the better of b1 = 0 and b2 = 0.
    m = fits.models.index("polynomial_2")
    candidates = []
    for keep in ([0, 1], [0, 2]):
        coefficients, *_ = np.linalg.lstsq(x[:, keep], values[:, 1], rcond=None)
        full = np.zeros(3)
        full[keep] = coefficients
        candidates.append((np.sum((values[:, 1] - x @ full) ** 2), full))
    expected = min(candidates, key=lambda c: c[0])[1]
    assert_allclose(
        parameters_of(fits, "polynomial_2")[1], expected, rtol=1e-9, atol=1e-12
    )
    assert fits.parameters_at_bounds[1, m] in ("b1", "b2")


def test_aicc_and_akaike_weights_by_hand() -> None:
    rng = np.random.default_rng(3)
    doses = np.repeat(HALF_LOG, 3)
    values = (6.0 + 0.05 * doses + rng.normal(0, 0.2, doses.size))[:, None]
    fits = fit_dose_curves(doses, values, models="epa")
    n = doses.size
    sizes = np.array([len(m.parameters) for m in models.SETS["epa"]])
    k = sizes + 1.0
    ll = fits.log_likelihood[0]
    assert_allclose(fits.aic[0], -2 * ll + 2 * k, rtol=1e-14)
    assert_allclose(
        fits.aicc[0], -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1), rtol=1e-14
    )
    delta = fits.aicc[0] - fits.aicc[0].min()
    weights = np.exp(-delta / 2) / np.exp(-delta / 2).sum()
    assert_allclose(fits.akaike_weight[0], weights, rtol=1e-12)
    assert fits.best[0] == int(np.argmin(fits.aicc[0]))


@pytest.mark.parametrize("model_set", ["epa", "dromics"])
def test_plug_in_likelihood_aicc_and_residuals_by_hand(model_set: str) -> None:
    """In the weighted and moderated modes: the normal log-density of the means at
    the documented curve with covariance s^2 C, AICc with n the samples, and the
    largest |ybar_k - f_k| / (s sqrt(C_kk))."""
    levels = np.array([0.0, 0.1, 0.4, 1.7, 9.0])
    n_samples = np.array([3.0, 2, 3, 3, 2])
    covariance = np.diag(1.0 / n_samples) + 0.01
    data = _CurveData(
        "weighted",
        levels,
        n_samples,
        np.array([[5.0, 5.4, 5.6, 6.3, 8.1], [4.0, 4.9, 5.3, 5.2, 4.1]]),
        np.stack([covariance, covariance * 1.5]),
        np.array([0.04, 0.09]),
        np.array([7.0, 7.0]),
        None,
    )
    fits = _fit_means(data, model_set)
    n = n_samples.sum()
    checked = 0
    for g in range(2):
        cov = data.covariance_scale[g] * data.covariance[g]
        for m, model in enumerate(fits.models):
            if fits.not_fitted_reason[g, m]:
                continue
            spec = models.model_of(model_set, model)
            f = documented(
                formula_name(model_set, model), parameters_of(fits, model)[g], levels
            )
            ll = stats.multivariate_normal.logpdf(data.means[g], mean=f, cov=cov)
            assert fits.log_likelihood[g, m] == pytest.approx(ll, rel=1e-9, abs=1e-9)
            k = spec.n_parameters + 1
            aicc = -2 * ll + 2 * k + 2 * k * (k + 1) / (n - k - 1)
            assert fits.aicc[g, m] == pytest.approx(aicc, rel=1e-9, abs=1e-9)
            residual = np.max(np.abs(data.means[g] - f) / np.sqrt(np.diag(cov)))
            assert fits.max_abs_scaled_residual[g, m] == pytest.approx(
                residual, rel=1e-6, abs=1e-9
            )
            checked += 1
    assert checked >= 6


def test_within_group_sds_are_the_samples_sds() -> None:
    """Under constant variance, each dose group's sample SD with n_k - 1 in the
    denominator, NaN for a group of one sample; NaN for means of a fit."""
    doses = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 3.0, 10.0, 10.0, 10.0])
    rng = np.random.default_rng(4)
    values = 5.0 + rng.normal(0.0, 0.3, (doses.size, 2))
    values[doses == 10.0, 1] = 6.0
    fits = fit_dose_curves(doses, values, models="epa")
    for g in range(2):
        expected = [
            np.std(values[doses == level, g], ddof=1)
            if np.sum(doses == level) > 1
            else np.nan
            for level in (0.0, 1.0, 3.0, 10.0)
        ]
        assert_allclose(fits.within_group_sd[g], expected, rtol=1e-12, equal_nan=True)
    assert fits.within_group_sd[1, 3] == 0.0
    with pytest.raises(ValueError, match="read-only"):
        fits.within_group_sd[0, 0] = 1.0
    data = _CurveData(
        "weighted",
        np.array([0.0, 0.1, 0.4, 1.7, 9.0]),
        np.array([3.0, 2, 3, 3, 2]),
        np.array([[5.0, 5.4, 5.6, 6.3, 8.1]]),
        np.diag(1.0 / np.array([3.0, 2, 3, 3, 2]))[None],
        np.array([0.04]),
        np.array([7.0]),
        None,
    )
    assert np.isnan(_fit_means(data, "epa").within_group_sd).all()


def test_constant_scaled_residuals_by_hand() -> None:
    """BMDS's scaled residuals of a straight line: (ybar_k - f_k) / (sigma / sqrt(n_k))
    with sigma^2 the residual sum of squares of the samples over n."""
    rng = np.random.default_rng(1)
    doses = np.repeat(HALF_LOG, [3, 2, 3, 3, 1, 3, 3, 2, 3])
    values = (5.0 + 0.02 * doses + rng.normal(0, 0.1, doses.size))[:, None]
    fits = fit_dose_curves(doses, values, models="epa")
    x = np.column_stack([np.ones(doses.size), doses])
    line, *_ = np.linalg.lstsq(x, values[:, 0], rcond=None)
    sigma = math.sqrt(np.sum((values[:, 0] - x @ line) ** 2) / doses.size)
    levels, counts = np.unique(doses, return_counts=True)
    means = np.array([values[doses == d, 0].mean() for d in levels])
    expected = np.max(
        np.abs(means - line[0] - line[1] * levels) / (sigma / np.sqrt(counts))
    )
    m = fits.models.index("linear")
    assert fits.max_abs_scaled_residual[0, m] == pytest.approx(expected, rel=1e-10)


def test_ties_infinite_and_unbounded_criteria() -> None:
    from cytepool.stats.dose_curves import _akaike_weights, _best

    not_fitted = np.full((4, 3), "", dtype=object)
    aicc = np.array(
        [
            [1.0, 1.0 + 5e-9, 3.0],  # a tie within 1e-8: the first model of the set
            [np.inf, 2.0, np.inf],  # infinite AICc: no weight
            [-np.inf, 2.0, -np.inf],  # perfect fits share the weight
            [np.inf, np.inf, np.inf],  # no finite AICc: no best model
        ]
    )
    best = _best(aicc, not_fitted)
    assert best.tolist() == [0, 1, 0, -1]
    weights = _akaike_weights(aicc, not_fitted)
    assert weights[1].tolist() == [0.0, 1.0, 0.0]
    assert weights[2].tolist() == [0.5, 0.0, 0.5]
    assert np.isnan(weights[3]).all()
    not_fitted[0, 0] = "the values do not vary"
    assert _best(aicc, not_fitted)[0] == 1
    assert np.isnan(_akaike_weights(aicc, not_fitted)[0, 0])


def test_lack_of_fit_by_hand() -> None:
    rng = np.random.default_rng(5)
    doses = np.repeat([0.0, 0.1, 0.4, 1.7, 9.0], [3, 2, 3, 3, 2])
    values = (5.0 + np.sqrt(doses) + rng.normal(0, 0.2, doses.size))[:, None]
    fits = fit_dose_curves(doses, values, models="epa")
    m = fits.models.index("linear")
    levels = np.unique(doses)
    within = sum(
        np.sum((values[doses == d, 0] - values[doses == d, 0].mean()) ** 2)
        for d in levels
    )
    line = np.polyfit(doses, values[:, 0], 1)
    rss = np.sum((values[:, 0] - np.polyval(line, doses)) ** 2)
    # The pure-error F of lack of fit: the line's excess over free group means, on
    # K - 2 and n - K degrees of freedom.
    n, k = doses.size, levels.size
    f = ((rss - within) / (k - 2)) / (within / (n - k))
    assert fits.lack_of_fit_p_value[0, m] == pytest.approx(
        float(stats.f.sf(f, k - 2, n - k)), rel=1e-10
    )
    # The weighted mode at unequally spaced doses: F of contrasts that span the null
    # space of [1, x], with the means' covariance.
    k = levels.size
    covariance = np.diag(1.0 / np.array([3, 2, 3, 3, 2])) + 0.01
    means = np.array([5.0, 5.4, 5.6, 6.3, 8.1])[None]
    data = _CurveData(
        "weighted",
        levels,
        np.array([3.0, 2, 3, 3, 2]),
        means,
        covariance[None],
        np.array([0.04]),
        np.array([7.0]),
        None,
    )
    weighted = _fit_means(data, "epa")
    q, _ = np.linalg.qr(np.column_stack([np.ones(k), levels]), mode="complete")
    contrasts = q[:, 2:]
    estimate = contrasts.T @ means[0]
    f = (
        estimate
        @ np.linalg.solve(contrasts.T @ covariance @ contrasts, estimate)
        / (k - 2)
        / 0.04
    )
    m = weighted.models.index("linear")
    assert weighted.lack_of_fit_p_value[0, m] == pytest.approx(
        float(stats.f.sf(f, k - 2, 7.0)), rel=1e-10
    )
    # On infinite degrees of freedom, a chi-square.
    data = _CurveData(
        "moderated",
        levels,
        np.array([3.0, 2, 3, 3, 2]),
        means,
        covariance[None],
        np.array([0.04]),
        np.array([np.inf]),
        None,
    )
    assert _fit_means(data, "epa").lack_of_fit_p_value[0, m] == pytest.approx(
        float(stats.chi2.sf(f * (k - 2), k - 2)), rel=1e-10
    )


def test_declines() -> None:
    rng = np.random.default_rng(7)
    doses = np.repeat([0.0, 0.1, 1.0, 10.0], 3)
    values = 2.0 + rng.normal(0, 0.2, (doses.size, 3))
    values[:, 1] -= 2.0  # dose-group means around 0
    values[:, 2] = 4.0  # no variation
    fits = fit_dose_curves(doses, values, models="epa")
    assert set(fits.models_not_fitted) == {"hill", "exponential_5"}
    assert "4 dose groups and 4 parameters" in fits.models_not_fitted["hill"]
    exponential = [
        fits.models.index(m)
        for m in ("exponential_2", "exponential_3", "exponential_4")
    ]
    others = [m for m in range(len(fits.models)) if m not in exponential]
    assert (fits.not_fitted_reason[0] == "").all()
    assert (fits.not_fitted_reason[1, exponential] != "").all()
    assert (fits.not_fitted_reason[1, others] == "").all()
    assert np.isnan(fits.parameters[1, exponential]).all()
    assert np.isnan(fits.akaike_weight[1, exponential]).all()
    assert fits.akaike_weight[1, others].sum() == pytest.approx(1.0)
    assert (fits.not_fitted_reason[2] == "the values do not vary").all()
    assert fits.best[2] == -1
    # A mean of exactly 0 is not fitted either; the boundary is at 0, not below it.
    values[:, 1] = np.where(doses == 0.0, 0.0, 1.0) + rng.normal(0, 0.1, doses.size) * (
        doses > 0
    )
    values[doses == 0.0, 1] = [-0.1, 0.0, 0.1]
    fits = fit_dose_curves(doses, values, models="epa")
    assert (fits.not_fitted_reason[1, exponential] != "").all()
    with pytest.raises(ValueError, match="needs 5 dose groups"):
        fit_dose_curves(doses, values, models="ll4")


def test_bounds_are_reported() -> None:
    doses = np.repeat(HALF_LOG, 3)
    # Group means that step exactly at the lowest dose, with variation within groups:
    # Hill's curve is the step, which any k far below the lowest dose gives to
    # rounding, k = 0 among them.
    step = np.where(doses > 0, 7.0, 5.0) + np.tile([-0.1, 0.0, 0.1], HALF_LOG.size)
    fits = fit_dose_curves(doses, step[:, None], models="epa")
    m = fits.models.index("hill")
    hill = parameters_of(fits, "hill")[0]
    assert hill[2] < 0.01
    assert_allclose(
        documented("hill", hill, HALF_LOG), np.where(HALF_LOG > 0, 7.0, 5.0), atol=1e-12
    )
    n = doses.size
    within = 2 * 0.01 * HALF_LOG.size
    assert fits.log_likelihood[0, m] == pytest.approx(
        -0.5 * n * (np.log(2 * np.pi * within / n) + 1), rel=1e-12
    )
    # With noise, any k below the lowest dose with a steep n fits nearly as well, and
    # the maximum is at least the step's.
    rng = np.random.default_rng(11)
    noisy = np.where(doses > 0, 7.0, 5.0) + rng.normal(0, 0.1, doses.size)
    fits = fit_dose_curves(doses, noisy[:, None], models="epa")
    treated, vehicle = noisy[doses > 0], noisy[doses == 0]
    rss = np.sum((treated - treated.mean()) ** 2) + np.sum(
        (vehicle - vehicle.mean()) ** 2
    )
    assert fits.log_likelihood[0, m] >= -0.5 * n * (np.log(2 * np.pi * rss / n) + 1)
    # The straight line of the power model: n at its lower bound of 1.
    power = fits.models.index("power")
    assert fits.parameters_at_bounds[0, power] == "n"
    assert parameters_of(fits, "power")[0, 2] == 1.0
    # A plateau 30 times the vehicle's log2 level, beyond the 20 of pybmds's table of
    # priors, which pybmds does not enforce: c is recovered, off any bound. A plateau
    # 1,200 times it, as from 0.005 to 6 log2 CPM: c at its bound of 1,000.
    within_groups = np.tile([-1e-4, 0.0, 1e-4], HALF_LOG.size)
    for a, c in ((0.2, 30.0), (0.005, 1200.0)):
        induced = np.repeat(documented("exponential_4", [a, 1.0, c], HALF_LOG), 3)
        fits = fit_dose_curves(doses, (induced + within_groups)[:, None], models="epa")
        m = fits.models.index("exponential_4")
        found = parameters_of(fits, "exponential_4")[0, 2]
        at_bound = "c" in fits.parameters_at_bounds[0, m].split(", ")
        if c < 1000.0:
            assert found == pytest.approx(c, rel=1e-6) and not at_bound
        else:
            assert found == 1000.0 and at_bound


def test_turns_and_the_group_they_rest_on() -> None:
    rng = np.random.default_rng(13)
    doses = np.repeat(HALF_LOG, 3)
    # Flat genes with noise and one dose group raised by 1.5 log2 units, an outlying
    # group; and bells over three groups, with the same noise.
    spike = 6.0 + rng.normal(0, 0.15, (doses.size, 8))
    spike[doses == 0.3] += 1.5
    log_dose = np.log10(np.where(doses > 0, doses, 1e-9))
    bell = 6.0 + 1.5 * np.exp(-0.5 * ((log_dose - np.log10(0.3)) / 0.6) ** 2)
    bell = bell[:, None] + rng.normal(0, 0.15, (doses.size, 8))
    fits = fit_dose_curves(doses, np.hstack([spike, bell]), models="dromics")
    rows = np.arange(16)
    assert np.isfinite(fits.turn_dose[rows, fits.best]).all()
    # Without the raised group, the best fit of the set does not turn: the turn rests
    # on it. The noise of a flat gene can support a turning curve on its own, rarely.
    driven = fits.turn_driven_by_dose[:8]
    assert np.count_nonzero(driven == 0.3) >= 7
    assert np.isnan(driven[driven != 0.3]).all()
    assert np.isnan(fits.turn_driven_by_dose[8:]).all()
    assert "16 best fits turn, " in repr(fits)
    assert f"{np.count_nonzero(np.isfinite(driven))} driven by one dose group" in repr(
        fits
    )


def test_locations_reach_beyond_the_doses() -> None:
    """Hill's k and the log-logistic e reach 5 times the top dose, BMDS's bound: a
    curve whose midpoint lies beyond the doses is recovered, and one beyond the
    bound stops there."""
    for model_set, model, truth in (
        ("epa", "hill", [6.0, 4.0, 60.0, 2.0]),
        ("ll4", "ll4", [2.0, 10.0, 6.0, 60.0]),
    ):
        curve = documented(formula_name(model_set, model), truth, HALF_LOG)
        doses, values = replicated(HALF_LOG, curve)
        fits = fit_dose_curves(doses, values, models=model_set)
        assert_allclose(parameters_of(fits, model)[0], truth, rtol=1e-4)
    curve = documented("hill", [6.0, 8.0, 400.0, 1.5], HALF_LOG)
    doses, values = replicated(HALF_LOG, curve)
    values[::3, 0] += 1e-3
    fits = fit_dose_curves(doses, values, models="epa")
    m = fits.models.index("hill")
    assert parameters_of(fits, "hill")[0, 2] == 150.0
    assert "k" in fits.parameters_at_bounds[0, m].split(", ")


def test_gauss_probit_bounds_hold() -> None:
    """Each Gauss-probit peak or inflection lies within the doses and each width is
    at least a third of the largest gap between adjacent doses, on dose or on ln
    dose among the positive doses."""
    rng = np.random.default_rng(23)
    doses = np.repeat(HALF_LOG, 3)
    values = 6.0 + rng.normal(0, 0.15, (doses.size, 12))
    values[doses == 0.3, :6] += 1.5
    fits = fit_dose_curves(doses, values, models="dromics")
    positive = np.log(HALF_LOG[1:])
    for model in fits.models:
        if "gauss_probit" not in model:
            continue
        theta = parameters_of(fits, model)
        peak = theta[:, 2] if model.endswith("symmetric") else theta[:, 3]
        if model.startswith("log"):
            assert np.all(theta[:, 0] >= np.max(np.diff(positive)) / 3 * (1 - 1e-12))
            assert np.all((peak >= 0.01) & (peak <= 30.0))
        else:
            assert np.all(theta[:, 0] >= np.max(np.diff(HALF_LOG)) / 3 * (1 - 1e-12))
            assert np.all((peak >= 0.0) & (peak <= 30.0))


def test_a_turn_below_the_lowest_dose_counts() -> None:
    """A turn between the vehicle and the lowest dose lies within the doses, strictly
    between 0 and D: the curve rises and falls over them."""
    # log-Gauss-probit (b, c, d, e, f): z* = (c - d) / (f sqrt(2 pi)) < 0, so the
    # turn lies below e, here below the lowest dose of 0.01.
    theta = np.array([[1.0, 8.0, 6.0, 0.01, -3.0]])
    z = 2.0 / (-3.0 * math.sqrt(2 * math.pi))
    expected = 0.01 * math.exp(z)
    assert 0 < expected < 0.01
    found = models.turn_doses(models.LOG_GAUSS_PROBIT, theta, 30.0)
    assert found[0] == pytest.approx(expected, rel=1e-14)
    # A turn beyond the top dose does not count.
    theta[0, 3] = 30.0
    theta[0, 4] = 3.0
    assert np.isnan(models.turn_doses(models.LOG_GAUSS_PROBIT, theta, 30.0)[0])


def test_a_turn_can_rest_on_the_vehicle() -> None:
    """A low vehicle before treated doses that fall steadily makes a curve that rises
    and falls; without the vehicle's group, the best fit falls: the vehicle's group,
    at dose 0, drives the turn."""
    rng = np.random.default_rng(29)
    doses = np.repeat(HALF_LOG, 3)
    treated = 8.0 - 0.4 * np.log10(np.where(doses > 0, doses, 1.0) / 0.01)
    values = np.where(doses > 0, treated, 5.0)[:, None] + rng.normal(
        0, 0.1, (doses.size, 4)
    )
    fits = fit_dose_curves(doses, values, models="dromics")
    rows = np.arange(4)
    assert np.isfinite(fits.turn_dose[rows, fits.best]).all()
    assert_array_equal(fits.turn_driven_by_dose, 0.0)


def test_a_peak_at_the_top_dose_is_not_a_turn() -> None:
    """A curve whose slope changes sign at the top dose rises over every dose: it does
    not turn within them, also when the peak sits at its bound, e = D."""
    for kind, theta in (
        (models.GAUSS_PROBIT, [2.0, 7.0, 7.0, 30.0, 1.0]),
        (models.GAUSS_PROBIT_SYMMETRIC, [2.0, 7.0, 30.0, 1.0]),
        (models.LOG_GAUSS_PROBIT_SYMMETRIC, [1.0, 7.0, 30.0, 1.0]),
    ):
        found = models.turn_doses(kind, np.array([theta]), 30.0)
        assert np.isnan(found[0]), kind
        # Just below the top dose, it turns there.
        below = list(theta)
        below[3 if kind == models.GAUSS_PROBIT else 2] = 29.9
        assert models.turn_doses(kind, np.array([below]), 30.0)[0] == pytest.approx(
            29.9, rel=1e-14
        )
    # Genes that rise at the top dose alone: no best fit turns.
    rng = np.random.default_rng(37)
    doses = np.repeat(HALF_LOG, 3)
    values = np.where(doses == 30.0, 7.5, 6.0)[:, None] + rng.normal(
        0, 0.15, (doses.size, 8)
    )
    fits = fit_dose_curves(doses, values, models="dromics")
    rows = np.arange(8)
    assert np.isnan(fits.turn_dose[rows, fits.best]).all()


def test_turn_doses_are_extrema_of_the_documented_mean() -> None:
    """Without the closed form: where a fit turns, the documented mean has an
    extremum, its differences on either side of opposite signs."""
    rng = np.random.default_rng(13)
    doses = np.repeat(HALF_LOG, 3)
    log_dose = np.log10(np.where(doses > 0, doses, 1e-9))
    bell = 6.0 + 1.5 * np.exp(-0.5 * ((log_dose - np.log10(0.3)) / 0.6) ** 2)
    values = bell[:, None] + rng.normal(0, 0.15, (doses.size, 8))
    values[doses == 3.0, 4:] += 1.5  # and spikes at one dose
    fits = fit_dose_curves(doses, values, models="dromics")
    checked = 0
    for m, model in enumerate(fits.models):
        theta = parameters_of(fits, model)
        for g in range(values.shape[1]):
            x = fits.turn_dose[g, m]
            if not np.isfinite(x):
                continue
            step = 1e-3 * x if model.startswith("log") else 1e-3 * HALF_LOG[-1]
            low, mid, high = documented(
                model, theta[g], np.array([x - step, x, x + step])
            )
            assert (mid - low) * (high - mid) < 0, (g, model, x)
            checked += 1
    assert checked >= 16


def test_the_lower_of_two_driving_doses_is_named() -> None:
    """A plateau above a low vehicle and a low top dose: without either group, the
    best fit no longer turns, and the lower dose, the vehicle's, is reported."""
    rng = np.random.default_rng(3)
    doses = np.repeat(HALF_LOG, 3)
    level = np.where((doses > 0) & (doses < 30), 7.0, 5.0)
    values = level[:, None] + rng.normal(0, 0.1, (doses.size, 6))
    fits = fit_dose_curves(doses, values, models="dromics")
    rows = np.arange(6)
    assert np.isfinite(fits.turn_dose[rows, fits.best]).all()
    driven = fits.turn_driven_by_dose
    assert np.isin(driven, [0.0, 30.0]).all(), driven
    assert np.count_nonzero(driven == 0.0) >= 4, driven


def test_the_data_without_a_group_are_those_of_its_samples() -> None:
    rng = np.random.default_rng(5)
    doses = np.repeat(HALF_LOG, [3, 2, 3, 3, 1, 3, 3, 2, 3])
    values = 6.0 + rng.normal(0, 0.2, (doses.size, 4))
    full = _constant_variance_data(doses, values)
    genes = np.array([0, 2, 3])
    for k, dose in enumerate(full.doses):
        keep = doses != dose
        reduced = full.without_group(k, genes)
        expected = _constant_variance_data(doses[keep], values[keep][:, genes])
        assert_array_equal(reduced.doses, expected.doses)
        assert_array_equal(reduced.n_samples, expected.n_samples)
        assert_allclose(reduced.means, expected.means, rtol=1e-14)
        assert reduced.within is not None and expected.within is not None
        assert_allclose(reduced.within, expected.within, rtol=1e-12, atol=1e-15)
        assert_allclose(reduced.covariance, expected.covariance, rtol=1e-14)


@pytest.mark.parametrize("seed", [7, 11])
def test_turn_drivers_follow_the_rule_with_refits_of_the_samples(seed: int) -> None:
    """The documented rule computed with the public function on the samples without
    each dose group: the lowest dose whose removal leaves a best fit that does not
    turn within the doses kept."""
    rng = np.random.default_rng(seed)
    doses = np.repeat(HALF_LOG, 3)
    log_dose = np.log10(np.where(doses > 0, doses, 1e-3))
    columns = [
        6.0 + 1.2 * np.exp(-0.5 * ((log_dose - centre) / 0.5) ** 2) * (doses > 0)
        for centre in (-1.5, -0.5, 0.5, 1.2)
    ]
    columns += [np.where(doses == dose, 7.0, 6.0) for dose in (0.01, 1.0, 30.0)]
    columns.append(np.where(doses >= 10, 5.0, np.where(doses > 0, 7.0, 6.0)))
    values = np.column_stack(columns) + rng.normal(0, 0.15, (doses.size, len(columns)))
    fits = fit_dose_curves(doses, values, models="dromics")
    expected = np.full(values.shape[1], np.nan)
    for g in range(values.shape[1]):
        if fits.best[g] < 0 or not np.isfinite(fits.turn_dose[g, fits.best[g]]):
            continue
        for dose in fits.doses:
            keep = doses != dose
            refit = fit_dose_curves(doses[keep], values[keep][:, [g]], models="dromics")
            b = refit.best[0]
            if b < 0 or not np.isfinite(refit.turn_dose[0, b]):
                expected[g] = dose
                break
    turning = np.isfinite(fits.turn_dose[np.arange(values.shape[1]), fits.best])
    assert np.count_nonzero(turning) >= 4
    assert_array_equal(fits.turn_driven_by_dose, expected)


def test_turns_at_five_dose_groups_are_not_checked() -> None:
    """At 5 dose groups, no model that can turn has a spare group among the 4 left
    without one, so no refit could turn: no dose group is named, and the printout
    says that the turns were not checked."""
    levels = np.array([0.0, 0.1, 0.3, 1.0, 3.0])
    doses = np.repeat(levels, 4)
    log_dose = np.log10(np.where(doses > 0, doses, 1e-3))
    rng = np.random.default_rng(2)
    # Bells over three groups, which no single group makes.
    values = np.column_stack(
        [
            6.0 + 2.0 * np.exp(-0.5 * ((log_dose - centre) / 0.35) ** 2) * (doses > 0)
            for centre in (-0.8, -0.5, -0.2)
        ]
    ) + rng.normal(0, 0.1, (doses.size, 3))
    fits = fit_dose_curves(doses, values, models="dromics")
    rows = np.arange(3)
    assert np.isfinite(fits.turn_dose[rows, fits.best]).all()
    assert np.isnan(fits.turn_driven_by_dose).all()
    assert "3 best fits turn, not checked for a dose group that drives them" in repr(
        fits
    )
    # At 6 dose groups the symmetric bells, of 4 parameters, can be refitted.
    six = np.repeat(np.r_[levels, 10.0], 4)
    log_six = np.log10(np.where(six > 0, six, 1e-3))
    bells = 6.0 + 2.0 * np.exp(-0.5 * ((log_six + 0.5) / 0.35) ** 2) * (six > 0)
    fits = fit_dose_curves(
        six, bells[:, None] + rng.normal(0, 0.1, (six.size, 1)), models="dromics"
    )
    assert "driven by one dose group" in repr(fits)


def test_doses_without_a_vehicle_or_over_nine_decades() -> None:
    truth = [1.2, 9.0, 6.0, 0.5]
    doses = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0])
    curve = documented("ll4", truth, doses)
    fits = fit_dose_curves(*replicated(doses, curve), models="ll4")
    assert_allclose(parameters_of(fits, "ll4")[0], truth, rtol=1e-5)
    assert fits.doses[0] == 0.01
    # Doses from 1e-6 to 1e3: the fits do not depend on the unit.
    wide = np.r_[0.0, np.geomspace(1e-6, 1e3, 9)]
    rng = np.random.default_rng(31)
    values = 6.0 + 2.0 / (1 + (1e-2 / np.where(wide > 0, wide, 1e-300)) ** 0.8)
    values = np.repeat(values, 3)[:, None] + rng.normal(0, 0.1, (wide.size * 3, 3))
    base = fit_dose_curves(np.repeat(wide, 3), values, models="epa")
    scaled = fit_dose_curves(np.repeat(wide, 3) * 1e3, values, models="epa")
    assert np.isfinite(base.log_likelihood).all()
    assert_allclose(scaled.log_likelihood, base.log_likelihood, rtol=1e-8)


def test_the_refinement_stops_at_its_limit() -> None:
    """A refinement that reaches its limit of iterations reports that it did not
    converge, and keeps the best point found."""
    from cytepool.stats import _dose_curve_search as search

    scale = models.DoseScale.of(HALF_LOG)
    kind = models.HILL
    lower, upper = models.bounds(kind, scale)
    means = documented("hill", [6.0, 2.0, 0.8, 2.2], HALF_LOG)
    root = np.diag(np.sqrt(np.full(HALF_LOG.size, 3.0)))
    z = root @ means
    start = np.array([6.0, 1.0, 10.0, 1.0])
    theta = start.copy()
    value, converged = search.refine(
        kind, models.FREE, 2, theta, lower, upper, HALF_LOG, means, z, root, 0.1, 1
    )
    assert not converged
    first = float(np.sum(3.0 * (means - documented("hill", start, HALF_LOG)) ** 2))
    assert value < first
    theta = start.copy()
    value, converged = search.refine(
        kind, models.FREE, 2, theta, lower, upper, HALF_LOG, means, z, root, 0.1, 500
    )
    # Converged: what a Gauss-Newton step could gain is below 1e-12 of Q plus 0.1.
    assert converged and value < 1e-12
    # A start whose means overflow is no start: the refinement stops at once.
    kind = models.EXP2
    theta = np.array([1.0, 1e3])
    value, converged = search.refine(
        kind,
        models.NON_NEGATIVE,
        1,
        theta,
        np.array([0.0, -np.inf]),
        np.array([np.inf, np.inf]),
        HALF_LOG,
        means,
        z,
        root,
        0.1,
        500,
    )
    assert value == np.inf and not converged


def test_degenerate_inputs() -> None:
    doses = np.array([0.0, 0.0, 0.1, 1.0, 10.0, 30.0])
    # No replicate at the treated doses: the likelihood rests on Q, the lack of fit
    # has no within-group variance only when the vehicle's samples agree.
    values = np.array(
        [[5.0, 5.2, 5.5, 6.3, 7.0, 7.1], [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]]
    ).T
    fits = fit_dose_curves(doses, values, models="epa")
    assert np.isfinite(fits.log_likelihood[0, fits.models.index("linear")])
    assert (fits.not_fitted_reason[1] == "the values do not vary").all()
    # Values on a line exactly, without replicates: the within-group sum of squares
    # 0 and Q at rounding, a likelihood without bound that the line takes.
    exact = np.column_stack([3.0 + 0.2 * np.unique(doses)])
    fits = fit_dose_curves(np.unique(doses), exact, models="epa")
    m = fits.models.index("linear")
    assert fits.log_likelihood[0, m] > 100.0
    assert fits.best[0] == m
    assert fits.akaike_weight[0, m] == pytest.approx(1.0)
    assert np.isnan(fits.lack_of_fit_p_value[0, m])


def test_results_do_not_depend_on_order_units_or_threads() -> None:
    rng = np.random.default_rng(17)
    doses = np.repeat(HALF_LOG, 3)
    values = 6.0 + 1.5 / (1 + (0.5 / np.where(doses > 0, doses, 1e-12)) ** 1.8)
    values = values[:, None] + rng.normal(0, 0.15, (doses.size, 5))
    base = fit_dose_curves(doses, values, models="epa", n_threads=1)
    four = fit_dose_curves(doses, values, models="epa", n_threads=4)
    assert_array_equal(base.parameters, four.parameters)
    assert_array_equal(base.log_likelihood, four.log_likelihood)
    order = rng.permutation(doses.size)
    shuffled = fit_dose_curves(doses[order], values[order], models="epa")
    assert_allclose(shuffled.log_likelihood, base.log_likelihood, rtol=1e-11)
    genes = rng.permutation(5)
    swapped = fit_dose_curves(doses, values[:, genes], models="epa")
    assert_allclose(swapped.log_likelihood, base.log_likelihood[genes], rtol=1e-11)
    # Doses in other units: the same likelihoods, locations scaled and rates divided.
    scaled = fit_dose_curves(doses * 1000, values, models="epa")
    assert_allclose(scaled.log_likelihood, base.log_likelihood, rtol=1e-9)
    assert_allclose(
        parameters_of(scaled, "hill")[:, 2],
        parameters_of(base, "hill")[:, 2] * 1000,
        rtol=1e-5,
    )
    assert_allclose(
        parameters_of(scaled, "exponential_2")[:, 1],
        parameters_of(base, "exponential_2")[:, 1] / 1000,
        rtol=1e-5,
    )
    # Values moved up: the same likelihoods for the models whose level is a parameter.
    moved = fit_dose_curves(doses, values + 500.0, models="epa")
    for model in ("linear", "polynomial_2", "power", "hill"):
        m = base.models.index(model)
        assert_allclose(
            moved.log_likelihood[:, m], base.log_likelihood[:, m], rtol=1e-8
        )


def test_arguments_are_refused_with_a_fix() -> None:
    doses = np.repeat([0.0, 0.1, 1.0, 10.0, 30.0], 2)
    values = np.ones((doses.size, 2)) + np.arange(doses.size)[:, None] * 0.1
    with pytest.raises(ValueError, match="one dose each"):
        fit_dose_curves(doses[:-1], values)
    with pytest.raises(ValueError, match="2-D"):
        fit_dose_curves(doses, values[:, 0])
    bad = values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fit_dose_curves(doses, bad)
    with pytest.raises(ValueError, match="at least 0"):
        fit_dose_curves(-doses, values)
    with pytest.raises(ValueError, match="two dose groups"):
        fit_dose_curves(np.zeros(doses.size), values)
    with pytest.raises(ValueError, match="models names a set of models, 'epa' "):
        fit_dose_curves(doses, values, models="bmds")
    with pytest.raises(ValueError, match=r"did you mean 'epa'\?"):
        fit_dose_curves(doses, values, models="EPA")
    # A model's name, or a list of names, is pointed to the sets that hold it.
    with pytest.raises(ValueError, match="'hill' is in 'epa': pass the set"):
        fit_dose_curves(doses, values, models="hill")
    with pytest.raises(
        TypeError, match="'hill' is in 'epa', 'linear' is in 'epa' and 'dromics'"
    ):
        fit_dose_curves(doses, values, models=["hill", "linear"])  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="models names a set"):
        fit_dose_curves(doses, values, models=None)  # type: ignore[arg-type]
    # Values whose squares the fits' sums of squares cannot hold.
    with pytest.raises(ValueError, match="at most 1e\\+100 in size"):
        fit_dose_curves(doses, values * 1e160)
    with pytest.raises(TypeError, match="masked"):
        fit_dose_curves(doses, np.ma.masked_array(values))
    with pytest.raises(TypeError, match="real numbers"):
        fit_dose_curves(doses, values.astype(str))
    with pytest.raises(ValueError, match="1e-8"):
        fit_dose_curves(np.r_[doses[:-1], 30.0 * (1 + 1e-12)], values)
    with pytest.raises(ValueError, match="n_threads"):
        fit_dose_curves(doses, values, n_threads=0)


def test_the_table_reads_back() -> None:
    rng = np.random.default_rng(19)
    doses = np.repeat(HALF_LOG, 3)
    values = 4.0 + rng.normal(0, 0.2, (doses.size, 3))
    for model_set in SETS:
        fits = fit_dose_curves(doses, values, models=model_set)
        table = fits.table(feature_ids=["a", "b", "c"])
        assert isinstance(table, Table)
        assert table.columns == (
            "feature_id",
            "model",
            *models.SET_PARAMETERS[model_set],
            "n_parameters",
            "spare_groups",
            "log_likelihood",
            "aic",
            "aicc",
            "akaike_weight",
            "best",
            "lack_of_fit_p_value",
            "max_abs_scaled_residual",
            "parameters_at_bounds",
            "turn_dose",
            "turn_driven_by_dose",
            "converged",
            "not_fitted_reason",
        )
        n_models = len(fits.models)
        assert table["feature_id"].tolist() == [
            g for g in "abc" for _ in range(n_models)
        ]
        assert table["model"].tolist() == list(fits.models) * 3
        assert_array_equal(table["log_likelihood"], fits.log_likelihood.ravel())
        assert int(table["best"].sum()) == 3
        assert table["spare_groups"].tolist()[:n_models] == [
            9 - len(m.parameters) for m in models.SETS[model_set]
        ]
        for name in models.SET_PARAMETERS[model_set]:
            used = [name in m.parameters for m in models.SETS[model_set]]
            column = np.asarray(table[name])[:n_models]
            assert np.isnan(column[~np.asarray(used)]).all()
        assert fits.table()["feature_id"][0] == "feature_0"
        assert repr(fits).startswith(
            f"DoseCurveFits ({model_set}, constant variance): 3 genes"
        )
        assert not fits.parameters.flags.writeable
        with pytest.raises(AttributeError):
            fits.model_set = "ll4"  # type: ignore[misc]


def test_the_table_puts_the_turn_driver_on_the_best_row() -> None:
    rng = np.random.default_rng(13)
    doses = np.repeat(HALF_LOG, 3)
    values = 6.0 + rng.normal(0, 0.15, (doses.size, 6))
    values[doses == 0.3] += 1.5
    fits = fit_dose_curves(doses, values, models="dromics")
    assert np.isfinite(fits.turn_driven_by_dose).any()
    table = fits.table()
    n_models = len(fits.models)
    column = np.asarray(table["turn_driven_by_dose"]).reshape(-1, n_models)
    best = np.asarray(table["best"]).reshape(-1, n_models)
    assert_array_equal(column[best], fits.turn_driven_by_dose)
    assert np.isnan(column[~best]).all()


def test_the_printout_names_models_not_fitted() -> None:
    # Four dose groups leave Hill and exponential 5, of 4 parameters, no spare group.
    doses = np.repeat([0.0, 0.1, 1.0, 10.0], 3)
    values = 5.0 + np.random.default_rng(23).normal(0, 0.2, (doses.size, 2))
    fits = fit_dose_curves(doses, values, models="epa")
    assert set(fits.models_not_fitted) == {"hill", "exponential_5"}
    assert "  not fitted (no spare dose group): hill, exponential_5" in repr(fits)


# ---------------------------------------------------------------------------------------
# Against the references
# ---------------------------------------------------------------------------------------


LIKELIHOOD = 1e-8  # the Validation page's scale-free gap between two maxima
PARAMETERS = 10.0  # standard errors per sqrt(2 max(dLL, 1e-12)), the page's bound
DROMICS_FAMILIES = {
    "L": ("linear",),
    "E": ("exponential",),
    "H": ("ll4",),
    "GP": ("gauss_probit", "gauss_probit_monotone", "gauss_probit_symmetric"),
    "lGP": (
        "log_gauss_probit",
        "log_gauss_probit_monotone",
        "log_gauss_probit_symmetric",
    ),
}
DROMICS_NAMES = {
    "linear": "L",
    "exponential": "E",
    "Hill": "H",
    "Gauss-probit": "GP",
    "log-Gauss-probit": "lGP",
}


@functools.cache
def references() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / "references.json").read_text())
    return found


@functools.cache
def constant_fits(series: str, model_set: str) -> DoseCurveFits:
    found = references()["constant"][series]
    return fit_dose_curves(
        np.asarray(found["doses"], dtype=float),
        np.asarray(found["values"], dtype=float),
        models=model_set,
    )


def residual_variance_gap(
    fits: DoseCurveFits, g: int, m: int, reference_ll: float
) -> float:
    """How far CytePool's maximum lies below a reference's, as the relative
    residual variance; negative where it is the larger."""
    n = float(np.sum(fits.n_samples))
    return math.expm1(2 * (reference_ll - fits.log_likelihood[g, m]) / n)


def samples_gap(
    series: str, fits: DoseCurveFits, g: int, model: str, q_ref: float
) -> float:
    """(Q - Q_ref) / (S_w + Q_ref) of a fit's reported parameters, with Q = sum_k n_k
    (ybar_k - f(x_k))^2 by the documented formula and the within-group sum of squares
    S_w, both from the series' samples."""
    found = references()["constant"][series]
    doses = np.asarray(found["doses"], dtype=float)
    values = np.asarray(found["values"], dtype=float)[:, g]
    levels = np.unique(doses)
    theta = parameters_of(fits, model)[g]
    f = documented(formula_name(fits.model_set, model), theta, levels)
    q = sum(
        np.sum(doses == d) * (values[doses == d].mean() - f[k]) ** 2
        for k, d in enumerate(levels)
    )
    within = sum(
        np.sum((values[doses == d] - values[doses == d].mean()) ** 2) for d in levels
    )
    return float((q - q_ref) / (within + q_ref))


def identified(
    model: str, theta: np.ndarray, levels: np.ndarray, sizes: np.ndarray
) -> bool:
    """Whether the data identify a fit's parameters, the Validation page's rule: the
    Jacobian of the documented mean at the dose groups, weighted by their sizes, with
    columns scaled to unit length, has a condition number of at most 1e6."""
    jac = np.empty((levels.size, theta.size))
    for j in range(theta.size):
        h = 1e-6 * max(abs(theta[j]), 1e-3)
        up, down = theta.copy(), theta.copy()
        up[j] += h
        down[j] -= h
        change = documented(model, up, levels) - documented(model, down, levels)
        jac[:, j] = np.sqrt(sizes) * change / (2 * h)
    norms = np.linalg.norm(jac, axis=0)
    if not np.all(np.isfinite(jac)) or np.any(norms == 0):
        return False
    singular = np.linalg.svd(jac / norms, compute_uv=False)
    return bool(singular[-1] > 0 and singular[0] / singular[-1] <= 1e6)


def within_documented_bounds(model: str, theta: np.ndarray, doses: np.ndarray) -> bool:
    """Whether reported parameters lie within the chapter's bounds of the model."""
    levels = np.unique(doses)
    top, lowest = levels[-1], levels[levels > 0][0]
    rate = 100.0 / top
    bounds: dict[str, list[tuple[float, float]]] = {
        "linear": [(-np.inf, np.inf)] * 2,
        "polynomial_2": [(-np.inf, np.inf)] * 3,
        "power": [(-np.inf, np.inf), (-np.inf, np.inf), (1, 18)],
        "hill": [(-np.inf, np.inf), (-np.inf, np.inf), (0, 5 * top), (1, 18)],
        "exponential_2": [(0, np.inf), (-rate, rate)],
        "exponential_3": [(0, np.inf), (-rate, rate), (1, 18)],
        "exponential_4": [(0, np.inf), (0, rate), (0, 1000)],
        "exponential_5": [(0, np.inf), (0, rate), (0, 1000), (1, 18)],
        "ll4": [
            (0.1, 18),
            (-np.inf, np.inf),
            (-np.inf, np.inf),
            (lowest / 100, 5 * top),
        ],
    }
    if model == "polynomial_2" and theta[1] * theta[2] < 0:
        return False
    if model == "exponential":
        return bool(top / 100 <= abs(theta[2]) <= 1e4 * top)
    if "gauss_probit" in model:
        log = model.startswith("log")
        positive = np.log(levels[levels > 0]) if log else levels
        gap = float(np.max(np.diff(positive)))
        width_top = 10 * (positive[-1] - positive[0]) if log else 10 * top
        peak = (lowest, top) if log else (0.0, top)
        e = theta[2] if model.endswith("symmetric") else theta[3]
        return bool(gap / 3 <= theta[0] <= width_top and peak[0] <= e <= peak[1])
    return all(a <= t <= b for t, (a, b) in zip(theta, bounds[model], strict=True))


def test_constant_maxima_reach_scipys_best_of_200_starts() -> None:
    checked = 0
    for series, found in references()["constant"].items():
        for model_set, by_model in found["scipy"].items():
            fits = constant_fits(series, model_set)
            for model, by_gene in by_model.items():
                for g, q_ref in by_gene.items():
                    gap = samples_gap(series, fits, int(g), model, q_ref)
                    assert gap <= LIKELIHOOD, (series, model, g, gap)
                    checked += 1
    assert checked >= 400


@pytest.mark.parametrize("variance", ["weighted", "moderated"])
def test_plug_in_maxima_reach_scipys_best_of_200_starts(variance: str) -> None:
    found = references()[variance]
    means = np.asarray(found["means"], dtype=float)
    covariance = np.asarray(found["covariance"], dtype=float)
    scale = np.asarray(found["covariance_scale"], dtype=float)
    doses = np.asarray(found["doses"], dtype=float)
    data = _CurveData(
        variance,
        doses,
        np.asarray(found["n_samples"], dtype=float),
        means,
        covariance,
        scale,
        np.asarray(found["df"], dtype=float),
        None,
    )
    checked = 0
    for model_set, by_model in found["scipy"].items():
        fits = _fit_means(data, model_set)
        for model, by_gene in by_model.items():
            for g, q_ref in ((int(g), q) for g, q in by_gene.items()):
                theta = parameters_of(fits, model)[g]
                root = np.linalg.inv(np.linalg.cholesky(covariance[g]))
                r = root @ (
                    means[g] - documented(formula_name(model_set, model), theta, doses)
                )
                gap = (float(r @ r) - q_ref) / (doses.size * scale[g])
                assert gap <= LIKELIHOOD, (model_set, model, g, gap)
                checked += 1
    assert checked >= 50


def test_epa_fits_reach_pybmds_maxima_with_its_statistics() -> None:
    """pybmds 25.2, each model run increasing and decreasing and the larger maximum
    kept: where it reaches higher, its estimate lies beyond CytePool's bounds; where
    both reach the maximum, the AIC's penalty, test 4, the scaled residuals and the
    parameters agree. Test 4 is the likelihood ratio n log(1 + (K - p) F / (n - K))
    of CytePool's F on a chi-square of K - p degrees of freedom."""
    beyond, matched, compared = 0, 0, 0
    for series, found in references()["constant"].items():
        fits = constant_fits(series, "epa")
        doses = np.asarray(found["doses"], dtype=float)
        n_groups = np.unique(doses).size
        n = doses.size
        for model, by_gene in found["pybmds"].items():
            m = fits.models.index(model)
            spec = models.model_of("epa", model)
            for g, runs in ((int(g), r) for g, r in by_gene.items()):
                usable = {
                    d: r
                    for d, r in runs.items()
                    if r["log_likelihood"] is not None and r["log_likelihood"] > -9e3
                }
                if not usable or fits.not_fitted_reason[g, m]:
                    continue
                direction = max(usable, key=lambda d: usable[d]["log_likelihood"])
                best = usable[direction]
                named = dict(zip(best["names"], best["values"], strict=True))
                # pybmds's exponential 3 has b >= 0 and its direction as a setting.
                sign = 1.0 if direction == "True" else -1.0
                theta = np.array(
                    [
                        named[p]
                        * (sign if (p == "b" and model == "exponential_3") else 1.0)
                        for p in spec.parameters
                    ]
                )
                gap = residual_variance_gap(fits, g, m, best["log_likelihood"])
                if gap > LIKELIHOOD:
                    assert not within_documented_bounds(model, theta, doses), (model, g)
                    beyond += 1
                    continue
                if gap < -LIKELIHOOD:
                    continue
                matched += 1
                ll = fits.log_likelihood[g, m]
                assert best["aic"] + 2 * best["log_likelihood"] == pytest.approx(
                    fits.aic[g, m] + 2 * ll, rel=1e-12
                )
                spare = n_groups - len(spec.parameters)
                assert best["test4_df"] == spare
                f = stats.f.isf(fits.lack_of_fit_p_value[g, m], spare, n - n_groups)
                p_ours = stats.chi2.sf(
                    n * math.log1p(spare * f / (n - n_groups)), spare
                )
                p_ref = best["test4_p"]
                assert abs(p_ours - p_ref) <= 1e-6 * p_ref * max(
                    1.0, abs(math.log(max(p_ref, 1e-300)))
                )
                # Two maxima within 1e-8 of the residual variance put each scaled
                # residual within 2 sqrt(n 1e-8) of the other's.
                residual = np.max(np.abs(best["scaled_residuals"]))
                assert abs(
                    fits.max_abs_scaled_residual[g, m] - residual
                ) <= 2 * math.sqrt(n * LIKELIHOOD)
                errors = np.array(
                    [
                        dict(zip(best["names"], best["se"])).get(p, np.nan)
                        for p in spec.parameters
                    ],
                    dtype=float,
                )
                ours = parameters_of(fits, model)[g]
                if (
                    fits.parameters_at_bounds[g, m]
                    or any(best["bounded"])
                    or not (np.isfinite(errors).all() and (errors > 0).all())
                    or not identified(
                        formula_name("epa", model), ours, fits.doses, fits.n_samples
                    )
                ):
                    continue
                spread = errors * math.sqrt(
                    2 * max(abs(ll - best["log_likelihood"]), 1e-12)
                )
                assert np.all(np.abs(ours - theta) <= PARAMETERS * spread), (model, g)
                compared += 1
    assert matched >= 100
    assert compared >= 20
    assert beyond <= 5


def test_ll4_fits_reach_drc_maxima_with_its_parameters() -> None:
    """drc 4.0-0's LL.4, whose (b, c, d, e) with b < 0 are CytePool's (-b, d, c,
    e)."""
    checked, compared = 0, 0
    for series, found in references()["constant"].items():
        doses = np.asarray(found["doses"], dtype=float)
        if np.unique(doses).size < 5:
            continue
        fits = constant_fits(series, "ll4")
        for g, run in enumerate(found["drc"]):
            if run["failed"]:
                continue
            b, c, d, e = run["coef"]
            theta = np.array([b, c, d, e] if b > 0 else [-b, d, c, e])
            gap = residual_variance_gap(fits, g, 0, run["loglik"])
            if gap > LIKELIHOOD:
                assert not within_documented_bounds("ll4", theta, doses), g
                continue
            checked += 1
            ours = parameters_of(fits, "ll4")[g]
            if (
                gap < -LIKELIHOOD
                or fits.parameters_at_bounds[g, 0]
                or not identified("ll4", ours, fits.doses, fits.n_samples)
            ):
                continue
            se = np.asarray(run["se"], dtype=float)
            errors = se if b > 0 else se[[0, 2, 1, 3]]
            dll = abs(fits.log_likelihood[g, 0] - run["loglik"])
            spread = errors * math.sqrt(2 * max(dll, 1e-12))
            assert np.all(np.abs(ours - theta) <= PARAMETERS * spread), g
            compared += 1
    assert checked >= 15
    assert compared >= 1


def test_dromics_criteria_and_turns() -> None:
    """DRomics 2.6.3's drcfit: CytePool's lowest AICc of each family is not above
    DRomics', whose criteria are printed to 2 decimals, unless DRomics' estimate lies
    beyond CytePool's bounds; and at DRomics' estimates of a bell or U, CytePool's
    turn is DRomics' xextrem."""
    compared, turns = 0, 0
    for series, found in references()["constant"].items():
        doses = np.asarray(found["doses"], dtype=float)
        if np.unique(doses).size < 5:
            continue
        fits = constant_fits(series, "dromics")
        n = float(doses.size)
        chosen = {row["id"]: row for row in found["dromics_fitres"]}
        for criteria in found["dromics_criteria"]:
            g = int(criteria["id"][1:])
            row = chosen.get(criteria["id"])
            for family, members in DROMICS_FAMILIES.items():
                ref = criteria[f"InfoCrit.{family}"]
                # The family's lowest AICc among its models fitted at these doses.
                ours = min(
                    (
                        fits.aicc[g, fits.models.index(m)]
                        for m in members
                        if m in fits.models
                    ),
                    default=np.inf,
                )
                if ref is None or not np.isfinite(ours):
                    continue
                if (ours - ref - 0.005) / n > LIKELIHOOD:
                    # Higher only where DRomics' estimate is beyond CytePool's bounds or
                    # not reported.
                    assert (
                        row is None
                        or DROMICS_NAMES[row["model"]] != family
                        or not (
                            within_documented_bounds(
                                dromics_form(row),
                                np.array(
                                    [
                                        row[p]
                                        for p in dromics_letters(dromics_form(row))
                                    ],
                                    dtype=float,
                                ),
                                doses,
                            )
                        )
                    ), (series, g, family)
                    assert family != "L"
                    continue
                compared += 1
            if row is None or row.get("trend") not in ("bell", "U"):
                continue
            # The turn at DRomics' own estimates, by CytePool's closed form.
            spec = models.model_of("dromics", dromics_form(row))
            theta = np.array([row[p] for p in spec.parameters], dtype=float)
            x = models.turn_doses(
                spec.kind, models.internal(spec.kind, theta)[None], float(doses.max())
            )[0]
            assert x == pytest.approx(row["xextrem"], rel=1e-3)
            turns += 1
    assert compared >= 30
    assert turns >= 1


def dromics_form(row: dict[str, Any]) -> str:
    """CytePool's name of the model DRomics chose for an item."""
    family = DROMICS_NAMES[row["model"]]
    if family in ("L", "E", "H"):
        return DROMICS_FAMILIES[family][0]
    prefix = "log_gauss_probit" if family == "lGP" else "gauss_probit"
    if row["nbpar"] == 5:
        return prefix
    return prefix + ("_monotone" if row["f"] == 0 else "_symmetric")


def dromics_letters(model: str) -> tuple[str, ...]:
    return models.model_of("dromics", model).parameters


def test_turns_rest_on_the_groups_the_documented_rule_finds() -> None:
    """The dose group that drives each turn against validation/dose_curves.py's own
    coding of the rule with scipy's fits of the whole set on K - 1 groups: on the
    series under constant and weighted variance, and on genes that turn, or do not,
    by construction at 5, 6 and 9 dose groups, where no group is named at 5."""
    found = references()
    constructed = found["turn_cases"]
    checked, unchecked = 0, 0
    for verdict in found["turns"]:
        label, g = verdict["case"], verdict["gene"]
        series, *rest = label.split()
        expected = np.nan if verdict["reference"] is None else verdict["reference"]
        if label in constructed:
            fits = constructed_fits(label)
        elif rest == ["dromics"] and series in found["constant"]:
            fits = constant_fits(series, "dromics")
        elif len(rest) == 3 and series == "ladder" and rest[0] == "abundant":
            plug_in = found[rest[1]]
            if g not in plug_in["genes"]:
                continue
            position = plug_in["genes"].index(g)
            fits = _fit_means(plug_in_data(plug_in, [position]), "dromics")
            g = 0
        else:
            continue
        assert_array_equal(fits.turn_driven_by_dose[g], expected)
        checked += 1
        unchecked += fits.doses.size == 5
    assert checked >= 40
    assert unchecked >= 10


@functools.cache
def constructed_fits(label: str) -> DoseCurveFits:
    """The "dromics" fits of a case of genes that turn, or not, by construction."""
    case = references()["turn_cases"][label]
    if "values" in case:
        return fit_dose_curves(
            np.asarray(case["doses"], dtype=float),
            np.asarray(case["values"], dtype=float),
            models="dromics",
        )
    return _fit_means(plug_in_data(case), "dromics")


def plug_in_data(found: dict[str, Any], rows: list[int] | None = None) -> _CurveData:
    """The curve data of a weighted or moderated fixture, of some of its genes."""
    pick = slice(None) if rows is None else rows
    return _CurveData(
        found.get("variance", "weighted"),
        np.asarray(found["doses"], dtype=float),
        np.asarray(found["n_samples"], dtype=float),
        np.asarray(found["means"], dtype=float)[pick],
        np.asarray(found["covariance"], dtype=float)[pick],
        np.asarray(found["covariance_scale"], dtype=float)[pick],
        np.asarray(found["df"], dtype=float)[pick],
        None,
    )
