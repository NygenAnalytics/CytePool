"""Tests for the EC50s, Hill slopes and plateaus of dose curves, ec50s and EC50s.

Reference intervals in ``tests/data/ec50/profiles.json`` come from mpmath 1.3.0 and
drc 4.0-0's coefficients in ``drc.json``, written by ``validation/ec50.py``, whose
reference process computes the documented profile at 30 digits without CytePool. The
other tests check what the references cannot: curves written here from the documented
formulas, GraphPad's equation, a dense profile computed here with its own variance and
F quantile, bend points by root search on the curve, and invariants of units, order,
threads and values.
"""

import dataclasses
import doctest
import json
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import optimize, stats

from cytepool.stats.dose_curves import (
    DoseCurveFits,
    _CurveData,
    _fit_means,
    fit_dose_curves,
)
from cytepool.stats import potency
from cytepool.stats.potency import EC50s, ec50s
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "ec50"
HALF_LOG = np.array([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
BEND = 4.6805  # Sebaugh and McCray (2003)


def ll4(x: Any, b: float, c: float, d: float, e: float) -> np.ndarray:
    """c + (d - c) / (1 + (x / e)^b), d at dose 0, as the chapter writes it."""
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore"):
        return c + (d - c) / (1.0 + np.where(x > 0, (x / e) ** b, 0.0))


def hill(x: Any, g: float, v: float, k: float, n: float) -> np.ndarray:
    """g + v x^n / (k^n + x^n), g at dose 0 and g + v above it when k is 0."""
    x = np.asarray(x, dtype=float)
    if k == 0.0:
        return g + v * (x > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return g + v * np.where(
            x > 0, 1.0 / (1.0 + (k / np.where(x > 0, x, 1.0)) ** n), 0.0
        )


def graphpad(
    x: Any, bottom: float, top: float, ec50: float, slope: float
) -> np.ndarray:
    """GraphPad Prism's variable-slope equation, X = log10 dose:
    Y = Bottom + (Top - Bottom) / (1 + 10^((LogEC50 - X) HillSlope))."""
    x = np.asarray(x, dtype=float)
    return bottom + (top - bottom) / (
        1.0 + 10.0 ** ((math.log10(ec50) - np.log10(x)) * slope)
    )


def samples_on(curve: np.ndarray, spread: float = 0.2) -> np.ndarray:
    """Three samples per dose group whose mean is the curve: -spread, 0, +spread."""
    return (curve[:, None] + np.array([-spread, 0.0, spread])[None, :]).ravel()


def column(found: EC50s, name: str) -> np.ndarray:
    return np.asarray(getattr(found, name))


# ---------------------------------------------------------------------------------------
# The EC50, Hill slope and plateaus
# ---------------------------------------------------------------------------------------

KNOWN = [
    # (set, model, parameters, (ec50, hill slope, bottom, top)) on HALF_LOG
    ("ll4", "ll4", (1.5, 8.0, 6.0, 0.3), (0.3, 1.5, 6.0, 8.0)),  # rise: c > d
    ("ll4", "ll4", (2.0, 4.0, 6.5, 1.0), (1.0, -2.0, 4.0, 6.5)),  # fall
    ("epa", "hill", (6.0, 1.5, 0.3, 2.0), (0.3, 2.0, 6.0, 7.5)),  # rise: v > 0
    ("epa", "hill", (7.0, -2.0, 1.0, 1.5), (1.0, -1.5, 5.0, 7.0)),  # fall
]


@pytest.mark.parametrize(("model_set", "model", "parameters", "expected"), KNOWN)
def test_ec50s_of_known_curves(
    model_set: str, model: str, parameters: tuple, expected: tuple
) -> None:
    curve = (ll4 if model == "ll4" else hill)(HALF_LOG, *parameters)
    x = np.repeat(HALF_LOG, 3)
    fits = fit_dose_curves(x, samples_on(curve)[:, None], models=model_set)
    found = ec50s(fits)
    assert found.model == model
    got = [
        found.ec50[0],
        found.hill_slope[0],
        found.bottom_plateau[0],
        found.top_plateau[0],
    ]
    assert_allclose(got, expected, rtol=1e-5, atol=1e-6)
    assert found.status[0] == "within the doses"


@pytest.mark.parametrize("model_set", ["ll4", "epa", "dromics"])
def test_graphpad_equation_gives_the_curve(model_set: str) -> None:
    rng = np.random.default_rng(5)
    x = np.repeat(HALF_LOG, 3)
    truths = [(1.5, 8.0, 6.0, 0.3), (2.0, 4.0, 6.5, 1.0), (0.7, 7.0, 6.0, 3.0)]
    values = np.column_stack(
        [ll4(x, *t) + rng.normal(0.0, 0.2, x.size) for t in truths]
    )
    fits = fit_dose_curves(x, values, models=model_set)
    found = ec50s(fits)
    m = fits.models.index(found.model)
    letters = "bcde" if found.model == "ll4" else "gvkn"
    positive = HALF_LOG[1:]
    for g in range(len(truths)):
        p = [fits.parameters[g, m, fits.parameter_names.index(c)] for c in letters]
        curve = (ll4 if found.model == "ll4" else hill)(positive, *p)
        equation = graphpad(
            positive,
            found.bottom_plateau[g],
            found.top_plateau[g],
            found.ec50[g],
            found.hill_slope[g],
        )
        assert_allclose(equation, curve, rtol=1e-12, atol=1e-12)
        # GraphPad's slope is positive for a rise.
        assert np.sign(found.hill_slope[g]) == np.sign(curve[-1] - curve[0])


def test_drc_slope_sign_convention_by_hand() -> None:
    """The chapter's table of conventions: drc's LL.4 with b < 0 for a rise is
    CytePool's curve with -b, so GraphPad's slope is -b sign(d - c) of drc's (b, c, d,
    e) for either sign of b."""
    x = HALF_LOG[1:]
    for b, c, d, e in (
        (-1.5, 6.0, 8.0, 0.3),
        (1.5, 8.0, 6.0, 0.3),
        (2.0, 6.5, 4.0, 1.0),
    ):
        with np.errstate(divide="ignore"):
            drc = c + (d - c) / (1.0 + np.exp(b * (np.log(x) - math.log(e))))
        slope = -b * math.copysign(1.0, d - c)
        assert_allclose(graphpad(x, min(c, d), max(c, d), e, slope), drc, rtol=1e-12)


# ---------------------------------------------------------------------------------------
# The interval, against a dense profile computed here
# ---------------------------------------------------------------------------------------


def scan_of(model: str, doses: np.ndarray) -> np.ndarray:
    """The documented scan: 48 EC50s evenly spaced on log scale from a hundredth of the
    lowest positive dose to 5 times the top dose, with 0 first for the Hill model."""
    positive = doses[doses > 0]
    low, high = positive[0] / 100.0, 5.0 * doses[-1]
    points = np.exp(np.linspace(math.log(low), math.log(high), 48))
    points[0], points[-1] = low, high
    return np.r_[0.0, points] if model == "hill" else points


def share(model: str, x: np.ndarray, slope: float, ec50: float) -> np.ndarray:
    """The share of the change a Hill curve has covered at each dose."""
    if model == "ll4":
        return ll4(x, slope, 1.0, 0.0, ec50)
    return hill(x, 0.0, 1.0, ec50, slope)


def level_sums(
    shares: np.ndarray, means: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """The smallest weighted sum of squares of the means about start + change h, for
    each row of shares h, by the 2 x 2 normal equations; a flat curve where h does not
    vary at the doses."""
    ones = np.ones(means.size)
    s00 = ones @ weights @ ones
    s01 = shares @ weights @ ones
    s11 = np.einsum("ik,kl,il->i", shares, weights, shares)
    t0 = ones @ weights @ means
    t1 = shares @ weights @ means
    det = s00 * s11 - s01 * s01
    flat = det <= 1e-12 * s00 * np.maximum(s11, 1e-300)
    change = np.where(flat, 0.0, (s00 * t1 - s01 * t0) / np.where(flat, 1.0, det))
    start = (t0 - change * s01) / s00
    residual = means[None, :] - start[:, None] - change[:, None] * shares
    return np.einsum("ik,kl,il->i", residual, weights, residual)


def dense_profile(model: str, doses, means, weights, ec50: float) -> float:
    """The profile at an EC50: 600 slopes on a log grid within the documented bounds,
    then golden-section search around the best of each local minimum."""
    low, high = (0.1, 18.0) if model == "ll4" else (1.0, 18.0)
    grid = np.linspace(math.log(low), math.log(high), 600)

    def q(u: float) -> float:
        return float(
            level_sums(share(model, doses, math.exp(u), ec50)[None, :], means, weights)[
                0
            ]
        )

    values = level_sums(
        np.array([share(model, doses, math.exp(u), ec50) for u in grid]), means, weights
    )
    best = float(values.min())
    for i in np.flatnonzero(
        (values <= np.r_[np.inf, values[:-1]]) & (values <= np.r_[values[1:], np.inf])
    ):
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, grid.size - 1)]
        found = optimize.minimize_scalar(
            q, bounds=(a, b), method="bounded", options={"xatol": 1e-12}
        )
        best = min(best, float(found.fun))
    return best


def dense_interval(model, doses, means, weights, estimate, objective, s2, threshold):
    """The region's lowest and highest EC50 at the documented scan, the estimate in
    it, refined by bisection; 0 and infinity at the scan's ends."""
    scan = scan_of(model, doses)
    deviance = np.array(
        [
            (dense_profile(model, doses, means, weights, e) - objective) / s2
            for e in scan
        ]
    )
    inside = scan[deviance <= threshold]
    ends = []
    for side, end in ((0, min([estimate, *inside])), (1, max([estimate, *inside]))):
        if (side == 0 and end <= scan[0]) or (side == 1 and end >= scan[-1]):
            ends.append(0.0 if side == 0 else math.inf)
            continue
        beyond = scan[scan < end].max() if side == 0 else scan[scan > end].min()
        logs = beyond > 0
        a, b = (math.log(beyond), math.log(end)) if logs else (beyond, end)
        # 50 halvings narrow a scan step on log EC50 to double precision.
        for _ in range(50):
            middle = 0.5 * (a + b)
            e = math.exp(middle) if logs else middle
            d = (dense_profile(model, doses, means, weights, e) - objective) / s2
            a, b = (middle, b) if d > threshold else (a, middle)
        ends.append(math.exp(0.5 * (a + b)) if logs else 0.5 * (a + b))
    return ends


def fitted_objective(fits: DoseCurveFits, found: EC50s, g: int) -> float:
    """The weighted sum of squares of a gene's means about its Hill curve."""
    m = fits.models.index(found.model)
    letters = "bcde" if found.model == "ll4" else "gvkn"
    p = [fits.parameters[g, m, fits.parameter_names.index(c)] for c in letters]
    curve = (ll4 if found.model == "ll4" else hill)(fits.doses, *p)
    residual = fits.means[g] - curve
    return float(residual @ np.linalg.solve(fits.covariance[g], residual))


@pytest.mark.parametrize("model_set", ["ll4", "epa"])
def test_intervals_match_a_dense_profile(model_set: str) -> None:
    rng = np.random.default_rng(23)
    x = np.repeat(HALF_LOG, 3)
    truths = [
        (1.5, 7.0, 6.0, 0.3),
        (0.8, 7.0, 6.0, 2.0),
        (4.0, 7.0, 6.0, 20.0),
        (1.5, 5.0, 6.0, 0.05),
        (1.0, 6.0, 6.0, 1.0),
    ]
    values = np.column_stack(
        [ll4(x, *t) + rng.normal(0.0, 0.25, x.size) for t in truths]
    )
    fits = fit_dose_curves(x, values, models=model_set)
    found = ec50s(fits)
    n = x.size
    # The curve's residual variance on n - 4 degrees of freedom, and GraphPad's F.
    threshold = stats.f.ppf(0.95, 1, n - 4)
    for g in range(len(truths)):
        objective = fitted_objective(fits, found, g)
        within = float(np.sum((values[:, g] - fits.means[g].repeat(3)) ** 2))
        s2 = (within + objective) / (n - 4)
        expected = dense_interval(
            found.model,
            fits.doses,
            fits.means[g],
            np.diag(fits.n_samples),
            found.ec50[g],
            objective,
            s2,
            threshold,
        )
        got = [found.ec50_lower[g], found.ec50_upper[g]]
        for a, b in zip(got, expected, strict=True):
            if math.isfinite(b) and b > 0:
                assert a == pytest.approx(b, rel=1e-6)
            else:
                assert a == b
        assert found.threshold[g] == pytest.approx(threshold, rel=1e-12)


def test_a_region_in_two_pieces_ends_at_the_far_piece() -> None:
    # Two steps, at the lowest dose and at the top dose: the Hill curve fits either,
    # and the region of the profile is two pieces apart.
    x = np.repeat(HALF_LOG, 3)
    means = 6.0 + 0.5 * (np.arange(9) >= 1) + 0.8 * (np.arange(9) >= 8)
    fits = fit_dose_curves(x, samples_on(means, 0.5)[:, None], models="epa")
    found = ec50s(fits)
    objective = fitted_objective(fits, found, 0)
    s2 = (
        float(np.sum((samples_on(means, 0.5) - means.repeat(3)) ** 2)) + objective
    ) / 23
    threshold = stats.f.ppf(0.95, 1, 23)
    weights = np.diag(fits.n_samples)
    scan = scan_of("hill", HALF_LOG)
    deviance = np.array(
        [
            (dense_profile("hill", HALF_LOG, means, weights, e) - objective) / s2
            for e in scan
        ]
    )
    inside = deviance <= threshold
    assert np.sum(np.diff(inside.astype(int)) == 1) + inside[0] == 2  # two pieces
    expected = dense_interval(
        "hill", HALF_LOG, means, weights, found.ec50[0], objective, s2, threshold
    )
    got = [found.ec50_lower[0], found.ec50_upper[0]]
    for a, b in zip(got, expected, strict=True):
        assert a == (
            b if not (math.isfinite(b) and b > 0) else pytest.approx(b, rel=1e-6)
        )
    # The interval spans scan points outside the region, between its pieces.
    between = (scan > got[0]) & (scan < got[1]) & ~inside
    assert between.any()


def test_intervals_hold_the_estimate_and_open_at_the_scan_ends() -> None:
    rng = np.random.default_rng(31)
    x = np.repeat(HALF_LOG, 3)
    truths = [(1.5, 7.0, 6.0, 10**u) for u in np.linspace(-3, 2, 12)]
    truths += [(1.0, 6.0, 6.0, 1.0)] * 3  # flat
    values = np.column_stack(
        [ll4(x, *t) + rng.normal(0.0, 0.3, x.size) for t in truths]
    )
    for model_set in ("ll4", "epa"):
        fits = fit_dose_curves(x, values, models=model_set)
        found = ec50s(fits)
        ok = np.isfinite(found.threshold) & ~np.isnan(found.ec50_lower)
        assert ok.sum() >= 12
        assert np.all(found.ec50_lower[ok] <= found.ec50[ok])
        assert np.all(found.ec50[ok] <= found.ec50_upper[ok])
        scan = scan_of(found.model, HALF_LOG)
        # An end is open exactly where the profile at the scan's end is inside.
        for g in np.flatnonzero(ok):
            objective = fitted_objective(fits, found, g)
            within = float(np.sum((values[:, g] - fits.means[g].repeat(3)) ** 2))
            s2 = (within + objective) / 23
            weights = np.diag(fits.n_samples)
            for end, point in (
                (found.ec50_lower[g] == 0.0, scan[0]),
                (np.isinf(found.ec50_upper[g]), scan[-1]),
            ):
                d = (
                    dense_profile(found.model, HALF_LOG, fits.means[g], weights, point)
                    - objective
                ) / s2
                assert (
                    end == (d <= found.threshold[g])
                    or abs(d - found.threshold[g]) < 1e-6
                )
        # Flat genes: the whole range.
        assert_array_equal(found.ec50_lower[-3:], 0.0)
        assert_array_equal(found.ec50_upper[-3:], np.inf)


def test_a_fit_short_of_its_maximum_gets_no_interval() -> None:
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(2)
    values = np.column_stack(
        [ll4(x, 1.5, 7.0, 6.0, 0.3) + rng.normal(0.0, 0.2, x.size) for _ in range(3)]
    )
    fits = fit_dose_curves(x, values, models="ll4")
    worse = np.array(fits.parameters)
    worse[1, 0, 3] *= 3.0  # gene 1's EC50 moved off its maximum
    found = ec50s(dataclasses.replace(fits, parameters=worse))
    assert np.isnan(found.ec50_lower[1]) and np.isnan(found.ec50_upper[1])
    assert np.isfinite(found.ec50_lower[[0, 2]]).all()
    assert found.ec50[1] == pytest.approx(3.0 * fits.parameters[1, 0, 3])
    assert (
        "intervals not computed for 1 gene: the profile found a curve better"
        in repr(found)
    )


def test_the_bends_lie_where_the_chapter_says() -> None:
    # The chapter's formula: the bends lie where the curve has covered 1 / (1 + k) and
    # k / (1 + k) of its change, found here by root search on the documented curve.
    # The strict-count test below checks the code's bends.
    for slope in (0.5, 1.0, 1.7, 6.0):
        low = optimize.brentq(
            lambda d: float(ll4(d, slope, 1.0, 0.0, 1.0)) - 1 / (1 + BEND), 1e-12, 1.0
        )
        high = optimize.brentq(
            lambda d: float(ll4(d, slope, 1.0, 0.0, 1.0)) - BEND / (1 + BEND), 1.0, 1e12
        )
        assert low == pytest.approx(BEND ** (-1 / slope), rel=1e-10)
        assert high == pytest.approx(BEND ** (1 / slope), rel=1e-10)


@pytest.mark.parametrize("rises", [True, False])
@pytest.mark.parametrize(
    ("doses_below", "doses_above", "expected"),
    [(2, 2, True), (1, 2, False), (2, 1, False), (3, 3, True), (0, 3, False)],
)
def test_two_doses_beyond_each_bend_counts_strictly(
    rises: bool, doses_below: int, doses_above: int, expected: bool
) -> None:
    # A curve of slope 1 and EC50 1 exactly has its bends at 1 / k and k, whether it
    # rises or falls: doses placed strictly beyond them, and one exactly on each bend,
    # which does not count; the vehicle never does.
    low, high = 1.0 / BEND, BEND
    doses = np.r_[
        0.0,
        low * 0.5 ** np.arange(doses_below, 0, -1),
        low,
        1.0,
        high,
        high * 2.0 ** np.arange(1, doses_above + 1),
    ]
    c, d = (8.0, 6.0) if rises else (6.0, 8.0)
    curve = ll4(doses, 1.0, c, d, 1.0)
    fits = fit_dose_curves(
        np.repeat(doses, 3), samples_on(curve, 0.01)[:, None], models="ll4"
    )
    exact = np.array(fits.parameters)
    exact[0, 0, :] = (1.0, c, d, 1.0)  # b, c, d, e
    found = ec50s(dataclasses.replace(fits, parameters=exact))
    assert found.hill_slope[0] == (1.0 if rises else -1.0)
    assert found.two_doses_beyond_each_bend[0] == expected


def test_a_curve_with_equal_levels_has_no_hill_slope() -> None:
    # The two levels set equal: no slope and no bend check, for both models.
    x = np.repeat(HALF_LOG, 3)
    values = (6.0 + np.tile([-0.2, 0.0, 0.2], 9))[:, None]
    for model_set, row in (
        ("ll4", (1.5, 6.0, 6.0, 0.3)),
        ("epa", (6.0, 0.0, 0.3, 2.0)),
    ):
        fits = fit_dose_curves(x, values, models=model_set)
        m = fits.models.index("ll4" if model_set == "ll4" else "hill")
        exact = np.array(fits.parameters)
        exact[0, m, :4] = row
        found = ec50s(dataclasses.replace(fits, parameters=exact))
        assert np.isnan(found.hill_slope[0])
        assert not found.two_doses_beyond_each_bend[0]
        assert found.bottom_plateau[0] == found.top_plateau[0] == 6.0


def test_best_model_is_the_lowest_aicc() -> None:
    # A sigmoid, whose best model in "epa" is the Hill model, and a straight line in
    # dose, whose best is not: best_model against the lowest AICc found here.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(1)
    values = np.column_stack(
        [
            ll4(x, 2.0, 8.0, 6.0, 0.3) + rng.normal(0.0, 0.1, x.size),
            6.0 + 0.05 * x + rng.normal(0.0, 0.1, x.size),
        ]
    )
    fits = fit_dose_curves(x, values, models="epa")
    found = ec50s(fits)
    aicc = np.where(np.isfinite(fits.aicc), fits.aicc, np.inf)
    lowest = np.argmin(aicc, axis=1) == fits.models.index("hill")
    assert_array_equal(found.best_model, lowest)
    assert found.best_model.tolist() == [True, False]


def test_statuses_at_their_edges() -> None:
    x = np.repeat(HALF_LOG, 3)
    values = np.column_stack(
        [samples_on(ll4(HALF_LOG, 1.5, 8.0, 6.0, e)) for e in (0.3, 0.3, 0.3, 0.3)]
        + [np.full(x.size, 6.0)]
    )
    fits = fit_dose_curves(x, values, models="ll4")
    parameters = np.array(fits.parameters)
    for g, e in enumerate((0.01, 30.0, 0.01 * (1 - 1e-12), 30.0 * (1 + 1e-12))):
        parameters[g, 0, 3] = e
    found = ec50s(dataclasses.replace(fits, parameters=parameters))
    assert found.status.tolist() == [
        "within the doses",
        "within the doses",
        "below the lowest dose",
        "above the top dose",
        "no curve",
    ]
    assert np.isnan(found.ec50[4]) and np.isnan(found.hill_slope[4])
    assert found.parameters_at_bounds[4] == ""


def test_units_order_and_threads() -> None:
    rng = np.random.default_rng(41)
    x = np.repeat(HALF_LOG, 3)
    truths = [
        (1.5, 7.0, 6.0, 0.3),
        (0.8, 5.0, 6.0, 2.0),
        (4.0, 7.0, 6.0, 20.0),
        (2.0, 7.0, 6.0, 0.003),
    ]
    values = np.column_stack(
        [ll4(x, *t) + rng.normal(0.0, 0.25, x.size) for t in truths]
    )
    for model_set in ("ll4", "epa"):
        base = ec50s(fit_dose_curves(x, values, models=model_set), n_threads=1)
        scaled = ec50s(
            fit_dose_curves(x * 1000.0, values, models=model_set), n_threads=4
        )
        for name in ("ec50", "ec50_lower", "ec50_upper"):
            assert_allclose(
                column(scaled, name), 1000.0 * column(base, name), rtol=1e-6
            )
        for name in ("hill_slope", "bottom_plateau", "top_plateau"):
            assert_allclose(
                column(scaled, name), column(base, name), rtol=1e-6, atol=1e-9
            )
        order = [2, 0, 3, 1]
        reordered = ec50s(
            fit_dose_curves(x, values[:, order], models=model_set), n_threads=4
        )
        for name in ("ec50", "ec50_lower", "ec50_upper", "hill_slope"):
            assert_array_equal(column(reordered, name), column(base, name)[order])
        threads = ec50s(fit_dose_curves(x, values, models=model_set), n_threads=4)
        for name in ("ec50", "ec50_lower", "ec50_upper", "hill_slope", "top_plateau"):
            assert_array_equal(column(threads, name), column(base, name))


def test_values_shifted_and_scaled() -> None:
    rng = np.random.default_rng(43)
    x = np.repeat(HALF_LOG, 3)
    truths = [(1.5, 7.0, 6.0, 0.3), (0.8, 5.0, 6.0, 2.0)]
    values = np.column_stack(
        [ll4(x, *t) + rng.normal(0.0, 0.25, x.size) for t in truths]
    )
    base = ec50s(fit_dose_curves(x, values, models="ll4"))
    shifted = ec50s(fit_dose_curves(x, values + 3.0, models="ll4"))
    doubled = ec50s(fit_dose_curves(x, values * 2.0, models="ll4"))
    # Values far below 1, whose spread is still far above their rounding: the test of
    # rounding is relative to the values.
    tiny = ec50s(fit_dose_curves(x, values * 1e-14, models="ll4"))
    for name in ("ec50", "ec50_lower", "ec50_upper", "hill_slope"):
        assert_allclose(column(shifted, name), column(base, name), rtol=1e-6)
        assert_allclose(column(doubled, name), column(base, name), rtol=1e-6)
        assert_allclose(column(tiny, name), column(base, name), rtol=1e-6)
    for name in ("bottom_plateau", "top_plateau"):
        assert_allclose(column(shifted, name), column(base, name) + 3.0, rtol=1e-8)
        assert_allclose(column(doubled, name), column(base, name) * 2.0, rtol=1e-8)


def test_no_replicates_no_interval() -> None:
    curve = ll4(HALF_LOG, 1.5, 8.0, 6.0, 0.3) + np.array(
        [0, 0.1, -0.1, 0.05, 0, 0.1, -0.05, 0, 0.02]
    )
    fits = fit_dose_curves(HALF_LOG, curve[:, None], models="ll4")
    with pytest.raises(
        ValueError,
        match="intervals need replicates within dose groups.*intervals=False",
    ):
        ec50s(fits)
    found = ec50s(fits, intervals=False)
    assert np.isfinite(found.ec50[0]) and np.isfinite(found.hill_slope[0])
    assert np.isnan(found.ec50_lower[0]) and np.isnan(found.ec50_upper[0])
    assert np.isnan(found.threshold[0])
    assert not found.intervals
    assert "; no intervals\n" in repr(found)
    assert "intervals open" not in repr(found)
    # With replicates, intervals=False leaves them out too.
    x = np.repeat(HALF_LOG, 3)
    replicated = fit_dose_curves(
        x, samples_on(ll4(HALF_LOG, 1.5, 8.0, 6.0, 0.3))[:, None], models="ll4"
    )
    without = ec50s(replicated, intervals=False)
    assert (
        np.isnan(without.ec50_lower[0]) and without.ec50[0] == ec50s(replicated).ec50[0]
    )
    with pytest.raises(TypeError, match="intervals"):
        ec50s(replicated, intervals=1)  # type: ignore[arg-type]


def test_values_on_the_curve_without_spread_get_no_interval() -> None:
    # s^2 is 0 to rounding: the deviance has no scale.
    x = np.repeat(HALF_LOG, 3)
    values = np.repeat(ll4(HALF_LOG, 1.5, 8.0, 6.0, 0.3), 3)
    found = ec50s(fit_dose_curves(x, values[:, None], models="ll4"))
    assert found.ec50[0] == pytest.approx(0.3, rel=1e-6)
    assert np.isnan(found.ec50_lower[0]) and np.isnan(found.threshold[0])


def test_the_bmdexpress_set_reads_its_hill_model_within_its_own_bounds() -> None:
    # The rule sets' set fits BMDS's Hill model with k up to 1,000 times the top dose:
    # the EC50s and intervals of "epa" where both ranges hold the region, and limits at
    # its own bound rather than at the 5 times the top dose of "epa".
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(31)
    values = ll4(x, 1.5, 7.0, 6.0, 0.3) + rng.normal(0.0, 0.2, x.size)
    epa = ec50s(fit_dose_curves(x, values[:, None], models="epa"))
    fits = fit_dose_curves(x, np.column_stack([values] * 3), models="bmdexpress")
    wide = ec50s(fits)
    assert wide.model == "hill"
    for name in ("ec50", "ec50_lower", "ec50_upper", "hill_slope"):
        assert column(wide, name)[0] == pytest.approx(column(epa, name)[0], rel=1e-6)
    parameters = np.array(fits.parameters)
    m, k = fits.models.index("hill"), fits.parameter_names.index("k")
    parameters[1, m, k] = 5.0 * 30.0
    parameters[2, m, k] = 1000.0 * 30.0
    limits = ec50s(dataclasses.replace(fits, parameters=parameters), intervals=False)
    named = ["ec50" in text.split(", ") for text in limits.parameters_at_bounds]
    assert named == [False, False, True]


def test_the_hill_model_s_ec50_of_zero() -> None:
    # A response complete by the lowest dose: k at 0, its slope undefined, and the
    # interval's lower end 0 when the profile at 0 lies inside.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(7)
    curve = 6.0 + 1.0 * (HALF_LOG > 0)
    values = np.column_stack(
        [np.repeat(curve, 3) + rng.normal(0.0, 0.2, x.size) for _ in range(30)]
    )
    fits = fit_dose_curves(x, values, models="epa")
    found = ec50s(fits)
    zero = found.ec50 == 0.0
    assert zero.any()
    assert np.isnan(found.hill_slope[zero]).all()
    assert (found.ec50_lower[zero] == 0.0).all()
    assert np.all(found.status[zero] == "below the lowest dose")
    assert not found.two_doses_beyond_each_bend[zero].any()
    assert all("ec50" in text for text in found.parameters_at_bounds[zero])
    # The upper end of one such gene, against the dense profile.
    g = int(np.flatnonzero(zero)[0])
    objective = fitted_objective(fits, found, g)
    within = float(np.sum((values[:, g] - fits.means[g].repeat(3)) ** 2))
    expected = dense_interval(
        "hill",
        HALF_LOG,
        fits.means[g],
        np.diag(fits.n_samples),
        0.0,
        objective,
        (within + objective) / 23,
        stats.f.ppf(0.95, 1, 23),
    )
    assert expected[0] == 0.0
    assert found.ec50_upper[g] == pytest.approx(expected[1], rel=1e-6)


def test_fits_without_a_vehicle_and_with_a_one_sample_group() -> None:
    # No dose 0, and a top dose of one sample: weights n_k and s^2 on n - 4.
    doses = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    x = np.r_[np.repeat(doses[:-1], 3), doses[-1]]
    rng = np.random.default_rng(13)
    values = ll4(x, 1.5, 8.0, 6.0, 0.3)[:, None] + rng.normal(0.0, 0.2, (x.size, 2))
    fits = fit_dose_curves(x, values, models="ll4")
    found = ec50s(fits)
    assert found.lowest_dose == 0.01 and found.top_dose == 30.0
    threshold = stats.f.ppf(0.95, 1, x.size - 4)
    assert found.threshold[0] == pytest.approx(threshold)
    for g in range(values.shape[1]):
        objective = fitted_objective(fits, found, g)
        counts = fits.n_samples.astype(int)
        within = float(np.sum((values[:, g] - np.repeat(fits.means[g], counts)) ** 2))
        expected = dense_interval(
            "ll4",
            fits.doses,
            fits.means[g],
            np.diag(fits.n_samples),
            found.ec50[g],
            objective,
            (within + objective) / (x.size - 4),
            threshold,
        )
        assert_allclose([found.ec50_lower[g], found.ec50_upper[g]], expected, rtol=1e-6)


def test_moderated_intervals_read_the_f_quantile_and_chi_square_at_infinite_df() -> (
    None
):
    rng = np.random.default_rng(17)
    doses = HALF_LOG
    k = doses.size
    means = np.vstack(
        [ll4(doses, 1.5, 7.0, 6.0, 0.3) + rng.normal(0, 0.05, k) for _ in range(3)]
    )
    covariance = np.broadcast_to(np.diag(np.full(k, 1.0 / 3.0)), (3, k, k)).copy()
    for df, expected in (
        (10.0, stats.f.ppf(0.95, 1, 10)),
        (np.inf, stats.chi2.ppf(0.95, 1)),
    ):
        data = _CurveData(
            "moderated",
            doses,
            np.full(k, 3.0),
            means,
            covariance,
            np.full(3, 0.04),
            np.full(3, df),
            None,
        )
        found = ec50s(_fit_means(data, "ll4"))
        assert_allclose(found.threshold, expected, rtol=1e-10)
        assert np.all(found.ec50_lower < found.ec50) and np.all(
            found.ec50 < found.ec50_upper
        )


def test_arguments_are_refused() -> None:
    with pytest.raises(TypeError, match="fits must be a DoseCurveFits"):
        ec50s("fits")  # type: ignore[arg-type]
    x = np.repeat([0.0, 0.1, 1.0, 10.0], 3)
    values = np.random.default_rng(1).normal(6.0, 0.2, (x.size, 2))
    fits = fit_dose_curves(x, values, models="epa")
    with pytest.raises(
        ValueError,
        match=r"model 'hill' was not fitted: no spare dose group",
    ):
        ec50s(fits)
    good = fit_dose_curves(
        np.repeat(HALF_LOG, 2),
        np.ones((18, 1)) + np.arange(18)[:, None] * 0.01,
        models="ll4",
    )
    with pytest.raises(TypeError, match="n_threads"):
        ec50s(good, n_threads=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="n_threads"):
        ec50s(good, n_threads=0)


def test_the_table_reads_back() -> None:
    rng = np.random.default_rng(3)
    x = np.repeat(HALF_LOG, 3)
    values = np.column_stack(
        [
            ll4(x, 1.5, 7.0, 6.0, 0.3) + rng.normal(0.0, 0.2, x.size),
            np.full(x.size, 6.0),
        ]
    )
    found = ec50s(fit_dose_curves(x, values, models="epa"))
    table = found.table(feature_ids=["Cyp1a1", "Alb"])
    assert isinstance(table, Table)
    assert table.columns == (
        "feature_id",
        "best_model",
        "ec50",
        "ec50_lower",
        "ec50_upper",
        "hill_slope",
        "bottom_plateau",
        "top_plateau",
        "ec50_status",
        "two_doses_beyond_each_bend",
        "parameters_at_bounds",
    )
    assert table["feature_id"].tolist() == ["Cyp1a1", "Alb"]
    assert_array_equal(table["ec50_upper"], found.ec50_upper)
    assert table["ec50_status"].tolist() == found.status.tolist()
    assert ec50s(fit_dose_curves(x, values, models="epa")).table()[
        "feature_id"
    ].tolist() == ["feature_0", "feature_1"]
    with pytest.raises(ValueError, match="one id per feature"):
        found.table(feature_ids=["a"])
    again = pickle.loads(pickle.dumps(found))
    for name in table.columns:
        assert_array_equal(again.table()[name], found.table()[name])
    with pytest.raises(ValueError):
        again.ec50[0] = 1.0  # read-only
    text = repr(found)
    assert text.startswith(
        "EC50s (hill curve of the 'epa' set, constant variance): 2 genes"
    )
    assert "1 within the doses, 1 no curve" in text


def test_the_printout_counts_limits() -> None:
    # A step between two doses fits with its slope at the bound of 18, a limit.
    x = np.repeat(HALF_LOG, 3)
    noise = np.tile([-0.1, 0.0, 0.1], 9)
    step = np.where(x > 0.2, 7.0, 6.0) + noise
    rise = ll4(x, 1.5, 7.0, 6.0, 0.3) + noise
    found = ec50s(fit_dose_curves(x, np.column_stack([step, rise]), models="ll4"))
    assert found.parameters_at_bounds.tolist() == ["hill_slope", ""]
    assert found.hill_slope[0] == 18.0
    text = repr(found)
    assert "  genes with a parameter at a bound (a limit, not an estimate): 1\n" in text
    assert text.endswith("  intervals open below: 0; open above: 0")


def test_an_ll4_ec50_at_its_upper_bound_is_named_a_limit() -> None:
    # A rise at the top doses pins many fits' EC50 at 5 times the top dose: a limit,
    # named as the EC50's, with an interval open above.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(20)
    values = ll4(x, 4.0, 7.0, 6.0, 20.0)[:, None] + rng.normal(0.0, 0.25, (x.size, 12))
    found = ec50s(fit_dose_curves(x, values, models="ll4"))
    pinned = found.ec50 == 5.0 * 30.0
    assert pinned.sum() >= 3
    assert all(
        "ec50" in text.split(", ") for text in found.parameters_at_bounds[pinned]
    )
    assert np.isinf(found.ec50_upper[pinned]).all()
    assert (found.status[pinned] == "above the top dose").all()


def test_a_parameter_a_hair_short_of_its_bound_is_named_a_limit() -> None:
    # The refinement can stop short of a bound that its objective keeps falling
    # toward: within 0.01% of the bound, the parameter counts as at it.
    x = np.repeat(HALF_LOG, 3)
    step = np.where(x > 0.2, 7.0, 6.0) + np.tile([-0.1, 0.0, 0.1], 9)
    fits = fit_dose_curves(x, np.column_stack([step] * 5), models="ll4")
    parameters = np.array(fits.parameters)
    for g, (b, e) in enumerate(
        (
            (18.0 * (1 - 5e-5), 0.5),
            (18.0 * (1 - 2e-4), 0.5),
            (2.0, 150.0 * (1 - 5e-5)),
            (2.0, 1e-4 * (1 + 5e-5)),
            (0.1 * (1 + 5e-5), 0.5),
        )
    ):
        parameters[g, 0, [0, 3]] = b, e
    found = ec50s(dataclasses.replace(fits, parameters=parameters), intervals=False)
    assert found.parameters_at_bounds.tolist() == [
        "hill_slope",
        "",
        "ec50",
        "ec50",
        "hill_slope",
    ]
    # A fit of the references' first case that stopped 9e-7 short of 18.
    case = fixture_fits(load("profiles.json")["cases"][0])
    b = case.parameters[6, 0, case.parameter_names.index("b")]
    assert 0.0 < 18.0 - b < 18.0 * 1e-4
    assert ec50s(case, intervals=False).parameters_at_bounds[6] == "hill_slope"


# ---------------------------------------------------------------------------------------
# Against the references
# ---------------------------------------------------------------------------------------


def load(name: str) -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / name).read_text())
    return found


def fixture_fits(record: dict[str, Any]) -> DoseCurveFits:
    """The fits of a fixture's case: refitted from its samples under constant
    variance, or from its dose-group means, covariance and variances otherwise."""
    if "plug_in" not in record:
        values = np.array(record["values"], dtype=float).T
        return fit_dose_curves(
            np.array(record["doses"]), values, models=record["model_set"]
        )
    p = record["plug_in"]
    data = _CurveData(
        record["variance"],
        np.array(p["doses"]),
        np.array(p["n_samples"]),
        np.array(p["means"]),
        np.array(p["covariance"]),
        np.array(p["covariance_scale"]),
        np.array(p["df"]),
        None,
    )
    return _fit_means(data, record["model_set"])


@pytest.mark.parametrize("case", range(len(load("profiles.json")["cases"])))
def test_intervals_match_mpmath(case: int) -> None:
    record = load("profiles.json")["cases"][case]
    found = ec50s(fixture_fits(record))
    compared = 0
    for g, ends in enumerate(record["mpmath"]):
        if ends is None:
            continue
        compared += 1
        for ours, theirs in zip(
            (found.ec50_lower[g], found.ec50_upper[g]), ends, strict=True
        ):
            if math.isfinite(theirs) and theirs > 0.0:
                # Brent's roots stop within 1e-10 on ln E; the validation's tier is 1e-6.
                assert ours == pytest.approx(theirs, rel=1e-8)
            else:
                assert ours == theirs
    assert compared >= 5


def test_the_printout_counts_open_ends_as_mpmath_finds_them() -> None:
    record = load("profiles.json")["cases"][0]
    ends = np.array(record["mpmath"], dtype=float)
    low, high = int(np.sum(ends[:, 0] == 0.0)), int(np.sum(np.isinf(ends[:, 1])))
    assert low != high
    found = ec50s(fixture_fits(record))
    assert f"  intervals open below: {low}; open above: {high}" in repr(found)


def test_hill_slope_signs_match_drc() -> None:
    """GraphPad's Hill slope has the sign of -b sign(d - c) of drc's LL.4 estimates,
    where drc reaches CytePool's maximum; its size, at a maximum flat along the slope,
    the map's exact tests pin."""
    cases = load("profiles.json")["cases"]
    compared = 0
    for item in load("drc.json")["genes"]:
        c, g = (int(v) for v in item["key"].split("|"))
        record = cases[c]
        fits = fixture_fits(record)
        found = ec50s(fits)
        values = np.array(record["values"], dtype=float)[g]
        within = float(
            np.sum((values - fits.means[g].repeat(fits.n_samples.astype(int))) ** 2)
        )
        rss = within + fitted_objective(fits, found, g)
        if item["rss"] > rss * (1.0 + 1e-8):
            continue  # drc stopped below CytePool's maximum
        b, c_, d, _ = item["coefficients"]
        assert np.sign(found.hill_slope[g]) == np.sign(-b * math.copysign(1.0, d - c_))
        compared += 1
    assert compared >= 4


def test_an_interval_narrower_than_one_scan_step() -> None:
    # Little noise: the region holds no scan point, and both ends lie between the scan
    # points around the estimate.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(29)
    values = ll4(x, 2.0, 8.0, 6.0, 0.5) + rng.normal(0.0, 0.003, x.size)
    fits = fit_dose_curves(x, values[:, None], models="ll4")
    found = ec50s(fits)
    scan = scan_of("ll4", HALF_LOG)
    low, high = found.ec50_lower[0], found.ec50_upper[0]
    assert low < found.ec50[0] < high
    assert not np.any((scan >= low) & (scan <= high))
    objective = fitted_objective(fits, found, 0)
    s2 = (float(np.sum((values - fits.means[0].repeat(3)) ** 2)) + objective) / 23
    expected = dense_interval(
        "ll4",
        HALF_LOG,
        fits.means[0],
        np.diag(fits.n_samples),
        found.ec50[0],
        objective,
        s2,
        stats.f.ppf(0.95, 1, 23),
    )
    assert_allclose([low, high], expected, rtol=1e-6)


def test_the_hill_model_s_interval_can_end_below_a_hundredth_of_the_lowest_dose() -> (
    None
):
    # A Hill curve complete well before the lowest dose, measured precisely: its EC50
    # lies between 0 and the grid's lowest positive EC50, and so does the interval's
    # lower end, found from the profile at an EC50 of 0, which lies outside.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(37)
    values = hill(x, 6.0, 2.0, 5e-5, 1.0) + rng.normal(0.0, 5e-4, x.size)
    fits = fit_dose_curves(x, values[:, None], models="epa")
    found = ec50s(fits)
    assert 0.0 < found.ec50_lower[0] < 1e-4  # a hundredth of the lowest dose
    objective = fitted_objective(fits, found, 0)
    s2 = (float(np.sum((values - fits.means[0].repeat(3)) ** 2)) + objective) / 23
    expected = dense_interval(
        "hill",
        HALF_LOG,
        fits.means[0],
        np.diag(fits.n_samples),
        found.ec50[0],
        objective,
        s2,
        stats.f.ppf(0.95, 1, 23),
    )
    assert_allclose([found.ec50_lower[0], found.ec50_upper[0]], expected, rtol=1e-6)


def test_hill_intervals_follow_the_dose_unit() -> None:
    # The lower end found from an EC50 of 0 is relative to the doses: the same doses
    # in other units, such as molar doses, give the same interval in those units.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(37)
    values = hill(x, 6.0, 2.0, 5e-5, 1.0) + rng.normal(0.0, 5e-4, x.size)
    base = ec50s(fit_dose_curves(x, values[:, None], models="epa"))
    assert 0.0 < base.ec50_lower[0] < 1e-4
    for factor in (1e-9, 1e-3, 1e3):
        scaled = ec50s(fit_dose_curves(x * factor, values[:, None], models="epa"))
        for name in ("ec50", "ec50_lower", "ec50_upper"):
            assert column(scaled, name)[0] == pytest.approx(
                factor * column(base, name)[0], rel=1e-8
            )


def test_a_fit_that_did_not_converge_keeps_its_interval() -> None:
    # A steep rise at the top doses whose fit stopped at its iteration limit: read as
    # it stands, with the interval of the dense profile.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(20)
    values = ll4(x, 4.0, 7.0, 6.0, 20.0)[:, None] + rng.normal(0.0, 0.25, (x.size, 300))
    fits = fit_dose_curves(x, values[:, [8]], models="ll4")
    assert not fits.converged[0, 0]
    found = ec50s(fits)
    objective = fitted_objective(fits, found, 0)
    within = float(np.sum((values[:, 8] - fits.means[0].repeat(3)) ** 2))
    expected = dense_interval(
        "ll4",
        HALF_LOG,
        fits.means[0],
        np.diag(fits.n_samples),
        found.ec50[0],
        objective,
        (within + objective) / 23,
        stats.f.ppf(0.95, 1, 23),
    )
    for ours, theirs in zip((found.ec50_lower[0], found.ec50_upper[0]), expected):
        assert ours == (
            theirs if math.isinf(theirs) else pytest.approx(theirs, rel=1e-6)
        )


def test_the_module_examples_run() -> None:
    assert doctest.testmod(potency, raise_on_error=False).failed == 0


def test_the_chapter_s_ec50_example() -> None:
    """The EC50 example of the potency chapter, on the series of the dose chapter, and
    the numbers its text states."""
    import re

    chapter = (
        Path(__file__).parents[1] / "docs" / "source" / "statistics" / "potency.md"
    )
    text = chapter.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    namespace: dict[str, Any] = {}
    # The chapter's series, then its EC50 example.
    exec(blocks[0], namespace)  # noqa: S102
    exec(next(b for b in blocks if '"ec50s"' in b), namespace)  # noqa: S102
    potencies = namespace["potencies"]
    status = np.asarray(potencies["ec50_status"])
    counts = [
        int(np.sum(status == s))
        for s in ("within the doses", "below the lowest dose", "above the top dose")
    ]
    words = " ".join(text.split())
    assert (
        f"Of the hepatocytes' {len(potencies)} genes, {counts[0]} have an EC50 within the doses, {counts[1]} below the lowest dose and {counts[2]} above the top dose"
        in words
    )
    gated = np.asarray(potencies["passes_gate"])
    lower = np.asarray(potencies["ec50_lower"])
    upper = np.asarray(potencies["ec50_upper"])
    readable = (
        gated
        & (status == "within the doses")
        & (lower > 0)
        & np.isfinite(upper)
        & np.asarray(potencies["two_doses_beyond_each_bend"])
    )
    assert (
        f"{int(readable.sum())} of the {int(gated.sum())} have an EC50 within the doses, a closed interval and two doses beyond each bend"
        in words
    )
    bounded = int(np.sum(np.asarray(potencies["parameters_at_bounds"]) != ""))
    assert (
        f"{bounded} of the hepatocytes' {len(potencies)} genes have a parameter at a bound"
        in words
    )
    # The printout of the same EC50s.
    printout = repr(namespace["ll4"].ec50s("hepatocyte"))
    assert (
        f"  {counts[0]} within the doses, {counts[1]} below the lowest dose" in printout
    )
    assert (
        f"genes with a parameter at a bound (a limit, not an estimate): {bounded}"
        in (printout)
    )


def test_a_step_s_interval_lies_inside_the_gap_of_its_doses() -> None:
    # The chapter's limit: a step between doses 3 and 10 fits with its slope at 18,
    # and the bounded slope cuts the gap's outer parts off its interval.
    x = np.repeat(HALF_LOG, 3)
    rng = np.random.default_rng(11)
    values = 6.0 + 1.0 * (x >= 6.0) + rng.normal(0.0, 0.25, x.size)
    found = ec50s(fit_dose_curves(x, values[:, None], models="ll4"))
    assert found.hill_slope[0] == 18.0
    assert found.parameters_at_bounds[0] == "hill_slope"
    low, high = (
        round(float(found.ec50_lower[0]), 2),
        round(float(found.ec50_upper[0]), 2),
    )
    assert 3.0 < low < high < 10.0
    chapter = (
        Path(__file__).parents[1] / "docs" / "source" / "statistics" / "potency.md"
    )
    assert (
        f"a step between doses 3 and 10 got the interval [{low}, {high}]"
        in " ".join(chapter.read_text().split())
    )


def test_a_steep_precise_curve_whose_slope_grid_misses_the_profile() -> None:
    # A steep curve measured precisely: near its EC50 the grid of 24 slopes sits
    # deviance units above the profile, past the threshold, so the profile must be
    # refined wherever the grid lies outside the region, or the region shrinks.
    rng = np.random.default_rng(0)
    slope, ec50 = rng.uniform(2.0, 14.0), 10 ** rng.uniform(-1.5, 1.0)
    sd = 10 ** rng.uniform(-2.5, -1.0)
    x = np.repeat(HALF_LOG, 3)
    curve = ll4(x, slope, 7.0, 6.0, ec50)
    values = curve[:, None] + rng.normal(0.0, sd, (x.size, 4))
    fits = fit_dose_curves(x, values[:, [1]], models="ll4")
    found = ec50s(fits)
    objective = fitted_objective(fits, found, 0)
    within = float(np.sum((values[:, 1] - fits.means[0].repeat(3)) ** 2))
    s2 = (within + objective) / 23
    threshold = stats.f.ppf(0.95, 1, 23)
    # The grid's bound and the profile at the scan's EC50s near the estimate.
    scan = scan_of("ll4", HALF_LOG)
    near = scan[np.argsort(np.abs(np.log(scan / found.ec50[0])))[:2]]
    weights = np.diag(fits.n_samples)
    grid_slopes = np.exp(np.linspace(math.log(0.1), math.log(18.0), 24))
    straddles = 0
    for e in near:
        shares = np.array([share("ll4", HALF_LOG, b, e) for b in grid_slopes])
        bound = (level_sums(shares, fits.means[0], weights).min() - objective) / s2
        exact = (
            dense_profile("ll4", HALF_LOG, fits.means[0], weights, e) - objective
        ) / s2
        straddles += int(bound > threshold >= exact)
    assert straddles >= 1
    expected = dense_interval(
        "ll4", HALF_LOG, fits.means[0], weights, found.ec50[0], objective, s2, threshold
    )
    assert_allclose([found.ec50_lower[0], found.ec50_upper[0]], expected, rtol=1e-6)
