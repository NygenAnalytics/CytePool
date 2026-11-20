"""Tests for LOWESS, its span rule and natural cubic spline bases.

Reference values in ``tests/data/smoothing`` come from R's ``stats::lowess``
and ``splines::ns`` (bases, knots, knots on the distinct values and ``predict`` at new
points), Hmisc's ``rcspline.eval`` (knots by Harrell's rule and restricted cubic
columns) and rms's ``anova(ols(y ~ rcs(x, nk)))``, written by
``validation/smoothing.py``. The span rule is checked here against values derived by
hand, and in ``tests/test_voom.py`` against limma's ``chooseLowessSpan``. The weighted
LOWESS behind the trend of the likelihood estimator is checked here against values
derived by hand from its rules, and in ``tests/test_empirical_bayes.py`` against
limma's ``weightedLowess``, directly and through the trend.
"""

import dataclasses
import doctest
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from cytepool.stats import smoothing
from cytepool.stats.smoothing import (
    NaturalSplineBasis,
    _anchor_spacing,
    _weighted_lowess,
    choose_lowess_span,
    lowess,
    natural_spline_basis,
)

DATA = Path(__file__).parent / "data" / "smoothing"
LOWESS_CASES = (
    "span_0.1",
    "span_0.3",
    "span_0.4",
    "span_0.5",
    "span_1.0",
    "span_1.7",
    "iterations_0",
    "iterations_1",
    "iterations_2",
    "iterations_4",
    "delta_0",
    "delta_large",
    "delta_beyond_range",
    "integer_x",
    "integer_x_delta_0",
    "integer_x_large_delta",
    "paired_x_span_0.1",
    "paired_x_span_0.15",
    "paired_x_span_0.3",
    "tie_block_wider_than_window",
    "ties_at_both_ends",
    "n_1",
    "n_2",
    "n_3",
    "n_4",
    "n_5",
    "n_2_tied",
    "heavy_tails",
    "constant_y",
    "line_with_outliers_zero_mad",
    "all_weights_zero",
    "constant_x",
    "two_x_values",
    "span_times_n_just_below_29",
    "span_times_n_30_minus_5e-8",
    "span_times_n_30_minus_2e-7",
    "tricube_upper_cutoff",
    "tricube_lower_cutoff",
    "local_constant_switch_below",
    "local_constant_switch_above",
    "bisquare_cutoffs",
    "x_offset_1e6",
    "x_scale_1e-8",
    "y_scale_1e10",
)
SPLINE_CASES = (
    "df_1",
    "df_2",
    "df_3",
    "df_5",
    "df_2_intercept",
    "df_3_intercept",
    "df_5_intercept",
    "df_7_intercept",
    "no_knots",
    "explicit_knots",
    "explicit_knots_unsorted",
    "boundary_knots_inside_data_df_3",
    "boundary_knots_inside_data_df_4",
    "boundary_knots_inside_data_knots",
    "boundary_knots_reversed_beyond_data",
    "knot_multiplicity_2",
    "knot_multiplicity_2_intercept",
    "knot_multiplicity_3",
    "knot_multiplicity_3_intercept",
    "knot_multiplicity_4",
    "knot_multiplicity_4_intercept",
    "knot_multiplicity_5",
    "knot_multiplicity_5_intercept",
    "integer_x_repeated_quantile_knots",
    "three_distinct_values",
    "quantile_knot_on_lower_boundary",
    "quantile_knot_on_lower_boundary_intercept",
    "quantile_knot_on_upper_boundary",
    "quantile_knots_on_both_boundaries",
    "two_distinct_values",
    "nan_in_x",
)
# Cases for which R's ns() fails.
FAILING_SPLINE_CASES = (
    "fails_every_knot_on_upper_boundary",
    "fails_every_knot_on_lower_boundary",
    "fails_knot_on_boundary_after_shove",
    "fails_knot_beyond_boundary",
)


@pytest.fixture(scope="module")
def lowess_reference() -> dict[str, dict[str, Any]]:
    """R's lowess inputs and outputs, by case."""
    with np.load(DATA / "lowess.npz") as data:
        cases = {}
        for name, (span, iterations, delta) in zip(data["names"], data["parameters"]):
            x, y, fitted = data[str(name)]
            cases[str(name)] = {
                "x": x,
                "y": y,
                "fitted": fitted,
                "span": float(span),
                "iterations": int(iterations),
                "delta": None if np.isnan(delta) else float(delta),
            }
    return cases


@pytest.fixture(scope="module")
def spline_reference() -> dict[str, dict[str, Any]]:
    """R's ns inputs and bases (None where R fails), by case."""
    with np.load(DATA / "natural_splines.npz") as data:
        cases = {}
        for name, df, intercept in zip(data["names"], data["df"], data["intercept"]):
            name = str(name)
            keywords: dict[str, Any] = {
                "df": None if df < 0 else int(df),
                "intercept": bool(intercept),
            }
            for key in ("knots", "boundary_knots"):
                if f"{name}__{key}" in data.files:
                    keywords[key] = data[f"{name}__{key}"]
            basis = data[f"{name}__basis"] if f"{name}__basis" in data.files else None
            cases[name] = {
                "x": data[f"{name}__x"],
                "keywords": keywords,
                "basis": basis,
            }
    return cases


def least_squares_fits(basis: np.ndarray, responses: np.ndarray) -> tuple[Any, int]:
    """Fitted values of least squares on the rows without NaN, and the rank."""
    complete = ~np.isnan(basis).any(axis=1)
    u, s, _ = np.linalg.svd(basis[complete], full_matrices=False)
    u = u[:, : int(np.count_nonzero(s > s[0] * 1e-10))]
    fits = np.full((basis.shape[0], responses.shape[1]), np.nan)
    fits[complete] = u @ (u.T @ responses[complete])
    return fits, u.shape[1]


def assert_spans_the_columns(
    actual: np.ndarray, desired: np.ndarray, tolerance: float = 1e-10
) -> None:
    """Assert that every column of ``desired`` is a combination of the columns of
    ``actual``, to ``tolerance`` relative to that column's largest absolute value, by
    combinations of full rank, so that both span the same space.

    Regressing ``desired`` on ``actual`` keeps the error of a well-conditioned
    ``actual`` small when ``desired`` is ill conditioned, as Harrell's truncated
    cubes are, where least squares fits on ``desired`` would lose digits.
    """
    assert actual.shape == desired.shape
    weights = np.linalg.lstsq(actual, desired, rcond=None)[0]
    residuals = np.max(np.abs(desired - actual @ weights), axis=0)
    assert np.max(residuals / np.max(np.abs(desired), axis=0)) <= tolerance
    assert np.linalg.matrix_rank(weights) == desired.shape[1]


def assert_same_column_space(
    actual: np.ndarray, desired: np.ndarray, tolerance: float = 1e-10
) -> None:
    """Assert equal shapes, NaN rows, ranks and least squares fitted values."""
    assert actual.shape == desired.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(desired))
    responses = np.random.default_rng(5).normal(0.0, 1.0, (actual.shape[0], 3))
    actual_fits, actual_rank = least_squares_fits(actual, responses)
    desired_fits, desired_rank = least_squares_fits(desired, responses)
    assert actual_rank == desired_rank
    complete = ~np.isnan(desired_fits[:, 0])
    scale = np.max(np.abs(desired_fits[complete]))
    np.testing.assert_allclose(
        actual_fits[complete], desired_fits[complete], rtol=0, atol=tolerance * scale
    )


def test_fixtures_hold_every_case(
    lowess_reference: dict[str, Any], spline_reference: dict[str, Any]
) -> None:
    assert set(lowess_reference) == set(LOWESS_CASES)
    assert set(spline_reference) == set(SPLINE_CASES + FAILING_SPLINE_CASES)


@pytest.mark.parametrize("case", LOWESS_CASES)
def test_lowess_matches_r(case: str, lowess_reference: dict[str, Any]) -> None:
    data = lowess_reference[case]
    fitted = lowess(
        data["x"],
        data["y"],
        span=data["span"],
        iterations=data["iterations"],
        delta=data["delta"],
    )
    # The validation finds R's values reproduced to the last bit; 1e-12 of
    # the largest value leaves room for other platforms' rounding.
    scale = np.max(np.abs(data["fitted"]))
    np.testing.assert_allclose(fitted, data["fitted"], rtol=0, atol=1e-12 * scale)


@pytest.mark.parametrize(
    ("span", "iterations", "delta"),
    [
        (0.04, 0, None),  # two-point neighbourhoods
        (0.04, 3, 0.0),
        (0.3, 0, 0.0),
        (0.3, 3, None),
        (2 / 3, 3, 2.0),
        (1.5, 3, None),
    ],
)
def test_lowess_reproduces_a_straight_line(
    span: float, iterations: int, delta: float | None
) -> None:
    x = np.random.default_rng(11).uniform(0.0, 10.0, 50)
    y = 3.0 - 2.0 * x
    fitted = lowess(x, y, span=span, iterations=iterations, delta=delta)
    np.testing.assert_allclose(fitted, y, rtol=0, atol=1e-12 * np.max(np.abs(y)))


@pytest.mark.parametrize("ties", [False, True])
def test_lowess_does_not_depend_on_the_order_of_the_points(ties: bool) -> None:
    rng = np.random.default_rng(13)
    x = rng.integers(0, 12, 70).astype(float) if ties else rng.uniform(0, 10, 70)
    y = np.sin(x) + rng.normal(0.0, 0.4, 70)
    order = rng.permutation(70)
    fitted = lowess(x, y, span=0.3)
    permuted = lowess(x[order], y[order], span=0.3)
    if ties:
        # Sums over tied points run in another order: rounding differs.
        np.testing.assert_allclose(permuted, fitted[order], rtol=1e-12, atol=1e-14)
        for value in np.unique(x):
            assert np.ptp(fitted[x == value]) == 0.0
    else:
        np.testing.assert_array_equal(permuted, fitted[order])


def test_lowess_is_equivariant_under_affine_maps() -> None:
    rng = np.random.default_rng(17)
    x = rng.uniform(0.0, 10.0, 60)
    y = np.sin(x) + rng.standard_cauchy(60) * 0.2
    fitted = lowess(x, y, span=0.4)
    # Scaling by powers of two is exact, so the fit is too.
    np.testing.assert_array_equal(lowess(4.0 * x, 2.0 * y, span=0.4), 2.0 * fitted)
    np.testing.assert_allclose(
        lowess(x + 1000.0, y - 7.0, span=0.4), fitted - 7.0, rtol=0, atol=1e-10
    )


def test_lowess_span_of_one_or_more_uses_every_point() -> None:
    rng = np.random.default_rng(19)
    x, y = rng.uniform(0.0, 1.0, 30), rng.normal(0.0, 1.0, 30)
    np.testing.assert_array_equal(lowess(x, y, span=5.0), lowess(x, y, span=1.0))
    assert np.max(np.abs(lowess(x, y, span=0.9) - lowess(x, y, span=1.0))) > 1e-3


def test_lowess_two_point_neighbourhoods_leave_distinct_points_unsmoothed() -> None:
    # The other point of a two-point neighbourhood lies at the radius, where
    # the tricube weight is zero; with delta=0 no point is interpolated.
    rng = np.random.default_rng(23)
    x, y = rng.uniform(0.0, 1.0, 40), rng.normal(0.0, 1.0, 40)
    np.testing.assert_array_equal(lowess(x, y, span=1e-3, delta=0.0), y)
    # By default, points within 1% of the range of x of a fitted point are
    # interpolated instead.
    assert np.any(lowess(x, y, span=1e-3) != y)


def test_lowess_small_samples_and_ties() -> None:
    np.testing.assert_array_equal(lowess([2.0], [5.0]), [5.0])
    np.testing.assert_array_equal(lowess([1.0, 2.0], [5.0, 7.0]), [5.0, 7.0])
    np.testing.assert_array_equal(lowess([1.0, 1.0], [5.0, 7.0]), [6.0, 6.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 10.0])
    fitted = lowess(np.full(5, 2.0), y, iterations=0)
    np.testing.assert_allclose(fitted, np.full(5, 4.0), rtol=1e-15)
    # Integers and lists are read as floats.
    np.testing.assert_array_equal(
        lowess([3, 1, 2, 5, 4], [1, 2, 3, 4, 5]),
        lowess(np.array([3.0, 1, 2, 5, 4]), np.array([1.0, 2, 3, 4, 5])),
    )


def test_lowess_iterations_stop_when_the_median_absolute_residual_vanishes() -> None:
    # A line with three outliers: the median absolute residual is zero up to
    # rounding, so the robustness iterations stop and the outliers keep their
    # pull on the fit.
    x = np.arange(101.0)
    y = 2.0 + 0.5 * x
    y[[20, 50, 80]] += 7.0
    first = lowess(x, y, span=0.1, iterations=0, delta=0.0)
    np.testing.assert_array_equal(lowess(x, y, span=0.1, delta=0.0), first)
    assert first[50] > 2.0 + 0.5 * 50 + 1.0  # pulled up by the outlier


@pytest.mark.parametrize(
    ("x", "y", "keywords", "error", "message"),
    [
        ([1.0, 2.0], [1.0], {}, ValueError, "same length, got 2 and 1"),
        ([], [], {}, ValueError, "at least one point"),
        ([[1.0, 2.0]], [[1.0, 2.0]], {}, ValueError, "1-D array"),
        ([1.0, np.nan], [1.0, 2.0], {}, ValueError, "x must be finite"),
        ([1.0, 2.0], [1.0, np.inf], {}, ValueError, "y must be finite"),
        ([1e308, -1e308], [1.0, 2.0], {}, ValueError, "range of x must be finite"),
        (["a", "b"], [1.0, 2.0], {}, TypeError, "x must hold real numbers"),
        ([1.0, 2.0], [True, False], {}, TypeError, "y must hold real numbers"),
        ([1.0, 2.0], [1.0, 2.0], {"span": 0.0}, ValueError, "positive and finite"),
        ([1.0, 2.0], [1.0, 2.0], {"span": np.nan}, ValueError, "positive and finite"),
        ([1.0, 2.0], [1.0, 2.0], {"span": "0.5"}, TypeError, "span must be a real"),
        ([1.0, 2.0], [1.0, 2.0], {"iterations": -1}, ValueError, "non-negative"),
        ([1.0, 2.0], [1.0, 2.0], {"iterations": 2.5}, TypeError, "an integer"),
        ([1.0, 2.0], [1.0, 2.0], {"iterations": True}, TypeError, "an integer"),
        ([1.0, 2.0], [1.0, 2.0], {"delta": -0.1}, ValueError, "non-negative and"),
        ([1.0, 2.0], [1.0, 2.0], {"delta": np.inf}, ValueError, "non-negative and"),
    ],
)
def test_lowess_rejects_invalid_input(
    x: Any, y: Any, keywords: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        lowess(x, y, **keywords)


@pytest.mark.parametrize(
    ("n", "settings", "expected"),
    [
        # 50 / n is a perfect cube: the span is 0.3 + 0.7 times its cube root.
        (400, {}, 0.65),
        (1350, {}, 8 / 15),
        (3200, {}, 0.475),
        (6250, {}, 0.44),
        (50_000, {}, 0.37),
        # A square root, and the linear rule: 10 / 40 and 25 / 100 are 1/4.
        (40, {"small_n": 10, "min_span": 0.2, "power": 0.5}, 0.6),
        (100, {"small_n": 25, "power": 1.0}, 0.475),
        # Up to small_n points, and for a power of 0, the span is 1; with a small_n
        # of 0, it is min_span from 1 point on.
        (1, {"small_n": 1, "min_span": 0.05}, 1.0),
        (10**6, {"power": 0.0}, 1.0),
        (0, {"small_n": 0}, 1.0),
        (7, {"small_n": 0, "min_span": 0.25}, 0.25),
    ],
)
def test_lowess_span_rule_gives_the_values_of_its_formula(
    n: int, settings: dict[str, Any], expected: float
) -> None:
    # Within a few units in the last place: 1/3 and 0.7 are not exact in binary.
    np.testing.assert_allclose(choose_lowess_span(n, **settings), expected, rtol=1e-15)


def test_lowess_span_rule_falls_from_one_toward_its_minimum() -> None:
    spans = np.array([choose_lowess_span(n) for n in range(5001)])
    np.testing.assert_array_equal(spans[:51], 1.0)
    assert np.all(np.diff(spans[50:]) < 0.0)
    # Above the minimum by 0.7 (50 / n)^(1/3): 2.6e-5 at 1e15 points.
    assert 0.3 < choose_lowess_span(10**15) < 0.3 + 3e-5
    assert choose_lowess_span(np.int64(400)) == choose_lowess_span(400)


@pytest.mark.parametrize(
    ("n", "settings", "error", "message"),
    [
        (-1, {}, ValueError, "n must be non-negative, got -1"),
        (400.0, {}, TypeError, "n must be an integer, got float"),
        (True, {}, TypeError, "n must be an integer, got bool"),
        (400, {"small_n": -1}, ValueError, "small_n must be non-negative, got -1"),
        (400, {"small_n": 50.5}, TypeError, "small_n must be an integer, got float"),
        (400, {"min_span": 0.0}, ValueError, r"min_span must be in \(0, 1\], got 0.0"),
        (400, {"min_span": 1.5}, ValueError, r"min_span must be in \(0, 1\], got 1.5"),
        (400, {"min_span": np.nan}, ValueError, r"min_span must be in \(0, 1\]"),
        (400, {"min_span": "0.3"}, TypeError, "min_span must be a real number"),
        (400, {"power": -0.5}, ValueError, r"power must be in \[0, 1\], got -0.5"),
        (400, {"power": 2.0}, ValueError, r"power must be in \[0, 1\], got 2.0"),
        (400, {"power": None}, TypeError, "power must be a real number"),
    ],
)
def test_lowess_span_rule_rejects_invalid_input(
    n: Any, settings: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        choose_lowess_span(n, **settings)


@pytest.mark.parametrize("case", SPLINE_CASES)
def test_natural_spline_basis_matches_r(
    case: str, spline_reference: dict[str, Any]
) -> None:
    data = spline_reference[case]
    basis = natural_spline_basis(data["x"], **data["keywords"])
    assert_same_column_space(basis, data["basis"])


@pytest.mark.parametrize("case", FAILING_SPLINE_CASES)
def test_natural_spline_basis_fails_where_r_fails(
    case: str, spline_reference: dict[str, Any]
) -> None:
    data = spline_reference[case]
    assert data["basis"] is None
    with pytest.raises(ValueError, match="knot"):
        natural_spline_basis(data["x"], **data["keywords"])


def truncated_power_basis(x: np.ndarray, knots: np.ndarray) -> np.ndarray:
    """Natural cubic splines with knots xi_1 < ... < xi_K, boundary ones included.

    The basis 1, x, d_k - d_(K-1) with d_k(x) = ((x - xi_k)_+^3 - (x -
    xi_K)_+^3) / (xi_K - xi_k), k = 1, ..., K - 2 (Hastie, Tibshirani and
    Friedman 2009, equations 5.4 and 5.5).
    """

    def d(k: int) -> np.ndarray:
        cube = np.maximum(x - knots[k], 0.0) ** 3 - np.maximum(x - knots[-1], 0.0) ** 3
        return cube / (knots[-1] - knots[k])

    columns = [np.ones_like(x), x] + [
        d(k) - d(knots.size - 2) for k in range(knots.size - 2)
    ]
    return np.column_stack(columns)


@pytest.mark.parametrize("knots", [[], [0.5], [0.2, 0.45, 0.8]])
def test_natural_spline_basis_spans_the_truncated_power_basis(knots: list) -> None:
    # Points on both sides beyond the boundary knots, where both are linear.
    x = np.random.default_rng(29).uniform(-0.4, 1.3, 90)
    basis = natural_spline_basis(
        x, knots=knots, intercept=True, boundary_knots=[0.0, 1.0]
    )
    reference = truncated_power_basis(x, np.array([0.0, *knots, 1.0]))
    assert_same_column_space(basis, reference)


@pytest.mark.parametrize("df", [1, 2, 4, 6])
def test_natural_spline_basis_without_intercept_vanishes_at_the_lower_knot(
    df: int,
) -> None:
    x = np.random.default_rng(31).exponential(1.0, 50)
    basis = natural_spline_basis(x, df=df)
    assert basis.shape == (50, df)
    assert np.linalg.matrix_rank(basis) == df
    lower = x.min()
    np.testing.assert_allclose(basis[np.argmin(x)], 0.0, atol=1e-15)
    # Lines through (lower, 0) lie in the space; constants do not.
    line = 0.7 * (x - lower)
    coefficients = np.linalg.lstsq(basis, line, rcond=None)[0]
    np.testing.assert_allclose(basis @ coefficients, line, rtol=0, atol=1e-12)
    ones = np.ones(50)
    residual = ones - basis @ np.linalg.lstsq(basis, ones, rcond=None)[0]
    assert np.linalg.norm(residual) > 0.1


@pytest.mark.parametrize("df", [2, 3, 5])
def test_natural_spline_basis_with_intercept_reproduces_lines(df: int) -> None:
    x = np.random.default_rng(37).normal(5.0, 2.0, 40)
    basis = natural_spline_basis(x, df=df, intercept=True)
    assert basis.shape == (40, df)
    assert np.linalg.matrix_rank(basis) == df
    line = 1.5 - 0.3 * x
    coefficients = np.linalg.lstsq(basis, line, rcond=None)[0]
    np.testing.assert_allclose(basis @ coefficients, line, rtol=0, atol=1e-12)


def test_quantile_knots_are_type_7_quantiles() -> None:
    x = np.arange(11.0)
    np.testing.assert_array_equal(
        natural_spline_basis(x, df=4), natural_spline_basis(x, knots=[2.5, 5.0, 7.5])
    )
    # Quantiles 0 and 1 of the values below: the knot on the lower boundary
    # knot moves to one eighth of the way to the next knot.
    tied = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(
        natural_spline_basis(tied, df=3), natural_spline_basis(tied, knots=[0.125, 1.0])
    )
    # Only values within the boundary knots count.
    wide = np.concatenate([[-50.0], np.arange(11.0), [60.0]])
    np.testing.assert_array_equal(
        natural_spline_basis(wide, df=4, boundary_knots=[0.0, 10.0]),
        natural_spline_basis(wide, knots=[2.5, 5.0, 7.5], boundary_knots=[0.0, 10.0]),
    )


def test_natural_spline_basis_rows_of_missing_values() -> None:
    x = np.random.default_rng(41).uniform(0.0, 3.0, 25)
    missing = x.copy()
    missing[[4, 9]] = np.nan
    basis = natural_spline_basis(missing, df=4, intercept=True)
    assert np.isnan(basis[[4, 9]]).all()
    kept = np.delete(np.arange(25), [4, 9])
    np.testing.assert_array_equal(
        basis[kept], natural_spline_basis(x[kept], df=4, intercept=True)
    )


def test_natural_spline_basis_reads_boundary_knots_in_either_order() -> None:
    x = np.linspace(-1.0, 2.0, 13)
    np.testing.assert_array_equal(
        natural_spline_basis(x, knots=[0.5], boundary_knots=[1.5, 0.0]),
        natural_spline_basis(x, knots=[0.5], boundary_knots=[0.0, 1.5]),
    )


@pytest.mark.parametrize(
    ("x", "keywords", "error", "message"),
    [
        ([[1.0, 2.0]], {"df": 2}, ValueError, "1-D array"),
        ([1.0, np.inf, 2.0], {"df": 2}, ValueError, "infinite"),
        (["a", "b"], {"df": 2}, TypeError, "x must hold real numbers"),
        ([1.0, 2.0, 3.0], {"df": 2.0}, TypeError, "df must be an integer"),
        ([1.0, 2.0, 3.0], {"df": 0}, ValueError, "at least 1 with intercept=False"),
        (
            [1.0, 2.0, 3.0],
            {"df": 1, "intercept": True},
            ValueError,
            "at least 2 with intercept=True",
        ),
        ([1.0, 2.0, 3.0], {"df": 2, "intercept": 1}, TypeError, "intercept must be"),
        ([1.0, 2.0, 3.0], {"df": 4, "knots": [2.0]}, ValueError, "disagrees with 1"),
        ([1.0, 2.0, 3.0], {"knots": [np.nan]}, ValueError, "knots must be finite"),
        ([1.0, 2.0, 3.0], {"knots": [1.0]}, ValueError, r"strictly inside .*\[1\.0\]"),
        ([1.0, 2.0, 3.0], {"knots": [3.5]}, ValueError, "strictly inside"),
        ([1.0, 2.0], {"boundary_knots": [1.0]}, ValueError, "two values"),
        ([1.0, 2.0], {"boundary_knots": [1.0, 1.0]}, ValueError, "distinct"),
        ([1.0, 2.0], {"boundary_knots": [0.0, np.nan]}, ValueError, "finite"),
        ([2.0, 2.0, 2.0], {"df": 1}, ValueError, "two distinct values"),
        ([np.nan, np.nan], {"df": 1}, ValueError, "no observed"),
        (
            [5.0, 6.0],
            {"df": 2, "boundary_knots": [0.0, 1.0]},
            ValueError,
            "no value of x lies within",
        ),
        ([0.0] * 6 + [1.0], {"df": 3}, ValueError, "lower boundary knot"),
    ],
)
def test_natural_spline_basis_rejects_invalid_input(
    x: Any, keywords: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message):
        natural_spline_basis(x, **keywords)


def test_natural_spline_basis_needs_no_x_when_knots_are_given() -> None:
    basis = natural_spline_basis([], knots=[0.5], boundary_knots=[0.0, 1.0])
    assert basis.shape == (0, 2)
    x = np.linspace(0.0, 1.0, 9)
    np.testing.assert_array_equal(
        natural_spline_basis(x, knots=0.5), natural_spline_basis(x, knots=[0.5])
    )


def test_weighted_lowess_of_three_distinct_values_gives_group_means() -> None:
    # With span 1 the window of each value reaches the farthest one, whose tricube
    # weight is zero: the line through two groups passes through their weighted
    # means, and the middle group is alone. So every point gets the weighted mean
    # of its group, as in limma's weightedLowess.
    rng = np.random.default_rng(1)
    x = np.repeat([0.0, 1.0, 2.0], 30)[rng.permutation(90)]
    y = rng.normal(size=90)
    w = rng.uniform(0.1, 2.0, 90)
    fitted = _weighted_lowess(x, y, w, 1.0)
    for level in (0.0, 1.0, 2.0):
        group = x == level
        mean = np.sum(w[group] * y[group]) / np.sum(w[group])
        np.testing.assert_allclose(fitted[group], mean, rtol=1e-13)


@pytest.mark.parametrize("span", [0.2, 0.5, 1.0])
def test_weighted_lowess_reproduces_a_line(span: float) -> None:
    # Weighted least squares fits a line exactly, and so does linear interpolation
    # between anchors, whatever the weights and the window: this checks the
    # arithmetic of the local fits, not the choice of windows or anchors.
    rng = np.random.default_rng(2)
    x = np.r_[rng.uniform(0.0, 10.0, 600), 3.0, 3.0, 3.0]
    w = rng.uniform(0.1, 3.0, x.size)
    np.testing.assert_allclose(
        _weighted_lowess(x, 2.0 - 0.5 * x, w, span), 2.0 - 0.5 * x, atol=1e-12
    )


def test_weighted_lowess_window_of_ties_holds_every_tie() -> None:
    # Windows grow by whole groups of tied points, so a window that reaches its
    # weight within the points tied with its anchor holds all of them; its radius
    # is zero, below 1e-7, so the fit is their prior-weighted mean (2.2, and
    # 8.5 / 6.5 for the ties at x = 2 below).
    y = np.array([1.0, 2.0, 4.0, 0.5, 3.0])
    w = np.array([1.0, 2.0, 0.5, 1.0, 3.0])
    np.testing.assert_allclose(_weighted_lowess(np.full(5, 2.0), y, w, 0.3), 2.2)
    x = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 3.0])
    fitted = _weighted_lowess(
        x, np.r_[0.0, y], np.r_[1.0, 2.0, 0.5, 1.0, 3.0, 1.0], 0.3
    )
    np.testing.assert_allclose(fitted, [0.0, *[8.5 / 6.5] * 4, 3.0], rtol=1e-15)


def test_weighted_lowess_narrow_windows_give_prior_weighted_means() -> None:
    # Span 0.5 of three unit weights: each window holds a point and one neighbour,
    # the right one when both are as near. Windows narrower than 1e-7 give the
    # prior-weighted mean of their points: (0 + 1) / 2, then (1 + 5) / 2 twice.
    # Spread 2^30 times wider, the neighbour at the window's edge gets a tricube
    # weight of zero, and each point keeps its own value.
    a = 2.0**-25  # about 3e-8
    x = np.array([0.0, a, 2.0 * a])
    y = np.array([0.0, 1.0, 5.0])
    w = np.ones(3)
    np.testing.assert_array_equal(_weighted_lowess(x, y, w, 0.5), [0.5, 3.0, 3.0])
    np.testing.assert_array_equal(_weighted_lowess(x * 2.0**30, y, w, 0.5), y)


def test_weighted_lowess_fits_a_mean_where_x_barely_varies() -> None:
    # With span 1, the window of the first point holds the three, at distances 0, s
    # and 2s = h: tricube weights 1, (7/8)^3 and 0. The fit there is the line
    # through the first two points, which passes through the first (0), unless
    # the weighted sum of squares of x about its mean, w1 w2 / (w1 + w2) s^2, is
    # below 1e-7, when it is their weighted mean. The bound is absolute, so it
    # depends on the scale of the weights as well as on s.
    second = 2.0 * (7.0 / 8.0) ** 3
    mean = second / (1.0 + second)
    sum_of_squares = second / (1.0 + second)  # times s^2 and the weights' scale
    y = np.array([0.0, 1.0, 5.0])
    for s, scale, expected in ((3e-4, 1.0, mean), (5e-4, 1.0, 0.0), (3e-4, 10.0, 0.0)):
        assert (sum_of_squares * s * s * scale < 1e-7) == (expected == mean)
        x = np.array([0.0, s, 2.0 * s])
        w = scale * np.array([1.0, 2.0, 1.5])
        assert _weighted_lowess(x, y, w, 1.0)[0] == pytest.approx(expected, abs=1e-12)


def test_anchor_spacing_leaves_out_large_gaps() -> None:
    # delta solves sum(min(gaps, delta)) = npts delta (found by probing limma): for
    # 1,000 equally spaced points it is the range over 200; for two clusters of
    # 150 points 1 apart and a gap of 1,001, 298 / 199, as limma's anchors show.
    assert _anchor_spacing(np.arange(1000.0), 200) == 999.0 / 200.0
    clusters = np.r_[np.arange(1.0, 151.0), np.arange(1151.0, 1301.0)]
    assert _anchor_spacing(clusters, 200) == 298.0 / 199.0
    # every distinct value is an anchor up to npts of them, ties counted once
    assert _anchor_spacing(np.repeat(np.arange(150.0), 3), 200) == 0.0
    # with npts + 1 values, the smallest gap: one pair shares an anchor
    x = np.sort(np.random.default_rng(3).uniform(0.0, 1.0, 201))
    assert _anchor_spacing(x, 200) == np.min(np.diff(x))


# Bases that keep their knots: R's rules, Harrell's rule, both forms.


@pytest.fixture(scope="module")
def spline_arrays() -> dict[str, np.ndarray]:
    """Every array of natural_splines.npz, by key."""
    with np.load(DATA / "natural_splines.npz") as data:
        return {key: data[key] for key in data.files}


@pytest.fixture(scope="module")
def hmisc() -> dict[str, np.ndarray]:
    """Every array of restricted_cubic_splines.npz (Hmisc and rms), by key."""
    with np.load(DATA / "restricted_cubic_splines.npz") as data:
        return {key: data[key] for key in data.files}


def all_knots(basis: NaturalSplineBasis) -> np.ndarray:
    """The knots of a basis, the boundary knots included, in order."""
    lower, upper = basis.boundary_knots
    return np.array([lower, *basis.knots, upper])


def basis_of(data: dict[str, Any], **options: Any) -> NaturalSplineBasis:
    """The basis that natural_spline_basis builds for a fixture case."""
    keywords = dict(data["keywords"])
    knots = keywords.pop("knots", "quantiles")
    return NaturalSplineBasis(data["x"], knots=knots, **keywords, **options)


def test_quantile_knots_are_rs_to_the_last_bit(
    spline_reference: dict[str, Any], spline_arrays: dict[str, np.ndarray]
) -> None:
    # R's quantile arithmetic, with its position counted from 1: counted from 0, the
    # knots of two_distinct_values and of normal150 at df 7 (below) move in the last
    # bit.
    compared = 0
    for name, data in spline_reference.items():
        keywords = data["keywords"]
        if data["basis"] is None or keywords["df"] is None or "knots" in keywords:
            continue
        basis = basis_of(data)
        assert_array_equal(basis.knots, spline_arrays[f"{name}__r_knots"], name)
        assert_array_equal(
            basis.boundary_knots, spline_arrays[f"{name}__r_boundary_knots"], name
        )
        compared += 1
    assert compared == 18


@pytest.mark.parametrize("rule", ["quantiles", "levels"])
def test_knots_of_seeded_datasets_are_rs(
    rule: str, spline_arrays: dict[str, np.ndarray]
) -> None:
    # attr(ns(x, df), "knots") and attr(ns(unique(x), df), "knots") at df 2 to 8, with
    # and without an intercept, on samples, dose ladders and ties; where R fails, so
    # must CytePool.
    key = "knots" if rule == "quantiles" else "levels"
    compared = failed = 0
    for name in spline_arrays["knot_datasets"]:
        x = spline_arrays[f"knots__{name}__x"]
        for df in range(2, 9):
            for intercept in (False, True):
                prefix = f"knots__{name}__df{df}_{int(intercept)}"
                if f"{prefix}__{key}" not in spline_arrays:
                    with pytest.raises(ValueError, match="knot"):
                        NaturalSplineBasis(x, df, knots=rule, intercept=intercept)
                    failed += 1
                    continue
                basis = NaturalSplineBasis(x, df, knots=rule, intercept=intercept)
                label = f"{name} df={df} intercept={intercept}"
                assert_array_equal(
                    basis.knots, spline_arrays[f"{prefix}__{key}"], label
                )
                if rule == "quantiles":
                    boundary = spline_arrays[f"{prefix}__boundary"]
                    assert_array_equal(basis.boundary_knots, boundary, label)
                compared += 1
    assert compared + failed == 9 * 14 and compared > 100


def test_level_knots_are_rs_on_the_distinct_values(
    spline_reference: dict[str, Any], spline_arrays: dict[str, np.ndarray]
) -> None:
    compared = 0
    for name, data in spline_reference.items():
        if f"{name}__r_levels" not in spline_arrays:
            continue
        basis = basis_of(data | {"keywords": data["keywords"] | {"knots": "levels"}})
        assert_array_equal(basis.knots, spline_arrays[f"{name}__r_levels"], name)
        compared += 1
    assert compared >= 15


def test_the_basis_at_new_points_matches_rs_predict(
    spline_reference: dict[str, Any], spline_arrays: dict[str, np.ndarray]
) -> None:
    # Random responses fitted at x on either basis, and predicted at new points: inside
    # the boundary knots, at the knots, beyond both boundary knots, and NaN.
    rng = np.random.default_rng(13)
    compared = 0
    for name, data in spline_reference.items():
        if f"{name}__r_newx" not in spline_arrays:
            continue
        train = data["basis"]
        complete = ~np.isnan(train).any(axis=1)
        if np.linalg.matrix_rank(train[complete]) < train.shape[1]:
            continue
        basis = basis_of(data)
        newx = spline_arrays[f"{name}__r_newx"]
        responses = rng.normal(0.0, 1.0, (int(complete.sum()), 2))
        r_fit = np.linalg.lstsq(train[complete], responses, rcond=None)[0]
        fit = np.linalg.lstsq(basis.matrix(data["x"])[complete], responses, rcond=None)[
            0
        ]
        expected = spline_arrays[f"{name}__r_newx_basis"] @ r_fit
        predicted = basis.matrix(newx) @ fit
        missing = np.isnan(expected[:, 0])
        assert_array_equal(np.isnan(predicted[:, 0]), missing, name)
        assert newx[~missing].min() < basis.boundary_knots[0] < basis.boundary_knots[1]
        assert newx[~missing].max() > basis.boundary_knots[1]
        scale = np.max(np.abs(expected[~missing]))
        assert_allclose(
            predicted[~missing],
            expected[~missing],
            rtol=0,
            atol=1e-10 * scale,
            err_msg=name,
        )
        compared += 1
    assert compared >= 20


# Datasets that tie at an extreme in 5% of their values or more, with no value between
# the extremes as often (derived by hand from each dataset's design): Hmisc's help page
# says that it then places the knots by another method, which CytePool refuses.
TIED_AT_AN_EXTREME = {
    "ties_at_min_10pct",  # 10 zeros of 100
    "ties_at_min_5pct",  # 10 zeros of 200, at the threshold
    "ties_at_max_10pct",
    "ties_at_both_ends",
    "ties_at_min_n46",  # 6 zeros of 46
    "ties_at_min_few_values",  # 20 zeros, then 1 to 5 once each
    "ties_at_max_few_values",
    "ties_at_min_seven_values",  # 30 zeros, then 1 to 6 once each
}
# 9 zeros of 200 (4.5%) and ties between the extremes: Harrell's percentiles apply.
NOT_TIED = {"ties_at_min_4.5pct", "ties_at_min_and_between"}


def test_harrell_knots_match_hmisc(hmisc: dict[str, np.ndarray]) -> None:
    assert TIED_AT_AN_EXTREME | NOT_TIED <= set(hmisc["harrell_datasets"])
    identical = refused_ties = 0
    for name in hmisc["harrell_datasets"]:
        x = hmisc[f"harrell__{name}__x"]
        m = np.unique(x[~np.isnan(x)]).size
        for nk in range(2, 10):
            hmisc_knots = hmisc.get(f"harrell__{name}__nk{nk}")
            tied = name in TIED_AT_AN_EXTREME and 3 <= nk < m - 2
            label = f"{name} nk={nk}"
            try:
                basis = NaturalSplineBasis(x, nk - 1, knots="harrell")
            except ValueError as error:
                # Refused only where Hmisc fails, places fewer knots than asked, or
                # switches to its method for tied values.
                assert hmisc_knots is None or hmisc_knots.size < nk or tied, label
                if tied:
                    assert "tie at the" in str(error), label
                    refused_ties += 1
                continue
            assert not tied, label
            assert hmisc_knots is not None, label
            assert_array_equal(all_knots(basis), hmisc_knots, label)
            identical += 1
    assert identical >= 80 and refused_ties >= 25


def test_harrell_knots_of_small_samples_and_ladders(
    hmisc: dict[str, np.ndarray],
) -> None:
    # Fewer than 100 values: the outer knots are the 5th smallest and largest values,
    # sorted with the percentiles; 100 values take the percentiles. At nk = m - 2 the
    # knots are the distinct values but the extremes. The ladder at 5 knots, Hmisc's
    # default, needs the probabilities as R's seq() computes them.
    small = hmisc["harrell__uniform12__x"]
    knots = all_knots(NaturalSplineBasis(small, 4, knots="harrell"))
    ordered = np.sort(small)
    # Here the 5th smallest value lies above the 27.5th percentile, which the sort
    # makes the lower boundary knot, as in Hmisc.
    assert {ordered[4], ordered[-5]} <= set(knots) and knots[0] < ordered[4]
    assert_array_equal(knots, hmisc["harrell__uniform12__nk5"])
    large = hmisc["harrell__exponential100__x"]
    assert (
        NaturalSplineBasis(large, 2, knots="harrell").boundary_knots[0]
        != np.sort(large)[4]
    )
    five = np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 4)
    assert NaturalSplineBasis(five, 2, knots="harrell").knots == (3.0,)
    ladder = NaturalSplineBasis(hmisc["harrell__ladder__x"], 4, knots="harrell")
    assert_array_equal(all_knots(ladder), hmisc["harrell__ladder__nk5"])


@pytest.mark.parametrize(
    ("x", "df", "message"),
    [
        (np.r_[np.zeros(10), np.linspace(0.1, 1.0, 90)], 4, "10.0% of the values of x tie at the smallest value, 0.0"),
        (np.r_[np.linspace(0.0, 0.9, 90), np.ones(10)], 4, "tie at the largest value, 1.0"),
        (np.r_[np.zeros(10), np.linspace(0.1, 0.9, 80), np.ones(10)], 3, "tie at the smallest value"),
        (np.repeat([1.0, 2.0, 3.0, 4.0], 10), 2, "3 knots strictly inside the range of x only among at least 5 distinct values, and x has 4; Hmisc would place 2"),
        (np.linspace(0.0, 1.0, 9), 7, "x has 9; Hmisc would place 7"),
        # The 5th smallest value and the median are both 5.
        (np.r_[1.0, 2.0, 3.0, 4.0, np.full(10, 5.0), np.arange(6.0, 12.0)], 2, "two knots at 5.0, where Hmisc would drop one"),
        (np.linspace(0.0, 1.0, 50), 1, "at least 3 knots, so df must be at least 2 with intercept=False, got 1"),
    ],
)  # fmt: skip
def test_harrell_refuses_where_hmisc_departs(
    x: np.ndarray, df: int, message: str
) -> None:
    with pytest.raises(ValueError, match="Harrell's rule|tie at the"):
        NaturalSplineBasis(x, df, knots="harrell")
    with pytest.raises(ValueError) as error:
        NaturalSplineBasis(x, df, knots="harrell")
    assert message in str(error.value)


@pytest.mark.parametrize("form", ["b_spline", "restricted_cubic"])
def test_harrell_with_an_intercept_places_df_knots(
    form: str, hmisc: dict[str, np.ndarray]
) -> None:
    # ols(y ~ rcs(x, nk)) has an intercept and nk - 1 spline columns, nk in all: with
    # intercept=True, df = nk gives rms's nk knots and nk columns.
    for name in hmisc["rms_cases"]:
        x, nk = hmisc[f"rms__{name}__x"], int(hmisc[f"rms__{name}__nk"])
        basis = NaturalSplineBasis(x, nk, knots="harrell", intercept=True, form=form)
        assert basis.df == nk and basis.matrix(x).shape == (x.size, nk), name
        assert_array_equal(all_knots(basis), hmisc[f"rms__{name}__knots"], name)


def test_restricted_cubic_columns_match_hmisc(hmisc: dict[str, np.ndarray]) -> None:
    # rcspline.eval(x, knots, inclx = TRUE): x, whose counterpart here is x minus the
    # lower boundary knot, then Harrell's terms, which with an intercept span the same
    # curves as the columns here; 3 to 8 knots, points to 50, far beyond the last knot.
    compared = 0
    for name in hmisc["restricted_cubic_cases"]:
        knots = hmisc[f"rcs__{name}__knots"]
        newx = hmisc[f"rcs__{name}__newx"]
        if name == "repeated_knot":
            with pytest.raises(ValueError, match="takes distinct knots"):
                NaturalSplineBasis(
                    knots=knots[1:-1],
                    boundary_knots=knots[[0, -1]],
                    form="restricted_cubic",
                )
            continue
        ours = NaturalSplineBasis(
            knots=knots[1:-1], boundary_knots=knots[[0, -1]], form="restricted_cubic"
        ).matrix(newx)
        reference = hmisc[f"rcs__{name}__basis"]
        present = ~np.isnan(reference[:, 0])
        assert_allclose(
            ours[present, 0] + knots[0],
            reference[present, 0],
            rtol=0,
            atol=1e-12 * np.max(np.abs(reference[present, 0])),
            err_msg=name,
        )
        ones = np.ones((present.sum(), 1))
        assert_spans_the_columns(
            np.hstack([ones, ours[present]]), np.hstack([ones, reference[present]])
        )
        assert_array_equal(np.isnan(ours), np.isnan(reference))
        compared += 1
    assert compared == 7


def test_restricted_cubic_values_by_hand() -> None:
    # Knots 1, 2 and 3: the first column is x - 1, and the second is a combination of
    # x - 1 and Harrell's term N_1(x) = ((x - 1)_+^3 - 2 (x - 2)_+^3 + (x - 3)_+^3) / 4,
    # which is 0 up to 1, 0.5^3 / 4 at 1.5, (1.5^3 - 2 x 0.5^3) / 4 at 2.5 and
    # (27 - 16 + 1) / 4 = 3 at 4, with a share of N_1 that is not 0.
    x = np.array([0.0, 1.0, 1.5, 2.5, 4.0])
    line_and_term = np.column_stack([x - 1.0, [0.0, 0.0, 0.03125, 0.78125, 3.0]])
    for intercept in (False, True):
        basis = NaturalSplineBasis(
            knots=[2.0],
            boundary_knots=[1.0, 3.0],
            intercept=intercept,
            form="restricted_cubic",
        )
        columns = basis.matrix(x)
        assert columns.shape == (5, 2 + intercept)
        if intercept:
            assert_array_equal(columns[:, 0], 1.0)
            columns = columns[:, 1:]
        assert_array_equal(columns[:, 0], x - 1.0)
        shares = np.linalg.lstsq(line_and_term, columns[:, 1], rcond=None)[0]
        assert_allclose(line_and_term @ shares, columns[:, 1], rtol=0, atol=1e-14)
        assert abs(shares[1]) > 0.5


def test_restricted_cubic_columns_stay_well_conditioned_where_knots_crowd() -> None:
    # Doses from 0.01 to 30 on a linear scale crowd the knots below 1. Harrell's
    # truncated cubes are nearly collinear there: with an intercept, the smallest
    # eigenvalue of the correlation of their coefficients falls to 4.9e-9 of the
    # largest at df 3 and to 0 at df 7, below the 1e-8 under which limma's F-tests drop
    # a dimension. The columns here keep it above 2.5e-3.
    doses = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    for df in range(1, 8):
        basis = NaturalSplineBasis(doses, df, knots="levels", form="restricted_cubic")
        design = np.column_stack([np.ones(doses.size), basis.matrix(doses)])
        covariance = np.linalg.inv(design.T @ design)[1:, 1:]
        scale = np.sqrt(np.diag(covariance))
        eigenvalues = np.linalg.eigvalsh(covariance / np.outer(scale, scale))
        assert eigenvalues[0] > 1e-3 * eigenvalues[-1], df


def f_statistics(
    design: np.ndarray, y: np.ndarray, dropped: list[list[int]]
) -> list[float]:
    """F-statistics of least squares fits that drop each list of columns."""

    def rss(columns: np.ndarray) -> float:
        residuals = y - columns @ np.linalg.lstsq(columns, y, rcond=None)[0]
        return float(residuals @ residuals)

    full, df_residual = rss(design), y.size - design.shape[1]
    return [
        ((rss(np.delete(design, columns, axis=1)) - full) / len(columns))
        / (full / df_residual)
        for columns in dropped
    ]


def test_spline_f_tests_match_rms(hmisc: dict[str, np.ndarray]) -> None:
    # anova(ols(y ~ rcs(x, nk))): the overall F of all spline columns, the non-linear F
    # of the columns after the first, and the overall F of the B-spline form on the
    # same knots, since both forms span the same curves.
    for name in hmisc["rms_cases"]:
        x, y = hmisc[f"rms__{name}__x"], hmisc[f"rms__{name}__y"]
        nk = int(hmisc[f"rms__{name}__nk"])
        overall, nonlinear = hmisc[f"rms__{name}__anova"][:2]
        basis = NaturalSplineBasis(x, nk - 1, knots="harrell", form="restricted_cubic")
        assert_array_equal(all_knots(basis), hmisc[f"rms__{name}__knots"])
        complete = ~np.isnan(x)
        design = np.column_stack([np.ones(complete.sum()), basis.matrix(x[complete])])
        found = f_statistics(
            design, y[complete], [list(range(1, nk)), list(range(2, nk))]
        )
        assert_allclose(found, [overall, nonlinear], rtol=1e-10, err_msg=name)
        b_spline = NaturalSplineBasis(
            knots=basis.knots, boundary_knots=basis.boundary_knots
        )
        design = np.column_stack(
            [np.ones(complete.sum()), b_spline.matrix(x[complete])]
        )
        found = f_statistics(design, y[complete], [list(range(1, nk))])
        assert_allclose(found, [overall], rtol=1e-10, err_msg=name)


@pytest.mark.parametrize("form", ["b_spline", "restricted_cubic"])
@pytest.mark.parametrize("knots", [[], [0.5], [0.2, 0.45, 0.8]])
def test_both_forms_span_the_truncated_power_basis(form: str, knots: list) -> None:
    x = np.random.default_rng(43).uniform(-0.4, 1.3, 90)
    basis = NaturalSplineBasis(
        knots=knots, boundary_knots=[0.0, 1.0], intercept=True, form=form
    )
    reference = truncated_power_basis(x, np.array([0.0, *knots, 1.0]))
    assert_same_column_space(basis.matrix(x), reference)


@pytest.mark.parametrize("form", ["b_spline", "restricted_cubic"])
def test_columns_vanish_at_the_lower_knot_and_are_linear_beyond(form: str) -> None:
    x = np.random.default_rng(47).normal(0.0, 1.0, 40)
    basis = NaturalSplineBasis(x, 4, form=form)
    lower, upper = basis.boundary_knots
    assert_allclose(basis.matrix([lower]), 0.0, atol=1e-15)
    # Second differences of equally spaced points vanish below the lower boundary knot
    # and beyond the upper one, where every column is a line.
    for start, step in ((lower - 3.0, 0.5), (upper + 0.01, 0.5)):
        columns = basis.matrix(start + step * np.arange(6))
        second = np.diff(columns, n=2, axis=0)
        assert np.max(np.abs(second)) <= 1e-11 * np.max(np.abs(columns)), start
    # Within the knots the columns are not lines.
    inside = basis.matrix(np.linspace(lower, upper, 6))
    assert np.max(np.abs(np.diff(inside, n=2, axis=0))) > 1e-3


def test_rows_of_missing_values_and_points_beyond_the_data() -> None:
    x = np.random.default_rng(53).uniform(0.0, 3.0, 25)
    basis = NaturalSplineBasis(x, 3, form="restricted_cubic")
    columns = basis.matrix([np.nan, 1.0, -10.0, 100.0])
    assert np.isnan(columns[0]).all() and np.isfinite(columns[1:]).all()
    with pytest.raises(ValueError, match="infinite"):
        basis.matrix([1.0, np.inf])
    with pytest.raises(TypeError, match="x must hold real numbers"):
        basis.matrix(["a"])
    with pytest.raises(ValueError, match="1-D"):
        basis.matrix([[1.0]])


@pytest.mark.parametrize(
    ("x", "keywords", "error", "message"),
    [
        ([1.0, 2.0, 3.0], {"knots": "splines"}, ValueError, "knots must be a rule, one of 'quantiles', 'levels', 'harrell', or the interior knots, got 'splines'"),
        ([1.0, 2.0, 3.0], {"knots": None}, TypeError, "knots must be a rule"),
        ([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], {"df": 2, "knots": "harrell", "boundary_knots": [0, 7]}, ValueError, "places the boundary knots itself"),
        ([1.0, 2.0, 3.0], {"knots": "harrell"}, ValueError, "needs x, whose percentiles place the knots, and df"),
        (None, {"df": 3}, ValueError, "places the knots on the values of x; give x"),
        (None, {"knots": [0.5]}, ValueError, "give x or boundary_knots"),
        (None, {"knots": [0.5, 0.5], "boundary_knots": [0, 1], "form": "restricted_cubic"}, ValueError, "takes distinct knots, and the knots given repeats 0.5"),
        (np.repeat([0.0, 1.0, 2.0, 3.0], [1, 6, 1, 1]), {"df": 3, "form": "restricted_cubic"}, ValueError, "R's quantile rule repeats 1.0"),
        ([1.0, 2.0, 3.0], {"form": "b-spline"}, ValueError, "form must be one of 'b_spline', 'restricted_cubic'"),
        ([1.0, 2.0, 3.0], {"intercept": 1}, TypeError, "intercept must be True or False"),
        ([1.0, 2.0, 3.0], {"df": 2.0}, TypeError, "df must be an integer"),
        ([1.0, 2.0, 3.0], {"df": 1, "intercept": True}, ValueError, "at least 2 with intercept=True"),
        ([1.0, np.inf], {}, ValueError, "infinite"),
    ],
)  # fmt: skip
def test_rules_and_arguments_are_refused_with_a_fix(
    x: Any, keywords: dict[str, Any], error: type[Exception], message: str
) -> None:
    with pytest.raises(error, match=message.replace("(", r"\(").replace(")", r"\)")):
        NaturalSplineBasis(x, **keywords)


@pytest.mark.parametrize("form", ["b_spline", "restricted_cubic"])
def test_a_basis_survives_json_and_pickle(form: str) -> None:
    # With an intercept and the restricted cubic form, a record that lost either
    # would rebuild another basis.
    x = np.random.default_rng(59).normal(0.0, 1.0, 30)
    basis = NaturalSplineBasis(x, 4, knots="levels", intercept=True, form=form)
    record = json.loads(json.dumps(basis.to_dict()))
    assert set(record) == {"knots", "boundary_knots", "intercept", "form"}
    rebuilt = NaturalSplineBasis(**record)
    assert rebuilt == basis and hash(rebuilt) == hash(basis)
    points = np.linspace(-4.0, 4.0, 17)
    assert_array_equal(rebuilt.matrix(points), basis.matrix(points))
    unpickled = pickle.loads(pickle.dumps(basis))
    assert unpickled == basis
    assert_array_equal(unpickled.matrix(points), basis.matrix(points))
    # The repr names the knots and rebuilds the basis.
    assert eval(repr(basis), {"NaturalSplineBasis": NaturalSplineBasis}) == basis
    assert NaturalSplineBasis(**(record | {"intercept": False})) != basis
    other_form = "b_spline" if form == "restricted_cubic" else "restricted_cubic"
    assert NaturalSplineBasis(**(record | {"form": other_form})) != basis
    with pytest.raises(dataclasses.FrozenInstanceError):
        basis.knots = ()  # type: ignore[misc]


def test_natural_spline_basis_takes_no_rule() -> None:
    # R's fixtures pin the function's columns; here, rules are the class's: the
    # function keeps its knots as values or none.
    with pytest.raises(TypeError, match=r"NaturalSplineBasis\(x, df, knots='levels'\)"):
        natural_spline_basis(np.arange(10.0), 3, knots="levels")
    # Without df or knots, no interior knots, as R's ns(x) gives (case no_knots).
    assert NaturalSplineBasis(np.arange(10.0)).knots == ()
    assert NaturalSplineBasis(np.arange(10.0), knots="levels").df == 1


def test_the_examples_in_the_docstrings_run() -> None:
    assert doctest.testmod(smoothing, raise_on_error=False).failed == 0
