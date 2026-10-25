"""Tests for LOWESS, its span rule and natural cubic spline bases.

Reference values in ``tests/data/smoothing`` come from R's ``stats::lowess``
and ``splines::ns``, written by ``validation/smoothing.py``. The span rule is
checked here against values derived by hand, and in ``tests/test_voom.py``
against limma's ``chooseLowessSpan``.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from cytepool.stats.smoothing import choose_lowess_span, lowess, natural_spline_basis

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
