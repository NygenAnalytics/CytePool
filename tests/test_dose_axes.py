"""Tests for dose axes: positions, spline bases on doses and design columns.

Reference values in ``tests/data/dose_axes`` come from R (``splines::ns``,
``model.matrix``, ``lm``, ``anova`` and ``predict``) and Hmisc's ``rcspline.eval``,
written by ``validation/dose_axes.py``. Positions are checked against values derived
by hand; the meaning of ``dose_treated`` against the means of the dose groups; and the
tests of a saturated spline against a factor of dose groups and the lack-of-fit F,
each fitted independently by least squares.
"""

import doctest
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import Design, DoseAxis, Table, dose_axes, from_bulk, limma_trend
from cytepool.stats.smoothing import NaturalSplineBasis

DATA = Path(__file__).parent / "data" / "dose_axes"
CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/linear_models.md"
)
# A ladder like the TCDD liver series: the vehicle and eight doses, 2 or 3 each.
TCDD = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
LADDER = np.repeat(np.r_[0.0, TCDD], [4, 2, 3, 3, 2, 3, 2, 3, 3])
FORMS = ("restricted_cubic", "b_spline")
# The cases of dose_axes.npz where DoseAxis raises and R does not.
TIED_KNOTS = (
    "unbalanced_ladder | own group, log10 | restricted_cubic | quantiles | df 3"
)
TIED_VALUES = (
    "exposures | on axis, log10 with offset | restricted_cubic | harrell | df 4"
)


def ug_per_kg(**options: Any) -> DoseAxis:
    return DoseAxis("dose", unit="ug/kg", metric="administered", **options)


def table_of(doses: np.ndarray) -> Table:
    return Table({"sample_id": [f"S{i}" for i in range(len(doses))], "dose": doses})


def matrix_of(columns: dict[str, np.ndarray]) -> np.ndarray:
    """The design of the columns, with an intercept first."""
    first = next(iter(columns.values()))
    return np.column_stack([np.ones(first.size), *columns.values()])


def residual_sum_of_squares(design: np.ndarray, y: np.ndarray) -> np.ndarray:
    residuals = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    return np.sum(residuals**2, axis=0)


def f_statistic(full: np.ndarray, reduced: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The F-statistic of a nested pair of least squares fits."""
    rank_full, rank_reduced = (np.linalg.matrix_rank(m) for m in (full, reduced))
    rss_full = residual_sum_of_squares(full, y)
    extra = (residual_sum_of_squares(reduced, y) - rss_full) / (
        rank_full - rank_reduced
    )
    return extra / (rss_full / (y.shape[0] - rank_full))


def same_column_space(ours: np.ndarray, reference: np.ndarray) -> float:
    """Largest difference of least squares fits of random responses on either
    matrix, relative to the largest fit."""
    responses = np.random.default_rng(5).normal(0.0, 1.0, (ours.shape[0], 3))

    def fits(matrix: np.ndarray) -> np.ndarray:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        u = u[:, : int(np.count_nonzero(s > s[0] * 1e-10))]
        return u @ (u.T @ responses)

    expected = fits(reference)
    return float(np.max(np.abs(fits(ours) - expected)) / np.max(np.abs(expected)))


@pytest.fixture(scope="module")
def r_cases() -> list[dict[str, Any]]:
    """The cases of dose_axes.npz: the axis, doses, responses, rule, and R's values."""
    with np.load(DATA / "dose_axes.npz") as data:
        cases = []
        for i, name in enumerate(data["names"]):
            prefix = f"case{i}__"
            case = {
                key.removeprefix(prefix): data[key]
                for key in data.files
                if key.startswith(prefix)
            }
            cases.append(case | {"name": str(name)})
    return cases


def axis_of(case: dict[str, Any]) -> DoseAxis:
    offset = float(case["offset"])
    return DoseAxis(
        "dose",
        unit=str(case["unit"]),
        metric=str(case["metric"]),
        scale=str(case["scale"]),
        vehicle=str(case["vehicle"]),
        offset=None if np.isnan(offset) else offset,
    )


def basis_of(case: dict[str, Any]) -> NaturalSplineBasis:
    rule = str(case["rule"])
    return axis_of(case).spline_basis(
        table_of(case["doses"]),
        int(case["df"]),
        knots=case["given"] if rule == "given" else rule,
        form=str(case["form"]),
    )


# Positions.


def test_dose_positions_by_hand() -> None:
    doses = np.array([0.0, 10.0, 100.0])
    own_group = ug_per_kg()
    assert_array_equal(own_group.positions(doses), [np.nan, 1.0, 2.0])
    # The offset is added to every dose, the vehicle's included.
    on_axis = ug_per_kg(vehicle="on_axis", offset=1.0)
    assert_allclose(
        on_axis.positions(doses), [0.0, np.log10(11.0), np.log10(101.0)], rtol=1e-15
    )
    assert_array_equal(ug_per_kg(scale="linear").positions(doses), [np.nan, 10, 100])
    linear = ug_per_kg(scale="linear", vehicle="on_axis")
    assert_array_equal(linear.positions(doses), doses)
    # doses() inverts positions(); NaN stays NaN.
    assert_allclose(own_group.doses([np.nan, 1.0, 2.0]), [np.nan, 10.0, 100.0])
    assert_allclose(on_axis.doses(on_axis.positions(doses)), doses, atol=1e-13)
    assert_array_equal(linear.doses(doses), doses)
    assert own_group.positions([3]).dtype == np.float64


# Columns, designs, tests and curves against R.


def test_design_columns_match_r(r_cases: list[dict[str, Any]]) -> None:
    compared = 0
    for case in r_cases:
        if case["name"] in (TIED_KNOTS, TIED_VALUES):
            continue
        axis, samples, name = axis_of(case), table_of(case["doses"]), case["name"]
        assert_allclose(axis.positions(case["doses"]), case["positions"], rtol=1e-12)
        basis = basis_of(case)
        lower, upper = basis.boundary_knots
        knots = np.array([lower, *basis.knots, upper])
        scale = np.max(np.abs(case["knots"]))
        assert_allclose(knots, case["knots"], rtol=0, atol=1e-12 * scale, err_msg=name)
        columns = axis.design_columns(samples, basis)
        names = list(columns)
        ours = np.column_stack(list(columns.values()))
        reference = case["columns"]
        assert ours.shape == reference.shape, name
        spline = slice(None)
        if axis.vehicle == "own_group":
            assert names[0] == "dose_treated"
            assert_array_equal(columns["dose_treated"], case["treated"])
            assert_array_equal(reference[:, 0], case["treated"])
            spline = slice(1, None)
        design = Design(samples.with_columns(columns), "~ " + " + ".join(names))
        assert design.columns == ("(Intercept)", *names)
        if case["form"] == "restricted_cubic":
            # Element by element, relative to each column's largest value.
            largest = np.max(np.abs(reference), axis=0)
            assert np.max(np.abs(ours - reference) / largest) <= 1e-10, name
            assert np.max(np.abs(design.matrix - case["design"])) <= 1e-10 * np.max(
                np.abs(case["design"])
            )
        else:
            # The spline columns alone: the vehicle's rows hold 0 in both, and the
            # column space of the whole design does not depend on them.
            assert same_column_space(ours[:, spline], reference[:, spline]) <= 1e-10
            assert same_column_space(design.matrix, case["design"]) <= 1e-10
        # F-statistics of nested fits: any dose, among the treated doses, non-linear.
        splines = [j + 1 for j, n in enumerate(names) if n.startswith("dose_s")]
        dropped = {
            "any_dose": list(range(1, len(names) + 1)),
            "among_doses": splines,
            "nonlinear": splines[1:],
        }
        found = np.column_stack(
            [
                f_statistic(
                    design.matrix,
                    np.delete(design.matrix, dropped[test], axis=1),
                    case["y"],
                )
                for test in case["tests"]
            ]
        )
        assert_allclose(found, case["f"], rtol=1e-10, err_msg=name)
        # The fitted curves on a grid of doses: 0, each dose, between them, beyond.
        grid = matrix_of(axis.design_columns({"dose": case["grid"]}, basis))
        coefficients = np.linalg.lstsq(design.matrix, case["y"], rcond=None)[0]
        curves = grid @ coefficients
        scale = np.max(np.abs(case["curves"]))
        assert_allclose(
            curves, case["curves"], rtol=0, atol=1e-10 * scale, err_msg=name
        )
        compared += 1
    assert compared == 8


def test_coinciding_knots_are_refused_where_r_repeats_one(
    r_cases: list[dict[str, Any]],
) -> None:
    # 10 of the 16 treated samples have 0.1 mg/kg: R's rule places the knots at 1/3
    # and 2/3 of them both there, and ns() takes the repeated knot.
    (case,) = [c for c in r_cases if c["name"] == TIED_KNOTS]
    assert np.unique(case["knots"]).size < case["knots"].size
    message = (
        "the knots that R's quantile rule places on the doses coincide at 0.1 mg/kg"
    )
    for form in FORMS:
        with pytest.raises(ValueError, match=re.escape(message)):
            basis_of(case | {"form": np.array(form)})
    # On the distinct doses, the knots cannot coincide.
    levels = axis_of(case).spline_basis(table_of(case["doses"]), 3)
    assert len(set(levels.knots)) == 2


def test_hmisc_method_for_tied_values_is_refused(r_cases: list[dict[str, Any]]) -> None:
    # The 6 zeros of 30 exposures sit at log10(0.01) on the axis: Hmisc places 5 knots
    # by its method for values tied at an extreme, which DoseAxis refuses.
    (case,) = [c for c in r_cases if c["name"] == TIED_VALUES]
    assert str(case["rule"]) == "harrell"
    assert case["knots"].size == int(case["df"]) + 1
    with pytest.raises(ValueError, match="tie at the smallest value") as error:
        basis_of(case)
    assert "x is log10 of the doses plus 0.01 ng/mL" in str(error.value)


# What the columns mean.


def test_treated_is_the_lowest_dose_against_the_vehicle(
    r_cases: list[dict[str, Any]],
) -> None:
    # With a column for every dose group (df = 7 for 8 doses), the coefficient of
    # dose_treated is the mean of the lowest dose minus the vehicle's, because every
    # spline column is 0 at the lowest dose and for the vehicle.
    y = np.random.default_rng(67).normal(0.0, 1.0, LADDER.size) + np.log1p(LADDER)
    axis, samples = ug_per_kg(), table_of(LADDER)
    expected = y[LADDER == 0.01].mean() - y[LADDER == 0.0].mean()
    for form in FORMS:
        columns = axis.design_columns(samples, axis.spline_basis(samples, 7, form=form))
        coefficients = np.linalg.lstsq(matrix_of(columns), y, rcond=None)[0]
        assert_allclose(coefficients[1], expected, rtol=1e-12, err_msg=form)
    # Below saturation it is the jump from the vehicle to the curve at the lowest
    # dose: the same in both forms, and in R's columns.
    case = next(c for c in r_cases if "levels | df 3" in c["name"])
    r_coefficients = np.linalg.lstsq(case["design"], case["y"], rcond=None)[0]
    samples = table_of(case["doses"])
    for form in FORMS:
        basis = axis_of(case).spline_basis(samples, 3, form=form)
        columns = axis_of(case).design_columns(samples, basis)
        coefficients = np.linalg.lstsq(matrix_of(columns), case["y"], rcond=None)[0]
        assert_allclose(coefficients[1], r_coefficients[1], rtol=1e-10, err_msg=form)


def dose_series(seed: int) -> Any:
    """A bulk cube of the ladder, 300 genes, 60 of which rise with log dose."""
    rng = np.random.default_rng(seed)
    effect = np.log10(np.where(LADDER > 0.0, LADDER, 0.003)) + 2.5
    means = rng.uniform(20.0, 500.0, 300)
    log_fold = np.outer(effect, np.r_[rng.normal(0.0, 0.5, 60), np.zeros(240)])
    counts = rng.poisson(means * np.exp(log_fold))
    samples = {
        "sample_id": [f"S{i}" for i in range(LADDER.size)],
        "dose": LADDER,
        "dose_group": [f"d{dose:g}" for dose in LADDER],
    }
    return from_bulk(counts, samples=samples)


@pytest.mark.parametrize("vehicle", ["own_group", "on_axis"])
def test_a_saturated_spline_is_the_dose_group_test(vehicle: str) -> None:
    # With df one less than the distinct doses on the axis, the columns span every
    # dose group, so the F-test of all of them is the F-test of a factor of dose
    # groups against the vehicle, gene by gene, whatever the vehicle's rows hold.
    cube = dose_series(71)
    offset = 0.003 if vehicle == "on_axis" else None
    axis = ug_per_kg(vehicle=vehicle, offset=offset)
    levels = 8 if vehicle == "own_group" else 9
    basis = axis.spline_basis(cube.samples, levels - 1)
    columns = axis.design_columns(cube.samples, basis)
    samples = cube.samples.with_columns(columns)
    names = list(columns)
    spline = limma_trend(
        cube,
        Design(samples, "~ " + " + ".join(names)),
        contrasts=names,
        f_tests={"any_dose": names},
    ).table("bulk", "any_dose")
    order = ["d0", *(f"d{dose:g}" for dose in TCDD)]
    groups = Design(samples, "~ dose_group", levels={"dose_group": order})
    factor = list(groups.columns[1:])
    dose_groups = limma_trend(
        cube, groups, contrasts=factor, f_tests={"any_dose": factor}
    ).table("bulk", "any_dose")
    assert_array_equal(spline["feature_id"], dose_groups["feature_id"])
    # The F-statistic reads the correlation of the coefficients, whose condition number
    # is about 3e6 (vehicle in its own group) and 1e7 (on the axis) for the spline:
    # rounding leaves about 1e-10 between the two parametrizations.
    assert_allclose(spline["f_statistic"], dose_groups["f_statistic"], rtol=1e-8)
    # p-values far in the tail, down to 1e-314, as differences relative to
    # max(p, 1e-300) max(1, |log p|), about the error in log p relative to |log p|.
    p = dose_groups["p_value"]
    scale = np.maximum(p, 1e-300) * np.maximum(
        1.0, np.abs(np.log(np.maximum(p, 1e-300)))
    )
    assert np.max(np.abs(spline["p_value"] - p) / scale) <= 1e-8


def test_the_nonlinear_test_at_saturation_is_lack_of_fit() -> None:
    # At df = 7 the columns after the first test departure from a line in log10 dose
    # among the treated doses: the lack-of-fit F of dose_treated and log10 dose against
    # the dose groups.
    y = np.random.default_rng(73).normal(0.0, 1.0, (LADDER.size, 3))
    axis, samples = ug_per_kg(), table_of(LADDER)
    columns = axis.design_columns(
        samples, axis.spline_basis(samples, 7, form="restricted_cubic")
    )
    full = matrix_of(columns)
    nonlinear = f_statistic(full, full[:, :3], y)
    treated = (LADDER > 0.0).astype(float)
    log_dose = np.where(LADDER > 0.0, np.log10(np.where(LADDER > 0, LADDER, 1.0)), 0.0)
    line = np.column_stack([np.ones(LADDER.size), treated, log_dose])
    groups = (LADDER[:, None] == np.unique(LADDER)[None, :]).astype(float)
    assert_allclose(nonlinear, f_statistic(groups, line, y), rtol=1e-10)


def test_vehicle_rows_change_no_f_test_and_no_curve() -> None:
    # The vehicle's rows of the spline columns hold one value; any other value adds a
    # multiple of 1 - dose_treated, which the intercept and dose_treated span.
    y = np.random.default_rng(79).normal(0.0, 1.0, (LADDER.size, 2))
    axis, samples = ug_per_kg(), table_of(LADDER)
    columns = axis.design_columns(samples, axis.spline_basis(samples, 3))
    full = matrix_of(columns)
    moved = full.copy()
    moved[LADDER == 0.0, 2:] = [0.7, -1.3, 2.1]
    for design in (full, moved):
        assert_allclose(
            f_statistic(design, design[:, :2], y),
            f_statistic(full, full[:, :2], y),
            rtol=1e-10,
        )
    fitted = full @ np.linalg.lstsq(full, y, rcond=None)[0]
    assert_allclose(
        moved @ np.linalg.lstsq(moved, y, rcond=None)[0], fitted, atol=1e-12
    )


# Invariances.


def test_units_and_order_do_not_matter() -> None:
    samples, in_ng = {"dose": LADDER}, {"dose": LADDER * 1000.0}
    for form in FORMS:
        # log10(1000 d) and 3 + log10(d) can differ in the last bit.
        axis = ug_per_kg()
        first = axis.design_columns(samples, axis.spline_basis(samples, 3, form=form))
        second = axis.design_columns(in_ng, axis.spline_basis(in_ng, 3, form=form))
        for name, values in first.items():
            assert_allclose(second[name], values, rtol=0, atol=1e-12 * 30)
        # With the vehicle on the axis, the offset scales with the doses.
        axis, scaled = (
            ug_per_kg(vehicle="on_axis", offset=offset) for offset in (0.003, 3.0)
        )
        first = axis.design_columns(samples, axis.spline_basis(samples, 3, form=form))
        second = scaled.design_columns(in_ng, scaled.spline_basis(in_ng, 3, form=form))
        for name, values in first.items():
            assert_allclose(second[name], values, rtol=0, atol=1e-12 * 30)
    # On the linear scale the restricted cubic columns scale with the doses, and the
    # B-spline columns do not change.
    axis = ug_per_kg(scale="linear")
    for form, factor in (("restricted_cubic", 1000.0), ("b_spline", 1.0)):
        first = axis.design_columns(samples, axis.spline_basis(samples, 3, form=form))
        second = axis.design_columns(in_ng, axis.spline_basis(in_ng, 3, form=form))
        for name, values in first.items():
            if name != "dose_treated":
                assert_allclose(second[name], factor * values, rtol=1e-12, atol=1e-9)
    # The order of the samples changes the order of the rows only.
    order = np.random.default_rng(83).permutation(LADDER.size)
    axis = ug_per_kg()
    basis = axis.spline_basis(samples, 4, knots="quantiles")
    assert axis.spline_basis({"dose": LADDER[order]}, 4, knots="quantiles") == basis
    permuted = axis.design_columns({"dose": LADDER[order]}, basis)
    for name, values in axis.design_columns(samples, basis).items():
        assert_array_equal(permuted[name], values[order])


# Refusals.


def test_coinciding_knots_and_too_many_columns_are_refused() -> None:
    axis, samples = ug_per_kg(), table_of(LADDER)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "a spline of 8 columns needs 9 distinct doses on the axis, and the samples "
            "have 8, 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and 30 ug/kg (the treated doses: "
            "the vehicle is a group of its own); lower df to at most 7"
        ),
    ):
        axis.spline_basis(samples, 8)
    on_axis = ug_per_kg(vehicle="on_axis", offset=0.003)
    assert on_axis.spline_basis(samples, 8).df == 8
    for doses, found in (([0.0, 0.0, 1.0, 1.0], "1 ug/kg"), ([0.0, 0.0], "none")):
        with pytest.raises(ValueError, match=f"have {found} \\(the treated doses"):
            axis.spline_basis({"dose": doses}, 1)
    with pytest.raises(ValueError, match="the knots given coincide at 0.1 ug/kg"):
        axis.spline_basis(samples, knots=[0.1, 0.1])
    with pytest.raises(ValueError, match="df, the number of spline columns, is needed"):
        axis.spline_basis(samples)
    with pytest.raises(ValueError, match="df=4 disagrees with 2 knots"):
        axis.spline_basis(samples, 4, knots=[0.1, 1.0])
    with pytest.raises(ValueError, match="knots must be a rule"):
        axis.spline_basis(samples, 2, knots="evenly")


@pytest.mark.parametrize(
    ("knots", "message"),
    [
        ([0.0, 1.0], "strictly between the lowest and highest dose on the axis, 0.01 and 30 ug/kg"),
        ([100.0], "got 100 ug/kg"),
        ([0.01], "got 0.01 ug/kg"),
        ([-1.0], "knots must be finite and at least 0"),
        ([np.nan], "knots must be finite and at least 0"),
        (["a"], "knots must hold numbers"),
    ],
)  # fmt: skip
def test_given_knots_are_doses_strictly_inside(knots: list, message: str) -> None:
    error = TypeError if knots == ["a"] else ValueError
    with pytest.raises(error, match=re.escape(message)):
        ug_per_kg().spline_basis(table_of(LADDER), knots=knots)


def test_given_knots_in_any_order_are_sorted() -> None:
    axis, samples = ug_per_kg(), table_of(LADDER)
    basis = axis.spline_basis(samples, knots=[1.0, 0.1], form="b_spline")
    assert basis == axis.spline_basis(samples, 3, knots=[0.1, 1.0], form="b_spline")
    assert_allclose(axis.doses(basis.knots), [0.1, 1.0], rtol=1e-14)
    assert axis.spline_basis(samples, knots=0.3).df == 2


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"column": ""}, ValueError, "column must not be empty"),
        ({"column": 3}, TypeError, "column must be a str, got int"),
        ({"unit": " "}, ValueError, "unit must not be empty"),
        ({"unit": None}, TypeError, "unit must be a str, got NoneType"),
        ({"metric": "dose"}, ValueError, "metric must be one of 'administered', 'nominal', 'measured', 'internal', got 'dose'"),
        ({"scale": "log"}, ValueError, "scale must be one of 'log10', 'linear', got 'log'"),
        ({"vehicle": "zero"}, ValueError, "vehicle must be one of 'own_group', 'on_axis'"),
        ({"offset": 0.1}, ValueError, "offset places the vehicle on a log10 axis, and applies only with scale='log10' and vehicle='on_axis'; got scale='log10' and vehicle='own_group'"),
        ({"vehicle": "on_axis"}, ValueError, "the vehicle's dose of 0 has no logarithm; with scale='log10' and vehicle='on_axis', give offset"),
        ({"vehicle": "on_axis", "offset": 0.0}, ValueError, "offset must be positive and finite, got 0.0"),
        ({"vehicle": "on_axis", "offset": np.nan}, ValueError, "offset must be positive and finite, got nan"),
        ({"vehicle": "on_axis", "offset": "0.1"}, TypeError, "offset must be a real number, got str"),
        ({"scale": "linear", "vehicle": "on_axis", "offset": 1.0}, ValueError, "applies only with scale='log10'"),
    ],
)  # fmt: skip
def test_axis_arguments_are_refused_with_a_fix(
    arguments: dict[str, Any], error: type[Exception], message: str
) -> None:
    options = {"column": "dose", "unit": "uM", "metric": "nominal"} | arguments
    with pytest.raises(error, match=re.escape(message)):
        DoseAxis(**options)


@pytest.mark.parametrize(
    ("samples", "error", "message"),
    [
        ({"dose": [0.0, -1.0, 1.0]}, ValueError, "samples['dose'] must be finite and at least 0, with 0 for the vehicle; got -1.0"),
        ({"dose": [0.0, np.nan, 1.0]}, ValueError, "samples['dose'] must be finite and at least 0, with 0 for the vehicle; got nan"),
        ({"dose": ["0", "1"]}, TypeError, "samples['dose'] must hold numbers, got dtype object"),
        ({"dose": [False, True]}, TypeError, "samples['dose'] must hold numbers, got dtype bool"),
        ({"Dose": [0.0, 1.0]}, ValueError, "the dose column 'dose' is not a column of the sample table (did you mean 'Dose'?); columns are: 'Dose'"),
        ([0.0, 1.0], TypeError, "samples must be a Table, a mapping of column names to 1-D values or a pandas DataFrame, got list"),
    ],
)  # fmt: skip
def test_doses_are_refused_with_a_fix(
    samples: Any, error: type[Exception], message: str
) -> None:
    axis = ug_per_kg()
    basis = NaturalSplineBasis(knots=[], boundary_knots=[0.0, 1.0])
    for call in (
        lambda: axis.spline_basis(samples, 1),
        lambda: axis.design_columns(samples, basis),
    ):
        with pytest.raises(error, match=re.escape(message)):
            call()


def test_messages_say_what_x_and_the_doses_on_the_axis_are() -> None:
    # A quarter of the treated doses tie at the lowest: Harrell's rule refuses, and
    # the message says what x is on a linear axis.
    doses = np.r_[0.0, 0.0, np.ones(10), np.arange(2.0, 32.0)]
    linear = ug_per_kg(scale="linear")
    with pytest.raises(ValueError, match="tie at the smallest value, 1.0") as error:
        linear.spline_basis({"dose": doses}, 3, knots="harrell")
    assert str(error.value).endswith("(on this axis, x is the treated doses in ug/kg)")
    # With the vehicle on the axis, its dose counts among the doses on the axis.
    on_axis = ug_per_kg(vehicle="on_axis", offset=0.003)
    with pytest.raises(ValueError) as error:
        on_axis.spline_basis(table_of(LADDER), 9)
    assert str(error.value) == (
        "a spline of 9 columns needs 10 distinct doses on the axis, and the samples "
        "have 9, 0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10 and 30 ug/kg; lower df to at most 8"
    )


def test_positions_and_doses_refuse_what_is_not_a_dose() -> None:
    axis = ug_per_kg()
    with pytest.raises(ValueError, match="doses must be finite and at least 0"):
        axis.positions([1.0, np.inf])
    with pytest.raises(ValueError, match="doses must be a 1-D array"):
        axis.positions([[1.0]])
    with pytest.raises(ValueError, match="positions must not be infinite"):
        axis.doses([1.0, -np.inf])
    with pytest.raises(TypeError, match="positions must hold real numbers"):
        axis.doses(["1"])
    with pytest.raises(ValueError, match="positions must be a 1-D array"):
        axis.doses(1.0)


def test_doses_that_differ_only_by_rounding_are_refused() -> None:
    # 0.1 * 3 is 0.30000000000000004, one dose that a conversion of units left as two.
    doses = np.array([0.0, 0.0, 0.1 * 3, 0.3, 1.0, 1.0, 10.0, 10.0])
    axis = ug_per_kg()
    with pytest.raises(ValueError, match="differ by less than 1e-8 of the dose"):
        axis.spline_basis({"dose": doses}, 2)
    assert axis.spline_basis({"dose": np.round(doses, 10)}, 2).df == 2


def test_design_columns_refuse_bases_they_cannot_use() -> None:
    axis, samples = ug_per_kg(), table_of(LADDER)
    with pytest.raises(TypeError, match="basis must be a NaturalSplineBasis"):
        axis.design_columns(samples, np.ones((LADDER.size, 2)))  # type: ignore[arg-type]
    with_intercept = NaturalSplineBasis(np.log10(TCDD), 3, intercept=True)
    with pytest.raises(ValueError, match="the basis has an intercept"):
        axis.design_columns(samples, with_intercept)


# Tables, records and the defaults of the design.


def test_samples_as_a_table_a_mapping_or_a_data_frame() -> None:
    pandas = pytest.importorskip("pandas")
    axis = ug_per_kg()
    basis = axis.spline_basis({"dose": LADDER}, 3)
    expected = axis.design_columns(table_of(LADDER), basis)
    frames = (
        pandas.DataFrame({"dose": LADDER}),
        # A categorical of numbers is read as its numbers.
        pandas.DataFrame({"dose": pandas.Categorical(LADDER)}),
    )
    for samples in ({"dose": list(LADDER)}, *frames):
        found = axis.design_columns(samples, basis)
        assert list(found) == list(expected)
        for name, values in expected.items():
            assert_array_equal(found[name], values)
            assert found[name].dtype == np.float64
    text = pandas.DataFrame({"dose": pandas.Categorical([f"{d:g}" for d in LADDER])})
    with pytest.raises(TypeError, match=r"samples\['dose'\] must hold numbers"):
        axis.spline_basis(text, 3)


def test_columns_without_a_vehicle_or_on_a_grid() -> None:
    # A design without vehicle samples: dose_treated is 1 everywhere and Design.check
    # reports it aliased with the intercept.
    axis = ug_per_kg()
    treated = LADDER[LADDER > 0.0]
    columns = axis.design_columns(
        table_of(treated), axis.spline_basis({"dose": treated}, 3)
    )
    assert_array_equal(columns["dose_treated"], 1.0)
    samples = table_of(treated).with_columns(columns)
    report = Design(samples, "~ " + " + ".join(columns)).check()
    assert "rank_deficient" in report.kinds() and report.aliased == ("dose_treated",)
    # The design with the vehicle passes the check.
    basis = axis.spline_basis({"dose": LADDER}, 3)
    full = table_of(LADDER).with_columns(axis.design_columns(table_of(LADDER), basis))
    assert Design(full, "~ dose_treated + dose_s1 + dose_s2 + dose_s3").check().ok
    # A grid of doses beyond the highest gets the columns of the same curve.
    grid = axis.design_columns({"dose": [0.0, 30.0, 300.0]}, basis)
    assert_array_equal(grid["dose_treated"], [0.0, 1.0, 1.0])
    assert grid["dose_s1"][2] == pytest.approx(grid["dose_s1"][1] + 1.0)


def test_an_axis_survives_its_record() -> None:
    for axis in (
        ug_per_kg(),
        ug_per_kg(scale="linear", vehicle="on_axis"),
        DoseAxis(
            "exposure", unit="ng/mL", metric="measured", vehicle="on_axis", offset=1
        ),
    ):
        record = json.loads(json.dumps(axis.to_dict()))
        assert set(record) == {"column", "unit", "metric", "scale", "vehicle", "offset"}
        assert DoseAxis(**record) == axis
    assert (
        DoseAxis("x", unit="Gy", metric="internal", vehicle="on_axis", offset=1).offset
        == 1.0
    )
    assert isinstance(
        DoseAxis("x", unit="Gy", metric="internal", vehicle="on_axis", offset=1).offset,
        float,
    )


def test_the_chapter_example_runs() -> None:
    # The code of "Dose axes and the vehicle", run on a dose series.
    text = CHAPTER.read_text()
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    (example,) = [block for block in blocks if "cp.DoseAxis(" in block]
    namespace = {"cp": cytepool, "cube": dose_series(89)}
    exec(example, namespace)  # noqa: S102
    result = namespace["result"]
    assert set(result.f_tests) == {"any_dose", "among_doses", "nonlinear"}
    tables = {test: result.table("bulk", test) for test in result.f_tests}
    # 60 genes respond along log dose; the F-test of any dose finds most of them.
    responsive = np.isin(
        tables["any_dose"]["feature_id"], [f"feature_{i}" for i in range(60)]
    )
    called = tables["any_dose"]["adjusted_p_value"] < 0.05
    assert called[responsive].mean() > 0.5 and called[~responsive].mean() < 0.1


def test_the_examples_in_the_docstrings_run() -> None:
    assert doctest.testmod(dose_axes, raise_on_error=False).failed == 0
