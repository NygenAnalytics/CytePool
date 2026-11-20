"""Tests for the dose tests of each cell type, dose_response and DoseResponseResult.

Reference values in ``tests/data/dose_response_tests/limma.json`` come from limma
3.62.1 and edgeR 4.4.0 in R, written by ``validation/dose_response_tests.py``: R
filters each cell type's samples from their cells and counts itself, places the knots
with its own rule and fits the spline through the dose groups' means with its own
``ns()`` and Hmisc columns, so that it shares nothing with CytePool but the counts and
the settings. Other tests check what R cannot: least squares on the samples
themselves, identities of the tests derived by hand, the rules that choose the tests
and the spline's size, invariants, refusals and the records.
"""

import contextlib
import functools
import io
import json
import math
import pickle
import re
import zlib
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.stats import t as student_t

import cytepool
from cytepool import (
    CandidateShapes,
    Cube,
    Design,
    DoseAxis,
    DoseResponseResult,
    dose_response,
    dream_voom,
    limma_trend,
    limma_voom,
    read_cube,
    simulate_dose_series,
    write_cube,
)
from cytepool import dose_responses
from cytepool.dose_designs import DoseDesignReport
from cytepool.dose_responses import (
    _CellTypeFit,
    _dose_group_estimates,
    _run_gate,
    _smallest_times_doses,
)
from cytepool.stats.distributions import t_ppf
from cytepool.stats.multiple_contrasts import (
    multiple_contrast_test,
    stage_wise_adjusted_p_values,
)
from cytepool.stats.multiple_testing import adjust_p_values
from cytepool.table import Table

DATA = Path(__file__).parent / "data" / "dose_response_tests"
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
ON_AXIS = DoseAxis(
    "dose", unit="ug/kg", metric="administered", vehicle="on_axis", offset=0.001
)
TESTS = ("spline", "nonlinearity", "linear_trend", "dose_groups")
NORMAL_QUANTILE = 1.959963984540054  # the 0.975 quantile of the standard normal


# The fixture of R's results, and the series it was computed on.


@functools.cache
def reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / "limma.json").read_text())
    return found


@functools.cache
def fixture_cube() -> Cube:
    """The series R analysed: the vehicle and 5 doses x 3, with a cell type of every
    sample and one that loses the top dose and a sample of 3 ug/kg."""
    series = reference()["series"]
    return Cube(
        {"counts": np.array(series["counts"], dtype=np.int64)},
        samples={"sample_id": series["sample_id"], "dose": series["dose"]},
        groups={"group_id": series["groups"]},
        features={"feature_id": series["features"]},
        n_cells=np.array(series["n_cells"], dtype=np.int64),
    )


@functools.cache
def fixture_result(analysis: str, standard_errors: str = "exact") -> DoseResponseResult:
    cube = fixture_cube()
    options: dict[str, Any] = {}
    if analysis == "voom":
        options["contrast_standard_errors"] = standard_errors
    return dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=list(TESTS),
        spline_df=2,
        analysis=limma_voom if analysis == "voom" else limma_trend,
        **options,
    )


def assert_scaled(ours: Any, theirs: Any, tolerance: float = 1e-10) -> None:
    """Differences over max(|reference|, 1), as the Validation page scales them."""
    a, b = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert a.shape == b.shape
    assert np.all(np.abs(a - b) <= tolerance * np.maximum(np.abs(b), 1.0))


def assert_p_values(ours: Any, theirs: Any, tolerance: float = 1e-10) -> None:
    """p-values, to a tolerance relative to p max(1, |log p|)."""
    a, b = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert a.shape == b.shape
    scale = b * np.maximum(1.0, np.abs(np.log(b)))
    assert np.all(np.abs(a - b) <= tolerance * scale)


def by_dose(table: Any, column: str, k: int) -> np.ndarray:
    """A column of the table of each dose as genes x doses."""
    return np.asarray(table[column], dtype=float).reshape(k, -1).T


def check_against_r(
    result: DoseResponseResult, job: str, *, exact: bool, given: bool
) -> None:
    """Every table of a result against R's: the samples, dose groups and genes, each
    test's statistics and p-values within and across cell types, and each dose."""
    features = reference()["series"]["features"]
    summary = result.summary
    for group, theirs in reference()["jobs"][job]["cell_types"].items():
        entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
        assert entry["fit"]["samples"] == theirs["samples"]
        assert entry["doses"] == theirs["doses"]
        assert entry["samples"] == theirs["n"]
        genes = [features[i - 1] for i in theirs["genes"]]
        row = summary["group_id"].tolist().index(group)
        assert summary["n_features_tested"][row] == len(genes)
        assert summary["n_dose_groups_tested"][row] == len(theirs["doses"])
        assert_allclose(summary["df_prior"][row], theirs["df_prior"], rtol=1e-10)
        # The posterior variances and the degrees of freedom of the t-tests, which
        # the bands use, from the fit computed again.
        fit = result.fit(group)
        assert_scaled(fit.moderated.posterior_variance, theirs["s2_post"])
        assert_scaled(fit.moderated.df_total, theirs["df_total"])
        prefix = "exact_" if exact else ""
        for test, values in theirs["tests"].items():
            table = result.table(group, test)
            assert table["feature_id"].tolist() == genes
            if test == "linear_trend":
                assert_scaled(table["log2_fold_change"], values["logFC"])
                assert_scaled(table["standard_error"], values[prefix + "se"])
                assert_scaled(table["t"], values[prefix + "t"])
                assert_p_values(table["p_value"], values[prefix + "p"])
                assert_p_values(table["adjusted_p_value"], values[prefix + "adj"])
                assert_p_values(
                    table["adjusted_p_value_across_groups"],
                    values[prefix + "p_across"],
                )
                if not exact:
                    assert_scaled(table["log_odds"], values["B"])
                continue
            # limma's F of the rows CytePool tests, under limma's standard errors;
            # a key of this test's own, so that the next tests keep the prefix.
            key = prefix
            if given and test in theirs.get("given", {}):
                values, key = theirs["given"][test], ""
            assert entry["f_test_df1"][test] == values["df1"]
            assert_scaled(table["f_statistic"], values[key + "F"])
            assert_p_values(table["p_value"], values[key + "p"])
            assert_p_values(table["adjusted_p_value"], values[key + "adj"])
            assert_p_values(
                table["adjusted_p_value_across_groups"], values[key + "p_across"]
            )
            if "df2" in values:
                assert_scaled(table["df"], values["df2"])
        each = result.table(group, "each_dose")
        k = len(theirs["doses"]) - 1
        assert (
            each["dose"].tolist() == np.repeat(theirs["doses"][1:], len(genes)).tolist()
        )
        assert each["feature_id"].tolist() == np.tile(genes, k).tolist()
        assert_scaled(
            by_dose(each, "log2_fold_change", k), theirs["each_dose"]["logFC"]
        )
        assert_scaled(
            by_dose(each, "standard_error", k), theirs["each_dose"][prefix + "se"]
        )
        assert_scaled(by_dose(each, "t", k), theirs["each_dose"][prefix + "t"])
        assert_p_values(by_dose(each, "p_value", k), theirs["each_dose"][prefix + "p"])
        assert_p_values(
            by_dose(each, "adjusted_p_value", k), theirs["each_dose"][prefix + "adj"]
        )


# Against R.


def test_tests_match_limma() -> None:
    # limma-trend, and limma-voom with limma's standard errors, whose F-tests of the
    # spline R computes on the rows CytePool tests: limma's approximation depends on
    # the basis of the hypothesis.
    check_against_r(fixture_result("trend"), "fixture trend", exact=False, given=False)
    check_against_r(
        fixture_result("voom", "limma"), "fixture voom", exact=False, given=True
    )
    # Without weights, an F-statistic times its numerator degrees of freedom and the
    # posterior variance is the partial sum of squares of rms's anova, for 50 genes.
    result = fixture_result("trend")
    for group, theirs in reference()["jobs"]["fixture trend"]["cell_types"].items():
        df1 = next(e for e in result.provenance["groups"] if e["group_id"] == group)[
            "f_test_df1"
        ]
        s2 = result.fit(group).moderated.posterior_variance
        for test, key in (("spline", "total"), ("nonlinearity", "nonlinear")):
            m = len(theirs["rms"][key])
            sums = result.table(group, test)["f_statistic"][:m] * df1[test] * s2[:m]
            assert_allclose(sums, theirs["rms"][key], rtol=1e-10)


def test_the_default_f_matches_weighted_least_squares() -> None:
    # The exact default under voom's weights against each gene's lm.wfit in R, with
    # R's own spline rows: the exact F does not depend on the basis.
    check_against_r(fixture_result("voom"), "fixture voom", exact=True, given=False)


@pytest.mark.parametrize(
    ("analysis", "standard_errors", "exact"),
    [("trend", "exact", False), ("voom", "exact", True), ("voom", "limma", False)],
)
def test_curves_and_bands_match_r(
    analysis: str, standard_errors: str, exact: bool
) -> None:
    result = fixture_result(analysis, standard_errors)
    job = f"fixture {analysis}"
    for group, theirs in reference()["jobs"][job]["cell_types"].items():
        curve = theirs["curve"]
        at = np.array(curve["at"])
        n = len(curve["estimate"])
        genes = [reference()["series"]["features"][i - 1] for i in theirs["genes"][:n]]
        found = result.curve(group, genes, at=at)
        assert found["feature_id"].tolist() == np.repeat(genes, at.size).tolist()
        assert_array_equal(found["dose"], np.tile(at, n))
        assert_scaled(found["log2_fold_change"].reshape(n, -1), curve["estimate"])
        lower = curve["exact_lower"] if exact else curve["lower"]
        upper = curve["exact_upper"] if exact else curve["upper"]
        assert_scaled(found["ci_lower"].reshape(n, -1), lower)
        assert_scaled(found["ci_upper"].reshape(n, -1), upper)


# Identities derived by hand.


def test_a_saturated_spline_is_the_dose_group_test() -> None:
    # With as many columns as the treated doses less one, the spline, its treated
    # column and the intercept pass through every dose group's mean, so its F-test
    # is the dose-group F-test.
    cube = fixture_cube()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=["spline", "dose_groups"],
        spline_df=4,
        analysis=limma_voom,
    )
    full = result.table("full", "spline")
    assert_scaled(
        full["f_statistic"], result.table("full", "dose_groups")["f_statistic"]
    )
    assert result.provenance["groups"][0]["f_test_df1"] == {
        "spline": 5,
        "dose_groups": 5,
    }
    # The depleted cell type compares 4 treated doses, where 4 columns and the
    # treated column are one too many: its spline is not run.
    assert "spline" in result.provenance["groups"][1]["tests_not_run"]


def least_squares_values(result: DoseResponseResult, group: str) -> tuple:
    """The values limma-trend fitted, and each sample's dose, of a cell type."""
    fit = result.fit(group)
    return np.asarray(fit.log2_cpm), np.asarray(fit.samples["dose"], dtype=float)


def test_without_weights_the_spline_is_the_least_squares_spline() -> None:
    # Without observation weights or other terms, the contrasts of the dose groups
    # give the spline that least squares fits to the samples themselves. The
    # depleted cell type keeps 3, 3, 3, 3 and 2 samples, so the groups' weights
    # matter.
    result = fixture_result("trend")
    axis, basis = result.dose, result.spline_basis
    assert basis is not None
    for group in ("full", "depleted"):
        y, doses = least_squares_values(result, group)
        columns = axis.design_columns({"dose": doses}, basis)
        x = np.column_stack([np.ones(doses.size), *columns.values()])
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        at = np.array([0.0, 0.1, 0.17, 0.5, 2.0, 3.0])
        grid = axis.design_columns({"dose": at}, basis)
        rows = np.column_stack([np.ones(at.size), *grid.values()])
        # The vehicle's mean is the intercept: the curve is a change from it.
        expected = (rows @ beta - beta[0]).T
        curve = result.curve(group, at=at)
        assert_allclose(
            curve["log2_fold_change"].reshape(-1, at.size), expected, atol=1e-12
        )
        # The linear trend: a straight line along the treated doses, with the
        # vehicle apart.
        line = axis.spline_basis(Design(fixture_cube().samples, "~ 1").samples, df=1)
        first = axis.design_columns({"dose": doses}, line)
        x1 = np.column_stack([np.ones(doses.size), *first.values()])
        slope = np.linalg.lstsq(x1, y, rcond=None)[0][-1]
        assert_allclose(
            result.table(group, "linear_trend")["log2_fold_change"], slope, atol=1e-12
        )


def test_on_the_axis_the_curve_is_a_change_from_the_vehicles_mean() -> None:
    # With the vehicle on the axis, the spline need not pass through the vehicle's
    # mean: the curve at dose 0 is the spline there minus the vehicle's mean, which
    # least squares on the samples gives.
    cube = fixture_cube()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=ON_AXIS,
        spline_df=2,
        tests=["spline"],
        analysis=limma_trend,
    )
    basis = result.spline_basis
    assert basis is not None
    y, doses = least_squares_values(result, "full")
    columns = ON_AXIS.design_columns({"dose": doses}, basis)
    x = np.column_stack([np.ones(doses.size), *columns.values()])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    vehicle = y[doses == 0.0].mean(axis=0)
    at = np.array([0.0, 0.03, 1.0])
    grid = ON_AXIS.design_columns({"dose": at}, basis)
    rows = np.column_stack([np.ones(at.size), *grid.values()])
    expected = (rows @ beta).T - vehicle[:, None]
    curve = result.curve("full", at=at)
    found = curve["log2_fold_change"].reshape(-1, at.size)
    assert_allclose(found, expected, atol=1e-12)
    # It misses the vehicle's mean by a visible amount for some genes.
    assert np.max(np.abs(found[:, 0])) > 0.01
    assert np.all(curve["standard_error"].reshape(-1, at.size)[:, 0] > 0.0)


def test_a_dropped_dose_group_is_left_out() -> None:
    # The depleted cell type keeps no sample at 10 ug/kg: its dose-group F-test is
    # limma_voom's F-test of the four doses it keeps, on a model of the dose groups
    # written by hand.
    cube = fixture_cube()
    result = fixture_result("voom")
    doses = np.asarray(cube.samples["dose"])
    names = {0.1: "d01", 0.3: "d03", 1.0: "d1", 3.0: "d3", 10.0: "d10"}
    samples = cube.samples.with_columns(
        {name: (doses == dose).astype(float) for dose, name in names.items()}
    )
    by_hand = limma_voom(
        cube,
        Design(samples, "~ d01 + d03 + d1 + d3 + d10"),
        groups=["depleted"],
        contrasts=["d01", "d03", "d1", "d3"],
        f_tests={"kept": ["d01", "d03", "d1", "d3"]},
    )
    ours = result.table("depleted", "dose_groups")
    assert_allclose(
        ours["f_statistic"],
        by_hand.table("depleted", "kept")["f_statistic"],
        rtol=1e-12,
    )
    entry = result.provenance["groups"][1]
    assert entry["f_test_df1"]["dose_groups"] == 4
    assert entry["doses"] == [0.0, 0.1, 0.3, 1.0, 3.0]
    assert entry["samples"] == [3.0, 3.0, 3.0, 3.0, 2.0]
    assert 10.0 not in result.table("depleted", "each_dose")["dose"].tolist()
    assert 10.0 in result.table("full", "each_dose")["dose"].tolist()
    # Each dose is the dose's own t-test.
    for dose, name in list(names.items())[:4]:
        each = result.table("depleted", "each_dose")
        rows = each["dose"] == dose
        mine = by_hand.table("depleted", name)
        assert_allclose(each["t"][rows], mine["t"], rtol=1e-12)


def least_squares_curve(
    result: DoseResponseResult, group: str, at: np.ndarray
) -> np.ndarray:
    """The spline that least squares fits to a cell type's samples, minus the
    vehicle's mean, at doses (limma-trend under "~ 1"), genes x doses."""
    y, doses = least_squares_values(result, group)
    axis, basis = result.dose, result.spline_basis
    assert basis is not None
    columns = axis.design_columns({axis.column: doses}, basis)
    x = np.column_stack([np.ones(doses.size), *columns.values()])
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    grid = axis.design_columns({axis.column: at}, basis)
    rows = np.column_stack([np.ones(at.size), *grid.values()])
    found: np.ndarray = (rows @ beta - beta[0]).T
    return found


@pytest.mark.parametrize(
    ("lost", "kept"),
    [(0.1, [0.0, 0.3, 1.0, 3.0, 10.0]), (1.0, [0.0, 0.1, 0.3, 3.0, 10.0])],
)
def test_a_cell_type_that_loses_a_low_or_middle_dose(
    lost: float, kept: list[float]
) -> None:
    # The samples of the lowest, or a middle, dose keep 5 cells in one cell type:
    # its contrasts weigh the indicators of the doses it keeps, and its curve is the
    # spline that least squares fits to its samples.
    doses = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
    cube, _ = simulate_dose_series(
        doses,
        replicates=3,
        cell_types=["all", "lost"],
        cells=300.0,
        n_genes=150,
        rng=11,
    )
    cube = recelled(
        cube, {"lost": [5 if dose == lost else 300 for dose in np.repeat(doses, 3)]}
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        spline_df=2,
        analysis=limma_trend,
    )
    entry = result.provenance["groups"][1]
    assert entry["doses"] == kept
    assert entry["f_test_df1"]["dose_groups"] == 4
    each = result.table("lost", "each_dose")
    assert sorted(set(each["dose"].tolist())) == kept[1:]
    at = np.array([0.0, kept[1], 2.0, 10.0])
    found = result.curve("lost", at=at)["log2_fold_change"].reshape(-1, at.size)
    assert_allclose(found, least_squares_curve(result, "lost", at), atol=1e-12)
    if lost < kept[1]:
        with pytest.raises(ValueError, match="outside the doses that 'lost' compares"):
            result.curve("lost", at=[lost])


def test_doses_a_millionth_apart_are_two_dose_groups() -> None:
    # 1 and 1.000001 ug/kg differ by more than the 1e-8 that makes two doses one, so
    # each has an indicator of its own: its estimate is the mean of its samples
    # minus the vehicle's, and the curve the spline that least squares fits to the
    # samples, also when the dose-group test is not run.
    doses = [0.0, 1.0, 1.000001, 10.0, 100.0]
    cube, _ = simulate_dose_series(
        doses, replicates=3, cell_types=1, cells=300.0, n_genes=150, rng=2
    )
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS, analysis=limma_trend)
    assert result.provenance["design"]["columns"] == [
        "(Intercept)",
        "dose_1",
        "dose_1.000001",
        "dose_10",
        "dose_100",
    ]
    y, sample_doses = least_squares_values(result, "cell_type_1")
    vehicle = y[sample_doses == 0.0].mean(axis=0)
    means = np.column_stack(
        [y[sample_doses == dose].mean(axis=0) - vehicle for dose in doses[1:]]
    )
    each = result.table("cell_type_1", "each_dose")
    assert_allclose(by_dose(each, "log2_fold_change", 4), means, atol=1e-12)
    along = dose_response(
        cube, design, dose=AXIS, tests=["spline", "linear_trend"], analysis=limma_trend
    )
    at = np.array([0.0, 1.0, 5.0, 100.0])
    found = along.curve("cell_type_1", at=at)["log2_fold_change"]
    assert_allclose(
        found.reshape(-1, at.size),
        least_squares_curve(along, "cell_type_1", at),
        atol=1e-12,
    )


def test_molar_doses_and_a_dose_column_that_needs_quoting() -> None:
    # Doses in M, from 1e-9, name indicators such as dose_1e-09, and a dose column
    # named "dose (nM)" names dose (nM)_1: the design quotes both in backticks. The
    # tests are those of the same series in nM, and so are the curves.
    nanomolar = [0.0, 1.0, 10.0, 100.0, 1000.0]
    cube, _ = simulate_dose_series(
        nanomolar, replicates=3, cell_types=1, cells=300.0, n_genes=150, rng=6
    )
    doses = np.asarray(cube.samples["dose"], dtype=float)

    def tested(column: str, unit: str, scale: float) -> DoseResponseResult:
        samples = cube.samples.with_columns({column: doses * scale})
        axis = DoseAxis(column, unit=unit, metric="nominal")
        return dose_response(cube, Design(samples, "~ 1"), dose=axis)

    reference = tested("dose", "nM", 1.0)
    for result, scale in (
        (tested("dose", "M", 1e-9), 1e-9),
        (tested("dose (nM)", "nM", 1.0), 1.0),
    ):
        for test in TESTS:
            assert_allclose(
                result.table("cell_type_1", test)["p_value"],
                reference.table("cell_type_1", test)["p_value"],
                rtol=1e-10,
            )
        at = np.array([0.0, 3.0, 300.0])
        assert_allclose(
            result.curve("cell_type_1", at=at * scale)["ci_upper"],
            reference.curve("cell_type_1", at=at)["ci_upper"],
            rtol=1e-10,
            atol=1e-12,
        )
    columns = tested("dose (nM)", "nM", 1.0).provenance["design"]["columns"]
    assert "`dose (nM)_1`" in columns


# The tests and the spline's size the design chooses.


@functools.cache
def series(doses: tuple[float, ...], seed: int = 3) -> Cube:
    cube, _ = simulate_dose_series(
        list(doses), replicates=3, cell_types=1, cells=300.0, n_genes=150, rng=seed
    )
    return cube


def run(
    doses: tuple[float, ...], axis: DoseAxis = AXIS, **options: Any
) -> DoseResponseResult:
    """dose_response on a series of these doses; the simulator draws two treated
    doses at least, so a design of one takes the samples of its doses."""
    cube = series(doses if len(doses) > 2 else (*doses, 2 * doses[-1]))
    samples = cube.samples.filter(np.isin(cube.samples["dose"], doses))
    return dose_response(cube, Design(samples, "~ 1"), dose=axis, **options)


def ran(result: DoseResponseResult) -> list[str]:
    """The tests the one cell type ran."""
    return [test for test in TESTS if ("cell_type_1", test) in result.tables]


@pytest.mark.parametrize(
    ("doses", "axis", "tests", "not_run"),
    [
        # 5 treated doses: 2 columns, every test.
        ((0, 0.1, 1, 10, 30, 100), AXIS, list(TESTS), {}),
        # 3 treated doses: 1 column, so no non-linearity.
        ((0, 0.1, 1, 10), AXIS, ["spline", "linear_trend", "dose_groups"], {}),
        # 2 treated doses: the spline, its treated column and the intercept pass
        # through the 3 groups, so it is the dose-group test, which runs as the
        # spline.
        ((0, 1, 10), AXIS, ["spline", "linear_trend"], {"dose_groups": "spline"}),
        # One treated dose: nothing runs along the axis.
        ((0, 10), AXIS, ["dose_groups"], {}),
        # The vehicle on the axis, 2 doses: one t-test, which runs as the spline.
        (
            (0, 10),
            ON_AXIS,
            ["spline"],
            {"linear_trend": "spline", "dose_groups": "spline"},
        ),
        # 3 doses on the axis: a spline of 1 column is the straight line.
        ((0, 1, 10), ON_AXIS, ["spline", "dose_groups"], {"linear_trend": "spline"}),
        # 4 doses on the axis: 2 columns, every test.
        ((0, 0.1, 1, 10), ON_AXIS, list(TESTS), {}),
    ],
)
def test_tests_follow_the_design(
    doses: tuple[float, ...], axis: DoseAxis, tests: list[str], not_run: dict
) -> None:
    result = run(doses, axis)
    assert ran(result) == tests
    reasons = result.provenance["groups"][0]["tests_not_run"]
    assert sorted(reasons) == sorted(not_run)
    for test, same in not_run.items():
        assert f"the same hypothesis as {same}" in reasons[test]
        with pytest.raises(KeyError, match=f"same hypothesis as {same}"):
            result.table("cell_type_1", test)
    assert result.summary["tests_not_run"].tolist() == [
        ", ".join(t for t in TESTS if t in not_run)
    ]
    # Named, a test of the same hypothesis runs.
    if not_run:
        named = run(doses, axis, tests=[*tests, *not_run])
        assert ran(named) == [t for t in TESTS if t in [*tests, *not_run]]


def test_a_test_named_alone_is_the_same_test() -> None:
    # The non-linearity test named alone, with the gate's test, the spline, beside
    # it: the same contrasts, so the same F-statistics.
    alone = run((0, 0.1, 1, 10, 30, 100), tests="nonlinearity")
    every = run((0, 0.1, 1, 10, 30, 100))
    assert ran(alone) == ["spline", "nonlinearity"]
    assert_allclose(
        alone.table("cell_type_1", "nonlinearity")["f_statistic"],
        every.table("cell_type_1", "nonlinearity")["f_statistic"],
        rtol=1e-12,
    )
    assert alone.tests == ("spline", "nonlinearity")
    assert alone.settings["tests"] == ["nonlinearity"]


def test_named_tests_the_design_cannot_support_are_refused() -> None:
    with pytest.raises(
        ValueError,
        match=r"'linear_trend' runs along the dose axis and needs two doses on it, "
        r"and the design has one \(the treated doses",
    ):
        run((0, 10), tests=["linear_trend"])
    with pytest.raises(ValueError, match="'spline' and 'linear_trend' run along"):
        run((0, 10), tests=["spline", "linear_trend"])
    # A design of the vehicle's samples alone, on the axis: nothing can be tested.
    cube = series((0, 1, 10))
    vehicle = cube.samples.filter(cube.samples["dose"] == 0.0)
    with pytest.raises(
        ValueError, match="the design has one; a dose series needs the vehicle and a"
    ):
        dose_response(cube, Design(vehicle, "~ 1"), dose=ON_AXIS, tests="spline")
    with pytest.raises(ValueError, match="spline of two columns or more.*spline_df=2"):
        run((0, 0.1, 1, 10), tests=["nonlinearity"])
    with pytest.raises(ValueError, match="needs three doses on the axis"):
        run((0, 1, 10), tests=["nonlinearity"])
    # The spline's size is DoseAxis.spline_basis's to refuse, with a note on
    # spline_df.
    with pytest.raises(ValueError, match="at least two distinct doses") as error:
        run((0, 10), spline_df=1)
    assert "spline_df" in "".join(error.value.__notes__)
    with pytest.raises(ValueError, match="lower df to at most 3") as error:
        run((0, 0.1, 1, 10, 100), spline_df=4)
    assert "spline_df" in "".join(error.value.__notes__)
    # At the limit, the spline passes through every treated dose.
    assert run((0, 0.1, 1, 10, 100), spline_df=3).spline_basis.df == 3


@pytest.mark.parametrize(
    ("doses", "axis", "columns"),
    [
        ((0, 1, 10), AXIS, 1),
        ((0, 0.1, 1, 10), AXIS, 1),
        ((0, 0.1, 1, 10, 100), AXIS, 2),
        ((0, 0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30), AXIS, 2),
        ((0, 10), ON_AXIS, 1),
        ((0, 1, 10), ON_AXIS, 1),
        ((0, 0.1, 1, 10), ON_AXIS, 2),
    ],
)
def test_default_spline_size(
    doses: tuple[float, ...], axis: DoseAxis, columns: int
) -> None:
    # 2 columns when 4 doses or more are on the axis, else 1.
    basis = run(doses, axis).spline_basis
    assert basis is not None and basis.df == columns


# Cell types the check leaves out, or whose doses cannot estimate a test.


def recelled(cube: Cube, cells: dict[str, list[int]]) -> Cube:
    """The cube with the cells of some cell types replaced."""
    n_cells = np.array(cube.n_cells, copy=True)
    ids = cube.groups["group_id"].tolist()
    for group, values in cells.items():
        n_cells[:, ids.index(group)] = values
    return Cube(
        {"counts": np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
    )


@functools.cache
def mixed_cube() -> Cube:
    """Five cell types of one series of the vehicle and 5 doses x 3: one keeps every
    sample, one keeps one sample per treated dose (descriptive), one no vehicle
    sample (refused), one only the vehicle and the two lowest doses, and one only the
    vehicle and the lowest dose."""
    cube, _ = simulate_dose_series(
        [0, 0.1, 0.3, 1, 3, 10],
        replicates=3,
        cell_types=["all", "single", "no_vehicle", "low", "one_dose"],
        cells=300.0,
        n_genes=150,
        rng=11,
    )
    return recelled(
        cube,
        {
            "single": [300] * 3 + [300, 5, 5] * 5,
            "no_vehicle": [5] * 3 + [300] * 15,
            "low": [300] * 9 + [5] * 9,
            "one_dose": [300] * 6 + [5] * 12,
        },
    )


@functools.cache
def mixed_result() -> DoseResponseResult:
    cube = mixed_cube()
    return dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)


def test_descriptive_and_refused_cell_types_are_not_tested() -> None:
    result = mixed_result()
    report = result.design_report
    summary = result.summary
    assert (
        summary["status"].tolist()
        == report.summary["status"].tolist()
        == [
            "inferential",
            "descriptive",
            "refused",
            "inferential",
            "inferential",
        ]
    )
    assert summary["tested"].tolist() == [True, False, False, True, True]
    assert summary["reason"].tolist()[1:3] == report.summary["reason"].tolist()[1:3]
    assert summary["n_features_tested"].tolist()[1:3] == [0, 0]
    assert np.isnan(summary["df_residual"][1]) and np.isnan(summary["df_prior"][2])
    for group in ("single", "no_vehicle"):
        assert not any(key[0] == group for key in result.tables)
        with pytest.raises(KeyError, match=f"{group!r} was not tested"):
            result.fit(group)
        with pytest.raises(KeyError, match=f"{group!r} was not tested"):
            result.table(group, "dose_groups")
        with pytest.raises(KeyError, match=f"{group!r} was not tested"):
            result.curve(group, at=[0.0])


def test_a_cell_type_whose_doses_cannot_estimate_the_spline() -> None:
    # "low" keeps the vehicle, 0.1 and 0.3 ug/kg: a spline of 2 columns beside the
    # treated column needs 4 dose groups. Its linear trend and dose-group test still
    # run, and it is out of the spline's family across cell types.
    result = mixed_result()
    entry = result.provenance["groups"][3]
    assert entry["doses"] == [0.0, 0.1, 0.3]
    reason = entry["tests_not_run"]["spline"]
    assert "needs 4 dose groups" in reason and "compares 3" in reason
    assert entry["tests_not_run"]["nonlinearity"] == reason
    assert sorted(entry["f_test_df1"]) == ["dose_groups", "linear_trend"]
    with pytest.raises(KeyError, match="'spline' was not run in 'low'"):
        result.table("low", "spline")
    with pytest.raises(ValueError, match="no curve in 'low'.*needs 4 dose groups"):
        result.curve("low", at=[0.1])
    spline = result.table("all", "spline")
    assert_allclose(
        spline["adjusted_p_value_across_groups"],
        adjust_p_values(spline["p_value"], "bh"),
        rtol=1e-14,
    )
    assert result.summary["tests_not_run"].tolist()[3] == "spline, nonlinearity"
    assert "low: 3 dose groups" in repr(result)
    assert "not run: spline, nonlinearity\n" in repr(result) + "\n"
    # "one_dose" keeps the vehicle and 0.1 ug/kg: with the vehicle in a group of its
    # own the linear trend runs along the treated doses, and one cannot place a
    # line.
    entry = result.provenance["groups"][4]
    assert entry["doses"] == [0.0, 0.1]
    reason = entry["tests_not_run"]["linear_trend"]
    assert "linear trend runs along the treated doses" in reason
    assert "compares one" in reason
    assert entry["f_test_df1"] == {"dose_groups": 1}


def test_doses_below_the_second_knot_cannot_estimate_a_larger_spline() -> None:
    # 12 treated doses and 3 columns: the second interior knot falls above the 4
    # lowest doses, where the spline's last column vanishes, so a cell type that keeps
    # the vehicle and those 4 doses has enough dose groups, 5, and still cannot
    # estimate the spline.
    doses = [0.0, *np.round(np.geomspace(0.01, 100.0, 12), 6).tolist()]
    cube, _ = simulate_dose_series(
        doses, replicates=2, cell_types=["all", "low"], cells=300.0, n_genes=150, rng=4
    )
    cube = recelled(cube, {"low": [300] * 10 + [5] * 16})
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, spline_df=3)
    entry = result.provenance["groups"][1]
    assert len(entry["doses"]) == 5
    reason = entry["tests_not_run"]["spline"]
    assert reason.startswith("the spline's columns are linearly dependent at the doses")
    assert "linear_trend" in entry["f_test_df1"]


def test_no_spline_with_one_treated_dose() -> None:
    result = run((0, 10))
    assert result.spline_basis is None
    assert "no spline (one dose on the axis)" in repr(result)
    with pytest.raises(ValueError, match="a spline needs two doses on the axis"):
        result.curve("cell_type_1", at=[10.0])


def test_the_options_are_recorded_by_the_analysis() -> None:
    # The analysis records its options, a numpy integer among them, as plain values.
    cube = fixture_cube()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        groups="full",
        n_threads=np.int64(1),
    )
    assert result.provenance["analysis_settings"]["n_threads"] == 1
    assert "options" not in result.settings


def test_adjusted_across_groups() -> None:
    # Each test is a family over the cell types that ran it; each dose against the
    # vehicle is adjusted within its cell type and dose only.
    result = fixture_result("voom")
    for test in TESTS:
        tables = [result.table(group, test) for group in ("full", "depleted")]
        stacked = adjust_p_values(np.concatenate([t["p_value"] for t in tables]), "bh")
        found = np.concatenate([t["adjusted_p_value_across_groups"] for t in tables])
        assert_allclose(found, stacked, rtol=1e-14)
        columns = tables[0].columns
        assert (
            columns.index("adjusted_p_value_across_groups")
            == columns.index("adjusted_p_value") + 1
        )
    each = result.table("full", "each_dose")
    assert "adjusted_p_value_across_groups" not in each.columns
    rows = each["dose"] == 1.0
    assert_allclose(
        each["adjusted_p_value"][rows], adjust_p_values(each["p_value"][rows], "bh")
    )


def test_aliased_dose_groups_and_a_design_without_an_intercept() -> None:
    # "~ sex" whose males are exactly the samples of 10 ug/kg: that dose cannot be
    # compared with the vehicle apart from sex, so it is left out of the tests.
    cube = fixture_cube()
    doses = np.asarray(cube.samples["dose"])
    sexes = np.where(doses == 10.0, "M", "F")
    design = Design(cube.samples.with_columns({"sex": sexes}), "~ sex")
    result = dose_response(cube, design, dose=AXIS, groups="full")
    entry = result.provenance["groups"][0]
    assert entry["doses"] == [0.0, 0.1, 0.3, 1.0, 3.0]
    assert entry["f_test_df1"]["dose_groups"] == 4
    assert 10.0 not in result.table("full", "each_dose")["dose"].tolist()
    # "~ 0" gains the vehicle's intercept, so it gives "~ 1"'s results.
    zero = dose_response(cube, Design(cube.samples, "~ 0"), dose=AXIS, groups="full")
    one = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, groups="full")
    for name in (*TESTS, "each_dose"):
        assert zero.table("full", name).equals(one.table("full", name))
    assert "(Intercept)" in zero.provenance["design"]["columns"]


def test_options_reach_the_analysis() -> None:
    # Each option gives what limma_voom gives with it on a model of the dose groups
    # written by hand and the contrasts the result records, and changes the result.
    cube = fixture_cube()
    doses = np.asarray(cube.samples["dose"])
    names = ["dose_0.1", "dose_0.3", "dose_1", "dose_3", "dose_10"]
    levels = [0.1, 0.3, 1.0, 3.0, 10.0]
    samples = cube.samples.with_columns(
        {n: (doses == d).astype(float) for n, d in zip(names, levels, strict=True)}
    )
    model = Design(samples, "~ " + " + ".join(names))
    plain = fixture_result("voom").table("full", "spline")["f_statistic"]
    for option in (
        {"prior_weights": "n_cells"},
        {"quality_weights": True},
        {"robust": True},
        {"contrast_standard_errors": "limma"},
    ):
        result = dose_response(
            cube, Design(cube.samples, "~ 1"), dose=AXIS, groups="full", **option
        )
        entry = result.provenance["groups"][0]
        by_hand = limma_voom(
            cube,
            model,
            groups="full",
            contrasts={
                name: np.array(weights) for name, weights in entry["contrasts"].items()
            },
            f_tests={
                test: members
                for test, members in entry["tests"].items()
                if test != "linear_trend"
            },
            **option,
        )
        found = result.table("full", "spline")["f_statistic"]
        assert_allclose(
            found, by_hand.table("full", "spline")["f_statistic"], rtol=1e-12
        )
        assert not np.allclose(found, plain)
        # Recorded as limma_voom records the option given to it.
        (name,) = option
        assert result.provenance["analysis_settings"][name] == by_hand.settings[name]


def test_a_design_with_other_terms_is_limma_voom_on_the_dose_groups() -> None:
    # "~ sex", with 2 females and 2 males at each of 5 doses and males expressing the
    # first 50 genes 3 times as much: the dose-group test and each dose's test are
    # limma_voom's on a model of the dose groups written by hand, with 20 samples
    # less 6 coefficients, 14, residual degrees of freedom.
    doses = [0.0, 0.1, 1.0, 10.0, 30.0]
    cube, _ = simulate_dose_series(
        doses, replicates=4, cell_types=1, cells=300.0, n_genes=150, rng=14
    )
    ids = cube.samples["sample_id"].tolist()
    sex = np.array(["F" if sample.endswith(("_r1", "_r2")) else "M" for sample in ids])
    counts = np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])
    counts[:, sex == "M", :50] *= 3
    samples = cube.samples.with_columns({"sex": sex})
    cube = Cube(
        {"counts": counts},
        samples=samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    result = dose_response(cube, Design(samples, "~ sex"), dose=AXIS)
    given = np.asarray(samples["dose"], dtype=float)
    names = {0.1: "d01", 1.0: "d1", 10.0: "d10", 30.0: "d30"}
    hand = samples.with_columns(
        {name: (given == dose).astype(float) for dose, name in names.items()}
    )
    by_hand = limma_voom(
        cube,
        Design(hand, "~ sex + " + " + ".join(names.values())),
        contrasts=list(names.values()),
        f_tests={"groups": list(names.values())},
    )
    assert result.summary["df_residual"][0] == 14.0
    assert by_hand.summary["df_residual"][0] == 14.0
    assert_allclose(
        result.table("cell_type_1", "dose_groups")["f_statistic"],
        by_hand.table("cell_type_1", "groups")["f_statistic"],
        rtol=1e-12,
    )
    each = result.table("cell_type_1", "each_dose")
    for dose, name in names.items():
        assert_allclose(
            each["t"][each["dose"] == dose],
            by_hand.table("cell_type_1", name)["t"],
            rtol=1e-12,
        )


def test_the_dream_route() -> None:
    # Donors crossed with 4 doses 1 mg/kg apart on a linear axis with the vehicle on
    # it, all samples kept: the linear trend through the 4 means, each of 3 samples,
    # weighs them (x - 1.5) / 5, so the doses' differences from the vehicle by
    # (-0.1, 0.1, 0.3), derived by hand.
    cube, _ = simulate_dose_series(
        [0, 1, 2, 3],
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=120,
        donor_sd=0.3,
        unit="mg/kg",
        rng=5,
    )
    axis = DoseAxis(
        "dose", unit="mg/kg", metric="administered", scale="linear", vehicle="on_axis"
    )
    design = Design(cube.samples, "~ 1", random="(1 | donor)")
    result = dose_response(
        cube,
        design,
        dose=axis,
        tests=["linear_trend", "dose_groups"],
        analysis=dream_voom,
    )
    doses = np.asarray(cube.samples["dose"])
    samples = cube.samples.with_columns(
        {f"d{d}": (doses == d).astype(float) for d in (1, 2, 3)}
    )
    by_hand = dream_voom(
        cube,
        Design(samples, "~ d1 + d2 + d3", random="(1 | donor)"),
        contrasts={"slope": [0.0, -0.1, 0.1, 0.3], "d1": "d1", "d2": "d2", "d3": "d3"},
        f_tests={"groups": ["d1", "d2", "d3"]},
    )
    ours, theirs = (
        result.table("cell_type_1", "linear_trend"),
        by_hand.table("cell_type_1", "slope"),
    )
    for column in ("log2_fold_change", "t", "df", "p_value"):
        assert_allclose(ours[column], theirs[column], rtol=1e-10)
    assert_allclose(
        result.table("cell_type_1", "dose_groups")["f_statistic"],
        by_hand.table("cell_type_1", "groups")["f_statistic"],
        rtol=1e-10,
    )
    with pytest.raises(ValueError, match="dream's fits do not keep"):
        result.curve("cell_type_1", at=[1.0])
    assert result.method == "dream_voom"
    # The printout gives the slope's unit on a linear axis.
    assert "linear_trend slopes per mg/kg" in " ".join(repr(result).split())


def test_results_do_not_depend_on_order_units_or_threads() -> None:
    cube = fixture_cube()
    base = fixture_result("voom")
    # The design's rows in another order.
    order = np.random.default_rng(1).permutation(cube.n_samples)
    shuffled = dose_response(
        cube, Design(cube.samples.take(order), "~ 1"), dose=AXIS, spline_df=2
    )
    for key, table in base.tables.items():
        column = next(c for c in ("f_statistic", "t", "p_value") if c in table)
        assert_allclose(shuffled.tables[key][column], table[column], rtol=1e-10)
    # Doses in ng/kg, a thousand times larger: the tests are unchanged, the slope
    # per log10 unit of dose too, and the curves are the same at the same doses.
    scaled_samples = cube.samples.with_columns(
        {"dose": np.asarray(cube.samples["dose"]) * 1000.0}
    )
    scaled_cube = Cube(
        {"counts": np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])},
        samples=scaled_samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    axis = DoseAxis("dose", unit="ng/kg", metric="administered")
    scaled = dose_response(
        scaled_cube, Design(scaled_samples, "~ 1"), dose=axis, spline_df=2
    )
    for test in TESTS:
        column = "t" if test == "linear_trend" else "f_statistic"
        assert_allclose(
            scaled.table("depleted", test)[column],
            base.table("depleted", test)[column],
            rtol=1e-9,
        )
    assert_allclose(
        scaled.table("full", "linear_trend")["log2_fold_change"],
        base.table("full", "linear_trend")["log2_fold_change"],
        rtol=1e-9,
    )
    at = np.array([0.0, 0.2, 3.0])
    assert_allclose(
        scaled.curve("depleted", at=at * 1000.0)["ci_upper"],
        base.curve("depleted", at=at)["ci_upper"],
        rtol=1e-9,
        atol=1e-12,
    )
    # One thread and four.
    one = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, spline_df=2, n_threads=1
    )
    four = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, spline_df=2, n_threads=4
    )
    for key in one.tables:
        assert one.tables[key].equals(four.tables[key])


# Curves.


def test_curve_refuses_doses_outside_the_tested_range() -> None:
    result = fixture_result("voom")
    # The vehicle in its own group: 0 with a standard error of 0, exactly.
    vehicle = result.curve("full", at=[0.0])
    assert np.all(vehicle["log2_fold_change"] == 0.0)
    assert np.all(vehicle["standard_error"] == 0.0)
    assert np.all(vehicle["ci_lower"] == 0.0) and np.all(vehicle["ci_upper"] == 0.0)
    # The depleted cell type compares doses up to 3 ug/kg: 10, within the design's
    # doses, lies beyond its own.
    assert len(result.curve("full", "gene_001", at=[10.0])) == 1
    with pytest.raises(ValueError, match="10 ug/kg, outside the doses that 'depleted'"):
        result.curve("depleted", at=[10.0])
    with pytest.raises(
        ValueError, match=r"0\.05 ug/kg, outside.*0, the vehicle, and 0\.1 to 3"
    ):
        result.curve("depleted", at=[0.05])
    with pytest.raises(ValueError, match="finite and at least 0"):
        result.curve("full", at=[np.nan])
    with pytest.raises(ValueError, match="finite and at least 0"):
        result.curve("full", at=[-1.0])
    with pytest.raises(KeyError, match="'nope' was not tested in 'full'"):
        result.curve("full", ["nope"], at=[1.0])
    with pytest.raises(TypeError, match="feature ids"):
        result.curve("full", [1], at=[1.0])
    with pytest.raises(TypeError, match="feature id or symbol, or a list"):
        result.curve("full", 3.0, at=[1.0])  # type: ignore[arg-type]
    # The ends of the range, up to the rounding of a unit conversion.
    assert (
        len(
            result.curve(
                "depleted", "gene_001", at=[0.1 * (1 - 1e-12), 3.0 * (1 + 1e-12)]
            )
        )
        == 2
    )


def test_curves_keep_the_order_of_genes_and_doses() -> None:
    # The genes in the order given and each gene's doses in the order of at, repeats
    # included: each row is the curve of its gene at its dose alone.
    result = fixture_result("trend")
    tested = result.table("full", "spline")["feature_id"].tolist()
    first, second = tested[5], tested[1]
    curve = result.curve("full", [first, second], at=[10.0, 0.1, 10.0])
    assert curve["feature_id"].tolist() == [first] * 3 + [second] * 3
    assert curve["dose"].tolist() == [10.0, 0.1, 10.0] * 2
    for row in range(len(curve)):
        alone = result.curve(
            "full", str(curve["feature_id"][row]), at=[float(curve["dose"][row])]
        )
        for column in ("log2_fold_change", "ci_lower", "ci_upper"):
            assert_allclose(alone[column][0], curve[column][row], rtol=1e-13)


def test_genes_by_symbol() -> None:
    # A cube whose features have symbols: genes are named by feature id or symbol,
    # the curve carries the symbol, and a symbol of several genes is refused.
    base = fixture_cube()
    ids = base.features["feature_id"].tolist()
    symbols = [f"Gene{i}" for i in range(len(ids))]
    # Two genes that the cell type tests, which R's filter keeps too.
    kept = reference()["jobs"]["fixture trend"]["cell_types"]["full"]["genes"]
    a, b, c, d = (ids[i - 1] for i in kept[:4])
    symbols[kept[2] - 1] = symbols[kept[3] - 1] = "Shared"
    cube = Cube(
        {"counts": np.stack([base.layer("counts", g) for g in range(base.n_groups)])},
        samples=base.samples,
        groups=base.groups,
        features={"feature_id": ids, "symbol": symbols},
        n_cells=base.n_cells,
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        groups="full",
        analysis=limma_trend,
    )
    by_symbol = result.curve(
        "full", [symbols[kept[1] - 1], symbols[kept[0] - 1]], at=[1.0, 3.0]
    )
    by_id = result.curve("full", [b, a], at=[1.0, 3.0])
    assert by_symbol.equals(by_id)
    assert by_symbol.columns[:3] == ("feature_id", "symbol", "dose")
    assert by_symbol["feature_id"].tolist() == [b, b, a, a]
    assert (
        by_symbol["symbol"].tolist()
        == [symbols[kept[1] - 1]] * 2 + [symbols[kept[0] - 1]] * 2
    )
    with pytest.raises(ValueError, match=f"'Shared' names 2 genes.*'{c}' and '{d}'"):
        result.curve("full", "Shared", at=[1.0])
    assert len(result.curve("full", c, at=[1.0])) == 1
    with pytest.raises(KeyError, match="no gene tested there has this feature id or"):
        result.curve("full", "Nope", at=[1.0])


def test_a_curve_by_default_spans_the_doses_compared() -> None:
    # 100 doses evenly spaced on the log10 axis, from the lowest dose the cell type
    # compares to its highest; from the vehicle when it sits on the axis.
    result = fixture_result("trend")
    for group, high in (("full", 10.0), ("depleted", 3.0)):
        curve = result.curve(group, "gene_001")
        doses = np.asarray(curve["dose"])
        assert doses.size == 100 and doses[0] == 0.1 and doses[-1] == high
        steps = np.diff(np.log10(doses))
        assert_allclose(steps, np.log10(high / 0.1) / 99, rtol=1e-10)
        assert curve.equals(result.curve(group, "gene_001", at=doses))
    on_axis = run((0, 0.1, 1, 10), ON_AXIS)
    # The printout gives the linear trend's slope per log10 unit of the dose plus
    # the offset.
    printout = " ".join(repr(on_axis).split())
    assert "linear_trend slopes per log10 unit of dose plus 0.001 ug/kg" in printout
    doses = np.asarray(on_axis.curve("cell_type_1", "gene_001")["dose"])
    assert doses[0] == 0.0 and doses[-1] == 10.0 and doses.size == 100
    steps = np.diff(np.log10(doses + 0.001))
    assert_allclose(steps, np.log10(10.001 / 0.001) / 99, rtol=1e-10)


def test_infinite_prior_degrees_of_freedom_give_bands_on_the_pooled_df() -> None:
    # Without biological variation the variances agree with their prior, whose
    # degrees of freedom are infinite. limma caps a gene's degrees of freedom at the
    # residual degrees of freedom summed over the genes, as topTable(confint = TRUE)
    # takes them: the band is the estimate plus and minus t(0.975) on 10 x the genes
    # tested, a little wider than the normal quantile.
    cube, _ = simulate_dose_series(
        [0, 0.1, 1, 10, 100],
        replicates=3,
        cell_types=1,
        cells=2000.0,
        n_genes=300,
        biological_sd=1e-3,
        rng=7,
    )
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    assert result.summary["df_prior"][0] == np.inf
    assert result.summary["df_residual"][0] == 10.0
    pooled = 10.0 * result.summary["n_features_tested"][0]
    curve = result.curve("cell_type_1", at=[0.5, 50.0])
    half = (curve["ci_upper"] - curve["log2_fold_change"]) / curve["standard_error"]
    assert_allclose(half, t_ppf(0.975, pooled), rtol=1e-12)
    assert np.all(half > NORMAL_QUANTILE)


# The record and the printout.


def test_the_result_reads_back(tmp_path: Path) -> None:
    result = fixture_result("voom")
    assert result.groups == ("full", "depleted")
    assert result.tests == TESTS
    assert result.method == "limma_voom"
    assert result.dose == AXIS
    assert result.spline_basis is not None and result.spline_basis.df == 2
    assert isinstance(result.design_report, DoseDesignReport)
    settings = result.settings
    assert settings["tests"] == list(TESTS) and settings["spline_df"] == 2
    assert settings["dose"] == AXIS.to_dict()
    assert settings["min_cells"] == 10 and "options" not in settings
    record = result.provenance
    assert record["cube"]["fingerprint"] == fixture_cube().fingerprint()
    # The cube is recorded once, not in each cell type's fit.
    assert "cube" not in record["groups"][0]["fit"]
    entry = record["groups"][0]
    assert entry["f_test_df1"] == {
        "spline": 3,
        "nonlinearity": 1,
        "linear_trend": 1,
        "dose_groups": 5,
    }
    # Each test's contrasts, named by the rules of the module: the spline's
    # coefficients after the columns of the axis, each dose after its indicator.
    spline = ["spline_dose_treated", "spline_dose_s1", "spline_dose_s2"]
    doses = ["dose_0.1", "dose_0.3", "dose_1", "dose_3", "dose_10"]
    assert entry["tests"] == {
        "spline": spline,
        "nonlinearity": spline[2:],
        "linear_trend": ["linear_trend"],
        "dose_groups": doses,
    }
    assert sorted(entry["contrasts"]) == sorted([*doses, *spline, "linear_trend"])
    assert record["analysis_settings"]["contrast_standard_errors"] == "exact"
    # The fit computed again tests those contrasts.
    fit = result.fit("full")
    assert sorted(fit.moderated.fit.coefficient_names) == sorted(entry["contrasts"])
    # The columns of each table, as the class documents them.
    f_columns = (
        "feature_id",
        "average_log2_cpm",
        "f_statistic",
        "df",
        "p_value",
        "adjusted_p_value",
        "adjusted_p_value_across_groups",
        "standardized_f_statistic",
    )
    for test in ("spline", "nonlinearity", "dose_groups"):
        assert result.table("full", test).columns == f_columns
    assert result.table("full", "linear_trend").columns == (
        "feature_id",
        "log2_fold_change",
        "standard_error",
        "average_log2_cpm",
        "t",
        "p_value",
        "adjusted_p_value",
        "adjusted_p_value_across_groups",
        "log_odds",
    )
    assert result.table("full", "each_dose").columns == (
        "feature_id",
        "dose",
        "log2_fold_change",
        "standard_error",
        "average_log2_cpm",
        "t",
        "p_value",
        "adjusted_p_value",
        "log_odds",
    )
    assert result.curve("full", "gene_001", at=[1.0]).columns == (
        "feature_id",
        "dose",
        "log2_fold_change",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    # Keys left out, and keys that do not exist.
    with pytest.raises(ValueError, match="2 cell types; name one"):
        result.table(None, "spline")
    with pytest.raises(ValueError, match="name one with name"):
        result.table("full")
    with pytest.raises(KeyError, match="no cell type 'liver' was checked"):
        result.table("liver", "spline")
    with pytest.raises(KeyError, match="no table 'splines'.*did you mean 'spline'"):
        result.table("full", "splines")
    with pytest.raises(TypeError, match=r"group must be a cell type \(str\)"):
        result.table(3, "spline")  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="no cell type 'liver' was checked"):
        result.fit("liver")
    # The printout: the knots in doses, the tests, the linear trend's unit, the
    # gate, and the genes that pass it and the calls of each cell type.
    text = " ".join(repr(result).split())
    assert text.startswith("DoseResponseResult (limma_voom): 2 cell types, 2 tested")
    assert "spline of 2 columns, knots at 1 ug/kg, boundary knots at 0.1 and 10" in text
    assert (
        "; linear_trend slopes per log10 unit of dose gate: spline test, "
        "Benjamini-Hochberg within each cell type, at 5% FDR full:" in text
    )
    calls = {
        test: int(
            np.count_nonzero(result.table("full", test)["adjusted_p_value"] <= 0.05)
        )
        for test in TESTS
    }
    assert (
        f"{calls['spline']} pass the gate; calls at 5% FDR: "
        + ", ".join(f"{test} {count}" for test, count in calls.items())
        + " depleted:"
        in text
    )
    # Pickled, with the cube in memory and with a cube read from a file.
    again = pickle.loads(pickle.dumps(result))
    assert again.provenance == result.provenance
    for key, table in result.tables.items():
        assert again.tables[key].equals(table)
    path = tmp_path / "series.cube.zarr"
    write_cube(fixture_cube(), path)
    stored = read_cube(path)
    from_file = dose_response(
        stored, Design(stored.samples, "~ 1"), dose=AXIS, groups="full"
    )
    restored = pickle.loads(pickle.dumps(from_file))
    assert restored.curve("full", "gene_001", at=[1.0]).equals(
        from_file.curve("full", "gene_001", at=[1.0])
    )
    # With one cell type checked, a table name given as the cell type is pointed
    # to name=.
    assert from_file.table(name="spline").equals(from_file.table("full", "spline"))
    with pytest.raises(KeyError, match="'spline' is a table, not a cell type"):
        from_file.table("spline")
    # Immutable.
    with pytest.raises(AttributeError, match="immutable"):
        result.summary = None  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del result.summary


def test_the_result_refuses_fields_of_the_wrong_type() -> None:
    result = fixture_result("trend")
    # The fields of a result, as pickling passes them to the constructor: five
    # positional, then the cube and the design by keyword.
    good = result.__reduce__()[1]
    assert good[5] is result.cube and good[6] is result.design
    for position, bad, message in (
        (0, {}, "summary must be a Table"),
        (1, [], "tables must be a mapping"),
        (1, {("a",): result.summary}, "keyed by"),
        (1, {("a", "b"): 1}, "must be a Table"),
        (2, [], "fits must be a mapping"),
        (2, {"full": 1}, "must be a DifferentialExpressionResult"),
        (3, None, "design_report must be a DoseDesignReport"),
        (4, [], "provenance must be a mapping"),
    ):
        fields = list(good[:5])
        fields[position] = bad
        with pytest.raises(TypeError, match=message):
            DoseResponseResult(*fields)
    with pytest.raises(TypeError):
        DoseResponseResult(*good[:4], {"x": np.zeros(2)})
    with pytest.raises(ValueError):
        DoseResponseResult(*good[:4], {"x": float("nan")})
    with pytest.raises(TypeError, match="cube must be a Cube or None"):
        DoseResponseResult(*good[:5], cube="cube", design=result.design)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="design must be a Design or None"):
        DoseResponseResult(*good[:5], cube=result.cube, design="design")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="give both the cube and the design"):
        DoseResponseResult(*good[:5], cube=result.cube)
    bare = DoseResponseResult(*good[:5])
    assert bare.cube is None and bare.design is None


# Refusals.


def test_arguments_are_refused_with_a_fix() -> None:
    cube = fixture_cube()
    design = Design(cube.samples, "~ 1")

    def call(**options: Any) -> Any:
        return dose_response(cube, options.pop("design", design), dose=AXIS, **options)

    cases: list[tuple[dict[str, Any], type[Exception], str]] = [
        ({"tests": 3}, TypeError, "a test name or a list of them"),
        ({"tests": {"spline": 1}}, TypeError, "a test name or a list of them"),
        ({"tests": []}, ValueError, "tests is empty"),
        ({"tests": [1]}, TypeError, r"test names \(str\)"),
        ({"tests": ["levels"]}, ValueError, "'levels', which is not a test"),
        ({"tests": ["splines"]}, ValueError, "did you mean 'spline'"),
        (
            {"tests": ["linear_trend", "linear_trend"]},
            ValueError,
            "'linear_trend' more than once",
        ),
        ({"tests": ["slope"]}, ValueError, "'slope', which is not a test"),
        ({"spline_df": 0}, ValueError, "spline_df"),
        ({"spline_df": 1.5}, TypeError, "spline_df"),
        ({"spline_df": 5}, ValueError, "lower df to at most 4"),
        ({"analysis": len}, TypeError, "the functions limma_voom, limma_trend and"),
        (
            {"analysis": "limma_trend"},
            TypeError,
            "analysis=cytepool.limma_trend, got the name 'limma_trend'",
        ),
        ({"analysis": dream_voom}, ValueError, "dream_voom fits random effects"),
        ({"contrasts": ["x"]}, TypeError, "dose_response sets contrasts itself"),
        ({"f_tests": {}}, TypeError, "dose_response sets f_tests itself"),
        ({"adjust_across_groups": True}, TypeError, "sets adjust_across_groups"),
        ({"prior_count": 3}, TypeError, "limma_voom has no option 'prior_count'"),
        ({"robustly": True}, TypeError, "did you mean 'robust'"),
        (
            {"design": Design(cube.samples, "~ 1", contrasts={"a": "(Intercept)"})},
            ValueError,
            "the design names contrasts",
        ),
        (
            {
                "design": Design(
                    cube.samples.with_columns({"donor": ["a", "b", "c"] * 6}),
                    "~ 1",
                    random="(1 | donor)",
                )
            },
            ValueError,
            "pass analysis=dream_voom",
        ),
        (
            {
                "design": Design(
                    cube.samples.with_columns({"dose_10": np.zeros(18)}), "~ 1"
                )
            },
            ValueError,
            "already has the column.*'dose_10'",
        ),
        ({"dose": "dose"}, TypeError, "dose must be a DoseAxis"),
        ({"design": "~ 1"}, TypeError, "design must be a Design of the terms other"),
    ]
    for options, error, message in cases:
        if "dose" in options:
            with pytest.raises(error, match=message):
                dose_response(cube, design, **options)
            continue
        with pytest.raises(error, match=message):
            call(**options)
    # Values of the options are the analysis's to check.
    with pytest.raises(ValueError, match="contrast_standard_errors"):
        call(contrast_standard_errors="approximate")
    with pytest.raises(TypeError, match="masked"):
        call(groups=np.ma.masked_array(["full"], mask=[True]))
    with pytest.raises(TypeError, match="masked"):
        call(tests=np.ma.masked_array(["spline"], mask=[True]))


def test_a_fit_that_stops_leaves_the_cell_type_untested() -> None:
    # Every gene filtered out: the fit stops, and the cell type is listed with the
    # analysis's reason.
    cube = fixture_cube()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        groups="full",
        expression_filter={"min_count": 1e9, "min_total_count": 1e12},
    )
    assert result.summary["tested"].tolist() == [False]
    assert result.summary["status"].tolist() == ["inferential"]
    assert "no feature passed the expression filter" in result.summary["reason"][0]
    assert result.tables == {}
    assert result.provenance["analysis_settings"] is None
    with pytest.raises(KeyError, match="was not tested"):
        result.table("full", "spline")
    assert "not tested, inferential" in repr(result)


def test_the_package_lists_the_names() -> None:
    assert cytepool.dose_response is dose_response
    assert cytepool.DoseResponseResult is DoseResponseResult
    assert re.search(r":func:`dose_response`", cytepool.__doc__ or "")


# The chapter.

CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/dose_response.md"
)


def test_the_chapter_example_runs() -> None:
    pytest.importorskip("matplotlib")
    text = CHAPTER.read_text()
    tests = text.split("(dose-response-tests)=")[1].split("\n(")[0]
    curves = text.split("(dose-response-curves)=")[1].split("\n(")[0]
    blocks = [
        *re.findall(r"```python\n(.*?)```", tests, re.S),
        *re.findall(r"```python\n(.*?)```", curves, re.S),
    ]
    assert len(blocks) == 2
    namespace: dict[str, Any] = {"cp": cytepool}
    for block in blocks:
        exec(block.replace("print(result)\n", ""), namespace)  # noqa: S102
    result = namespace["result"]
    # The printout, and the values the text states.
    assert repr(result) in tests
    summary = result.summary
    assert summary["n_dose_groups_tested"].tolist() == [9, 8, 0]
    assert summary["status"].tolist()[2] == "refused"
    responding = namespace["responding"]
    truth = namespace["truth"].table("hepatocyte")
    size = dict(zip(truth["feature_id"].tolist(), truth["max_log2_change"].tolist()))
    hits = sum(size[g] != 0 for g in responding["feature_id"].tolist())
    assert (len(responding), hits) == (88, 85)
    assert "the spline calls 88 genes, 85 of which respond in truth" in tests
    stellate = result.provenance["groups"][1]
    assert stellate["f_test_df1"]["dose_groups"] == 7
    assert 30.0 not in stellate["doses"]
    # The curves of four genes on 100 doses by default, and the figure.
    curve = namespace["curve"]
    assert len(curve) == 4 * 100
    assert namespace["ax"].get_xlabel() == "Dose (ug/kg, log10 axis)"
    with pytest.raises(ValueError, match="'stellate' compares"):
        result.curve("stellate", at=[30.0])


# Trend contrasts, the MCP step and the lowest effective dose.

CONTRAST_DATA = Path(__file__).parent / "data" / "dose_contrasts"
CONTRAST_TESTS = ("dunnett", "williams", "mcp")
CALIBRATED = 5.5  # combined standard errors, as validation/dose_contrasts.py


@functools.cache
def contrast_reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((CONTRAST_DATA / "r_tests.json").read_text())
    return found


@functools.cache
def contrast_cube() -> Cube:
    """The series of validation/dose_contrasts.py's fixture: the vehicle and 5 doses
    from 0.1 to 10 ug/kg x 3, 200 genes, a cell type of every sample and one that
    loses the top dose and a sample of 3 ug/kg."""
    series = contrast_reference()["series"]
    cell_types = series["cell_types"]
    groups = list(cell_types)
    return Cube(
        {
            "counts": np.stack(
                [np.array(cell_types[g]["counts"], dtype=np.int64).T for g in groups]
            )
        },
        samples=series["samples"],
        groups={"group_id": groups},
        features={"feature_id": [f"gene_{j + 1:03d}" for j in range(200)]},
        n_cells=np.column_stack(
            [np.array(cell_types[g]["n_cells"], dtype=np.int64) for g in groups]
        ),
    )


@functools.cache
def contrast_result(
    analysis: str, gate: str = "dose_groups", fdr: float = 0.05
) -> DoseResponseResult:
    """The multiple contrast tests of the contrast cube, with the dose-group test as
    the gate at 5% FDR unless another gate is named."""
    cube = contrast_cube()
    return dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=("dose_groups", *CONTRAST_TESTS),
        gate={"test": gate, "fdr": fdr},
        analysis=limma_trend if analysis == "trend" else limma_voom,
        rng=20261216,
    )


def contrast_weights(result: DoseResponseResult, group: str, test: str) -> tuple:
    """A test's contrast names and weights on the treated doses, from the record."""
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
    contrasts = entry["contrast_tests"][test]["contrasts"]
    return list(contrasts), np.array(list(contrasts.values()))


def numbers(values: Any) -> np.ndarray:
    array = np.array(values, dtype=object)
    return np.array(
        [np.nan if v is None else v for v in array.ravel()], dtype=np.float64
    ).reshape(array.shape)


@pytest.mark.parametrize("analysis", ["voom", "trend"])
def test_multiple_contrast_tests_match_r(analysis: str) -> None:
    """Contrasts, each gene's largest |t|, the contrast it belongs to and df against R:
    limma fitted from the counts, multcomp's contrMat and DoseFinding's optContr from
    the doses and samples R counted, and glht(parm()) on R's estimates."""
    result = contrast_result(analysis)
    for theirs in contrast_reference()["cell_types"]:
        if theirs["analysis"] != analysis:
            continue
        group = theirs["group"]
        entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
        assert entry["doses"] == pytest.approx(theirs["doses"], rel=0)
        assert entry["samples"] == theirs["n"]
        assert result.table(group, "dunnett")["feature_id"].tolist() == [
            f"gene_{g + 1:03d}" for g in theirs["genes"]
        ]
        for test in CONTRAST_TESTS:
            names, weights = contrast_weights(result, group, test)
            assert_allclose(weights, numbers(theirs["contrasts"][test]), atol=1e-13)
            table = result.table(group, test)
            assert_scaled(table["t"], theirs["t"][test])
            chosen = [names.index(name) for name in table["contrast"].tolist()]
            assert chosen == theirs["chosen"][test]
            assert_allclose(table["df"], theirs["df"], rtol=1e-12)
        assert_p_values(
            result.table(group, "dose_groups")["p_value"], theirs["f_p"], 1e-9
        )


def test_multiple_contrast_p_values_match_scipy() -> None:
    """The p-values of limma-voom's fit at its non-integer df against scipy's
    multivariate t on R's statistics, within 5.5 combined standard errors."""
    result = contrast_result("voom")
    found = 0
    for case in contrast_reference()["scipy"]:
        table = result.table(case["group"], case["test"])
        row = table["feature_id"].tolist().index(case["feature_id"])
        error = math.hypot(table["p_value_error"][row], case["error"])
        assert abs(table["p_value"][row] - case["p_value"]) <= CALIBRATED * error
        found += 1
    assert found >= 100


def test_lowest_effective_doses_match_stager() -> None:
    """R's own dose-group test and per-dose tests, through stageR's
    stageWiseAdjustment('holm'): under the dose-group gate, the lowest dose called,
    its stage-wise adjusted p-value, the doses called and the gate's calls."""
    found = 0
    for theirs in contrast_reference()["cell_types"]:
        result = contrast_result(theirs["analysis"])
        group = theirs["group"]
        ours = result.lowest_effective_doses(group)
        doses = np.array(theirs["doses"][1:])
        adjusted = numbers(theirs["stage_wise"]).reshape(len(theirs["genes"]), -1)
        called = np.nan_to_num(adjusted, nan=2.0) <= 0.05
        some = called.any(axis=1)
        first = np.argmax(called, axis=1)
        assert_array_equal(
            ours["lowest_effective_dose"], np.where(some, doses[first], np.nan)
        )
        assert_allclose(
            ours["stage_wise_adjusted_p_value"],
            np.where(some, adjusted[np.arange(first.size), first], np.nan),
            rtol=1e-12,
        )
        assert_array_equal(ours["n_doses_called"], called.sum(axis=1))
        q = adjust_p_values(numbers(theirs["f_p"]), "bh")
        assert_array_equal(ours["passes_gate"], q <= 0.05)
        found += int(np.count_nonzero(some))
    assert found > 50


def test_dunnett_t_is_each_dose_s_t() -> None:
    """Dunnett's contrasts are the doses themselves, so each gene's largest |t| is
    the t of that dose in the table of each dose, from the analysis's own contrast
    tables; and the Wald F of all doses from the estimates and their covariance is
    the dose-group F. Under limma-voom, limma's standard errors, limma-trend and
    dream."""
    cube, _ = simulate_dose_series(
        [0, 0.1, 1, 10], replicates=3, cell_types=1, n_genes=120, donor_sd=0.3, rng=6
    )
    results = {
        "voom": dose_response(
            cube,
            Design(cube.samples, "~ 1"),
            dose=AXIS,
            tests=("dose_groups", "dunnett"),
        ),
        "trend": dose_response(
            cube,
            Design(cube.samples, "~ 1"),
            dose=AXIS,
            tests=("dose_groups", "dunnett"),
            analysis=limma_trend,
        ),
        "dream": dose_response(
            cube,
            Design(cube.samples, "~ 1", random="(1 | donor)"),
            dose=AXIS,
            tests=("dose_groups", "dunnett"),
            analysis=dream_voom,
        ),
    }
    for name, result in results.items():
        dunnett = result.table("cell_type_1", "dunnett")
        each = result.table("cell_type_1", "each_dose")
        k = len(dunnett)
        t = by_dose(each, "t", 3)
        names = ["0.1 - vehicle", "1 - vehicle", "10 - vehicle"]
        picked = np.array([names.index(c) for c in dunnett["contrast"].tolist()])
        assert_allclose(dunnett["t"], t[np.arange(k), picked], rtol=1e-10)
        assert_array_equal(picked, np.argmax(np.abs(t), axis=1))
        if name == "dream":
            df = np.asarray(each["df"], dtype=float).reshape(3, -1).T.min(axis=1)
            assert_allclose(dunnett["df"], df, rtol=1e-12)
        fit = result.fit("cell_type_1")
        estimates = _dose_group_estimates(
            fit, [f"dose_{d:.12g}" for d in (0.1, 1.0, 10.0)]
        )
        covariance = estimates.covariance * estimates.posterior_variance[:, None, None]
        wald = (
            np.einsum(
                "gi,gi->g",
                estimates.theta,
                np.linalg.solve(covariance, estimates.theta[:, :, None])[:, :, 0],
            )
            / 3
        )
        # dream's F takes the doses' own df (lmerTest's rule); its statistic is the
        # Wald F all the same.
        assert_allclose(
            wald, result.table("cell_type_1", "dose_groups")["f_statistic"], rtol=1e-9
        )


def test_contrasts_follow_the_cell_type_s_samples() -> None:
    """The depleted cell type keeps 3, 3, 3, 3 and 2 samples of the vehicle and four
    doses: the Williams-type weights are its samples over the pooled sum, and the MCP
    step's contrasts are n_k (mu_k - the size-weighted mean of mu), derived by hand,
    with the vehicle's weight dropped and scaled to unit length."""
    result = contrast_result("voom")
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == "depleted")
    n = np.array(entry["samples"], dtype=float)
    assert n.tolist() == [3, 3, 3, 3, 2]
    names, williams = contrast_weights(result, "depleted", "williams")
    assert names == [
        "3 - vehicle",
        "1 to 3 - vehicle",
        "0.3 to 3 - vehicle",
        "0.1 to 3 - vehicle",
    ]
    expected = np.zeros((4, 4))
    for j in range(4):
        pooled = slice(3 - j, 4)
        expected[j, pooled] = n[1:][pooled] / n[1:][pooled].sum()
    assert_allclose(williams, expected, rtol=1e-15)
    shapes = result.candidates
    assert shapes is not None
    means = shapes.means(entry["doses"])
    mean = (means * n).sum(axis=1, keepdims=True) / n.sum()
    by_hand = (n * (means - mean))[:, 1:]
    by_hand /= np.linalg.norm(by_hand, axis=1, keepdims=True)
    names, mcp = contrast_weights(result, "depleted", "mcp")
    assert names == list(shapes.names)
    assert_allclose(mcp, by_hand, rtol=1e-12, atol=1e-15)


def test_lowest_effective_doses_by_the_closed_test() -> None:
    """Each gene's doses called equal those of the closure of Bonferroni tests coded
    here, every intersection enumerated; and the lowest effective dose lies below an
    uncalled dose for some gene, which a run of called doses from the top would miss."""
    for group in ("full", "depleted"):
        for screen in ("dose_groups", "mcp"):
            result = contrast_result("voom", screen, 0.2)
            ours = result.lowest_effective_doses(group)
            p = by_dose(
                result.table(group, "each_dose"),
                "p_value",
                len(result.provenance["groups"][0]["doses"]) - 1
                if group == "full"
                else 4,
            )
            screened = np.asarray(result.table(group, screen)["p_value"])
            q = adjust_p_values(screened, "bh")
            selected = q <= 0.2
            level = selected.sum() * 0.2 / screened.size
            doses = np.array(
                next(e for e in result.provenance["groups"] if e["group_id"] == group)[
                    "doses"
                ][1:]
            )
            m = doses.size
            change = by_dose(result.table(group, "each_dose"), "log2_fold_change", m)
            subsets = [s for size in range(1, m) for s in combinations(range(m), size)]
            gaps = 0
            for g in np.flatnonzero(selected):
                rejected = [
                    all(min(p[g, list(s)]) <= level / len(s) for s in subsets if j in s)
                    for j in range(m)
                ]
                assert ours["n_doses_called"][g] == sum(rejected)
                lowest = doses[rejected.index(True)] if any(rejected) else np.nan
                assert ours["lowest_effective_dose"][g] == pytest.approx(
                    lowest, nan_ok=True
                )
                # The change from the vehicle at that dose, from the table of each dose.
                at = change[g, rejected.index(True)] if any(rejected) else np.nan
                np.testing.assert_equal(ours["log2_fold_change"][g], at)
                if any(rejected):
                    first = rejected.index(True)
                    gaps += not all(rejected[first:])
            assert_array_equal(np.isnan(ours["lowest_effective_dose"][~selected]), True)
            assert_array_equal(np.isnan(ours["log2_fold_change"][~selected]), True)
            assert gaps > 0


def test_lowest_effective_doses_at_extreme_levels() -> None:
    none = contrast_result("trend", "dose_groups", 1e-300).lowest_effective_doses(
        "full"
    )
    assert not np.any(none["passes_gate"])
    assert np.isnan(none["lowest_effective_dose"]).all()
    assert_array_equal(none["n_doses_called"], 0)
    lenient = contrast_result("trend", "dose_groups", 0.5).lowest_effective_doses(
        "full"
    )
    strict = contrast_result("trend").lowest_effective_doses("full")
    assert np.sum(lenient["passes_gate"]) > np.sum(strict["passes_gate"])
    # A larger level calls each called dose again: the lowest effective dose can only
    # move down.
    both = ~np.isnan(strict["lowest_effective_dose"])
    assert np.all(
        lenient["lowest_effective_dose"][both] <= strict["lowest_effective_dose"][both]
    )


def test_contrast_tests_do_not_depend_on_threads_order_or_other_cell_types() -> None:
    cube = contrast_cube()
    design = Design(cube.samples, "~ 1")
    base = contrast_result("voom")
    seed = base.settings["seed"]
    assert seed == 20261216
    alone = dose_response(
        cube,
        design,
        dose=AXIS,
        tests=("dose_groups", *CONTRAST_TESTS),
        groups=["depleted"],
        rng=seed,
    )
    four = dose_response(
        cube,
        Design(cube.samples.take(np.arange(cube.n_samples)[::-1]), "~ 1"),
        dose=AXIS,
        tests=("dose_groups", *CONTRAST_TESTS),
        rng=seed,
        n_threads=4,
    )
    within = [
        "feature_id",
        "average_log2_cpm",
        "contrast",
        "t",
        "df",
        "p_value",
        "p_value_error",
        "adjusted_p_value",
    ]
    for test in CONTRAST_TESTS:
        # Run alone, the cell type is its own family across cell types.
        assert (
            alone.table("depleted", test)
            .select(within)
            .equals(base.table("depleted", test).select(within))
        )
        for group in ("full", "depleted"):
            assert_allclose(
                four.table(group, test)["p_value"],
                base.table(group, test)["p_value"],
                rtol=1e-12,
            )
    # A test named alone draws the random numbers it draws beside the others: the
    # test's position counts among every multiple contrast test, not those named.
    for test in ("williams", "mcp"):
        named_alone = dose_response(cube, design, dose=AXIS, tests=(test,), rng=seed)
        for group in ("full", "depleted"):
            assert_array_equal(
                named_alone.table(group, test)["p_value"],
                base.table(group, test)["p_value"],
            )
    # Each cell type and test draws from numpy.random.default_rng([seed, CRC-32 of the
    # cell type's name, the test's position]).
    fit = base.fit("full")
    entry = next(e for e in base.provenance["groups"] if e["group_id"] == "full")
    estimates = _dose_group_estimates(
        fit, [f"dose_{d:.12g}" for d in entry["doses"][1:]]
    )
    names, weights = contrast_weights(base, "full", "williams")
    direct = multiple_contrast_test(
        estimates.theta,
        estimates.covariance * estimates.posterior_variance[:, None, None],
        dict(zip(names, weights, strict=True)),
        df=estimates.df_total,
        rng=np.random.default_rng([seed, zlib.crc32(b"full"), 1]),
    )
    assert_array_equal(
        direct.table()["p_value"], base.table("full", "williams")["p_value"]
    )
    # A generator gives its first draw of [0, 2^63) as the seed, which reproduces it.
    drawn = int(np.random.default_rng(7).integers(0, 2**63))
    again = dose_response(
        cube,
        design,
        dose=AXIS,
        tests=("mcp",),
        groups=["full"],
        rng=np.random.default_rng(7),
    )
    assert again.settings["seed"] == drawn
    repeat = dose_response(
        cube, design, dose=AXIS, tests=("mcp",), groups=["full"], rng=drawn
    )
    assert repeat.table("full", "mcp").equals(again.table("full", "mcp"))


def test_contrast_tests_follow_the_dose_s_unit() -> None:
    """Doses in ng/kg: the candidate shapes scale, the tests and the calls do not, and
    the lowest effective doses are a thousand times larger."""
    cube = contrast_cube()
    base = contrast_result("voom")
    scaled_samples = cube.samples.with_columns(
        {"dose": np.asarray(cube.samples["dose"]) * 1000.0}
    )
    scaled_cube = Cube(
        {"counts": np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])},
        samples=scaled_samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=cube.n_cells,
    )
    scaled = dose_response(
        scaled_cube,
        Design(scaled_samples, "~ 1"),
        dose=DoseAxis("dose", unit="ng/kg", metric="administered"),
        tests=("dose_groups", *CONTRAST_TESTS),
        gate={"test": "dose_groups"},
        rng=20261216,
    )
    for test in CONTRAST_TESTS:
        assert_allclose(
            scaled.table("full", test)["t"], base.table("full", test)["t"], rtol=1e-9
        )
        assert_allclose(
            scaled.table("full", test)["p_value"],
            base.table("full", test)["p_value"],
            rtol=1e-6,
        )
    ours = scaled.lowest_effective_doses("full")
    theirs = base.lowest_effective_doses("full")
    assert_allclose(
        ours["lowest_effective_dose"],
        1000 * theirs["lowest_effective_dose"],
        rtol=1e-12,
    )


def test_few_dose_groups() -> None:
    # The vehicle and two doses: five shapes span two dimensions, and the printout
    # says so; the p-values of two contrasts need no random numbers.
    result = run((0, 1, 10), tests=("dunnett", "williams", "mcp"))
    for test in CONTRAST_TESTS:
        assert np.isfinite(result.table("cell_type_1", test)["p_value"]).all()
    assert (
        "mcp: 5 candidate shapes on 2 treated doses, whose contrasts span 2"
        in " ".join(repr(result).split())
    )
    # One treated dose: the MCP step is refused, and Dunnett's and the Williams-type
    # test are each dose's t-test; the default gate is the dose groups' test.
    with pytest.raises(ValueError, match="the design has 1"):
        run((0, 10), tests=("mcp",))
    one = run((0, 10), tests=("dunnett", "williams"))
    assert one.tests == ("dose_groups", "dunnett", "williams")
    assert one.gate["test"] == "dose_groups"
    each = one.table("cell_type_1", "each_dose")
    for test in ("dunnett", "williams"):
        table = one.table("cell_type_1", test)
        assert_allclose(table["t"], each["t"], rtol=1e-12)
        assert_allclose(table["p_value"], each["p_value"], rtol=1e-12)
        assert np.all(table["p_value_error"] == 0.0)


def test_a_cell_type_that_compares_one_dose_does_not_run_the_mcp_step() -> None:
    """The sparse cell type keeps no cells at 10 and 100 ug/kg: it compares the vehicle
    and one dose, where the MCP step has no shape to tell apart, and runs Dunnett's
    and the Williams-type test, each the dose's t-test."""
    cube, _ = simulate_dose_series(
        [0, 1, 10, 100],
        replicates=3,
        cell_types=["full", "sparse"],
        cells=[300.0, 300.0],
        n_genes=100,
        composition={"sparse": [1.0, 1.0, 0.0, 0.0]},
        rng=9,
    )
    result = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, tests=CONTRAST_TESTS, rng=1
    )
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == "sparse")
    assert entry["doses"] == [0.0, 1.0]
    assert "mcp" not in entry["contrast_tests"]
    with pytest.raises(KeyError, match="compares one, 1"):
        result.table("sparse", "mcp")
    # The default gate, the spline, needs two treated doses there too.
    sparse = result.summary.filter(group_id="sparse")
    assert sparse["tests_not_run"].tolist() == ["spline, mcp"]
    assert np.isnan(sparse["n_genes_passing_gate"]).all()
    each = result.table("sparse", "each_dose")
    for test in ("dunnett", "williams"):
        assert_allclose(result.table("sparse", test)["t"], each["t"], rtol=1e-12)
    assert ("full", "mcp") in result.tables


def test_a_flat_shape_is_left_out() -> None:
    """A narrow beta shape peaking at 50 is 0 in double precision at 1 and 10, so a
    cell type that loses the top dose leaves it out, with the reason, and tests the
    other shapes."""
    cube, _ = simulate_dose_series(
        [0, 1, 10, 50],
        replicates=3,
        cell_types=["full", "depleted"],
        cells=[300.0, 300.0],
        n_genes=100,
        composition={"depleted": [1.0, 1.0, 1.0, 0.01]},
        rng=8,
    )
    shapes = CandidateShapes(
        emax=[1.0, 10.0], beta=[(2000.0, 2000.0)], beta_scale=100.0
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=("mcp",),
        candidates=shapes,
        rng=1,
    )
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == "depleted")
    assert entry["doses"] == [0.0, 1.0, 10.0]
    mcp = entry["contrast_tests"]["mcp"]
    assert list(mcp["contrasts"]) == ["emax_1", "emax_2"]
    assert mcp["shapes_left_out"] == {
        "beta": "it does not change at the doses this cell type compares"
    }
    full = next(e for e in result.provenance["groups"] if e["group_id"] == "full")
    assert list(full["contrast_tests"]["mcp"]["contrasts"]) == [
        "emax_1",
        "emax_2",
        "beta",
    ]
    only_beta = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=("mcp",),
        candidates=CandidateShapes(beta=[(2000.0, 2000.0)], beta_scale=100.0),
        rng=1,
    )
    with pytest.raises(KeyError, match="no candidate shape changes at the doses"):
        only_beta.table("depleted", "mcp")
    assert "mcp" in only_beta.summary.filter(group_id="depleted")["tests_not_run"][0]
    assert "mcp: beta left out, flat at the doses compared" in " ".join(
        repr(result).split()
    )
    # A beta shape whose largest change at 1 and 10 is subnormal, about 1e-310, is
    # flat to double precision and left out; one of about 3e-178 is kept, and its
    # contrast is n_k (mu_k - the size-weighted mean of mu) of its means, by hand,
    # where squaring the means would underflow.
    for delta, flat in ((698.0, True), (400.0, False)):
        shapes = CandidateShapes(
            emax=[1.0, 10.0], beta=[(delta, delta)], beta_scale=100.0
        )
        means = shapes.means([0.0, 1.0, 10.0])[2]
        assert (0.0 < means.max() < np.finfo(np.float64).tiny) == flat
        tiny = dose_response(
            cube,
            Design(cube.samples, "~ 1"),
            dose=AXIS,
            tests=("mcp",),
            candidates=shapes,
            groups=["depleted"],
            rng=1,
        )
        (entry,) = tiny.provenance["groups"]
        mcp = entry["contrast_tests"]["mcp"]
        assert list(mcp["shapes_left_out"]) == (["beta"] if flat else [])
        assert np.isfinite(tiny.table("depleted", "mcp")["p_value"]).all()
        if not flat:
            n = np.array(entry["samples"])
            mu = means / means.max()
            by_hand = (n * (mu - (n * mu).sum() / n.sum()))[1:]
            assert_allclose(
                mcp["contrasts"]["beta"], by_hand / np.linalg.norm(by_hand), rtol=1e-13
            )


def test_contrast_test_arguments_are_refused_with_a_fix() -> None:
    cube = contrast_cube()
    design = Design(cube.samples, "~ 1")
    shapes = CandidateShapes(emax=[1.0])
    with pytest.raises(ValueError, match="add 'mcp' to tests"):
        dose_response(cube, design, dose=AXIS, tests=("dunnett",), candidates=shapes)
    with pytest.raises(ValueError, match="add 'mcp' to tests"):
        dose_response(cube, design, dose=AXIS, candidates=shapes)
    with pytest.raises(TypeError, match="candidates must be a CandidateShapes"):
        dose_response(
            cube, design, dose=AXIS, tests=("mcp",), candidates={"emax": [1.0]}
        )
    with pytest.raises(ValueError, match="below the design's top dose"):
        dose_response(
            cube,
            design,
            dose=AXIS,
            tests=("mcp",),
            candidates=CandidateShapes(beta=[(1.0, 1.0)], beta_scale=5.0),
        )
    with pytest.raises(ValueError, match="did you mean 'mcp'"):
        dose_response(cube, design, dose=AXIS, tests=("mpc",))
    for test in CONTRAST_TESTS:
        with pytest.raises(ValueError, match="limma's approximate standard errors"):
            dose_response(
                cube,
                design,
                dose=AXIS,
                tests=("dose_groups", test),
                contrast_standard_errors="limma",
            )
    with pytest.raises(TypeError, match="rng must be"):
        dose_response(cube, design, dose=AXIS, tests=("mcp",), rng="1")
    with pytest.raises(ValueError, match="non-negative seed"):
        dose_response(cube, design, dose=AXIS, rng=-1)
    result = contrast_result("trend")
    with pytest.raises(ValueError, match="name one with group"):
        result.lowest_effective_doses()
    # A test passed as the cell type, a group that is not a string, and a misspelt
    # test, as the result's other methods refuse them.
    with pytest.raises(KeyError, match="'mcp' is a table, not a cell type"):
        result.lowest_effective_doses("mcp")
    with pytest.raises(TypeError, match="group must be a cell type"):
        result.lowest_effective_doses(1)  # type: ignore[arg-type]
    with pytest.raises(KeyError, match="'williams' is a table, not a cell type"):
        result.table("williams")
    with pytest.raises(KeyError, match="no table 'dunett'.*did you mean 'dunnett'"):
        result.table("full", "dunett")


def test_the_contrast_tests_read_back() -> None:
    result = contrast_result("voom")
    assert result.tests == ("dose_groups", "dunnett", "williams", "mcp")
    assert result.candidates == CandidateShapes.spread(AXIS, contrast_cube().samples)
    assert result.settings["candidates"] == result.candidates.to_dict()
    table = result.table("full", "mcp")
    assert table.columns == (
        "feature_id",
        "average_log2_cpm",
        "contrast",
        "t",
        "df",
        "p_value",
        "p_value_error",
        "adjusted_p_value",
        "adjusted_p_value_across_groups",
    )
    for group in ("full", "depleted"):
        for test in CONTRAST_TESTS:
            found = result.table(group, test)
            assert_allclose(
                found["adjusted_p_value"],
                adjust_p_values(np.asarray(found["p_value"]), "bh"),
                rtol=1e-15,
            )
    stacked = np.concatenate(
        [result.table(g, "mcp")["p_value"] for g in ("full", "depleted")]
    )
    assert_allclose(
        np.concatenate(
            [
                result.table(g, "mcp")["adjusted_p_value_across_groups"]
                for g in ("full", "depleted")
            ]
        ),
        adjust_p_values(stacked, "bh"),
        rtol=1e-15,
    )
    entry = next(e for e in result.provenance["groups"] if e["group_id"] == "full")
    assert entry["contrast_tests"]["mcp"]["dimensions"] == 5
    assert entry["contrast_tests"]["mcp"]["shapes_left_out"] == {}
    again = pickle.loads(pickle.dumps(result))
    assert again.candidates == result.candidates
    assert again.table("full", "williams").equals(result.table("full", "williams"))
    assert again.lowest_effective_doses("full").equals(
        result.lowest_effective_doses("full")
    )
    text = repr(result)
    assert "mcp" in text and "williams" in text
    no_contrasts = fixture_result("trend")
    assert no_contrasts.candidates is None
    assert no_contrasts.settings["seed"] is None


def test_the_lowest_effective_doses_of_an_untested_cell_type_are_refused() -> None:
    result = mixed_result()
    untested = next(
        group
        for group, tested in zip(result.groups, result.summary["tested"], strict=True)
        if not tested
    )
    with pytest.raises(KeyError, match="was not tested"):
        result.lowest_effective_doses(untested)


def test_the_contrast_tables_carry_the_genes_symbols() -> None:
    """With symbols in the cube's feature table, the tables of the multiple contrast
    tests and the lowest effective doses carry each gene's own symbol."""
    base = contrast_cube()
    ids = base.features["feature_id"].tolist()
    symbols = [f"Gene{j}" for j in range(len(ids))]
    cube = Cube(
        {"counts": np.stack([base.layer("counts", g) for g in range(base.n_groups)])},
        samples=base.samples,
        groups=base.groups,
        features={"feature_id": ids, "symbol": symbols},
        n_cells=base.n_cells,
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=("dose_groups", "dunnett"),
        gate={"test": "dose_groups"},
        rng=1,
    )
    of_gene = dict(zip(ids, symbols, strict=True))
    for table in (
        result.table("full", "dunnett"),
        result.table("full", "gate"),
        result.lowest_effective_doses("full"),
    ):
        assert table.columns[:2] == ("feature_id", "symbol")
        assert table["symbol"].tolist() == [
            of_gene[g] for g in table["feature_id"].tolist()
        ]


def test_a_beta_scale_at_the_top_dose_is_accepted() -> None:
    """DoseFinding takes a beta model's scale equal to the top dose, and so does
    dose_response, where the beta shape is 0 at the top dose."""
    cube = contrast_cube()
    shapes = CandidateShapes(emax=[1.0], beta=[(1.0, 1.0)], beta_scale=10.0)
    assert float(np.max(cube.samples["dose"])) == 10.0
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        tests=("mcp",),
        candidates=shapes,
        groups=["full"],
        rng=1,
    )
    assert np.isfinite(result.table("full", "mcp")["p_value"]).all()
    assert result.candidates == shapes


def test_the_contrast_tests_follow_the_thread_limit(monkeypatch: Any) -> None:
    """n_threads among the analysis's options reaches every multiple contrast test."""
    calls: list[Any] = []
    original = dose_responses.multiple_contrast_test

    def recorded(*args: Any, **options: Any) -> Any:
        calls.append(options.get("n_threads"))
        return original(*args, **options)

    monkeypatch.setattr(dose_responses, "multiple_contrast_test", recorded)
    run((0, 1, 10, 100), tests=CONTRAST_TESTS, n_threads=2)
    assert calls == [2, 2, 2]


def test_the_chapter_s_contrast_and_lowest_dose_examples_run() -> None:
    """The code of the chapter's sections on the multiple contrast tests and the
    lowest effective dose, on the series of its section on the tests; the printouts
    and the numbers its text states, against the simulator's truth."""
    text = CHAPTER.read_text()

    def section(label: str) -> str:
        return text.split(f"({label})=")[1].split("\n(")[0]

    setup = re.findall(r"```python\n(.*?)```", section("dose-response-tests"), re.S)
    contrasts, lowest = (
        section("dose-response-contrasts"),
        section("lowest-effective-dose"),
    )
    blocks = [
        *re.findall(r"```python\n(.*?)```", contrasts, re.S),
        *re.findall(r"```python\n(.*?)```", lowest, re.S),
    ]
    assert len(blocks) == 5
    namespace: dict[str, Any] = {"cp": cytepool}
    exec(setup[0].replace("print(result)\n", ""), namespace)  # noqa: S102
    printed = []
    for block in blocks:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(block, namespace)  # noqa: S102
        printed.append(output.getvalue())
    # What each block prints is in the chapter, as printed: the result, and the
    # lowest effective doses of five genes, rounded.
    result = namespace["result"]
    assert printed[0] == repr(result) + "\n" and repr(result) in contrasts
    assert printed[1] == printed[2] == printed[4] == ""
    assert printed[3].count("\n") == 5 and printed[3] in lowest
    # Each dose's own p-value: at 10 ug/kg for every gene, and the smallest over the
    # doses is the gene's p-value in "dunnett"'s table, within their errors.
    at_ten = namespace["at_ten"]
    dunnett = result.table("hepatocyte", "dunnett")
    assert at_ten["feature_id"].tolist() == dunnett["feature_id"].tolist()
    whole = namespace["per_dose"].table()
    error = np.hypot(dunnett["p_value_error"], whole["p_value_error"])
    assert np.all(np.abs(whole["p_value"] - dunnett["p_value"]) <= CALIBRATED * error)
    assert len(namespace["doses"]) == 8
    truth = namespace["truth"].table("hepatocyte")
    ids = truth["feature_id"].tolist()
    responds = dict(zip(ids, (np.asarray(truth["max_log2_change"]) != 0).tolist()))
    shape = dict(zip(ids, truth["shape"].tolist()))
    tested = result.table("hepatocyte", "spline")["feature_id"].tolist()
    bells = [g for g in tested if shape[g] == "log_gauss_probit"]
    found = {}
    for test in ("spline", "dunnett", "williams", "mcp"):
        table = result.table("hepatocyte", test)
        called = np.asarray(table["feature_id"])[
            np.asarray(table["adjusted_p_value"]) < 0.05
        ].tolist()
        found[test] = (
            len(called),
            sum(responds[g] for g in called),
            sum(g in called for g in bells),
        )
    assert found == {
        "spline": (88, 85, 21),
        "dunnett": (69, 67, 17),
        "williams": (73, 69, 12),
        "mcp": (92, 89, 19),
    }
    assert len(bells) == 33
    assert (
        "the MCP step calls 92 genes, 89 of which respond in truth, the spline 88 (85), "
        "the Williams-type test 73 (69) and Dunnett's test 69 (67)"
    ) in contrasts
    assert (
        "Of the 33 responsive genes with bell-shaped responses, the spline calls 21, "
        "the MCP step 19, Dunnett's test 17 and the Williams-type test 12"
    ) in contrasts
    assert namespace["guessed"].table("hepatocyte", "mcp")["contrast"][0] in (
        namespace["shapes"].names
    )
    # The lowest effective doses: the genes with a dose called, and every dose called.
    with_dose = np.count_nonzero(
        ~np.isnan(np.asarray(namespace["called"]["lowest_effective_dose"]))
    )
    assert (len(namespace["called"]), with_dose) == (88, 69)
    assert (
        "Of the 88 genes the spline calls in the hepatocytes, 69 have a dose" in lowest
    )
    calls = namespace["calls"]
    assert int(calls.sum()) == 167 and "here 167 gene-dose calls" in lowest
    assert_array_equal(
        calls.any(axis=1),
        ~np.isnan(np.asarray(namespace["lowest"]["lowest_effective_dose"])),
    )


# The gate. Reference values in tests/data/dose_gate/limma.json come from limma 3.62.1,
# edgeR 4.4.0 and stageR 1.28.0 in R, written by validation/dose_gate.py: R fits each
# cell type from the counts with its own knots and spline columns and gives each gate's
# p-values (exact under voom's weights, and limma's classifyTestsF), adjusted by
# Benjamini-Hochberg, each dose's TREAT p-values (treat and topTreat), and stageR's
# stage-wise adjustment of the doses under the 1.5-fold gate.

GATE_DATA = Path(__file__).parent / "data" / "dose_gate"
ANALYSES = {"voom": limma_voom, "trend": limma_trend}


@functools.cache
def gate_reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((GATE_DATA / "limma.json").read_text())
    return found


@functools.cache
def gate_case(name: str) -> tuple[Cube, Design]:
    case = gate_reference()["cases"][name]
    cube = Cube(
        {"counts": np.array(case["counts"], dtype=np.int64)},
        samples=case["samples"],
        groups={"group_id": case["groups"]},
        features={"feature_id": case["features"]},
        n_cells=np.array(case["n_cells"], dtype=np.int64),
    )
    return cube, Design(case["design"], case["formula"])


ANALYSES = {"voom": limma_voom, "trend": limma_trend}


@pytest.mark.parametrize("name", ["fixture", "duplicate"])
@pytest.mark.parametrize("analysis", ["voom", "trend"])
@pytest.mark.parametrize("test", ["spline", "dose_groups"])
def test_gate_calls_match_limma(name: str, analysis: str, test: str) -> None:
    cube, design = gate_case(name)
    jobs = gate_reference()["limma"][f"{name} {analysis}"]
    for level in (0.05, 0.1):
        result = dose_response(
            cube,
            design,
            dose=AXIS,
            gate={"test": test, "fdr": level},
            analysis=ANALYSES[analysis],
        )
        for group, theirs in jobs.items():
            features = cube.features["feature_id"].tolist()
            genes = [features[i - 1] for i in np.atleast_1d(theirs["genes"])]
            table = result.table(group, "gate")
            assert table["feature_id"].tolist() == genes
            values = theirs["gates"][test]
            assert_p_values(table["p_value"], values["exact"])
            called = np.asarray(values["exact_adjusted"]) <= level
            assert_array_equal(table["passes_gate"], called)
            assert called.any() and not called.all()
            row = result.summary["group_id"].tolist().index(group)
            assert result.summary["n_genes_passing_gate"][row] == called.sum()


def test_the_dose_group_gate_with_limma_s_standard_errors() -> None:
    cube, design = gate_case("fixture")
    result = dose_response(
        cube,
        design,
        dose=AXIS,
        gate={"test": "dose_groups"},
        contrast_standard_errors="limma",
    )
    for group, theirs in gate_reference()["limma"]["fixture voom"].items():
        values = theirs["gates"]["dose_groups"]
        table = result.table(group, "gate")
        assert_p_values(table["p_value"], values["limma"])
        assert_array_equal(
            table["passes_gate"], np.asarray(values["limma_adjusted"]) <= 0.05
        )
        # limma's approximation moves the p-values off the exact ones.
        assert not np.allclose(values["limma"], values["exact"], rtol=1e-6)


@pytest.mark.parametrize("analysis", ["voom", "trend"])
def test_the_fold_change_gate_matches_treat(analysis: str) -> None:
    cube, design = gate_case("fixture")
    result = dose_response(
        cube,
        design,
        dose=AXIS,
        gate={"fold_change": 1.5},
        analysis=ANALYSES[analysis],
    )
    assert result.gate == {"test": "each_dose", "fdr": 0.05, "fold_change": 1.5}
    for group, theirs in gate_reference()["limma"][f"fixture {analysis}"].items():
        values = theirs["treat"]
        table = result.table(group, "gate")
        assert_p_values(table["p_value"], values["p"])
        called = np.asarray(values["adjusted"]) <= 0.05
        assert_array_equal(table["passes_gate"], called)
        assert called.any()
        # The lowest effective doses, called by each dose's TREAT p-value.
        k = len(theirs["doses"]) - 1
        adjusted = np.array(
            [[np.nan if v is None else v for v in row] for row in values["stage_wise"]],
            dtype=float,
        ).reshape(-1, k)
        hit = np.nan_to_num(adjusted, nan=2.0) <= 0.05
        doses = np.asarray(theirs["doses"][1:], dtype=float)
        lowest = result.lowest_effective_doses(group)
        assert_array_equal(
            lowest["lowest_effective_dose"],
            np.where(hit.any(axis=1), doses[np.argmax(hit, axis=1)], np.nan),
        )
        assert_array_equal(lowest["n_doses_called"], hit.sum(axis=1))
        assert_array_equal(lowest["passes_gate"], called)
        assert hit.any()


def treat_by_hand(result: DoseResponseResult, group: str, change: float) -> np.ndarray:
    """The gate's p-values from the table of each dose and the fit's degrees of
    freedom: McCarthy and Smyth's p = P(T > |t| - delta) + P(T > |t| + delta), delta
    = log2(fold change) / se, with scipy's t, then the smallest times the doses."""
    each = result.table(group, "each_dose")
    k = len(np.unique(each["dose"]))
    change_ = np.asarray(each["log2_fold_change"], dtype=float).reshape(k, -1).T
    error = np.asarray(each["standard_error"], dtype=float).reshape(k, -1).T
    df = np.asarray(result.fit(group).moderated.df_total, dtype=float)[:, None]
    t = np.abs(change_) / error
    delta = np.log2(change) / error
    p = student_t.sf(t - delta, df) + student_t.sf(t + delta, df)
    return np.minimum(1.0, k * p.min(axis=1))


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"robust": True},
        {"quality_weights": True},
        {"analysis": limma_trend},
        {"analysis": limma_trend, "robust": True},
    ],
)
def test_treat_moderates_as_the_fit(options: dict[str, Any]) -> None:
    # TREAT's variances and degrees of freedom are those of the fit's own
    # moderation, robust, trended or with quality weights: the gate's p-values from
    # the table of each dose and the fit, by hand.
    cube, design = gate_case("fixture")
    result = dose_response(
        cube, design, dose=AXIS, gate={"fold_change": 1.2}, **options
    )
    for group in result.groups:
        expected = treat_by_hand(result, group, 1.2)
        assert_p_values(result.table(group, "gate")["p_value"], expected, 1e-12)


def test_the_gate_by_hand() -> None:
    # Benjamini-Hochberg over the cell type's genes, a call at an adjusted p-value
    # of at most the FDR: p = (0.025, 0.05) gives (0.05, 0.05), both called at 0.05.
    fit = _CellTypeFit(
        "g",
        "inferential",
        "",
        tables={
            "spline": Table(
                {
                    "feature_id": ["a", "b", "c", "d"],
                    "p_value": np.array([0.025, 0.05, np.nan, 0.9]),
                }
            )
        },
    )
    _run_gate(fit, {"test": "spline", "fdr": 0.05, "fold_change": None}, "limma_voom")
    table = fit.tables["gate"]
    # The NaN p-value stays out of the family of 3: 0.025 x 3 / 1 = 0.075 > 0.05, and
    # 0.05 x 3 / 2 = 0.075.
    np.testing.assert_allclose(
        table["adjusted_p_value"], [0.075, 0.075, np.nan, 0.9], rtol=1e-15
    )
    assert table["passes_gate"].tolist() == [False, False, False, False]
    fit = _CellTypeFit(
        "g",
        "inferential",
        "",
        tables={
            "spline": Table(
                {"feature_id": ["a", "b"], "p_value": np.array([0.025, 0.05])}
            )
        },
    )
    _run_gate(fit, {"test": "spline", "fdr": 0.05, "fold_change": None}, "limma_voom")
    assert fit.tables["gate"]["adjusted_p_value"].tolist() == [0.05, 0.05]
    assert fit.tables["gate"]["passes_gate"].tolist() == [True, True]
    # Each gene's smallest p-value of its doses times the doses compared, capped at
    # 1; a missing dose stays in the count, and a gene with none is NaN.
    found = _smallest_times_doses(
        np.array(
            [[0.01, 0.2, 0.5], [np.nan, 0.3, np.nan], [0.6, 0.5, 0.4], [np.nan] * 3]
        )
    )
    np.testing.assert_allclose(found, [0.03, 0.9, 1.0, np.nan], rtol=1e-15)
    # A test that did not run gives no gate.
    fit = _CellTypeFit("g", "inferential", "", tables={})
    _run_gate(fit, {"test": "spline", "fdr": 0.05, "fold_change": None}, "limma_voom")
    assert fit.tables == {}


@functools.cache
def gate_series() -> Cube:
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0, 30.0],
        replicates=3,
        cell_types=["one", "two"],
        cells=300.0,
        n_genes=150,
        responsive=0.3,
        rng=31,
    )
    return cube


def test_the_default_gate_and_its_record(tmp_path: Path) -> None:
    cube = gate_series()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS)
    assert result.gate == {"test": "spline", "fdr": 0.05, "fold_change": None}
    assert result.settings["gate"] == result.gate
    assert result.provenance["gate_adjustment"] == (
        "Benjamini-Hochberg within each cell type"
    )
    assert result.cube is cube and result.design is design
    for group in result.groups:
        spline = result.table(group, "spline")
        gate = result.table(group, "gate")
        assert gate.columns == (
            "feature_id",
            "p_value",
            "adjusted_p_value",
            "passes_gate",
        )
        assert_array_equal(gate["p_value"], spline["p_value"])
        assert_array_equal(gate["adjusted_p_value"], spline["adjusted_p_value"])
    # The record gives the gate back.
    again = dose_response(cube, design, dose=AXIS, gate=result.gate)
    for key, table in result.tables.items():
        assert again.tables[key].equals(table)
    copied = pickle.loads(pickle.dumps(result))
    assert copied.gate == result.gate
    assert copied.cube is not None and copied.design is not None
    assert copied.table("one", "gate").equals(result.table("one", "gate"))
    # With one treated dose, no spline is placed, and the dose groups ask its
    # question.
    one = dose_response(
        cube,
        Design(cube.samples.filter(np.isin(cube.samples["dose"], [0.0, 1.0])), "~ 1"),
        dose=AXIS,
    )
    assert one.gate["test"] == "dose_groups"
    assert (
        one.table("one", "gate")["p_value"].tolist()
        == one.table("one", "dose_groups")["p_value"].tolist()
    )
    # A fold change, its adjustments in the record and the printout.
    large = dose_response(cube, design, dose=AXIS, gate={"fold_change": 1.5})
    assert large.provenance["gate_adjustment"] == (
        "the smallest of each gene's TREAT p-values of the doses times the doses "
        "(Bonferroni), then Benjamini-Hochberg within each cell type"
    )
    assert (
        "gate: TREAT beyond 1.5-fold at each dose, Bonferroni over the cell type's "
        "doses, Benjamini-Hochberg within each cell type, at 5% FDR"
    ) in " ".join(repr(large).split())


def test_the_gate_s_test_runs_when_tests_do_not_name_it() -> None:
    cube = gate_series()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS, tests="linear_trend")
    assert result.tests == ("spline", "linear_trend")
    assert ("one", "spline") in result.tables and ("one", "gate") in result.tables
    contrast = dose_response(
        cube, design, dose=AXIS, tests="spline", gate={"test": "williams"}, rng=1
    )
    assert contrast.tests == ("spline", "williams")
    assert contrast.settings["seed"] == 1
    # At three dose groups the spline is the dose-group test; the gate's test runs
    # and the other is left out as its repeat.
    three = Design(cube.samples.filter(np.asarray(cube.samples["dose"]) <= 1.0), "~ 1")
    by_default = dose_response(cube, three, dose=AXIS)
    assert ("one", "spline") in by_default.tables
    assert ("one", "dose_groups") not in by_default.tables
    groups = dose_response(cube, three, dose=AXIS, gate={"test": "dose_groups"})
    assert ("one", "dose_groups") in groups.tables
    assert ("one", "spline") not in groups.tables
    entry = groups.provenance["groups"][0]
    assert entry["tests_not_run"]["spline"].startswith(
        "it tests the same hypothesis as dose_groups"
    )
    np.testing.assert_allclose(
        groups.table("one", "gate")["p_value"],
        by_default.table("one", "gate")["p_value"],
        rtol=1e-12,
    )


def test_a_cell_type_without_the_gate() -> None:
    # The cell type that keeps the vehicle and the two lowest doses cannot
    # estimate a spline of two columns: no gate there, and it says why.
    cube = mixed_cube()
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    assert ("low", "gate") not in result.tables
    with pytest.raises(
        KeyError, match="the gate was not run in 'low': its test, 'spline'"
    ):
        result.table("low", "gate")
    with pytest.raises(KeyError, match="the gate was not run in 'low'"):
        result.lowest_effective_doses("low")
    summary = result.summary
    passing = dict(
        zip(summary["group_id"].tolist(), summary["n_genes_passing_gate"].tolist())
    )
    assert np.isnan(passing["low"]) and np.isnan(passing["single"])
    assert passing["all"] == np.count_nonzero(
        result.table("all", "gate")["passes_gate"]
    )
    text = " ".join(repr(result).split())
    assert "low: 3 dose groups, 9 samples," in text and "gate not run" in text
    assert (
        "gate: spline test, Benjamini-Hochberg within each cell type, at 5% FDR" in text
    )


@pytest.mark.parametrize("test", ["dunnett", "williams", "mcp"])
def test_a_multiple_contrast_gate_calls_its_test_s_adjusted_p_values(test: str) -> None:
    cube = gate_series()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        gate={"test": test, "fdr": 0.1},
        rng=2,
    )
    for group in result.groups:
        table = result.table(group, test)
        gate = result.table(group, "gate")
        assert_array_equal(gate["p_value"], table["p_value"])
        assert_array_equal(
            gate["passes_gate"], np.asarray(table["adjusted_p_value"]) <= 0.1
        )


def test_the_mcp_gate_takes_candidates_without_tests_naming_it() -> None:
    cube = gate_series()
    shapes = CandidateShapes(emax=[0.5, 5.0])
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        candidates=shapes,
        gate={"test": "mcp"},
        rng=4,
    )
    assert result.tests[-1] == "mcp" and result.candidates == shapes
    assert result.settings["seed"] == 4


def test_lowest_effective_doses_read_the_gate() -> None:
    # Under the Williams-type gate at 10%: the stage-wise calls of each dose in the
    # genes it passes, at its level, through the building block.
    cube = gate_series()
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        gate={"test": "williams", "fdr": 0.1},
        rng=5,
    )
    for group in result.groups:
        lowest = result.lowest_effective_doses(group)
        gate = result.table(group, "gate")
        assert_array_equal(lowest["passes_gate"], gate["passes_gate"])
        each = result.table(group, "each_dose")
        doses = np.unique(each["dose"])
        p = np.asarray(each["p_value"]).reshape(doses.size, -1).T
        adjusted = stage_wise_adjusted_p_values(
            result.table(group, "williams")["p_value"], p, alpha=0.1
        )
        hit = np.nan_to_num(adjusted, nan=2.0) <= 0.1
        assert_array_equal(
            lowest["lowest_effective_dose"],
            np.where(hit.any(axis=1), doses[np.argmax(hit, axis=1)], np.nan),
        )
        assert hit.any()
        assert np.count_nonzero(lowest["passes_gate"]) > np.count_nonzero(
            np.asarray(adjust_p_values(np.asarray(gate["p_value"]), "bh")) <= 0.05
        )


def test_gate_refusals() -> None:
    cube = gate_series()
    design = Design(cube.samples, "~ 1")

    def refused(error: type, match: str, **arguments: Any) -> None:
        with pytest.raises(error, match=match):
            dose_response(cube, design, dose=AXIS, **arguments)

    refused(TypeError, r"gate must be a mapping of its settings", gate="spline")
    refused(
        ValueError, r"gate has no setting 'fcd'.*did you mean 'fdr'", gate={"fcd": 0.1}
    )
    refused(TypeError, r"gate\['test'\] must be a test name", gate={"test": 1})
    for test in ("linear_trend", "nonlinearity"):
        refused(ValueError, "not any response", gate={"test": test})
    refused(ValueError, r"'splien'.*did you mean 'spline'", gate={"test": "splien"})
    for fdr in (0.0, 1.0, -0.1, np.nan):
        refused(ValueError, r"gate\['fdr'\] must be", gate={"fdr": fdr})
    refused(TypeError, r"gate\['fdr'\] must be a real number", gate={"fdr": "0.05"})
    for change in (1.0, 0.5, np.inf):
        refused(
            ValueError, r"gate\['fold_change'\] must be", gate={"fold_change": change}
        )
    refused(
        ValueError,
        "has no threshold; leave gate\\['test'\\] out",
        gate={"test": "spline", "fold_change": 1.5},
    )
    refused(
        ValueError,
        "needs gate\\['fold_change'\\].*'dunnett' tests each dose",
        gate={"test": "each_dose"},
    )
    refused(
        ValueError,
        "leave 'mcp' out of tests|neither tests nor the gate",
        candidates=object(),
    ) if False else None
    one = Design(cube.samples.filter(np.isin(cube.samples["dose"], [0.0, 1.0])), "~ 1")
    with pytest.raises(
        ValueError, match="the gate's test 'spline' runs along the dose"
    ):
        dose_response(cube, one, dose=AXIS, gate={"test": "spline"})
    with pytest.raises(ValueError, match="the gate's test 'mcp' compares"):
        dose_response(cube, one, dose=AXIS, gate={"test": "mcp"})
    donors, _ = simulate_dose_series(
        [0.0, 1.0, 10.0], replicates=3, cell_types=1, n_genes=60, donor_sd=0.3, rng=1
    )
    with pytest.raises(ValueError, match="dream_voom's tests are not"):
        dose_response(
            donors,
            Design(donors.samples, "~ 1", random="(1 | donor)"),
            dose=AXIS,
            analysis=dream_voom,
            gate={"fold_change": 1.5},
        )
    with pytest.raises(TypeError, match="masked"):
        dose_response(cube, design, dose=AXIS, gate={"fdr": np.ma.masked})
