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

import functools
import json
import pickle
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import (
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
from cytepool import dose_responses as module
from cytepool.dose_designs import DoseDesignReport
from cytepool.stats.distributions import t_ppf
from cytepool.stats.multiple_testing import adjust_p_values

DATA = Path(__file__).parent / "data" / "dose_response_tests"
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
ON_AXIS = DoseAxis(
    "dose", unit="ug/kg", metric="administered", vehicle="on_axis", offset=0.001
)
TESTS = ("spline", "nonlinearity", "slope", "dose_groups")
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
    for group, theirs in reference()["jobs"][job]["cell_types"].items():
        entry = next(e for e in result.provenance["groups"] if e["group_id"] == group)
        assert entry["fit"]["samples"] == theirs["samples"]
        assert entry["doses"] == theirs["doses"]
        assert entry["samples"] == theirs["n"]
        genes = [features[i - 1] for i in theirs["genes"]]
        prefix = "exact_" if exact else ""
        for test, values in theirs["tests"].items():
            table = result.table(group, test)
            assert table["feature_id"].tolist() == genes
            if test == "slope":
                assert_scaled(table["log2_fold_change"], values["logFC"])
                assert_scaled(table["standard_error"], values[prefix + "se"])
                assert_scaled(table["t"], values[prefix + "t"])
                assert_p_values(table["p_value"], values[prefix + "p"])
                assert_p_values(table["adjusted_p_value"], values[prefix + "adj"])
                assert_p_values(
                    table["adjusted_p_value_across_groups"],
                    values[prefix + "p_across"],
                )
                continue
            if given and test in theirs.get("given", {}):
                values, prefix = theirs["given"][test], ""
            assert entry["f_test_df1"][test] == values["df1"]
            assert_scaled(table["f_statistic"], values[prefix + "F"])
            assert_p_values(table["p_value"], values[prefix + "p"])
            assert_p_values(table["adjusted_p_value"], values[prefix + "adj"])
            assert_p_values(
                table["adjusted_p_value_across_groups"], values[prefix + "p_across"]
            )
            if "df2" in values:
                assert_scaled(table["df"], values["df2"])
        each = result.table(group, "each_dose")
        k = len(theirs["doses"]) - 1
        assert (
            each["dose"].tolist() == np.repeat(theirs["doses"][1:], len(genes)).tolist()
        )
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
    fit = result.fits[group].fit(group)
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
        # The slope: a straight line along the treated doses, with the vehicle
        # apart.
        line = axis.spline_basis(Design(fixture_cube().samples, "~ 1").samples, df=1)
        first = axis.design_columns({"dose": doses}, line)
        x1 = np.column_stack([np.ones(doses.size), *first.values()])
        slope = np.linalg.lstsq(x1, y, rcond=None)[0][-1]
        assert_allclose(
            result.table(group, "slope")["log2_fold_change"], slope, atol=1e-12
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
        ((0, 0.1, 1, 10), AXIS, ["spline", "slope", "dose_groups"], {}),
        # 2 treated doses: the spline, its treated column and the intercept pass
        # through the 3 groups, so it is the dose-group test, which runs as the
        # spline.
        ((0, 1, 10), AXIS, ["spline", "slope"], {"dose_groups": "spline"}),
        # One treated dose: nothing runs along the axis.
        ((0, 10), AXIS, ["dose_groups"], {}),
        # The vehicle on the axis, 2 doses: one t-test, which runs as the spline.
        ((0, 10), ON_AXIS, ["spline"], {"slope": "spline", "dose_groups": "spline"}),
        # 3 doses on the axis: a spline of 1 column is the straight line.
        ((0, 1, 10), ON_AXIS, ["spline", "dose_groups"], {"slope": "spline"}),
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


def test_named_tests_the_design_cannot_support_are_refused() -> None:
    with pytest.raises(ValueError, match="need two doses on it, and the design has 1"):
        run((0, 10), tests=["slope"])
    with pytest.raises(ValueError, match="spline of two columns or more.*spline_df=2"):
        run((0, 0.1, 1, 10), tests=["nonlinearity"])
    with pytest.raises(ValueError, match="needs three doses on the axis"):
        run((0, 1, 10), tests=["nonlinearity"])
    with pytest.raises(ValueError, match="leave spline_df out"):
        run((0, 10), spline_df=1)
    with pytest.raises(ValueError, match="lower spline_df to at most 3"):
        run((0, 0.1, 1, 10, 100), spline_df=4)
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
    """Four cell types of one series of the vehicle and 5 doses x 3: one keeps every
    sample, one keeps one sample per treated dose (descriptive), one no vehicle
    sample (refused), and one only the vehicle and the two lowest doses."""
    cube, _ = simulate_dose_series(
        [0, 0.1, 0.3, 1, 3, 10],
        replicates=3,
        cell_types=["all", "single", "no_vehicle", "low"],
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
        ]
    )
    assert summary["tested"].tolist() == [True, False, False, True]
    assert summary["reason"].tolist()[1:3] == report.summary["reason"].tolist()[1:3]
    assert summary["n_features_tested"].tolist()[1:3] == [0, 0]
    assert np.isnan(summary["df_residual"][1]) and np.isnan(summary["df_prior"][2])
    for group in ("single", "no_vehicle"):
        assert not any(key[0] == group for key in result.tables)
        assert group not in result.fits
        with pytest.raises(KeyError, match=f"{group!r} was not tested"):
            result.table(group, "dose_groups")
        with pytest.raises(KeyError, match=f"{group!r} was not tested"):
            result.curve(group, at=[0.0])


def test_a_cell_type_whose_doses_cannot_estimate_the_spline() -> None:
    # "low" keeps the vehicle, 0.1 and 0.3 ug/kg: a spline of 2 columns beside the
    # treated column needs 4 dose groups. Its slope and dose-group test still run,
    # and it is out of the spline's family across cell types.
    result = mixed_result()
    entry = result.provenance["groups"][3]
    assert entry["doses"] == [0.0, 0.1, 0.3]
    reason = entry["tests_not_run"]["spline"]
    assert "needs 4 dose groups" in reason and "compares 3" in reason
    assert entry["tests_not_run"]["nonlinearity"] == reason
    assert sorted(entry["f_test_df1"]) == ["dose_groups", "slope"]
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
        tests = result.fits["full"].f_tests
        by_hand = limma_voom(
            cube,
            model,
            groups="full",
            contrasts={
                name: np.array(weights) for name, weights in entry["contrasts"].items()
            },
            f_tests={test: list(members) for test, members in tests.items()},
            **option,
        )
        found = result.table("full", "spline")["f_statistic"]
        assert_allclose(
            found, by_hand.table("full", "spline")["f_statistic"], rtol=1e-12
        )
        assert not np.allclose(found, plain)
        assert result.settings["options"] == option


def test_the_dream_route() -> None:
    # Donors crossed with 4 doses 1 mg/kg apart on a linear axis with the vehicle on
    # it, all samples kept: the slope through the 4 means, each of 3 samples, weighs
    # them (x - 1.5) / 5, so the doses' differences from the vehicle by (-0.1, 0.1,
    # 0.3), derived by hand.
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
        cube, design, dose=axis, tests=["slope", "dose_groups"], analysis=dream_voom
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
        result.table("cell_type_1", "slope"),
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


def test_results_do_not_depend_on_order_units_or_threads() -> None:
    cube = fixture_cube()
    base = fixture_result("voom")
    # The design's rows in another order.
    order = np.random.default_rng(1).permutation(cube.n_samples)
    shuffled = dose_response(
        cube, Design(cube.samples.take(order), "~ 1"), dose=AXIS, spline_df=2
    )
    for key, table in base.tables.items():
        assert_allclose(
            shuffled.tables[key]["f_statistic" if "f_statistic" in table else "t"],
            table["f_statistic" if "f_statistic" in table else "t"],
            rtol=1e-10,
        )
    # Doses in ng/kg, a thousand times larger: the tests are unchanged, the slope
    # per log10 dose too, and the curves are the same at the same doses.
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
        column = "t" if test == "slope" else "f_statistic"
        assert_allclose(
            scaled.table("depleted", test)[column],
            base.table("depleted", test)[column],
            rtol=1e-9,
        )
    assert_allclose(
        scaled.table("full", "slope")["log2_fold_change"],
        base.table("full", "slope")["log2_fold_change"],
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
    with pytest.raises(TypeError, match="feature id or a list"):
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
    assert settings["min_cells"] == 10 and settings["options"] == {
        "contrast_standard_errors": "exact"
    }
    record = result.provenance
    assert record["cube"]["fingerprint"] == fixture_cube().fingerprint()
    # The cube is recorded once, not in each cell type's fit.
    assert "cube" not in record["groups"][0]["fit"]
    assert "cube" not in result.fits["full"].provenance
    assert record["groups"][0]["f_test_df1"] == {
        "spline": 3,
        "nonlinearity": 1,
        "slope": 1,
        "dose_groups": 5,
    }
    assert record["analysis_settings"]["contrast_standard_errors"] == "exact"
    # Keys left out, and keys that do not exist.
    with pytest.raises(ValueError, match="2 cell types; name one"):
        result.table(None, "spline")
    with pytest.raises(ValueError, match="name one with name"):
        result.table("full")
    with pytest.raises(KeyError, match="no cell type 'liver' was checked"):
        result.table("liver", "spline")
    with pytest.raises(KeyError, match="no table 'splines'.*did you mean 'spline'"):
        result.table("full", "splines")
    # The printout: the knots in doses, the tests and the calls of each cell type.
    text = repr(result)
    assert text.startswith("DoseResponseResult (limma_voom): 2 cell types, 2 tested")
    assert "spline of 2 columns, knots at 1 ug/kg, boundary knots at 0.1 and 10" in text
    calls = int(
        np.count_nonzero(result.table("full", "spline")["adjusted_p_value"] < 0.05)
    )
    assert f"spline {calls}" in text
    assert "slope" in text and "(per log10 dose)" in text
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
    # Immutable.
    with pytest.raises(AttributeError, match="immutable"):
        result.summary = None  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del result.summary


def test_the_result_refuses_fields_of_the_wrong_type() -> None:
    result = fixture_result("trend")
    good = (
        result.summary,
        result.tables,
        result.fits,
        result.design_report,
        result.provenance,
    )
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
        fields = list(good)
        fields[position] = bad
        with pytest.raises(TypeError, match=message):
            DoseResponseResult(*fields)
    with pytest.raises(TypeError):
        DoseResponseResult(*good[:4], {"x": np.zeros(2)})
    with pytest.raises(ValueError):
        DoseResponseResult(*good[:4], {"x": float("nan")})


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
        ({"tests": ["slope", "slope"]}, ValueError, "'slope' more than once"),
        ({"spline_df": 0}, ValueError, "spline_df"),
        ({"spline_df": 1.5}, TypeError, "spline_df"),
        ({"spline_df": 5}, ValueError, "lower spline_df to at most 4"),
        ({"analysis": len}, TypeError, "limma_voom, limma_trend or dream_voom"),
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


def test_a_check_that_the_fit_contradicts_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The weights of the spline come from the check's samples of each dose group:
    # a fit that used others stops the analysis.
    cube = fixture_cube()
    real = module.check_dose_design

    def wrong(*args: Any, **options: Any) -> DoseDesignReport:
        report = real(*args, **options)
        doses = report.doses
        counted = np.array(doses["n_samples_used"], copy=True)
        counted[1] += 1
        return DoseDesignReport(
            report.summary,
            doses.with_columns({"n_samples_used": counted}),
            report.samples,
            report.minimums,
            report.design_check,
            report.settings,
        )

    monkeypatch.setattr(module, "check_dose_design", wrong)
    with pytest.raises(RuntimeError, match="the check and the analysis disagree"):
        dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, groups="full")


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
    # The curves of four genes on 50 doses, and the figure.
    curve = namespace["curve"]
    assert len(curve) == 4 * 50
    assert namespace["ax"].get_xlabel() == "Dose (ug/kg, log10 axis)"
    with pytest.raises(ValueError, match="'stellate' compares"):
        result.curve("stellate", at=[30.0])
