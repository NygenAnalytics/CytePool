"""Tests for the F-tests of limma_voom, limma_trend and dream_voom on a cube.

Reference values in ``tests/data/differential_expression`` come from limma 3.62.1 and
edgeR 4.4.0, run on the pseudobulk counts of the cells in ``cells.npz`` and of a
simulated dose series in ``dose_spline.npz``: ``topTable(coef = ...)`` and
``classifyTestsF`` of each F-test's contrasts, the F-statistics of nested weighted
least-squares fits with limma's posterior variances, and limma-trend on R's ``ns()``
columns; ``validation/differential_expression.py`` writes both files. The other
tests check what a wrong implementation would break: the exact F-statistic against
nested fits computed here, its invariance to how a set of contrasts is written, the
rank of dependent contrasts, the F-test of one contrast against its t-test, the
closed forms of F(2, d) and F(2, infinity), the adjustment across cell types against
scipy's, and identical columns in the three methods. ``test_moderation.py`` tests the
F-test code that the three methods share.
"""

import dataclasses
import pickle
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats
from test_differential_expression import (
    ANALYSES,
    EXACT,
    cells_cube,
    fixture,
    paired_design,
    scaled_difference,
    simulated_cube,
)
from test_kenward_roger_and_f_tests import cube_and_design, cube_result

from cytepool import (
    Cube,
    Design,
    DifferentialExpressionResult,
    Table,
    dream_voom,
    from_bulk,
    limma_trend,
    limma_voom,
)
from cytepool.differential_expression import _limma_f_test
from cytepool.stats.empirical_bayes import moderated_statistics
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.smoothing import natural_spline_basis

DATA = Path(__file__).parent / "data" / "differential_expression"
ACROSS = "adjusted_p_value_across_groups"
CONTRASTS = {
    "stim_vs_ctrl": "conditionstim",
    "D2_vs_D1": "donorD2",
    "D2_vs_D4": "donorD2 - donorD4",
    "D3_vs_D1": "donorD3",
    "D3_vs_D2": "donorD3 - donorD2",
}
# The F-tests of the fixture, as the validation script names them: the stimulation
# together with a donor difference that the NK cells can estimate without donor D1,
# and two donor differences, one of which (D2_vs_D1) they cannot. limma refuses both
# in the NK cells, as each holds D2_vs_D4, which weighs the aliased column donorD4.
F_TESTS = {
    "condition_and_donor": ["stim_vs_ctrl", "D2_vs_D4"],
    "donors": ["D2_vs_D1", "D2_vs_D4"],
}
F_COLUMNS = (
    "feature_id",
    "log2_fold_change_stim_vs_ctrl",
    "log2_fold_change_D2_vs_D4",
    "average_log2_cpm",
    "f_statistic",
    "df",
    "p_value",
    "adjusted_p_value",
    "standardized_f_statistic",
)
METHODS = {
    "limma_voom": limma_voom,
    "limma_trend": limma_trend,
}
# The rows of limma's F-test fixtures after the logFC of each contrast: topTable's
# AveExpr, F, P.Value and adj.P.Val, and classifyTestsF's df2.
F_ROWS = ("average_log2_cpm", "f_statistic", "p_value", "adjusted_p_value", "df")


def design(**options: Any) -> Design:
    return paired_design(cells_cube().samples, contrasts=CONTRASTS, **options)


@cache
def analysis(method: str, **options: Any) -> DifferentialExpressionResult:
    return METHODS[method](cells_cube(), design(), f_tests=F_TESTS, **options)


def nested_f(result: DifferentialExpressionResult, group: str, members: list[str]):
    """The F-statistics and p-values of an F-test from nested weighted least-squares
    fits: (RSS0 - RSS1) / (r s~^2), with RSS1 of the group's design, RSS0 of the
    design restricted by C_S' b = 0, r the rank of C_S and s~^2 the posterior
    variance; computed here with numpy and scipy, apart from the package's linear
    algebra."""
    fit = result.fit(group)
    linear = fit.moderated.fit
    x, values, weights = linear.design, fit.log2_cpm, fit.weights
    positions = [linear.coefficient_names.index(m) for m in members]
    c = linear.contrasts[:, positions]
    rank = np.linalg.matrix_rank(c)
    # The null space of C_S': the coefficients that the F-test's null leaves free.
    null = np.linalg.svd(c.T)[2][rank:].T
    residual = np.empty((2, values.shape[1]))
    for g in range(values.shape[1]):
        root = np.sqrt(weights[:, g])
        for k, columns in enumerate((x, x @ null)):
            a, y = columns * root[:, None], values[:, g] * root
            beta = np.linalg.lstsq(a, y, rcond=None)[0]
            residual[k, g] = np.sum((y - a @ beta) ** 2)
    f_value = (residual[1] - residual[0]) / (rank * fit.moderated.posterior_variance)
    df2 = linear.df_residual + fit.moderated.df_prior
    return f_value, stats.f.sf(f_value, rank, df2)


def assert_f_test_matches(
    table: Table, reference: np.ndarray, members: list[str], tolerance: float
) -> None:
    """An F-test's table against limma's: the logFC of each contrast (when given),
    then AveExpr, F, P.Value, adj.P.Val and df2, scaled as the validation scales
    them; infinite df must be infinite in both."""
    for i, member in enumerate(members):
        found = scaled_difference(
            table[f"log2_fold_change_{member}"], reference[i], False
        )
        assert found <= tolerance, (member, found)
    rows = reference[len(reference) - len(F_ROWS) :]
    for column, expected in zip(F_ROWS, rows, strict=True):
        ours = np.asarray(table[column], dtype=float)
        if column == "df":
            np.testing.assert_array_equal(np.isinf(ours), np.isinf(expected))
            ours, expected = ours[np.isfinite(ours)], expected[np.isfinite(expected)]
            if not ours.size:
                continue
        found = scaled_difference(ours, expected, column.endswith("p_value"))
        assert found <= tolerance, (column, found)


# Against limma, from cells to F-tests.


@pytest.mark.parametrize("analysis", ANALYSES)
def test_f_tests_match_limma(analysis: str) -> None:
    function, options, tolerance = ANALYSES[analysis]
    cube = cells_cube()
    result = function(cube, paired_design(cube.samples), f_tests=F_TESTS, **options)
    limma = fixture("limma")
    t_cells = result.provenance["groups"][result.groups.index("T")]
    for name, members in F_TESTS.items():
        reference = limma[f"{analysis} T F {name}"]
        assert_f_test_matches(result.table("T", name), reference, members, tolerance)
        assert t_cells["f_test_df1"][name] == int(limma[f"{analysis} T F {name} df1"])
    if analysis == "trend":
        # limma's df2 per feature: infinite where the robust prior df are, and
        # finite for the outlying features, the case where each feature's own df2
        # matter.
        df2 = limma["trend T F condition_and_donor"][-1]
        assert np.isinf(df2).sum() >= 10 and np.isfinite(df2).sum() >= 5


@pytest.mark.parametrize("analysis", ANALYSES)
def test_rank_deficient_f_tests_differ_from_limma_only_where_limma_is_wrong(
    analysis: str,
) -> None:
    # In the NK cells limma refuses both F-tests, as each holds D2_vs_D4. Its
    # F-test of stim_vs_ctrl and D2_vs_D1 tests what condition_and_donor tests, as
    # donorD2 there means D2 against D4; CytePool runs condition_and_donor and not
    # donors, whose D2_vs_D1 the NK cells cannot estimate.
    function, options, tolerance = ANALYSES[analysis]
    cube = cells_cube()
    result = function(cube, paired_design(cube.samples), f_tests=F_TESTS, **options)
    limma = fixture("limma")
    for name in F_TESTS:
        assert "non-estimable" in str(limma[f"{analysis} NK F {name} error"])
    reading = limma[f"{analysis} NK F condition_and_donor as limma reads it"]
    table = result.table("NK", "condition_and_donor")
    assert_f_test_matches(table, reading, F_TESTS["condition_and_donor"], tolerance)
    nk = result.provenance["groups"][result.groups.index("NK")]
    assert nk["f_test_df1"] == {"condition_and_donor": 2}
    expected_df1 = limma[f"{analysis} NK F condition_and_donor as limma reads it df1"]
    assert int(expected_df1) == 2
    assert ("NK", "donors") not in result.tables


@pytest.mark.parametrize("analysis", ["voom", "robust"])
def test_the_default_matches_nested_weighted_fits_in_r(analysis: str) -> None:
    """R's lm.wfit of the full and the restricted design, with voom's weights and
    limma's posterior variances: the exact F-statistic, which limma does not give."""
    _, options, tolerance = ANALYSES[analysis]
    exact = {k: v for k, v in options.items() if k != "contrast_standard_errors"}
    cube = cells_cube()
    result = limma_voom(cube, paired_design(cube.samples), f_tests=F_TESTS, **exact)
    limma = fixture("limma")
    compared = 0
    for (group, name), table in result.tables.items():
        if name not in F_TESTS:
            continue
        f_value, p_value = limma[f"{analysis} {group} F {name} nested"]
        assert scaled_difference(table["f_statistic"], f_value, False) <= tolerance
        assert scaled_difference(table["p_value"], p_value, True) <= tolerance
        compared += 1
    assert compared == 3  # both F-tests in the T cells, one in the NK cells


@cache
def spline_study() -> tuple[Any, Design, dict[str, np.ndarray]]:
    """The dose series of dose_spline.npz: one cell type, the vehicle and four doses
    in duplicate, and the design of natural_spline_basis(dose, df=3)."""
    data = fixture("dose_spline")
    samples = {"sample_id": data["sample_id"], "dose": data["dose"]}
    cube = from_bulk(data["counts"], samples=samples)
    basis = natural_spline_basis(data["dose"], df=3)
    names = ["dose_s1", "dose_s2", "dose_s3"]
    columns = cube.samples.with_columns(dict(zip(names, basis.T, strict=True)))
    return cube, Design(columns, "~ " + " + ".join(names), contrasts=names), data


def test_a_spline_f_test_is_limmas_on_r_ns_columns() -> None:
    """limma-trend's F-test of the columns of natural_spline_basis is limma's
    topTable(coef = 2:4) on R's ~ ns(dose, df = 3): the two bases span the same
    curves, so the F-test, unlike the coefficients, is the same."""
    cube, design, data = spline_study()
    names = list(design.contrasts.names)
    result = limma_trend(cube, design, f_tests={"dose": names})
    key = "dose spline, limma-trend"
    tested = cube.features["feature_id"][data[f"{key} keep"]]
    table = result.table("bulk", "dose")
    assert table["feature_id"].tolist() == tested.tolist()
    assert_f_test_matches(table, data[f"{key} F dose"], [], EXACT)
    assert result.provenance["groups"][0]["f_test_df1"] == {"dose": 3}
    assert int(data[f"{key} F dose df1"]) == 3


def test_a_spline_f_test_by_default_matches_nested_fits_on_r_ns_columns() -> None:
    cube, design, data = spline_study()
    result = limma_voom(cube, design, f_tests={"dose": list(design.contrasts.names)})
    table = result.table("bulk", "dose")
    f_value, p_value = data["dose spline, limma-voom F dose nested"]
    assert scaled_difference(table["f_statistic"], f_value, False) <= EXACT
    assert scaled_difference(table["p_value"], p_value, True) <= EXACT


# The exact F-statistic and limma's.


def test_the_default_f_statistic_is_that_of_nested_weighted_fits() -> None:
    result = analysis("limma_voom")
    limma = analysis("limma_voom", contrast_standard_errors="limma")
    for group in ("T", "NK"):
        members = F_TESTS["condition_and_donor"]
        f_value, p_value = nested_f(result, group, members)
        table = result.table(group, "condition_and_donor")
        assert scaled_difference(table["f_statistic"], f_value, False) <= EXACT
        assert scaled_difference(table["p_value"], p_value, True) <= EXACT
        # limma's F-statistic takes the correlation of the design without weights.
        approximate = limma.table(group, "condition_and_donor")["f_statistic"]
        assert np.max(np.abs(approximate - f_value) / f_value) > 1e-3


def test_writing_the_same_contrasts_differently_changes_only_limmas_f() -> None:
    # Three pairs of contrasts that span the differences among donors D1, D2 and D3.
    pairs = {
        "a": ["D2_vs_D1", "D3_vs_D1"],
        "b": ["D2_vs_D1", "D3_vs_D2"],
        "c": ["D3_vs_D1", "D3_vs_D2"],
    }
    cube = cells_cube()
    for errors, same in (("exact", True), ("limma", False)):
        result = limma_voom(
            cube,
            design(),
            f_tests=pairs,
            groups="T",
            contrast_standard_errors=errors,
        )
        tables = [result.table("T", name) for name in pairs]
        largest = max(
            scaled_difference(table["f_statistic"], tables[0]["f_statistic"], False)
            for table in tables[1:]
        )
        assert (largest <= EXACT) == same, (errors, largest)
        if same:
            for table in tables[1:]:
                np.testing.assert_array_equal(table["df"], tables[0]["df"])
                p = scaled_difference(table["p_value"], tables[0]["p_value"], True)
                assert p <= EXACT
        else:
            assert largest > 1e-6


@pytest.mark.parametrize("method", METHODS)
def test_dependent_contrasts_count_once(method: str) -> None:
    # D3_vs_D1 = D2_vs_D1 + D3_vs_D2: three contrasts in two directions.
    tests = {
        "dependent": ["D2_vs_D1", "D3_vs_D2", "D3_vs_D1"],
        "pair": ["D2_vs_D1", "D3_vs_D1"],
    }
    result = METHODS[method](cells_cube(), design(), f_tests=tests, groups="T")
    assert result.provenance["groups"][0]["f_test_df1"] == {"dependent": 2, "pair": 2}
    dependent, pair = result.table("T", "dependent"), result.table("T", "pair")
    f = scaled_difference(dependent["f_statistic"], pair["f_statistic"], False)
    assert f <= EXACT
    # The same F on the same numerator and denominator df has the same p-value.
    np.testing.assert_array_equal(dependent["df"], pair["df"])
    np.testing.assert_allclose(dependent["p_value"], pair["p_value"], rtol=1e-12)
    # F(2, infinity) is a chi-squared on 2 df over 2, whose upper tail is exp(-x):
    # the standardized F-statistic x has the test's p-value exp(-x).
    np.testing.assert_allclose(
        np.exp(-dependent["standardized_f_statistic"]),
        dependent["p_value"],
        rtol=1e-12,
    )


@pytest.mark.parametrize("method", METHODS)
def test_the_f_test_of_one_contrast_is_its_t_test(method: str) -> None:
    result = METHODS[method](
        cells_cube(),
        design(),
        f_tests={"stimulation": ["stim_vs_ctrl"]},
        groups="T",
        expression_filter=False,
        normalization=None,
    )
    f, t = result.table("T", "stimulation"), result.table("T", "stim_vs_ctrl")
    np.testing.assert_allclose(f["f_statistic"], t["t"] ** 2, rtol=1e-12)
    assert result.provenance["groups"][0]["f_test_df1"] == {"stimulation": 1}
    # Their p-values agree where the cap of the t-tests' df, the residual df summed
    # over the features, does not bind; it binds where the prior df are infinite.
    moderated = result.fit("T").moderated
    uncapped = f["df"] <= moderated.fit.df_residual.sum()
    assert uncapped.sum() >= 10
    assert scaled_difference(f["p_value"][uncapped], t["p_value"][uncapped], True) <= (
        1e-12
    )


def test_an_f_test_of_every_contrast_is_ebayes_f() -> None:
    result = limma_voom(cells_cube(), design(), f_tests={"all": list(CONTRASTS)})
    table, moderated = result.table("T", "all"), result.fit("T").moderated
    np.testing.assert_array_equal(table["f_statistic"], moderated.f_statistic)
    np.testing.assert_array_equal(table["p_value"], moderated.f_p_value)
    # Five contrasts in four directions: D3_vs_D2 = D3_vs_D1 - D2_vs_D1.
    record = result.provenance["groups"][result.groups.index("T")]
    assert record["f_test_df1"] == {"all": moderated.f_df1} == {"all": 4}


# The denominator degrees of freedom and the standardized F-statistic.


def test_on_infinite_df_the_standardized_f_statistic_is_the_f_statistic() -> None:
    result = analysis(
        "limma_trend", expression_filter=False, normalization=None, robust=True
    )
    table = result.table("NK", "condition_and_donor")
    assert np.isinf(table["df"]).all()
    np.testing.assert_allclose(
        table["standardized_f_statistic"], table["f_statistic"], rtol=1e-12
    )


def test_limma_f_tests_standardize_on_each_features_df() -> None:
    # One feature with a large effect, whose p-value underflows on the 9 residual
    # df plus 1,000 prior df: the standardized F-statistic comes from the log of the
    # tail probability on those df, where F on (2, d) gives (d / 2) log(1 + 2F / d).
    rng = np.random.default_rng(1)
    groups = np.repeat(np.eye(3), 4, axis=0)
    x = np.column_stack([np.ones(12), groups[:, 1:]])
    effects = np.zeros((30, 3))
    effects[:, 0] = 5.0
    effects[0, 1:] = [40.0, -30.0]
    effects[1:, 1] = 0.1 * rng.normal(size=29)
    values = x @ effects.T + rng.normal(0.0, 0.05, (12, 30))
    moderated = moderated_statistics(fit_linear_model(values, x))
    finite = dataclasses.replace(moderated, df_prior=np.full(30, 1_000.0))
    test = _limma_f_test(finite, [1, 2])
    d, f = test.df2[0], test.f_statistic[0]
    assert d == 1_009.0 and test.p_value[0] == 0.0
    np.testing.assert_allclose(
        test.standardized_f_statistic[0], d / 2.0 * np.log1p(2.0 * f / d), rtol=1e-12
    )


# Columns, tables and records shared by the three methods.


def test_the_three_methods_give_f_tests_the_same_columns() -> None:
    cube, mixed = cube_and_design()
    # Gene symbols, as real cubes usually have, join the identity columns.
    named = Cube(
        {"counts": cube.layer("counts")},
        samples=cube.samples,
        groups=cube.groups,
        features={
            "feature_id": cube.features["feature_id"],
            "symbol": [f"SYM_{f}" for f in cube.features["feature_id"]],
        },
        n_cells=cube.n_cells,
    )
    fixed = Design(
        cube.samples,
        "~ time + age",
        levels={"time": ["t0", "t1", "t2"]},
        contrasts={
            "t1_vs_t0": "timet1",
            "t2_vs_t0": "timet2",
            "t2_vs_t1": "timet2 - timet1",
        },
    )
    tests = {"time": ["t1_vs_t0", "t2_vs_t0"]}
    for adjust in (False, True):
        results = {
            "limma_voom": limma_voom(
                named, fixed, f_tests=tests, adjust_across_groups=adjust
            ),
            "limma_trend": limma_trend(
                named, fixed, f_tests=tests, adjust_across_groups=adjust
            ),
            "dream_voom": dream_voom(
                named, mixed, f_tests=tests, span=0.5, adjust_across_groups=adjust
            ),
        }
        columns = {result.table("small", "time").columns for result in results.values()}
        assert len(columns) == 1
        (found,) = columns
        assert found[:3] == ("feature_id", "symbol", "log2_fold_change_t1_vs_t0")
        assert (ACROSS in found) == adjust
        for result in results.values():
            f_table = result.table("small", "time")
            for member in tests["time"]:
                contrast = result.table("small", member)
                np.testing.assert_array_equal(
                    f_table[f"log2_fold_change_{member}"], contrast["log2_fold_change"]
                )
            for column in ("feature_id", "symbol", "average_log2_cpm"):
                np.testing.assert_array_equal(f_table[column], contrast[column])
    # dream's tables are those of the analysis without the adjustment.
    dream = cube_result().table("small", "time")
    assert dream.equals(
        dream_voom(cube, mixed, f_tests=tests, span=0.5).table("small", "time")
    )


def test_an_f_test_has_the_columns_of_the_result_and_its_contrasts_tables() -> None:
    for method in METHODS:
        result = analysis(method)
        table = result.table("T", "condition_and_donor")
        assert table.columns == F_COLUMNS
        for member in F_TESTS["condition_and_donor"]:
            np.testing.assert_array_equal(
                table[f"log2_fold_change_{member}"],
                result.table("T", member)["log2_fold_change"],
            )


def test_the_columns_follow_the_order_of_the_contrasts_given() -> None:
    cube = cells_cube()
    given = {
        "forward": ["stim_vs_ctrl", "D2_vs_D4", "D3_vs_D2"],
        "reversed": ["D3_vs_D2", "D2_vs_D4", "stim_vs_ctrl"],
    }
    result = limma_voom(cube, design(), f_tests=given)
    for group in ("T", "NK"):
        forward, backward = (
            result.table(group, "forward"),
            result.table(group, "reversed"),
        )
        assert backward.columns[1:4] == tuple(
            f"log2_fold_change_{m}" for m in given["reversed"]
        )
        for column in ("f_statistic", "p_value", "standardized_f_statistic"):
            np.testing.assert_allclose(backward[column], forward[column], rtol=1e-12)
        np.testing.assert_array_equal(backward["df"], forward["df"])


def test_each_f_test_is_adjusted_alone_within_and_across_groups() -> None:
    result = analysis("limma_voom", adjust_across_groups=True)
    for name, groups in (("condition_and_donor", ["NK", "T"]), ("donors", ["T"])):
        tables = [result.table(group, name) for group in groups]
        for table in tables:
            np.testing.assert_allclose(
                table["adjusted_p_value"],
                stats.false_discovery_control(table["p_value"], method="bh"),
                rtol=1e-12,
            )
        stacked = np.concatenate([table["p_value"] for table in tables])
        expected = stats.false_discovery_control(stacked, method="bh")
        np.testing.assert_allclose(
            np.concatenate([table[ACROSS] for table in tables]), expected, rtol=1e-12
        )
        assert tables[0].columns[7:9] == ("adjusted_p_value", ACROSS)


# Groups that cannot run an F-test.


@pytest.mark.parametrize("method", METHODS)
def test_a_group_runs_only_the_f_tests_whose_contrasts_it_estimates(
    method: str,
) -> None:
    # Without donor D1, the NK cells cannot estimate D2_vs_D1, a member of "donors".
    result = analysis(method)
    nk = result.provenance["groups"][result.groups.index("NK")]
    assert nk["f_test_df1"] == {"condition_and_donor": 2}
    assert nk["not_estimable"]["donors"] == (
        "the samples of this group cannot estimate its contrast 'D2_vs_D1'"
    )
    assert result.summary["not_estimable"][result.groups.index("NK")] == (
        "D2_vs_D1, D3_vs_D1, donors"
    )
    assert ("NK", "donors") not in result.tables
    with pytest.raises(
        KeyError,
        match="F-test 'donors' was not tested in group 'NK': the samples of this "
        "group cannot estimate its contrast 'D2_vs_D1'",
    ):
        result.table("NK", "donors")
    assert ("T", "donors") in result.tables


def test_an_f_test_no_group_can_run_leaves_a_reason_and_no_table() -> None:
    cube = cells_cube()
    result = limma_voom(
        cube,
        design(),
        groups="NK",
        f_tests={"donors": F_TESTS["donors"], "from_d1": ["D2_vs_D1", "D3_vs_D1"]},
        adjust_across_groups=True,
    )
    assert ("NK", "donors") not in result.tables
    record = result.provenance["groups"][0]
    assert (
        record["f_test_df1"] == {} and "D2_vs_D1" in record["not_estimable"]["donors"]
    )
    # Without donor D1, neither of from_d1's contrasts.
    assert record["not_estimable"]["from_d1"] == (
        "the samples of this group cannot estimate its contrasts 'D2_vs_D1' and "
        "'D3_vs_D1'"
    )


def test_a_group_not_fitted_records_no_f_tests() -> None:
    cube = simulated_cube({"common": [20] * 8, "rare": [20, 20, 20, 0, 3, 0, 0, 5]})
    tests = {"both": ["stim_vs_ctrl", "D2_vs_D4"]}
    for function in METHODS.values():
        result = function(
            cube, paired_design(cube.samples), f_tests=tests, adjust_across_groups=True
        )
        common, rare = result.provenance["groups"]
        assert common["f_test_df1"] == {"both": 2} and rare["f_test_df1"] is None
        with pytest.raises(KeyError, match=r"'rare' was not fitted \(no_residual_df\)"):
            result.table("rare", "both")
        # The family across groups is the fitted group's alone.
        table = result.table("common", "both")
        np.testing.assert_array_equal(table[ACROSS], table["adjusted_p_value"])


# What F-tests leave unchanged.


@pytest.mark.parametrize("method", METHODS)
def test_f_tests_change_nothing_else(method: str) -> None:
    function = METHODS[method]
    without = function(cells_cube(), design())
    with_f = analysis(method)
    for key, table in without.tables.items():
        assert with_f.tables[key].equals(table)
    assert set(with_f.tables) - set(without.tables) == {
        ("T", "condition_and_donor"),
        ("T", "donors"),
        ("NK", "condition_and_donor"),
    }
    for column in without.summary.columns:
        if column != "not_estimable":
            assert with_f.summary[column].tolist() == without.summary[column].tolist()
    assert with_f.samples.equals(without.samples)
    for group in ("T", "NK"):
        np.testing.assert_array_equal(
            with_f.fit(group).moderated.t, without.fit(group).moderated.t
        )
    # An empty mapping is no F-test.
    empty = function(cells_cube(), design(), f_tests={})
    assert empty.settings["f_tests"] is None and empty.f_tests == {}
    assert empty.provenance["groups"] == without.provenance["groups"]
    assert [g["f_test_df1"] for g in without.provenance["groups"]] == [{}, {}]


def test_f_tables_do_not_depend_on_threads_or_the_order_of_the_designs_rows() -> None:
    cube = cells_cube()
    order = np.random.default_rng(5).permutation(cube.n_samples)
    shuffled = paired_design(cube.samples.take(order), contrasts=CONTRASTS)
    one = limma_voom(cube, design(), f_tests=F_TESTS, n_threads=1)
    four = limma_voom(cube, design(), f_tests=F_TESTS, n_threads=4)
    rows = limma_voom(cube, shuffled, f_tests=F_TESTS)
    for key, table in analysis("limma_voom").tables.items():
        assert one.tables[key].equals(table)
        assert four.tables[key].equals(table)
        assert rows.tables[key].equals(table)


def test_the_result_keeps_its_f_tests_in_records_repr_and_pickles() -> None:
    for method in METHODS:
        result = analysis(method)
        assert result.f_tests == {name: tuple(m) for name, m in F_TESTS.items()}
        assert result.settings["f_tests"] == F_TESTS
        restored = pickle.loads(pickle.dumps(result))
        assert restored.f_tests == result.f_tests
        for key, table in result.tables.items():
            assert restored.tables[key].equals(table)
        lines = repr(result).splitlines()
        assert max(map(len, lines)) <= 80
        text = " ".join(" ".join(lines).split())
        assert (
            "5 contrasts:" in text and "2 F-tests: condition_and_donor, donors" in text
        )
        assert "not estimable: D2_vs_D1, D3_vs_D1, donors" in text
    single = limma_voom(
        cells_cube(),
        design(),
        contrasts=["conditionstim"],
        f_tests={"stimulation": ["conditionstim"]},
        groups="T",
    )
    text = " ".join(repr(single).split())
    assert "1 contrast: conditionstim; 1 F-test: stimulation" in text
    with pytest.raises(ValueError, match="the result has 2 contrasts and F-tests"):
        single.table("T")
    assert single.table("T", "stimulation").columns[1] == (
        "log2_fold_change_conditionstim"
    )
    with pytest.raises(KeyError, match="no contrast or F-test 'stimulations' was"):
        single.table("T", "stimulations")


def test_a_record_without_the_f_test_setting_reads_as_no_f_tests() -> None:
    # limma_voom results recorded before it took F-tests, such as those pickled with
    # CytePool 0.1.0, have no "f_tests" setting: they read, refit and pickle as
    # results without F-tests, as they did then.
    result = METHODS["limma_voom"](cells_cube(), design())
    provenance = result.provenance
    del provenance["settings"]["f_tests"]
    old = DifferentialExpressionResult(
        result.method,
        result.contrasts,
        result.summary,
        result.samples,
        result.tables,
        provenance,
        cube=result.cube,
        design=result.design,
    )
    assert "f_tests" not in old.settings
    assert old.f_tests == {}
    assert "F-test" not in repr(old)
    contrast = result.contrasts.names[0]
    assert old.table("T", contrast).columns == result.table("T", contrast).columns
    refit = old.fit("T").moderated
    np.testing.assert_array_equal(refit.t, result.fit("T").moderated.t)
    assert pickle.loads(pickle.dumps(old)).f_tests == {}
    with pytest.raises(KeyError, match="no contrast 'dose' was"):
        old.table("T", "dose")


# Arguments.


@pytest.mark.parametrize("method", [*METHODS, "dream_voom"])
@pytest.mark.parametrize(
    ("f_tests", "error", "message"),
    [
        (["stim_vs_ctrl"], TypeError, "f_tests must be a mapping"),
        ({1: ["stim_vs_ctrl"]}, TypeError, r"f_tests must be keyed by names \(str\)"),
        ({"x": "stim_vs_ctrl"}, TypeError, r"f_tests\['x'\] must be a list"),
        ({"x": []}, ValueError, "must name at least one contrast"),
        ({"x": ["stim_vs_ctrl", 2]}, ValueError, "must name at least one contrast"),
        ({"x": ["stim_vs_ctrl"] * 2}, ValueError, "'stim_vs_ctrl' more than once"),
        ({"stim_vs_ctrl": ["stim_vs_ctrl"]}, ValueError, "has the name of a contrast"),
        ({"x": ["stim_vs_ctrl", "D9"]}, ValueError, "names 'D9', which is not among"),
    ],
)
def test_invalid_f_tests_are_refused_alike(
    method: str, f_tests: Any, error: type[Exception], message: str
) -> None:
    samples = cells_cube().samples
    contrasts = {"stim_vs_ctrl": "conditionstim"}
    if method == "dream_voom":
        function: Any = dream_voom
        chosen = Design(
            samples, "~ condition", random="(1 | donor)", contrasts=contrasts
        )
    else:
        function, chosen = METHODS[method], paired_design(samples, contrasts=contrasts)
    with pytest.raises(error, match=message):
        function(cells_cube(), chosen, f_tests=f_tests)
