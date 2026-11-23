"""Tests of NTP's and ETAP's rule sets, the documents' choices of model and BMDS's
recommended model: each step against ETAP's text, scipy and values derived by hand,
the rule table against its steps run through public functions, and BMDExpress
2.30.520, pybmds 25.2 and R through the fixtures of validation/toxicology_rule_sets.py."""

import json
import math
import pickle
import re
import zlib
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
from scipy.stats import chi2, f_oneway, false_discovery_control

from cytepool import (
    BenchmarkResponse,
    Cube,
    Design,
    DoseAxis,
    GeneSets,
    PathwayScores,
    dose_response,
    from_bulk,
    simulate_dose_series,
)
from cytepool._rule_sets import (
    ETAP,
    NTP,
    anova_premodelling_test,
    largest_log2_fold_changes,
    passes_anova_premodelling_test,
    passes_prefilter,
    values_by_etap,
)
from cytepool.stats._benchmark_dose_flags import (
    FlagInputs,
    accepted_rows,
    failed_checks,
)
from cytepool.stats.dose_curves import fit_dose_curves
from cytepool.stats.potency import (
    _first_with_bounds,
    _flagged_hill,
    _per_model,
    _pybmds_counted,
    _recommended,
    benchmark_doses,
)
from cytepool.stats.trend_tests import williams_trend_test
from cytepool.table import Table


def test_values_follow_etap() -> None:
    """ETAP Sec. 3.4.5.2: probes with a mean count below 5 are removed; counts per
    million are count x 1,000,000 / the sum of the remaining probes' counts; then
    log2(CPM + 1)."""
    rng = np.random.default_rng(1)
    counts = rng.poisson(40.0, (10, 6)).astype(float)
    counts[:, 1] = [5] * 10  # mean exactly 5: kept
    counts[:, 2] = [5] * 9 + [4]  # mean 4.9: removed
    counts[:, 3] = 0.0  # removed
    kept, values = values_by_etap(counts)
    assert kept.tolist() == [True, True, False, False, True, True]
    remaining = counts[:, kept]
    expected = np.log2(remaining * 1_000_000 / remaining.sum(axis=1, keepdims=True) + 1)
    np.testing.assert_allclose(values, expected, rtol=1e-14)


def test_a_cell_type_of_few_counts_is_not_analysed() -> None:
    """When no gene has a mean count of 5, ETAP's values have no gene, and the rule
    set says so in the cell type's status and outcomes."""
    kept, values = values_by_etap(np.full((6, 4), 4.0))
    assert not kept.any() and values.shape == (6, 0)
    levels = np.array([0.0, 0.1, 1.0, 10.0])
    doses = np.repeat(levels, 3)
    counts = np.random.default_rng(4).poisson(3.0, (doses.size, 30))
    samples = {"sample_id": [f"s{i}" for i in range(doses.size)], "dose": doses}
    cube = from_bulk(counts, samples=samples)
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        rules="ntp_2018",
        rng=1,
        expression_filter={"min_count": 1, "min_total_count": 1},
    )
    status = "not analysed: no gene has a mean count of at least 5"
    assert result.summary["rule_set_status"].tolist() == [status]
    group = result.groups[0]
    assert len(result.table(group, "rule_set")) == 0
    found = dict(
        zip(*(result.rules(group)[c].tolist() for c in ("setting", "outcome")))
    )
    assert found["values"] == "0 genes" and found["ANOVA pre-modelling test"] == status


def test_anova_matches_scipy_and_adjusts_over_genes_that_vary() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(4), (3, 4, 4, 4))
    values = rng.normal(size=(15, 30)) + rng.normal(size=(4, 30))[labels]
    values[:, 5] = 3.0  # one value: no test, left out of the adjustment
    values[:, 6] = np.repeat([1.0, 2.0, 3.0, 4.0], (3, 4, 4, 4))  # no spread within
    f, p, adjusted = anova_premodelling_test(values, labels, 4)
    varying = [g for g in range(30) if g not in (5, 6)]
    reference = f_oneway(*[values[labels == k][:, varying] for k in range(4)], axis=0)
    np.testing.assert_allclose(f[varying], reference.statistic, rtol=1e-12)
    np.testing.assert_allclose(p[varying], reference.pvalue, rtol=1e-11)
    assert np.isnan(f[5]) and np.isnan(p[5]) and np.isnan(adjusted[5])
    assert f[6] == np.inf and p[6] == 0.0
    tested = [g for g in range(30) if g != 5]
    np.testing.assert_allclose(
        adjusted[tested], false_discovery_control(p[tested]), rtol=1e-12
    )


def test_anova_passes_below_its_threshold_strictly() -> None:
    """NTP p. 5 and ETAP Sec. 3.4.5.3: "FDR < 0.05"; NaN passes not."""
    adjusted = np.array([0.05, np.nextafter(0.05, 0.0), 0.0, np.nan, 1.0])
    assert passes_anova_premodelling_test(adjusted).tolist() == [
        False,
        True,
        True,
        False,
        False,
    ]


def test_largest_change_is_signed_at_the_dose_of_largest_size() -> None:
    labels = np.repeat(np.arange(4), 2)
    # Gene 0: means 1, 3, 0.5, 2: the largest change is +2 at the first dose.
    # Gene 1: means 2, 1, 3, 2.5: changes -1, +1, +0.5, a tie of size; the lower
    # dose's -1 is taken.
    values = np.column_stack(
        [
            np.repeat([1.0, 3.0, 0.5, 2.0], 2) + [0.1, -0.1] * 4,
            np.repeat([2.0, 1.0, 3.0, 2.5], 2) + [0.2, -0.2] * 4,
        ]
    )
    np.testing.assert_allclose(
        largest_log2_fold_changes(values, labels, 4), [2.0, -1.0], atol=1e-14
    )


def test_prefilter_thresholds() -> None:
    """Williams' p below 0.05, strictly; a change of 1.5-fold or more, inclusive (ETAP:
    "1.5x or greater"); NaN passes neither."""
    change = np.log2(1.5)
    p = np.array([0.04, 0.05, 0.04, 0.04, 0.04, np.nan, 0.01])
    changes = np.array(
        [change, change, -change, np.nextafter(change, 0), 1.0, 1.0, np.nan]
    )
    assert passes_prefilter(p, changes).tolist() == [
        True,
        False,
        True,
        False,
        True,
        False,
        False,
    ]


def test_settings_name_their_sources() -> None:
    """Every setting has a value and a source; the BMDExpress settings name the
    options of BMDExpress's command line that set them."""
    for rule_set in (NTP, ETAP):
        names = [setting.setting for setting in rule_set.settings]
        assert len(set(names)) == len(names)
        for setting in (*rule_set.settings, *rule_set.not_applied):
            assert setting.value and setting.source
    assert ETAP.bmr_sd == 1.349 and NTP.bmr_sd == 1.0
    assert ETAP.hill_k_fraction == 1 / 3 and NTP.hill_k_fraction is None
    etap = {setting.setting: setting for setting in ETAP.settings}
    assert "bmdBMDLRatioMin 20" in etap["acceptance"].bmdexpress
    assert "kParameterValue 3" in etap["model choice"].bmdexpress
    ntp = {setting.setting: setting for setting in NTP.settings}
    assert "bmduBMDLRatioMin 40" in ntp["acceptance"].bmdexpress
    assert "flagHillWithKParameter false" in ntp["model choice"].bmdexpress


# The rule sets in dose_response, against their steps run through public functions.

AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
LADDER = (0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)


@pytest.fixture(scope="module")
def series():
    cube, truth = simulate_dose_series(
        LADDER, replicates=4, cell_types=2, cells=300.0, n_genes=300, rng=20261122
    )
    return cube, truth


@pytest.fixture(scope="module")
def etap(series):
    cube, _ = series
    return dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, rules="etap_2024", rng=5
    )


def test_the_rule_table_is_its_steps_run_through_public_functions(series, etap) -> None:
    cube, _ = series
    group = etap.groups[0]
    table = etap.table(group, "rule_set")
    samples = etap.design_report.samples.filter(group_id=group)
    used = np.asarray(samples["used"], dtype=bool)
    ids = np.asarray(samples["sample_id"].tolist(), dtype=object)[used]
    doses = np.asarray(samples["dose"], dtype=float)[used]
    rows = [cube.samples["sample_id"].tolist().index(s) for s in ids]
    counts = np.asarray(cube.layer("counts", group)[rows], dtype=float)
    keep = counts.mean(axis=0) >= 5
    remaining = counts[:, keep]
    values = np.log2(remaining * 1e6 / remaining.sum(axis=1, keepdims=True) + 1)
    assert table["feature_id"].tolist() == cube.features["feature_id"][keep].tolist()
    levels = np.unique(doses)
    anova = f_oneway(*[values[doses == level] for level in levels], axis=0)
    np.testing.assert_allclose(table["anova_p_value"], anova.pvalue, rtol=1e-10)
    np.testing.assert_allclose(
        table["anova_adjusted_p_value"],
        false_discovery_control(anova.pvalue),
        rtol=1e-10,
    )
    # The cell type's permutations, from the seed recorded and the stream documented.
    generator = np.random.default_rng(
        [etap.settings["seed"], zlib.crc32(group.encode("utf-8")), 2000]
    )
    trend = williams_trend_test(doses, values, rng=generator)
    np.testing.assert_array_equal(table["williams_trend_p_value"], trend.p_value)
    means = np.array([values[doses == level].mean(axis=0) for level in levels])
    changes = means[1:] - means[0]
    largest = changes[np.argmax(np.abs(changes), axis=0), np.arange(changes.shape[1])]
    np.testing.assert_allclose(table["max_log2_fold_change"], largest, atol=1e-12)
    passing = (trend.p_value < 0.05) & (np.abs(largest) >= np.log2(1.5))
    np.testing.assert_array_equal(table["passes_prefilter"], passing)
    fits = fit_dose_curves(doses, values[:, passing], models="bmdexpress")
    np.testing.assert_array_equal(etap.curve_fits(group).parameters, fits.parameters)
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    picked = table.filter(table["passes_prefilter"])
    assert picked["model"].tolist() == found.gene_models.tolist()
    for column in ("bmd", "bmdl", "bmdu"):
        np.testing.assert_array_equal(picked[column], getattr(found, column)[0])
    np.testing.assert_array_equal(picked["fit_p_value"], found.fit_p_value)
    np.testing.assert_array_equal(picked["bmd_vehicle_sd"], found.bmd_vehicle_sd)
    np.testing.assert_array_equal(picked["accepted"], found.accepted("etap_2024")[0])
    others = table.filter(~table["passes_prefilter"])
    assert set(others["model"].tolist()) <= {""} and not others["accepted"].any()
    assert np.isnan(others["bmd"]).all()


def test_rules_refuse_what_they_set_and_designs_they_do_not_fit(series) -> None:
    cube, _ = series
    design = Design(cube.samples, "~ 1")
    for option, value in (
        ("models", "epa"),
        ("variance", "weighted"),
        ("bmr", BenchmarkResponse.sd(1)),
        ("flags", "etap_2024"),
    ):
        with pytest.raises(ValueError, match=f"'{option}' cannot be given with it"):
            dose_response(cube, design, dose=AXIS, rules="etap_2024", **{option: value})
    with pytest.raises(
        ValueError, match="rules must be one of 'ntp_2018', 'etap_2024'"
    ):
        dose_response(cube, design, dose=AXIS, rules="etap")
    sexes = np.where(np.arange(len(cube.samples)) % 2 == 0, "F", "M")
    with_sex = cube.samples.with_columns({"sex": sexes})
    with pytest.raises(ValueError, match="fits dose alone to independent animals"):
        dose_response(cube, Design(with_sex, "~ sex"), dose=AXIS, rules="ntp_2018")
    with pytest.raises(ValueError, match="selection is BMDS's recommendation"):
        dose_response(cube, design, dose=AXIS, rules="ntp_2018", selection="bmds_2026")
    with pytest.raises(ValueError, match="give models='epa' and bmr"):
        dose_response(cube, design, dose=AXIS, selection="bmds_2026")
    with pytest.raises(ValueError, match="use variance='constant'"):
        dose_response(
            cube,
            design,
            dose=AXIS,
            models="epa",
            variance="weighted",
            bmr=BenchmarkResponse.sd(1),
            selection="bmds_2026",
        )
    with pytest.raises(ValueError, match="selection must be one of 'bmds_2026'"):
        dose_response(cube, design, dose=AXIS, selection="lowest_aic")
    masked = np.ma.masked_array(["etap_2024"], mask=[True])
    for option in ("rules", "selection"):
        with pytest.raises(TypeError, match=f"{option} is a numpy masked array"):
            dose_response(cube, design, dose=AXIS, **{option: masked})
    # Scores have no counts, which the rule sets read.
    ids = cube.samples["sample_id"].tolist()
    scores = PathwayScores(
        np.random.default_rng(3).normal(size=(len(ids), 2)),
        Table({"profile_id": ids, "sample_id": ids, "dose": cube.samples["dose"]}),
        GeneSets({name: [f"{name}_g{i}" for i in range(10)] for name in ("a", "b")}),
        "gsva",
        MappingProxyType({}),
    )
    for rules in ("ntp_2018", "etap_2024"):
        with pytest.raises(ValueError, match="pathway scores and activities have none"):
            dose_response(scores, design, dose=AXIS, rules=rules)


def test_bmds_s_choice_leaves_out_cell_types_without_bounds() -> None:
    """A cell type that supports only description has no BMDLs for BMDS's checks, so
    under selection='bmds_2026' it keeps its curves and gets no BMDs, and says why."""
    cube, _ = simulate_dose_series(
        [0.0, 1.0, 10.0], replicates=[2, 1, 1], cell_types=1, n_genes=40, rng=2
    )
    result = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        models="epa",
        bmr=BenchmarkResponse.sd(1),
        selection="bmds_2026",
    )
    group = result.groups[0]
    assert result.summary["status"].tolist() == ["descriptive"]
    assert result.curve_fits(group).models == ("linear", "exponential_2")
    reason = "reads each model's BMDL, and a cell type that supports only description"
    with pytest.raises(KeyError, match=reason):
        result.benchmark_doses(group)
    with pytest.raises(KeyError, match=reason):
        result.table(group, "benchmark_doses")
    assert "BMDs at" not in repr(result)


def test_rules_and_their_outcomes_read_back(series, etap) -> None:
    group = etap.groups[0]
    settings = etap.rules()
    assert settings.columns == ("setting", "value", "bmdexpress", "source")
    assert settings["setting"].tolist()[:4] == [
        "data set",
        "design minimums",
        "values",
        "ANOVA pre-modelling test",
    ]
    outcomes = etap.rules(group)
    found = dict(zip(outcomes["setting"].tolist(), outcomes["outcome"].tolist()))
    record = next(e for e in etap.provenance["groups"] if e["group_id"] == group)
    counts = record["rule_set"]
    table = etap.table(group, "rule_set")
    assert found["values"] == f"{len(table):,} genes"
    passing = int(np.sum(table["anova_adjusted_p_value"] < 0.05))
    assert found["ANOVA pre-modelling test"] == f"passed: {passing} genes at FDR < 0.05"
    assert (
        found["fold change"]
        == f"{int(table['passes_prefilter'].sum())} genes pass both"
    )
    accepted = int(table["accepted"].sum())
    assert (
        found["acceptance"]
        == f"{accepted} of {int(table['passes_prefilter'].sum())} accepted"
    )
    # 4 animals per dose group, but the ladder's steps are not half a log10.
    assert found["design minimums"].startswith("deviation: dose_spacing")
    summary = etap.summary
    row = summary["group_id"].tolist().index(group)
    assert summary["rule_set_status"][row] == "analysed"
    assert summary["n_genes_accepted"][row] == accepted == counts["n_genes_accepted"]
    text = " ".join(repr(etap).split())
    assert "rules: etap_2024 (US EPA 2024), curves bmdexpress" in text
    assert f"{accepted} accepted" in text
    copy = pickle.loads(pickle.dumps(etap))
    assert copy.table(group, "rule_set").equals(table)
    assert copy.rules(group).equals(outcomes)
    assert etap.settings["rules"] == "etap_2024" and etap.settings["seed"] == 5
    assert "flags_etap_2024" in etap.table(group, "benchmark_doses").columns
    assert "passes_gate" not in etap.table(group, "benchmark_doses").columns
    with pytest.raises(ValueError, match="ran without a rule set"):
        dose_response(
            series[0], Design(series[0].samples, "~ 1"), dose=AXIS, tests="dose_groups"
        ).rules()


def test_the_model_figure_draws_the_rule_set_s_means(series, etap) -> None:
    """Under rules, plot_dose_response with a model draws the rule set's curves over
    the means of its log2(CPM + 1) values, minus the vehicle's."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from cytepool.plotting import plot_dose_response

    group = etap.groups[0]
    table = etap.table(group, "rule_set")
    gene = table.filter(table["passes_prefilter"])["feature_id"][0]
    fits = etap.curve_fits(group)
    ids = etap.table(group, "curves")["feature_id"][:: len(fits.models)].tolist()
    means = fits.means[ids.index(gene)]
    fig, ax = plt.subplots()
    plot_dose_response(etap, group, genes=[gene], model="best", ax=ax)
    (dots,) = [a for a in ax.get_children() if a.get_gid() == "doses"]
    np.testing.assert_allclose(dots.get_offsets()[:, 1], means - means[0], atol=1e-12)
    plt.close(fig)


def test_without_rules_the_analysis_is_unchanged(series, etap) -> None:
    cube, _ = series
    plain = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, rng=5)
    for group in plain.groups:
        for name in ("spline", "dose_groups", "gate"):
            assert plain.table(group, name).equals(etap.table(group, name))
    assert "rule_set" not in plain.provenance["groups"][0] or (
        plain.provenance["groups"][0]["rule_set"] is None
    )
    assert "rule_set_status" not in plain.summary


def test_seeds_do_not_depend_on_other_cell_types(series, etap) -> None:
    cube, _ = series
    group = etap.groups[1]
    alone = dose_response(
        cube,
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        rules="etap_2024",
        groups=[group],
        rng=5,
    )
    assert alone.table(group, "rule_set").equals(etap.table(group, "rule_set"))
    again = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, rules="etap_2024", rng=6
    )
    assert not np.array_equal(
        again.table(group, "rule_set")["williams_trend_p_value"],
        etap.table(group, "rule_set")["williams_trend_p_value"],
    )


def test_a_cell_type_without_signal_is_inactive() -> None:
    """With no responsive gene, the ANOVA pre-modelling test passes no gene, the
    cell type is inactive, and its table keeps the ANOVA columns alone."""
    cube, _ = simulate_dose_series(
        LADDER, replicates=3, cell_types=1, n_genes=150, responsive=0.0, rng=11
    )
    result = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, rules="ntp_2018", rng=1
    )
    group = result.groups[0]
    table = result.table(group, "rule_set")
    assert (table["anova_adjusted_p_value"] >= 0.05).all()
    assert np.isnan(table["williams_trend_p_value"]).all()
    assert not table["passes_prefilter"].any() and not table["accepted"].any()
    status = "inactive: no gene passes the ANOVA pre-modelling test"
    assert result.summary["rule_set_status"].tolist() == [status]
    with pytest.raises(KeyError, match=f"no curves in '{group}': {status}"):
        result.curve_fits(group)
    outcomes = result.rules(group)
    found = dict(zip(outcomes["setting"].tolist(), outcomes["outcome"].tolist()))
    assert found["ANOVA pre-modelling test"] == status
    assert found["acceptance"] == ""


def test_a_cell_type_where_no_gene_passes_the_prefilter_has_no_curves() -> None:
    """A gene that rises steadily by 1.2-fold passes the ANOVA pre-modelling test and
    Williams' trend test but not the 1.5-fold change, so the cell type is analysed,
    with no gene to fit."""
    levels = np.array([0.0, 0.1, 1.0, 10.0])
    doses = np.repeat(levels, 3)
    means = np.full((doses.size, 40), 2000.0)
    means[:, 0] *= 1.2 ** (np.searchsorted(levels, doses) / 3)
    counts = np.random.default_rng(3).poisson(means)
    samples = {"sample_id": [f"s{i}" for i in range(doses.size)], "dose": doses}
    cube = from_bulk(counts, samples=samples)
    result = dose_response(
        cube, Design(cube.samples, "~ 1"), dose=AXIS, rules="ntp_2018", rng=1
    )
    group = result.groups[0]
    table = result.table(group, "rule_set")
    assert table["anova_adjusted_p_value"][0] < 0.05
    assert table["williams_trend_p_value"][0] < 0.05
    assert abs(table["max_log2_fold_change"][0]) < np.log2(1.5)
    assert not table["passes_prefilter"].any()
    assert result.summary["rule_set_status"].tolist() == ["analysed"]
    assert result.summary["n_genes_passing_prefilter"].tolist() == [0]
    with pytest.raises(KeyError, match="no gene passes the prefilter of ntp_2018"):
        result.curve_fits(group)


def with_cells(cube: Cube, n_cells: np.ndarray) -> Cube:
    """The cube with other numbers of cells."""
    return Cube(
        {"counts": np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
    )


def test_etap_leaves_out_short_dose_levels_and_needs_three_vehicles(series) -> None:
    """ETAP Sec. 3.4.5.2: a dose level with fewer than 2 passing samples is left out
    of the analysis; with fewer than 3 passing vehicle samples the cell type is not
    modelled."""
    cube, _ = series
    group = cube.groups["group_id"][0]
    doses = np.asarray(cube.samples["dose"], dtype=float)
    lowest = np.flatnonzero(doses == 0.03)
    vehicle = np.flatnonzero(doses == 0.0)
    # Three of the four samples at 0.03 ug/kg keep too few cells to pass in the first
    # cell type, which leaves one; two of the four at 0.1 ug/kg, which leaves two.
    cells = np.array(cube.n_cells, copy=True)
    cells[lowest[:3], 0] = 5
    cells[np.flatnonzero(doses == 0.1)[:2], 0] = 5
    result = dose_response(
        with_cells(cube, cells),
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        rules="etap_2024",
        groups=[group],
        rng=5,
    )
    record = result.provenance["groups"][0]["rule_set"]
    assert record["dose_levels_left_out"] == [0.03]
    assert 0.03 not in result.curve_fits(group).doses.tolist()
    assert result.curve_fits(group).doses[1] == 0.1  # the lowest dose the rules read
    used = result.design_report.samples.filter(group_id=group)
    at = np.asarray(used["dose"], dtype=float) == 0.1
    assert np.count_nonzero(np.asarray(used["used"], dtype=bool) & at) == 2
    found = dict(
        zip(*(result.rules(group)[c].tolist() for c in ("setting", "outcome")))
    )
    assert "dose levels left out: 0.03" in found["data set"]
    cells[vehicle[:2], 0] = 5
    short = dose_response(
        with_cells(cube, cells),
        Design(cube.samples, "~ 1"),
        dose=AXIS,
        rules="etap_2024",
        groups=[group],
        rng=5,
    )
    status = "not modelled: 2 passing vehicle samples, ETAP needs 3"
    assert short.summary["rule_set_status"].tolist() == [status]
    found = dict(zip(*(short.rules(group)[c].tolist() for c in ("setting", "outcome"))))
    assert found["data set"] == status
    with pytest.raises(KeyError, match=f"etap_2024 did not run in '{group}': {status}"):
        short.table(group, "rule_set")


# The documents' choices of model, on real fits with one quantity set by hand.


HALF_LOG = np.repeat([0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 4)


def gene_values(seed: int, mean: np.ndarray, sd: float = 0.1) -> np.ndarray:
    return (8.0 + mean + np.random.default_rng(seed).normal(0.0, sd, HALF_LOG.size))[
        :, None
    ]


def exact_line(slope: float) -> np.ndarray:
    """A straight line whose dose-group means lie on it exactly: the replicates'
    deviations sum to 0 in each group."""
    deviations = np.tile([0.1, -0.1, 0.05, -0.05], HALF_LOG.size // 4)
    return (8.0 + slope * HALF_LOG + deviations)[:, None]


def test_aic_leaves_out_parameters_at_the_bounds_of_bmds_2s_programs() -> None:
    """On a line whose means lie on it, power at n = 1 (bounded, so 2 mean
    parameters) ties the line, and the tie goes to the line, first in the set. A Hill
    fit with n at 1 counts 3, so it beats the quadratic of 3 when its log-likelihood
    is higher by 0.5, which counting n would reverse."""
    fits = fit_dose_curves(HALF_LOG, exact_line(0.04), models="bmdexpress")
    power = fits.models.index("power")
    assert fits.parameters_at_bounds[0, power] == "n"
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    assert found.gene_models[0] == "linear" and found.model_choice[0] == "lowest AIC"
    shallow = gene_values(7, 0.35 * np.log10(1.0 + HALF_LOG / 0.001), sd=0.05)
    fits = fit_dose_curves(HALF_LOG, shallow, models="bmdexpress")
    hill = fits.models.index("hill")
    assert "n" in fits.parameters_at_bounds[0, hill].split(", ")
    ll = np.full(fits.log_likelihood.shape, -50.0)
    ll[0, fits.models.index("polynomial_2")] = 10.0
    ll[0, hill] = 10.5
    forced = replace(fits, log_likelihood=ll)
    found = benchmark_doses(forced, BenchmarkResponse.sd(1.0), model="ntp_2018")
    assert found.gene_models[0] == "hill"


def test_etap_leaves_out_a_hill_fit_whose_k_is_below_a_third_of_the_lowest_dose() -> (
    None
):
    """ETAP Sec. 3.4.5.3: on a step at the lowest dose, Hill's k sits at 0, and ETAP
    chooses another model where NTP keeps Hill; the flag reads k < (lowest positive
    dose) / 3, so k at exactly a third stays."""
    step = gene_values(2, 1.5 * (HALF_LOG > 0))
    fits = fit_dose_curves(HALF_LOG, step, models="bmdexpress")
    etap = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    ntp = benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="ntp_2018")
    assert ntp.gene_models[0] == "hill"
    assert etap.gene_models[0] != "hill"
    assert etap.model_choice[0] == (
        "lowest AIC, Hill left out: k below a third of the lowest dose"
    )
    hill = fits.models.index("hill")
    parameters = fits.parameters.copy()
    k = fits.parameter_names.index("k")
    flags = []
    for value in (0.1 / 3, np.nextafter(0.1 / 3, 0.0), np.nan):
        parameters[0, hill, k] = value
        flags.append(bool(_flagged_hill(replace(fits, parameters=parameters))[0, hill]))
    assert flags == [False, True, False]


def test_a_model_without_bounds_or_a_bmd_gives_way_to_the_next() -> None:
    """A model whose bounds are not numbers (here Hill's height moved off its
    maximum, so the profile finds a higher likelihood than the fit's) leaves the
    choice; so does a model whose change never reaches the BMR (a plateau below it),
    which gets no bounds at all; the next by AIC is chosen."""
    rise = gene_values(3, 1.4 * HALF_LOG / (3.0 + HALF_LOG))
    fits = fit_dose_curves(HALF_LOG, rise, models="bmdexpress")
    hill, linear = fits.models.index("hill"), fits.models.index("linear")
    ll = np.full(fits.log_likelihood.shape, -50.0)
    ll[0, hill], ll[0, linear] = 35.0, 29.0
    parameters = fits.parameters.copy()
    parameters[0, hill, fits.parameter_names.index("v")] *= 0.95
    moved = replace(fits, log_likelihood=ll, parameters=parameters)
    found = benchmark_doses(moved, BenchmarkResponse.sd(1.349), model="etap_2024")
    assert found.gene_models[0] == "linear"
    assert found.model_choice[0] == "lowest AIC of the models with a BMD and bounds"
    flat = gene_values(8, 0.0 * HALF_LOG)
    fits = fit_dose_curves(HALF_LOG, flat, models="bmdexpress")
    parameters = fits.parameters.copy()
    parameters[0, hill, fits.parameter_names.index("v")] = 0.03  # 0.3 SD
    low = replace(fits, parameters=parameters)
    plateau = benchmark_doses(low, BenchmarkResponse.sd(1.349), model="hill")
    assert plateau.status[0, 0] == "not reached"
    ll = np.full(fits.log_likelihood.shape, -50.0)
    ll[0, hill], ll[0, linear] = 35.0, 29.0
    found = benchmark_doses(
        replace(low, log_likelihood=ll), BenchmarkResponse.sd(1.349), model="etap_2024"
    )
    assert found.gene_models[0] == "linear"
    # A line whose change reaches the BMR only beyond the search, near 10,000 ug/kg,
    # as BMDS 2 reports such BMDs: chosen, with no BMD, so that the gene fails the
    # checks, where a choice of reached models alone would choose none.
    fits = fit_dose_curves(HALF_LOG, exact_line(1e-5), models="bmdexpress")
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    assert found.gene_models[0] == "linear" and found.status[0, 0] == "not reached"
    assert not found.accepted("etap_2024")[0, 0]


def test_the_documents_bounds_end_at_the_piece_that_holds_the_bmd() -> None:
    """BMDS 2 bounds a BMD by the piece of the profile's region that holds it. On a
    weak Hill response whose region below the BMD has a second piece, from a step
    just above dose 0, the whole region's BMDL is 0 and the piece's lies between that
    piece and the BMD; on a line, whose region is one piece, both are the same."""
    x = np.repeat([0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0], 3)
    y = np.array(
        [5.8463, 5.7155, 6.2424, 6.6158, 6.4698, 6.3317, 6.774, 6.882, 6.1706]
        + [6.3315, 6.8764, 6.1794, 6.4309, 6.9166, 6.3524, 6.663, 6.4653, 6.5553]
        + [6.5106, 5.9315, 6.5692, 7.2186, 6.923, 6.8938, 7.0501, 6.6016, 7.0952]
    )
    line = 6.0 + 0.04 * x + np.tile([0.1, -0.1, 0.0], 9)
    fits = fit_dose_curves(x, np.column_stack([y, line]), models="bmdexpress")
    bmr = BenchmarkResponse.absolute(0.3)
    hill = fits.models.index("hill")
    _, whole = _per_model(fits, bmr, hill, True)
    _, piece = _per_model(fits, bmr, hill, True, piece=True)
    assert 3.0 < whole["bmd"][0] < 10.0 and whole["bmd"][0] == piece["bmd"][0]
    assert whole["bmdl"][0] == 0.0
    assert 0.03 < piece["bmdl"][0] < whole["bmd"][0]
    assert piece["bmdu"][0] <= whole["bmdu"][0]
    linear = fits.models.index("linear")
    _, whole = _per_model(fits, bmr, linear, True)
    _, piece = _per_model(fits, bmr, linear, True, piece=True)
    for bound in ("bmdl", "bmdu"):
        assert piece[bound][1] == whole[bound][1]
    # Above the BMD: a quadratic that rises and then falls can return to the BMR's
    # change far beyond the doses, so a near-linear gene's whole region reaches the
    # search's end, and the piece ends within the doses.
    doses = np.repeat([0.0, 0.3, 1.0, 3.0, 10.0, 30.0], 4)
    rising = 8.0 + 0.03 * doses + np.random.default_rng(0).normal(0.0, 0.15, doses.size)
    near = fit_dose_curves(doses, rising[:, None], models="bmdexpress")
    quadratic = near.models.index("polynomial_2")
    _, whole = _per_model(near, BenchmarkResponse.sd(1.0), quadratic, True)
    _, piece = _per_model(near, BenchmarkResponse.sd(1.0), quadratic, True, piece=True)
    assert whole["bmdu"][0] == np.inf and whole["bmdu"][0] > piece["bmdu"][0]
    assert whole["bmd"][0] < piece["bmdu"][0] < 30.0
    # The documents' choices read the piece: exponential 5 here, whose region also
    # reaches down to a step above dose 0.
    found = benchmark_doses(fits, bmr, model="ntp_2018")
    m = fits.models.index(str(found.gene_models[0]))
    _, whole = _per_model(fits, bmr, m, True)
    _, piece = _per_model(fits, bmr, m, True, piece=True)
    assert whole["bmdl"][0] == 0.0 < piece["bmdl"][0] == found.bmdl[0, 0]


def test_test_4_is_the_likelihood_ratio_of_the_samples() -> None:
    """BMDS's test 4 on K - (p - b) degrees of freedom, b the parameters at a bound of
    BMDS 2's programs, from the samples' sums of squares about the curve and about
    the dose-group means: on a step at the lowest dose NTP chooses Hill, with k at 0
    and n at 1, so b = 2."""
    gene = gene_values(2, 1.5 * (HALF_LOG > 0))
    fits = fit_dose_curves(HALF_LOG, gene, models="bmdexpress")
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="ntp_2018")
    assert found.gene_models[0] == "hill"
    m = fits.models.index("hill")
    k, n = (fits.parameters[0, m, fits.parameter_names.index(q)] for q in ("k", "n"))
    assert (k, n) == (0.0, 1.0)
    levels = np.unique(HALF_LOG)
    y = gene[:, 0]
    groups = [y[HALF_LOG == level] for level in levels]
    within = sum(((g - g.mean()) ** 2).sum() for g in groups)
    fitted = curve_values_of(fits, m, levels)
    about = sum(((g - fitted[i]) ** 2).sum() for i, g in enumerate(groups))
    statistic = y.size * np.log(about / within)
    assert found.fit_p_value[0] == pytest.approx(
        chi2.sf(statistic, levels.size - 2), rel=1e-8
    )


def curve_values_of(fits, m: int, doses: np.ndarray) -> np.ndarray:
    from cytepool.stats._dose_curve_models import curve_values, model_of

    model = model_of(fits.model_set, fits.models[m])
    names = [fits.parameter_names.index(p) for p in model.parameters]
    return curve_values(model, fits.parameters[0, m, names][None, :], doses)[0]


def test_the_vehicle_reading_of_etap_is_a_line_s_closed_form() -> None:
    """For a straight line, the BMD at 1.349 times the vehicle replicates' sample SD
    is that size over the slope."""
    line = exact_line(0.05)
    fits = fit_dose_curves(HALF_LOG, line, models="bmdexpress")
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    assert found.gene_models[0] == "linear"
    slope = fits.parameters[
        0, fits.models.index("linear"), fits.parameter_names.index("b1")
    ]
    sd = np.std(line[HALF_LOG == 0.0, 0], ddof=1)
    assert found.bmd_vehicle_sd[0] == pytest.approx(1.349 * sd / abs(slope), rel=1e-6)
    ntp = benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="ntp_2018")
    assert np.isnan(ntp.bmd_vehicle_sd).all()
    # No vehicle reading of an absolute BMR, nor of a vehicle without spread.
    absolute = benchmark_doses(fits, BenchmarkResponse.absolute(0.3), model="etap_2024")
    assert np.isnan(absolute.bmd_vehicle_sd).all()
    still = line.copy()
    still[HALF_LOG == 0.0, 0] = 8.0
    flat = benchmark_doses(
        fit_dose_curves(HALF_LOG, still, models="bmdexpress"),
        BenchmarkResponse.sd(1.349),
        model="etap_2024",
    )
    assert np.isnan(flat.bmd_vehicle_sd).all() and np.isfinite(flat.bmd[0]).all()
    # The table carries the choice and its test 4 under a document's choice only;
    # the 1 SD row added after 1.349 SD keeps the gene's model and its p-value.
    table = found.table()
    assert table.columns[:5] == (
        "feature_id",
        "bmr",
        "model",
        "model_choice",
        "fit_p_value",
    )
    assert table["model_choice"].tolist() == [found.model_choice[0]] * 2
    assert table["fit_p_value"].tolist() == [found.fit_p_value[0]] * 2
    assert (
        "model_choice" not in benchmark_doses(fits, BenchmarkResponse.sd(1.0)).table()
    )


def test_the_documents_choices_refuse_what_they_cannot_read() -> None:
    """Each choice is its program's, on the set it fits, under constant variance and
    with bounds."""
    fits = fit_dose_curves(HALF_LOG, exact_line(0.05), models="bmdexpress")
    epa = fit_dose_curves(HALF_LOG, exact_line(0.05), models="epa")
    bmr = BenchmarkResponse.sd(1.0)
    with pytest.raises(ValueError, match="fit the curves with models='bmdexpress'"):
        benchmark_doses(epa, bmr, model="ntp_2018")
    with pytest.raises(
        ValueError, match="pybmds 25.2 .* fit the curves with models='epa'"
    ):
        benchmark_doses(fits, bmr, model="bmds_2026")
    with pytest.raises(ValueError, match="fit the curves under constant variance"):
        benchmark_doses(replace(fits, variance="weighted"), bmr, model="etap_2024")
    with pytest.raises(ValueError, match="keeps a model only where its BMD has bounds"):
        benchmark_doses(fits, bmr, model="ntp_2018", bounds=False)


def test_bmds_recommends_by_aic_within_3_fold_and_by_bmdl_beyond() -> None:
    """The BMDS User Guide's rule as pybmds 25.2 applies it to the viable models."""
    aic = np.array(
        [[10.0, 9.0, 11.0], [10.0, 9.0, 11.0], [5.0, 5.0, 7.0], [1.0, 2.0, 3.0]]
    )
    bmdl = np.array(
        [[1.0, 3.0, 2.0], [1.0, 3.0000001, 2.0], [1.0, 1.5, 1.2], [1.0, 2.0, 3.0]]
    )
    viable = np.array(
        [[True, True, True], [True, True, True], [True, True, False], [False] * 3]
    )
    n_parameters = np.array([4.0, 3.0, 5.0])
    chosen, why = _recommended(aic, bmdl, viable, n_parameters)
    # Exactly 3-fold: the lowest AIC; just beyond: the lowest BMDL; a tie of AIC goes to
    # the fewer parameters, here the later model; none viable: none.
    assert chosen.tolist() == [1, 0, 1, -1]
    assert why.tolist() == [
        "lowest AIC of the viable models, BMDLs within 3-fold",
        "lowest BMDL of the viable models, BMDLs more than 3-fold apart",
        "lowest AIC of the viable models, BMDLs within 3-fold",
        "no viable model",
    ]
    # A tie between the first and the last goes to the first, of fewer parameters; with
    # as many parameters, a tie goes to the first in set order.
    tie = np.array([[5.0, 6.0, 5.0]])
    every = np.ones((1, 3), dtype=bool)
    near = np.array([[1.0, 1.5, 1.2]])
    assert _recommended(tie, near, every, n_parameters)[0].tolist() == [0]
    assert _recommended(tie, near, every, np.array([5.0, 3.0, 4.0]))[0].tolist() == [2]
    assert _recommended(tie, near, every, np.full(3, 3.0))[0].tolist() == [0]


def test_acceptance_at_each_threshold() -> None:
    """NTP pp. 8 to 9 and ETAP Sec. 3.4.5.3, each criterion at its edge, as the
    documents state them: below the highest dose, fit p-value above 0.1, BMDU / BMDL
    below 40 (NTP) and BMD / BMDL below 20 (ETAP); NTP's 10-fold flag is kept."""
    line = gene_values(6, 0.05 * HALF_LOG)
    fits = fit_dose_curves(HALF_LOG, np.repeat(line, 10, axis=1), models="bmdexpress")
    found = benchmark_doses(fits, BenchmarkResponse.sd(1.349), model="etap_2024")
    top, lowest = 30.0, 0.1
    rows = {
        "good": (1.0, 0.5, 2.0, 0.5),
        "at the top dose": (top, 0.5 * top, 2 * top, 0.5),
        "fit p 0.1": (1.0, 0.5, 2.0, 0.1),
        "BMD/BMDL 20": (2.0, 0.1, 3.0, 0.5),
        "BMDU/BMDL 40": (1.0, 0.1, 4.0, 0.5),
        "a tenth of the lowest": (lowest / 10, lowest / 20, lowest / 5, 0.5),
        "below a tenth": (0.9 * lowest / 10, lowest / 20, lowest / 5, 0.5),
        "BMDL 0": (1.0, 0.0, 2.0, 0.5),
        "BMDU infinite": (1.0, 0.5, np.inf, 0.5),
        "no fit p-value": (1.0, 0.5, 2.0, np.nan),
    }
    bmd, bmdl, bmdu, p = (np.array([row[i] for row in rows.values()]) for i in range(4))
    edges = replace(
        found,
        bmd=np.tile(bmd, (2, 1)),
        bmdl=np.tile(bmdl, (2, 1)),
        bmdu=np.tile(bmdu, (2, 1)),
        fit_p_value=p,
    )
    ntp = dict(zip(rows, edges.accepted("ntp_2018")[0].tolist()))
    etap = dict(zip(rows, edges.accepted("etap_2024")[0].tolist()))
    assert ntp == {
        "good": True,
        "at the top dose": False,
        "fit p 0.1": False,
        "BMD/BMDL 20": True,
        "BMDU/BMDL 40": False,
        "a tenth of the lowest": True,
        "below a tenth": True,
        "BMDL 0": False,
        "BMDU infinite": False,
        "no fit p-value": False,
    }
    assert etap == {
        "good": True,
        "at the top dose": False,
        "fit p 0.1": False,
        "BMD/BMDL 20": False,
        "BMDU/BMDL 40": True,
        "a tenth of the lowest": True,
        "below a tenth": True,
        "BMDL 0": False,
        "BMDU infinite": True,
        "no fit p-value": False,
    }
    # The checks skip the fit p-value a row lacks; acceptance does not.
    flags = edges.table(flags="etap_2024")["flags_etap_2024"].tolist()
    assert flags[list(rows).index("no fit p-value")] == ""
    with pytest.raises(ValueError, match="BMDS's bins"):
        edges.accepted("bmds_2026")
    with pytest.raises(ValueError, match="these BMDs have none"):
        replace(edges, bounds=False).accepted("ntp_2018")


# Against BMDExpress 2.30.520, pybmds 25.2 and R, from validation/toxicology_rule_sets.py.


DATA = Path(__file__).parent / "data" / "toxicology_rule_sets"
MODELS = (
    "linear",
    "polynomial_2",
    "power",
    "hill",
    "exponential_2",
    "exponential_3",
    "exponential_4",
    "exponential_5",
)
BEST = {
    "linear": "Linear",
    "polynomial_2": "Poly 2",
    "power": "Power",
    "hill": "Hill",
    "exponential_2": "Exp 2",
    "exponential_3": "Exp 3",
    "exponential_4": "Exp 4",
    "exponential_5": "Exp 5",
}


def bmdexpress() -> dict:
    return json.loads((DATA / "bmdexpress.json").read_text())


def numbers(values: list) -> np.ndarray:
    return np.array([math.nan if v is None else v for v in values], dtype=float)


@pytest.mark.parametrize("rule", ["ntp_2018", "etap_2024"])
def test_the_choice_on_bmdexpress_s_own_fits_is_bmdexpress_s(rule: str) -> None:
    """The lowest AIC among models with a BMD, BMDL and BMDU, ties to the set's order,
    ETAP's Hill rule: on BMDExpress's own AICs, BMDs, bounds and Hill k, the rule gives
    BMDExpress's model."""
    for key, entry in bmdexpress().items():
        found = entry[rule]
        doses = np.array(entry["doses"])
        lowest = float(np.float32(doses[doses > 0].min()))
        aic = np.column_stack([numbers(found[f"AIC {m}"]) for m in MODELS])
        bmd = np.column_stack([numbers(found[f"BMD {m}"]) for m in MODELS])
        bmdl = np.column_stack([numbers(found[f"BMDL {m}"]) for m in MODELS])
        bmdu = np.column_stack([numbers(found[f"BMDU {m}"]) for m in MODELS])
        open_ = np.isfinite(aic) & np.isfinite(bmd)
        if rule == "etap_2024":
            open_[:, MODELS.index("hill")] &= ~(numbers(found["hill_k"]) < lowest / 3)
        chosen = _first_with_bounds(
            aic, open_, lambda g, m: ~np.isnan(bmdl[g, m]) & ~np.isnan(bmdu[g, m])
        )
        ours = [BEST[MODELS[c]] if c >= 0 else "" for c in chosen]
        assert ours == found["best"], key


def test_the_bmdexpress_set_reaches_bmdexpress_s_maxima() -> None:
    """CytePool's "bmdexpress" fits reach the maximum BMDS 2's programs report, to
    1e-4 in log-likelihood, but where exponential 4 and 5 hold c at CytePool's limit
    of 1,000 and BMDS 2's c goes beyond it."""
    for key, entry in bmdexpress().items():
        doses = np.array(entry["doses"])
        fits = fit_dose_curves(doses, np.array(entry["values"]), models="bmdexpress")
        shift = doses.size / 2 * math.log(2 * math.pi)
        c = fits.parameter_names.index("c")
        for m, model in enumerate(fits.models):
            theirs = numbers(entry["ntp_2018"][f"fitLogLikelihood {model}"]) - shift
            gap = theirs - fits.log_likelihood[:, m]
            capped = fits.parameters[:, m, c] == 1000.0
            assert np.all((gap <= 1e-4) | np.isnan(theirs) | capped), (key, model)


@pytest.mark.parametrize("rule", ["ntp_2018", "etap_2024"])
def test_acceptance_keeps_what_bmdexpress_s_filters_keep(rule: str) -> None:
    """The documents' checks on BMDExpress's best models keep the genes its own
    category filters keep; NTP's BMDs more than 10-fold below the lowest dose are
    kept by NTP and removed by BMDExpress's nFoldBelowLowestDose filter."""
    for key, entry in bmdexpress().items():
        found = entry[rule]
        doses = np.array(entry["doses"])
        n = len(found["best"])
        missing = np.full(n, np.nan)
        inputs = FlagInputs(
            bmd=numbers(found["Best BMD"]),
            bmdl=numbers(found["Best BMDL"]),
            bmdu=numbers(found["Best BMDU"]),
            bounds=True,
            lowest_dose=float(np.float32(doses[doses > 0].min())),
            top_dose=float(np.float32(doses.max())),
            fit_p_value=numbers(found["Best fitPValue"]),
            residual_near_bmd=missing,
            residual_at_vehicle=missing,
            vehicle_sd_ratio=missing,
            constant_variance_p_value=missing,
        )
        kept = accepted_rows(rule, inputs).tolist()
        flagged = [
            "(NTP keeps the gene)" in text for text in failed_checks(rule, inputs)
        ]
        expected = (
            [k and not f for k, f in zip(kept, flagged)] if rule == "ntp_2018" else kept
        )
        tie = numbers(found["Best fitPValue"]) == 0.1
        assert [e for e, t in zip(expected, tie) if not t] == [
            k for k, t in zip(found["kept_by_filters"], tie) if not t
        ], key


@pytest.mark.parametrize("rule", ["ntp_2018", "etap_2024"])
def test_the_documents_bounds_are_bmds_2s_where_the_fits_agree(rule: str) -> None:
    """Where CytePool chooses BMDExpress's model and both reach its maximum, the
    bounds of the BMD are BMDS 2's within 5e-3, or 0 and infinity where BMDS 2's lie
    beyond a hundredth of the lowest dose and 50 times the top dose. BMDS 2's search
    can stop short of the piece's end, so a few bounds lie at CytePool's search ends
    or within 3-fold, as the validation's row reports them; the whole region would
    put many more far out."""
    compared = close = 0
    for key, entry in bmdexpress().items():
        doses = np.array(entry["doses"])
        lowest, top = doses[doses > 0].min(), doses.max()
        fits = fit_dose_curves(doses, np.array(entry["values"]), models="bmdexpress")
        bmr = BenchmarkResponse.sd(1.349 if rule == "etap_2024" else 1.0)
        found = benchmark_doses(fits, bmr, model=rule)
        theirs = entry[rule]
        shift = doses.size / 2 * math.log(2 * math.pi)
        for i, best in enumerate(theirs["best"]):
            model = str(found.gene_models[i])
            if not model or BEST[model] != best:
                continue
            m = fits.models.index(model)
            their_ll = numbers(theirs[f"fitLogLikelihood {model}"])[i] - shift
            if not abs(fits.log_likelihood[i, m] - their_ll) < 1e-3:
                continue
            if found.status[0, i] == "below a hundredth of the lowest dose":
                continue
            for ours, label in ((found.bmdl[0, i], "BMDL"), (found.bmdu[0, i], "BMDU")):
                bound = numbers(theirs[f"Best {label}"])[i]
                if not math.isfinite(bound):
                    continue
                if label == "BMDL" and bound < lowest / 100:
                    assert ours == 0.0, (key, i)
                    continue
                if label == "BMDU" and bound > 50 * top:
                    assert ours == math.inf, (key, i)
                    continue
                compared += 1
                if ours == pytest.approx(bound, rel=5e-3):
                    close += 1
                elif ours not in (0.0, math.inf):
                    assert 1 / 3 < ours / bound < 3, (key, i, label)
    assert compared > 80 and close >= 0.9 * compared


def test_acceptance_at_the_edges_is_r_s_reading_of_the_documents() -> None:
    edges = json.loads((DATA / "acceptance.json").read_text())
    rows = numbers([v for row in edges["rows"] for v in row]).reshape(-1, 4)
    found = {}
    for rule in ("ntp_2018", "etap_2024"):
        n = rows.shape[0]
        missing = np.full(n, np.nan)
        inputs = FlagInputs(
            bmd=rows[:, 0],
            bmdl=rows[:, 1],
            bmdu=rows[:, 2],
            bounds=True,
            lowest_dose=edges["lowest"],
            top_dose=edges["top"],
            fit_p_value=rows[:, 3],
            residual_near_bmd=missing,
            residual_at_vehicle=missing,
            vehicle_sd_ratio=missing,
            constant_variance_p_value=missing,
        )
        found[rule] = accepted_rows(rule, inputs).tolist()
        if rule == "ntp_2018":
            texts = failed_checks(rule, inputs)
            assert ["(NTP keeps the gene)" in x for x in texts] == edges["ntp_flag"]
    assert found["ntp_2018"] == edges["ntp"]
    assert found["etap_2024"] == edges["etap"]


def test_bmds_recommends_what_pybmds_recommends_where_the_fits_agree() -> None:
    """Of the seeded genes where both programs reach every model's maximum with the
    same parameters counted, as the validation's row compares them, each gets
    pybmds's recommendation; 15 of the 60 qualify in the recorded run."""
    found = json.loads((DATA / "pybmds.json").read_text())
    doses = np.array(found["doses"])
    fits = fit_dose_curves(doses, np.array(found["values"]), models="epa")
    recommended = benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="bmds_2026")
    counted = _pybmds_counted(fits)
    order = [
        "linear",
        "polynomial_2",
        "power",
        "hill",
        "exponential_3",
        "exponential_5",
    ]
    compared = 0
    for g, index in enumerate(found["recommended"]):
        rows = [fits.models.index(m) for m in order]
        lls = fits.log_likelihood[g, rows]
        theirs = np.array(found["log_likelihood"][g], dtype=float)
        # pybmds counts the parameters off its bounds, as _pybmds_counted does.
        free = np.sum(~np.isnan(fits.parameters[g, rows]), axis=1)
        free = free - np.array(found["n_bounded"][g])
        if np.all(np.abs(lls - theirs) < 1e-3) and np.all(free == counted[g, rows]):
            compared += 1
            assert recommended.gene_models[g] == (
                order[index] if index is not None else ""
            ), g
    assert compared >= 15


# The chapter's table of both rule sets and the numbers of its examples.

CHAPTER = Path(__file__).parents[1] / "docs" / "source" / "statistics" / "potency.md"


def chapter_cell(value: str, source: str, document: str) -> str:
    """A setting's value with its source, the document's name and what it takes
    from ETAP left out, as the chapter's table writes it."""
    parts = [p for p in source.split("; ") if not p.startswith("taken from ETAP")]
    return f"{value} ({'; '.join(p.removeprefix(f'{document}, ') for p in parts)})"


def test_the_chapter_s_table_is_the_rule_sets_data() -> None:
    """Each row of the chapter's table is a setting of both rule sets, in workflow
    order, with each document's value and source and BMDExpress's setting; the
    steps not applied are named in the text."""
    text = CHAPTER.read_text()
    ntp = {setting.setting: setting for setting in NTP.settings}
    assert list(ntp) == [setting.setting for setting in ETAP.settings]
    rows = []
    for etap in ETAP.settings:
        other = ntp[etap.setting]
        value = "as ETAP" if other.value == etap.value else other.value
        bmdexpress = (
            etap.bmdexpress
            if etap.bmdexpress == other.bmdexpress
            else f"ETAP: {etap.bmdexpress}; NTP: {other.bmdexpress}"
        )
        rows.append(
            f"| {etap.setting} | {chapter_cell(etap.value, etap.source, 'US EPA 2024')} "
            f"| {chapter_cell(value, other.source, 'NTP 2018')} | {bmdexpress} |"
        )
    header = "| Setting | ETAP (US EPA 2024) | NTP 2018 | BMDExpress 2.30.520 |\n"
    table = header + "| --- | --- | --- | --- |\n" + "\n".join(rows) + "\n"
    assert table in text
    for name in ("sequencing quality", "outliers", "one gene per feature"):
        assert name in text
    names = {s.setting for s in (*ETAP.not_applied, *NTP.not_applied)}
    assert names == {
        "sample quality",
        "outlier samples",
        "one gene per feature",
        "models without a spare dose group",
        "calibration of the prefilter",
    }


def test_the_chapter_s_rule_set_examples_run() -> None:
    """The chapter's examples of ETAP's rule set and BMDS's recommended model, on the
    chapter's series, and the numbers its text states."""
    text = " ".join(CHAPTER.read_text().split())
    series, rules_block, bmds_block = re.findall(
        r"```python\n(.*?)```", CHAPTER.read_text(), re.S
    )
    namespace: dict = {}
    # The series and its axis, without the chapter's first analysis.
    exec(series[: series.index("bmr = [")], namespace)  # noqa: S102
    exec(rules_block, namespace)  # noqa: S102
    etap, settings, genes = (namespace[k] for k in ("etap", "settings", "genes"))
    found = dict(zip(settings["setting"].tolist(), settings["outcome"].tolist()))
    anova = int(found["ANOVA pre-modelling test"].split()[1])
    kept = int(found["values"].split()[0].replace(",", ""))
    passing = int(np.count_nonzero(genes["passes_prefilter"]))
    accepted = int(np.count_nonzero(genes["accepted"]))
    assert (
        f"{anova} of the {kept:,} genes with a mean count of at least 5 pass the ANOVA "
        f"pre-modelling test" in text
    )
    assert (
        f"{passing} pass Williams' trend test and the fold change, and {accepted} of them"
        in text
    )
    assert found["design minimums"].startswith("deviation: dose_spacing")
    assert "animals_per_sex_and_dose (3 at 0 ug/kg" in found["design minimums"]
    rows = np.asarray(genes["accepted"], dtype=bool)
    ratio = np.asarray(genes["bmd_vehicle_sd"])[rows] / np.asarray(genes["bmd"])[rows]
    quartiles = np.nanpercentile(ratio, [25, 50, 75])
    assert (
        f"a median of {quartiles[1]:.2f} times the BMD, with quartiles of "
        f"{quartiles[0]:.2f} and {quartiles[2]:.2f}" in text
    )
    exec(bmds_block, namespace)  # noqa: S102
    table = namespace["recommended"]
    gate = np.asarray(table["passes_gate"], dtype=bool)
    none = int(np.count_nonzero(gate & (table["model_choice"] == "no viable model")))
    fits = namespace["bmds"].curve_fits("hepatocyte")
    doses = benchmark_doses(fits, BenchmarkResponse.sd(1.0), model="linear")
    variance = np.asarray(doses.constant_variance_p_value < 0.05)
    unequal = int(np.count_nonzero(gate & variance))
    # A failed test of one variance makes every model questionable.
    assert np.all(table["model_choice"][gate & variance] == "no viable model")
    assert (
        f"{none} of the {int(gate.sum())} genes that pass the gate have no viable model "
        f"at 1 SD; in {unequal} of them" in text
    )
