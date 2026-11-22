"""Tests for dose tests of pathway scores and activities, and tests in one direction.

Reference values in ``tests/data/dose_response_signatures/`` come from
``validation/dose_response_signatures.py``: ``limma.json`` holds GSVA scores of
simulated dose series with limma 3.62.1's tests of them, which R computed from the
scores, the samples' doses and its own knots and spline columns, and for scores with
missing values from each score's own covariance; ``dosefinding.json`` holds scores
tested alone with DoseFinding 1.4-2's contrasts and t-statistics, multcomp 1.4-32's
Dunnett's and Williams' t-statistics, and their p-values from 10 million draws of the
multivariate t in R. Other tests check what R cannot: an ordinary linear model fitted
by hand, the route of a cube's own values, invariants of one-sided p-values,
refusals and the records.
"""

import functools
import json
import math
import pickle
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import stats

from cytepool import (
    Activities,
    Design,
    DoseAxis,
    GeneSets,
    Network,
    PathwayScores,
    Table,
    dose_response,
    dream_voom,
    gsva,
    limma_trend,
    permutation_check,
    simulate_dose_series,
    ulm,
)
from cytepool.dose_responses import _one_sided
from cytepool.stats.multiple_contrasts import stage_wise_adjusted_p_values
from cytepool.stats.multiple_testing import adjust_p_values

DATA = Path(__file__).parent / "data" / "dose_response_signatures"
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
TESTS = ("spline", "nonlinearity", "linear_trend", "dose_groups")
CALIBRATED = 5.5  # combined standard errors, as the validation script
# DoseFinding's names of the directions, which key its models in the fixture.
DOSEFINDING = {"up": "increasing", "down": "decreasing"}


@functools.cache
def limma_reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / "limma.json").read_text())
    return found


@functools.cache
def dosefinding_reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / "dosefinding.json").read_text())
    return found


def numbers(value: Any) -> np.ndarray:
    """R's numbers as float64, with its NA and Inf, which it writes as text."""

    def floats(item: Any) -> Any:
        if isinstance(item, list):
            return [floats(x) for x in item]
        if item is None or item in ("NA", "NaN"):
            return math.nan
        if isinstance(item, str):
            return float(item.replace("Inf", "inf"))
        return item

    return np.array(floats(value), dtype=np.float64)


def scores_of(record: dict[str, Any], group: str) -> PathwayScores:
    """Pathway scores of a fixture's matrix, with a set of the recorded size for each
    score, whose genes the tests do not read."""
    ids = record["sample_id"]
    profiles = {"profile_id": ids, "sample_id": ids, "dose": record["dose"]}
    if "sex" in record:
        profiles["sex"] = record["sex"]
    sets = GeneSets(
        {
            name: [f"{name}_{i}" for i in range(size)]
            for name, size in zip(record["names"], record["sizes"], strict=True)
        }
    )
    return PathwayScores(
        numbers(record["scores"]),
        Table(profiles),
        sets,
        "gsva",
        MappingProxyType({"group": group}),
    )


def design_of(scores: PathwayScores, formula: str = "~ 1") -> Design:
    columns = ["sample_id", "dose", *(["sex"] if "sex" in scores.profiles else [])]
    return Design(scores.profiles.select(columns), formula)


def assert_scaled(ours: Any, theirs: Any, tolerance: float = 1e-10) -> None:
    """Differences over max(|reference|, 1), with NaN where the reference has it."""
    a, b = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert a.shape == b.shape
    assert_array_equal(np.isnan(a), np.isnan(b))
    keep = ~np.isnan(b)
    same = a[keep] == b[keep]  # infinities alike
    with np.errstate(invalid="ignore"):
        difference = np.where(same, 0.0, np.abs(a[keep] - b[keep]))
    assert np.all(difference <= tolerance * np.maximum(np.abs(b[keep]), 1.0))


def assert_p_values(ours: Any, theirs: Any, tolerance: float = 1e-10) -> None:
    """p-values, to a tolerance relative to p max(1, |log p|), NaN where R has NA."""
    a, b = np.asarray(ours, dtype=float), np.asarray(theirs, dtype=float)
    assert a.shape == b.shape
    assert_array_equal(np.isnan(a), np.isnan(b))
    keep = ~np.isnan(b)
    scale = b[keep] * np.maximum(1.0, np.abs(np.log(b[keep])))
    assert np.all(np.abs(a[keep] - b[keep]) <= tolerance * scale)


JOBS = {
    # job: (scores, group, formula, options, tolerance)
    "ladder": ("ladder", "full", "~ 1", {}, 1e-10),
    "ladder, trend": ("ladder", "full", "~ 1", {"trend": True}, 1e-10),
    "ladder, robust": ("ladder", "full", "~ 1", {"robust": True}, 1e-6),
    "missing": ("missing", "full", "~ 1", {}, 1e-10),
    "missing, trend": ("missing", "full", "~ 1", {"trend": True}, 1e-10),
    "sex": ("sex", "pDC", "~ sex", {}, 1e-10),
}


@functools.cache
def job_result(job: str, **extra: Any) -> Any:
    case, group, formula, options, _ = JOBS[job]
    scores = scores_of(limma_reference()["scores"][case], group)
    moderation = dict(options)
    if moderation.get("trend") is True:
        # The set sizes as the trend's covariate, as R's reference took them.
        moderation["trend"] = scores.n_genes
    return dose_response(
        scores,
        design_of(scores, formula),
        dose=AXIS,
        **({"tests": TESTS, "spline_df": 2} | moderation | extra),
    )


# Against limma.


@pytest.mark.parametrize("job", list(JOBS))
def test_tests_of_scores_match_limma(job: str) -> None:
    result = job_result(job)
    _, group, _, _, tolerance = JOBS[job]
    theirs = limma_reference()["jobs"][job]
    entry = result.provenance["groups"][0]
    assert entry["fit"]["samples"] == theirs["samples"]
    assert entry["doses"] == numbers(theirs["doses"]).tolist()
    assert entry["samples"] == numbers(theirs["n"]).tolist()
    prior = numbers(theirs["df_prior"])
    assert_scaled(result.summary["df_prior"][0], np.median(prior), tolerance)
    for test in TESTS:
        table = result.table(group, test)
        values = theirs["tests"][test]
        if test == "linear_trend":
            for ours, key in (
                ("estimate", "estimate"),
                ("standard_error", "se"),
                ("average", "average"),
                ("t", "t"),
                ("log_odds", "B"),
            ):
                assert_scaled(table[ours], numbers(values[key]), tolerance)
        else:
            assert entry["f_test_df1"][test] == values["df1"]
            assert_scaled(table["f_statistic"], numbers(values["F"]), tolerance)
            df2 = np.broadcast_to(numbers(values["df2"]), len(table))
            assert_scaled(table["df"], df2, tolerance)
        assert_p_values(table["p_value"], numbers(values["p"]), tolerance)
        # One cell type: the adjustment across cell types is the one within.
        for column in ("adjusted_p_value", "adjusted_p_value_across_groups"):
            assert_p_values(table[column], numbers(values["adj"]), tolerance)
    each = result.table(group, "each_dose")
    k = len(theirs["doses"]) - 1

    def by_dose(column: str) -> np.ndarray:
        return np.asarray(each[column], dtype=float).reshape(k, -1).T

    for ours, key in (
        ("estimate", "estimate"),
        ("standard_error", "se"),
        ("t", "t"),
        ("log_odds", "B"),
    ):
        assert_scaled(by_dose(ours), numbers(theirs["each_dose"][key]), tolerance)
    assert_scaled(
        by_dose("average")[:, 0], numbers(theirs["each_dose"]["average"]), tolerance
    )
    assert_p_values(by_dose("p_value"), numbers(theirs["each_dose"]["p"]), tolerance)
    assert_p_values(
        by_dose("adjusted_p_value"), numbers(theirs["each_dose"]["adj"]), tolerance
    )


def test_scores_that_cannot_estimate_every_dose_group_have_no_tests() -> None:
    # In the fixture, "flat" is missing in the whole vehicle group and "random_05" in
    # every sample: R's lmFit drops a column of the first and fits nothing of the
    # second, and their tests are NaN, while their residual variances (the first's)
    # enter the prior, which the other scores' statistics check against limma.
    result = job_result("missing")
    names = limma_reference()["scores"]["missing"]["names"]
    unestimable = [
        names[i] for i in limma_reference()["jobs"]["missing"]["unestimable"]
    ]
    assert unestimable == ["flat", "random_05"]
    for test in (*TESTS, "each_dose"):
        table = result.table("full", test)
        rows = np.isin(np.asarray(table["gene_set"].tolist()), unestimable)
        assert np.isnan(np.asarray(table["p_value"], dtype=float)[rows]).all()
        assert not np.isnan(np.asarray(table["p_value"], dtype=float)[~rows]).any()


def test_scores_are_paired_with_the_design_by_sample() -> None:
    # The design's rows in the reverse order of the scores' profiles: each score keeps
    # its own sample's dose and sex, so every table is the one of the same order.
    scores = scores_of(limma_reference()["scores"]["sex"], "pDC")
    reversed_rows = {
        name: scores.profiles[name].tolist()[::-1]
        for name in ("sample_id", "dose", "sex")
    }
    result = dose_response(
        scores, Design(reversed_rows, "~ sex"), dose=AXIS, tests=TESTS, spline_df=2
    )
    same_order = job_result("sex")
    for test in (*TESTS, "each_dose"):
        ours, theirs = result.table(name=test), same_order.table(name=test)
        assert ours.columns == theirs.columns
        for column in ours.columns:
            a, b = np.asarray(ours[column]), np.asarray(theirs[column])
            if a.dtype.kind == "f":
                assert_allclose(a, b, rtol=1e-9, atol=1e-12)
            else:
                assert_array_equal(a, b)


def test_curves_of_scores_match_topTable() -> None:
    result = job_result("ladder")
    theirs = limma_reference()["jobs"]["ladder"]["curve"]
    at = np.asarray(theirs["at"])
    curve = result.curve("full", at=at)
    assert curve.columns == (
        "gene_set",
        "dose",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
    )
    for ours, key in (
        ("estimate", "estimate"),
        ("ci_lower", "lower"),
        ("ci_upper", "upper"),
    ):
        assert_scaled(
            np.asarray(curve[ours]).reshape(-1, at.size), numbers(theirs[key])
        )


@pytest.mark.parametrize("job", ["ladder", "ladder, trend", "missing"])
def test_one_sided_linear_trend_matches_pt(job: str) -> None:
    values = limma_reference()["jobs"][job]["tests"]["linear_trend"]
    for direction in ("up", "down"):
        table = job_result(job).table(name="linear_trend")
        one = job_result(job, direction=direction, tests=("linear_trend",))
        found = one.table(name="linear_trend")
        assert_p_values(found["p_value"], numbers(values[direction]))
        assert_p_values(found["adjusted_p_value"], numbers(values[f"{direction}_adj"]))
        # The statistics themselves stay those of the two-sided test.
        for column in ("estimate", "standard_error", "t", "log_odds"):
            assert_array_equal(found[column], table[column])


# Against DoseFinding, multcomp and draws of the multivariate t.


def alone(name: str) -> PathwayScores:
    return scores_of(dosefinding_reference()["scores"][name], "bulk")


@pytest.mark.parametrize("name", ["trial falling", "trial rising", "ladder falling"])
@pytest.mark.parametrize("direction", [None, "up", "down"])
def test_one_score_s_contrast_tests_match_dosefinding_and_multcomp(
    name: str, direction: str | None
) -> None:
    scores = alone(name)
    theirs = dosefinding_reference()["reference"][name]
    result = dose_response(
        scores,
        design_of(scores),
        dose=AXIS,
        tests=("dunnett", "williams", "mcp"),
        direction=direction,
        rng=dosefinding_reference()["seed"],
    )
    entry = result.provenance["groups"][0]
    n = numbers(theirs["n"])
    df = float(theirs["df"])
    # The MCP step's contrasts: DoseFinding's on the treated doses, scaled to unit
    # length, and their mirror images for the decreasing models.
    weights = entry["contrast_tests"]["mcp"]["contrasts"]
    ours = np.array(list(weights.values()))
    for side, sign in (("up", 1.0), ("down", -1.0)):
        models = theirs["mcp"][DOSEFINDING[side]]
        matrix = numbers(models["contrasts"]).reshape(n.size, -1)[1:].T
        unit = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        assert_scaled(sign * ours, unit)
    mcp = result.table(name="mcp")
    names = list(weights)
    if direction is None:
        reference = theirs["mcp"]["two_sided"]
        t_all, sign = numbers(theirs["mcp"]["increasing"]["t"]), 1.0
        # MCTtest reports |t| when two-sided.
        assert_scaled(
            numbers(theirs["mcp"]["increasing"]["t_two_sided"]), np.abs(t_all)
        )
    else:
        reference = theirs["mcp"][DOSEFINDING[direction]]
        t_all = numbers(reference["t"])
        # DoseFinding's decreasing models give t of the opposite sign.
        sign = 1.0 if direction == "up" else -1.0
    chosen = names.index(mcp["contrast"][0])
    assert chosen == reference["chosen"]
    assert_scaled(sign * mcp["t"][0], t_all[chosen])
    assert mcp["df"][0] == df
    within_errors(mcp, reference)
    side = {None: "two.sided", "up": "greater", "down": "less"}[direction]
    for test in ("dunnett", "williams"):
        table = result.table(name=test)
        values = theirs[test][side]
        order = list(entry["contrast_tests"][test]["contrasts"])
        assert order.index(table["contrast"][0]) == values["chosen"]
        assert_scaled(table["t"][0], numbers(values["t"])[values["chosen"]])
        within_errors(table, values)


def within_errors(table: Table, reference: dict[str, Any]) -> None:
    """A max-t p-value within CALIBRATED combined standard errors of the draws'."""
    ours, error = float(table["p_value"][0]), float(table["p_value_error"][0])
    combined = math.hypot(error, float(reference["se"]))
    assert abs(ours - float(reference["p"])) <= CALIBRATED * combined


def test_one_treated_dose_dunnett_is_each_dose_s_t_test() -> None:
    # With one treated dose, Dunnett's test has one contrast, the dose's own t-test,
    # whose one-sided p-value is half the two-sided one in the stated direction.
    rng = np.random.default_rng(3)
    doses = np.repeat([0.0, 1.0], 5)
    values = rng.normal(size=(10, 3)) - 0.8 * doses[:, None] * np.array([1, 0, -1])
    scores = array_scores(values, doses)
    design = design_of(scores)
    two = dose_response(scores, design, dose=AXIS, tests="dunnett")
    each = two.table(name="each_dose")
    for direction, toward in (("up", 1.0), ("down", -1.0)):
        one = dose_response(
            scores, design, dose=AXIS, tests="dunnett", direction=direction
        ).table(name="dunnett")
        t = np.asarray(each["t"], dtype=float)
        p = np.asarray(each["p_value"], dtype=float)
        expected = np.where(toward * t >= 0, p / 2, 1 - p / 2)
        assert_allclose(one["p_value"], expected, rtol=1e-12)
        assert_array_equal(one["p_value_error"], 0.0)


# What R cannot check.


def array_scores(
    values: np.ndarray, doses: np.ndarray, names: list[str] | None = None
) -> PathwayScores:
    """Pathway scores of an array of values, one sample per row, at the doses
    given."""
    ids = [f"s{i}" for i in range(values.shape[0])]
    names = names or [f"set_{j}" for j in range(values.shape[1])]
    return PathwayScores(
        values,
        Table({"profile_id": ids, "sample_id": ids, "dose": doses}),
        GeneSets({name: [f"{name}_g{i}" for i in range(10)] for name in names}),
        "gsva",
        MappingProxyType({}),
    )


def test_one_score_is_an_ordinary_linear_model() -> None:
    rng = np.random.default_rng(11)
    levels = np.array([0.0, 0.1, 1.0, 10.0])
    n = np.array([4, 3, 4, 5])
    doses = np.repeat(levels, n)
    values = rng.normal(size=doses.size) + 0.4 * np.log10(np.where(doses > 0, doses, 1))
    scores = array_scores(values[:, None], doses)
    result = dose_response(
        scores, design_of(scores), dose=AXIS, tests=("linear_trend", "dose_groups")
    )
    # One score is not moderated: its tests are those of its own linear model.
    assert result.summary["df_prior"][0] == 0.0
    means = np.array([values[doses == d].mean() for d in levels])
    residual = values - means[np.searchsorted(levels, doses)]
    df = doses.size - levels.size
    s2 = residual @ residual / df
    # The one-way analysis of variance of the dose groups.
    between = (n * (means - values.mean()) ** 2).sum() / (levels.size - 1)
    groups = result.table(name="dose_groups")
    assert_allclose(groups["f_statistic"], between / s2, rtol=1e-12)
    assert_allclose(
        groups["p_value"], stats.f.sf(between / s2, levels.size - 1, df), rtol=1e-10
    )
    # Each dose's t-test against the vehicle, on the pooled variance.
    each = result.table(name="each_dose")
    t = (means[1:] - means[0]) / np.sqrt(s2 * (1 / n[1:] + 1 / n[0]))
    assert_allclose(each["t"], t, rtol=1e-12)
    assert_allclose(each["p_value"], 2 * stats.t.sf(np.abs(t), df), rtol=1e-10)
    # The linear trend through the treated doses, by least squares on the samples:
    # an intercept, a jump from the vehicle and log10 dose from the lowest dose.
    treated = (doses > 0).astype(float)
    position = np.where(doses > 0, np.log10(np.where(doses > 0, doses, 1)) + 1, 0)
    x = np.column_stack([np.ones(doses.size), treated, position])
    slope = np.linalg.lstsq(x, values, rcond=None)[0][2]
    covariance = np.linalg.inv(x.T @ x)[2, 2]
    trend = result.table(name="linear_trend")
    assert_allclose(trend["estimate"], slope, rtol=1e-12)
    assert_allclose(trend["standard_error"], np.sqrt(s2 * covariance), rtol=1e-12)


def test_scores_follow_the_route_of_limma_trend() -> None:
    # limma-trend's values of a cube's cell type, given as one score per gene with
    # the trend on their mean, are the cube's own fit.
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0, 30.0], replicates=3, cell_types=1, n_genes=300, rng=21
    )
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS, analysis=limma_trend)
    fit = result.fit("cell_type_1")
    samples = fit.samples
    genes = fit.features["feature_id"].tolist()
    scores = PathwayScores(
        np.asarray(fit.log2_cpm),
        Table(
            {
                "profile_id": samples["sample_id"],
                "sample_id": samples["sample_id"],
                "dose": samples["dose"],
            }
        ),
        GeneSets({gene: [gene] for gene in genes}),
        "gsva",
        MappingProxyType({"group": "cell_type_1"}),
    )
    found = dose_response(scores, design, dose=AXIS, trend=True)
    for test in result.tests:
        ours, theirs = found.table(name=test), result.table(name=test)
        assert ours["gene_set"].tolist() == genes
        for column in ("f_statistic", "t", "p_value", "adjusted_p_value"):
            if column in theirs:
                assert_allclose(ours[column], theirs[column], rtol=1e-12)
    curve_ours = found.curve(genes=genes[:3])
    curve_theirs = result.curve(genes=genes[:3])
    assert_allclose(
        curve_ours["estimate"], curve_theirs["log2_fold_change"], rtol=1e-12
    )
    assert_allclose(curve_ours["ci_upper"], curve_theirs["ci_upper"], rtol=1e-12)


def test_activities_are_tested_as_pathway_scores() -> None:
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0], replicates=3, cell_types=["pDC"], n_genes=200, rng=4
    )
    genes = cube.features["feature_id"].tolist()
    network = Network(
        {
            "A": {gene: 1.0 for gene in genes[:20]},
            "B": {gene: (-1.0) ** j for j, gene in enumerate(genes[20:50])},
        }
    )
    activities = ulm(cube, network, group="pDC")
    scores = PathwayScores(
        activities.scores,
        activities.profiles,
        GeneSets({source: [source] for source in activities.sources}),
        "ulm",
        MappingProxyType({"group": "pDC"}),
    )
    design = Design(cube.samples, "~ 1")
    tests = ("spline", "linear_trend", "mcp")
    by_activity = dose_response(activities, design, dose=AXIS, tests=tests, rng=1)
    by_scores = dose_response(scores, design, dose=AXIS, tests=tests, rng=1)
    for test in (*tests, "each_dose"):
        one, other = by_activity.table(name=test), by_scores.table(name=test)
        assert one.columns[0] == "source" and other.columns[0] == "gene_set"
        assert one["source"].tolist() == other["gene_set"].tolist()
        for column in one.columns[1:]:
            assert_array_equal(one[column], other[column])
    assert by_activity.provenance["scores"]["type"] == "Activities"
    assert (
        by_activity.provenance["cube"]["fingerprint"] == cube.digests()["fingerprint"]
    )


# One-sided tests.


def test_one_sided_p_values_split_the_two_sided_one() -> None:
    result = job_result("ladder")
    two = result.table(name="linear_trend")
    up = job_result("ladder", direction="up", tests=("linear_trend",))
    down = job_result("ladder", direction="down", tests=("linear_trend",))
    p_up = np.asarray(up.table(name="linear_trend")["p_value"])
    p_down = np.asarray(down.table(name="linear_trend")["p_value"])
    assert_allclose(p_up + p_down, 1.0, rtol=1e-15)
    t, p = np.asarray(two["t"]), np.asarray(two["p_value"])
    assert_allclose(np.where(t > 0, p_up, p_down), p / 2, rtol=1e-15)
    assert up.settings["direction"] == "up"
    assert "one-sided, up" in repr(up)


def test_one_sided_p_values_at_zero_and_missing() -> None:
    # P(T >= t) of a two-sided p: half of it toward the direction, one minus half of
    # it away from it, one half at t = 0, and NaN where the test is missing.
    table = Table(
        {
            "t": [0.0, 2.0, -2.0, np.nan],
            "p_value": [1.0, 0.1, 0.1, np.nan],
            "adjusted_p_value": [1.0, 0.2, 0.2, np.nan],
        }
    )
    up = _one_sided(table, "up")
    down = _one_sided(table, "down")
    assert_array_equal(up["p_value"], [0.5, 0.05, 0.95, np.nan])
    assert_array_equal(down["p_value"], [0.5, 0.95, 0.05, np.nan])
    assert_allclose(up["adjusted_p_value"], adjust_p_values(up["p_value"], "bh"))
    assert_array_equal(up["t"], table["t"])


@pytest.mark.parametrize("analysis", ["voom", "dream"])
def test_direction_on_a_cube(analysis: str) -> None:
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0],
        replicates=3,
        cell_types=2,
        n_genes=150,
        donor_sd=0.2 if analysis == "dream" else None,
        rng=5,
    )
    random = "(1 | donor)" if analysis == "dream" else None
    design = Design(cube.samples, "~ 1", random=random)
    options: dict[str, Any] = {"analysis": dream_voom} if analysis == "dream" else {}
    two = dose_response(cube, design, dose=AXIS, tests="linear_trend", **options)
    one = dose_response(
        cube,
        design,
        dose=AXIS,
        tests="linear_trend",
        direction="down",
        **options,
    )
    stacked = []
    for group in one.groups:
        theirs, ours = (
            two.table(group, "linear_trend"),
            one.table(group, "linear_trend"),
        )
        t, p = np.asarray(theirs["t"]), np.asarray(theirs["p_value"])
        expected = np.where(t <= 0, p / 2, 1 - p / 2)
        assert_allclose(ours["p_value"], expected, rtol=1e-15)
        assert_allclose(ours["adjusted_p_value"], adjust_p_values(expected, "bh"))
        stacked.append(expected)
    across = adjust_p_values(np.concatenate(stacked), "bh")
    found = np.concatenate(
        [
            one.table(g, "linear_trend")["adjusted_p_value_across_groups"]
            for g in one.groups
        ]
    )
    assert_allclose(found, across)


def test_direction_is_refused_where_a_test_has_no_side() -> None:
    scores = alone("trial falling")
    design = design_of(scores)
    with pytest.raises(ValueError, match="tests=None runs the default tests"):
        dose_response(scores, design, dose=AXIS, direction="down")
    for test in ("spline", "nonlinearity", "dose_groups"):
        with pytest.raises(ValueError, match=f"'{test}' is an F-test"):
            dose_response(
                scores,
                design,
                dose=AXIS,
                tests=("mcp", test),
                direction="down",
                spline_df=1 if test == "spline" else None,
            )
    # DoseFinding's words name its models' directions, not CytePool's.
    for word in ("decreasing", "Down", "fall"):
        with pytest.raises(ValueError, match=r"direction must be 'up' \(DoseFinding"):
            dose_response(scores, design, dose=AXIS, tests="mcp", direction=word)
    with pytest.raises(TypeError, match="direction must be 'up' or 'down', or None"):
        dose_response(scores, design, dose=AXIS, tests="mcp", direction=-1)
    # A gate of a test without a sign.
    for gate, what in (
        ({"test": "spline"}, "a change either way"),
        ({"test": "dose_groups"}, "a change either way"),
        ({"fold_change": 1.5}, "each dose against a fold change either way"),
    ):
        with pytest.raises(ValueError, match=f"makes the gate one-sided.*{what}"):
            dose_response(
                scores,
                design,
                dose=AXIS,
                tests="mcp",
                direction="down",
                gate=gate,
            )


def test_a_direction_makes_the_gate_one_sided() -> None:
    # The gate is the first multiple contrast test that tests names, or the MCP
    # step, which then runs too; its p-values are that test's, in the direction
    # given.
    for tests, gate in (
        (("linear_trend", "mcp", "williams"), "williams"),
        (("linear_trend",), "mcp"),
    ):
        result = job_result("ladder", direction="down", tests=tests)
        assert result.gate == {"test": gate, "fdr": 0.05, "fold_change": None}
        assert set(result.tests) == {*tests, gate}
        assert_array_equal(
            result.table(name="gate")["p_value"], result.table(name=gate)["p_value"]
        )
        assert f"gate: {gate}" in repr(result)
    # The Williams-type gate for a fall passes the falling set and not the rising
    # one, and the gate for a rise the reverse.
    down = job_result("ladder", direction="down", tests=("williams",))
    up = job_result("ladder", direction="up", tests=("williams",))
    names = down.table(name="gate")["gene_set"].tolist()
    p_down = np.asarray(down.table(name="gate")["p_value"])
    p_up = np.asarray(up.table(name="gate")["p_value"])
    falling, rising = names.index("falling"), names.index("rising")
    assert p_down[falling] < 1e-3 and p_up[falling] > 0.99
    assert p_up[rising] < 1e-3 and p_down[rising] > 0.99


def test_direction_leaves_estimates_two_sided() -> None:
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0], replicates=3, cell_types=1, n_genes=200, rng=6
    )
    design = Design(cube.samples, "~ 1")
    tests = ("linear_trend", "dunnett", "mcp")
    two = dose_response(cube, design, dose=AXIS, tests=tests, rng=2)
    one = dose_response(cube, design, dose=AXIS, tests=tests, direction="up", rng=2)
    each, each_one = two.table(name="each_dose"), one.table(name="each_dose")
    # Each dose's tests; which genes pass the gate differs with the gate.
    for column in [name for name in each.columns if name != "passes_gate"]:
        assert_array_equal(each[column], each_one[column])
    assert_array_equal(two.curve()["ci_lower"], one.curve()["ci_lower"])
    # The gate: the first multiple contrast test named, one-sided, where the
    # two-sided result's is the spline.
    assert (two.gate["test"], one.gate["test"]) == ("spline", "dunnett")
    gate = one.table(name="gate")
    assert_array_equal(gate["p_value"], one.table(name="dunnett")["p_value"])
    # Lowest effective doses: the one-sided gate, and each dose two-sided.
    lowest = one.lowest_effective_doses()
    k = len(one.provenance["groups"][0]["doses"]) - 1
    p = np.asarray(each["p_value"]).reshape(k, -1).T
    adjusted = stage_wise_adjusted_p_values(np.asarray(gate["p_value"]), p, alpha=0.05)
    called = adjusted <= 0.05
    assert_array_equal(lowest["n_doses_called"], called.sum(axis=1))
    assert called.sum() > 0


# Refusals.


def test_scores_refuse_what_applies_to_a_cube() -> None:
    scores = alone("trial falling")
    design = design_of(scores)
    with pytest.raises(ValueError, match="scores are fitted as limma fits them"):
        dose_response(scores, design, dose=AXIS, analysis=limma_trend)
    with pytest.raises(ValueError, match="groups chooses cell types of a cube"):
        dose_response(scores, design, dose=AXIS, groups="bulk")
    for name, value in (("min_cells", 3), ("min_counts", 10.0)):
        with pytest.raises(ValueError, match=f"{name} chooses the samples"):
            dose_response(scores, design, dose=AXIS, **{name: value})
    with pytest.raises(TypeError, match="no option 'prior_weights'.*trend and robust"):
        dose_response(scores, design, dose=AXIS, prior_weights="n_cells")
    with pytest.raises(ValueError, match="one covariate value per feature"):
        dose_response(scores, design, dose=AXIS, trend=np.ones(3))
    samples = scores.profiles.select(["sample_id", "dose"]).with_columns(
        {"donor": [f"d{i % 4}" for i in range(len(scores.profiles))]}
    )
    with pytest.raises(ValueError, match="dose_response on scores fits fixed effects"):
        dose_response(scores, Design(samples, "~ 1", random="(1 | donor)"), dose=AXIS)
    other = Design(Table({"sample_id": ["x", "y"], "dose": [0.0, 1.0]}), "~ 1")
    with pytest.raises(ValueError, match="none of the design's samples"):
        dose_response(scores, other, dose=AXIS)
    contrast_activities = Activities(
        scores.scores[:2].T,
        scores.scores[:2].T,
        np.full((1, 2), 0.5),
        np.full((1, 2), 0.5),
        np.ones((1, 2), dtype=np.int64),
        Table({"profile_id": ["a"], "group_id": ["g"], "contrast": ["c"]}),
        Network({"s1": {"x": 1.0}, "s2": {"y": 1.0}}),
        "ulm",
        MappingProxyType({}),
    )
    with pytest.raises(ValueError, match="activities are of the statistics"):
        dose_response(contrast_activities, design, dose=AXIS)
    profiles = scores.profiles.select(["profile_id", "sample_id", "dose"])
    twice = PathwayScores(
        np.vstack([scores.scores, scores.scores[:1]]),
        Table(
            {
                name: [*profiles[name].tolist(), profiles[name].tolist()[0]]
                for name in profiles.columns
            }
        ),
        scores.gene_sets,
        "gsva",
        MappingProxyType({}),
    )
    with pytest.raises(ValueError, match="name the sample.*more than once"):
        dose_response(twice, design, dose=AXIS)
    with pytest.raises(ValueError, match="scores have no fold change"):
        dose_response(scores, design, dose=AXIS, gate={"fold_change": 1.5})
    with pytest.raises(ValueError, match="this result tests pathway scores"):
        permutation_check(job_result("ladder"))
    result = job_result("ladder")
    for genes in (5, {"falling": 1}):
        with pytest.raises(TypeError, match="genes must be a gene set's name"):
            result.curve("full", genes=genes)
    with pytest.raises(TypeError, match="genes must be names of gene sets"):
        result.curve("full", genes=[1])


# Curves of named models.


def test_curves_of_scores() -> None:
    scores = scores_of(limma_reference()["scores"]["missing"], "full")
    design = design_of(scores)
    names = list(scores.gene_sets)
    for variance in ("constant", "weighted", "moderated"):
        result = dose_response(
            scores, design, dose=AXIS, models="ll4", variance=variance
        )
        table = result.table(name="curves")
        assert table.columns[:3] == ("gene_set", "passes_gate", "model")
        # Each score's call in the table "gate", matched by its gene set.
        gate = result.table(name="gate")
        call = dict(zip(gate["gene_set"].tolist(), gate["passes_gate"].tolist()))
        assert table["passes_gate"].tolist() == [
            call[name] for name in table["gene_set"].tolist()
        ]
        assert 0 < np.count_nonzero(table["passes_gate"]) < len(table)
        record = result.provenance["groups"][0]["curves"]
        left_out = set(record["left_out"])
        # Under constant variance, scores with a missing value have no curve; in the
        # plug-in modes, only those whose dose groups cannot be estimated.
        expected = (
            {"rising", "flat", "random_05"}
            if variance == "constant"
            else {
                "flat",
                "random_05",
            }
        )
        assert left_out == expected
        assert table["gene_set"].tolist() == [n for n in names if n not in expected]
        curve = result.curve(genes="falling", model="ll4")
        assert curve.columns == ("gene_set", "model", "dose", "estimate")
    # Constant variance fits each score's samples, as fit_dose_curves does.
    from cytepool.stats.dose_curves import fit_dose_curves

    complete = scores_of(limma_reference()["scores"]["ladder"], "full")
    result = dose_response(complete, design_of(complete), dose=AXIS, models="ll4")
    direct = fit_dose_curves(
        np.asarray(complete.profiles["dose"]), complete.scores, models="ll4"
    )
    assert_allclose(result.curve_fits().parameters, direct.parameters, rtol=1e-12)


def test_scores_that_each_miss_a_value_have_no_curves() -> None:
    # Each score misses a sample the other has: under constant variance neither has
    # complete data, and the cell type has no curves, with the reason recorded.
    record = json.loads(json.dumps(limma_reference()["scores"]["ladder"]))
    values = numbers(record["scores"])[:, :2]
    values[0, 0] = values[1, 1] = np.nan
    names = record["names"][:2]
    record |= {"scores": values.tolist(), "names": names, "sizes": record["sizes"][:2]}
    scores = scores_of(record, "full")
    result = dose_response(scores, design_of(scores), dose=AXIS, models="ll4")
    entry = result.provenance["groups"][0]["curves"]
    assert entry["reason"] == "no gene set has complete data"
    assert sorted(entry["left_out"]) == sorted(names)
    with pytest.raises(KeyError, match="no gene set has complete data"):
        result.table(name="curves")


@pytest.mark.parametrize("variance", ["constant", "weighted"])
@pytest.mark.parametrize(
    ("model_set", "model"), [("epa", "linear"), ("epa", "hill"), ("ll4", "ll4")]
)
def test_curves_of_one_score_match_fitmod(
    variance: str, model_set: str, model: str
) -> None:
    # DoseFinding's fitMod of the samples (type "normal") or of the dose-group means
    # with the residual variance times diag(1 / n) (type "general"); its linear model
    # and sigEmax, e0 + eMax d^h / (ED50^h + d^h), the mean of the Hill and
    # log-logistic curves, with their bounds.
    scores = alone("ladder falling")
    theirs = dosefinding_reference()["curves"]["ladder falling"][variance][model]
    result = dose_response(
        scores, design_of(scores), dose=AXIS, models=model_set, variance=variance
    )
    fits = result.curve_fits()
    m = fits.models.index(model)
    log_likelihood = float(fits.log_likelihood[0, m])
    if variance == "constant":
        n = len(scores.profiles)
        objective = n * math.exp(-2 * log_likelihood / n - 1) / (2 * math.pi)
    else:
        covariance = fits.covariance_scale[0] * np.asarray(fits.covariance[0])
        objective = (
            -2 * log_likelihood
            - fits.doses.size * math.log(2 * math.pi)
            - np.linalg.slogdet(covariance)[1]
        )
    exact = model == "linear"
    assert_allclose(objective, theirs["objective"], rtol=1e-10 if exact else 1e-6)
    p = dict(zip(fits.parameter_names, fits.parameters[0, m].tolist(), strict=True))
    if model == "linear":
        ours = [p["g"], p["b1"]]
    elif model == "hill":
        ours = [p["g"], p["v"], p["k"], p["n"]]
    else:
        ours = [p["d"], p["c"] - p["d"], p["e"], p["b"]]
    assert_allclose(ours, numbers(theirs["coef"]), rtol=1e-10 if exact else 1e-4)


def test_descriptive_scores_get_curves_without_lack_of_fit() -> None:
    rng = np.random.default_rng(9)
    doses = np.array([0.0, 0.0, 0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    values = rng.normal(size=(9, 2)) + np.log10(np.where(doses > 0, doses, 0.01))[
        :, None
    ] * np.array([0.5, -0.5])
    scores = array_scores(values, doses)
    result = dose_response(scores, design_of(scores), dose=AXIS, models="ll4")
    assert result.design_report.status() == "descriptive"
    assert not result.summary["tested"][0]
    fits = result.curve_fits()
    assert fits.variance == "constant"
    assert np.isnan(fits.lack_of_fit_p_value).all()


# Records.


def test_the_result_of_scores_reads_back() -> None:
    result = job_result("ladder", direction="down", tests=("linear_trend", "mcp"))
    assert result.method == "limma"
    assert result.settings["analysis"] == "limma"
    assert result.settings["direction"] == "down"
    record = result.provenance
    assert record["scores"]["type"] == "PathwayScores"
    assert record["scores"]["names"] == limma_reference()["scores"]["ladder"]["names"]
    assert record["cube"] is None and record["layer"] is None
    assert record["analysis_settings"] == {"trend": False, "robust": False}
    text = repr(result)
    assert "DoseResponseResult (limma)" in text and "24 gene sets" in text
    with pytest.raises(ValueError, match="this result tests scores"):
        result.fit()
    again = pickle.loads(pickle.dumps(result))
    for key, table in result.tables.items():
        for column in table.columns:
            assert_array_equal(again.tables[key][column], table[column])
    for column in ("estimate", "ci_lower", "ci_upper"):
        assert_array_equal(again.curve()[column], result.curve()[column])
    # The gate of scores: the MCP step, the first multiple contrast test named, in
    # the direction given, over the gene sets.
    assert result.gate["test"] == "mcp"
    gate, mcp = result.table(name="gate"), result.table(name="mcp")
    assert gate.columns == ("gene_set", "p_value", "adjusted_p_value", "passes_gate")
    assert_array_equal(gate["p_value"], mcp["p_value"])
    assert_array_equal(gate["passes_gate"], np.asarray(mcp["adjusted_p_value"]) <= 0.05)
    lowest = result.lowest_effective_doses()
    assert lowest.columns == (
        "gene_set",
        "passes_gate",
        "lowest_effective_dose",
        "estimate",
        "stage_wise_adjusted_p_value",
        "n_doses_called",
    )
    assert mcp.columns == (
        "gene_set",
        "average",
        "contrast",
        "t",
        "df",
        "p_value",
        "p_value_error",
        "adjusted_p_value",
        "adjusted_p_value_across_groups",
    )


def test_scores_of_an_array_form_one_group() -> None:
    rng = np.random.default_rng(1)
    doses = np.repeat([0.0, 0.1, 1.0, 10.0], 3)
    genes = [f"g{j}" for j in range(60)]
    values = rng.normal(size=(12, 60))
    values[:, :10] += np.log10(np.where(doses > 0, doses, 0.01))[:, None]
    ids = [f"s{i}" for i in range(12)]
    scores = gsva(
        values,
        {"up": genes[:10], "other": genes[30:50]},
        feature_ids=genes,
        sample_ids=ids,
    )
    design = Design({"sample_id": ids[::-1], "dose": doses[::-1]}, "~ 1")
    result = dose_response(scores, design, dose=AXIS, tests="linear_trend")
    assert result.groups == ("bulk",)
    assert result.provenance["groups"][0]["fit"]["samples"] == ids[::-1]
    assert result.table(name="linear_trend")["t"][0] > 5


CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/dose_response.md"
)


def test_the_chapter_s_signature_example_runs() -> None:
    """The code of the chapter's section on signatures, on the series of its section
    on the tests: its printout and the numbers its text states, against the
    simulator's truth."""
    import contextlib
    import io
    import re

    import cytepool

    text = CHAPTER.read_text()

    def section(label: str) -> str:
        return text.split(f"({label})=")[1].split("\n(")[0]

    setup = re.findall(r"```python\n(.*?)```", section("dose-response-tests"), re.S)
    signatures_text = section("dose-response-signatures")
    (block,) = re.findall(r"```python\n(.*?)```", signatures_text, re.S)
    (printout,) = re.findall(r"```text\n(.*?)```", signatures_text, re.S)
    namespace: dict[str, Any] = {"cp": cytepool}
    exec(setup[0].replace("print(result)\n", ""), namespace)  # noqa: S102
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(block, namespace)  # noqa: S102
    assert output.getvalue() == printout
    scores, result = namespace["scores"], namespace["signatures"]
    # The falling set holds the truth's 30 falling Hill genes, 15 of which pass the
    # expression filter, and its fall is the one called.
    truth = namespace["truth"].table("hepatocyte")
    falling = (np.asarray(truth["shape"].tolist()) == "hill") & (
        np.asarray(truth["max_log2_change"]) < 0
    )
    assert int(falling.sum()) == 30
    assert scores.n_genes.tolist()[0] == 15
    assert "the 15 that pass the expression filter" in signatures_text
    assert result.summary["df_prior"][0] == 0.0
    emax = result.candidates.emax[1]
    assert f"reaches half its change at {emax:.1f} ug/kg" in signatures_text
    assert result.table(name="mcp")["contrast"].tolist()[0] == "emax_2"
