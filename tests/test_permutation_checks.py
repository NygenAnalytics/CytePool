"""Tests for permutation_check and PermutationCheck.

Reference values in ``tests/data/dose_gate/limma.json`` come from R, written by
``validation/dose_gate.py``: R reads the sample table itself, builds the blocks of
exchangeable units, counts and enumerates the assignments of the dose labels, and
refits limma 3.62.1 on each of the 90 assignments of a series of the vehicle and two
doses in duplicate. Other tests check what R does not: the assignments against brute
force over itertools on blocks that the tests build from the sample table, the
uniformity of random runs, runs made again from their recorded assignments by
dose_response called in the test, counts put into the summary by hand, invariants and
refusals.
"""

import contextlib
import functools
import io
import itertools
import json
import math
import pickle
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import cytepool
from cytepool import (
    Cube,
    Design,
    DoseAxis,
    DoseResponseResult,
    PermutationCheck,
    dose_response,
    dream_voom,
    limma_trend,
    permutation_check,
    simulate_dose_series,
)
from cytepool.permutation_checks import _rerun_tests, _statistics, _units

DATA = Path(__file__).parent / "data" / "dose_gate"
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")


# The fixture of R's results, and the series it was computed on.


@functools.cache
def reference() -> dict[str, Any]:
    found: dict[str, Any] = json.loads((DATA / "limma.json").read_text())
    return found


@functools.cache
def fixture(name: str) -> tuple[Cube, Design]:
    """A series R analysed, as a cube, and its design."""
    case = reference()["cases"][name]
    cube = Cube(
        {"counts": np.array(case["counts"], dtype=np.int64)},
        samples=case["samples"],
        groups={"group_id": case["groups"]},
        features={"feature_id": case["features"]},
        n_cells=np.array(case["n_cells"], dtype=np.int64),
    )
    return cube, Design(case["design"], case["formula"])


def series(doses: list[float], seed: int = 1, **options: Any) -> Cube:
    """A small simulated series of few genes, for runs that only count."""
    defaults: dict[str, Any] = {"cell_types": 1, "cells": 300.0, "n_genes": 60}
    cube, _ = simulate_dose_series(doses, rng=seed, **(defaults | options))
    return cube


def labels_of(check: PermutationCheck, group: str) -> list[tuple[int, ...]]:
    """The true assignment and each run's, as recorded."""
    entry = next(e for e in check.provenance["groups"] if e["group_id"] == group)
    return [tuple(entry["labels"])] + [tuple(a) for a in entry["assignments"]]


# Against R.


@pytest.mark.parametrize("gate", [{"test": "dose_groups"}, {"fold_change": 1.5}])
def test_each_run_matches_r_on_every_assignment(gate: dict[str, Any]) -> None:
    # The vehicle and two doses in duplicate: R refits limma on each of the 90
    # assignments it enumerates, and counts the genes the gate passes.
    cube, design = fixture("duplicate")
    theirs = reference()["cases"]["duplicate"]["blocks"]["cell_type_1"]
    key = "dose_groups" if "test" in gate else "each_dose"
    counts = {
        tuple(a): int(n)
        for a, n in zip(theirs["assignments"], theirs["counts"][key], strict=True)
    }
    result = dose_response(cube, design, dose=AXIS, gate=gate)
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    assert entry["exhaustive"] and len(entry["assignments"]) == 89
    runs = check.table("cell_type_1")["n_genes_passing_gate"]
    expected = [counts[tuple(a)] for a in entry["assignments"]]
    assert_array_equal(runs, expected)
    observed = counts[tuple(entry["labels"])]
    row = check.summary
    assert row["n_genes_passing_gate"][0] == observed
    others = np.array(expected)
    assert row["share_of_runs_with_passing_genes"][0] == np.mean(others >= 1)
    assert row["p_value"][0] == (1 + np.sum(others >= observed)) / 90
    renamings = theirs["renamings"][
        "every_group" if key == "dose_groups" else "vehicle_kept"
    ]
    assert renamings == (6 if key == "dose_groups" else 2)
    assert row["smallest_p_value"][0] == renamings / 90


@pytest.mark.parametrize("name", ["fixture", "duplicate", "sex"])
def test_units_and_assignments_match_r(name: str) -> None:
    # R builds the blocks from the sample table itself, counts the assignments as
    # products of binomial coefficients, and enumerates them where they are few.
    cube, design = fixture(name)
    blocks = reference()["cases"][name]["blocks"]
    exhaustive = name != "fixture"
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dose_groups"})
    check = permutation_check(result, n_permutations=100 if exhaustive else 1, rng=3)
    for entry in check.provenance["groups"]:
        theirs = blocks[entry["group_id"]]
        assert entry["units"] == theirs["kind"] == "samples"
        assert entry["n_assignments"] == int(theirs["n_assignments"])
        assert entry["samples"] == theirs["samples"]
        assert entry["labels"] == theirs["labels"]
        if exhaustive:
            mine = labels_of(check, entry["group_id"])
            assert len(mine) == len(set(mine)) == entry["n_assignments"]
            assert set(mine) == {tuple(row) for row in theirs["assignments"]}
    assert check.summary["exhaustive"].tolist() == [exhaustive] * len(check.summary)


# Against brute force over itertools.


def brute_force(
    labels: list[int], blocks: list[list[list[int]]]
) -> set[tuple[int, ...]]:
    """Every distinct assignment of the labels: the units of a block, each a list of
    sample positions, take each other's labels in every order, block by block."""
    found = set()
    orders = [itertools.permutations(range(len(block))) for block in blocks]
    for chosen in itertools.product(*(list(order) for order in orders)):
        assignment = list(labels)
        for block, order in zip(blocks, chosen, strict=True):
            for unit, source in zip(block, order, strict=True):
                for position in unit:
                    assignment[position] = labels[block[source][0]]
        found.add(tuple(assignment))
    return found


def blocks_by(values: list[Any], members: list[list[int]] | None = None) -> list:
    """Blocks of units with the same value, each unit a list of sample positions,
    one sample each unless ``members`` groups them."""
    units = members if members is not None else [[i] for i in range(len(values))]
    by_value: dict[Any, list[list[int]]] = {}
    for unit in units:
        by_value.setdefault(values[unit[0]], []).append(unit)
    return list(by_value.values())


@functools.cache
def designs() -> dict[str, tuple[Cube, Design, Any]]:
    """Cubes and designs with their blocks, built from the sample table by rules of
    the tests' own: one block; the sexes; animals moved whole; donors' own samples;
    a numeric covariate."""
    built = {}
    cube = series([0.0, 0.1, 1.0], replicates=2)
    built["vehicle and two doses x 2"] = (cube, Design(cube.samples, "~ 1"), None)
    cube = series([0.0, 0.1, 1.0, 10.0], replicates=[3, 2, 2, 1])
    built["groups of 3, 2, 2 and 1"] = (cube, Design(cube.samples, "~ 1"), None)
    cube = series([0.0, 1.0, 10.0], replicates=4)
    sex = [
        "F" if s.endswith(("_r1", "_r2")) else "M" for s in cube.samples["sample_id"]
    ]
    samples = cube.samples.with_columns({"sex": sex})
    built["~ sex"] = (cube, Design(samples, "~ sex"), "sex")
    cube = series([0.0, 1.0, 10.0], replicates=4)
    animal = [
        f"{s.split('_')[0]}_a{(int(s.split('_r')[1]) + 1) // 2}"
        for s in cube.samples["sample_id"]
    ]
    samples = cube.samples.with_columns({"animal": animal})
    built["animals nested in dose"] = (
        cube,
        Design(samples, "~ 1", random="(1 | animal)"),
        "animal",
    )
    cube = series([0.0, 1.0, 10.0], replicates=3, donor_sd=0.3)
    built["donors crossed with dose"] = (
        cube,
        Design(cube.samples, "~ 1", random="(1 | donor)"),
        "donor",
    )
    cube = series([0.0, 1.0, 10.0], replicates=2)
    age = np.arange(cube.n_samples, dtype=float)
    built["a numeric covariate"] = (
        cube,
        Design(cube.samples.with_columns({"age": age}), "~ age"),
        "age",
    )
    return built


def expected_blocks(name: str, design: Design, column: Any) -> tuple[list[int], list]:
    """The labels of the samples, and their blocks by the rule of each design."""
    samples = design.samples
    doses = np.asarray(samples["dose"], dtype=float)
    labels = np.searchsorted(np.unique(doses), doses).tolist()
    if column is None:
        return labels, [[[i] for i in range(len(labels))]]
    values = samples[column].tolist()
    if name == "animals nested in dose":
        members: dict[Any, list[int]] = {}
        for i, value in enumerate(values):
            members.setdefault(value, []).append(i)
        return labels, [list(members.values())]
    return labels, blocks_by(values)


@pytest.mark.parametrize("name", list(designs()))
def test_assignments_by_brute_force(name: str) -> None:
    cube, design, column = designs()[name]
    analysis = dream_voom if design.random else cytepool.limma_voom
    result = dose_response(
        cube, design, dose=AXIS, gate={"test": "dose_groups"}, analysis=analysis
    )
    labels, blocks = expected_blocks(name, design, column)
    expected = brute_force(labels, blocks)
    check = permutation_check(result, n_permutations=100, rng=5)
    entry = check.provenance["groups"][0]
    assert entry["samples"] == design.samples["sample_id"].tolist()
    assert entry["n_assignments"] == len(expected)
    if len(expected) == 1:
        assert not check.summary["checked"][0]
        assert "numeric covariate" in check.summary["reason"][0]
        return
    assert check.summary["n_assignments"][0] == len(expected)
    if len(expected) - 1 <= 100:
        found = labels_of(check, "cell_type_1")
        assert len(found) == len(set(found)) == len(expected)
        assert set(found) == expected
    else:
        assert all(tuple(a) in expected for a in entry["assignments"])
    kinds = {
        "animals nested in dose": "whole units",
        "donors crossed with dose": "doses within units",
    }
    assert entry["units"] == kinds.get(name, "samples")


def test_random_runs_are_uniform() -> None:
    # 9,000 draws on the 90 assignments of the vehicle and two doses in duplicate:
    # a chi-squared test of uniformity at the 0.001 level.
    from scipy.stats import chi2

    cube, design, _ = designs()["vehicle and two doses x 2"]
    units = _units(design, AXIS, design.samples["sample_id"].tolist())
    assert units.count() == 90
    generator = np.random.default_rng(11)
    counts = Counter(units.draw(generator) for _ in range(9000))
    assert len(counts) == 90
    observed = np.array(list(counts.values()), dtype=float)
    statistic = float(np.sum((observed - 100.0) ** 2 / 100.0))
    assert statistic < chi2.isf(0.001, 89)


def test_renamings_tie_and_set_the_smallest_p_value() -> None:
    # An order-free gate passes the same genes when an assignment only renames the
    # dose groups: the 90 counts of the dose-group gate fall in 15 sets of 6 equal
    # values, one per partition of the 6 samples into three pairs, and no p-value
    # can fall below 6/90. The Williams-type test tells the doses apart.
    cube, design = fixture("duplicate")
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dose_groups"})
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    counts = [int(check.summary["n_genes_passing_gate"][0])]
    counts += check.table("cell_type_1")["n_genes_passing_gate"].tolist()

    def partition(labels: tuple[int, ...]) -> frozenset:
        return frozenset(
            frozenset(i for i, label in enumerate(labels) if label == value)
            for value in set(labels)
        )

    by_partition: dict[frozenset, set[int]] = {}
    for labels, count in zip(labels_of(check, "cell_type_1"), counts, strict=True):
        by_partition.setdefault(partition(labels), set()).add(count)
    assert len(by_partition) == 15
    assert all(len(values) == 1 for values in by_partition.values())
    assert check.summary["smallest_p_value"][0] == 6 / 90
    assert check.summary["p_value"][0] >= 6 / 90
    assert entry["renamings"] == 6
    williams = dose_response(cube, design, dose=AXIS, gate={"test": "williams"})
    assert permutation_check(williams, rng=1).summary["smallest_p_value"][0] == 1 / 90
    # Swapping the two treated doses' labels changes the Williams-type p-values.
    doses = np.asarray(design.samples["dose"], dtype=float)
    swapped = np.where(doses == 0.1, 1.0, np.where(doses == 1.0, 0.1, doses))
    renamed = Design(design.samples.with_columns({"dose": swapped}), "~ 1")
    other = dose_response(cube, renamed, dose=AXIS, gate={"test": "williams"})
    first = williams.table("cell_type_1", "gate")["p_value"]
    second = other.table("cell_type_1", "gate")["p_value"]
    assert not np.allclose(first, second)


def test_the_summary_by_hand() -> None:
    # The share of runs with genes, the mean per run and (1 + b) / (1 + B), with b the
    # runs that pass at least as many genes as the true doses.
    found = _statistics(2, np.array([0.0, 2.0, 5.0, 1.0]))
    assert found == {
        "share_of_runs_with_passing_genes": 0.75,
        "mean_passing_genes_per_run": 2.0,
        "p_value": 3 / 5,
    }
    # No gene with the true doses: every run passes at least as many.
    assert _statistics(0, np.array([0.0, 0.0, 3.0]))["p_value"] == 1.0
    # More genes than any run: the smallest p-value of B runs.
    assert _statistics(9, np.array([0.0, 2.0, 5.0]))["p_value"] == 1 / 4


# Runs made again by the test.


@functools.cache
def responsive() -> Cube:
    """A series with responsive genes, whose runs pass varying numbers of genes."""
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0],
        replicates=3,
        cell_types=["one", "two"],
        cells=300.0,
        n_genes=120,
        responsive=0.4,
        rng=21,
    )
    return cube


def again(
    cube: Cube, design: Design, entry: dict[str, Any], run: int, **arguments: Any
) -> DoseResponseResult:
    """dose_response called by the test on the design with a run's recorded doses."""
    doses = np.asarray(design.samples["dose"], dtype=float).copy()
    ids = design.samples["sample_id"].tolist()
    for sample, label in zip(entry["samples"], entry["assignments"][run], strict=True):
        doses[ids.index(sample)] = entry["doses"][label]
    random = [str(effect) for effect in design.random] or None
    permuted = Design(
        design.samples.with_columns({"dose": doses}), design.formula, random=random
    )
    return dose_response(
        cube, permuted, dose=AXIS, groups=[entry["group_id"]], **arguments
    )


OPTIONS = {
    "limma-voom, robust, quality weights": {"robust": True, "quality_weights": True},
    "limma-voom, prior weights": {"prior_weights": "n_cells"},
    "limma-trend": {"analysis": limma_trend},
    "an expression filter of settings": {"expression_filter": {"min_count": 5}},
    "no expression filter": {"expression_filter": False},
    "the fold-change gate": {"gate": {"fold_change": 1.5}},
    "the Williams-type gate": {"gate": {"test": "williams"}, "rng": 7},
}


@pytest.mark.parametrize("name", list(OPTIONS))
def test_a_run_is_dose_response_on_the_permuted_doses(name: str) -> None:
    # The test rebuilds runs from the doses the provenance records and calls
    # dose_response with the options as given, so that an option lost on the way
    # from the record, or the wrong samples permuted, changes the counts.
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    arguments = dict(OPTIONS[name])
    result = dose_response(cube, design, dose=AXIS, **arguments)
    check = permutation_check(result, n_permutations=4, rng=2)
    seeds = arguments.pop("rng", None)
    del seeds  # the runs draw their own seeds for the multiple contrast tests
    for entry in check.provenance["groups"]:
        group = entry["group_id"]
        runs = check.table(group)["n_genes_passing_gate"]
        assert (
            entry["samples"]
            == result.provenance["groups"][result.groups.index(group)]["fit"]["samples"]
        )
        for run in (0, 3):
            found = again(cube, design, entry, run, **arguments)
            if name == "the Williams-type gate":
                continue  # Monte Carlo p-values from the run's own seed
            assert (
                np.count_nonzero(found.table(group, "gate")["passes_gate"]) == runs[run]
            )
    assert check.summary["share_of_runs_with_passing_genes"].tolist() != [0.0, 0.0]


def test_a_run_under_dream_is_dose_response_on_the_permuted_doses() -> None:
    cube, design, _ = designs()["donors crossed with dose"]
    result = dose_response(cube, design, dose=AXIS, analysis=dream_voom)
    check = permutation_check(result, n_permutations=3, rng=4)
    entry = check.provenance["groups"][0]
    runs = check.table("cell_type_1")["n_genes_passing_gate"]
    for run in range(3):
        found = again(cube, design, entry, run, analysis=dream_voom)
        assert (
            np.count_nonzero(found.table("cell_type_1", "gate")["passes_gate"])
            == runs[run]
        )
    # Donors keep their samples: each run permutes the doses within each donor.
    donors = design.samples["donor"].tolist()
    ids = design.samples["sample_id"].tolist()
    for assignment in entry["assignments"]:
        for donor in set(donors):
            mine = [
                assignment[entry["samples"].index(s)]
                for s, d in zip(ids, donors)
                if d == donor
            ]
            assert sorted(mine) == [0, 1, 2]


def test_runs_do_not_depend_on_threads_or_other_cell_types() -> None:
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS)
    first = permutation_check(result, n_permutations=5, rng=9)
    second = permutation_check(result, n_permutations=5, rng=9)
    assert first.provenance == second.provenance | {
        "versions": first.provenance["versions"]
    }
    for group in result.groups:
        assert first.table(group).equals(second.table(group))
    # One cell type alone runs as it does among the others.
    alone = permutation_check(result, n_permutations=5, groups="two", rng=9)
    assert alone.table("two").equals(first.table("two"))
    assert alone.provenance["groups"][0] == first.provenance["groups"][1]
    # The analysis on one thread or two gives the same runs.
    threads = dose_response(cube, design, dose=AXIS, n_threads=1)
    other = permutation_check(threads, n_permutations=5, rng=9)
    for group in result.groups:
        assert other.table(group).equals(first.table(group))
    # The seed of a cell type's runs: numpy.random.default_rng([seed, CRC-32 of its
    # name, 1000]), pinned by the first run's assignment.
    assert first.provenance["groups"][0]["assignments"][0] == [
        2,
        0,
        3,
        1,
        0,
        0,
        3,
        2,
        1,
        2,
        3,
        1,
    ]
    # A generator gives its seed to the record, which gives the same runs back.
    drawn = permutation_check(result, n_permutations=5, rng=np.random.default_rng(3))
    seed = drawn.settings["seed"]
    again_ = permutation_check(result, n_permutations=5, rng=seed)
    assert again_.provenance["groups"] == drawn.provenance["groups"]


# Cell types that cannot be checked, the guard and refusals.


def mixed() -> tuple[Cube, Design]:
    """A series with a cell type that is refused (no vehicle sample of enough cells),
    one whose doses cannot estimate the spline of the design, and one that can."""
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        replicates=2,
        cell_types=["kept", "few_doses", "no_vehicle"],
        cells=[300.0, 300.0, 300.0],
        n_genes=60,
        composition={
            "few_doses": [1.0, 1.0, 0.005, 0.005, 0.005, 0.005],
            "no_vehicle": [0.005, 1.0, 1.0, 1.0, 1.0, 1.0],
        },
        rng=13,
    )
    return cube, Design(cube.samples, "~ 1")


def test_cell_types_that_cannot_be_checked() -> None:
    cube, design = mixed()
    result = dose_response(cube, design, dose=AXIS, spline_df=2)
    check = permutation_check(result, n_permutations=2, rng=1)
    summary = check.summary
    assert summary["group_id"].tolist() == ["kept", "few_doses", "no_vehicle"]
    assert summary["checked"].tolist() == [True, False, False]
    reasons = summary["reason"].tolist()
    assert reasons[0] == ""
    assert reasons[1].startswith("the gate did not run: its test, 'spline'")
    assert reasons[2].startswith("not tested (refused)")
    assert np.isnan(summary["n_genes_passing_gate"][1:]).all()
    assert summary["n_runs"].tolist() == [2, 0, 0]
    with pytest.raises(KeyError, match="was not checked: the gate did not run"):
        check.table("few_doses")
    for group, why in (
        ("few_doses", "the gate did not run"),
        ("no_vehicle", "not tested"),
    ):
        with pytest.raises(KeyError, match=f"cannot be checked: {why}"):
            permutation_check(result, groups=group)
    # The fit's numeric covariate leaves no assignment but the true one.
    cube, design, _ = designs()["a numeric covariate"]
    single = permutation_check(dose_response(cube, design, dose=AXIS), rng=1)
    assert single.summary["n_assignments"][0] == 1
    assert single.summary["n_runs"][0] == 0
    assert "drop the covariate, or bin it" in single.summary["reason"][0]
    assert np.isfinite(single.summary["n_genes_passing_gate"][0])


def rebuilt(result: DoseResponseResult, **changes: Any) -> DoseResponseResult:
    """The result with parts of its provenance or its design replaced."""
    provenance = result.provenance
    for key, value in changes.pop("provenance", {}).items():
        provenance[key] = value
    parts = {
        "cube": result.cube,
        "design": result.design,
    } | changes
    return DoseResponseResult(
        result.summary,
        dict(result.tables),
        {
            group: result._fits[group]
            for group in result.groups
            if group in result._fits
        },
        result.design_report,
        provenance,
        **parts,
    )


def test_a_result_its_doses_do_not_give_again_is_refused() -> None:
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS)
    # A record that says robust where the fit was not: the true doses run again
    # robustly, and the gate's p-values differ.
    settings = result.provenance["analysis_settings"] | {"robust": True}
    changed = rebuilt(result, provenance={"analysis_settings": settings})
    with pytest.raises(ValueError, match="does not give the result's gate"):
        permutation_check(changed, n_permutations=2)
    # Another cube under the same record.
    other = series(
        [0.0, 0.1, 1.0, 10.0],
        replicates=3,
        cell_types=["one", "two"],
        n_genes=120,
        seed=22,
    )
    with pytest.raises(ValueError, match="ran again on a cube other than the result"):
        permutation_check(rebuilt(result, cube=other), n_permutations=2)


def test_a_run_that_does_not_run_the_gate_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cytepool import permutation_checks

    cube = responsive()
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, groups="one")
    calls = []

    def without_gate(*args: Any, **kwargs: Any) -> DoseResponseResult:
        found = dose_response(*args, **kwargs)
        calls.append(found)
        if len(calls) < 3:
            return found
        tables = {key: t for key, t in found.tables.items() if key[1] != "gate"}
        return DoseResponseResult(
            found.summary,
            tables,
            dict(found._fits),
            found.design_report,
            found.provenance,
        )

    monkeypatch.setattr(permutation_checks, "dose_response", without_gate)
    with pytest.raises(ValueError, match="run 2 of 'one' did not run the gate"):
        permutation_check(result, n_permutations=3, rng=1)


def test_refusals() -> None:
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS, groups="one")
    with pytest.raises(TypeError, match="result must be a DoseResponseResult"):
        permutation_check("result")  # type: ignore[arg-type]
    for value, error in (
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ):
        with pytest.raises(error, match="n_permutations"):
            permutation_check(result, n_permutations=value)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="groups must be a cell type"):
        permutation_check(result, groups=3)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="groups must be cell types"):
        permutation_check(result, groups=[3])  # type: ignore[list-item]
    with pytest.raises(KeyError, match="no cell type 'tow'.*did you mean 'two'"):
        permutation_check(dose_response(cube, design, dose=AXIS), groups="tow")
    with pytest.raises(ValueError, match="rng must be a non-negative seed"):
        permutation_check(result, rng=-1)
    with pytest.raises(TypeError, match="rng must be"):
        permutation_check(result, rng="1")  # type: ignore[arg-type]
    bare = rebuilt(result, cube=None, design=None)
    with pytest.raises(ValueError, match="keeps no cube and design"):
        permutation_check(bare)
    for random in ("(1 | donor) + (1 | dose_group)", "(1 + dose | donor)"):
        odd = rebuilt(result, design=Design(cube.samples, "~ 1", random=random))
        with pytest.raises(ValueError, match="leave no exchangeable units"):
            permutation_check(odd)


def test_the_check_reads_back(tmp_path: Path) -> None:
    cube = responsive()
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    check = permutation_check(result, n_permutations=6, rng=8)
    summary = check.summary
    assert summary.columns == (
        "group_id",
        "checked",
        "reason",
        "n_assignments",
        "exhaustive",
        "n_runs",
        "n_genes_passing_gate",
        "share_of_runs_with_passing_genes",
        "mean_passing_genes_per_run",
        "p_value",
        "smallest_p_value",
    )
    count = math.factorial(12) // math.factorial(3) ** 4
    assert summary["n_assignments"].tolist() == [float(count)] * 2
    assert summary["smallest_p_value"].tolist() == [1 / 7] * 2
    table = check.table("one")
    assert table.columns == ("run", "n_genes_passing_gate")
    assert table["run"].tolist() == [1, 2, 3, 4, 5, 6]
    record = check.provenance
    assert record["settings"] == {"n_permutations": 6, "groups": None, "seed": 8}
    assert record["result"]["gate"] == result.gate
    entry = record["groups"][0]
    assert entry["n_assignments"] == count and isinstance(entry["n_assignments"], int)
    assert len(entry["assignments"]) == 6
    assert all(len(a) == len(entry["samples"]) == 12 for a in entry["assignments"])
    assert entry["units"] == "samples" and entry["unit"] is None
    assert entry["blocks"] == [[[s] for s in entry["samples"]]]
    copied = pickle.loads(pickle.dumps(check))
    assert copied.summary.equals(summary)
    assert copied.table("two").equals(check.table("two"))
    assert copied.provenance == record
    text = " ".join(repr(check).split())
    assert text.startswith(
        "PermutationCheck of the gate, the spline test at 5% FDR: 2 cell types, 2 checked; seed 8"
    )
    assert "6 random runs of" in text
    with pytest.raises(ValueError, match="name one with group"):
        check.table()
    with pytest.raises(TypeError, match="group must be a cell type"):
        check.table(1)  # type: ignore[arg-type]
    with pytest.raises(AttributeError, match="immutable"):
        check.summary = summary  # type: ignore[misc]


def test_units_of_unequal_size_keep_the_dose_groups() -> None:
    # Two animals per dose, one with two samples and one with one: animals move
    # whole only among animals of the same size, so that each run keeps every dose
    # group's samples and animals. The second cell type drops a sample of few cells,
    # and its runs move only the samples it uses.
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0, 10.0],
        replicates=3,
        cell_types=["all", "dropped"],
        cells=300.0,
        n_genes=60,
        rng=17,
    )
    ids = cube.samples["sample_id"].tolist()
    n_cells = np.array(cube.n_cells, copy=True)
    n_cells[ids.index("d2_r3"), 1] = 2
    cube = Cube(
        {"counts": np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
    )
    animal = [f"{s.split('_')[0]}_{'a' if s[-1] in '12' else 'b'}" for s in ids]
    design = Design(
        cube.samples.with_columns({"animal": animal}), "~ 1", random="(1 | animal)"
    )
    result = dose_response(cube, design, dose=AXIS, analysis=dream_voom)
    check = permutation_check(result, n_permutations=30, rng=6)
    for entry in check.provenance["groups"]:
        fitted = next(
            e for e in result.provenance["groups"] if e["group_id"] == entry["group_id"]
        )
        assert entry["samples"] == fitted["fit"]["samples"]
        assert entry["units"] == "whole units"
        sizes = Counter(entry["labels"])
        for assignment in entry["assignments"]:
            assert Counter(assignment) == sizes
            by_animal: dict[str, set[int]] = {}
            for sample, label in zip(entry["samples"], assignment, strict=True):
                by_animal.setdefault(animal[ids.index(sample)], set()).add(label)
            assert all(len(labels) == 1 for labels in by_animal.values())
    first, second = check.provenance["groups"]
    assert first["n_assignments"] == math.factorial(4) ** 2
    # The cell type that drops d2_r3 has one animal of one sample fewer: the b
    # animals of three doses, and a of four doses form the blocks.
    assert second["n_assignments"] == math.factorial(4) * math.factorial(3)
    assert "d2_r3" not in second["samples"]


def test_runs_leave_the_batch_out() -> None:
    # Lanes of two samples that the true doses cross, so that no dose group is
    # confounded with a lane. By inclusion and exclusion, 9 x 6 - 18 + 6 = 42 of the
    # 89 other assignments put both samples of a dose in one lane: 6 for each of the 3
    # doses in each of the 3 lanes, 1 for each of 18 pairs of doses in two lanes, and
    # 1 for each of the 6 with every dose in a lane of its own. The result's check of
    # the batch is the user's; the runs leave the batch out, as a fit does, and so
    # each runs the gate.
    cube = series([0.0, 0.1, 1.0], replicates=2)
    assert cube.samples["dose"].tolist() == [0.0, 0.0, 0.1, 0.1, 1.0, 1.0]
    lanes = ["a", "b", "b", "c", "c", "a"]
    design = Design(cube.samples.with_columns({"lane": lanes}), "~ 1")
    result = dose_response(cube, design, dose=AXIS, batch="lane")
    assert result.settings["batch"] == "lane"
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    assert entry["exhaustive"] and len(entry["assignments"]) == 89
    confounded = 0
    for assignment in entry["assignments"]:
        by_dose: dict[int, set[str]] = {}
        for sample, label in zip(entry["samples"], assignment, strict=True):
            by_dose.setdefault(label, set()).add(lanes[entry["samples"].index(sample)])
        confounded += any(len(found) == 1 for found in by_dose.values())
    assert confounded == 42
    found = again(cube, design, entry, 0)
    assert check.table("cell_type_1")["n_genes_passing_gate"][0] == np.count_nonzero(
        found.table("cell_type_1", "gate")["passes_gate"]
    )


def test_a_check_refuses_wrong_parts_and_stays_as_built() -> None:
    cube = responsive()
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    check = permutation_check(result, n_permutations=2, rng=3)
    summary, provenance = check.summary, check.provenance
    tables = {group: check.table(group) for group in ("one", "two")}
    for parts, message in (
        (("summary", tables, provenance), "summary must be a Table"),
        ((summary, ["one"], provenance), "tables must be a mapping"),
        ((summary, {1: tables["one"]}, provenance), "keyed by cell type"),
        ((summary, {"one": "table"}, provenance), r"tables\['one'\] must be a Table"),
        ((summary, tables, "record"), "provenance must be a mapping"),
    ):
        with pytest.raises(TypeError, match=message):
            PermutationCheck(*parts)  # type: ignore[arg-type]
    with pytest.raises(AttributeError, match="immutable"):
        check.summary = summary  # type: ignore[misc]
    with pytest.raises(AttributeError, match="immutable"):
        del check.summary  # type: ignore[misc]
    with pytest.raises(KeyError, match="no cell type 'three' was asked for"):
        check.table("three")
    assert PermutationCheck(summary, tables, provenance).summary.equals(summary)


def test_the_printout_says_what_the_runs_can_give() -> None:
    # Series where no gene responds, whose true doses pass no gene: every run passes
    # at least as many, and p is 1.
    flat = series([0.0, 0.1, 1.0, 10.0], replicates=3, responsive=0.0)
    result = dose_response(flat, Design(flat.samples, "~ 1"), dose=AXIS)
    assert not np.any(result.table("cell_type_1", "gate")["passes_gate"])
    # 20 random runs can give 1/21, below 0.05: the line ends at p.
    text = " ".join(repr(permutation_check(result, n_permutations=20, rng=2)).split())
    assert "over 20 random runs of 369,600 assignments" in text
    assert text.endswith("genes per run; p = 1")
    # The vehicle and one dose in duplicate allow 6 assignments, and the dose-group
    # gate no p-value below 2/6: the line says so.
    pairs = series([0.0, 0.1, 1.0], replicates=2, responsive=0.0)
    samples = pairs.samples.filter(pairs.samples["dose"] <= 0.1)
    result = dose_response(pairs, Design(samples, "~ 1"), dose=AXIS)
    assert result.gate["test"] == "dose_groups"
    text = " ".join(repr(permutation_check(result, rng=2)).split())
    assert "over all 5 other assignments" in text
    assert text.endswith("p = 1; the smallest these runs can give is 0.33")
    # A fold-change gate in the heading.
    large = dose_response(
        responsive(), Design(responsive().samples, "~ 1"), dose=AXIS,
        gate={"fold_change": 1.5},
    )  # fmt: skip
    text = " ".join(repr(permutation_check(large, n_permutations=1, rng=2)).split())
    assert text.startswith(
        "PermutationCheck of the gate, TREAT beyond 1.5-fold at each dose, at 5% FDR:"
    )


def test_runs_take_the_tests_the_result_named() -> None:
    # The runs fit the contrasts of the tests along dose that the result named, with
    # the gate's test, so that the true doses give the result's gate p-values bit
    # for bit; multiple contrast tests other than the gate's are left out, and a
    # fold-change gate with no such test takes the dose groups'.
    assert _rerun_tests(None, "spline") is None
    assert _rerun_tests(["dose_groups", "williams"], "spline") == [
        "dose_groups",
        "spline",
    ]
    assert _rerun_tests(["spline", "dunnett"], "dunnett") == ["spline", "dunnett"]
    assert _rerun_tests(["williams"], "each_dose") == ["dose_groups"]
    assert _rerun_tests(["linear_trend"], "each_dose") == ["linear_trend"]
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    for tests, gate in (
        (("dose_groups", "williams"), None),
        (("linear_trend",), {"fold_change": 1.5}),
    ):
        result = dose_response(cube, design, dose=AXIS, tests=tests, gate=gate, rng=4)
        check = permutation_check(result, n_permutations=2, rng=5)
        for entry in check.provenance["groups"]:
            group = entry["group_id"]
            found = again(cube, design, entry, 1, tests=tests, gate=gate, rng=4)
            assert check.table(group)["n_genes_passing_gate"][1] == np.count_nonzero(
                found.table(group, "gate")["passes_gate"]
            )


def test_units_that_each_hold_one_dose_leave_one_assignment() -> None:
    # Animals of 2 samples received the vehicle, of 1 sample the first dose and of 3
    # the second: blocks of animals of the same size each hold one dose, so no dose
    # label can move.
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0], replicates=6, cell_types=1, cells=300.0, n_genes=60, rng=8
    )
    ids = cube.samples["sample_id"].tolist()
    kept = ids[:4] + ids[6:8] + ids[12:18]
    animal = ["v1", "v1", "v2", "v2", "a1", "a2", "b1", "b1", "b1", "b2", "b2", "b2"]
    samples = cube.samples.filter(np.isin(cube.samples["sample_id"], kept))
    assert samples["dose"].tolist() == [0.0] * 4 + [0.1] * 2 + [1.0] * 6
    design = Design(
        samples.with_columns({"animal": animal}), "~ 1", random="(1 | animal)"
    )
    result = dose_response(cube, design, dose=AXIS, analysis=dream_voom)
    check = permutation_check(result, rng=1)
    assert check.summary["n_assignments"].tolist() == [1.0]
    assert check.summary["reason"][0] == (
        "each block of units that the design treats alike holds one dose, so no "
        "dose label can move"
    )


# The chapter.

CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/dose_response.md"
)


def chapter_section(label: str) -> str:
    return CHAPTER.read_text().split(f"({label})=")[1].split("\n(")[0]


def test_the_chapter_s_check_example_runs() -> None:
    """The code of the chapter's section on the check, on the series of its section
    on the tests: what it prints and the numbers its text reads from the summary."""
    setup = re.findall(
        r"```python\n(.*?)```", chapter_section("dose-response-tests"), re.S
    )
    text = chapter_section("dose-permutation-check")
    blocks = re.findall(r"```python\n(.*?)```", text, re.S)
    assert len(blocks) == 1
    namespace: dict[str, Any] = {"cp": cytepool}
    exec(setup[0].replace("print(result)\n", ""), namespace)  # noqa: S102
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        exec(blocks[0], namespace)  # noqa: S102
    check = namespace["check"]
    assert output.getvalue() == repr(check) + "\n"
    assert "```text\n" + repr(check) + "\n```" in text
    summary = check.summary.filter(check.summary["checked"])
    assert summary["group_id"].tolist() == ["hepatocyte", "stellate"]
    assert summary["share_of_runs_with_passing_genes"].tolist() == [0.03, 0.03]
    assert summary["p_value"].tolist() == [1 / 101] * 2
    assert summary["smallest_p_value"].tolist() == [1 / 101] * 2
    assert "the gate passed genes in 3 of 100 runs in each cell type" in text
    assert "so p is 1/101, the smallest that 100 runs can give" in text


ROW = re.compile(
    r"\| The vehicle and (\d) doses? x (\d) \| ([\d,]+) \| (\d+)/([\d,]+), ([\d.]+) \|"
)


def test_the_chapter_s_table_of_the_smallest_p_values() -> None:
    """Each row of the chapter's table against a check of that design under the
    dose-group gate: its assignments, and its smallest p-value with every assignment
    run, or, where they are many, the renamings that tie: every dose group has the
    same size, so each of the K! orders of the K groups' labels passes the same
    genes."""
    text = chapter_section("dose-permutation-check")
    rows = ROW.findall(text)
    assert len(rows) == 6
    for treated, replicates, count, ties, over, shown in rows:
        doses = [0.0, 0.1, 1.0, 10.0][: int(treated) + 1]
        # The simulator places curves on two doses or more; one dose is taken from
        # a series of two.
        cube = series(
            [*doses, 30.0] if len(doses) == 2 else doses,
            n_genes=40,
            replicates=int(replicates),
        )
        samples = cube.samples.filter(cube.samples["dose"] <= max(doses))
        result = dose_response(
            cube, Design(samples, "~ 1"), dose=AXIS, gate={"test": "dose_groups"}
        )
        assignments = int(count.replace(",", ""))
        assert int(over.replace(",", "")) == assignments
        exhaustive = assignments - 1 <= 100
        check = permutation_check(
            result, n_permutations=100 if exhaustive else 1, rng=1
        )
        entry = check.provenance["groups"][0]
        assert entry["n_assignments"] == assignments
        if exhaustive:
            smallest = float(check.summary["smallest_p_value"][0])
            assert smallest == int(ties) / assignments
        else:
            assert int(ties) == math.factorial(len(doses))
        assert float(shown) == float(f"{int(ties) / assignments:.2g}")
    assert (
        "with `n_permutations` of 1,679 and 2,519, which run every assignment" in text
    )
    # The default gate at the vehicle and two doses in duplicate: a spline of one
    # column through the three groups' means, which ties as the dose groups do.
    cube = series([0.0, 0.1, 1.0], n_genes=40, replicates=2)
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    assert result.gate["test"] == "spline"
    check = permutation_check(result, rng=1)
    assert check.summary["smallest_p_value"].tolist() == [6 / 90]
    assert (
        "At the vehicle and two doses in duplicate, no p-value of the default" in text
    )
