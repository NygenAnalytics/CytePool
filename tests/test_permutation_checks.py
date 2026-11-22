"""Tests for permutation_check and PermutationCheck.

Reference values in ``tests/data/dose_gate/limma.json`` come from R, written by
``validation/dose_gate.py``: R reads the sample table itself, builds the blocks of
exchangeable units, counts and enumerates the assignments of the dose labels, and
refits limma 3.62.1 on each of the 90 assignments of a series of the vehicle and two
doses in duplicate and the 36 of a series of the vehicle and one dose in two sexes,
counting as ties the assignments whose refit gives the true doses' genes and
p-values. Other tests check what R does not: the assignments against brute force over
itertools on blocks that the tests build from the sample table, the uniformity of
random runs, runs made again from their recorded assignments by dose_response called
in the test, ties against the renamings and reflections of the doses, counts put
into the summary by hand, invariants and refusals.
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
    CandidateShapes,
    Cube,
    Design,
    DoseAxis,
    DoseResponseResult,
    PermutationCheck,
    Table,
    dose_response,
    dream_voom,
    limma_trend,
    permutation_check,
    simulate_dose_series,
)
from cytepool.permutation_checks import _statistics, _units

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


@pytest.mark.parametrize("name", ["duplicate", "sex"])
@pytest.mark.parametrize("gate", [{"test": "dose_groups"}, {"fold_change": 1.5}])
def test_each_run_matches_r_on_every_assignment(
    name: str, gate: dict[str, Any]
) -> None:
    # The vehicle and two doses in duplicate, and the vehicle and one dose in two
    # sexes: R refits limma on each assignment it enumerates, counts the genes the
    # gate passes, and the ties, whose refit gives the true doses' p-values.
    cube, design = fixture(name)
    theirs = reference()["cases"][name]["blocks"]["cell_type_1"]
    key = "dose_groups" if "test" in gate else "each_dose"
    counts = {
        tuple(a): int(n)
        for a, n in zip(theirs["assignments"], theirs["counts"][key], strict=True)
    }
    result = dose_response(cube, design, dose=AXIS, gate=gate)
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    total = len(counts)
    assert total == (90 if name == "duplicate" else 36)
    assert entry["exhaustive"] and len(entry["assignments"]) == total - 1
    runs = check.table("cell_type_1")["n_genes_passing_gate"]
    expected = [counts[tuple(a)] for a in entry["assignments"]]
    assert_array_equal(runs, expected)
    observed = counts[tuple(entry["labels"])]
    row = check.summary
    assert row["n_genes_passing_gate"][0] == observed
    others = np.array(expected)
    assert row["share_of_runs_with_passing_genes"][0] == np.mean(others >= 1)
    assert row["p_value"][0] == (1 + np.sum(others >= observed)) / total
    # The renamings that keep each sex's count of every dose: 3! of the duplicate's
    # three pairs under the dose groups, and the swap of the vehicle and the treated
    # doses, the identity included, under the fold-change gate and in sex.
    ties = int(theirs["ties"][key])
    assert ties == (6 if (name, key) == ("duplicate", "dose_groups") else 2)
    assert row["smallest_possible_p_value"][0] == ties / total
    assert len(entry["tied_runs"]) == ties - 1


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


def blocks_by(
    values: list[Any],
    members: list[list[int]] | None = None,
    units: list[list[int]] | None = None,
) -> list:
    """Blocks of units with the same value, each unit a list of sample positions,
    one sample each unless ``members`` groups them; with ``units``, ``values`` and
    ``members`` index the units, and each block holds their samples."""
    found = members if members is not None else [[i] for i in range(len(values))]
    by_value: dict[Any, list[list[int]]] = {}
    for unit in found:
        by_value.setdefault(values[unit[0]], []).append(unit)
    blocks = list(by_value.values())
    if units is None:
        return blocks
    return [[units[unit[0]] for unit in block] for block in blocks]


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
    # Animals of two samples, one female and one male animal at each dose: animals
    # move whole within each sex.
    cube = series([0.0, 1.0, 10.0], replicates=4)
    ids = cube.samples["sample_id"].tolist()
    animal = [f"{s.split('_')[0]}_a{(int(s.split('_r')[1]) + 1) // 2}" for s in ids]
    sex = ["F" if s.endswith(("_r1", "_r2")) else "M" for s in ids]
    samples = cube.samples.with_columns({"animal": animal, "sex": sex})
    built["animals nested in dose, by sex"] = (
        cube,
        Design(samples, "~ sex", random="(1 | animal)"),
        "animal",
    )
    # Donors crossed with dose, each donor's samples on two plates, so that the
    # doses move within each donor's plate.
    cube = series([0.0, 1.0, 10.0], replicates=3, donor_sd=0.3)
    donors = cube.samples["donor"].tolist()
    doses = np.asarray(cube.samples["dose"], dtype=float)
    level = np.searchsorted(np.unique(doses), doses)
    order = {donor: i for i, donor in enumerate(dict.fromkeys(donors))}
    plate = ["a" if (k + order[d]) % 3 < 2 else "b" for k, d in zip(level, donors)]
    samples = cube.samples.with_columns({"plate": plate})
    built["donors crossed with dose, by plate"] = (
        cube,
        Design(samples, "~ plate", random="(1 | donor)"),
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
    if name.startswith("animals nested in dose"):
        members: dict[Any, list[int]] = {}
        for i, value in enumerate(values):
            members.setdefault(value, []).append(i)
        if name == "animals nested in dose":
            return labels, [list(members.values())]
        # Animals of the same sex and size form a block.
        sexes = samples["sex"].tolist()
        units = list(members.values())
        keys = [(sexes[unit[0]], len(unit)) for unit in units]
        return labels, blocks_by(keys, [[i] for i in range(len(units))], units)
    if name == "donors crossed with dose, by plate":
        plates = samples["plate"].tolist()
        return labels, blocks_by(list(zip(values, plates, strict=True)))
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
    kinds = {"animals": "whole units", "donors": "doses within units"}
    assert entry["units"] == kinds.get(name.split()[0], "samples")


@pytest.mark.parametrize("name", ["vehicle and two doses x 2", "~ sex, one dose"])
def test_random_runs_are_uniform(name: str) -> None:
    # 100 draws per assignment, on the 90 assignments of the vehicle and two doses in
    # duplicate, and on the 36 of the vehicle and one dose with two of each in each
    # sex, two blocks of the same size: a chi-squared test of uniformity at the
    # 0.001 level, which one permutation reused for both blocks fails.
    from scipy.stats import chi2

    if name == "~ sex, one dose":
        cube = series([0.0, 1.0, 10.0], replicates=4)
        samples = cube.samples.filter(cube.samples["dose"] <= 1.0)
        ids = samples["sample_id"].tolist()
        sex = ["F" if s.endswith(("_r1", "_r2")) else "M" for s in ids]
        design = Design(samples.with_columns({"sex": sex}), "~ sex")
    else:
        _, design, _ = designs()[name]
    units = _units(design, AXIS, design.samples["sample_id"].tolist())
    total = units.count()
    assert total == (90 if name.startswith("vehicle") else 36)
    generator = np.random.default_rng(11)
    counts = Counter(units.draw(generator) for _ in range(100 * total))
    assert len(counts) == total
    observed = np.array(list(counts.values()), dtype=float)
    statistic = float(np.sum((observed - 100.0) ** 2 / 100.0))
    assert statistic < chi2.isf(0.001, total - 1)


def test_every_assignment_runs_when_the_runs_allow_it() -> None:
    # The duplicate's 90 assignments: 89 runs take every other assignment, and 88
    # draw at random.
    cube, design = fixture("duplicate")
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dose_groups"})
    every = permutation_check(result, n_permutations=89, rng=1)
    assert every.summary["exhaustive"].tolist() == [True]
    assert every.summary["n_runs"].tolist() == [89]
    random = permutation_check(result, n_permutations=88, rng=1)
    assert random.summary["exhaustive"].tolist() == [False]
    assert random.summary["n_runs"].tolist() == [88]


def partition(labels: tuple[int, ...]) -> frozenset:
    """The samples of each dose label, without the labels."""
    return frozenset(
        frozenset(i for i, label in enumerate(labels) if label == value)
        for value in set(labels)
    )


def test_renamings_tie_and_set_the_smallest_possible_p_value() -> None:
    # An order-free gate passes the same genes when an assignment only renames the
    # dose groups: the 90 counts of the dose-group gate fall in 15 sets of 6 equal
    # values, one per partition of the 6 samples into three pairs, the runs that tie
    # are the 5 other renamings of the true partition, and no p-value can fall below
    # 6/90. The Williams-type test tells the doses apart.
    cube, design = fixture("duplicate")
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dose_groups"})
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    counts = [int(check.summary["n_genes_passing_gate"][0])]
    counts += check.table("cell_type_1")["n_genes_passing_gate"].tolist()
    by_partition: dict[frozenset, set[int]] = {}
    for labels, count in zip(labels_of(check, "cell_type_1"), counts, strict=True):
        by_partition.setdefault(partition(labels), set()).add(count)
    assert len(by_partition) == 15
    assert all(len(values) == 1 for values in by_partition.values())
    true = partition(tuple(entry["labels"]))
    assert entry["tied_runs"] == [
        run
        for run, labels in enumerate(entry["assignments"], start=1)
        if partition(tuple(labels)) == true
    ]
    assert len(entry["tied_runs"]) == 5
    assert check.summary["smallest_possible_p_value"][0] == 6 / 90
    assert check.summary["p_value"][0] >= 6 / 90
    williams = dose_response(cube, design, dose=AXIS, gate={"test": "williams"})
    found = permutation_check(williams, rng=1)
    assert found.summary["smallest_possible_p_value"][0] == 1 / 90
    # Swapping the two treated doses' labels changes the Williams-type p-values.
    doses = np.asarray(design.samples["dose"], dtype=float)
    swapped = np.where(doses == 0.1, 1.0, np.where(doses == 1.0, 0.1, doses))
    renamed = Design(design.samples.with_columns({"dose": swapped}), "~ 1")
    other = dose_response(cube, renamed, dose=AXIS, gate={"test": "williams"})
    first = williams.table("cell_type_1", "gate")["p_value"]
    second = other.table("cell_type_1", "gate")["p_value"]
    assert not np.allclose(first, second)


@pytest.mark.parametrize(
    "gate",
    [{"fold_change": 1.2}, {"test": "dunnett"}, {"test": "williams"}],
    ids=["fold change", "dunnett", "williams"],
)
def test_one_treated_dose_ties_with_its_swap(gate: dict[str, Any]) -> None:
    # At the vehicle and one dose in duplicate, every gate tests the dose against
    # the vehicle on both sides, so the run that swaps the two labels passes the
    # true doses' genes, and no p-value can fall below 2/6.
    pairs = series([0.0, 0.1, 1.0], replicates=2, responsive=0.5, seed=3)
    samples = pairs.samples.filter(pairs.samples["dose"] <= 0.1)
    result = dose_response(pairs, Design(samples, "~ 1"), dose=AXIS, gate=gate, rng=1)
    check = permutation_check(result, rng=2)
    entry = check.provenance["groups"][0]
    swap = tuple(1 - label for label in entry["labels"])
    (run,) = entry["tied_runs"]
    assert tuple(entry["assignments"][run - 1]) == swap
    passed = check.table("cell_type_1")["n_genes_passing_gate"][run - 1]
    assert passed == check.summary["n_genes_passing_gate"][0] > 0
    assert check.summary["smallest_possible_p_value"][0] == 2 / 6
    # The sex fixture's one dose under the fold-change gate: the swap in both sexes.
    cube, design = fixture("sex")
    sexes = dose_response(cube, design, dose=AXIS, gate={"fold_change": 1.5})
    found = permutation_check(sexes, rng=1)
    assert found.summary["smallest_possible_p_value"][0] == 2 / 36


def test_designs_without_renamings_that_tie() -> None:
    # A spline of two columns through four dose groups of 2, 2, 1 and 1 samples does
    # not pass through every mean, and the reflection of the doses would swap groups
    # of different sizes: no run ties, among all 179 others.
    cube = series(
        [0.0, 0.1, 1.0, 10.0], replicates=[2, 2, 1, 1], responsive=0.5, seed=3
    )
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS)
    entry = result.provenance["groups"][0]
    assert entry["f_test_df1"]["spline"] == 2 < len(entry["doses"]) - 1
    check = permutation_check(result, n_permutations=179, rng=1)
    assert check.summary["exhaustive"].tolist() == [True]
    assert check.summary["smallest_possible_p_value"][0] == 1 / 180
    # Females with the vehicle twice and the dose once, males the reverse: the
    # labels' counts differ in each sex, so no renaming keeps them, among 9.
    cube = series([0.0, 1.0, 10.0], replicates=3, responsive=0.5, seed=4)
    samples = cube.samples.filter(cube.samples["dose"] <= 1.0)
    sex = {"d0_r1": "F", "d0_r2": "F", "d0_r3": "M", "d1_r1": "F", "d1_r2": "M"}
    labels = [sex.get(s, "M") for s in samples["sample_id"].tolist()]
    design = Design(samples.with_columns({"sex": labels}), "~ sex")
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dose_groups"})
    check = permutation_check(result, rng=1)
    assert check.provenance["groups"][0]["n_assignments"] == 9
    assert check.summary["smallest_possible_p_value"][0] == 1 / 9


def test_dunnett_s_test_of_two_doses_ties_with_their_swap() -> None:
    # Dunnett's test of two treated doses uses no random numbers: each run passes the
    # genes of the run that swaps its two treated labels, and no p-value can fall
    # below 2/90.
    cube, design = fixture("duplicate")
    result = dose_response(cube, design, dose=AXIS, gate={"test": "dunnett"}, rng=1)
    check = permutation_check(result, rng=1)
    labels = labels_of(check, "cell_type_1")
    counts = [int(check.summary["n_genes_passing_gate"][0])]
    counts += check.table("cell_type_1")["n_genes_passing_gate"].tolist()
    by_labels = dict(zip(labels, counts, strict=True))
    swap = {0: 0, 1: 2, 2: 1}
    assert all(
        by_labels[tuple(swap[label] for label in assignment)] == count
        for assignment, count in by_labels.items()
    )
    assert any(counts[1:])
    assert check.summary["smallest_possible_p_value"][0] == 2 / 90


def test_the_reflection_of_symmetric_doses_ties() -> None:
    # The vehicle on a linear axis with doses 0, 5 and 10: the spline of one column
    # is a straight line, which the reflection of the doses turns over without
    # changing its test, so that one run ties and no p-value can fall below 2/90.
    linear = DoseAxis(
        "dose", unit="ug/kg", metric="administered", scale="linear", vehicle="on_axis"
    )
    cube = series([0.0, 5.0, 10.0], replicates=2, responsive=0.4, seed=2)
    result = dose_response(cube, Design(cube.samples, "~ 1"), dose=linear)
    assert result.provenance["groups"][0]["f_test_df1"]["spline"] == 1
    check = permutation_check(result, rng=1)
    entry = check.provenance["groups"][0]
    (run,) = entry["tied_runs"]
    assert tuple(entry["assignments"][run - 1]) == tuple(
        2 - label for label in entry["labels"]
    )
    assert check.summary["smallest_possible_p_value"][0] == 2 / 90


def test_the_summary_by_hand() -> None:
    # The share of runs with genes, the mean per run, (1 + b) / (1 + B), with b the
    # runs that pass at least as many genes as the true doses, and (1 + t) / (1 + B)
    # for the t runs that tie.
    found = _statistics(2, np.array([0, 2, 5, 1]), 1)
    assert found == {
        "share_of_runs_with_passing_genes": 0.75,
        "mean_passing_genes_per_run": 2.0,
        "p_value": 3 / 5,
        "smallest_possible_p_value": 2 / 5,
    }
    # No gene with the true doses: every run passes at least as many.
    assert _statistics(0, np.array([0, 0, 3]), 0)["p_value"] == 1.0
    # More genes than any run: the smallest p-value of B runs.
    found = _statistics(9, np.array([0, 2, 5]), 0)
    assert found["p_value"] == found["smallest_possible_p_value"] == 1 / 4


# Runs made again by the test.


@functools.cache
def strong() -> Cube:
    """The vehicle and two doses in duplicate, half the genes responding: many of the
    89 other assignments keep enough of a response to pass genes."""
    cube, _ = simulate_dose_series(
        [0.0, 0.1, 1.0],
        replicates=2,
        cell_types=1,
        cells=300.0,
        n_genes=80,
        responsive=0.5,
        rng=21,
    )
    return cube


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
    "Dunnett's gate": {"gate": {"test": "dunnett"}, "rng": 3},
    "Dunnett's gate for a rise": {"tests": "dunnett", "direction": "up"},
    "the MCP gate": {
        "gate": {"test": "mcp"},
        "candidates": CandidateShapes(emax=[0.5, 5.0]),
        "rng": 3,
    },
}


@pytest.mark.parametrize("name", list(OPTIONS))
def test_a_run_is_dose_response_on_the_permuted_doses(name: str) -> None:
    # The test rebuilds runs from the doses the provenance records and calls
    # dose_response with the options as given, and the result's seed, so that an
    # option lost on the way from the record, or the wrong samples permuted, changes
    # the counts: every run that passes genes, at least two, and two that pass none.
    cube = strong()
    design = Design(cube.samples, "~ 1")
    arguments = OPTIONS[name]
    result = dose_response(cube, design, dose=AXIS, **arguments)
    check = permutation_check(result, rng=2)
    entry = check.provenance["groups"][0]
    assert entry["samples"] == result.provenance["groups"][0]["fit"]["samples"]
    assert check.provenance["result"]["seed"] == result.settings["seed"]
    runs = check.table("cell_type_1")["n_genes_passing_gate"].tolist()
    passing = [run for run, count in enumerate(runs) if count]
    assert len(passing) >= 2 and len(set(runs)) >= 3
    empty = [run for run, count in enumerate(runs) if not count][:2]
    for run in passing + empty:
        found = again(cube, design, entry, run, **arguments)
        passed = found.table("cell_type_1", "gate")["passes_gate"]
        assert np.count_nonzero(passed) == runs[run]


def test_a_monte_carlo_gate_runs_with_the_result_s_seed() -> None:
    # The Williams-type test of three treated doses integrates its p-values by Monte
    # Carlo, so another seed moves them; every run takes the result's seed, so that
    # the true doses give the result's gate again and a run made again with that
    # seed passes the genes the check counted.
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    gate = {"test": "williams"}
    result = dose_response(cube, design, dose=AXIS, gate=gate, rng=7)
    moved = dose_response(cube, design, dose=AXIS, gate=gate, rng=8)
    first, second = (r.table("one", "gate")["p_value"] for r in (result, moved))
    assert not np.allclose(first, second, rtol=1e-6)
    check = permutation_check(result, n_permutations=2, rng=1)
    for entry in check.provenance["groups"]:
        group = entry["group_id"]
        for run in range(2):
            found = again(cube, design, entry, run, gate=gate, rng=7)
            passed = np.count_nonzero(found.table(group, "gate")["passes_gate"])
            assert passed == check.table(group)["n_genes_passing_gate"][run]


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
    # The reasons of the result's table of the gate.
    for group, reason in zip(summary["group_id"][1:], reasons[1:], strict=True):
        with pytest.raises(KeyError) as raised:
            result.table(group, "gate")
        assert reason == raised.value.args[0]
    assert reasons[1].startswith(
        "the gate was not run in 'few_doses': its test, 'spline'"
    )
    assert reasons[2].startswith("cell type 'no_vehicle' was not tested (refused)")
    assert np.isnan(summary["n_genes_passing_gate"][1:]).all()
    assert summary["n_runs"].tolist() == [2, 0, 0]
    with pytest.raises(KeyError, match="was not checked: the gate was not run"):
        check.table("few_doses")
    for group, why in (
        ("few_doses", "the gate was not run in 'few_doses'"),
        ("no_vehicle", "cell type 'no_vehicle' was not tested"),
    ):
        with pytest.raises(KeyError, match=why):
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


def with_gate(result: DoseResponseResult, **columns: Any) -> DoseResponseResult:
    """The result with columns of each cell type's table of the gate replaced."""
    tables = dict(result.tables)
    for (group, name), table in result.tables.items():
        if name == "gate":
            values = {column: table[column] for column in table.columns}
            for column, change in columns.items():
                values[column] = change(np.asarray(table[column]))
            tables[(group, name)] = Table(values)
    return DoseResponseResult(
        result.summary,
        tables,
        dict(result._fits),
        result.design_report,
        result.provenance,
        cube=result.cube,
        design=result.design,
    )


def test_a_result_its_doses_do_not_give_again_is_refused() -> None:
    cube = responsive()
    design = Design(cube.samples, "~ 1")
    result = dose_response(cube, design, dose=AXIS)
    message = "does not give the result's gate, the same genes with p-values equal"
    # A record that says robust where the fit was not: the true doses run again
    # robustly, and the gate's p-values differ.
    settings = result.provenance["analysis_settings"] | {"robust": True}
    changed = rebuilt(result, provenance={"analysis_settings": settings})
    with pytest.raises(ValueError, match=message):
        permutation_check(changed, n_permutations=2)
    # Another cube under the same record.
    other = series(
        [0.0, 0.1, 1.0, 10.0],
        replicates=3,
        cell_types=["one", "two"],
        n_genes=120,
        seed=22,
    )
    with pytest.raises(ValueError, match=message):
        permutation_check(rebuilt(result, cube=other), n_permutations=2)
    # p-values another machine's arithmetic moves by 1e-12 are the same; 1e-6 is
    # not, nor is a call that differs.
    near = with_gate(result, p_value=lambda p: p * (1 + 1e-12))
    assert permutation_check(near, n_permutations=1, rng=1).summary["checked"].all()
    far = with_gate(result, p_value=lambda p: p * (1 + 1e-6))
    with pytest.raises(ValueError, match=message):
        permutation_check(far, n_permutations=1)
    flipped = with_gate(result, passes_gate=lambda calls: np.r_[~calls[:1], calls[1:]])
    with pytest.raises(ValueError, match=message):
        permutation_check(flipped, n_permutations=1)


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
    # The groups, as dose_response refuses them.
    for groups, error, match in (
        (3, TypeError, "groups must be a group id or a list of group ids"),
        ([3], TypeError, "groups must be group ids"),
        ([], ValueError, "groups is empty"),
        (["one", "one"], ValueError, "group 'one' is listed more than once"),
        ("tow", KeyError, "no group with id 'tow'"),
        ("two", KeyError, "no cell type 'two' was checked"),
    ):
        with pytest.raises(error, match=match):
            permutation_check(result, groups=groups)  # type: ignore[arg-type]
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
        "smallest_possible_p_value",
    )
    count = math.factorial(12) // math.factorial(3) ** 4
    assert summary["n_assignments"].tolist() == [float(count)] * 2
    assert summary["smallest_possible_p_value"].tolist() == [1 / 7] * 2
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
    assert entry["tied_runs"] == [] and entry["n_genes_passing_gate"] > 0
    assert record["result"]["seed"] == result.settings["seed"]
    copied = pickle.loads(pickle.dumps(check))
    assert copied.summary.equals(summary)
    assert copied.table("two").equals(check.table("two"))
    assert copied.provenance == record
    text = " ".join(repr(check).split())
    assert text.startswith(
        "PermutationCheck of the gate (spline test, Benjamini-Hochberg within each "
        "cell type, at 5% FDR): 2 cell types, 2 checked; seed 8"
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
        "PermutationCheck of the gate (TREAT beyond 1.5-fold at each dose, Bonferroni "
        "over the cell type's doses, Benjamini-Hochberg within each cell type, at 5% "
        "FDR):"
    )


def test_runs_fit_the_gate_s_test_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    # Each run fits the gate's test alone, the dose groups' for a fold change, whose
    # fit holds each dose, and passes the genes that a run of the result's own tests
    # passes.
    from cytepool import permutation_checks

    cube = strong()
    design = Design(cube.samples, "~ 1")
    for tests, gate, alone in (
        (("dose_groups", "williams"), None, ["spline"]),
        (("linear_trend",), {"fold_change": 1.5}, ["dose_groups"]),
        (("spline",), {"test": "williams"}, ["williams"]),
    ):
        result = dose_response(cube, design, dose=AXIS, tests=tests, gate=gate, rng=4)
        seen = []

        def recorded(*args: Any, **kwargs: Any) -> DoseResponseResult:
            seen.append(kwargs["tests"])
            return dose_response(*args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(permutation_checks, "dose_response", recorded)
            check = permutation_check(result, n_permutations=3, rng=5)
        assert seen == [alone] * 4
        entry = check.provenance["groups"][0]
        for run in range(3):
            found = again(cube, design, entry, run, tests=tests, gate=gate, rng=4)
            assert check.table("cell_type_1")["n_genes_passing_gate"][
                run
            ] == np.count_nonzero(found.table("cell_type_1", "gate")["passes_gate"])


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
    assert summary["smallest_possible_p_value"].tolist() == [1 / 101] * 2
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
            smallest = float(check.summary["smallest_possible_p_value"][0])
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
    assert check.summary["smallest_possible_p_value"].tolist() == [6 / 90]
    assert (
        "At the vehicle and two doses in duplicate, with the vehicle in a group of "
        "its own as the default axis places it, no p-value of the default gate falls "
        "below 0.05"
    ) in text
