"""Tests for simulated dose series and their truths.

Reference values in ``tests/data/simulate_dose_series`` come from mpmath at 50 digits,
written by ``validation/simulate_dose_series.py``: the true curves, EC50s and
benchmark doses of curves taken from simulated series and of curves whose maximal
change equals the change asked, and the exact moments of the count model. Benchmark
doses are also checked against values derived by hand; the simulated cells and counts
against the closed-form moments of the documented model, which this module computes
itself and checks against mpmath's first; the allocation against counts by hand; and
the cube against what DoseAxis and limma_voom need of a dose series.
"""

import doctest
import math
import pickle
import re
from pathlib import Path
from typing import Any

import numba
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

import cytepool
from cytepool import (
    Design,
    DoseAxis,
    DoseTruth,
    limma_voom,
    simulate_dose_series,
    simulated_dose_series,
)

DATA = Path(__file__).parent / "data" / "simulate_dose_series"
CHAPTER = (
    Path(__file__).resolve().parents[1] / "docs/source/statistics/linear_models.md"
)
NAN = math.nan
PARAMETERS = ("shape", "max_change", "ec50", "slope", "peak_dose", "peak_width")
# The count model's constants, as simulate_dose_series documents them.
CELLS_SD, DEPTH_SD, CELL_DISPERSION, LN2 = 0.3, 0.2, 0.5, math.log(2.0)
FOLD_CHANGE = math.log2(1.1)
DOSES = [0.0, 0.01, 0.1, 1.0, 10.0, 30.0]


def truth_of(rows: list[tuple[Any, ...]], top: float = 10.0) -> DoseTruth:
    """A truth of one cell type, "liver", whose genes g0, g1, ... have the curves
    given as (shape, max_change, ec50, slope, peak_dose, peak_width)."""
    columns = {key: [row[i] for row in rows] for i, key in enumerate(PARAMETERS)}
    table = {
        "feature_id": [f"g{i}" for i in range(len(rows))],
        **columns,
        "counts_per_cell": [1.0] * len(rows),
        "expression_sd": [0.25] * len(rows),
    }
    return DoseTruth(
        [0.0, top / 10.0, top], {"liver": table}, unit="ug/kg", metric="administered"
    )


def fixture_truths() -> tuple[
    dict[str, np.ndarray], dict[float, tuple[DoseTruth, np.ndarray]]
]:
    """The curves of the fixture, and a truth for each top dose with its cases."""
    data = dict(np.load(DATA / "curves.npz"))
    truths = {}
    for top in np.unique(data["top"]):
        members = np.flatnonzero(data["top"] == top)
        rows = [
            (str(data["shape"][i]), *(float(data[key][i]) for key in PARAMETERS[1:]))
            for i in members
        ]
        truths[float(top)] = (truth_of(rows, top=float(top)), members)
    return data, truths


# The true curves, EC50s and benchmark doses.


def test_curves_match_mpmath() -> None:
    data, truths = fixture_truths()
    checked = 0
    for truth, members in truths.values():
        for position, case in enumerate(members):
            points = data["grid_case"] == case
            found = truth.curve("liver", [position], at=data["grid_dose"][points])[0]
            expected = data["grid_value"][points]
            scale = abs(float(data["max_change"][case]))
            if scale == 0.0:
                assert_array_equal(found, 0.0)
            else:
                assert np.max(np.abs(found - expected)) <= 1e-10 * scale, case
            # Exactly 0 at the vehicle, for every shape.
            assert_array_equal(found[data["grid_dose"][points] == 0.0], 0.0)
            checked += points.sum()
    # Twelve or more curves of each of the seven shapes, and flat genes.
    shapes = set(data["shape"].tolist())
    assert shapes == {*simulated_dose_series.SHAPES, "flat"}
    assert checked > 1000


def test_ec50_and_benchmark_doses_match_mpmath_roots() -> None:
    data, truths = fixture_truths()
    placed = np.isin(data["shape"], ["hill", "exponential", "frechet", "step"])
    # The EC50 of a table is the dose of half the maximal change.
    assert_allclose(data["ec50"][placed], data["ec50_root"][placed], rtol=1e-10)
    assert np.isnan(data["ec50_root"][~placed]).all()
    found = np.empty(data["root"].size)
    for truth, members in truths.values():
        for position, case in enumerate(members):
            asked = np.flatnonzero(data["root_case"] == case)
            for j in asked:
                found[j] = truth.benchmark_doses(
                    "liver", [position], change=data["root_change"][j]
                )[0]
    assert_array_equal(np.isnan(found), np.isnan(data["root"]))
    reached = ~np.isnan(data["root"])
    assert_allclose(found[reached], data["root"][reached], rtol=1e-10)
    # Some changes are never reached, and some are reached at |max_change| itself.
    assert 0 < np.isnan(data["root"]).sum() < reached.sum()


def test_benchmark_doses_by_hand() -> None:
    # Cases whose exponents and roots have bases other than 1, so that an exponent
    # of n for 1/n, or a root left out, changes the dose.
    truth = truth_of(
        [
            ("hill", 1.0, 2.0, 2.0, NAN, NAN),
            ("frechet", 4.0, 1.0, 2.0, NAN, NAN),
            ("exponential", -1.0, 3.0, NAN, NAN, NAN),
            ("log_gauss_probit", 1.0, NAN, NAN, 1.0, 0.5),
            ("linear", 1.0, NAN, NAN, NAN, NAN),
            ("power", 1.0, NAN, 2.0, NAN, NAN),
            ("step", 0.5, 3.0, NAN, NAN, NAN),
        ]
    )
    # hill: 2 (0.2 / 0.8)^(1/2) = 1; frechet: log2(4 / 0.25)^(-1/2) = 0.5;
    # exponential: -3 log2(1 - 0.75) = 6; bell: 10^(-0.5 sqrt(2 ln(e^2))) = 0.1;
    # linear: 10 x 0.25 = 2.5; power: 10 x 0.25^(1/2) = 5; step: its EC50.
    changes = [0.2, 0.25, 0.75, math.exp(-2.0), 0.25, 0.25, 0.5]
    assert_allclose(
        truth.benchmark_doses("liver", change=changes),
        [1.0, 0.5, 6.0, 0.1, 2.5, 5.0, 3.0],
        rtol=1e-14,
    )
    # The curves there, and at the EC50, the peak and the top dose.
    assert_allclose(truth.curve("liver", [0], at=[1.0, 2.0])[0], [0.2, 0.5])
    assert_allclose(truth.curve("liver", [1], at=[0.5, 1.0])[0], [0.25, 2.0])
    assert_allclose(truth.curve("liver", [2], at=[3.0, 6.0])[0], [-0.5, -0.75])
    assert_allclose(
        truth.curve("liver", [3], at=[0.1, 1.0, 10**0.5])[0],
        [math.exp(-2.0), 1.0, math.exp(-0.5)],
    )
    assert_allclose(truth.curve("liver", [4], at=[5.0, 10.0])[0], [0.5, 1.0])
    assert_allclose(truth.curve("liver", [5], at=[5.0, 10.0])[0], [0.25, 1.0])
    assert_array_equal(
        truth.curve("liver", [6], at=[2.999999, 3.0, 50.0])[0], [0, 0.5, 0.5]
    )
    # Every curve is 0 at the vehicle.
    assert_array_equal(truth.curve("liver", at=[0.0]), 0.0)


def test_never_reached_is_nan() -> None:
    truth = truth_of(
        [
            ("flat", 0.0, NAN, NAN, NAN, NAN),
            ("hill", 0.5, 1.0, 2.0, NAN, NAN),
            ("exponential", -0.5, 1.0, NAN, NAN, NAN),
            ("frechet", 0.5, 1.0, 2.0, NAN, NAN),
            ("step", 0.5, 1.0, NAN, NAN, NAN),
            ("log_gauss_probit", -0.5, NAN, NAN, 1.0, 0.5),
            ("linear", 0.5, NAN, NAN, NAN, NAN),
            ("power", -0.5, NAN, 2.0, NAN, NAN),
        ]
    )
    # At a change equal to |max_change|: a plateau is approached and never reached,
    # while a step and a peak reach it; lines and powers go on beyond the top dose.
    assert_array_equal(
        truth.benchmark_doses("liver", change=0.5),
        [NAN, NAN, NAN, NAN, 1.0, 1.0, 10.0, 10.0],
    )
    beyond = truth.benchmark_doses("liver", change=0.5 * (1.0 + 1e-12))
    assert_array_equal(np.isnan(beyond), [True] * 6 + [False, False])
    below = truth.benchmark_doses("liver", change=0.5 * (1.0 - 1e-9))
    assert_array_equal(np.isnan(below), [True] + [False] * 7)
    # Twice |max_change|: 20 for the line, 10 x 2^(1/2) for the power.
    assert_allclose(
        truth.benchmark_doses("liver", [6, 7], change=1.0), [20.0, 10.0 * 2**0.5]
    )


def test_every_benchmark_dose_is_the_lowest_crossing() -> None:
    _, truth = simulate_dose_series(DOSES, replicates=1, responsive=1.0, rng=21)
    for cell_type in truth.cell_types:
        shape = truth.table(cell_type)["shape"]
        for change in (0.2, FOLD_CHANGE, 1.0):
            doses = truth.benchmark_doses(cell_type, change=change)
            reached = np.flatnonzero(np.isfinite(doses))
            # Each gene's curve at its own benchmark dose, and just below it.
            at = np.diagonal(truth.curve(cell_type, reached, at=doses[reached]))
            before = np.diagonal(
                truth.curve(cell_type, reached, at=0.999 * doses[reached])
            )
            steps = shape[reached] == "step"
            assert_allclose(np.abs(at[~steps]), change, rtol=1e-12)
            assert np.all(np.abs(at[steps]) >= change)
            assert np.all(np.abs(before) < change)
            assert reached.size >= 500


# The moments of the count model.


def stirling(i: int, j: int) -> float:
    """A Stirling number of the second kind."""
    total = sum((-1) ** (j - t) * math.comb(j, t) * t**i for t in range(j + 1))
    return total / math.factorial(j)


def cells_raw(m: float) -> list[float]:
    """E[n^i], i = 0 to 4, of n ~ Poisson(m exp(0.3 x - 0.045)), x standard normal."""
    rate = [m**j * math.exp(CELLS_SD**2 * j * (j - 1) / 2) for j in range(5)]
    return [sum(stirling(i, j) * rate[j] for j in range(i + 1)) for i in range(5)]


def central(raw: list[float]) -> tuple[float, float, float]:
    """Mean, variance and fourth central moment from raw moments 1 to 4."""
    m1, m2, m3, m4 = raw
    return m1, m2 - m1**2, m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4


def count_moments(m: float, a: float, sigma: float) -> tuple[float, float, float]:
    """The mean, variance and fourth central moment of a count with expected cells m,
    a = counts_per_cell 2^f and a biological SD sigma in log2 units: the factorial
    moments E[(Y)_k] = a^k E[e^k] E[2^(k b)] E[prod over j < k of (n + 0.5 j)]."""
    n_raw = cells_raw(m)
    factorial = []
    for k in range(1, 5):
        coefficients = [1.0]
        for j in range(k):
            shifted = [0.0, *coefficients]
            for i, value in enumerate(coefficients):
                shifted[i] += value * CELL_DISPERSION * j
            coefficients = shifted
        cells = sum(c * n_raw[i] for i, c in enumerate(coefficients))
        depth = math.exp(DEPTH_SD**2 * k * (k - 1) / 2)
        effect = math.exp((k * sigma * LN2) ** 2 / 2)
        factorial.append(a**k * depth * effect * cells)
    f1, f2, f3, f4 = factorial
    return central([f1, f2 + f1, f3 + 3 * f2 + f1, f4 + 6 * f3 + 7 * f2 + f1])


def donor_covariance(
    m1: float, m2: float, a1: float, a2: float, sigma: float, tau: float
) -> float:
    """The covariance of one donor's counts at two doses."""
    shared = math.exp(2 * (tau * LN2) ** 2) - math.exp((tau * LN2) ** 2)
    return a1 * a2 * m1 * m2 * math.exp((sigma * LN2) ** 2) * shared


def z_scores(
    values: np.ndarray, moments: tuple[float, float, float]
) -> tuple[float, float]:
    """z-scores of a sample's mean and variance against exact moments; the variance's
    standard error from the exact fourth central moment."""
    n = values.size
    mean, variance, fourth = moments
    z_mean = (values.mean() - mean) / math.sqrt(variance / n)
    spread = math.sqrt((fourth - variance**2 * (n - 3) / (n - 1)) / n)
    return z_mean, (values.var(ddof=1) - variance) / spread


def test_the_moment_formulas_match_mpmath() -> None:
    data = dict(np.load(DATA / "moments.npz"))
    for m, a, sigma, expected in zip(
        data["m"], data["a"], data["sigma"], data["count_moments"]
    ):
        assert_allclose(count_moments(m, a, sigma), expected, rtol=1e-10)
    for m, expected in zip(data["cells_m"], data["cells_moments"]):
        assert_allclose(central(cells_raw(m)[1:]), expected, rtol=1e-10)
    for *values, expected in zip(
        *(data[f"cov_{key}"] for key in ("m1", "m2", "a1", "a2", "sigma", "tau")),
        data["covariance"],
    ):
        assert_allclose(donor_covariance(*values), expected, rtol=1e-10)


def test_cells_and_counts_have_the_moments_of_the_model() -> None:
    doses = [0.0, 0.1, 1.0, 10.0]
    factors = {"cell_type_1": [1.0] * 4, "cell_type_2": [1.0, 1.0, 0.7, 0.4]}
    cube, truth = simulate_dose_series(
        doses,
        replicates=4000,
        cells=[30.0, 300.0],
        n_genes=6,
        responsive=0.5,
        composition={"cell_type_2": factors["cell_type_2"]},
        rng=22,
    )
    dose = cube.samples["dose"]
    z = []
    for k, (cell_type, cells) in enumerate(zip(truth.cell_types, (30.0, 300.0))):
        table = truth.table(cell_type)
        changes = truth.curve(cell_type, at=doses)
        counts = cube.layer("counts", cell_type).astype(np.float64)
        for j, d in enumerate(doses):
            rows = dose == d
            m = cells * factors[cell_type][j]
            z += z_scores(
                cube.n_cells[rows, k].astype(np.float64), central(cells_raw(m)[1:])
            )
            for g in range(6):
                a = table["counts_per_cell"][g] * 2.0 ** changes[g, j]
                moments = count_moments(m, a, table["expression_sd"][g])
                z += z_scores(counts[rows, g], moments)
    assert len(z) == 2 * 4 * (1 + 6) * 2
    assert np.max(np.abs(z)) <= 4.0


def test_donor_effects_are_shared_across_doses() -> None:
    doses = [0.0, 1.0, 10.0]
    options = {"cell_types": 1, "cells": 300.0, "n_genes": 6, "responsive": 0.5}
    crossed, truth = simulate_dose_series(
        doses, replicates=3000, donor_sd=0.25, rng=23, **options
    )
    donors = crossed.samples["donor"].reshape(3, 3000)
    assert_array_equal(donors[0], donors[1])
    assert_array_equal(donors[0], donors[2])
    assert donors[0, 0] == "donor_0001" and len(set(donors[0])) == 3000
    table = truth.table("cell_type_1")
    changes = truth.curve("cell_type_1", at=doses)
    counts = (
        crossed.layer("counts", "cell_type_1").astype(np.float64).reshape(3, 3000, 6)
    )
    # Each dose's variance has the table's expression_sd, biological and donor.
    sd_factor = table["expression_sd"] / math.hypot(0.25, 0.25)
    z = []
    for j in range(3):
        for g in range(6):
            a = table["counts_per_cell"][g] * 2.0 ** changes[g, j]
            moments = count_moments(300.0, a, table["expression_sd"][g])
            z.append(z_scores(counts[j, :, g], moments)[1])
    # The vehicle and each dose covary within donors as one donor effect gives.
    for j in (1, 2):
        for g in range(6):
            a1 = table["counts_per_cell"][g]
            a2 = a1 * 2.0 ** changes[g, j]
            sigma, tau = 0.25 * sd_factor[g], 0.25 * sd_factor[g]
            expected = donor_covariance(300.0, 300.0, a1, a2, sigma, tau)
            products = (counts[0, :, g] - counts[0, :, g].mean()) * (
                counts[j, :, g] - counts[j, :, g].mean()
            )
            spread = products.std(ddof=1) / math.sqrt(products.size)
            z.append((products.sum() / (products.size - 1) - expected) / spread)
    assert np.max(np.abs(z)) <= 4.0
    # One animal per replicate: unique donors, and no covariance between doses.
    nested, _ = simulate_dose_series(doses, replicates=3000, rng=23, **options)
    assert len(set(nested.samples["donor"])) == 9000
    # A crossed design without donor variance: shared ids, no covariance.
    flat_donors, _ = simulate_dose_series(
        doses, replicates=3000, donor_sd=0.0, rng=23, **options
    )
    assert_array_equal(flat_donors.samples["donor"].reshape(3, 3000)[1], donors[0])
    for cube in (nested, flat_donors):
        values = (
            cube.layer("counts", "cell_type_1").astype(np.float64).reshape(3, 3000, 6)
        )
        for g in range(6):
            products = (values[0, :, g] - values[0, :, g].mean()) * (
                values[2, :, g] - values[2, :, g].mean()
            )
            spread = products.std(ddof=1) / math.sqrt(products.size)
            assert abs(products.mean() / spread) <= 4.0


# Allocation and placement.


def test_allocation_of_genes_shapes_changes_and_signs() -> None:
    _, truth = simulate_dose_series(DOSES, responsive=[0.2, 0.25], rng=24)
    first, second = (truth.table(c) for c in truth.cell_types)
    assert (first["shape"] != "flat").sum() == 200
    assert (second["shape"] != "flat").sum() == 250
    cells: dict[tuple[str, float], int] = {}
    for shape, change in zip(first["shape"].tolist(), first["max_change"].tolist()):
        cells[shape, change] = cells.get((shape, change), 0) + 1
    # 200 responsive genes: 60 hill, 30 exponential, 10 linear, 20 power, 40 bells,
    # 20 frechet and 20 steps; within each, the changes in turn, and rises and falls
    # in turn within each change, the first a rise.
    for shape, per_change in (
        ("hill", [10, 10, 10, 10, 10, 10]),
        ("exponential", [5, 5, 5, 5, 5, 5]),
        ("linear", [2, 2, 2, 1, 2, 1]),
        ("power", [4, 3, 4, 3, 3, 3]),
        ("log_gauss_probit", [7, 7, 7, 6, 7, 6]),
        ("frechet", [4, 3, 4, 3, 3, 3]),
        ("step", [4, 3, 4, 3, 3, 3]),
    ):
        signed = [
            cells.get((shape, sign * change), 0)
            for change in (0.5, 1.0, 2.0)
            for sign in (1.0, -1.0)
        ]
        assert signed == per_change, shape
    # Seven genes by the largest remainder: 2.1, 1.05, 0.35, 0.7, 1.4, 0.7, 0.7
    # give 2, 1, 0, 1, 1, 1, 1; floors alone would leave three genes out.
    _, small = simulate_dose_series(DOSES, n_genes=7, responsive=1.0, rng=25)
    shapes = small.table("cell_type_1")["shape"].tolist()
    assert sorted(shapes) == sorted(
        ["hill", "hill", "exponential", "power", "log_gauss_probit", "frechet", "step"]
    )
    # A share that rounds to no gene, a half rounded up, and a series without
    # responses, whose benchmark doses are all NaN.
    for n_genes, share, expected in ((1, 0.2, 0), (3, 0.5, 2), (50, 0.0, 0)):
        _, rounded = simulate_dose_series(
            DOSES, n_genes=n_genes, responsive=share, rng=26
        )
        table = rounded.table("cell_type_1")
        assert (table["shape"] != "flat").sum() == expected
        if expected == 0:
            assert np.isnan(table["bmd_expression_sd"]).all()
            assert np.isnan(table["bmd_fold_change"]).all()
    # A larger share adds genes and keeps the others.
    _, fewer = simulate_dose_series(DOSES, responsive=0.1, rng=27)
    _, more = simulate_dose_series(DOSES, responsive=0.3, rng=27)
    kept = fewer.table("cell_type_1")["shape"] != "flat"
    assert np.all(more.table("cell_type_1")["shape"][kept] != "flat")


def test_placement_puts_a_tenth_below_and_above_the_doses() -> None:
    placed = {"hill": 0.25, "exponential": 0.25, "frechet": 0.25, "step": 0.25}
    _, truth = simulate_dose_series(
        DOSES,
        replicates=1,
        cell_types=1,
        n_genes=20_000,
        responsive=1.0,
        shapes=placed,
        rng=28,
    )
    table = truth.table("cell_type_1")
    fold_change = table["bmd_fold_change"]
    spread = math.sqrt(0.1 * 0.9 / fold_change.size)
    assert abs(np.mean(fold_change < 0.01) - 0.1) <= 4 * spread
    assert abs(np.mean(fold_change > 30.0) - 0.1) <= 4 * spread
    # Within the doses, log-uniform: a fifth of the span holds a fifth of the rest.
    inside = np.log10(fold_change[(fold_change >= 0.01) & (fold_change <= 30.0)])
    lower = np.log10(0.01) + (np.log10(30.0) - np.log10(0.01)) / 5
    assert abs(np.mean(inside < lower) - 0.2) <= 4 * math.sqrt(0.16 / inside.size)
    # Slopes log-uniform from 0.5 to 4: a third lie below 1.
    for shape in ("hill", "frechet"):
        slopes = table["slope"][table["shape"] == shape]
        assert 0.5 <= slopes.min() and slopes.max() <= 4.0
        below_one = np.mean(slopes < 1.0)
        assert abs(below_one - 1 / 3) <= 4 * math.sqrt(2 / 9 / slopes.size)
    _, bells = simulate_dose_series(
        DOSES,
        replicates=1,
        cell_types=1,
        n_genes=2000,
        responsive=1.0,
        shapes={"log_gauss_probit": 0.5, "power": 0.5},
        rng=29,
    )
    table = bells.table("cell_type_1")
    bell = table["shape"] == "log_gauss_probit"
    peaks = table["peak_dose"][bell]
    assert np.all((peaks >= 0.01) & (peaks <= 30.0))
    # Log-uniform over the doses: half the peaks lie below their log midpoint.
    midpoint = math.sqrt(0.01 * 30.0)
    assert abs(np.mean(peaks < midpoint) - 0.5) <= 4 * math.sqrt(0.25 / peaks.size)
    # The median log10 spacing of 0.01, 0.1, 1, 10 and 30 is 1.
    widths = table["peak_width"][bell]
    assert 0.75 <= widths.min() and widths.max() <= 1.5
    powers = table["slope"][table["shape"] == "power"]
    assert 1.0 <= powers.min() and powers.max() <= 4.0


def test_truths_stay_when_the_design_changes() -> None:
    curve_columns = ("feature_id", *PARAMETERS, "counts_per_cell", "bmd_fold_change")
    base_cube, base = simulate_dose_series(DOSES, rng=30)
    _, other = simulate_dose_series(
        DOSES,
        replicates=[2, 4, 3, 3, 2, 5],
        cells=[40.0, 900.0],
        composition={"cell_type_1": [1.0, 1.0, 0.9, 0.5, 0.2, 0.1]},
        biological_sd=0.4,
        donor_sd=0.3,
        rng=30,
    )
    for cell_type in base.cell_types:
        before, after = base.table(cell_type), other.table(cell_type)
        for key in curve_columns:
            assert_array_equal(before[key], after[key])
        # The genes' SD factors stay, scaled by the SDs given.
        assert_allclose(
            after["expression_sd"] / math.hypot(0.4, 0.3),
            before["expression_sd"] / 0.25,
            rtol=1e-14,
        )
    # The cells per profile come from streams of their own, apart from the genes.
    other_genes, _ = simulate_dose_series(
        DOSES, n_genes=300, shapes={"step": 1.0}, responsive=0.5, rng=30
    )
    assert_array_equal(other_genes.n_cells, base_cube.n_cells)
    # A third cell type keeps the first two, counts included.
    larger_cube, larger = simulate_dose_series(
        DOSES, cell_types=["cell_type_1", "cell_type_2", "cell_type_3"], rng=30
    )
    for cell_type in base.cell_types:
        assert base.table(cell_type).equals(larger.table(cell_type))
        assert_array_equal(
            base_cube.layer("counts", cell_type), larger_cube.layer("counts", cell_type)
        )
    # Doses in other units: EC50s, peaks and benchmark doses scale, counts stay.
    scaled_cube, scaled = simulate_dose_series(
        [1000.0 * dose for dose in DOSES], unit="ng/kg", rng=30
    )
    for cell_type in base.cell_types:
        before, after = base.table(cell_type), scaled.table(cell_type)
        for key in ("ec50", "peak_dose", "bmd_expression_sd", "bmd_fold_change"):
            assert_allclose(after[key], 1000.0 * before[key], rtol=1e-12)
        assert_array_equal(
            base_cube.layer("counts", cell_type), scaled_cube.layer("counts", cell_type)
        )


def fingerprint(**options: Any) -> tuple[str, Any]:
    cube, truth = simulate_dose_series(DOSES, n_genes=200, **options)
    return cube.fingerprint(), truth


def test_the_same_seed_gives_the_same_series() -> None:
    seeded, truth = fingerprint(rng=31)
    again, other = fingerprint(rng=np.random.default_rng(31))
    assert seeded == again
    assert all(truth.table(c).equals(other.table(c)) for c in truth.cell_types)
    assert fingerprint(rng=32)[0] != seeded
    # The series follows the generator's state, whatever seed sequence built it:
    # jumped bit generators, and a state copied into another generator.
    jumped = [
        fingerprint(rng=np.random.Generator(np.random.PCG64(1).jumped()))[0]
        for _ in range(2)
    ]
    assert jumped[0] == jumped[1]
    source = np.random.default_rng(3)
    copy = np.random.Generator(np.random.PCG64())
    copy.bit_generator.state = source.bit_generator.state
    assert fingerprint(rng=copy)[0] == fingerprint(rng=source)[0]
    # A generator moves on: a second call, or draws before the call, give another.
    generator = np.random.default_rng(33)
    assert fingerprint(rng=generator)[0] != fingerprint(rng=generator)[0]
    drawn = np.random.default_rng(31)
    drawn.random()
    assert fingerprint(rng=drawn)[0] != seeded
    # Threads change nothing; the simulator has no compiled code, and this guards
    # against compiled code added later.
    previous = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        single = fingerprint(rng=31)[0]
    finally:
        numba.set_num_threads(previous)
    assert single == seeded


# The cube.


def test_the_cube_reads_as_a_dose_series() -> None:
    cube, truth = simulate_dose_series(
        [0, 0.1, 1, 10],
        replicates=[3, 2, 2, 1],
        cell_types=["hepatocyte", "stellate"],
        n_genes=12,
        composition={"stellate": [1.0, 1.0, 0.5, 0.0]},
        unit="mg/kg",
        rng=34,
    )
    assert cube.groups["group_id"].tolist() == ["hepatocyte", "stellate"]
    assert cube.samples.columns == ("sample_id", "dose", "dose_group", "donor")
    assert cube.samples["sample_id"].tolist() == [
        "d0_r1", "d0_r2", "d0_r3", "d1_r1", "d1_r2", "d2_r1", "d2_r2", "d3_r1"
    ]  # fmt: skip
    assert cube.samples["dose"].dtype == np.float64
    assert_array_equal(cube.samples["dose"], [0, 0, 0, 0.1, 0.1, 1, 1, 10])
    assert cube.samples["dose_group"].tolist() == ["d0"] * 3 + ["d1"] * 2 + [
        "d2"
    ] * 2 + ["d3"]
    assert cube.samples["donor"].tolist() == [f"donor_{i}" for i in range(1, 9)]
    assert cube.features["feature_id"].tolist()[:2] == ["gene_01", "gene_02"]
    assert cube.layer("counts", "hepatocyte").dtype == np.uint32
    assert cube.n_cells.dtype == np.int64 and cube.n_cells.shape == (8, 2)
    # A factor of 0 leaves the stellate profile at the top dose without cells.
    assert cube.n_cells[-1, 1] == 0
    assert_array_equal(cube.layer("counts", "stellate")[-1], 0)
    assert truth.cell_types == ("hepatocyte", "stellate")
    assert truth.doses == (0.0, 0.1, 1.0, 10.0)
    assert (truth.unit, truth.metric) == ("mg/kg", "administered")
    assert_array_equal(
        truth.table("stellate")["feature_id"], cube.features["feature_id"]
    )
    construction = cube.provenance["construction"]
    assert set(construction) == {
        "function", "cytepool_version", "numpy_version", "doses", "replicates",
        "cell_types", "cells", "n_genes", "responsive", "shapes", "max_changes",
        "biological_sd", "donor_sd", "composition", "unit", "metric", "seed",
    }  # fmt: skip
    assert construction["seed"] == 34 and construction["replicates"] == [3, 2, 2, 1]
    assert construction["composition"]["stellate"] == [1.0, 1.0, 0.5, 0.0]
    assert "Built by simulate_dose_series" in cube.history()
    # The dose axis of PR 3 reads the sample table as it stands.
    axis = DoseAxis("dose", unit=truth.unit, metric=truth.metric)
    basis = axis.spline_basis(cube.samples, df=2)
    columns = axis.design_columns(cube.samples, basis)
    assert_array_equal(columns["dose_treated"], [0, 0, 0, 1, 1, 1, 1, 1])
    # Dose groups and replicates padded with zeros sort in dose order.
    ladder = [0.0, *np.geomspace(0.001, 100.0, 11).round(6)]
    wide, _ = simulate_dose_series(
        ladder, replicates=10, n_genes=3, donor_sd=0.1, rng=35
    )
    groups = wide.samples["dose_group"].tolist()
    assert groups[0] == "d00" and sorted(set(groups)) == list(dict.fromkeys(groups))
    assert wide.samples["sample_id"][9] == "d00_r10"
    assert wide.samples["donor"][10] == "donor_01"


def test_estimated_changes_follow_the_truth() -> None:
    # Strong Hill responses: limma-voom's top dose against the vehicle recovers the
    # true change of genes with at least 10 expected counts at both, and calls flat
    # genes at about the 5% rate.
    cube, truth = simulate_dose_series(
        DOSES,
        replicates=6,
        cell_types=1,
        cells=500.0,
        shapes={"hill": 1.0},
        max_changes=(2.0,),
        rng=11,
    )
    levels = {"dose_group": [f"d{k}" for k in range(6)]}
    result = limma_voom(
        cube,
        Design(cube.samples, "~ dose_group", levels=levels),
        contrasts=["dose_groupd5"],
    )
    tested = result.table("cell_type_1", "dose_groupd5")
    table = truth.table("cell_type_1")
    rows = np.searchsorted(table["feature_id"], tested["feature_id"])
    assert_array_equal(table["feature_id"][rows], tested["feature_id"])
    true_change = truth.curve("cell_type_1", rows, at=[30.0])[:, 0]
    expected = table["counts_per_cell"][rows] * 500.0
    counted = (expected >= 10.0) & (expected * 2.0**true_change >= 10.0)
    responsive = table["shape"][rows] != "flat"
    chosen = responsive & counted
    slope = np.polyfit(true_change[chosen], tested["log2_fold_change"][chosen], 1)[0]
    correlation = np.corrcoef(true_change[chosen], tested["log2_fold_change"][chosen])
    assert chosen.sum() > 100
    assert 0.95 <= slope <= 1.05
    assert correlation[0, 1] > 0.98
    assert 0.02 <= np.mean(tested["p_value"][~responsive] < 0.05) <= 0.08


# The truth as a record.


def test_a_truth_survives_its_tables() -> None:
    _, truth = simulate_dose_series(DOSES, n_genes=300, rng=36)
    rebuilt = DoseTruth(
        truth.doses,
        {c: truth.table(c) for c in truth.cell_types},
        unit=truth.unit,
        metric=truth.metric,
    )
    grid = np.geomspace(0.001, 100.0, 40)
    for cell_type in truth.cell_types:
        assert rebuilt.table(cell_type).equals(truth.table(cell_type))
        assert_array_equal(
            rebuilt.curve(cell_type, at=grid), truth.curve(cell_type, at=grid)
        )
    # From plain columns too, with the derived columns left out.
    plain = {
        c: {
            key: truth.table(c)[key].tolist()
            for key in ("feature_id", *PARAMETERS, "counts_per_cell", "expression_sd")
        }
        for c in truth.cell_types
    }
    again = DoseTruth(list(truth.doses), plain, unit=truth.unit, metric=truth.metric)
    assert all(again.table(c).equals(truth.table(c)) for c in truth.cell_types)
    unpickled = pickle.loads(pickle.dumps(truth))
    assert all(unpickled.table(c).equals(truth.table(c)) for c in truth.cell_types)
    assert (unpickled.doses, unpickled.unit) == (truth.doses, truth.unit)
    with pytest.raises(AttributeError, match="immutable"):
        truth.unit = "mg/kg"  # type: ignore[misc]
    assert " ".join(repr(truth).split()) == (
        "DoseTruth: 2 cell types ('cell_type_1' and 'cell_type_2') x 300 genes; 6 "
        "doses from 0 to 30 ug/kg (administered)"
    )


def test_genes_are_chosen_as_a_cube_chooses_features() -> None:
    truth = truth_of(
        [
            ("hill", 1.0, 1.0, 1.0, NAN, NAN),
            ("linear", -2.0, NAN, NAN, NAN, NAN),
            ("flat", 0.0, NAN, NAN, NAN, NAN),
        ]
    )
    everything = truth.curve("liver", at=[5.0])
    assert_array_equal(truth.curve("liver", ["g2", "g0"], at=[5.0]), everything[[2, 0]])
    assert_array_equal(truth.curve("liver", "g1", at=[5.0]), everything[[1]])
    assert_array_equal(
        truth.curve("liver", [True, False, True], at=[5.0]), everything[[0, 2]]
    )
    assert_array_equal(truth.curve("liver", [-1], at=[5.0]), everything[[2]])
    assert truth.curve("liver", [], at=[5.0]).shape == (0, 1)
    assert truth.curve("liver", at=[]).shape == (3, 0)
    assert_allclose(
        truth.benchmark_doses("liver", ["g1", "g0"], change=[1.0, 0.5]), [5.0, 1.0]
    )
    with pytest.raises(ValueError, match="repeats 'g0'"):
        truth.curve("liver", ["g0", "g0"], at=[1.0])
    with pytest.raises(KeyError, match="no gene with id 'g9'"):
        truth.benchmark_doses("liver", ["g9"], change=0.5)


# Arguments.

REFUSED_SERIES = [
    ({"doses": "0, 1, 10"}, TypeError, "doses must hold real numbers"),
    ({"doses": [True, False]}, TypeError, "doses must hold real numbers"),
    ({"doses": [[0.0, 1.0, 10.0]]}, ValueError, "1-D"),
    ({"doses": [0.0, 1.0, math.inf]}, ValueError, "finite and at least 0"),
    ({"doses": [0.0, -1.0, 10.0]}, ValueError, "finite and at least 0"),
    ({"doses": [0.1, 1.0, 10.0]}, ValueError, "start with 0, the vehicle"),
    ({"doses": []}, ValueError, "start with 0, the vehicle"),
    ({"doses": [0.0, 10.0, 1.0]}, ValueError, "increase strictly"),
    ({"doses": [0.0, 1.0, 1.0, 10.0]}, ValueError, "increase strictly"),
    ({"doses": [0.0, 1.0]}, ValueError, "at least two doses above 0"),
    ({"doses": [0.0, 0.3, 0.1 * 3]}, ValueError, "differ by less than 1e-8"),
    ({"replicates": 0}, ValueError, "replicates must be at least 1"),
    ({"replicates": 2.0}, TypeError, "replicates must be an integer"),
    ({"replicates": [3, 3]}, ValueError, "one value or one per dose \\(6\\), got 2"),
    (
        {"replicates": [3, 3, 3, 0, 3, 3]},
        ValueError,
        "replicates\\[3\\] must be at least 1",
    ),
    ({"cell_types": 0}, ValueError, "cell_types must be at least 1"),
    ({"cell_types": "liver"}, TypeError, "number of cell types or a list"),
    ({"cell_types": []}, ValueError, "at least one cell type"),
    ({"cell_types": ["a", ""]}, ValueError, "non-empty str"),
    ({"cell_types": ["a", "a"]}, ValueError, "repeats 'a'"),
    ({"cells": 0.0}, ValueError, "cells must be positive and finite"),
    ({"cells": [100.0, math.nan]}, ValueError, "cells\\[1\\] must be positive"),
    ({"cells": [100.0]}, ValueError, "one per cell type \\(2\\)"),
    ({"n_genes": 0}, ValueError, "n_genes must be at least 1"),
    ({"n_genes": 10.0}, TypeError, "n_genes must be an integer"),
    ({"responsive": 1.5}, ValueError, "share from 0 to 1"),
    ({"responsive": "all"}, TypeError, "responsive must be a real number"),
    ({"shapes": ["hill"]}, TypeError, "mapping of shape name to share"),
    (
        {"shapes": {"hil": 1.0}},
        ValueError,
        "unknown shape 'hil' \\(did you mean 'hill'",
    ),
    ({"shapes": {"weibull": 1.0}}, ValueError, "unknown shape 'weibull'"),
    ({"shapes": {"hill": 0.5, "step": 0.4}}, ValueError, "sum to 1"),
    ({"shapes": {"hill": 1.5, "step": -0.5}}, ValueError, "non-negative"),
    ({"max_changes": []}, ValueError, "at least one value"),
    ({"max_changes": [0.5, 0.1]}, ValueError, "above log2\\(1.1\\)"),
    ({"max_changes": [-1.0]}, ValueError, "given as a size"),
    ({"biological_sd": 0.0}, ValueError, "biological_sd must be positive"),
    ({"donor_sd": -0.1}, ValueError, "donor_sd must be non-negative"),
    ({"donor_sd": "high"}, TypeError, "donor_sd must be a real number"),
    ({"composition": [1.0]}, TypeError, "mapping of cell type"),
    (
        {"composition": {"cell_type_3": [1.0] * 6}},
        ValueError,
        "unknown cell type 'cell_type_3'",
    ),
    ({"composition": {"cell_type_1": 0.5}}, TypeError, "one factor per dose"),
    ({"composition": {"cell_type_1": [1.0] * 5}}, ValueError, "one per dose \\(6\\)"),
    (
        {"composition": {"cell_type_1": [1.0, -1.0, 1, 1, 1, 1]}},
        ValueError,
        "non-negative",
    ),
    ({"composition": {"cell_type_1": [0.0] * 6}}, ValueError, "0 at every dose"),
    ({"unit": 1}, TypeError, "unit must be a str"),
    ({"unit": " "}, ValueError, "unit must not be empty"),
    ({"metric": "given"}, ValueError, "metric must be one of"),
    ({"rng": "seed"}, TypeError, "rng must be a numpy.random.Generator"),
    ({"rng": -1}, ValueError, "non-negative seed"),
]


@pytest.mark.parametrize(("change", "error", "message"), REFUSED_SERIES)
def test_arguments_are_refused_with_a_fix(
    change: dict[str, Any], error: type[Exception], message: str
) -> None:
    arguments: dict[str, Any] = {"doses": DOSES, "n_genes": 20, "rng": 1, **change}
    with pytest.raises(error, match=message):
        simulate_dose_series(arguments.pop("doses"), **arguments)


def good_table(**changes: Any) -> dict[str, Any]:
    table = {
        "feature_id": ["a", "b"],
        "shape": ["hill", "flat"],
        "max_change": [1.0, 0.0],
        "ec50": [1.0, NAN],
        "slope": [1.0, NAN],
        "peak_dose": [NAN, NAN],
        "peak_width": [NAN, NAN],
        "counts_per_cell": [1.0, 1.0],
        "expression_sd": [0.25, 0.25],
    }
    return {**table, **changes}


REFUSED_TRUTHS = [
    ({"curves": []}, TypeError, "mapping of cell type to its table"),
    ({"curves": {}}, ValueError, "at least one cell type"),
    ({"curves": {1: good_table()}}, TypeError, "non-empty str"),
    ({"curves": {"x": {"feature_id": ["a"]}}}, ValueError, "lacks the columns 'shape'"),
    ({"curves": {"x": good_table(feature_id=["a", "a"])}}, ValueError, "repeats 'a'"),
    ({"curves": {"x": good_table(feature_id=[1, 2])}}, TypeError, "non-empty str"),
    (
        {"curves": {"x": good_table(shape=["hil", "flat"])}},
        ValueError,
        "did you mean 'hill'",
    ),
    (
        {"curves": {"x": good_table(ec50=[0.0, NAN])}},
        ValueError,
        "ec50 of a must be finite and above 0",
    ),
    (
        {"curves": {"x": good_table(slope=[NAN, NAN])}},
        ValueError,
        "slope of a must be finite",
    ),
    (
        {"curves": {"x": good_table(peak_dose=[1.0, NAN])}},
        ValueError,
        "peak_dose of a must be NaN",
    ),
    (
        {"curves": {"x": good_table(ec50=[1.0, 2.0])}},
        ValueError,
        "ec50 of b must be NaN",
    ),
    (
        {"curves": {"x": good_table(max_change=[0.0, 0.0])}},
        ValueError,
        "max_change of a must be finite and not 0",
    ),
    (
        {"curves": {"x": good_table(max_change=[1.0, 0.5])}},
        ValueError,
        "max_change of b must be 0",
    ),
    (
        {"curves": {"x": good_table(counts_per_cell=[1.0, 0.0])}},
        ValueError,
        "counts_per_cell of b",
    ),
    (
        {"curves": {"x": good_table(expression_sd=[math.inf, 1.0])}},
        ValueError,
        "expression_sd of a",
    ),
    (
        {"curves": {"x": good_table(max_change=["1", "0"])}},
        TypeError,
        "must hold real numbers",
    ),
    (
        {"curves": {"x": good_table(), "y": good_table(feature_id=["b", "a"])}},
        ValueError,
        "the genes of curves\\['x'\\], in the same order",
    ),
    ({"doses": [0.0, 1.0]}, ValueError, "at least two doses above 0"),
    ({"unit": ""}, ValueError, "unit must not be empty"),
    ({"metric": "dose"}, ValueError, "metric must be one of"),
]


@pytest.mark.parametrize(("change", "error", "message"), REFUSED_TRUTHS)
def test_truths_refuse_tables_they_cannot_read(
    change: dict[str, Any], error: type[Exception], message: str
) -> None:
    arguments: dict[str, Any] = {
        "doses": [0.0, 1.0, 10.0],
        "curves": {"x": good_table()},
        "unit": "uM",
        "metric": "nominal",
        **change,
    }
    with pytest.raises(error, match=message):
        DoseTruth(arguments.pop("doses"), arguments.pop("curves"), **arguments)


def test_methods_refuse_what_they_cannot_read() -> None:
    truth = DoseTruth(
        [0.0, 1.0, 10.0], {"liver": good_table()}, unit="uM", metric="nominal"
    )
    with pytest.raises(KeyError, match="no cell type 'livr' \\(did you mean 'liver'"):
        truth.table("livr")
    with pytest.raises(TypeError, match="cell_type must be a str"):
        truth.curve(0, at=[1.0])  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at must be finite and at least 0"):
        truth.curve("liver", at=[-1.0])
    with pytest.raises(ValueError, match="1-D"):
        truth.curve("liver", at=1.0)
    with pytest.raises(TypeError, match="at must hold real numbers"):
        truth.curve("liver", at=["1"])
    with pytest.raises(ValueError, match="one value or one per gene \\(2\\)"):
        truth.benchmark_doses("liver", change=[0.5, 0.5, 0.5])
    for change in (0.0, -1.0, math.nan, math.inf):
        with pytest.raises(ValueError, match="finite and above 0"):
            truth.benchmark_doses("liver", change=change)
    with pytest.raises(TypeError, match="change must hold real numbers"):
        truth.benchmark_doses("liver", change="0.5")


def test_doses_the_simulator_refuses_are_those_a_dose_axis_reads_as_one() -> None:
    doses = [0.0, 0.3, 0.1 * 3, 1.0]
    with pytest.raises(ValueError, match="differ by less than 1e-8"):
        simulate_dose_series(doses, n_genes=5, rng=1)
    axis = DoseAxis("dose", unit="uM", metric="nominal")
    with pytest.raises(ValueError, match="differ by less than 1e-8"):
        axis.spline_basis({"dose": doses}, df=1)
    # Doses 1e-7 apart relative to the dose are two, for both.
    distinct = [0.0, 0.3, 0.3 * (1 + 1e-7), 1.0]
    cube, _ = simulate_dose_series(distinct, n_genes=5, rng=1)
    axis.spline_basis(cube.samples, df=1)


# The documentation.


def test_the_chapter_example_runs(capsys: pytest.CaptureFixture[str]) -> None:
    section = CHAPTER.read_text().split("(simulated-dose-series)=")[1]
    section = section.split("\n(")[0]
    blocks = re.findall(r"```python\n(.*?)```", section, re.S)
    assert len(blocks) == 1
    namespace: dict[str, Any] = {"cp": cytepool}
    exec(blocks[0], namespace)  # noqa: S102
    printed = capsys.readouterr().out
    # The shares the chapter quotes.
    assert printed.strip() in section
    assert "53% of responsive genes called, 0.2% of flat genes" in printed


def test_the_examples_in_the_docstrings_run() -> None:
    assert doctest.testmod(simulated_dose_series, raise_on_error=False).failed == 0
