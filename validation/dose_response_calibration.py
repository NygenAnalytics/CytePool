"""Error rates and power of the dose tests at the designs of dose series, on series
simulated from known curves.

The core grid of dose series simulated by ``simulate_dose_series``: dose groups K = 4
(the vehicle and 0.1, 1 and 10 ug/kg) and 9 (the vehicle and 0.01 to 30 ug/kg in
half-log steps), 2 and 3 replicates per dose, a median of 100 cells per profile, one
animal per profile, 2 cell types of 1,000 genes, and the composition unchanged.
``dose_response`` runs with its defaults on each series: limma-voom, the tests the
design supports and the spline's default size.

- Null rows: 1,000 null datasets per design cell, in which no gene responds. For each
  test, the Kolmogorov-Smirnov distance between its p-values and the uniform
  distribution, the shares of p-values below 0.05 and 0.01, and the share of cell
  types with any call at 5% FDR within the cell type, the error a gate on the test
  would make.
- Power rows: 10 datasets per design cell, in which a fifth of the genes respond with
  the simulator's default shapes and maximal changes (0.5, 1 and 2 log2 units). For
  each test, the share of the responsive genes it calls at 5% FDR, by shape; the mean
  false discovery proportion, the share of calls that are flat genes, within and
  across cell types; and the spline's power by its number of columns, from 1 to the
  most the treated doses allow.

Seeds: ``numpy.random.default_rng([20261201, kind, design_cell, dataset])``, with kind 0
for power and 1 for null datasets and the design cells numbered in the order of the
grid, as each row prints. Each estimate comes with its Monte Carlo standard error
between datasets and, for shares of p-values, the binomial bound of that error, which
p-values that a dataset's moderation ties together exceed. Every row is information.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root.

Usage:
    PYTHONPATH=. python validation/dose_response_calibration.py
    PYTHONPATH=. python validation/dose_response_calibration.py --null-datasets 100
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cytepool.design import Design
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import dose_response
from cytepool.simulated_dose_series import simulate_dose_series

SEED = 20261201
POWER, NULL = 0, 1
FDR = 0.05
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
DOSES = {
    4: [0.0, 0.1, 1.0, 10.0],
    9: [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
}
REPLICATES = (2, 3)
CELLS = 100.0
CELL_TYPES = 2
GENES = 1000
RESPONSIVE = 0.2
POWER_DATASETS = 10
TESTS = ("spline", "nonlinearity", "slope", "dose_groups")


@dataclass(frozen=True)
class DesignCell:
    number: int
    groups: int  # K, the vehicle included
    replicates: int

    @property
    def label(self) -> str:
        return f"K = {self.groups}, {self.replicates} replicates (design cell {self.number})"


def grid() -> list[DesignCell]:
    return [
        DesignCell(number, groups, replicates)
        for number, (groups, replicates) in enumerate(
            (k, r) for k in sorted(DOSES) for r in REPLICATES
        )
    ]


def series(cell: DesignCell, kind: int, dataset: int) -> tuple[Any, Any]:
    rng = np.random.default_rng([SEED, kind, cell.number, dataset])
    return simulate_dose_series(
        DOSES[cell.groups],
        replicates=cell.replicates,
        cell_types=CELL_TYPES,
        cells=CELLS,
        n_genes=GENES,
        responsive=RESPONSIVE if kind == POWER else 0.0,
        rng=rng,
    )


def analyse(cube: Any, **options: Any) -> Any:
    return dose_response(cube, Design(cube.samples, "~ 1"), dose=AXIS, **options)


# The null rows.


@dataclass
class Null:
    """The p-values of each test over the null datasets of a design cell, and per
    dataset and cell type, the shares below the thresholds and any call."""

    p_values: dict[str, list[np.ndarray]] = field(default_factory=dict)
    shares: dict[str, list[tuple[float, float]]] = field(default_factory=dict)
    any_call: dict[str, list[bool]] = field(default_factory=dict)


def null_rows(cell: DesignCell, datasets: int) -> list[tuple[str, ...]]:
    found = Null()
    for dataset in range(datasets):
        cube, _ = series(cell, NULL, dataset)
        result = analyse(cube)
        for (group, test), table in result.tables.items():
            if test == "each_dose":
                continue
            p = np.asarray(table["p_value"], dtype=float)
            found.p_values.setdefault(test, []).append(p)
            found.shares.setdefault(test, []).append(
                (float(np.mean(p < 0.05)), float(np.mean(p < 0.01)))
            )
            called = bool(np.any(np.asarray(table["adjusted_p_value"]) < FDR))
            found.any_call.setdefault(test, []).append(called)
    rows = []
    for test in TESTS:
        if test not in found.p_values:
            continue
        p = np.sort(np.concatenate(found.p_values[test]))
        n = p.size
        cumulative = np.arange(1, n + 1) / n
        distance = float(max(np.max(cumulative - p), np.max(p - (cumulative - 1 / n))))
        shares = np.array(found.shares[test])
        calls = np.array(found.any_call[test], dtype=float)
        parts = [f"KS distance {distance:.4f} ({n:,} p-values)"]
        for column, threshold in enumerate((0.05, 0.01)):
            share = float(np.mean(p < threshold))
            between = float(np.std(shares[:, column], ddof=1) / np.sqrt(len(shares)))
            bound = float(np.sqrt(threshold * (1 - threshold) / n))
            parts.append(
                f"below {threshold:g}: {share:.4f} (SE {between:.4f}; binomial "
                f"{bound:.4f})"
            )
        share = float(np.mean(calls))
        error = float(np.sqrt(share * (1 - share) / calls.size))
        parts.append(
            f"cell types with a call at 5% FDR: {share:.3f} (SE {error:.3f}, "
            f"{calls.size:,} cell types)"
        )
        rows.append(
            (
                f"{cell.label}: {test} under the null, {datasets:,} datasets",
                f"seed [{SEED}, {NULL}, {cell.number}, dataset]",
                "; ".join(parts),
                "n/a",
                "info",
            )
        )
    return rows


# The power rows.


def called(table: Any, column: str = "adjusted_p_value") -> set[str]:
    rows = np.asarray(table[column]) < FDR
    return set(np.asarray(table["feature_id"])[rows].tolist())


def power_rows(cell: DesignCell) -> list[tuple[str, ...]]:
    # Per test and shape: the share called in each cell type of each dataset.
    power: dict[tuple[str, str], list[float]] = {}
    within: dict[str, list[float]] = {}
    across: dict[str, list[float]] = {}
    by_size: dict[int, list[float]] = {}
    sizes: list[int] = []
    for dataset in range(POWER_DATASETS):
        cube, truth = series(cell, POWER, dataset)
        result = analyse(cube)
        calls_across: dict[str, tuple[int, int]] = {}
        for group in result.groups:
            table = truth.table(group)
            shape = dict(
                zip(table["feature_id"].tolist(), table["shape"].tolist(), strict=True)
            )
            change = dict(
                zip(
                    table["feature_id"].tolist(),
                    np.abs(table["max_log2_change"]).tolist(),
                    strict=True,
                )
            )
            for test in result.tests:
                if (group, test) not in result.tables:
                    continue
                tested = result.table(group, test)
                genes = tested["feature_id"].tolist()
                hits = called(tested)
                responsive = [g for g in genes if change[g] > 0]
                for name in sorted({shape[g] for g in responsive}):
                    members = [g for g in responsive if shape[g] == name]
                    power.setdefault((test, name), []).append(
                        sum(g in hits for g in members) / len(members)
                    )
                power.setdefault((test, "all"), []).append(
                    sum(g in hits for g in responsive) / len(responsive)
                )
                false = sum(change[g] == 0 for g in hits)
                within.setdefault(test, []).append(false / max(len(hits), 1))
                hits_across = called(tested, "adjusted_p_value_across_groups")
                false_across = sum(change[g] == 0 for g in hits_across)
                total, wrong = calls_across.get(test, (0, 0))
                calls_across[test] = (total + len(hits_across), wrong + false_across)
            # The spline's power by its number of columns.
            treated = cell.groups - 1
            sizes = list(range(1, treated))
            for size in sizes:
                sized = analyse(
                    cube, tests=["spline"], spline_df=size, groups=group
                ).table(group, "spline")
                genes = sized["feature_id"].tolist()
                hits = called(sized)
                responsive = [g for g in genes if change[g] > 0]
                by_size.setdefault(size, []).append(
                    sum(g in hits for g in responsive) / len(responsive)
                )
        for test, (total, wrong) in calls_across.items():
            across.setdefault(test, []).append(wrong / max(total, 1))
    rows = []

    def mean(values: list[float]) -> str:
        values_array = np.asarray(values)
        error = np.std(values_array, ddof=1) / np.sqrt(values_array.size)
        return f"{np.mean(values_array):.3f} (SE {error:.3f})"

    for test in TESTS:
        if (test, "all") not in power:
            continue
        shapes = sorted(name for t, name in power if t == test and name != "all")
        by_shape = ", ".join(
            f"{name} {np.mean(power[test, name]):.2f}" for name in shapes
        )
        rows.append(
            (
                f"{cell.label}: {test}, power at 5% FDR, {POWER_DATASETS} datasets",
                f"seed [{SEED}, {POWER}, {cell.number}, dataset]",
                f"{mean(power[test, 'all'])}; by shape: {by_shape}",
                "n/a",
                "info",
            )
        )
        rows.append(
            (
                f"{cell.label}: {test}, the mean false discovery proportion at 5% FDR",
                f"seed [{SEED}, {POWER}, {cell.number}, dataset]",
                f"within cell types {mean(within[test])}; across cell types "
                f"{mean(across[test])}",
                "n/a",
                "info",
            )
        )
    rows.append(
        (
            f"{cell.label}: the spline's power at 5% FDR by its number of columns, "
            f"{POWER_DATASETS} datasets",
            f"seed [{SEED}, {POWER}, {cell.number}, dataset]",
            "; ".join(f"{size}: {mean(by_size[size])}" for size in sizes),
            "n/a",
            "info",
        )
    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--null-datasets",
        type=int,
        default=1000,
        help="null datasets per design cell (1,000 in the core grid)",
    )
    arguments = parser.parse_args()
    rows = []
    for cell in grid():
        rows += null_rows(cell, arguments.null_datasets)
        rows += power_rows(cell)
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"{len(rows)} checks, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
