"""Error rates and power of the dose tests at the designs of dose series, and the
coverage of the curves' bands, on series simulated from known curves.

A grid of designs simulated by ``simulate_dose_series``: dose groups K = 4 (the
vehicle and 0.1, 1 and 10 ug/kg) and 9 (the vehicle and 0.01 to 30 ug/kg in half-log
steps), 2 and 3 replicates per dose, a median of 100 cells per profile, one animal
per profile, 2 cell types of 1,000 genes, and the composition unchanged.
``dose_response`` runs with its defaults on each series: limma-voom, the tests the
design supports and the spline's default size.

- Null rows: 1,000 null datasets per design cell, in which no gene responds. For each
  test, the Kolmogorov-Smirnov distance between its p-values and the uniform
  distribution, the shares of p-values below 0.05 and 0.01, and the share of cell
  types with any call at 5% FDR within the cell type, the error a gate on the test
  would make. Then what the calls share: the share of cell types whose prior degrees
  of freedom are infinite, with a call by any test in those and in the others, and
  where the dose-group test's calls fall among the genes' true SDs between vehicle
  replicates (``expression_sd``).
- Power rows: 10 datasets per design cell, in which a fifth of the genes respond with
  the simulator's default shapes and maximal changes (0.5, 1 and 2 log2 units). For
  each test, the share of the responsive genes it calls at 5% FDR, by shape; the mean
  false discovery proportion, the share of calls that are flat genes, within and
  across cell types; and the spline's power by its number of columns, from 1 to the
  most the treated doses allow.
- Band rows: 12 datasets of K = 9 x 3 replicates, one cell type of 600 genes of which
  30% respond, with 3,000 cells per profile, so that counting compresses no change.
  At each treated dose, the share of the curves' pointwise 95% bands that hold the
  true change, for flat and responsive genes and by shape, against the band's own
  target, the spline fitted by least squares to the true means of the dose groups,
  and against each dose's 95% interval from the table of each dose.

Seeds: ``numpy.random.default_rng([20261201, kind, design_cell, dataset])``, with kind 0
for power, 1 for null and 2 for the band's datasets, the design cells numbered in the
order of the grid, as each row prints, and the band's design numbered 0. Each
estimate comes with its Monte Carlo standard error between datasets and, for shares
of p-values, the binomial bound of that error, which p-values that a dataset's
moderation ties together exceed. Every row is information.

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
from cytepool.stats.distributions import t_ppf

SEED = 20261201
POWER, NULL, BAND = 0, 1, 2
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
TESTS = ("spline", "nonlinearity", "linear_trend", "dose_groups")
BAND_DATASETS = 12
BAND_CELLS = 3000.0
BAND_GENES = 600
BAND_RESPONSIVE = 0.3


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
    # Per cell type: infinite prior degrees of freedom, and a call by any test; per
    # call of the dose-group test, the quantile of the gene's true SD among the genes
    # tested in its cell type.
    infinite_prior: list[bool] = field(default_factory=list)
    called_by_any: list[bool] = field(default_factory=list)
    sd_quantiles: list[float] = field(default_factory=list)


def null_rows(cell: DesignCell, datasets: int) -> list[tuple[str, ...]]:
    found = Null()
    for dataset in range(datasets):
        cube, truth = series(cell, NULL, dataset)
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
        for row, group in enumerate(result.groups):
            if (group, "dose_groups") not in result.tables:
                continue
            found.infinite_prior.append(bool(np.isinf(result.summary["df_prior"][row])))
            found.called_by_any.append(
                any(
                    bool(np.any(np.asarray(table["adjusted_p_value"]) < FDR))
                    for (owner, test), table in result.tables.items()
                    if owner == group and test != "each_dose"
                )
            )
            tested = result.table(group, "dose_groups")
            genes = truth.table(group)
            sd = dict(
                zip(
                    genes["feature_id"].tolist(),
                    np.asarray(genes["expression_sd"], dtype=float).tolist(),
                    strict=True,
                )
            )
            values = np.array([sd[gene] for gene in tested["feature_id"].tolist()])
            quantile = (np.argsort(np.argsort(values)) + 0.5) / values.size
            calls = np.asarray(tested["adjusted_p_value"]) < FDR
            found.sd_quantiles.extend(quantile[calls].tolist())
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
    rows.append(
        (
            f"{cell.label}: what the null calls share, {datasets:,} datasets",
            f"seed [{SEED}, {NULL}, {cell.number}, dataset]",
            null_calls(found),
            "n/a",
            "info",
        )
    )
    return rows


def null_calls(found: Null) -> str:
    """The prior degrees of freedom of the cell types with a call, and where the
    dose-group test's calls fall among the genes' true SDs."""
    infinite = np.array(found.infinite_prior, dtype=bool)
    called = np.array(found.called_by_any, dtype=float)
    quantiles = np.array(found.sd_quantiles)
    with_infinite = f"{np.mean(called[infinite]):.3f}" if infinite.any() else "none"
    others = f"{np.mean(called[~infinite]):.3f}" if (~infinite).any() else "none"
    parts = [
        f"prior df infinite in {np.mean(infinite):.3f} of {infinite.size:,} cell "
        f"types, with a call by any test at 5% FDR in {with_infinite} of those and "
        f"{others} of the others",
        f"dose-group test: {quantiles.size:,} calls"
        + (
            f", {np.mean(quantiles >= 0.9):.2f} of them among the tenth of the genes "
            "tested of largest true SD between vehicle replicates"
            if quantiles.size
            else ""
        ),
    ]
    return "; ".join(parts)


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


# The band rows.


@dataclass
class Coverage:
    """Per dataset and kind of gene, whether each gene's band, and its dose's
    interval, holds the true change at each treated dose."""

    band: dict[str, list[np.ndarray]] = field(default_factory=dict)
    interval: dict[str, list[np.ndarray]] = field(default_factory=dict)
    target: list[np.ndarray] = field(default_factory=list)

    def add(self, kind: str, band: np.ndarray, interval: np.ndarray) -> None:
        self.band.setdefault(kind, []).append(band.ravel())
        self.interval.setdefault(kind, []).append(interval.ravel())


def band_rows() -> list[tuple[str, ...]]:
    doses = np.array(DOSES[9])
    treated = doses[1:]
    found = Coverage()
    for dataset in range(BAND_DATASETS):
        rng = np.random.default_rng([SEED, BAND, 0, dataset])
        cube, truth = simulate_dose_series(
            doses.tolist(),
            replicates=3,
            cell_types=1,
            cells=BAND_CELLS,
            n_genes=BAND_GENES,
            responsive=BAND_RESPONSIVE,
            rng=rng,
        )
        result = analyse(cube)
        group = result.groups[0]
        curve = result.curve(group, at=treated)
        genes = curve["feature_id"].tolist()[:: treated.size]
        true = truth.curve(group, genes, at=treated)  # genes x treated doses
        lower = np.asarray(curve["ci_lower"]).reshape(len(genes), -1)
        upper = np.asarray(curve["ci_upper"]).reshape(len(genes), -1)
        band = (lower <= true) & (true <= upper)
        # The band's own target: the spline that least squares fits to the true means
        # of the dose groups, each weighted by its samples, as a change from the
        # vehicle's mean.
        (entry,) = result.provenance["groups"]
        root = np.sqrt(np.asarray(entry["samples"], dtype=float))[:, None]
        columns = AXIS.design_columns({"dose": doses}, result.spline_basis)
        z = np.column_stack([np.ones(doses.size), *columns.values()])
        means = np.column_stack([np.zeros(len(genes)), true]).T  # doses x genes
        beta = np.linalg.lstsq(z * root, means * root, rcond=None)[0]
        fitted = (z @ beta).T
        target = fitted[:, 1:] - fitted[:, :1]
        # Each dose's 95% interval, on the degrees of freedom of the gene's t-tests.
        each = result.table(group, "each_dose")
        estimate = np.asarray(each["log2_fold_change"]).reshape(treated.size, -1).T
        error = np.asarray(each["standard_error"]).reshape(treated.size, -1).T
        quantile = np.asarray(t_ppf(0.975, result.fit(group).moderated.df_total))
        interval = np.abs(estimate - true) <= quantile[:, None] * error
        table = truth.table(group)
        shape_of = dict(
            zip(table["feature_id"].tolist(), table["shape"].tolist(), strict=True)
        )
        shapes = np.array([shape_of[gene] for gene in genes])
        responsive = shapes != "flat"
        found.add("flat", band[~responsive], interval[~responsive])
        found.add("responsive", band[responsive], interval[responsive])
        for shape in np.unique(shapes[responsive]):
            rows = shapes == shape
            found.add(shape, band[rows], interval[rows])
        found.target.append(((lower <= target) & (target <= upper))[responsive].ravel())

    def share(values: list[np.ndarray]) -> str:
        """The share over every gene and dose, with the SE between datasets."""
        per_dataset = np.array([np.mean(v) for v in values if v.size])
        error = np.std(per_dataset, ddof=1) / np.sqrt(per_dataset.size)
        return f"{np.mean(np.concatenate(values)):.3f} (SE {error:.3f})"

    label = (
        f"K = 9, 3 replicates, {BAND_CELLS:,.0f} cells: the 95% band at the treated "
        f"doses, {BAND_DATASETS} datasets"
    )
    seed = f"seed [{SEED}, {BAND}, 0, dataset]"
    shapes = sorted(set(found.band) - {"flat", "responsive"})
    by_shape = ", ".join(
        f"{shape} {np.mean(np.concatenate(found.band[shape])):.3f}" for shape in shapes
    )
    count = sum(v.size for v in found.band["flat"])
    rows = [
        (
            f"{label}: flat genes ({count:,} gene-doses)",
            seed,
            f"band {share(found.band['flat'])}; each dose's interval "
            f"{share(found.interval['flat'])}",
            "n/a",
            "info",
        ),
        (
            f"{label}: responsive genes "
            f"({sum(v.size for v in found.band['responsive']):,} gene-doses)",
            seed,
            f"band {share(found.band['responsive'])}, by shape: {by_shape}; each "
            f"dose's interval {share(found.interval['responsive'])}",
            "n/a",
            "info",
        ),
        (
            f"{label}: responsive genes, the band against the spline fitted to the "
            "true means of the dose groups",
            seed,
            share(found.target),
            "n/a",
            "info",
        ),
    ]
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
    rows += band_rows()
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    print(f"{len(rows)} checks, 0 failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
