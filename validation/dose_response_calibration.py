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
- Shape rows: the curves of named models on the power datasets of K = 9: for
  ``"epa"`` under constant, weighted and moderated variance and for ``"ll4"`` and
  ``"dromics"`` under weighted variance, among the responsive genes, the share whose
  best model by AICc contains the true shape (hill: hill and ll4; exponential:
  exponential_4 and exponential_5, and DRomics' exponential; linear: linear,
  polynomial_2 and power; power: power; log_gauss_probit: the log-Gauss-probit forms;
  frechet and step: none), the share whose best fit has the true direction or, for
  bells, turns, and the mean Akaike weight on the models that contain the true
  shape; for ``"dromics"``, the share of best fits that turn among the true bells, the
  other responsive genes and the flat ones, and the share of those turns that one
  dose group drives.
- Potency rows: the benchmark doses of the same ``"epa"`` curves under constant,
  weighted and moderated variance at a 1.1-fold change, ``bmr`` of a change of
  log2(1.1), among the genes whose true shape lies in the set (hill, exponential,
  linear, power) and whose true benchmark dose at that change (``bmd_fold_change``)
  lies within the doses: the share with a BMD, the median and interquartile range of
  log10 of the estimated over the true BMD, and one-sided BMDL and BMDU coverage of the
  true BMD, also by the gene's expected counts per profile at the vehicle (below 10,
  10 to 100, 100 or more), since the prior counts of log-CPM values compress the
  observed changes of genes with few counts, and by whether the best model contains
  the true shape; and log10 of the 1 SD BMD over the true BMD at one SD of true
  expression (``bmd_expression_sd``), as bias only, since the observed SD includes
  counting noise.
- Straight-line rows: lines with normal errors simulated directly at K = 9 x 3, whose
  one-sided BMDLs have exact references: Fieller's t-based bound for a change of 0.4
  and the noncentral t bound for 1.349 SD, each of which must cover the true BMD in 95%
  of datasets within 4 standard errors (calibrated), beside the coverage of
  ``benchmark_doses``'s profile-likelihood BMDL and BMDU of the line, and of the
  bound of a change held at 1.349 times the vehicle replicates' sample SD, which
  ignores that SD's error.
- Band rows: 12 datasets of K = 9 x 3 replicates, one cell type of 600 genes of which
  30% respond, with 3,000 cells per profile, so that counting compresses no change.
  At each treated dose, the share of the curves' pointwise 95% bands that hold the
  true change, for flat and responsive genes and by shape, against the band's own
  target, the spline fitted by least squares to the true means of the dose groups,
  and against each dose's 95% interval from the table of each dose.
- Multiple contrast rows: ``dunnett``, ``williams`` and ``mcp``, which take about a
  millisecond per gene, on the first 200 null datasets of each design cell
  (``--contrast-datasets``) and on the power datasets, with the rows of the other tests.
  A control row on the same null datasets: calls of each dose at an unadjusted p-value
  below 0.05, which must call a dose in most null cell types.
- Gate rows: the share of null cell types in which a gate passes any gene at 5% FDR:
  the default gate, the spline test, and the dose-group gate, counted on the null
  datasets above, and the gate of a 1.5-fold change by TREAT, on the same datasets.
  Each against 5% plus 4 binomial standard errors of the cell types it counts, 6.95%
  for the 2,000 cell types of 1,000 datasets: a check of the spline and fold-change
  gates at 3 replicates, and a known limit, printed with the bound, for the
  dose-group gate and at 2 replicates. Then ``permutation_check`` with 100 runs of the
  default gate on the first 100 null datasets of each design cell
  (``--check-datasets``): the mean share of its runs that pass genes against the
  default gate's share on the other null datasets, within 4 standard errors of the
  difference, and the share of cell types whose p-value is at most 0.05, at most 5%
  plus 4 binomial standard errors. A control row: the unadjusted Williams-type gate,
  any gene at a p-value below 0.05 with no adjustment across genes, on the multiple
  contrast tests' null datasets, must exceed 5% by more than 4 binomial standard
  errors of the cell types it counts. On the
  power datasets, the 1.5-fold gate's power by maximal change and its false discovery
  proportion, a false call being a gene whose true change never exceeds 1.5-fold at
  the cell type's doses, and the check's p-values. And the design of Lee et al.
  (2024), vehicle-only data placed in 4 mock dose groups of 6 samples, 200 datasets:
  the default gate's share, and the share with a gene at an unadjusted Williams-type
  p-value below 0.05 and a change of 1.5-fold at a dose, the prefilter of their
  study (information).
- Stage-wise rows: the calls of ``lowest_effective_doses`` with the spline as gate,
  on the power datasets. The overall false discovery proportion, the share of the genes
  called with a false call, a flat gene's or a call at a dose where the true change is
  exactly 0, as below a step's dose; the share of false gene-dose calls among all of
  them; the share of the genes the screen calls that get a lowest effective dose; and
  the lowest effective dose against the gene's true 1.1-fold benchmark dose.

Seeds: ``numpy.random.default_rng([20261201, kind, design_cell, dataset])``, with kind 0
for power, 1 for null, 2 for the band's and 4 for the vehicle-only datasets, the design
cells numbered in the order of the grid, as each row prints, and the band's and the
vehicle-only design numbered 0; the multiple contrast tests of a dataset take their
seed from kind 3, the permutation checks theirs from kind 5, and the straight lines
from kind 6 with the true BMD's case. Each estimate comes with its Monte Carlo standard
error between datasets and, for shares of p-values and coverages, the binomial bound
of that error, which p-values that a dataset's moderation ties together exceed. Every
row is information but the spline and fold-change gates' rows at 3 replicates, the
permutation check's rows, the control rows and the straight lines' exact references,
which fail the run when they fail.

Requirements:
    Python 3.13 with numpy and numba, and scipy for the straight lines' references,
    run from the repository root.

Usage:
    PYTHONPATH=. python validation/dose_response_calibration.py
    PYTHONPATH=. python validation/dose_response_calibration.py --null-datasets 100
    PYTHONPATH=. python validation/dose_response_calibration.py --contrast-datasets 50
    PYTHONPATH=. python validation/dose_response_calibration.py --check-datasets 20
"""

import argparse
import sys
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cytepool.benchmark_responses import BenchmarkResponse
from cytepool.design import Design
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import dose_response
from cytepool.permutation_checks import permutation_check
from cytepool.simulated_dose_series import simulate_dose_series
from cytepool.stats.distributions import t_ppf
from cytepool.stats.multiple_contrasts import stage_wise_adjusted_p_values

SEED = 20261201
POWER, NULL, BAND, CONTRASTS, VEHICLE_ONLY, CHECKS, LINES = 0, 1, 2, 3, 4, 5, 6
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
CONTRAST_TESTS = ("dunnett", "williams", "mcp")
BAND_DATASETS = 12
BAND_CELLS = 3000.0
BAND_GENES = 600
BAND_RESPONSIVE = 0.3
FOLD_CHANGE = 1.5
CHECK_RUNS = 100
# The vehicle-only design of Lee et al. (2024): 4 mock dose groups of 6 samples.
VEHICLE_ONLY_DOSES = [0.0, 0.1, 1.0, 10.0]
VEHICLE_ONLY_REPLICATES = 6
VEHICLE_ONLY_DATASETS = 200
# The benchmark response of the truth's bmd_fold_change, a 1.1-fold change.
POTENCY_CHANGE = float(np.log2(1.1))
IN_EPA_SET = ("hill", "exponential", "linear", "power")
COUNT_STRATA = ((0.0, 10.0), (10.0, 100.0), (100.0, np.inf))
LINE_DATASETS = 4000
LINE_SD = 0.3
LINE_CHANGE = 0.4
LINE_SD_MULTIPLE = 1.349
LINE_BMDS = (1.0, 10.0)


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


# The curves' shapes: which models of each set contain each true shape.
CONTAINING = {
    "epa": {
        "hill": ("hill",),
        "exponential": ("exponential_4", "exponential_5"),
        "linear": ("linear", "polynomial_2", "power"),
        "power": ("power",),
    },
    "ll4": {"hill": ("ll4",)},
    "dromics": {
        "hill": ("ll4",),
        "exponential": ("exponential",),
        "linear": ("linear",),
        "log_gauss_probit": (
            "log_gauss_probit",
            "log_gauss_probit_monotone",
            "log_gauss_probit_symmetric",
        ),
    },
}
CURVE_RUNS = (
    ("epa", "constant"),
    ("epa", "weighted"),
    ("epa", "moderated"),
    ("ll4", "weighted"),
    ("dromics", "weighted"),
)


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
    # Per dataset and gate, whether each cell type's gate passes a gene; per cell
    # type of the checked datasets, the share of the check's runs that pass genes
    # and its p-value.
    gates: dict[str, list[list[bool]]] = field(default_factory=dict)
    check_shares: list[list[float]] = field(default_factory=list)
    check_p_values: list[float] = field(default_factory=list)


def null_rows(cell: DesignCell, datasets: int, checked: int) -> list[tuple[str, ...]]:
    found = Null()
    for dataset in range(datasets):
        cube, truth = series(cell, NULL, dataset)
        result = analyse(cube)
        changes = analyse(cube, tests="dose_groups", gate={"fold_change": FOLD_CHANGE})
        for gate, source in (
            ("spline", result),
            ("dose_groups", result),
            ("fold_change", changes),
        ):
            found.gates.setdefault(gate, []).append(
                [gate_calls(source, group, gate) for group in result.groups]
            )
        if dataset < checked:
            check = permutation_check(
                result, n_permutations=CHECK_RUNS, rng=check_seed(cell, dataset)
            )
            found.check_shares.append(
                check.summary["share_of_runs_with_passing_genes"].tolist()
            )
            found.check_p_values += check.summary["p_value"].tolist()
        for (group, test), table in result.tables.items():
            if test in ("each_dose", "gate"):
                continue
            p = np.asarray(table["p_value"], dtype=float)
            found.p_values.setdefault(test, []).append(p)
            found.shares.setdefault(test, []).append(
                (float(np.mean(p < 0.05)), float(np.mean(p < 0.01)))
            )
            called = bool(np.any(np.asarray(table["adjusted_p_value"]) <= FDR))
            found.any_call.setdefault(test, []).append(called)
        for row, group in enumerate(result.groups):
            if (group, "dose_groups") not in result.tables:
                continue
            found.infinite_prior.append(bool(np.isinf(result.summary["df_prior"][row])))
            found.called_by_any.append(
                any(
                    bool(np.any(np.asarray(table["adjusted_p_value"]) <= FDR))
                    for (owner, test), table in result.tables.items()
                    if owner == group and test not in ("each_dose", "gate")
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
            calls = np.asarray(tested["adjusted_p_value"]) <= FDR
            found.sd_quantiles.extend(quantile[calls].tolist())
    rows = summary_rows(cell, found, TESTS, datasets)
    rows.append(
        (
            f"{cell.label}: what the null calls share, {datasets:,} datasets",
            f"seed [{SEED}, {NULL}, {cell.number}, dataset]",
            null_calls(found),
            "n/a",
            "info",
        )
    )
    return rows + gate_null_rows(cell, found, datasets, checked)


def null_bound(cell_types: int) -> float:
    """The largest share of null cell types with a call that is in line with 5%: 5%
    plus 4 binomial standard errors of the cell types counted."""
    return float(0.05 + 4 * np.sqrt(0.05 * 0.95 / cell_types))


def gate_calls(result: Any, group: str, gate: str) -> bool:
    """Whether a gate passes a gene of a cell type at 5% FDR: the result's own gate,
    or the gate of the dose groups' test, whose adjusted p-values it calls."""
    if gate == "dose_groups":
        table = result.table(group, "dose_groups")
        return bool(np.any(np.asarray(table["adjusted_p_value"]) <= FDR))
    return bool(np.any(result.table(group, "gate")["passes_gate"]))


def check_seed(cell: DesignCell, dataset: int) -> np.random.Generator:
    """The generator from which a dataset's permutation check draws its seed."""
    return np.random.default_rng([SEED, CHECKS, cell.number, dataset])


def share_text(calls: np.ndarray) -> tuple[float, float, str]:
    """The share of cell types with a call over datasets x cell types, its standard
    error between datasets, and the text of both with the binomial bound."""
    share = float(np.mean(calls))
    between = float(np.std(np.mean(calls, axis=1), ddof=1) / np.sqrt(len(calls)))
    binomial = float(np.sqrt(share * (1 - share) / calls.size))
    text = (
        f"{share:.4f} (SE between datasets {between:.4f}; binomial {binomial:.4f}, "
        f"{calls.size:,} cell types)"
    )
    return share, between, text


def gate_null_rows(
    cell: DesignCell, found: Null, datasets: int, checked: int
) -> list[tuple[str, ...]]:
    """The gates' shares of null cell types with a gene passing, and the permutation
    check's agreement with the default gate's share."""
    rows = []
    labels = {
        "spline": "the default gate, the spline test,",
        "dose_groups": "the dose-group gate",
        "fold_change": f"the {FOLD_CHANGE:g}-fold gate, by TREAT,",
    }
    reference = f"seed [{SEED}, {NULL}, {cell.number}, dataset]"
    for gate, label in labels.items():
        calls = np.array(found.gates[gate], dtype=float)
        share, _, text = share_text(calls)
        bound = null_bound(calls.size)
        # The dose-group gate exceeds the bound at 3 replicates too.
        known = cell.replicates < 3 or gate == "dose_groups"
        if share <= bound:
            result = "pass"
        else:
            result = "known limit" if known else "FAIL"
        rows.append(
            (
                f"{cell.label}: {label} at 5% FDR under the null, {datasets:,} "
                "datasets: cell types with a gene passing",
                reference,
                text,
                f"at most {bound:.4f}",
                result,
            )
        )
    if not checked:
        return rows
    per_cell_type = np.array(found.check_shares, dtype=float)
    mean = float(np.mean(per_cell_type))
    error = float(np.std(np.mean(per_cell_type, axis=1), ddof=1) / np.sqrt(checked))
    # The default gate's share on the datasets the check did not run on, which are
    # independent of the check's.
    others = np.array(found.gates["spline"][checked:], dtype=float)
    null, null_error, _ = share_text(others)
    bound = 4 * np.sqrt(error**2 + null_error**2)
    difference = abs(mean - null)
    rows.append(
        (
            f"{cell.label}: permutation_check, {CHECK_RUNS} runs of the default gate "
            f"on the first {checked:,} null datasets: the mean share of runs that pass "
            f"genes, against the default gate's share on the other {len(others):,}",
            f"seed [{SEED}, {CHECKS}, {cell.number}, dataset]",
            f"{mean:.4f} (SE {error:.4f}) against {null:.4f} (SE {null_error:.4f}); "
            f"difference {difference:.4f}",
            f"at most {bound:.4f}",
            "pass" if difference <= bound else "FAIL",
        )
    )
    p = np.array(found.check_p_values, dtype=float)
    share = float(np.mean(p <= 0.05))
    limit = null_bound(p.size)
    rows.append(
        (
            f"{cell.label}: permutation_check on the same datasets: cell types with a "
            "p-value at most 0.05",
            f"seed [{SEED}, {CHECKS}, {cell.number}, dataset]",
            f"{share:.4f} ({p.size:,} cell types); p-values of 1 in {np.mean(p == 1):.3f}",
            f"at most {limit:.4f}",
            "pass" if share <= limit else "FAIL",
        )
    )
    return rows


def summary_rows(
    cell: DesignCell, found: Null, tests: tuple[str, ...], datasets: int
) -> list[tuple[str, ...]]:
    """Per test: the Kolmogorov-Smirnov distance, the shares of p-values below 0.05
    and 0.01, and the share of cell types with a call."""
    rows = []
    for test in tests:
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


def contrast_seed(cell: DesignCell, dataset: int) -> np.random.Generator:
    """The generator from which dose_response draws the seed of a dataset's multiple
    contrast tests."""
    return np.random.default_rng([SEED, CONTRASTS, cell.number, dataset])


def contrast_null_rows(cell: DesignCell, datasets: int) -> list[tuple[str, ...]]:
    """The multiple contrast tests on the first null datasets of a design cell, and
    the control row of each dose's calls at an unadjusted 0.05."""
    found = Null()
    any_dose = []
    unadjusted = []  # a gene at a Williams-type p-value below 0.05, unadjusted
    for dataset in range(datasets):
        cube, _ = series(cell, NULL, dataset)
        result = analyse(cube, tests=CONTRAST_TESTS, rng=contrast_seed(cell, dataset))
        for group in result.groups:
            for test in CONTRAST_TESTS:
                table = result.table(group, test)
                p = np.asarray(table["p_value"], dtype=float)
                found.p_values.setdefault(test, []).append(p)
                found.shares.setdefault(test, []).append(
                    (float(np.mean(p < 0.05)), float(np.mean(p < 0.01)))
                )
                calls = np.asarray(table["adjusted_p_value"]) <= FDR
                found.any_call.setdefault(test, []).append(bool(np.any(calls)))
            each = np.asarray(result.table(group, "each_dose")["p_value"], dtype=float)
            any_dose.append(bool(np.any(each < 0.05)))
            williams = np.asarray(result.table(group, "williams")["p_value"])
            unadjusted.append(bool(np.any(williams < 0.05)))
    rows = summary_rows(cell, found, CONTRAST_TESTS, datasets)
    share = float(np.mean(any_dose))
    rows.append(
        (
            f"{cell.label}: a call of each dose at an unadjusted p-value below 0.05, "
            f"under the null, {datasets:,} datasets",
            f"seed [{SEED}, {NULL}, {cell.number}, dataset]",
            f"cell types with a call: {share:.3f} ({len(any_dose):,} cell types)",
            "above 0.5",
            "control" if share > 0.5 else "FAIL",
        )
    )
    share = float(np.mean(unadjusted))
    bound = null_bound(len(unadjusted))
    rows.append(
        (
            f"{cell.label}: the unadjusted Williams-type gate, a gene at a p-value below "
            f"0.05 with no adjustment across genes, under the null, {datasets:,} datasets",
            f"seed [{SEED}, {NULL}, {cell.number}, dataset]",
            f"cell types with a gene passing: {share:.3f} ({len(unadjusted):,} cell "
            "types)",
            f"above {bound:.4f}",
            "control" if share > bound else "FAIL",
        )
    )
    return rows


# The power rows.


def called(table: Any, column: str = "adjusted_p_value") -> set[str]:
    rows = np.asarray(table[column]) <= FDR
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


def contrast_power_rows(cell: DesignCell) -> list[tuple[str, ...]]:
    """The multiple contrast tests and the stage-wise calls on the power datasets."""
    power: dict[tuple[str, str], list[float]] = {}
    within: dict[str, list[float]] = {}
    overall: list[float] = []  # per cell type, the false discovery proportion of genes
    pairs = [0, 0]  # gene-dose calls, and those false
    screened = [0, 0]  # genes the screen calls, and those with a dose called
    distance: list[float] = []  # log10 of the lowest effective dose over the BMD
    for dataset in range(POWER_DATASETS):
        cube, truth = series(cell, POWER, dataset)
        result = analyse(
            cube, tests=("spline", *CONTRAST_TESTS), rng=contrast_seed(cell, dataset)
        )
        for group in result.groups:
            table = truth.table(group)
            ids = table["feature_id"].tolist()
            shape = dict(zip(ids, table["shape"].tolist(), strict=True))
            change = dict(
                zip(ids, np.abs(table["max_log2_change"]).tolist(), strict=True)
            )
            for test in CONTRAST_TESTS:
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
            # Every dose called in the genes the spline calls, from the building
            # block; a call is false where the true change at the dose is exactly 0.
            spline = result.table(group, "spline")
            each = result.table(group, "each_dose")
            genes = spline["feature_id"].tolist()
            entry = next(
                e for e in result.provenance["groups"] if e["group_id"] == group
            )
            doses = np.asarray(entry["doses"][1:], dtype=np.float64)
            p = np.asarray(each["p_value"], dtype=np.float64).reshape(doses.size, -1).T
            adjusted = stage_wise_adjusted_p_values(
                np.asarray(spline["p_value"]), p, alpha=FDR
            )
            dose_calls = np.nan_to_num(adjusted, nan=2.0) <= FDR
            true = truth.curve(group, genes, at=doses)
            lowest = result.lowest_effective_doses(group)
            responds = np.asarray(lowest["passes_gate"], dtype=bool)
            flat = np.array([change[g] == 0 for g in genes])
            wrong = dose_calls & (true == 0.0)
            false_genes = (responds & flat) | wrong.any(axis=1)
            overall.append(float(false_genes.sum() / max(responds.sum(), 1)))
            pairs[0] += int(dose_calls.sum())
            pairs[1] += int(wrong.sum())
            screened[0] += int(responds.sum())
            led = np.asarray(lowest["lowest_effective_dose"], dtype=np.float64)
            screened[1] += int(np.count_nonzero(responds & ~np.isnan(led)))
            bmd = np.asarray(table["bmd_fold_change"], dtype=np.float64)
            rows_ids = [ids.index(g) for g in genes]
            both = ~np.isnan(led) & np.isfinite(bmd[rows_ids]) & ~flat
            distance += np.log10(led[both] / bmd[rows_ids][both]).tolist()
    rows = []

    def mean(values: list[float]) -> str:
        values_array = np.asarray(values)
        error = np.std(values_array, ddof=1) / np.sqrt(values_array.size)
        return f"{np.mean(values_array):.3f} (SE {error:.3f})"

    reference = f"seed [{SEED}, {POWER}, {cell.number}, dataset]"
    for test in CONTRAST_TESTS:
        shapes = sorted(name for t, name in power if t == test and name != "all")
        by_shape = ", ".join(
            f"{name} {np.mean(power[test, name]):.2f}" for name in shapes
        )
        rows.append(
            (
                f"{cell.label}: {test}, power at 5% FDR, {POWER_DATASETS} datasets",
                reference,
                f"{mean(power[test, 'all'])}; by shape: {by_shape}",
                "n/a",
                "info",
            )
        )
        rows.append(
            (
                f"{cell.label}: {test}, the mean false discovery proportion at 5% FDR",
                reference,
                f"within cell types {mean(within[test])}",
                "n/a",
                "info",
            )
        )
    rows.append(
        (
            f"{cell.label}: lowest_effective_doses, the overall false discovery "
            f"proportion at 5%, {POWER_DATASETS} datasets",
            reference,
            f"{mean(overall)}: genes the spline calls with a false call, a flat gene's "
            "or a dose's where the true change is 0",
            "n/a",
            "info",
        )
    )
    distance_text = (
        f"median {np.median(distance):.2f}, quartiles {np.quantile(distance, 0.25):.2f} "
        f"and {np.quantile(distance, 0.75):.2f}, {len(distance):,} genes"
        if distance
        else "no gene"
    )
    rows.append(
        (
            f"{cell.label}: lowest_effective_doses, calls and doses",
            reference,
            f"false gene-dose calls {pairs[1]:,} of {pairs[0]:,}; genes the spline calls "
            f"with a dose called {screened[1]:,} of {screened[0]:,}; log10 of the "
            f"lowest effective dose over the true 1.1-fold BMD: {distance_text}",
            "n/a",
            "info",
        )
    )
    return rows


def gate_power_rows(cell: DesignCell) -> list[tuple[str, ...]]:
    """The fold-change gate on the power datasets: its power by maximal change and
    its false discovery proportion, a false call a gene whose true change never
    exceeds the fold change at the cell type's doses; and the permutation check of
    the default gate there."""
    power: dict[float, list[float]] = {}
    false: list[float] = []
    p_values: list[float] = []
    shares: list[float] = []
    threshold = np.log2(FOLD_CHANGE)
    for dataset in range(POWER_DATASETS):
        cube, truth = series(cell, POWER, dataset)
        result = analyse(cube, tests="dose_groups", gate={"fold_change": FOLD_CHANGE})
        for group in result.groups:
            gate = result.table(group, "gate")
            genes = gate["feature_id"].tolist()
            passed = np.asarray(gate["passes_gate"], dtype=bool)
            entry = next(
                e for e in result.provenance["groups"] if e["group_id"] == group
            )
            doses = np.asarray(entry["doses"][1:], dtype=float)
            beyond = np.abs(truth.curve(group, genes, at=doses)).max(axis=1) > threshold
            table = truth.table(group)
            change = dict(
                zip(
                    table["feature_id"].tolist(),
                    np.abs(table["max_log2_change"]).tolist(),
                    strict=True,
                )
            )
            sizes = np.array([change[gene] for gene in genes])
            for size in np.unique(sizes[sizes > 0]):
                members = sizes == size
                power.setdefault(float(size), []).append(
                    float(np.mean(passed[members]))
                )
            false.append(float(np.sum(passed & ~beyond) / max(np.sum(passed), 1)))
        default = analyse(cube)
        check = permutation_check(
            default, n_permutations=CHECK_RUNS, rng=check_seed(cell, dataset)
        )
        p_values += check.summary["p_value"].tolist()
        shares += check.summary["share_of_runs_with_passing_genes"].tolist()

    def mean(values: list[float]) -> str:
        array = np.asarray(values)
        error = np.std(array, ddof=1) / np.sqrt(array.size)
        return f"{np.mean(array):.3f} (SE {error:.3f})"

    reference = f"seed [{SEED}, {POWER}, {cell.number}, dataset]"
    by_size = ", ".join(
        f"{size:g} log2: {mean(v)}" for size, v in sorted(power.items())
    )
    return [
        (
            f"{cell.label}: the {FOLD_CHANGE:g}-fold gate, power at 5% FDR by maximal "
            f"change, {POWER_DATASETS} datasets",
            reference,
            by_size,
            "n/a",
            "info",
        ),
        (
            f"{cell.label}: the {FOLD_CHANGE:g}-fold gate, the mean false discovery "
            f"proportion, a false call a gene never beyond {FOLD_CHANGE:g}-fold at the "
            "doses",
            reference,
            mean(false),
            "n/a",
            "info",
        ),
        (
            f"{cell.label}: permutation_check of the default gate on the power datasets",
            reference,
            f"p-values at most 0.05 in {np.mean(np.asarray(p_values) <= 0.05):.3f} of "
            f"{len(p_values)} cell types; runs that pass genes: median share "
            f"{np.median(shares):.2f}",
            "n/a",
            "info",
        ),
    ]


def vehicle_only_rows(datasets: int) -> list[tuple[str, ...]]:
    """The design of Lee et al. (2024), vehicle-only data in 4 mock dose groups of 6
    samples: the default gate, and their prefilter, a gene at an unadjusted
    Williams-type p-value below 0.05 and a 1.5-fold change at some dose."""
    gate: list[list[bool]] = []
    prefilter: list[list[bool]] = []
    threshold = np.log2(FOLD_CHANGE)
    for dataset in range(datasets):
        rng = np.random.default_rng([SEED, VEHICLE_ONLY, 0, dataset])
        cube, _ = simulate_dose_series(
            VEHICLE_ONLY_DOSES,
            replicates=VEHICLE_ONLY_REPLICATES,
            cell_types=CELL_TYPES,
            cells=CELLS,
            n_genes=GENES,
            responsive=0.0,
            rng=rng,
        )
        result = analyse(
            cube,
            tests=("spline", "williams"),
            rng=np.random.default_rng([SEED, CONTRASTS, 99, dataset]),
        )
        gate.append([gate_calls(result, group, "spline") for group in result.groups])
        found = []
        for group in result.groups:
            p = np.asarray(result.table(group, "williams")["p_value"])
            each = result.table(group, "each_dose")
            k = len(np.unique(each["dose"]))
            changes = np.abs(np.asarray(each["log2_fold_change"])).reshape(k, -1).T
            found.append(bool(np.any((p < 0.05) & (changes >= threshold).any(axis=1))))
        prefilter.append(found)
    reference = f"seed [{SEED}, {VEHICLE_ONLY}, 0, dataset]"
    label = (
        f"vehicle-only data in 4 mock dose groups of {VEHICLE_ONLY_REPLICATES} "
        f"samples (Lee et al. 2024), {datasets} datasets"
    )
    return [
        (
            f"{label}: the default gate, cell types with a gene passing",
            reference,
            share_text(np.array(gate, dtype=float))[2],
            "n/a",
            "info",
        ),
        (
            f"{label}: a gene at an unadjusted Williams-type p-value below 0.05 and a "
            f"{FOLD_CHANGE:g}-fold change at a dose, cell types with one",
            reference,
            share_text(np.array(prefilter, dtype=float))[2],
            "n/a",
            "info",
        ),
    ]


# The shape rows.


def shape_rows(cell: DesignCell) -> list[tuple[str, ...]]:
    """How often the best curve of each set and mode finds the true shape, and the
    benchmark doses of the "epa" curves, on the power datasets of a design cell."""
    found: dict[tuple[str, str, str], list[float]] = {}
    potency = Potency()
    for dataset in range(POWER_DATASETS):
        cube, truth = series(cell, POWER, dataset)
        for model_set, variance in CURVE_RUNS:
            options = (
                {"bmr": BenchmarkResponse.absolute(POTENCY_CHANGE)}
                if model_set == "epa"
                else {}
            )
            result = analyse(cube, models=model_set, variance=variance, **options)
            for group in result.groups:
                if model_set == "epa":
                    potency.add(variance, dataset, result, group, truth)
                fits = result.curve_fits(group)
                table = truth.table(group)
                info = {
                    gene: (shape, change)
                    for gene, shape, change in zip(
                        table["feature_id"].tolist(),
                        table["shape"].tolist(),
                        table["max_log2_change"].tolist(),
                        strict=True,
                    )
                }
                genes = result.table(group, "curves")["feature_id"][
                    :: len(fits.models)
                ].tolist()
                # Each best curve's change from dose 0 to the top dose.
                ends = result.curve(
                    group, model="best", at=[0.0, float(fits.doses[-1])]
                )
                change_of = dict(
                    zip(
                        ends["feature_id"][::2].tolist(),
                        np.diff(
                            np.asarray(ends["log2_fold_change"]).reshape(-1, 2), axis=1
                        )[:, 0].tolist(),
                        strict=True,
                    )
                )
                key = (model_set, variance)
                for g, gene in enumerate(genes):
                    shape, change = info[gene]
                    best = fits.best[g]
                    if best < 0:
                        continue
                    turns = bool(np.isfinite(fits.turn_dose[g, best]))
                    alone = bool(np.isfinite(fits.turn_driven_by_dose[g]))
                    kind = (
                        "bell"
                        if shape == "log_gauss_probit"
                        else "flat"
                        if change == 0
                        else "monotone"
                    )
                    if model_set == "dromics":
                        found.setdefault((*key, f"turns {kind}"), []).append(turns)
                        if turns:
                            found.setdefault((*key, f"alone {kind}"), []).append(alone)
                    if change == 0:
                        continue
                    members = CONTAINING[model_set].get(shape, ())
                    found.setdefault((*key, "contains"), []).append(
                        fits.models[best] in members
                    )
                    weight = sum(
                        fits.akaike_weight[g, fits.models.index(m)]
                        for m in members
                        if m in fits.models
                        and np.isfinite(fits.akaike_weight[g, fits.models.index(m)])
                    )
                    found.setdefault((*key, "weight"), []).append(weight)
                    # The direction of the best fit: the sign of its change from the
                    # vehicle's mean to the top dose, or a turn for a bell.
                    if kind == "bell":
                        right = turns
                    else:
                        right = (
                            np.sign(change_of[gene]) == np.sign(change) and not turns
                        )
                    found.setdefault((*key, "direction"), []).append(bool(right))
    rows = []
    source = f"seed [{SEED}, {POWER}, {cell.number}, dataset]"
    for model_set, variance in CURVE_RUNS:
        key = (model_set, variance)
        text = (
            f"best model contains the true shape {np.mean(found[(*key, 'contains')]):.3f}, "
            f"direction or turn right {np.mean(found[(*key, 'direction')]):.3f}, mean "
            f"Akaike weight on the true family {np.mean(found[(*key, 'weight')]):.3f}"
        )
        if model_set == "dromics":
            turning = "; ".join(
                f"{kind}: turn {np.mean(found[(*key, f'turns {kind}')]):.3f}, of which driven "
                f"by one dose group {np.mean(found.get((*key, f'alone {kind}'), [np.nan])):.3f}"
                for kind in ("bell", "monotone", "flat")
            )
            text += f"; best fits that turn, {turning}"
        rows.append(
            (
                f"{cell.label}: curves of {model_set!r}, {variance} variance, "
                f"{len(found[(*key, 'contains')]):,} responsive genes of "
                f"{POWER_DATASETS} datasets",
                source,
                text,
                "n/a",
                "info",
            )
        )
    return rows + potency.rows(cell)


# The potency rows.


@dataclass
class Potency:
    """Per variance mode, one record per dataset of the genes whose true shape lies in
    the "epa" set and whose true BMD at a 1.1-fold change lies within the doses: their
    expected counts per profile at the vehicle, whether the best model contains the
    true shape, the estimated BMD, BMDL and BMDU, the true BMD, and the 1 SD BMD with
    the true BMD at one SD of true expression."""

    records: dict[str, list[dict[str, np.ndarray]]] = field(default_factory=dict)

    def add(
        self, variance: str, dataset: int, result: Any, group: str, truth: Any
    ) -> None:
        fits = result.curve_fits(group)
        doses = result.benchmark_doses(group)
        table = truth.table(group)
        genes = result.table(group, "curves")["feature_id"][
            :: len(fits.models)
        ].tolist()
        position = {gene: i for i, gene in enumerate(table["feature_id"].tolist())}
        rows = np.array([position[gene] for gene in genes], dtype=np.int64)
        shape = np.asarray(table["shape"])[rows]
        true = np.asarray(table["bmd_fold_change"], dtype=float)[rows]
        positive = fits.doses[fits.doses > 0.0]
        with np.errstate(invalid="ignore"):
            kept = (
                np.isin(shape, IN_EPA_SET)
                & (true >= positive[0])
                & (true <= fits.doses[-1])
            )
        best = [fits.models[b] if b >= 0 else "" for b in fits.best]
        contains = np.array(
            [best[g] in CONTAINING["epa"].get(shape[g], ()) for g in range(len(genes))]
        )
        counts = np.asarray(table["counts_per_cell"], dtype=float)[rows] * CELLS
        # The 1 SD comparison is the last BMR.
        record = {
            "counts": counts[kept],
            "contains": contains[kept],
            "true": true[kept],
            "bmd": doses.bmd[0][kept],
            "bmdl": doses.bmdl[0][kept],
            "bmdu": doses.bmdu[0][kept],
            "bmd_sd": doses.bmd[-1][kept],
            "true_sd": np.asarray(table["bmd_expression_sd"], dtype=float)[rows][kept],
        }
        per_dataset = self.records.setdefault(variance, [])
        if len(per_dataset) <= dataset:
            per_dataset.append(record)
        else:
            per_dataset[dataset] = {
                key: np.concatenate([per_dataset[dataset][key], value])
                for key, value in record.items()
            }

    def rows(self, cell: DesignCell) -> list[tuple[str, ...]]:
        out = []
        source = f"seed [{SEED}, {POWER}, {cell.number}, dataset]"
        for variance, datasets in self.records.items():
            every = {
                key: np.concatenate([d[key] for d in datasets]) for key in datasets[0]
            }
            found = np.isfinite(every["bmd"])

            def coverage(select: Any) -> str:
                """One-sided BMDL and BMDU coverage of the genes with a BMD, with the
                SE between datasets and the binomial bound."""
                parts = []
                for bound, holds in (
                    ("BMDL", lambda d: d["bmdl"] <= d["true"]),
                    ("BMDU", lambda d: d["bmdu"] >= d["true"]),
                ):
                    hits, shares = [], []
                    for d in datasets:
                        mask = select(d) & np.isfinite(d["bmd"])
                        if mask.any():
                            hits.append(holds(d)[mask])
                            shares.append(float(np.mean(holds(d)[mask])))
                    pooled = np.concatenate(hits) if hits else np.zeros(0)
                    if pooled.size == 0:
                        return "no genes"
                    share = float(np.mean(pooled))
                    between = (
                        float(np.std(shares, ddof=1) / np.sqrt(len(shares)))
                        if len(shares) > 1
                        else float("nan")
                    )
                    binomial = float(np.sqrt(0.95 * 0.05 / pooled.size))
                    parts.append(
                        f"{bound} {share:.3f} (SE {between:.3f}; binomial {binomial:.3f})"
                    )
                count = int(
                    sum(
                        np.count_nonzero(select(d) & np.isfinite(d["bmd"]))
                        for d in datasets
                    )
                )
                return ", ".join(parts) + f", {count:,} genes"

            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.log10(every["bmd"][found] / every["true"][found])
                ok_sd = np.isfinite(every["bmd_sd"]) & np.isfinite(every["true_sd"])
                ratio_sd = np.log10(every["bmd_sd"][ok_sd] / every["true_sd"][ok_sd])
            q1, median, q3 = np.quantile(ratio, [0.25, 0.5, 0.75])
            s1, sd_median, s3 = np.quantile(ratio_sd, [0.25, 0.5, 0.75])
            strata = "; ".join(
                f"counts per profile {low:g} to {high:g}: "
                + coverage(
                    lambda d, low=low, high=high: (
                        (d["counts"] >= low) & (d["counts"] < high)
                    )
                )
                if np.isfinite(high)
                else f"counts per profile {low:g} or more: "
                + coverage(lambda d, low=low: d["counts"] >= low)
                for low, high in COUNT_STRATA
            )
            text = (
                f"BMD found {np.mean(found):.3f}; log10(BMD / true BMD) median "
                f"{median:+.3f}, interquartile range {q1:+.3f} to {q3:+.3f}; coverage "
                f"{coverage(lambda d: np.ones(d['bmd'].size, dtype=bool))}; {strata}; "
                "best model contains the true shape: "
                f"{coverage(lambda d: d['contains'])}; it does not: "
                f"{coverage(lambda d: ~d['contains'])}; 1 SD BMD against the true BMD at "
                f"one SD of true expression, log10 median {sd_median:+.3f}, "
                f"interquartile range {s1:+.3f} to {s3:+.3f} (bias only)"
            )
            out.append(
                (
                    f"{cell.label}: benchmark doses of 'epa', {variance} variance, at a "
                    f"1.1-fold change, {every['bmd'].size:,} genes of the set's shapes "
                    f"with a true BMD within the doses, {len(datasets)} datasets",
                    source,
                    text,
                    "n/a",
                    "info",
                )
            )
        return out


# The straight-line rows.


def line_rows() -> list[tuple[str, ...]]:
    """Coverage of one-sided BMDLs of straight lines with normal errors against their
    exact references: Fieller's t-based bound for an absolute change, and for a
    multiple of the SD the bound that inverts the noncentral t of the slope over its
    standard error; with benchmark_doses's profile-likelihood bounds beside."""
    from scipy.stats import chi2, nct
    from scipy.stats import t as student_t

    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.potency import benchmark_doses

    doses = np.repeat(np.array(DOSES[9]), 3)
    n = doses.size
    sxx = float(np.sum((doses - doses.mean()) ** 2))
    quantile = float(student_t.ppf(0.95, n - 2))
    rows = []
    for case, true_bmd in enumerate(LINE_BMDS):
        rng = np.random.default_rng([SEED, LINES, case, 0])
        slope = LINE_CHANGE / true_bmd
        values = (
            6.0 + slope * doses[:, None] + rng.normal(0.0, LINE_SD, (n, LINE_DATASETS))
        )
        estimate = (doses - doses.mean()) @ (values - values.mean(axis=0)) / sxx
        residuals = (
            values - values.mean(axis=0) - np.outer(doses - doses.mean(), estimate)
        )
        s = np.sqrt(np.sum(residuals**2, axis=0) / (n - 2))
        error = s / np.sqrt(sxx)
        # Fieller: the slope's upper t bound gives the BMDL of a change R.
        fieller = LINE_CHANGE / (estimate + quantile * error)
        # The SD-based BMD is v / (b1 / sigma); the slope over its standard error has
        # a noncentral t distribution on n - 2 degrees of freedom with noncentrality
        # (b1 / sigma) sqrt(Sxx), whose upper 95% bound gives the BMDL.
        t_observed = estimate / error
        low, high = np.full(LINE_DATASETS, -50.0), np.full(LINE_DATASETS, 200.0)
        for _ in range(100):
            middle = 0.5 * (low + high)
            above = nct.cdf(t_observed, n - 2, middle) > 0.05
            low = np.where(above, middle, low)
            high = np.where(above, high, middle)
        noncentral = LINE_SD_MULTIPLE * np.sqrt(sxx) / (0.5 * (low + high))
        true_sd_bmd = LINE_SD_MULTIPLE * LINE_SD / slope
        # The bound of a change held at 1.349 times the vehicle replicates' sample SD,
        # which ignores that SD's error: the line's profile bound of a fixed change R
        # is R / (b1 + sqrt((exp(chi2 / n) - 1) RSS / Sxx)).
        vehicle_sd = np.std(values[doses == 0.0], axis=0, ddof=1)
        rss = np.sum(residuals**2, axis=0)
        widening = np.sqrt(np.expm1(float(chi2.ppf(0.90, 1)) / n) * rss / sxx)
        held = LINE_SD_MULTIPLE * vehicle_sd / (estimate + widening)
        fits = fit_dose_curves(doses, values, models="epa")
        found = benchmark_doses(
            fits,
            [
                BenchmarkResponse.absolute(LINE_CHANGE),
                BenchmarkResponse.sd(LINE_SD_MULTIPLE),
            ],
            model="linear",
        )
        for label, reference, truth, row in (
            (f"a change of {LINE_CHANGE:g}", fieller, true_bmd, 0),
            (f"{LINE_SD_MULTIPLE:g} SD", noncentral, true_sd_bmd, 1),
        ):
            share = float(np.mean(reference <= truth))
            standard = float(np.sqrt(0.95 * 0.05 / LINE_DATASETS))
            ours_low = float(np.mean(found.bmdl[row] <= truth))
            ours_high = float(np.mean(found.bmdu[row] >= truth))
            exact = "Fieller's t-based bound" if row == 0 else "the noncentral t bound"
            rows.append(
                (
                    f"Straight lines with normal errors, K = 9 x 3, SD {LINE_SD:g}, true "
                    f"BMD {truth:g} at {label}, {LINE_DATASETS:,} datasets: {exact}",
                    f"seed [{SEED}, {LINES}, {case}, 0]",
                    f"BMDL coverage {share:.4f} (SE {standard:.4f})",
                    "0.95 within 4 SE",
                    "pass" if abs(share - 0.95) <= 4 * standard else "FAIL",
                )
            )
            rows.append(
                (
                    f"Straight lines, true BMD {truth:g} at {label}: benchmark_doses's "
                    "profile-likelihood bounds of the line",
                    f"seed [{SEED}, {LINES}, {case}, 0]",
                    f"BMDL coverage {ours_low:.4f}, BMDU coverage {ours_high:.4f} (SE "
                    f"{standard:.4f})",
                    "n/a",
                    "info",
                )
            )
        rows.append(
            (
                f"Straight lines, true BMD {true_sd_bmd:g} at {LINE_SD_MULTIPLE:g} SD: "
                f"the bound of a change held at {LINE_SD_MULTIPLE:g} times the vehicle "
                "replicates' sample SD, which benchmark_doses does not compute",
                f"seed [{SEED}, {LINES}, {case}, 0]",
                f"BMDL coverage {float(np.mean(held <= true_sd_bmd)):.4f} (SE "
                f"{np.sqrt(0.95 * 0.05 / LINE_DATASETS):.4f})",
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
    parser.add_argument(
        "--contrast-datasets",
        type=int,
        default=200,
        help="null datasets per design cell of the multiple contrast tests (200)",
    )
    parser.add_argument(
        "--check-datasets",
        type=int,
        default=100,
        help="null datasets per design cell of the permutation checks (100)",
    )
    parser.add_argument(
        "--vehicle-only-datasets",
        type=int,
        default=VEHICLE_ONLY_DATASETS,
        help="datasets of the vehicle-only design (200)",
    )
    arguments = parser.parse_args()
    if arguments.check_datasets >= arguments.null_datasets:
        parser.error(
            "--check-datasets must be below --null-datasets: the checks are compared "
            "with the null datasets they did not run on"
        )
    rows = []
    for cell in grid():
        rows += null_rows(cell, arguments.null_datasets, arguments.check_datasets)
        rows += power_rows(cell)
        rows += gate_power_rows(cell)
        rows += contrast_null_rows(cell, arguments.contrast_datasets)
        rows += contrast_power_rows(cell)
        if cell.groups == 9:
            rows += shape_rows(cell)
    rows += vehicle_only_rows(arguments.vehicle_only_datasets)
    rows += band_rows()
    rows += line_rows()
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    failed = sum(row[4] == "FAIL" for row in rows)
    print(f"{len(rows)} checks, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
