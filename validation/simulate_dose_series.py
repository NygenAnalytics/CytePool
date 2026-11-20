"""Check simulate_dose_series and DoseTruth against mpmath: true curves, EC50s and
benchmark doses, the moments of cells and counts, and series that seeds and threads
leave identical.

Four series from fixed seeds: ``ladder``, the vehicle and eight doses from 0.01 to 30
ug/kg as in the TCDD liver series, 3 replicates, two cell types of 500 and 30 cells
per profile, every one of 1,000 genes responsive (seed 20261127); ``half_log``, the
vehicle and 0.1 to 3.16 uM in half-log steps, 2 replicates from donors crossed with
dose (``donor_sd=0.2``), every one of 400 genes responsive (20261128); ``moments``,
the doses 0, 0.1, 1 and 10 with 10,000 replicates each, 30 and 300 cells per profile,
the second cell type at 70% and 40% of its cells at the two top doses, half of 50
genes responsive (20261129); and ``donors``, the same with 10,000 donors crossed with
dose (``donor_sd=0.25``, 20261130).

mpmath, run in a separate process at 50 digits on the exact binary values of the
truth tables' parameters (``--reference DIRECTORY``, which imports mpmath but not
CytePool), evaluates the documented formula of each shape at 0, at each dose, at the
geometric midpoints of neighbouring doses, at a tenth of the lowest positive dose and
at ten times the top dose; finds each EC50 as the root of |f| = |max_log2_change| / 2 and
each benchmark dose as the lowest root of |f| = c, by bisection on log10 dose, at c =
``expression_sd``, 1.349 ``expression_sd`` (ETAP's multiple), log2(1.1), 0.5 and
|max_log2_change|; and computes the exact moments of the count model to the fourth: the
Poisson-lognormal moments of the cells per profile, the marginal moments of the counts
through the factorial moments of the negative binomial distribution, and the
covariance of one donor's counts at two doses. A curve that approaches its plateau
never reaches a change of |max_log2_change| or more, and a peak or a step reaches a change
up to |max_log2_change|; within those rules the roots come from bisection, not from the
closed forms that CytePool uses. Thirteen curves with a change equal to their
|max_log2_change|, and flat genes, are added for the fixtures.

Rows: the true curves by shape (exact, 1e-10 of |max_log2_change|, and 0 at dose 0); the
EC50s (exact, 1e-10 relative); the benchmark doses by shape (exact, 1e-10 relative)
and where they are NaN (identical); the responsive genes and the counts of each shape,
maximal change and direction against the documented allocation, coded here
(identical); the shares of 1.1-fold benchmark doses below and above the doses for the
shapes placed by them (calibrated against a tenth, within 4 binomial standard errors)
and the bells' peaks within the doses (identical); the means and variances of the
cells and counts per profile and the covariances of donors (calibrated, within 4
standard errors: of a mean from the exact variance, of a variance from the exact
fourth central moment, of a covariance from the spread of the products); two models
the counts must not follow, a dispersion of 0.5 for every profile and effects in
natural-log units (controls: some comparison must exceed 4 standard errors); the four
series at 1, 2 and 8 numba threads in three processes, an integer seed against the
generator it seeds, generators in the same state, and the truth under other designs
(identical); and, for information, the z-scores, the shares of each shape's benchmark
doses below and above the doses, where the EC50s of the placed shapes fall at 9 and 4
dose groups, how many doses see each bell at four evenly and two unevenly spaced
designs, the SD of observed log2-CPM over ``expression_sd`` and the observed change at
the top dose over the true one, by expected count, the shift of the expected library
size along dose, and the run time of one series.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree, and mpmath for the reference.
    R: none.

Usage:
    PYTHONPATH=. python validation/simulate_dose_series.py
    PYTHONPATH=. python validation/simulate_dose_series.py --write-fixtures
"""

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "simulate_dose_series"
)
EXACT = 1e-10
# Calibrated rows: within this many standard errors.
BOUND = 4.0
DIGITS = 50
SHAPES = (
    "hill",
    "exponential",
    "linear",
    "power",
    "log_gauss_probit",
    "frechet",
    "step",
)
APPROACHED = ("hill", "exponential", "frechet")
PLACED = ("hill", "exponential", "frechet", "step")
PARAMETERS = ("max_log2_change", "ec50", "slope", "peak_dose", "peak_width")
LADDER = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
HALF_LOG = [0.0, 0.1, 0.316, 1.0, 3.16]
MOMENT_DOSES = [0.0, 0.1, 1.0, 10.0]
COMPOSITION = [1.0, 1.0, 0.7, 0.4]
SERIES: dict[str, dict[str, Any]] = {
    "ladder": {
        "doses": LADDER,
        "replicates": 3,
        "cell_types": ["hepatocyte", "stellate"],
        "cells": [500.0, 30.0],
        "n_genes": 1000,
        "responsive": 1.0,
        "rng": 20261127,
    },
    "half_log": {
        "doses": HALF_LOG,
        "replicates": 2,
        "n_genes": 400,
        "responsive": 1.0,
        "donor_sd": 0.2,
        "unit": "uM",
        "metric": "nominal",
        "rng": 20261128,
    },
    "moments": {
        "doses": MOMENT_DOSES,
        "replicates": 10_000,
        "cells": [30.0, 300.0],
        "n_genes": 50,
        "responsive": 0.5,
        "composition": {"cell_type_2": COMPOSITION},
        "rng": 20261129,
    },
    "donors": {
        "doses": MOMENT_DOSES,
        "replicates": 10_000,
        "cells": [30.0, 300.0],
        "n_genes": 50,
        "responsive": 0.5,
        "composition": {"cell_type_2": COMPOSITION},
        "donor_sd": 0.25,
        "rng": 20261130,
    },
}
# The count model's constants, as simulate_dose_series documents them.
CELLS_SD, DEPTH_SD, CELL_DISPERSION = 0.3, 0.2, 0.5
BIOLOGICAL_SD = 0.25
DEFAULT_SHARES = {
    "hill": 0.30,
    "exponential": 0.15,
    "linear": 0.05,
    "power": 0.10,
    "log_gauss_probit": 0.20,
    "frechet": 0.10,
    "step": 0.10,
}
DEFAULT_CHANGES = (0.5, 1.0, 2.0)
# The genes of each cell type whose count moments are compared: four flat and four
# responsive, at these quantiles of the ranks of counts_per_cell.
RANKS = (0.1, 0.4, 0.7, 0.95)
# The designs at which the bells' visibility is reported, with their seeds: four whose
# positive doses are evenly or nearly evenly spaced in log dose, and two that are not.
BELL_DESIGNS = {
    "3 dose groups, 0, 1 and 10": ([0.0, 1.0, 10.0], 20261141),
    "4 dose groups, 0 and 0.1 to 10 a decade apart": ([0.0, 0.1, 1.0, 10.0], 20261142),
    "6 dose groups, 0 and 0.01 to 30": ([0.0, 0.01, 0.1, 1.0, 10.0, 30.0], 20261143),
    "9 dose groups, the ladder": (LADDER, 20261144),
    "5 dose groups, 0, 1, 3, 10 and 100": ([0.0, 1.0, 3.0, 10.0, 100.0], 20261152),
    "6 dose groups, 0, 1, 2, 3, 4 and 1000": (
        [0.0, 1.0, 2.0, 3.0, 4.0, 1000.0],
        20261153,
    ),
}
# The designs at which the EC50s of the placed shapes are located, with their seeds.
EC50_DESIGNS = {
    "9 dose groups, the ladder": (LADDER, 20261154),
    "4 dose groups, 0 and 0.1 to 10 a decade apart": ([0.0, 0.1, 1.0, 10.0], 20261155),
}
THREADS = (1, 8)
# Curves per shape kept in the fixtures: the largest differences and a random sample.
FIXTURE_CURVES = 12
FIXTURE_MOMENTS = 10


# ---------------------------------------------------------------------------------------
# The reference: mpmath, without CytePool
# ---------------------------------------------------------------------------------------


def reference_main(directory: Path) -> None:
    """Read inputs.npz from the directory; write reference.npz and versions.json."""
    import mpmath as mp

    mp.mp.dps = DIGITS
    inputs = dict(np.load(directory / "inputs.npz"))
    ln2 = mp.log(2)

    def curve(case: tuple[Any, ...], d: Any) -> Any:
        shape, m, e, n, p, w, top = case
        if d == 0:
            return mp.mpf(0)
        if shape == "hill":
            return m / (1 + (e / d) ** n)
        if shape == "exponential":
            return m * (1 - mp.power(2, -d / e))
        if shape == "linear":
            return m * d / top
        if shape == "power":
            return m * (d / top) ** n
        if shape == "log_gauss_probit":
            return m * mp.exp(-((mp.log10(d / p) / w) ** 2) / 2)
        if shape == "frechet":
            return m * mp.power(2, -((e / d) ** n))
        if shape == "step":
            return m if d >= e else mp.mpf(0)
        return mp.mpf(0)

    def lowest_root(case: tuple[Any, ...], c: Any) -> float:
        """The lowest dose at which |f| reaches c, by bisection on log10 dose."""
        shape, m = case[0], case[1]
        if shape == "flat":
            return math.nan
        if shape in APPROACHED and not abs(m) > c:
            return math.nan
        if shape in ("log_gauss_probit", "step") and not abs(m) >= c:
            return math.nan
        lo = mp.mpf(-40)
        hi = mp.log10(case[4]) if shape == "log_gauss_probit" else mp.mpf(40)
        if (
            not abs(curve(case, mp.power(10, lo)))
            < c
            <= abs(curve(case, mp.power(10, hi)))
        ):
            raise ArithmeticError(f"no bracket for {case} at {c}")
        while hi - lo > mp.mpf(10) ** -30:
            middle = (lo + hi) / 2
            if abs(curve(case, mp.power(10, middle))) >= c:
                hi = middle
            else:
                lo = middle
        return float(mp.power(10, hi))

    columns = [inputs[key] for key in ("shape", *PARAMETERS, "top")]
    cases = [
        (str(shape), *(mp.mpf(float(value)) for value in values))
        for shape, *values in zip(*columns)
    ]
    out: dict[str, Any] = {
        "grid_value": [
            float(curve(cases[i], mp.mpf(float(d))))
            for i, d in zip(inputs["grid_case"], inputs["grid_dose"])
        ],
        "ec50": [
            lowest_root(case, abs(case[1]) / 2) if case[0] in PLACED else math.nan
            for case in cases
        ],
        "root": [
            lowest_root(cases[i], mp.mpf(float(c)))
            for i, c in zip(inputs["root_case"], inputs["root_change"])
        ],
    }

    def cells_raw(m: Any) -> list[Any]:
        """E[n^i], i = 0 to 4, of n ~ Poisson(m exp(s x - s^2 / 2)), x standard
        normal: the Poisson moments in Stirling numbers of the second kind, and
        the log-normal moments of the rate."""

        def stirling(i: int, j: int) -> Any:
            total = sum((-1) ** (j - t) * math.comb(j, t) * t**i for t in range(j + 1))
            return mp.mpf(total) / math.factorial(j)

        s2 = mp.mpf(CELLS_SD) ** 2
        rate = [m**j * mp.exp(s2 * j * (j - 1) / 2) for j in range(5)]
        return [sum(stirling(i, j) * rate[j] for j in range(i + 1)) for i in range(5)]

    def counts_raw(
        m: Any, a: Any, sigma: Any, *, per_cell: bool = True, base: Any = ln2
    ) -> list[Any]:
        """Raw moments 1 to 4 of the counts, from the factorial moments E[(Y)_k] =
        a^k E[e^k] E[2^(k b)] E[prod over j < k of (n + 0.5 j)]; with ``per_cell``
        False, a dispersion of 0.5 for every profile, E[n^k] prod (1 + 0.5 j); with
        ``base`` 1, effects in natural-log units."""
        n_raw = cells_raw(m)
        factorial = []
        for k in range(1, 5):
            if per_cell:
                # The coefficients of prod over j < k of (n + 0.5 j), in powers of n.
                coefficients = [mp.mpf(1)]
                for j in range(k):
                    shifted = [mp.mpf(0), *coefficients]
                    for i, value in enumerate(coefficients):
                        shifted[i] += value * CELL_DISPERSION * j
                    coefficients = shifted
                cells = mp.fsum(c * n_raw[i] for i, c in enumerate(coefficients))
            else:
                cells = n_raw[k] * mp.fprod(1 + CELL_DISPERSION * j for j in range(k))
            depth = mp.exp(mp.mpf(DEPTH_SD) ** 2 * k * (k - 1) / 2)
            effect = mp.exp((k * sigma * base) ** 2 / 2)
            factorial.append(a**k * depth * effect * cells)
        f1, f2, f3, f4 = factorial
        return [f1, f2 + f1, f3 + 3 * f2 + f1, f4 + 6 * f3 + 7 * f2 + f1]

    def central(raw: list[Any]) -> list[float]:
        """The mean, the variance and the fourth central moment."""
        m1, m2, m3, m4 = raw
        return [
            float(m1),
            float(m2 - m1**2),
            float(m4 - 4 * m3 * m1 + 6 * m2 * m1**2 - 3 * m1**4),
        ]

    out["cells_moments"] = [
        central(cells_raw(mp.mpf(float(m)))[1:]) for m in inputs["cells_m"]
    ]
    for key, options in (
        ("count_moments", {}),
        ("count_moments_dispersion", {"per_cell": False}),
        ("count_moments_natural", {"base": mp.mpf(1)}),
    ):
        out[key] = [
            central(counts_raw(*(mp.mpf(float(v)) for v in values), **options))
            for values in zip(
                inputs["count_m"], inputs["count_a"], inputs["count_sigma"]
            )
        ]
    covariances = []
    for values in zip(
        *(inputs[f"cov_{key}"] for key in ("m1", "m2", "a1", "a2", "sigma", "tau"))
    ):
        m1, m2, a1, a2, sigma, tau = (mp.mpf(float(v)) for v in values)
        shared = mp.exp(2 * (tau * ln2) ** 2) - mp.exp((tau * ln2) ** 2)
        covariances.append(
            float(a1 * a2 * m1 * m2 * mp.exp((sigma * ln2) ** 2) * shared)
        )
    out["covariance"] = covariances
    np.savez(directory / "reference.npz", **{k: np.asarray(v) for k, v in out.items()})
    versions = {"python": sys.version.split()[0], "mpmath": mp.__version__}
    (directory / "versions.json").write_text(json.dumps(versions))


def run_reference(
    inputs: dict[str, np.ndarray],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Run the reference in another process on the inputs."""
    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        np.savez(directory / "inputs.npz", **inputs)
        subprocess.run(
            [sys.executable, __file__, "--reference", str(directory)], check=True
        )
        reference = dict(np.load(directory / "reference.npz"))
        versions: dict[str, str] = json.loads((directory / "versions.json").read_text())
    return reference, versions


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------


def simulate(name: str, **changes: Any) -> Any:
    """A series of SERIES, with some of its arguments changed."""
    from cytepool import simulate_dose_series

    arguments = {**SERIES[name], **changes}
    return simulate_dose_series(arguments.pop("doses"), **arguments)


def table_digest(table: Any) -> str:
    """SHA-256 of a table's column names and values."""
    digest = hashlib.sha256()
    for name in table.columns:
        column = table[name]
        digest.update(name.encode() + b"\0")
        if column.dtype.kind == "O":
            digest.update("\0".join(column.tolist()).encode())
        else:
            digest.update(np.ascontiguousarray(column).tobytes())
    return digest.hexdigest()


def series_digests(simulated: dict[str, Any]) -> dict[str, Any]:
    """The fingerprint of each series' cube and the digests of its truth's tables."""
    return {
        name: {
            "fingerprint": cube.fingerprint(),
            "tables": {c: table_digest(truth.table(c)) for c in truth.cell_types},
        }
        for name, (cube, truth) in simulated.items()
    }


def simulate_main(directory: Path) -> None:
    """Write the digests of the four series, as this process computes them."""
    import numba

    simulated = {name: simulate(name) for name in SERIES}
    record = {"threads": numba.get_num_threads(), "series": series_digests(simulated)}
    (directory / "digests.json").write_text(json.dumps(record))


def digests_in_process(threads: int) -> dict[str, Any]:
    """The digests of the four series, computed by this script in another process
    with the given number of numba threads."""
    with tempfile.TemporaryDirectory() as tmp:
        environment = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
        subprocess.run(
            [sys.executable, __file__, "--simulate", tmp], check=True, env=environment
        )
        record: dict[str, Any] = json.loads((Path(tmp) / "digests.json").read_text())
    return record


# ---------------------------------------------------------------------------------------
# The allocation, coded from simulate_dose_series' documentation
# ---------------------------------------------------------------------------------------


def documented_allocation(
    count: int, shares: dict[str, float], changes: tuple[float, ...]
) -> Counter[tuple[str, float]]:
    """The number of genes of each shape and signed maximal change among ``count``
    responsive genes: shapes by the largest remainder of their shares, ties in the
    order given; the changes in turn within each shape; rises and falls in turn
    within each shape and change, the first a rise."""
    names = list(shares)
    quotas = [round(count * shares[name], 9) for name in names]
    seats = [math.floor(quota) for quota in quotas]
    order = sorted(range(len(names)), key=lambda i: (-(quotas[i] - seats[i]), i))
    for i in order[: count - sum(seats)]:
        seats[i] += 1
    found: Counter[tuple[str, float]] = Counter()
    for name, seat in zip(names, seats):
        for position in range(seat):
            change = changes[position % len(changes)]
            rise = (position // len(changes)) % 2 == 0
            found[name, change if rise else -change] += 1
    return found


# ---------------------------------------------------------------------------------------
# The printed checks
# ---------------------------------------------------------------------------------------


class Rows:
    """The printed checks, each with its reference, value, tolerance and outcome."""

    def __init__(self) -> None:
        self.rows: list[list[str]] = []
        self.failures = 0
        self.z_scores: list[float] = []

    def add(
        self,
        check: str,
        reference: str,
        value: str,
        tolerance: str,
        passed: bool | None,
        *,
        control: bool = False,
    ) -> None:
        """A check; ``passed`` None for information."""
        if passed is None:
            outcome = "information"
        elif not passed:
            outcome = "FAIL"
        else:
            outcome = "control" if control else "pass"
        self.failures += outcome == "FAIL"
        self.rows.append([check, reference, value, tolerance, outcome])

    def calibrated(self, check: str, reference: str, z: Any) -> None:
        """A family of calibrated comparisons, by their z-scores."""
        z = np.asarray(z, dtype=np.float64).ravel()
        self.z_scores += np.abs(z).tolist()
        largest = float(np.max(np.abs(z)))
        if z.size == 1:
            self.add(
                check, reference, f"z {z[0]:.2f}", f"{BOUND:g} SE", largest <= BOUND
            )
            return
        self.add(
            f"{check} ({z.size} comparisons)",
            reference,
            f"largest abs(z) {largest:.2f}",
            f"{BOUND:g} SE",
            largest <= BOUND,
        )

    def print(self) -> None:
        print("| check | reference | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


# ---------------------------------------------------------------------------------------
# Curves, EC50s and benchmark doses
# ---------------------------------------------------------------------------------------


def grid_of(doses: np.ndarray) -> np.ndarray:
    """0, each dose, the geometric midpoints of neighbouring positive doses, a tenth
    of the lowest positive dose and ten times the top dose."""
    positive = doses[1:]
    return np.concatenate(
        [
            doses,
            np.sqrt(positive[:-1] * positive[1:]),
            [positive[0] / 10.0, positive[-1] * 10.0],
        ]
    )


def series_cases(truths: dict[str, Any]) -> dict[str, np.ndarray]:
    """Every curve of the truths, its doses on the grid and the changes of its
    benchmark doses."""
    parts: dict[str, list[Any]] = {
        key: []
        for key in (
            "shape",
            *PARAMETERS,
            "top",
            "expression_sd",
            "bmd_expression_sd",
            "bmd_fold_change",
            "source",
        )
    }
    grid_case: list[int] = []
    grid_dose: list[float] = []
    root_case: list[int] = []
    root_change: list[float] = []
    for series, truth in truths.items():
        doses = np.asarray(truth.doses)
        grid = grid_of(doses)
        for cell_type in truth.cell_types:
            table = truth.table(cell_type)
            first = len(parts["shape"])
            for key in (
                "shape",
                *PARAMETERS,
                "expression_sd",
                "bmd_expression_sd",
                "bmd_fold_change",
            ):
                parts[key] += table[key].tolist()
            parts["top"] += [doses[-1]] * len(table)
            parts["source"] += [
                f"{series} {cell_type} {gene}" for gene in table["feature_id"].tolist()
            ]
            for i in range(len(table)):
                case = first + i
                grid_case += [case] * grid.size
                grid_dose += grid.tolist()
                sd = float(table["expression_sd"][i])
                reach = abs(float(table["max_log2_change"][i]))
                for change in (sd, 1.349 * sd, math.log2(1.1), 0.5, reach or 0.5):
                    root_case.append(case)
                    root_change.append(change)
    arrays = {key: np.asarray(values) for key, values in parts.items()}
    arrays["shape"] = arrays["shape"].astype(str)
    arrays["source"] = arrays["source"].astype(str)
    arrays["grid_case"] = np.asarray(grid_case)
    arrays["grid_dose"] = np.asarray(grid_dose)
    arrays["root_case"] = np.asarray(root_case)
    arrays["root_change"] = np.asarray(root_change)
    return arrays


def boundary_cases(first_case: int) -> dict[str, np.ndarray]:
    """Curves at a change equal to their |max_log2_change|, 0.6 of it and 1.5 times it,
    and a flat gene, for the fixtures: where each shape reaches the change."""
    rows = [
        ("hill", 0.5, 2.0, 1.5, math.nan, math.nan),
        ("hill", -0.25, 0.3, 0.8, math.nan, math.nan),
        ("exponential", 0.5, 2.0, math.nan, math.nan, math.nan),
        ("exponential", -1.0, 0.05, math.nan, math.nan, math.nan),
        ("linear", 0.5, math.nan, math.nan, math.nan, math.nan),
        ("power", -0.5, math.nan, 2.5, math.nan, math.nan),
        ("log_gauss_probit", 0.5, math.nan, math.nan, 1.0, 0.4),
        ("log_gauss_probit", -2.0, math.nan, math.nan, 0.05, 0.7),
        ("frechet", 0.5, 2.0, 1.2, math.nan, math.nan),
        ("frechet", -1.5, 0.02, 3.0, math.nan, math.nan),
        ("step", 0.5, 2.0, math.nan, math.nan, math.nan),
        ("step", -0.75, 7.0, math.nan, math.nan, math.nan),
        ("flat", 0.0, math.nan, math.nan, math.nan, math.nan),
    ]
    arrays = {
        key: np.asarray([row[i] for row in rows])
        for i, key in enumerate(("shape", *PARAMETERS))
    }
    arrays["shape"] = arrays["shape"].astype(str)
    arrays["top"] = np.full(len(rows), 30.0)
    arrays["expression_sd"] = np.full(len(rows), 0.25)
    arrays["bmd_expression_sd"] = np.full(len(rows), np.nan)
    arrays["bmd_fold_change"] = np.full(len(rows), np.nan)
    arrays["source"] = np.asarray([f"boundary {i}" for i in range(len(rows))])
    cases = np.arange(len(rows)) + first_case
    grid = grid_of(np.asarray(LADDER))
    arrays["grid_case"] = np.repeat(cases, grid.size)
    arrays["grid_dose"] = np.tile(grid, len(rows))
    reach = np.abs(arrays["max_log2_change"])
    at = np.where(reach > 0.0, reach, 0.5)
    arrays["root_case"] = np.concatenate([cases, cases, cases])
    arrays["root_change"] = np.concatenate([at, 0.6 * at, 1.5 * at])
    return arrays


def joined(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return {key: np.concatenate([first[key], second[key]]) for key in first}


def cytepool_curves(cases: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """DoseTruth's curves on the grid and benchmark doses at the changes, of truths
    built from the cases' parameters, one per top dose."""
    from cytepool import DoseTruth

    values = np.empty(cases["grid_case"].size)
    roots = np.empty(cases["root_case"].size)
    # Which of its changes each benchmark dose is, so that a call takes each gene once.
    rank = np.zeros(cases["root_case"].size, dtype=np.int64)
    seen: Counter[int] = Counter()
    for j, case in enumerate(cases["root_case"].tolist()):
        rank[j] = seen[case]
        seen[case] += 1
    for top in np.unique(cases["top"]):
        members = np.flatnonzero(cases["top"] == top)
        position = np.full(cases["shape"].size, -1)
        position[members] = np.arange(members.size)
        table = {
            "feature_id": [f"case_{i}" for i in members],
            **{
                key: cases[key][members]
                for key in ("shape", *PARAMETERS, "expression_sd")
            },
            "counts_per_cell": np.ones(members.size),
        }
        truth = DoseTruth(
            [0.0, top / 10.0, top],
            {"cases": table},
            unit="ug/kg",
            metric="administered",
        )
        points = np.flatnonzero(position[cases["grid_case"]] >= 0)
        doses, column = np.unique(cases["grid_dose"][points], return_inverse=True)
        curves = truth.curve("cases", at=doses)
        values[points] = curves[position[cases["grid_case"][points]], column]
        for r in np.unique(rank):
            asked = np.flatnonzero((rank == r) & (position[cases["root_case"]] >= 0))
            roots[asked] = truth.benchmark_doses(
                "cases",
                position[cases["root_case"][asked]],
                log2_change=cases["root_change"][asked],
            )
    return values, roots


def check_curves(
    rows: Rows, cases: dict[str, np.ndarray], reference: dict[str, np.ndarray]
) -> np.ndarray:
    """The rows of the curves, EC50s and benchmark doses; the largest difference of
    each case."""
    values, roots = cytepool_curves(cases)
    errors = np.zeros(cases["shape"].size)
    series = ~np.char.startswith(cases["source"], "boundary")
    scale = np.abs(cases["max_log2_change"])[cases["grid_case"]]
    difference = np.abs(values - reference["grid_value"])
    relative = np.where(
        scale > 0.0, difference / np.where(scale > 0.0, scale, 1.0), difference
    )
    np.maximum.at(errors, cases["grid_case"], relative)
    for shape in (*SHAPES, "flat"):
        genes = series & (cases["shape"] == shape)
        points = genes[cases["grid_case"]]
        largest = float(relative[points].max(initial=0.0))
        tolerance = 0.0 if shape == "flat" else EXACT
        rows.add(
            f"True curves, {shape}, {int(genes.sum()):,} genes at "
            f"{int(points.sum()):,} doses",
            "mpmath, the documented formula",
            f"{largest:.1e}",
            "0" if shape == "flat" else f"{EXACT:g} of max_log2_change",
            largest <= tolerance,
        )
    at_zero = cases["grid_dose"] == 0.0
    rows.add(
        f"True change at dose 0, {int(series.sum()):,} genes",
        "0",
        f"{int(np.count_nonzero(values[at_zero] != 0.0))} not 0",
        "0",
        bool(np.all(values[at_zero] == 0.0)),
    )
    for shape in PLACED:
        genes = np.flatnonzero(series & (cases["shape"] == shape))
        relative_ec50 = (
            np.abs(cases["ec50"][genes] - reference["ec50"][genes])
            / reference["ec50"][genes]
        )
        errors[genes] = np.maximum(errors[genes], relative_ec50)
        largest = float(relative_ec50.max(initial=0.0))
        rows.add(
            f"EC50s, {shape}, {genes.size:,} genes",
            "mpmath, root of abs(f) = abs(max_log2_change) / 2",
            f"{largest:.1e}",
            f"{EXACT:g}",
            largest <= EXACT,
        )
    theirs = reference["root"]
    both = np.isfinite(roots) & np.isfinite(theirs)
    relative_roots = np.zeros(roots.size)
    relative_roots[both] = np.abs(roots[both] - theirs[both]) / theirs[both]
    np.maximum.at(errors, cases["root_case"], relative_roots)
    in_series = series[cases["root_case"]]
    for shape in SHAPES:
        asked = in_series & both & (cases["shape"][cases["root_case"]] == shape)
        largest = float(relative_roots[asked].max(initial=0.0))
        rows.add(
            f"Benchmark doses, {shape}, {int(asked.sum()):,} at expression_sd, 1.349 "
            "expression_sd, log2(1.1), 0.5 and abs(max_log2_change)",
            "mpmath, lowest root of abs(f) = c",
            f"{largest:.1e}",
            f"{EXACT:g}",
            largest <= EXACT,
        )
    # The tables' columns: the first change of each curve is its expression_sd and
    # the third log2(1.1).
    for column, rank in (("bmd_expression_sd", 0), ("bmd_fold_change", 2)):
        asked = np.flatnonzero(in_series)[rank::5]
        stored = cases[column][cases["root_case"][asked]]
        root = theirs[asked]
        found = np.isfinite(stored) & np.isfinite(root)
        largest = float(
            (np.abs(stored[found] - root[found]) / root[found]).max(initial=0.0)
        )
        nan_differ = int(np.sum(np.isnan(stored) != np.isnan(root)))
        rows.add(
            f"The tables' {column}, {asked.size:,} genes",
            "mpmath, lowest root of abs(f) = c",
            f"{largest:.1e}; {nan_differ} NaN mismatches",
            f"{EXACT:g}",
            largest <= EXACT and nan_differ == 0,
        )
    mismatch = np.isnan(roots) != np.isnan(theirs)
    nan_count = int(np.isnan(theirs[in_series]).sum())
    rows.add(
        f"Benchmark doses that are NaN, {nan_count:,} of {int(in_series.sum()):,}: flat "
        "genes, and changes a curve never reaches",
        "mpmath",
        f"{int(mismatch[in_series].sum())} mismatches",
        "0",
        not mismatch[in_series].any(),
    )
    boundary = ~in_series
    rows.add(
        f"Benchmark doses of the {int((~series).sum())} added curves at abs(max_log2_change), "
        "0.6 and 1.5 times it",
        "mpmath",
        f"{int(mismatch[boundary].sum())} NaN mismatches, largest difference "
        f"{float(relative_roots[boundary].max(initial=0.0)):.1e}",
        f"0, {EXACT:g}",
        not mismatch[boundary].any()
        and float(relative_roots[boundary].max(initial=0.0)) <= EXACT,
    )
    errors[~series] = np.inf  # always kept in the fixtures
    return errors


# ---------------------------------------------------------------------------------------
# Allocation and placement
# ---------------------------------------------------------------------------------------


def check_allocation(rows: Rows, truths: dict[str, Any]) -> None:
    for name, truth in truths.items():
        settings = SERIES[name]
        mismatches = 0
        for k, cell_type in enumerate(truth.cell_types):
            table = truth.table(cell_type)
            n_genes = settings["n_genes"]
            share = settings["responsive"]
            responding = math.floor(round(share * n_genes, 9) + 0.5)
            found = Counter(
                (shape, change)
                for shape, change in zip(
                    table["shape"].tolist(), table["max_log2_change"].tolist()
                )
                if shape != "flat"
            )
            expected = documented_allocation(
                responding, DEFAULT_SHARES, DEFAULT_CHANGES
            )
            mismatches += sum((found - expected).values()) + sum(
                (expected - found).values()
            )
            mismatches += abs(int((table["shape"] != "flat").sum()) - responding)
        rows.add(
            f"Responsive genes of {name}, and their counts by shape, maximal change and "
            f"direction, {len(truth.cell_types)} cell types",
            "the documented allocation",
            f"{mismatches} mismatches",
            "0",
            mismatches == 0,
        )


def check_placement(rows: Rows, truths: dict[str, Any]) -> None:
    for name in ("ladder", "half_log"):
        truth = truths[name]
        doses = np.asarray(truth.doses)
        tables = [truth.table(c) for c in truth.cell_types]
        shape = np.concatenate([t["shape"] for t in tables])
        dose = np.concatenate([t["bmd_fold_change"] for t in tables])
        placed = np.isin(shape, PLACED)
        n = int(placed.sum())
        for side, share in (
            ("below the lowest dose", np.mean(dose[placed] < doses[1])),
            ("above the top dose", np.mean(dose[placed] > doses[-1])),
        ):
            z = (share - 0.1) / math.sqrt(0.1 * 0.9 / n)
            rows.calibrated(
                f"Share of 1.1-fold benchmark doses {side}, {name}, {n:,} hill, "
                f"exponential, frechet and step curves: {share:.3f}",
                "a tenth, binomial",
                [z],
            )
    outside = 0
    bells = 0
    for truth in truths.values():
        doses = np.asarray(truth.doses)
        for cell_type in truth.cell_types:
            table = truth.table(cell_type)
            peaks = table["peak_dose"][table["shape"] == "log_gauss_probit"]
            bells += peaks.size
            outside += int(np.sum((peaks < doses[1]) | (peaks > doses[-1])))
    rows.add(
        f"Peaks of the {bells:,} bells within the positive doses",
        "the documented placement",
        f"{outside} outside",
        "0",
        outside == 0,
    )


def information_shares(rows: Rows, truth: Any) -> None:
    """Where the 1.1-fold benchmark doses of each shape fall, in a default series."""
    doses = np.asarray(truth.doses)
    tables = [truth.table(c) for c in truth.cell_types]
    shape = np.concatenate([t["shape"] for t in tables])
    dose = np.concatenate([t["bmd_fold_change"] for t in tables])
    for name in SHAPES:
        chosen = dose[shape == name]
        rows.add(
            f"1.1-fold benchmark doses of the ladder's {chosen.size:,} {name} curves "
            "(every gene responsive, the default shares)",
            "information",
            f"{np.mean(chosen < doses[1]):.1%} below the lowest dose, "
            f"{np.mean(chosen > doses[-1]):.1%} above the top dose",
            "n/a",
            None,
        )
    responsive = shape != "flat"
    rows.add(
        f"1.1-fold benchmark doses of all {int(responsive.sum()):,} of the ladder's "
        "curves",
        "information",
        f"{np.mean(dose[responsive] < doses[1]):.1%} below, "
        f"{np.mean(dose[responsive] > doses[-1]):.1%} above",
        "n/a",
        None,
    )


def information_ec50s(rows: Rows) -> None:
    """Where the EC50s of the shapes placed by their benchmark dose fall: below the
    lowest dose, above the top dose, and above 5 times it, the bound of BMDS's Hill
    model on doses scaled to the top dose."""
    from cytepool import simulate_dose_series

    for design, (doses, seed) in EC50_DESIGNS.items():
        _, truth = simulate_dose_series(
            doses,
            replicates=1,
            cell_types=1,
            n_genes=4000,
            responsive=1.0,
            shapes=dict.fromkeys(PLACED, 0.25),
            rng=seed,
        )
        table = truth.table("cell_type_1")
        parts = []
        for shape in PLACED:
            ec50 = table["ec50"][table["shape"] == shape]
            parts.append(
                f"{shape} {np.mean(ec50 < doses[1]):.1%} below the lowest dose, "
                f"{np.mean(ec50 > doses[-1]):.1%} above the top dose, "
                f"{np.mean(ec50 > 5.0 * doses[-1]):.1%} above 5 times it"
            )
        rows.add(
            f"EC50s of 1,000 curves of each shape placed by its benchmark dose, at "
            f"{design}",
            "information",
            "; ".join(parts),
            "n/a",
            None,
        )


def information_bells(rows: Rows) -> None:
    """At how many positive doses each bell reaches half its peak, by design."""
    from cytepool import simulate_dose_series

    for design, (doses, seed) in BELL_DESIGNS.items():
        _, truth = simulate_dose_series(
            doses,
            replicates=1,
            cell_types=1,
            n_genes=2000,
            responsive=1.0,
            shapes={"log_gauss_probit": 1.0},
            rng=seed,
        )
        table = truth.table("cell_type_1")
        curves = truth.curve("cell_type_1", at=np.asarray(doses[1:]))
        relative = np.abs(curves) / np.abs(table["max_log2_change"])[:, None]
        seen = (relative >= 0.5).sum(axis=1)
        # A fall a method can see: the highest dose's change at least a quarter of
        # the peak below the largest change, which comes at a lower dose.
        falls = (np.argmax(relative, axis=1) < relative.shape[1] - 1) & (
            relative.max(axis=1) - relative[:, -1] >= 0.25
        )
        below = np.mean(table["bmd_fold_change"] < doses[1])
        rows.add(
            f"Bells at {design}, 2,000 genes",
            "information",
            f"half the peak reached at 2 or more doses {np.mean(seen >= 2):.0%}, at one "
            f"{np.mean(seen == 1):.0%}, at none {np.mean(seen == 0):.0%}; falls by a "
            f"quarter of the peak after its highest dose {np.mean(falls):.0%}; 1.1-fold "
            f"benchmark dose below the lowest dose {below:.0%}",
            "n/a",
            None,
        )


# ---------------------------------------------------------------------------------------
# Moments of cells and counts
# ---------------------------------------------------------------------------------------


def chosen_genes(table: Any) -> np.ndarray:
    """Four flat and four responsive genes, at spread ranks of counts_per_cell."""
    chosen = []
    for flat in (True, False):
        genes = np.flatnonzero((table["shape"] == "flat") == flat)
        order = genes[np.argsort(table["counts_per_cell"][genes], kind="stable")]
        chosen += [int(order[int(q * (order.size - 1))]) for q in RANKS]
    return np.asarray(chosen)


def moment_inputs(cube: Any, truth: Any, donor_sd: float) -> dict[str, Any]:
    """The expected cells, the factor a = counts_per_cell 2^f and the SD of each
    compared gene, type and dose, with the observed counts."""
    settings = SERIES["moments"]
    doses = np.asarray(MOMENT_DOSES)
    dose = np.asarray(cube.samples["dose"])
    found: dict[str, list[Any]] = {
        key: [] for key in ("m", "a", "sigma", "tau", "counts", "cells", "type", "dose")
    }
    for k, cell_type in enumerate(truth.cell_types):
        table = truth.table(cell_type)
        genes = chosen_genes(table)
        factors = COMPOSITION if cell_type == "cell_type_2" else [1.0] * doses.size
        changes = truth.curve(cell_type, genes, at=doses)
        counts = cube.layer("counts", cell_type)
        sd_factor = table["expression_sd"][genes] / math.hypot(BIOLOGICAL_SD, donor_sd)
        for j, d in enumerate(doses):
            rows = dose == d
            found["m"].append(np.full(genes.size, settings["cells"][k] * factors[j]))
            found["a"].append(table["counts_per_cell"][genes] * 2.0 ** changes[:, j])
            found["sigma"].append(sd_factor * BIOLOGICAL_SD)
            found["tau"].append(sd_factor * donor_sd)
            found["counts"].append(counts[rows][:, genes].astype(np.float64))
            found["cells"].append(cube.n_cells[rows, k].astype(np.float64))
            found["type"].append(k)
            found["dose"].append(j)
    return found


def z_of_moments(
    observed: np.ndarray, exact: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """z-scores of the sample mean and variance (columns of observations) against
    exact means, variances and fourth central moments (rows: mean, var, mu4)."""
    n = observed.shape[0]
    mean, variance, fourth = exact
    z_mean = (observed.mean(axis=0) - mean) / np.sqrt(variance / n)
    spread = np.sqrt((fourth - variance**2 * (n - 3) / (n - 1)) / n)
    z_variance = (observed.var(axis=0, ddof=1) - variance) / spread
    return z_mean, z_variance


# ---------------------------------------------------------------------------------------
# Information on the observed scale, the library sizes and the run time
# ---------------------------------------------------------------------------------------


def information_observed_sd(rows: Rows) -> None:
    """The SD of voom's log2-CPM between 2,000 vehicle replicates over expression_sd,
    by expected count at the vehicle."""
    from cytepool import simulate_dose_series

    edges = (0.0, 1.0, 5.0, 20.0, 100.0, math.inf)
    for i, cells in enumerate((30.0, 100.0, 500.0)):
        cube, truth = simulate_dose_series(
            LADDER,
            replicates=[2000] + [1] * (len(LADDER) - 1),
            cell_types=1,
            cells=cells,
            rng=20261148 + i,
        )
        counts = cube.layer("counts", "cell_type_1")[:2000].astype(np.float64)
        libraries = counts.sum(axis=1, keepdims=True)
        log_cpm = np.log2((counts + 0.5) / (libraries + 1.0) * 1e6)
        table = truth.table("cell_type_1")
        ratio = log_cpm.std(axis=0, ddof=1) / table["expression_sd"]
        expected = table["counts_per_cell"] * cells
        parts = []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = (expected >= low) & (expected < high)
            if chosen.sum() >= 5:
                label = (
                    f"{low:g} to {high:g}"
                    if math.isfinite(high)
                    else f"{low:g} or more"
                )
                parts.append(
                    f"{label}: {np.median(ratio[chosen]):.2f} ({int(chosen.sum())} genes)"
                )
        rows.add(
            f"SD of log2-CPM between 2,000 vehicle replicates over expression_sd, at "
            f"{cells:g} cells per profile, median by expected count",
            "information",
            "; ".join(parts),
            "n/a",
            None,
        )


def information_compression(rows: Rows) -> None:
    """The change of voom's log2-CPM from the vehicle to the top dose over the true
    change, by expected count at the vehicle: the prior count compresses the observed
    changes of genes with few counts."""
    from cytepool import simulate_dose_series

    doses = [0.0, 0.01, 0.1, 1.0, 10.0, 30.0]
    edges = (0.0, 1.0, 2.0, 5.0, 20.0, math.inf)
    for i, cells in enumerate((30.0, 100.0)):
        cube, truth = simulate_dose_series(
            doses,
            replicates=[2000, 1, 1, 1, 1, 2000],
            cell_types=1,
            cells=cells,
            n_genes=4000,
            responsive=0.2,
            shapes={"hill": 0.5, "exponential": 0.5},
            max_log2_changes=(1.0, 2.0),
            rng=20261156 + i,
        )
        counts = cube.layer("counts", "cell_type_1").astype(np.float64)
        libraries = counts.sum(axis=1, keepdims=True)
        log_cpm = np.log2((counts + 0.5) / (libraries + 1.0) * 1e6)
        dose = np.asarray(cube.samples["dose"])
        observed = log_cpm[dose == doses[-1]].mean(axis=0) - log_cpm[dose == 0.0].mean(
            axis=0
        )
        table = truth.table("cell_type_1")
        flat = table["shape"] == "flat"
        # The flat genes' median removes the shift of the library sizes, as TMM would.
        observed -= np.median(observed[flat])
        true = truth.curve("cell_type_1", at=[doses[-1]])[:, 0]
        ratio = np.divide(observed, true, out=np.zeros_like(true), where=true != 0.0)
        expected = table["counts_per_cell"] * cells
        responding = ~flat & (np.abs(true) > 0.5)
        parts = []
        for low, high in zip(edges[:-1], edges[1:]):
            chosen = responding & (expected >= low) & (expected < high)
            rises, falls = chosen & (true > 0.0), chosen & (true < 0.0)
            if min(rises.sum(), falls.sum()) < 5:
                continue
            label = (
                f"{low:g} to {high:g}" if math.isfinite(high) else f"{low:g} or more"
            )
            parts.append(
                f"{label}: {np.median(ratio[chosen]):.2f} (rises "
                f"{np.median(ratio[rises]):.2f}, falls {np.median(ratio[falls]):.2f}; "
                f"{int(chosen.sum())} genes)"
            )
        rows.add(
            "Change of log2-CPM from the vehicle to the top dose over the true change, "
            f"2,000 replicates of each, at {cells:g} cells per profile, median by "
            "expected count at the vehicle, responsive genes whose true change exceeds "
            "0.5",
            "information",
            "; ".join(parts),
            "n/a",
            None,
        )


def information_library_sizes(rows: Rows) -> None:
    """The log2 change of the expected library size per cell at the top dose."""
    from cytepool import simulate_dose_series

    shifts = []
    for i in range(50):
        _, truth = simulate_dose_series(
            LADDER, replicates=1, cell_types=1, rng=np.random.default_rng([20261145, i])
        )
        table = truth.table("cell_type_1")
        changes = truth.curve("cell_type_1", at=np.asarray(LADDER))
        totals = (table["counts_per_cell"][:, None] * 2.0**changes).sum(axis=0)
        shifts.append(np.log2(totals / totals[0]))
    top = np.asarray(shifts)[:, -1]
    rows.add(
        "Shift of the expected library size at the top dose, 50 default series of the "
        "ladder",
        "information",
        f"median {np.median(top):+.3f} log2; 95th percentile of the absolute shift "
        f"{np.quantile(np.abs(top), 0.95):.3f}",
        "n/a",
        None,
    )


def information_timing(rows: Rows) -> None:
    """The run time of one series, the cube and the truth included."""
    from cytepool import simulate_dose_series

    for replicates in (3, 4):

        def once() -> float:
            start = time.perf_counter()
            simulate_dose_series(LADDER, replicates=replicates, rng=20261146)
            return time.perf_counter() - start

        once()
        seconds = float(np.median([once() for _ in range(3)]))
        rows.add(
            f"Time of one series of 9 dose groups x {replicates} replicates x 2 cell "
            "types x 1,000 genes, median of 3 after a first call",
            "information",
            f"{seconds * 1e3:.1f} ms, {60 / seconds:,.0f} series a minute",
            "n/a",
            None,
        )


# ---------------------------------------------------------------------------------------
# Seeds and threads
# ---------------------------------------------------------------------------------------


def check_seeds(rows: Rows) -> None:
    from cytepool import simulate_dose_series

    first, truth = simulate("ladder")
    second, again = simulate(
        "ladder", rng=np.random.default_rng(SERIES["ladder"]["rng"])
    )
    same = first.fingerprint() == second.fingerprint() and all(
        truth.table(c).equals(again.table(c)) for c in truth.cell_types
    )
    rows.add(
        "An integer seed and numpy.random.default_rng(seed): the ladder's cube and truth",
        "identical",
        "identical" if same else "differ",
        "0",
        same,
    )
    jumped = [
        simulate_dose_series(
            HALF_LOG, rng=np.random.Generator(np.random.PCG64(20261147).jumped())
        )
        for _ in range(2)
    ]
    same = jumped[0][0].fingerprint() == jumped[1][0].fingerprint()
    rows.add(
        "Two generators in the same state, each from a jumped bit generator",
        "identical",
        "identical" if same else "differ",
        "0",
        same,
    )
    changed_cube, changed = simulate(
        "ladder",
        replicates=[4, 2, 3, 3, 2, 3, 3, 2, 4],
        cells=[80.0, 900.0, 200.0],
        composition={"stellate": [1.0, 1.0, 1.0, 1.0, 0.8, 0.6, 0.4, 0.2, 0.1]},
        donor_sd=0.3,
        cell_types=["hepatocyte", "stellate", "kupffer"],
    )
    differences = 0
    for cell_type in truth.cell_types:
        base, other = truth.table(cell_type), changed.table(cell_type)
        for key in (
            "feature_id",
            "shape",
            *PARAMETERS,
            "counts_per_cell",
            "bmd_fold_change",
        ):
            differences += not np.array_equal(
                base[key], other[key], equal_nan=base[key].dtype.kind == "f"
            )
        factors = other["expression_sd"] / math.hypot(BIOLOGICAL_SD, 0.3)
        differences += not np.allclose(
            factors, base["expression_sd"] / BIOLOGICAL_SD, rtol=1e-14, atol=0.0
        )
    rows.add(
        "The ladder's truth with other replicates, cells, composition and donor_sd, and "
        "a third cell type: curves, counts per cell and SD factors of the first two",
        "identical (SD factors to 1e-14)",
        f"{differences} columns differ",
        "0",
        differences == 0,
    )


def check_threads(rows: Rows, here: dict[str, Any]) -> None:
    import numba

    records = {numba.get_num_threads(): here}
    for threads in THREADS:
        record = digests_in_process(threads)
        records[record["threads"]] = record["series"]
    distinct = {json.dumps(series, sort_keys=True) for series in records.values()}
    rows.add(
        "Cubes and truths of the four series at "
        + ", ".join(str(t) for t in sorted(records))
        + " numba threads, each in its own process",
        "identical",
        "identical" if len(distinct) == 1 else "differ",
        "0",
        len(distinct) == 1,
    )


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


def fixture_moments(
    inputs: dict[str, np.ndarray], reference: dict[str, np.ndarray], width: int
) -> dict[str, np.ndarray]:
    """Ten count cases of the moments series and ten covariances, with their exact
    moments, and the cells' cases."""
    chosen = np.linspace(0, width - 1, FIXTURE_MOMENTS).round().astype(int)
    covariance_chosen = (
        np.linspace(0, reference["covariance"].size - 1, FIXTURE_MOMENTS)
        .round()
        .astype(int)
    )
    return {
        "m": inputs["count_m"][chosen],
        "a": inputs["count_a"][chosen],
        "sigma": inputs["count_sigma"][chosen],
        "count_moments": reference["count_moments"][chosen],
        "cells_m": inputs["cells_m"],
        "cells_moments": reference["cells_moments"],
        **{
            f"cov_{key}": inputs[f"cov_{key}"][covariance_chosen]
            for key in ("m1", "m2", "a1", "a2", "sigma", "tau")
        },
        "covariance": reference["covariance"][covariance_chosen],
    }


def write_fixtures(
    cases: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    errors: np.ndarray,
    moments: dict[str, np.ndarray],
) -> None:
    rng = np.random.default_rng(20261151)
    keep: list[int] = []
    for shape in (*SHAPES, "flat"):
        members = np.flatnonzero(cases["shape"] == shape)
        boundary = members[np.char.startswith(cases["source"][members], "boundary")]
        real = np.setdiff1d(members, boundary)
        worst = real[np.argsort(-errors[real], kind="stable")[: FIXTURE_CURVES // 2]]
        rest = np.setdiff1d(real, worst)
        extra = rng.choice(
            rest, min(FIXTURE_CURVES - worst.size, rest.size), replace=False
        )
        keep += sorted({*boundary.tolist(), *worst.tolist(), *extra.tolist()})
    keep_array = np.asarray(sorted(keep))
    renumber = np.full(cases["shape"].size, -1)
    renumber[keep_array] = np.arange(keep_array.size)
    points = renumber[cases["grid_case"]] >= 0
    asked = renumber[cases["root_case"]] >= 0
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        FIXTURE_DIRECTORY / "curves.npz",
        **{
            key: cases[key][keep_array]
            for key in ("shape", *PARAMETERS, "top", "source")
        },
        ec50_root=reference["ec50"][keep_array],
        grid_case=renumber[cases["grid_case"][points]],
        grid_dose=cases["grid_dose"][points],
        grid_value=reference["grid_value"][points],
        root_case=renumber[cases["root_case"][asked]],
        root_change=cases["root_change"][asked],
        root=reference["root"][asked],
    )
    print(f"wrote {FIXTURE_DIRECTORY / 'curves.npz'} ({keep_array.size} curves)")
    np.savez_compressed(FIXTURE_DIRECTORY / "moments.npz", **moments)
    print(f"wrote {FIXTURE_DIRECTORY / 'moments.npz'} ({moments['m'].size} cases)")


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def covariance_cases(
    donors: dict[str, Any],
) -> tuple[dict[str, np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """The inputs of the covariances within donors, the vehicle with each dose gene by
    gene, and the counts of each pair; replicate r of every dose is donor r."""
    inputs: dict[str, list[np.ndarray]] = {
        key: [] for key in ("m1", "m2", "a1", "a2", "sigma", "tau")
    }
    observed = []
    for i in range(len(donors["m"])):
        if donors["dose"][i] == 0:
            continue
        vehicle = next(
            j
            for j in range(len(donors["m"]))
            if donors["type"][j] == donors["type"][i] and donors["dose"][j] == 0
        )
        inputs["m1"].append(donors["m"][vehicle])
        inputs["m2"].append(donors["m"][i])
        inputs["a1"].append(donors["a"][vehicle])
        inputs["a2"].append(donors["a"][i])
        inputs["sigma"].append(donors["sigma"][i])
        inputs["tau"].append(donors["tau"][i])
        observed.append((donors["counts"][vehicle], donors["counts"][i]))
    return {key: np.concatenate(values) for key, values in inputs.items()}, observed


def check_moments(
    rows: Rows,
    moments: dict[str, Any],
    donors: dict[str, Any],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    reference: dict[str, np.ndarray],
) -> None:
    """The rows of the cells and counts against their exact moments, the covariances
    within donors, and the two models the counts must not follow."""
    exact_cells = reference["cells_moments"].T
    z_cells = [
        z_of_moments(cells[:, None], exact_cells[:, [i]])
        for i, cells in enumerate(moments["cells"])
    ]
    rows.calibrated(
        "Mean cells per profile, moments, 2 cell types x 4 doses",
        "mpmath, Poisson-lognormal",
        [z[0] for z in z_cells],
    )
    rows.calibrated(
        "Variance of cells per profile, moments, 2 cell types x 4 doses",
        "mpmath, Poisson-lognormal",
        [z[1] for z in z_cells],
    )
    # Counts: means and variances in the moments series; variances in the donors one.
    exact = reference["count_moments"].T
    width = len(moments["m"]) * moments["m"][0].size
    observed = np.concatenate(moments["counts"], axis=1)
    z_mean, z_variance = z_of_moments(observed, exact[:, :width])
    rows.calibrated(
        "Mean counts, moments, 8 genes x 2 cell types x 4 doses",
        "mpmath, exact marginal moments",
        z_mean,
    )
    rows.calibrated(
        "Variance of counts, moments, 8 genes x 2 cell types x 4 doses",
        "mpmath, exact marginal moments",
        z_variance,
    )
    observed_donors = np.concatenate(donors["counts"], axis=1)
    _, z_donor_variance = z_of_moments(observed_donors, exact[:, width:])
    rows.calibrated(
        "Variance of counts with donors crossed with dose, donors, 8 genes x 2 cell "
        "types x 4 doses, with expression_sd",
        "mpmath, exact marginal moments",
        z_donor_variance,
    )
    z_covariance = []
    for (vehicle, dosed), exact_covariance in zip(
        pairs, np.split(reference["covariance"], len(pairs))
    ):
        products = (vehicle - vehicle.mean(axis=0)) * (dosed - dosed.mean(axis=0))
        n = products.shape[0]
        estimate = products.sum(axis=0) / (n - 1)
        spread = products.std(axis=0, ddof=1) / math.sqrt(n)
        z_covariance.append((estimate - exact_covariance) / spread)
    rows.calibrated(
        "Covariance of one donor's counts at the vehicle and at each dose, donors, 8 "
        "genes x 2 cell types x 3 doses",
        "mpmath, exact",
        z_covariance,
    )
    for key, label in (
        (
            "count_moments_dispersion",
            "a dispersion of 0.5 for every profile, not 0.5 / n",
        ),
        ("count_moments_natural", "effects in natural-log units, not log2"),
    ):
        wrong = reference[key].T[:, :width]
        z_wrong = np.concatenate(z_of_moments(observed, wrong))
        largest = float(np.max(np.abs(z_wrong)))
        rows.add(
            f"Counts of the moments series against a model with {label} "
            f"({z_wrong.size} comparisons)",
            "mpmath, the wrong model's moments",
            f"largest abs(z) {largest:.1f}",
            f"above {BOUND:g} SE",
            largest > BOUND,
            control=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--simulate", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.reference is not None:
        reference_main(arguments.reference)
        return 0
    if arguments.simulate is not None:
        simulate_main(arguments.simulate)
        return 0

    rows = Rows()
    simulated = {name: simulate(name) for name in SERIES}
    truths = {name: truth for name, (_, truth) in simulated.items()}
    curves = series_cases(
        {name: truths[name] for name in ("ladder", "half_log", "moments")}
    )
    cases = joined(curves, boundary_cases(curves["shape"].size))
    moments = moment_inputs(*simulated["moments"], donor_sd=0.0)
    donors = moment_inputs(*simulated["donors"], donor_sd=SERIES["donors"]["donor_sd"])
    covariance_inputs, pairs = covariance_cases(donors)
    inputs = {
        **{
            key: cases[key]
            for key in (
                "shape",
                *PARAMETERS,
                "top",
                "grid_case",
                "grid_dose",
                "root_case",
                "root_change",
            )
        },
        "cells_m": np.asarray([float(m[0]) for m in moments["m"]]),
        "count_m": np.concatenate([*moments["m"], *donors["m"]]),
        "count_a": np.concatenate([*moments["a"], *donors["a"]]),
        "count_sigma": np.concatenate(
            [
                *moments["sigma"],
                *(np.hypot(s, t) for s, t in zip(donors["sigma"], donors["tau"])),
            ]
        ),
        **{f"cov_{key}": values for key, values in covariance_inputs.items()},
    }
    start = time.perf_counter()
    reference, versions = run_reference(inputs)
    reference_seconds = time.perf_counter() - start
    print("Reference: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    print(f"Reference computed in {reference_seconds:.0f} s")

    errors = check_curves(rows, cases, reference)
    check_allocation(rows, truths)
    check_placement(rows, truths)
    check_moments(rows, moments, donors, pairs, reference)
    z = np.asarray(rows.z_scores)
    rows.add(
        f"z-scores of the {z.size} calibrated comparisons",
        "standard normal",
        f"largest {z.max():.2f}; {np.mean(z > 2):.1%} above 2 (4.6% expected)",
        "n/a",
        None,
    )
    check_threads(rows, series_digests(simulated))
    check_seeds(rows)
    information_shares(rows, truths["ladder"])
    information_ec50s(rows)
    information_bells(rows)
    information_observed_sd(rows)
    information_compression(rows)
    information_library_sizes(rows)
    information_timing(rows)

    print()
    rows.print()
    if arguments.write_fixtures:
        if rows.failures:
            print("not writing fixtures: some checks failed")
        else:
            width = len(moments["m"]) * moments["m"][0].size
            write_fixtures(
                cases, reference, errors, fixture_moments(inputs, reference, width)
            )
    print(f"{len(rows.rows)} checks, {rows.failures} failed")
    return 1 if rows.failures else 0


if __name__ == "__main__":
    sys.exit(main())
