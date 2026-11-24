"""Check the EC50s, Hill slopes and plateaus of dose curves against mpmath, GraphPad
Prism's documented equation and drc.

Data from fixed seeds (20261240 to 20261243): ``ladder``, simulated by
``simulate_dose_series``, the vehicle and 8 half-log doses from 0.01 to 30 ug/kg x 3,
a cell type of 500 cells per profile and one of 30, each gene's values in the cell
type's fit by limma_voom refitted under constant variance, as dose_response refits
them; ``plug_in``, the vehicle and 5 doses from 0.1 to 10 ug/kg x 2, whose dose-group
means, covariance and variances dose_response fits in the weighted and moderated
modes; and ``edges``, genes drawn in this script with normal errors of SD 0.25 around
curves that test the ends: flat, an EC50 below the lowest dose, one above the top
dose, a step, a shallow curve, a fall, and two rises and two falls of slopes 1 and
2.5, at the vehicle and 8 half-log doses x 3.

mpmath 1.3.0, at 30 digits in another process of this script that never imports
CytePool, computes each interval from the documentation: the profile at an EC50 E,
the smallest weighted sum of squares of the dose-group means about a Hill curve with
EC50 E, its slope minimised from a grid of 60 slopes by golden-section search around
each local minimum within the documented bounds, its levels by weighted least
squares; the maximum by golden-section search over ln E around the best of the
documented scan, the 48 EC50s from a hundredth of the lowest dose to 5 times the top
dose (with 0 for the Hill model); the deviance (Q(E) - Q_max) / s^2, s^2 the curve's
residual variance on n - 4 degrees of freedom under constant variance and the
variance of the means' fit otherwise, against scipy 1.18.1's F(1, n - 4) or F(1, df)
at 0.95; and the ends, the lowest and highest E of the region at most the threshold,
by findroot (Illinois) between the region's extreme and the scan point beyond it, 0
or infinity where the region reaches the scan's first or last point. The same process
writes GraphPad's equation, Y = Bottom + (Top - Bottom) / (1 + 10^((log10 EC50 - X)
HillSlope)) with X = log10 dose, with CytePool's reported values, and the curve from
the fit's parameters by the documented formula, and compares them at the positive
doses. drc 4.0-0 fits ``LL.4`` to the constant-variance samples, where GraphPad's
Hill slope is -b sign(d - c) of drc's estimates, and prints ``ED(interval = "delta")``
and ``ED`` of ``LL2.4`` with ``interval = "fls"`` beside the profile intervals.

Tiers: close (1e-6 relative) for the intervals' finite ends against mpmath, and
identical for open ends and for the intervals CytePool leaves out because its scan
found a curve better than the fit; the fit within 1e-8 of mpmath's maximum, in units
of s^2; exact (1e-10) for GraphPad's equation; identical for the Hill slope's sign
against drc where drc reaches CytePool's maximum (residual sums of squares within
1e-8); information for drc's intervals.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root, with mpmath 1.3.0
    and scipy 1.18.1 (the reference process runs under the same interpreter).
    R 4.4 with drc 4.0-0 and jsonlite (``DOSE_RSCRIPT``, an Rscript executable;
    default ``Rscript``).

Usage:
    PYTHONPATH=. python validation/ec50.py
    PYTHONPATH=. python validation/ec50.py --write-fixtures
    PYTHONPATH=. python validation/ec50.py --benchmark
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "ec50"
CLOSE, EXACT, AT_MAXIMUM = 1e-6, 1e-10, 1e-8
LEVEL = 0.95
HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]

# ---------------------------------------------------------------------------------------
# The references, which never import CytePool
# ---------------------------------------------------------------------------------------

DIGITS = 30
SLOPES = 60  # the grid of the profile's slope
LOCATIONS = 48  # the documented scan of the EC50
GOLDEN_STEPS = 45  # golden-section steps of the slope, and of the maximum's ln E
# The documented bounds: the log-logistic slope from 0.1, the Hill exponent from 1,
# both to 18; EC50s from a hundredth of the lowest positive dose (0 for the Hill
# model) to 5 times the top dose.
SLOPE_BOUNDS = {"ll4": (0.1, 18.0), "hill": (1.0, 18.0)}


def _mp():
    import mpmath

    mpmath.mp.dps = DIGITS
    return mpmath


def documented_fraction(mp, x, slope, ec50):
    """The share of the change a Hill curve, log-logistic or BMDS's Hill model, has
    covered at dose x: 1 - 1 / (1 + (x / EC50)^slope), 0 at dose 0 and 1 above it when
    the EC50 is 0."""
    if x == 0:
        return mp.mpf(0)
    if ec50 == 0:
        return mp.mpf(1)
    return 1 - 1 / (1 + (mp.mpf(x) / ec50) ** slope)


def level_fit(mp, slope, ec50, task) -> Any:
    """The weighted least squares of the levels of a Hill curve with this slope and
    EC50, and its sum of squares: the curve is start + change h(x)."""
    x, y, w = task["mp_doses"], task["mp_means"], task["mp_weights"]
    h = [documented_fraction(mp, xi, slope, ec50) for xi in x]
    k = len(x)
    columns = mp.matrix([[1, hi] for hi in h])
    gram = columns.T * w * columns
    right = columns.T * w * mp.matrix(y)
    if abs(mp.det(gram)) <= mp.mpf(10) ** (-2 * DIGITS + 10) * (
        gram[0, 0] * gram[1, 1]
    ):
        # The two columns are one: a flat curve at the weighted mean.
        start = (mp.matrix([1] * k).T * w * mp.matrix(y))[0] / (
            mp.matrix([1] * k).T * w * mp.matrix([1] * k)
        )[0]
        coefficients = mp.matrix([start, 0])
    else:
        coefficients = mp.lu_solve(gram, right)
    residual = mp.matrix(y) - columns * coefficients
    return (residual.T * w * residual)[0], coefficients


def profile(mp, task, ec50) -> Any:
    """The smallest weighted sum of squares of a Hill curve with this EC50."""
    model = task["model"]
    low, high = (mp.log(v) for v in SLOPE_BOUNDS[model])
    grid = [low + (high - low) * i / (SLOPES - 1) for i in range(SLOPES)]
    values = [level_fit(mp, mp.exp(u), ec50, task)[0] for u in grid]
    best = min(values)
    golden = (mp.sqrt(5) - 1) / 2
    for i in range(SLOPES):
        left = values[i - 1] if i > 0 else mp.inf
        right = values[i + 1] if i < SLOPES - 1 else mp.inf
        if values[i] > left or values[i] > right:
            continue
        a, b = grid[max(i - 1, 0)], grid[min(i + 1, SLOPES - 1)]
        c, d = b - golden * (b - a), a + golden * (b - a)
        fc = level_fit(mp, mp.exp(c), ec50, task)[0]
        fd = level_fit(mp, mp.exp(d), ec50, task)[0]
        for _ in range(GOLDEN_STEPS):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - golden * (b - a)
                fc = level_fit(mp, mp.exp(c), ec50, task)[0]
            else:
                a, c, fc = c, d, fd
                d = a + golden * (b - a)
                fd = level_fit(mp, mp.exp(d), ec50, task)[0]
        best = min(best, fc, fd)
    return best


def scan_points(mp, task) -> list:
    """The documented scan: 48 EC50s evenly spaced on log scale from a hundredth of the
    lowest positive dose to 5 times the top dose, with 0 first for the Hill model."""
    positive = [d for d in task["doses"] if d > 0]
    low, high = mp.log(mp.mpf(positive[0]) / 100), mp.log(5 * mp.mpf(task["doses"][-1]))
    points = [
        mp.exp(low + (high - low) * i / (LOCATIONS - 1)) for i in range(LOCATIONS)
    ]
    points[0], points[-1] = mp.mpf(positive[0]) / 100, 5 * mp.mpf(task["doses"][-1])
    return ([mp.mpf(0)] if task["model"] == "hill" else []) + points


def reference_interval(task: dict[str, Any]) -> dict[str, Any]:
    """The documented interval of one gene's EC50, and the maximum it reads."""
    from scipy import stats

    mp = _mp()
    task["mp_doses"] = [mp.mpf(v) for v in task["doses"]]
    task["mp_means"] = [mp.mpf(v) for v in task["means"]]
    task["mp_weights"] = mp.matrix(task["weights"])
    scan = scan_points(mp, task)
    q = [profile(mp, task, e) for e in scan]
    # The maximum: golden-section search over ln E around the best scan point.
    i = min(range(len(scan)), key=lambda j: q[j])
    q_max = q[i]
    positive_scan = [j for j in range(len(scan)) if scan[j] > 0]
    if scan[i] > 0:
        a = mp.log(scan[max(i - 1, positive_scan[0])])
        b = mp.log(scan[min(i + 1, len(scan) - 1)])
        golden = (mp.sqrt(5) - 1) / 2
        c, d = b - golden * (b - a), a + golden * (b - a)
        fc, fd = profile(mp, task, mp.exp(c)), profile(mp, task, mp.exp(d))
        for _ in range(GOLDEN_STEPS):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - golden * (b - a)
                fc = profile(mp, task, mp.exp(c))
            else:
                a, c, fc = c, d, fd
                d = a + golden * (b - a)
                fd = profile(mp, task, mp.exp(d))
        q_max = min(q_max, fc, fd)
    if task["variance"] == "constant":
        n = sum(task["n_samples"])
        s2 = (mp.mpf(task["within"]) + q_max) / (n - 4)
        threshold = mp.mpf(float(stats.f.ppf(LEVEL, 1, n - 4)))
    else:
        s2 = mp.mpf(task["s2"])
        threshold = mp.mpf(float(stats.f.ppf(LEVEL, 1, task["df"])))
    deviance = [(v - q_max) / s2 for v in q]
    inside = [j for j in range(len(scan)) if deviance[j] <= threshold]
    estimate = mp.mpf(task["estimate"])
    low = min([scan[j] for j in inside] + [estimate])
    high = max([scan[j] for j in inside] + [estimate])
    ends = []
    for side, end in ((0, low), (1, high)):
        if (side == 0 and end <= scan[0]) or (side == 1 and end >= scan[-1]):
            ends.append(0.0 if side == 0 else math.inf)
            continue
        beyond = (
            max(s for s in scan if s < end)
            if side == 0
            else min(s for s in scan if s > end)
        )
        logs = beyond > 0

        def f(u, logs=logs):
            e = mp.exp(u) if logs else u
            return (profile(mp, task, e) - q_max) / s2 - threshold

        bracket = (mp.log(beyond), mp.log(end)) if logs else (beyond, end)
        u = mp.findroot(f, bracket, solver="illinois", tol=mp.mpf(10) ** -24)
        ends.append(float(mp.exp(u) if logs else u))
    return {
        "key": task["key"],
        "lower": ends[0],
        "upper": ends[1],
        "objective_max": float(q_max),
        "s2": float(s2),
    }


def graphpad_check(task: dict[str, Any]) -> dict[str, Any]:
    """The largest relative difference between GraphPad's equation with the reported
    values and the fitted curve from the documented formula, at the positive doses."""
    mp = _mp()
    model, p = task["model"], [mp.mpf(v) for v in task["parameters"]]
    worst = mp.mpf(0)
    for x in (mp.mpf(v) for v in task["doses"] if v > 0):
        if model == "ll4":
            b, c, d, e = p
            curve = c + (d - c) / (1 + (x / e) ** b)
        else:
            g, v, k, n = p
            curve = g + v * documented_fraction(mp, x, n, k)
        bottom, top = mp.mpf(task["bottom"]), mp.mpf(task["top"])
        slope, ec50 = mp.mpf(task["hill_slope"]), mp.mpf(task["ec50"])
        y = bottom + (top - bottom) / (
            1 + mp.mpf(10) ** ((mp.log10(ec50) - mp.log10(x)) * slope)
        )
        worst = max(worst, abs(y - curve) / max(abs(curve), 1))
    return {"key": task["key"], "difference": float(worst)}


def reference_main(directory: Path) -> None:
    from concurrent.futures import ProcessPoolExecutor

    import mpmath
    import scipy

    tasks = json.loads((directory / "reference_inputs.json").read_text())
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as pool:
        intervals = list(pool.map(reference_interval, tasks["intervals"]))
    graphpad = [graphpad_check(task) for task in tasks["graphpad"]]
    out = {
        "versions": {"mpmath": mpmath.__version__, "scipy": scipy.__version__},
        "intervals": intervals,
        "graphpad": graphpad,
    }
    (directory / "reference.json").write_text(json.dumps(out))


DRC = r"""
suppressPackageStartupMessages({library(drc); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = FALSE)
out <- list(versions = list(R = as.character(getRversion()),
                            drc = as.character(packageVersion("drc"))), genes = list())
for (g in input$genes) {
  x <- unlist(g$doses); y <- unlist(g$values)
  control <- drmc(relTol = 1e-12, maxIt = 10000, errorm = FALSE, noMessage = TRUE)
  fit <- tryCatch(drm(y ~ x, fct = LL.4(), control = control), error = function(e) NULL)
  res <- list(key = g$key)
  if (!is.null(fit) && !inherits(fit, "try-error") && !is.null(coef(fit))) {
    cf <- unname(coef(fit))
    res$coefficients <- cf; res$rss <- sum(residuals(fit)^2)
    ed <- tryCatch(ED(fit, 50, interval = "delta", display = FALSE), error = function(e) NULL)
    if (!is.null(ed)) res$delta <- unname(ed[1, c("Lower", "Upper")])
    fit2 <- tryCatch(drm(y ~ x, fct = LL2.4(), control = control), error = function(e) NULL)
    if (!is.null(fit2) && !is.null(coef(fit2))) {
      ed2 <- tryCatch(ED(fit2, 50, interval = "fls", display = FALSE), error = function(e) NULL)
      if (!is.null(ed2)) res$fls <- unname(ed2[1, c("Lower", "Upper")])
    }
  }
  out$genes[[length(out$genes) + 1]] <- res
}
write_json(out, args[[2]], digits = NA, auto_unbox = TRUE, na = "null")
"""


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "program.R").write_text(program)
        (root / "input.json").write_text(json.dumps(payload))
        command = [
            os.environ.get(variable, "Rscript"),
            "--vanilla",
            str(root / "program.R"),
            str(root / "input.json"),
            str(root / "output.json"),
        ]
        subprocess.run(command, check=True)
        return json.loads((root / "output.json").read_text())


def numbers(values: Any) -> np.ndarray:
    """R's numbers, with NA (null in JSON) as NaN."""
    return np.array([math.nan if v is None else v for v in values], dtype=float)


def run_reference(payload: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "reference_inputs.json").write_text(json.dumps(payload))
        subprocess.run(
            [sys.executable, "-I", __file__, "--reference", str(root)], check=True
        )
        return json.loads((root / "reference.json").read_text())


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------


class Rows:
    """The table's rows: check, reference, value, tolerance, result."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, str]] = []
        self.failures = 0
        self.worst: dict[str, str] = {}

    def check(
        self,
        check: str,
        reference: str,
        value: float,
        tolerance: float,
        count: int,
        worst: str = "",
    ) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        if not passed:
            self.worst[check] = worst
        self.rows.append(
            (
                f"{check} ({count:,})",
                reference,
                f"{value:.1e}",
                f"{tolerance:g}",
                "pass" if passed else "FAIL",
            )
        )

    def identical(
        self, check: str, reference: str, mismatches: int, count: int
    ) -> None:
        self.failures += mismatches > 0
        self.rows.append(
            (
                f"{check} ({count:,})",
                reference,
                f"{mismatches} mismatches",
                "0",
                "pass" if mismatches == 0 else "FAIL",
            )
        )

    def note(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "information"))

    def print(self) -> None:
        print("| check | reference | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")
        for check, worst in self.worst.items():
            print(f"differs: {check}: {worst}")


def axis() -> Any:
    from cytepool.dose_axes import DoseAxis

    return DoseAxis("dose", unit="ug/kg", metric="administered")


class Case:
    """Fits of one Hill curve, the genes compared and, under constant variance, the
    samples they were fitted to."""

    def __init__(self, name, fits, genes, ids, samples=None, plug_in=None):
        from cytepool.stats.potency import ec50s

        self.name = name
        self.fits = fits
        self.genes = genes  # rows of the fits compared
        self.ids = dict(zip(genes, ids, strict=True))  # their ids, by row
        self.samples = samples  # (doses, values) under constant variance
        self.plug_in = plug_in  # the _CurveData fields in the plug-in modes
        self.ec50s = ec50s(fits)
        self.model = self.ec50s.model


def picked_genes(truth: Any, cell_type: str, genes: list[str], count: int) -> list[int]:
    """Hill genes spread over their true EC50s, then two flat genes, among those
    tested."""
    table = truth.table(cell_type)
    row = {g: i for i, g in enumerate(table["feature_id"].tolist())}
    shapes, ec50 = table["shape"], np.asarray(table["ec50"], dtype=float)
    hills = [p for p, g in enumerate(genes) if shapes[row[g]] == "hill"]
    hills.sort(key=lambda p: ec50[row[genes[p]]])
    chosen = [hills[i] for i in np.linspace(0, len(hills) - 1, count).astype(int)]
    flats = [p for p, g in enumerate(genes) if shapes[row[g]] == "flat"][:2]
    return sorted(set(chosen + flats))


def ladder_cases() -> list[Case]:
    from cytepool.design import Design
    from cytepool.dose_responses import dose_response
    from cytepool.simulated_dose_series import simulate_dose_series
    from cytepool.stats.dose_curves import fit_dose_curves

    cube, truth = simulate_dose_series(
        HALF_LOG,
        rng=20261240,
        replicates=3,
        cell_types=["many", "few"],
        cells=[500.0, 30.0],
        n_genes=300,
    )
    design = Design(cube.samples, "~ 1")
    found = []
    for ct in ("many", "few"):
        result = dose_response(cube, design, dose=axis(), groups=[ct])
        fit = result.fit(ct)
        genes = fit.features["feature_id"].tolist()
        picked = picked_genes(truth, ct, genes, 6)
        doses = np.asarray(fit.samples["dose"], dtype=float)
        values = np.asarray(fit.log2_cpm)[:, picked]
        ids = [genes[p] for p in picked]
        for model_set in ("ll4", "epa"):
            fits = fit_dose_curves(doses, values, models=model_set)
            found.append(
                Case(
                    f"ladder, {ct}, {model_set}",
                    fits,
                    list(range(len(picked))),
                    ids,
                    samples=(doses, values),
                )
            )
    return found


def plug_in_cases() -> list[Case]:
    from cytepool.design import Design
    from cytepool.dose_responses import dose_response
    from cytepool.simulated_dose_series import simulate_dose_series

    cube, truth = simulate_dose_series(
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        rng=20261241,
        replicates=2,
        cell_types=1,
        cells=300.0,
        n_genes=300,
    )
    design = Design(cube.samples, "~ 1")
    found = []
    for variance in ("weighted", "moderated"):
        result = dose_response(
            cube, design, dose=axis(), models="ll4", variance=variance
        )
        ct = result.groups[0]
        fits = result.curve_fits(ct)
        genes = result.table(ct, "curves")["feature_id"].tolist()
        picked = picked_genes(truth, ct, genes, 5)
        found.append(
            Case(f"plug_in, {variance}", fits, picked, [genes[p] for p in picked])
        )
    return found


def edge_values() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Samples of curves that test the interval's ends, normal errors of SD 0.25."""
    curves = {
        # name: (b, c, d, e) of c + (d - c) / (1 + (x/e)^b), d at dose 0
        "flat": (1.0, 6.0, 6.0, 1.0),
        "EC50 below the lowest dose": (2.0, 7.0, 6.0, 0.002),
        "EC50 above the top dose": (2.0, 7.0, 6.0, 100.0),
        "step": (18.0, 7.0, 6.0, 0.5),
        "shallow": (0.3, 8.0, 6.0, 1.0),
        "fall": (1.5, 5.0, 6.0, 0.3),
        "rise, slope 1": (1.0, 7.5, 6.0, 0.1),
        "rise, slope 2.5": (2.5, 7.5, 6.0, 1.0),
        "fall, slope 1": (1.0, 4.5, 6.0, 0.3),
        "fall, slope 2.5": (2.5, 4.5, 6.0, 3.0),
    }
    rng = np.random.default_rng(20261242)
    x = np.repeat(HALF_LOG, 3)
    values = []
    for b, c, d, e in curves.values():
        with np.errstate(divide="ignore"):
            mean = c + (d - c) / (1.0 + np.where(x > 0, (x / e) ** b, 0.0))
        values.append(mean + rng.normal(0.0, 0.25, x.size))
    return x, np.column_stack(values), list(curves)


def edge_cases() -> list[Case]:
    from cytepool.stats.dose_curves import fit_dose_curves

    x, values, names = edge_values()
    return [
        Case(
            f"edges, {model_set}",
            fit_dose_curves(x, values, models=model_set),
            list(range(len(names))),
            names,
            samples=(x, values),
        )
        for model_set in ("ll4", "epa")
    ]


def reference_tasks(cases: list[Case]) -> dict[str, Any]:
    intervals, graphpad = [], []
    for c, case in enumerate(cases):
        fits, found = case.fits, case.ec50s
        m = fits.models.index(case.model)
        names = fits.parameter_names
        letters = "bcde" if case.model == "ll4" else "gvkn"
        columns = [names.index(p) for p in letters]
        for g in case.genes:
            key = f"{c}|{g}"
            if found.status[g] == "no curve":
                continue
            if np.isfinite(found.threshold[g]):
                weights = np.linalg.inv(fits.covariance[g])
                task = {
                    "key": key,
                    "model": case.model,
                    "variance": fits.variance,
                    "doses": fits.doses.tolist(),
                    "n_samples": fits.n_samples.tolist(),
                    "means": fits.means[g].tolist(),
                    "weights": weights.tolist(),
                    "estimate": float(found.ec50[g]),
                    "within": float(fits.covariance_scale[g] * fits.df[g]),
                    "s2": float(fits.covariance_scale[g]),
                    "df": float(fits.df[g]),
                }
                intervals.append(task)
            if np.isfinite(found.hill_slope[g]):
                graphpad.append(
                    {
                        "key": key,
                        "model": case.model,
                        "doses": fits.doses.tolist(),
                        "parameters": fits.parameters[g, m, columns].tolist(),
                        "ec50": float(found.ec50[g]),
                        "hill_slope": float(found.hill_slope[g]),
                        "bottom": float(found.bottom_plateau[g]),
                        "top": float(found.top_plateau[g]),
                    }
                )
    return {"intervals": intervals, "graphpad": graphpad}


def objective_at_fit(case: Case, g: int) -> float:
    """The weighted sum of squares of a gene's means about its fitted Hill curve, from
    the documented formula."""
    fits = case.fits
    m = fits.models.index(case.model)
    names = fits.parameter_names
    x = fits.doses
    if case.model == "ll4":
        b, c, d, e = (fits.parameters[g, m, names.index(p)] for p in "bcde")
        with np.errstate(divide="ignore"):
            curve = c + (d - c) / (1.0 + np.where(x > 0, (x / e) ** b, 0.0))
    else:
        gg, v, k, n = (fits.parameters[g, m, names.index(p)] for p in "gvkn")
        with np.errstate(divide="ignore", invalid="ignore"):
            share = np.where(
                x > 0, 1.0 / (1.0 + (k / np.where(x > 0, x, 1.0)) ** n), 0.0
            )
        curve = gg + v * share
    residual = fits.means[g] - curve
    return float(residual @ np.linalg.solve(fits.covariance[g], residual))


def relative(ours: float, theirs: float) -> float:
    """|ours - theirs| / |theirs| for finite ends; 0 for equal open ends, infinity for
    an open end against a finite one."""
    if ours == theirs:
        return 0.0
    if not (math.isfinite(ours) and math.isfinite(theirs)) or theirs == 0.0:
        return math.inf
    return abs(ours - theirs) / abs(theirs)


def compare_intervals(rows: Rows, cases: list[Case], reference: dict[str, Any]) -> list:
    by_key = {item["key"]: item for item in reference["intervals"]}
    worst, open_mismatch, maximum, count, matched = 0.0, 0, 0.0, 0, []
    worst_text = ""
    for c, case in enumerate(cases):
        found = case.ec50s
        for g in case.genes:
            item = by_key.get(f"{c}|{g}")
            if item is None:
                continue
            count += 1
            # CytePool's fit against mpmath's maximum, in units of s^2.
            gap = (objective_at_fit(case, g) - item["objective_max"]) / item["s2"]
            if np.isnan(found.ec50_lower[g]):
                # Left out: mpmath must find a better curve than the fit.
                open_mismatch += int(not gap > 1e-6)
                continue
            maximum = max(maximum, gap)
            ours = (float(found.ec50_lower[g]), float(found.ec50_upper[g]))
            theirs = (item["lower"], item["upper"])
            for a, b in zip(ours, theirs, strict=True):
                if (a == 0.0) != (b == 0.0) or math.isinf(a) != math.isinf(b):
                    open_mismatch += 1
                elif math.isfinite(a) and a > 0.0:
                    d = relative(a, b)
                    if d > worst:
                        worst, worst_text = (
                            d,
                            f"{case.name}, {case.ids[g]}: {a!r} against {b!r}",
                        )
            matched.append((c, g, theirs, item))
    rows.check(
        "EC50 intervals' finite ends",
        "mpmath 1.3.0, the documented profile",
        worst,
        CLOSE,
        count,
        worst_text,
    )
    rows.identical(
        "EC50 intervals' open ends, and those left out",
        "mpmath 1.3.0",
        open_mismatch,
        count,
    )
    rows.check(
        "The fit within mpmath's maximum, in units of s^2",
        "mpmath 1.3.0",
        maximum,
        AT_MAXIMUM,
        count,
    )
    return matched


def compare_graphpad(rows: Rows, reference: dict[str, Any]) -> None:
    values = [item["difference"] for item in reference["graphpad"]]
    rows.check(
        "GraphPad's equation with the EC50, Hill slope and plateaus gives the curve",
        "GraphPad Prism's documented equation",
        max(values, default=0.0),
        EXACT,
        len(values),
    )


def drc_payload(cases: list[Case]) -> dict[str, Any]:
    genes = []
    for c, case in enumerate(cases):
        if case.samples is None or case.model != "ll4":
            continue
        doses, values = case.samples
        for g in case.genes:
            genes.append(
                {
                    "key": f"{c}|{g}",
                    "doses": doses.tolist(),
                    "values": values[:, g].tolist(),
                }
            )
    return {"genes": genes}


def compare_drc(rows: Rows, cases: list[Case], drc: dict[str, Any]) -> None:
    signs, compared, lower_drc = 0, 0, 0
    delta_negative, widths_delta, widths_fls, profile_open = 0, [], [], 0
    delta_missing = 0
    reached = 0
    for item in drc["genes"]:
        c, g = (int(v) for v in item["key"].split("|"))
        case = cases[c]
        found = case.ec50s
        if "coefficients" not in item or found.status[g] == "no curve":
            continue
        coefficients = numbers(item["coefficients"])
        if not np.all(np.isfinite(coefficients)) or item["rss"] is None:
            continue  # drc did not converge
        fits = case.fits
        rss_ours = fits.covariance_scale[g] * fits.df[g] + objective_at_fit(case, g)
        b, cc, d, e = coefficients
        if item["rss"] > rss_ours * (1.0 + AT_MAXIMUM):
            lower_drc += 1
            continue
        reached += 1
        slope = found.hill_slope[g]
        theirs = -b * math.copysign(1.0, d - cc) if d != cc else math.nan
        if np.isfinite(slope) or math.isfinite(theirs):
            compared += 1
            signs += int(not (np.sign(slope) == np.sign(theirs)))
        # drc's intervals beside the profile's, at the same fit.
        low, high = float(found.ec50_lower[g]), float(found.ec50_upper[g])
        delta = numbers(item.get("delta", [None, None]))
        fls = numbers(item.get("fls", [None, None]))
        closed = math.isfinite(high) and low > 0
        if np.all(np.isfinite(delta)):
            delta_negative += int(delta[0] < 0)
            if closed:
                widths_delta.append((delta[1] - delta[0]) / (high - low))
            else:
                profile_open += 1
        else:
            delta_missing += 1
        if np.all(np.isfinite(fls)) and closed:
            widths_fls.append((fls[1] - fls[0]) / (high - low))
    rows.identical(
        "Hill slope's sign: -b sign(d - c) of drc's LL.4", "drc 4.0-0", signs, compared
    )
    rows.note(
        "Fits where drc stops below CytePool's maximum (by more than 1e-8 of its RSS)",
        "drc 4.0-0",
        f"{lower_drc} of {lower_drc + reached}",
    )
    rows.note(
        "Where drc reaches CytePool's maximum: drc's delta intervals with a lower end "
        "below 0; median width over the profile interval's, delta and LL2.4's fls",
        "drc 4.0-0",
        f"{delta_negative} of {len(widths_delta) + profile_open} "
        f"({delta_missing} without one); {np.median(widths_delta):.2f} and "
        f"{np.median(widths_fls):.2f} ({len(widths_delta)} and {len(widths_fls)} closed "
        "profile intervals)",
    )


def fixtures(
    cases: list[Case], matched: list, drc: dict[str, Any], versions: dict
) -> None:
    """The cases' inputs, CytePool-free references for the tests: mpmath's intervals,
    and drc's coefficients of the constant-variance ll4 genes."""
    records = []
    for c, case in enumerate(cases):
        record: dict[str, Any] = {
            "name": case.name,
            "model_set": case.fits.model_set,
            "variance": case.fits.variance,
            "ids": [case.ids[g] for g in case.genes],
        }
        if case.samples is not None:
            doses, values = case.samples
            record["doses"] = doses.tolist()
            record["values"] = values.T.tolist()
        else:
            fits = case.fits
            rows_ = case.genes
            record["plug_in"] = {
                "doses": fits.doses.tolist(),
                "n_samples": fits.n_samples.tolist(),
                "means": fits.means[rows_].tolist(),
                "covariance": fits.covariance[rows_].tolist(),
                "covariance_scale": fits.covariance_scale[rows_].tolist(),
                "df": fits.df[rows_].tolist(),
            }
        # mpmath's ends of each gene stored, in order; None where it computed none.
        ends = {g: [theirs[0], theirs[1]] for cc, g, theirs, _ in matched if cc == c}
        record["mpmath"] = [ends.get(g) for g in case.genes]
        records.append(record)
    drc_genes = [
        item
        for item in drc["genes"]
        if "coefficients" in item
        and item["rss"] is not None
        and np.all(np.isfinite(numbers(item["coefficients"])))
    ]
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, payload in (
        ("profiles.json", {"versions": versions, "cases": records}),
        ("drc.json", {"versions": versions, "genes": drc_genes}),
    ):
        path = FIXTURE_DIRECTORY / name
        path.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
        print(f"wrote {name} to {FIXTURE_DIRECTORY}")


def benchmark() -> None:
    """ec50s on 10,000 genes of ll4 curves, the vehicle and 8 half-log doses x 3, under
    constant variance, on 2 threads, with the ordinary least squares fit of 20,000
    genes x 200 samples of linear_models.py's benchmark as the machine's baseline."""
    from cytepool.stats.dose_curves import fit_dose_curves
    from cytepool.stats.linear_models import fit_linear_model
    from cytepool.stats.potency import ec50s

    rng = np.random.default_rng(20261243)
    x = np.repeat(HALF_LOG, 3)
    n_genes = 10_000
    e = 10.0 ** rng.uniform(-2.0, 1.5, n_genes)
    b = rng.uniform(0.5, 4.0, n_genes)
    change = rng.choice([0.0, 0.0, 1.0, -1.0], n_genes)
    with np.errstate(divide="ignore"):
        share = np.where(x[:, None] > 0, (x[:, None] / e) ** b, 0.0)
    values = (
        6.0
        + change * (1.0 - 1.0 / (1.0 + share))
        + rng.normal(0.0, 0.25, (x.size, n_genes))
    )
    fits = fit_dose_curves(x, values, models="ll4", n_threads=2)
    y = rng.normal(size=(200, 20_000))
    design = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])
    timings = {}
    for name, run in (
        ("ec50s", lambda: ec50s(fits, n_threads=2)),
        ("ols", lambda: fit_linear_model(y, design)),
    ):
        run()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        timings[name] = float(np.median(times))
    print("| run | seconds |")
    print("| --- | --- |")
    print(f"| ec50s, 10,000 genes, 2 threads | {timings['ec50s']:.1f} |")
    print(
        f"| fit_linear_model, 20,000 genes x 200 samples (baseline) | {timings['ols']:.3f} |"
    )
    print(f"{1e3 * timings['ec50s'] / n_genes:.2f} ms per gene")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time ec50s")
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.reference is not None:
        reference_main(arguments.reference)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0
    start = time.perf_counter()
    cases = ladder_cases() + plug_in_cases() + edge_cases()
    print(f"CytePool ran in {time.perf_counter() - start:.0f} s")
    start = time.perf_counter()
    reference = run_reference(reference_tasks(cases))
    drc = run_r(DRC, drc_payload(cases), "DOSE_RSCRIPT")
    print(f"References ran in {time.perf_counter() - start:.0f} s")
    versions = reference["versions"] | drc["versions"]
    print("Reference: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    rows = Rows()
    matched = compare_intervals(rows, cases, reference)
    compare_graphpad(rows, reference)
    compare_drc(rows, cases, drc)
    print()
    rows.print()
    if arguments.write_fixtures:
        if rows.failures:
            print("not writing fixtures: some checks failed")
        else:
            fixtures(cases, matched, drc, versions)
    print(f"{len(rows.rows)} checks, {rows.failures} failed")
    return 1 if rows.failures else 0


if __name__ == "__main__":
    sys.exit(main())
