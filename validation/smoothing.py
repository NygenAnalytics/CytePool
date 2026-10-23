"""Check LOWESS and natural cubic spline bases against R.

Compares ``cytepool.stats.smoothing.lowess`` with R's ``stats::lowess`` and
``natural_spline_basis`` with R's ``splines::ns`` on inputs generated from
fixed seeds.

LOWESS cases include mean-variance trends of 2,000 to 20,000 genes (as limma
and voom smooth them), spans from 0.1 to above 1, 0 to 4 robustness
iterations, delta of 0, the default and larger, ties (integer x, pairs, tie
blocks wider than the neighbourhood, ties at both ends), 1 to 5 points,
heavy-tailed outliers, constant x or y, data whose median absolute residual
is zero, neighbourhoods whose weights are all zero, spans whose product with
n is just below an integer, points placed at the cut-offs of the tricube and
bisquare weights and of the switch to a local constant fit, and x or y on
large or small scales. Fitted values are compared relative to the largest
absolute fitted value (tier close, at most 1e-6); the script also counts the
values identical to R's to the last bit.

Natural spline cases cover df with and without an intercept, explicit,
unsorted and repeated knots, boundary knots inside and beyond the data,
quantile knots that coincide because of ties or that R moves off a boundary
knot, x with few distinct values (rank-deficient bases), NaN in x, and
covariates of 2,000 genes. Bases are compared by their column spaces: the
least squares fitted values of random responses on either basis (tier
exact, at most 1e-10), their ranks and their rows of NaN. Cases for which R
fails must fail in CytePool too. Deliberate deviations, where CytePool
raises an error instead of returning what R returns, are listed as such.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer (the stats and splines packages ship with R). Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/smoothing.py
    PYTHONPATH=. python validation/smoothing.py --write-fixtures
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool.stats.smoothing import lowess, natural_spline_basis

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "smoothing"
CLOSE = 1e-6
EXACT = 1e-10
LARGEST_FIXTURE = 300

R_SCRIPT = r"""
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_doubles <- function(path, n) readBin(path, "double", n = n, size = 8, endian = "little")
write_doubles <- function(x, path) writeBin(as.double(x), path, size = 8, endian = "little")
numbers <- function(line) if (line == "none") numeric(0) else as.numeric(strsplit(line, ",")[[1]])
for (case in readLines(file.path(root, "lowess.txt"))) {
  dir <- file.path(root, "lowess", case)
  p <- read.table(file.path(dir, "parameters.txt"), header = TRUE)
  x <- read_doubles(file.path(dir, "x.bin"), p$n)
  y <- read_doubles(file.path(dir, "y.bin"), p$n)
  fit <- if (is.na(p$delta)) lowess(x, y, f = p$span, iter = p$iterations) else
    lowess(x, y, f = p$span, iter = p$iterations, delta = p$delta)
  write_doubles(fit$x, file.path(dir, "r_x.bin"))
  write_doubles(fit$y, file.path(dir, "r_y.bin"))
}
for (case in readLines(file.path(root, "ns.txt"))) {
  dir <- file.path(root, "ns", case)
  p <- readLines(file.path(dir, "parameters.txt"))
  x <- read_doubles(file.path(dir, "x.bin"), as.integer(p[1]))
  x[is.nan(x)] <- NA
  args <- list(x = x, intercept = p[3] == "TRUE")
  if (p[2] != "default") args$df <- as.integer(p[2])
  if (p[4] != "default") args$knots <- numbers(p[4])
  if (p[5] != "default") args$Boundary.knots <- numbers(p[5])
  basis <- tryCatch(suppressWarnings(do.call(splines::ns, args)), error = function(e) NULL)
  if (is.null(basis)) {
    writeLines("error", file.path(dir, "error.txt"))
  } else {
    write_doubles(dim(basis), file.path(dir, "r_dim.bin"))
    write_doubles(basis, file.path(dir, "r_basis.bin"))
  }
}
writeLines(paste("R", as.character(getRversion())), file.path(root, "versions.txt"))
"""


def lowess_cases() -> list[dict]:
    """Inputs for lowess: name, x, y, span, iterations, delta (None: default)."""
    rng = np.random.default_rng(20261023)

    def trend(n):  # log variances against average log expression, as in limma
        amean = rng.normal(6.0, 2.5, n)
        noise = np.log(rng.chisquare(4.0, n) / 4.0)
        return amean, np.log(0.05 + 2.0 / (1.0 + np.exp(amean - 4.0))) + noise

    amean = rng.normal(5.0, 2.0, 5000)
    sqrt_sd = 0.4 + 1.2 / (1.0 + np.exp(amean - 3.0)) + rng.normal(0.0, 0.1, 5000)
    rounded = np.round(rng.normal(5.0, 2.0, 20000), 2)
    noise = rng.normal(0.0, 0.3, 20000)
    x = rng.uniform(0.0, 10.0, 100)
    y = np.sin(x) + rng.normal(0.0, 0.3, 100)
    integers = rng.integers(0, 10, 80).astype(float)
    pairs = np.repeat(np.arange(20.0), 2)
    block = np.concatenate([np.arange(5.0), np.full(12, 5.0), np.arange(6.0, 20.0)])
    ends = np.concatenate([np.zeros(8), rng.uniform(0, 5, 20), np.full(7, 5.0)])
    line = np.arange(101.0)
    with_outliers = 2.0 + 0.5 * line
    with_outliers[[20, 50, 80]] += 7.0
    alternating = rng.normal(0, 1, 60)
    alternating[25:35] = 1000.0 * (-1.0) ** np.arange(10)
    x100 = rng.uniform(0.0, 10.0, 100)
    y100 = np.cos(x100) + rng.normal(0.0, 0.3, 100)
    upper_cut = [0.0, 0.3, 0.9995, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    lower_cut = [0.0, 0.0005, 0.5, 1.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    # A cluster of 10 points for one of whose fits the weighted sd of x is
    # 0.9995 or 1.0005 times 0.001 times the range of x (0.457458 is that sd
    # for a cluster of unit width).
    pattern = np.array([-1.0, -0.8, -0.55, -0.3, -0.1, 0.05, 0.3, 0.5, 0.75, 1.0])
    far = np.concatenate([np.arange(10.0), 991.0 + np.arange(10.0)])
    below, above = (
        np.concatenate([far[:10], 500.0 + factor / 0.457458 * pattern, far[10:]])
        for factor in (0.9995, 1.0005)
    )
    # Tie blocks as large as the neighbourhood are fitted by their means, so
    # the residuals are set exactly: median absolute residual 1, and
    # residuals at 0.9995 and 0.0005 times six times it.
    groups = [10.0 * b + np.array([-2, -1, 0, 1, 2.0]) for b in range(19)]
    groups.append(500.0 + np.array([5.997, 0.003, -2.0, -2.0, -2.0]))
    blocks = np.repeat(np.arange(20.0) * 100, 5)
    normal = rng.normal
    first_fit = dict(iterations=0, delta=0.0)
    cut_fit, third = first_fit | dict(span=0.4), first_fit | dict(span=1 / 3)
    one_iteration = dict(span=0.05, iterations=1, delta=0.0)

    specs = [
        ("trend_2000_genes", *trend(2000), dict(span=0.4)),
        ("trend_20000_genes", *trend(20000), dict(span=0.4)),
        ("trend_2000_genes_delta_0", *trend(2000), dict(span=0.4, delta=0.0)),
        ("voom_trend_5000_genes", amean, sqrt_sd, dict(span=0.5)),
        ("trend_20000_genes_rounded_x", rounded, np.sqrt(np.abs(rounded)) + noise, {}),
        *((f"span_{s}", x, y, dict(span=s)) for s in (0.1, 0.3, 0.4, 0.5, 1.0, 1.7)),
        *((f"iterations_{i}", x, y, dict(iterations=i)) for i in (0, 1, 2, 4)),
        ("delta_0", x, y, dict(span=0.3, delta=0.0)),
        ("delta_large", x, y, dict(span=0.3, delta=1.5)),
        ("delta_beyond_range", x, y, dict(delta=30.0)),
        ("integer_x", integers, np.sin(integers) + normal(0, 0.5, 80), dict(span=0.2)),
        ("integer_x_delta_0", integers, normal(0, 1, 80), dict(span=0.05, delta=0.0)),
        ("integer_x_large_delta", integers, normal(0, 1, 80), dict(delta=2.5)),
        *(
            (f"paired_x_span_{s}", pairs, normal(0, 1, 40), dict(span=s, delta=0.0))
            for s in (0.1, 0.15, 0.3)
        ),
        ("tie_block_wider_than_window", block, normal(0, 1, 31), dict(span=0.15)),
        ("ties_at_both_ends", ends, normal(0, 1, 35), dict(span=0.2, delta=0.0)),
        *((f"n_{n}", rng.uniform(0, 1, n), normal(0, 1, n), {}) for n in range(1, 6)),
        ("n_2_tied", [1.0, 1.0], [5.0, 7.0], {}),
        ("heavy_tails", x, y + 0.3 * rng.standard_cauchy(100), dict(iterations=4)),
        ("constant_y", x, np.full(100, 2.5), {}),
        ("line_with_outliers_zero_mad", line, with_outliers, dict(span=0.1, delta=0)),
        ("all_weights_zero", np.arange(60.0), alternating, dict(span=4 / 60, delta=0)),
        ("constant_x", np.full(9, 2.0), normal(0, 1, 9), {}),
        ("two_x_values", np.repeat([1.0, 2.0], 6), normal(0, 1, 12), {}),
        ("span_times_n_just_below_29", x100, y100, dict(span=0.29, delta=0.0)),
        ("span_times_n_30_minus_5e-8", x100, y100, dict(span=(30 - 5e-8) / 100)),
        ("span_times_n_30_minus_2e-7", x100, y100, dict(span=(30 - 2e-7) / 100)),
        ("tricube_upper_cutoff", upper_cut, normal(0, 1, 10), cut_fit),
        ("tricube_lower_cutoff", lower_cut, normal(0, 1, 10), cut_fit),
        ("local_constant_switch_below", below, normal(0, 1, 30), third),
        ("local_constant_switch_above", above, normal(0, 1, 30), third),
        ("bisquare_cutoffs", blocks, np.concatenate(groups), one_iteration),
        ("x_offset_1e6", 1e6 + x, y, {}),
        ("x_scale_1e-8", 1e-8 * x, y, {}),
        ("y_scale_1e10", x, 1e10 * y, dict(span=0.3)),
    ]
    defaults = dict(span=2 / 3, iterations=3, delta=None)
    return [
        dict(name=name, x=np.asarray(a, float), y=np.asarray(b, float))
        | defaults
        | options
        for name, a, b, options in specs
    ]


def spline_cases() -> list[dict]:
    """Inputs for ns: name, x, df, knots, intercept, boundary knots."""
    rng = np.random.default_rng(20261024)
    x = rng.exponential(1.0, 60)
    wide = rng.normal(0.0, 2.0, 70)
    grid = np.linspace(0.0, 3.0, 31)
    low = np.concatenate([np.zeros(15), rng.uniform(0, 5, 25)])
    high = np.concatenate([rng.uniform(0, 5, 25), np.full(15, 5.0)])
    missing = x.copy()
    missing[[3, 17, 40]] = np.nan
    amean = rng.normal(6.0, 2.5, 2000)
    on = dict(intercept=True)
    inside = dict(knots=[-0.5, 0.3, 1.0], boundary_knots=[-1.0, 1.5])
    beyond = dict(knots=[-0.5, 1.0], boundary_knots=[7.0, -6.0])
    three = [0.1, 0.7, 1.3, 1.3, 1.3, 2.9, 2.9, 2.9]
    ends = dict(boundary_knots=[0.0, 3.0])
    specs = [
        *((f"df_{df}", x, dict(df=df)) for df in (1, 2, 3, 5)),
        *((f"df_{df}_intercept", x, dict(df=df, **on)) for df in (2, 3, 5, 7)),
        ("no_knots", x, {}),
        ("explicit_knots", x, dict(knots=[0.3, 0.9, 2.0], **on)),
        ("explicit_knots_unsorted", x, dict(knots=[2.0, 0.3, 0.9])),
        ("boundary_knots_inside_data_df_3", wide, dict(df=3, boundary_knots=[-1.5, 2])),
        ("boundary_knots_inside_data_df_4", wide, dict(df=4, boundary_knots=[-1.5, 2])),
        ("boundary_knots_inside_data_knots", wide, inside | on),
        ("boundary_knots_reversed_beyond_data", wide, beyond),
        *(
            (
                f"knot_multiplicity_{m}{'_intercept' if i else ''}",
                grid,
                dict(knots=[1.0] * m + [2.0], intercept=i, **ends),
            )
            for m in (2, 3, 4, 5)
            for i in (False, True)
        ),
        ("integer_x_repeated_quantile_knots", rng.integers(0, 6, 80), dict(df=6, **on)),
        ("three_distinct_values", np.repeat([0.0, 1.0, 2.0], [4, 9, 5]), dict(df=5)),
        ("quantile_knot_on_lower_boundary", low, dict(df=4)),
        ("quantile_knot_on_lower_boundary_intercept", low, dict(df=4, **on)),
        ("quantile_knot_on_upper_boundary", high, dict(df=5, **on)),
        (
            "quantile_knots_on_both_boundaries",
            np.repeat([0, 1, 2], [10, 3, 10]),
            {"df": 3},
        ),
        ("two_distinct_values", [0.0, 0.0, 1.0, 1.0, 1.0], dict(df=3)),
        ("nan_in_x", missing, dict(df=4, **on)),
        *((f"covariate_2000_genes_df_{d}", amean, dict(df=d, **on)) for d in (2, 3, 4)),
        # R fails on these, so CytePool must raise.
        ("fails_every_knot_on_upper_boundary", [0, 0, 1.0, 1, 1], dict(df=3, **on)),
        ("fails_every_knot_on_lower_boundary", [0.0] * 14 + [1.0, 2.0], dict(df=5)),
        ("fails_knot_on_boundary_after_shove", three, dict(df=7)),
        ("fails_knot_beyond_boundary", grid, dict(knots=[1, 5], **ends)),
        # Deliberate deviations: R returns a basis, CytePool raises.
        ("deviation_knot_on_lower_boundary", grid, dict(knots=[0, 1], **ends)),
        ("deviation_df_below_minimum", x, dict(df=1, **on)),
    ]
    defaults = dict(df=None, knots=None, intercept=False, boundary_knots=None)
    return [
        dict(name=name, x=np.asarray(a, float)) | defaults | options
        for name, a, options in specs
    ]


def run_r(lowess_inputs: list[dict], spline_inputs: list[dict]) -> tuple[dict, str]:
    def text(values) -> str:
        values = np.atleast_1d(np.asarray(values, float))
        return ",".join(repr(float(v)) for v in values) if values.size else "none"

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for kind, inputs in (("lowess", lowess_inputs), ("ns", spline_inputs)):
            (root / f"{kind}.txt").write_text("".join(c["name"] + "\n" for c in inputs))
            for case in inputs:
                (root / kind / case["name"]).mkdir(parents=True)
                (root / kind / case["name"] / "x.bin").write_bytes(
                    case["x"].astype("<f8").tobytes()
                )
        for case in lowess_inputs:
            directory = root / "lowess" / case["name"]
            (directory / "y.bin").write_bytes(case["y"].astype("<f8").tobytes())
            delta = "NA" if case["delta"] is None else repr(float(case["delta"]))
            (directory / "parameters.txt").write_text(
                f"n span iterations delta\n{case['x'].size} {float(case['span'])!r} "
                f"{case['iterations']} {delta}\n"
            )
        for case in spline_inputs:
            lines = [
                str(case["x"].size),
                "default" if case["df"] is None else str(case["df"]),
                str(case["intercept"]).upper(),
                *(
                    "default" if case[key] is None else text(case[key])
                    for key in ("knots", "boundary_knots")
                ),
            ]
            path = root / "ns" / case["name"] / "parameters.txt"
            path.write_text("\n".join(lines) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        command = [rscript, "--vanilla", str(root / "compare.R"), str(root)]
        subprocess.run(command, check=True)
        results: dict = {}
        for case in lowess_inputs:
            directory = root / "lowess" / case["name"]
            results["lowess", case["name"]] = tuple(
                np.fromfile(directory / name, "<f8") for name in ("r_x.bin", "r_y.bin")
            )
        for case in spline_inputs:
            directory = root / "ns" / case["name"]
            if (directory / "error.txt").exists():
                results["ns", case["name"]] = None
                continue
            rows, columns = np.fromfile(directory / "r_dim.bin", "<f8").astype(int)
            basis = np.fromfile(directory / "r_basis.bin", "<f8").reshape(columns, rows)
            basis[np.isnan(basis)] = np.nan  # R's NA is a signalling NaN
            results["ns", case["name"]] = basis.T.copy()
        return results, (root / "versions.txt").read_text().strip()


class Table:
    def __init__(self) -> None:
        self.rows: list[tuple[str, ...]] = []
        self.failures = 0

    def add(self, check, reference, metric, value, tolerance, shown=None) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        shown = f"{value:.1e}" if shown is None else shown
        result = "pass" if passed else "FAIL"
        self.rows.append((check, reference, metric, shown, f"{tolerance:g}", result))

    def info(self, check, reference, metric, value, result="info") -> None:
        self.rows.append((check, reference, metric, value, "n/a", result))

    def print(self) -> None:
        print("| check | reference | metric | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def in_input_order(case: dict, r_x: np.ndarray, r_y: np.ndarray) -> np.ndarray | None:
    """R's fitted values (sorted by x) in the order of the input, if R's x agree."""
    order = np.argsort(case["x"], kind="stable")
    if not np.array_equal(case["x"][order], r_x):
        return None
    fitted = np.empty_like(r_y)
    fitted[order] = r_y
    return fitted


def compare_lowess(cases: list[dict], results: dict, table: Table) -> None:
    identical = total = 0
    for case in cases:
        reference = in_input_order(case, *results["lowess", case["name"]])
        options = {key: case[key] for key in ("span", "iterations", "delta")}
        ours = lowess(case["x"], case["y"], **options)
        difference = np.inf
        if reference is not None:
            scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
            difference = float(np.max(np.abs(ours - reference))) / scale
            identical += int(np.count_nonzero(ours == reference))
        total += ours.size
        check = f"lowess {case['name']} (n={case['x'].size})"
        table.add(check, "stats::lowess", "max diff / max abs", difference, CLOSE)
    check = "lowess: values identical to R's to the last bit"
    table.info(check, "stats::lowess", "values", f"{identical} of {total}")


def fitted_values(basis: np.ndarray, responses: np.ndarray) -> tuple[np.ndarray, int]:
    """Least squares fitted values on the rows without NaN, and the rank."""
    complete = ~np.isnan(basis).any(axis=1)
    u, s, _ = np.linalg.svd(basis[complete], full_matrices=False)
    u = u[:, : int(np.count_nonzero(s > s[0] * 1e-10))]
    fits = np.full((basis.shape[0], responses.shape[1]), np.nan)
    fits[complete] = u @ (u.T @ responses[complete])
    return fits, u.shape[1]


def compare_splines(cases: list[dict], results: dict, table: Table) -> None:
    rng = np.random.default_rng(7)
    for case in cases:
        name, reference = case["name"], results["ns", case["name"]]
        keywords = {k: case[k] for k in ("df", "knots", "intercept", "boundary_knots")}
        try:
            ours = natural_spline_basis(case["x"], **keywords)
        except ValueError:
            ours = None
        if name.startswith("deviation"):
            outcome = "R returns a basis; CytePool raises"
            if ours is not None or reference is None:
                outcome = "no deviation"
            table.info(f"ns {name}", "splines::ns", "behaviour", outcome, "deviation")
        elif reference is None or ours is None:
            mismatch = int((reference is None) != (ours is None))
            check = f"ns {name}: {'both raise' if reference is None else 'both work'}"
            table.add(check, "splines::ns", "mismatches", mismatch, 0, str(mismatch))
        elif ours.shape != reference.shape or not np.array_equal(
            np.isnan(ours), np.isnan(reference)
        ):
            table.add(f"ns {name}: shape, NaN rows", "splines::ns", "mismatches", 1, 0)
        else:
            responses = rng.normal(0.0, 1.0, (case["x"].size, 3))
            ours_fits, ours_rank = fitted_values(ours, responses)
            r_fits, rank = fitted_values(reference, responses)
            complete = ~np.isnan(r_fits[:, 0])
            scale = max(float(np.max(np.abs(r_fits[complete]))), np.finfo(float).tiny)
            difference = np.max(np.abs(ours_fits - r_fits)[complete]) / scale
            if ours_rank != rank:
                difference = np.inf
            check = f"ns {name} (n={case['x'].size}, rank {rank} of {ours.shape[1]})"
            metric = "LS fits: max diff / max abs"
            table.add(check, "splines::ns", metric, float(difference), EXACT)


def write_fixtures(lowess_inputs, spline_inputs, results) -> None:
    """Small cases: lowess.npz holds one (3, n) array of x, y and R's fitted
    values per case, with spans, iterations and delta (NaN: default) in
    ``parameters``; natural_splines.npz holds x, the arguments and R's basis
    (absent where R fails)."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    small = [c for c in lowess_inputs if c["x"].size <= LARGEST_FIXTURE]
    deltas = [np.nan if c["delta"] is None else c["delta"] for c in small]
    parameters = [[c["span"], c["iterations"], d] for c, d in zip(small, deltas)]
    arrays = {"names": np.array([c["name"] for c in small])}
    arrays["parameters"] = np.array(parameters)
    for c in small:
        fitted = in_input_order(c, *results["lowess", c["name"]])
        arrays[c["name"]] = np.vstack([c["x"], c["y"], fitted])
    np.savez_compressed(FIXTURE_DIRECTORY / "lowess.npz", **arrays)
    small = [
        c
        for c in spline_inputs
        if c["x"].size <= LARGEST_FIXTURE and not c["name"].startswith("deviation")
    ]
    arrays = {
        "names": np.array([c["name"] for c in small]),
        "df": np.array([-1 if c["df"] is None else c["df"] for c in small]),
        "intercept": np.array([c["intercept"] for c in small]),
    }
    for c in small:
        arrays[c["name"] + "__x"] = c["x"]
        for key in ("knots", "boundary_knots"):
            if c[key] is not None:
                arrays[f"{c['name']}__{key}"] = np.asarray(c[key], float)
        if results["ns", c["name"]] is not None:
            arrays[c["name"] + "__basis"] = results["ns", c["name"]]
    np.savez_compressed(FIXTURE_DIRECTORY / "natural_splines.npz", **arrays)
    print(f"wrote lowess.npz and natural_splines.npz to {FIXTURE_DIRECTORY}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    lowess_inputs, spline_inputs = lowess_cases(), spline_cases()
    results, version = run_r(lowess_inputs, spline_inputs)
    print(f"Reference: {version} (stats::lowess, splines::ns)")
    table = Table()
    compare_lowess(lowess_inputs, results, table)
    compare_splines(spline_inputs, results, table)
    table.print()
    if arguments.write_fixtures:
        if table.failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(lowess_inputs, spline_inputs, results)
    print(f"{len(table.rows)} checks, {table.failures} failed")
    return 1 if table.failures else 0


if __name__ == "__main__":
    sys.exit(main())
