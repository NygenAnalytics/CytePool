"""Check DoseAxis against R: positions, spline columns, designs, F-tests and curves.

Three designs from fixed seeds (``designs``): a ladder like the TCDD liver series, the
vehicle (4 samples) and eight doses from 0.01 to 30 ug/kg with 2 or 3 samples each;
an in vitro half-log series, 6 vehicle wells and four doses from 0.1 to 3.16 uM in
duplicate; and 24 distinct measured exposures with 6 zeros. Each design is read on
four axes: the vehicle in a group of its own, on the log10 and on the linear scale;
and the vehicle on the axis, on the log10 scale at an offset added to every dose and
on the linear scale. On each axis, both forms of the spline, with knots by R's rule
on the distinct doses (``"levels"``) and on the samples (``"quantiles"``) at every df
from 1 to the number of distinct doses on the axis minus one (at most 8), by
Harrell's rule (``"harrell"``) at df 2 to that number minus 3 (at most 8), and given
as doses.

R computes, from the doses alone: the positions (``log10(dose)``, ``log10(dose +
offset)`` or the dose, ``NA`` for the vehicle in its own group); ``treated``,
``as.numeric(dose > 0)``; the knots with R's own functions,
``attr(ns(unique(x), df), "knots")``, ``attr(ns(x, df), "knots")``,
``rcspline.eval(x, nk = df + 1, knots.only = TRUE)`` (Hmisc) or the positions of the
given doses, with the range of the positions as the boundary knots except under
Harrell's rule; the spline columns, ``ns(x, knots, Boundary.knots)`` or
``rcspline.eval(x, knots, inclx = TRUE)`` with the lower boundary knot subtracted from
its first column (``x`` minus it with two knots), 0 for the vehicle in its own group;
``model.matrix`` of ``~ dose_treated + dose_s1 + ...`` (``~ dose_s1 + ...`` with the
vehicle on the axis); the F-statistics of nested ``lm`` fits of four seeded responses
(``anova``): any dose (every column but the intercept), among the treated doses (the
spline columns, with the vehicle in its own group) and departure from a straight line
(the restricted cubic columns after the first, from df 2); and ``predict`` of the full
fit on a grid of doses: 0, each dose, the geometric mean of each pair of neighbouring
doses, and twice the highest.

CytePool computes the same with ``DoseAxis.positions``, ``DoseAxis.spline_basis``,
``DoseAxis.design_columns`` and ``Design``, and the F-statistics and predictions by
least squares on its own columns. Rows per design and axis: positions; treated
(identical); the knots; the restricted cubic columns, element by element relative to
each column's largest absolute value; the B-spline columns through their column space
(least squares fits of random responses on the spline columns alone, relative to the
largest fit), since they differ from ``ns()``'s by a change of basis; the design
matrices against ``model.matrix`` (element by element, and through the column space
for the B-spline form); the F-statistics (relative); and the curves on the grid
(relative to the largest absolute value). Tier exact, 1e-10 throughout, except
``treated`` (identical): R's and numpy's ``log10`` can differ in the last bit.
F-statistics and curves are compared where the design's condition number is at most
1e5, so that rounding stays below the tier; the cases above it, restricted cubic
columns on a linear axis over doses that span decades, are counted with their largest
condition number and the cases where R's ``lm`` drops a column as aliased. Cases
where one program places knots and the other fails count as outcomes that differ
(tolerance 0), except the deliberate differences: where R's rule places two knots at
one dose of a replicated design, ``ns`` returns a basis with a repeated knot and
DoseAxis raises; where Hmisc's ``rcspline.eval`` switches to its method for values tied
at an extreme (the 6 zeros of the exposures on an axis with the vehicle), drops a
repeated knot or places fewer knots than asked, Hmisc returns knots and DoseAxis
raises.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the CRAN package Hmisc (the splines package ships with R).
    Set ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/dose_axes.py
    PYTHONPATH=. python validation/dose_axes.py --write-fixtures
"""

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from cytepool import Design, DoseAxis, Table

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_axes"
EXACT = 1e-10
GENES = 4
LARGEST_DF = 8
# F-statistics and curves are compared at the exact tier where the design's condition
# number is at most this: their rounding errors, about the condition number times
# 1e-16, then stay below 1e-10. Above it, as for restricted cubic columns on a linear
# axis over doses that span decades, they depend on how each program solves the least
# squares problem, and R's lm can drop columns as aliased.
CONDITION_LIMIT = 1e5

R_SCRIPT = r"""
suppressPackageStartupMessages({library(splines); library(Hmisc)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_all <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8,
                                   endian = "little")
write_doubles <- function(x, path) writeBin(as.double(x), path, size = 8, endian = "little")
quietly <- function(expr) {
  out <- NULL
  invisible(capture.output(
    out <- tryCatch(suppressWarnings(expr), error = function(e) NULL)
  ))
  out
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, "cases", case)
  p <- readLines(file.path(dir, "parameters.txt"))
  scale <- p[1]; vehicle <- p[2]; offset <- as.numeric(p[3]); rule <- p[4]
  df <- as.integer(p[5]); form <- p[6]
  dose <- read_all(file.path(dir, "doses.bin"))
  y <- matrix(read_all(file.path(dir, "y.bin")), nrow = length(dose))
  grid <- read_all(file.path(dir, "grid.bin"))
  position <- function(d) {
    if (vehicle == "own_group") {
      out <- rep(NA_real_, length(d))
      treated <- d > 0
      out[treated] <- if (scale == "log10") log10(d[treated]) else d[treated]
      out
    } else if (scale == "log10") log10(d + offset) else d
  }
  x <- position(dose)
  write_doubles(x, file.path(dir, "r_positions.bin"))
  write_doubles(as.numeric(dose > 0), file.path(dir, "r_treated.bin"))
  on <- !is.na(x)
  xs <- x[on]
  with_boundary <- function(b) c(attr(b, "Boundary.knots")[1], attr(b, "knots"),
                                 attr(b, "Boundary.knots")[2])
  knots <- quietly(switch(rule,
    levels = with_boundary(ns(unique(xs), df = df)),
    quantiles = with_boundary(ns(xs, df = df)),
    harrell = rcspline.eval(xs, nk = df + 1, knots.only = TRUE),
    given = c(min(xs), position(read_all(file.path(dir, "given.bin"))), max(xs))))
  if (is.null(knots)) next
  write_doubles(knots, file.path(dir, "r_knots.bin"))
  # Columns only where the knots are those asked for: df + 1 distinct knots, or
  # repeated ones in the B-spline form, as ns() takes them.
  if (length(knots) != df + 1 || (form == "restricted_cubic" && anyDuplicated(knots))) next
  interior <- knots[-c(1, length(knots))]
  boundary <- knots[c(1, length(knots))]
  if (form == "b_spline") {
    basis <- ns(xs, knots = interior, Boundary.knots = boundary)
    spline <- function(z) predict(basis, z)
  } else {
    spline <- function(z) {
      if (length(knots) == 2) return(matrix(z - knots[1], ncol = 1))
      m <- rcspline.eval(z, knots = knots, inclx = TRUE)
      m[, 1] <- m[, 1] - knots[1]
      m
    }
  }
  names <- paste0("dose_s", seq_len(df))
  columns_at <- function(d) {
    z <- position(d)
    columns <- matrix(0, length(d), df, dimnames = list(NULL, names))
    columns[!is.na(z), ] <- spline(z[!is.na(z)])
    data <- data.frame(columns)
    if (vehicle == "own_group") data <- cbind(dose_treated = as.numeric(d > 0), data)
    data
  }
  data <- columns_at(dose)
  write_doubles(as.matrix(data), file.path(dir, "r_columns.bin"))
  terms <- colnames(data)
  full <- reformulate(terms)
  design <- model.matrix(full, data)
  write_doubles(dim(design), file.path(dir, "r_design_dim.bin"))
  write_doubles(design, file.path(dir, "r_design.bin"))
  # Nested F-tests: any dose, among the treated doses, departure from a line.
  reduced <- list(any_dose = character(0))
  if (vehicle == "own_group") reduced$among_doses <- "dose_treated"
  if (form == "restricted_cubic" && df >= 2) {
    reduced$nonlinear <- c(if (vehicle == "own_group") "dose_treated", "dose_s1")
  }
  grid_data <- columns_at(grid)
  f <- NULL
  curves <- NULL
  for (g in seq_len(ncol(y))) {
    data$y <- y[, g]
    fit <- lm(update(full, y ~ .), data)
    for (name in names(reduced)) {
      kept <- reduced[[name]]
      nested <- lm(if (length(kept)) reformulate(kept, "y") else y ~ 1, data)
      f <- c(f, anova(nested, fit)$F[2])
    }
    curves <- cbind(curves, predict(fit, grid_data))
  }
  write_doubles(fit$rank, file.path(dir, "r_rank.bin"))
  write_doubles(f, file.path(dir, "r_f.bin"))
  writeLines(names(reduced), file.path(dir, "r_tests.txt"))
  write_doubles(curves, file.path(dir, "r_curves.bin"))
}
version <- function(name) paste(name, packageDescription(name, fields = "Version"))
writeLines(paste0("R ", as.character(getRversion()), " (splines::ns, stats::lm), ",
                  version("Hmisc")), file.path(root, "versions.txt"))
"""


def designs() -> dict[str, dict]:
    """The dose designs, each with its unit, metric, offset and given knots (doses)."""
    rng = np.random.default_rng(20261124)
    ladder = np.array([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])
    replicates = np.r_[4, rng.integers(2, 4, ladder.size)]
    half_log = 10.0 ** np.array([-1.0, -0.5, 0.0, 0.5])
    exposures = np.sort(rng.lognormal(0.0, 1.0, 24))
    return {
        "tcdd_ladder": {
            "doses": np.repeat(np.r_[0.0, ladder], replicates),
            "unit": "ug/kg",
            "metric": "administered",
            "offset": 0.003,
            "given": [0.1, 1.0],
        },
        "half_log_wells": {
            "doses": np.r_[np.zeros(6), np.repeat(half_log, 2)],
            "unit": "uM",
            "metric": "nominal",
            "offset": 0.03,
            "given": [0.5],
        },
        "exposures": {
            "doses": np.r_[np.zeros(6), exposures],
            "unit": "ng/mL",
            "metric": "measured",
            "offset": 0.01,
            "given": [float(np.quantile(exposures, 0.3)), float(np.median(exposures))],
        },
    }


AXES = {
    "own group, log10": {"scale": "log10", "vehicle": "own_group"},
    "own group, linear": {"scale": "linear", "vehicle": "own_group"},
    "on axis, log10 with offset": {"scale": "log10", "vehicle": "on_axis"},
    "on axis, linear": {"scale": "linear", "vehicle": "on_axis"},
}
FORMS = ("restricted_cubic", "b_spline")


def cases() -> list[dict]:
    """Every design x axis x form x rule x df, with responses and a grid of doses."""
    rng = np.random.default_rng(20261125)
    found = []
    for design_name, design in designs().items():
        doses = design["doses"]
        positive = np.unique(doses[doses > 0])
        grid = np.r_[
            0.0, positive, np.sqrt(positive[:-1] * positive[1:]), 2 * positive[-1]
        ]
        # Responses that rise along log dose, with noise; the vehicle sits lower.
        level = np.log10(np.where(doses > 0, doses, positive[0] / 10.0))
        shape = 1.0 / (1.0 + np.exp(-2.0 * (level - np.median(level))))
        y = (
            rng.normal(5.0, 1.0, GENES)
            + np.outer(shape, rng.normal(0.0, 1.5, GENES))
            + rng.normal(0.0, 0.3, (doses.size, GENES))
        )
        for axis_name, options in AXES.items():
            offset = (
                design["offset"]
                if options == AXES["on axis, log10 with offset"]
                else None
            )
            axis = DoseAxis(
                "dose",
                unit=design["unit"],
                metric=design["metric"],
                offset=offset,
                **options,
            )
            m = np.unique(axis.positions(doses)[~np.isnan(axis.positions(doses))]).size
            plans = [
                *(("levels", df) for df in range(1, min(m - 1, LARGEST_DF) + 1)),
                *(("quantiles", df) for df in range(1, min(m - 1, LARGEST_DF) + 1)),
                *(("harrell", df) for df in range(2, min(m - 3, LARGEST_DF) + 1)),
                ("given", len(design["given"]) + 1),
            ]
            for form in FORMS:
                for rule, df in plans:
                    found.append(
                        {
                            "name": f"{design_name} | {axis_name} | {form} | {rule} | df {df}",
                            "design": design_name,
                            "axis_name": axis_name,
                            "axis": axis,
                            "doses": doses,
                            "y": y,
                            "grid": grid,
                            "rule": rule,
                            "df": df,
                            "form": form,
                            "given": np.asarray(design["given"], float),
                        }
                    )
    return found


def run_r(inputs: list[dict]) -> tuple[dict, str]:
    def write(path: Path, values) -> None:
        path.write_bytes(np.asarray(values, float).astype("<f8").tobytes())

    def read(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        values = np.fromfile(path, "<f8")
        values[np.isnan(values)] = np.nan  # R's NA is a signalling NaN
        return values

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        labels = [f"case{i}" for i in range(len(inputs))]
        (root / "cases.txt").write_text("".join(f"{label}\n" for label in labels))
        for label, case in zip(labels, inputs):
            directory = root / "cases" / label
            directory.mkdir(parents=True)
            axis = case["axis"]
            offset = "NA" if axis.offset is None else repr(axis.offset)
            parameters = [
                axis.scale,
                axis.vehicle,
                offset,
                case["rule"],
                str(case["df"]),
                case["form"],
            ]
            (directory / "parameters.txt").write_text("\n".join(parameters) + "\n")
            write(directory / "doses.bin", case["doses"])
            write(directory / "y.bin", case["y"].T.ravel())
            write(directory / "grid.bin", case["grid"])
            write(directory / "given.bin", case["given"])
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        results = {}
        for label, case in zip(labels, inputs):
            directory = root / "cases" / label
            n, grid = case["doses"].size, case["grid"].size
            result = {
                "positions": read(directory / "r_positions.bin"),
                "treated": read(directory / "r_treated.bin"),
                "knots": read(directory / "r_knots.bin"),
            }
            columns = read(directory / "r_columns.bin")
            if columns is not None:
                rows, width = read(directory / "r_design_dim.bin").astype(int)
                result |= {
                    "columns": columns.reshape(-1, n).T.copy(),
                    "design": read(directory / "r_design.bin")
                    .reshape(width, rows)
                    .T.copy(),
                    "tests": (directory / "r_tests.txt").read_text().split(),
                    "f": read(directory / "r_f.bin").reshape(GENES, -1),
                    "curves": read(directory / "r_curves.bin")
                    .reshape(GENES, grid)
                    .T.copy(),
                    "rank": int(read(directory / "r_rank.bin")[0]),
                }
            results[case["name"]] = result
        return results, (root / "versions.txt").read_text().strip()


class Checks:
    """The largest difference of each kind of check, and the outcomes, of one design
    on one axis."""

    def __init__(self) -> None:
        self.largest: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.differing: list[str] = []
        # Condition numbers above CONDITION_LIMIT, and whether R's lm lost rank.
        self.ill_conditioned: list[tuple[float, bool]] = []

    def record(self, kind: str, value: float) -> None:
        self.largest[kind] = max(self.largest.get(kind, 0.0), value)
        self.counts[kind] = self.counts.get(kind, 0) + 1


def relative(ours: np.ndarray, reference: np.ndarray) -> float:
    """Largest difference relative to the largest absolute reference value, with NaN
    where both are NaN; infinite where NaN appears in one only."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape or not np.array_equal(
        np.isnan(ours), np.isnan(reference)
    ):
        return np.inf
    present = ~np.isnan(reference)
    if not present.any():
        return 0.0
    scale = max(float(np.max(np.abs(reference[present]))), np.finfo(float).tiny)
    return float(np.max(np.abs(ours - reference)[present])) / scale


def by_column(ours: np.ndarray, reference: np.ndarray) -> float:
    """Largest difference of each column relative to its largest absolute value."""
    if ours.shape != reference.shape:
        return np.inf
    scale = np.maximum(np.max(np.abs(reference), axis=0), np.finfo(float).tiny)
    return float(np.max(np.abs(ours - reference) / scale))


def column_space(ours: np.ndarray, reference: np.ndarray) -> float:
    """Least squares fits of random responses on either matrix, relative to the
    largest fit; infinite when the shapes or ranks differ."""
    if ours.shape != reference.shape:
        return np.inf
    responses = np.random.default_rng(3).normal(0.0, 1.0, (ours.shape[0], 3))

    def fits(matrix: np.ndarray) -> tuple[np.ndarray, int]:
        u, s, _ = np.linalg.svd(matrix, full_matrices=False)
        u = u[:, : int(np.count_nonzero(s > s[0] * 1e-10))]
        return u @ (u.T @ responses), u.shape[1]

    ours_fits, ours_rank = fits(ours)
    reference_fits, rank = fits(reference)
    if ours_rank != rank:
        return np.inf
    return relative(ours_fits, reference_fits)


def f_statistics(
    design: np.ndarray, y: np.ndarray, dropped: list[list[int]]
) -> np.ndarray:
    """F-statistics, genes x tests, of least squares fits that drop each set of
    columns."""

    def rss(columns: np.ndarray) -> np.ndarray:
        coefficients = np.linalg.lstsq(columns, y, rcond=None)[0]
        return np.sum((y - columns @ coefficients) ** 2, axis=0)

    full = rss(design)
    df_residual = y.shape[0] - design.shape[1]
    statistics = []
    for columns in dropped:
        kept = [j for j in range(design.shape[1]) if j not in columns]
        statistics.append(
            ((rss(design[:, kept]) - full) / len(columns)) / (full / df_residual)
        )
    return np.column_stack(statistics)


def departure(case: dict, reference: dict, message: str) -> str | None:
    """The deliberate difference that explains why DoseAxis raises where R placed
    knots, or None."""
    knots = reference["knots"]
    if knots is None:
        return None
    if case["rule"] == "quantiles" and np.unique(knots).size < knots.size:
        return "R's rule places two knots at one dose: ns() takes them, DoseAxis raises"
    if case["rule"] == "harrell":
        if "tie at" in message:
            return "Hmisc switches to its method for values tied at an extreme"
        if knots.size < case["df"] + 1:
            return "Hmisc drops a repeated knot or places fewer knots than asked"
    return None


def compare(inputs: list[dict], results: dict) -> tuple[dict, dict]:
    """Per design and axis, the largest differences; and the deliberate differences."""
    checks: dict[tuple[str, str], Checks] = {}
    departures: dict[str, list[str]] = {}
    for case in inputs:
        reference = results[case["name"]]
        found = checks.setdefault((case["design"], case["axis_name"]), Checks())
        axis, doses = case["axis"], case["doses"]
        samples = Table(
            {"sample_id": [f"S{i}" for i in range(doses.size)], "dose": doses}
        )
        found.record(
            "positions", relative(axis.positions(doses), reference["positions"])
        )
        knots = case["given"] if case["rule"] == "given" else case["rule"]
        try:
            basis = axis.spline_basis(
                samples, case["df"], knots=knots, form=case["form"]
            )
        except ValueError as error:
            explained = departure(case, reference, str(error))
            if explained is not None:
                departures.setdefault(explained, []).append(case["name"])
            elif reference["knots"] is not None:
                found.differing.append(f"{case['name']}: {error}")
            continue
        if reference.get("columns") is None:
            found.differing.append(f"{case['name']}: R placed no knots or columns")
            continue
        lower, upper = basis.boundary_knots
        found.record(
            "knots",
            relative(np.array([lower, *basis.knots, upper]), reference["knots"]),
        )
        columns = axis.design_columns(samples, basis)
        if "dose_treated" in columns:
            same = np.array_equal(columns["dose_treated"], reference["treated"])
            found.record("treated", 0.0 if same else np.inf)
        names = list(columns)
        ours = np.column_stack([columns[name] for name in names])
        spline = slice(1, None) if "dose_treated" in columns else slice(None)
        if case["form"] == "restricted_cubic":
            found.record(
                "restricted cubic columns", by_column(ours, reference["columns"])
            )
        else:
            found.record(
                "B-spline columns",
                column_space(ours[:, spline], reference["columns"][:, spline]),
            )
        design = Design(samples.with_columns(columns), "~ " + " + ".join(names)).matrix
        if case["form"] == "restricted_cubic":
            found.record("design matrices", by_column(design, reference["design"]))
        else:
            found.record("design matrices", column_space(design, reference["design"]))
        condition = float(np.linalg.cond(design))
        if condition > CONDITION_LIMIT:
            found.ill_conditioned.append(
                (condition, reference["rank"] < design.shape[1])
            )
            found.record("cases", 0.0)
            continue
        position = {name: j for j, name in enumerate(["(Intercept)", *names])}
        dropped = {
            "any_dose": list(range(1, len(names) + 1)),
            "among_doses": [position[n] for n in names if n.startswith("dose_s")],
            "nonlinear": [position[n] for n in names if n.startswith("dose_s")][1:],
        }
        f = f_statistics(design, case["y"], [dropped[t] for t in reference["tests"]])
        found.record("F-statistics", relative(f, reference["f"]))
        grid_columns = axis.design_columns({"dose": case["grid"]}, basis)
        grid_design = np.column_stack(
            [np.ones(case["grid"].size), *(grid_columns[name] for name in names)]
        )
        coefficients = np.linalg.lstsq(design, case["y"], rcond=None)[0]
        found.record(
            "curves", relative(grid_design @ coefficients, reference["curves"])
        )
        found.record("cases", 0.0)
    return checks, departures


ROWS = {
    "positions": "positions",
    "treated": "dose_treated",
    "knots": "knots and boundary knots",
    "restricted cubic columns": "restricted cubic columns",
    "B-spline columns": "B-spline columns (column space)",
    "design matrices": "design matrices",
    "F-statistics": "F-statistics",
    "curves": "curves on the grid",
}
REFERENCES = {
    "positions": "log10",
    "treated": "dose > 0",
    "knots": "ns, rcspline.eval",
    "restricted cubic columns": "rcspline.eval",
    "B-spline columns": "ns",
    "design matrices": "model.matrix",
    "F-statistics": "anova(lm)",
    "curves": "predict(lm)",
}


def table(checks: dict, departures: dict) -> tuple[list[tuple[str, ...]], int]:
    rows, failures = [], 0
    for (design, axis), found in checks.items():
        cases = found.counts.get("cases", 0)
        for kind, label in ROWS.items():
            if kind not in found.largest:
                continue
            value = found.largest[kind]
            tolerance = 0.0 if kind == "treated" else EXACT
            passed = value <= tolerance
            failures += not passed
            shown = "0 mismatches" if kind == "treated" and passed else f"{value:.1e}"
            check = f"{design}, {axis}: {label} ({found.counts[kind]} cases)"
            rows.append(
                (
                    check,
                    REFERENCES[kind],
                    shown,
                    f"{tolerance:g}",
                    "pass" if passed else "FAIL",
                )
            )
        if found.ill_conditioned:
            largest = max(condition for condition, _ in found.ill_conditioned)
            lost = sum(rank_lost for _, rank_lost in found.ill_conditioned)
            check = (
                f"{design}, {axis}: F-statistics and curves of "
                f"{len(found.ill_conditioned)} cases whose design has a condition "
                f"number above {CONDITION_LIMIT:g}, up to {largest:.1e}; R's lm drops a "
                f"column as aliased in {lost}"
            )
            rows.append((check, "lm", "not compared", "n/a", "info"))
        differing = len(found.differing)
        failures += differing > 0
        check = f"{design}, {axis}: outcomes that differ ({cases} cases compared)"
        rows.append(
            (check, "R", str(differing), "0", "pass" if not differing else "FAIL")
        )
        for line in found.differing:
            print(f"differs: {line}")
    for description, names in departures.items():
        rows.append(
            (
                f"{description}: {len(names)} cases",
                "R",
                "DoseAxis raises",
                "n/a",
                "deviation",
            )
        )
    return rows, failures


FIXTURE_CASES = (
    "tcdd_ladder | own group, log10 | restricted_cubic | levels | df 3",
    "tcdd_ladder | own group, log10 | restricted_cubic | levels | df 7",
    "tcdd_ladder | own group, log10 | b_spline | levels | df 7",
    "tcdd_ladder | own group, log10 | b_spline | quantiles | df 3",
    "tcdd_ladder | own group, log10 | b_spline | given | df 3",
    "half_log_wells | on axis, log10 with offset | restricted_cubic | levels | df 2",
    "half_log_wells | on axis, linear | b_spline | levels | df 4",
    "exposures | own group, linear | restricted_cubic | harrell | df 4",
    "exposures | on axis, log10 with offset | restricted_cubic | harrell | df 4",
)


def write_fixtures(inputs: list[dict], results: dict, departures: dict) -> None:
    """A few cases: the axis, doses, responses, grid, rule, df, form and given knots,
    and R's positions, treated, knots, columns, design matrix, F-statistics (with the
    names of the tests) and curves; and a case of each deliberate difference."""
    tied = [n for d, names in departures.items() if d.startswith("R's") for n in names]
    chosen = [*FIXTURE_CASES, *tied[:1]]
    by_name = {case["name"]: case for case in inputs}
    arrays: dict[str, np.ndarray] = {"names": np.array(chosen)}
    for i, name in enumerate(chosen):
        case, reference = by_name[name], results[name]
        axis = case["axis"]
        record = {
            "unit": axis.unit,
            "metric": axis.metric,
            "scale": axis.scale,
            "vehicle": axis.vehicle,
            "offset": np.nan if axis.offset is None else axis.offset,
            "rule": case["rule"],
            "df": case["df"],
            "form": case["form"],
            "doses": case["doses"],
            "y": case["y"],
            "grid": case["grid"],
            "given": case["given"],
            **{key: value for key, value in reference.items() if value is not None},
        }
        for key, value in record.items():
            arrays[f"case{i}__{key}"] = np.asarray(value)
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_DIRECTORY / "dose_axes.npz", **arrays)
    print(f"wrote dose_axes.npz to {FIXTURE_DIRECTORY}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    inputs = cases()
    results, version = run_r(inputs)
    print(f"Reference: {version}")
    checks, departures = compare(inputs, results)
    rows, failures = table(checks, departures)
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(inputs, results, departures)
    print(f"{len(rows)} checks, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
