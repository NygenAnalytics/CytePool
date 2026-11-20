"""Check LOWESS and natural cubic spline bases against R, Hmisc and rms.

Compares ``cytepool.stats.smoothing.lowess`` with R's ``stats::lowess``, and
``natural_spline_basis`` and ``NaturalSplineBasis`` with R's ``splines::ns``, Hmisc's
``rcspline.eval`` and rms's ``rcs``, on inputs generated from fixed seeds.

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

The knots that ``NaturalSplineBasis`` places must equal R's to the last bit
(tolerance 0): with R's rule (``"quantiles"``), ``attr(ns(x, df), "knots")`` and
``"Boundary.knots"`` of every spline case with ``df``, and of 11 seeded datasets (from
30 to 2,000 values, dose ladders, ties) at df 2 to 8 with and without an intercept;
with the rule on the distinct values (``"levels"``), ``attr(ns(unique(x), df),
"knots")``. A basis evaluated at new points, inside the boundary knots, at the knots,
beyond both boundary knots and at NaN, is compared with R's ``predict(basis, newx)``
through least squares: random responses are fitted at x on either basis and
predicted at the new points (tier exact, relative to the largest prediction), in
the cases where the basis has full column rank.

Harrell's rule (``"harrell"``) is compared with Hmisc's ``rcspline.eval(x, nk,
knots.only = TRUE)`` for nk from 2 to 9 on 26 seeded datasets: samples of 6 to 200
values, NaN, dose ladders and values tied at an extreme or between the extremes. Its
knots must be identical; where Hmisc returns fewer knots than asked, switches to the
method it uses for values tied at an extreme, or drops a repeated knot, CytePool
raises an error instead (deliberate differences); where Hmisc fails, CytePool must
fail too. A case counts as Hmisc's method for ties only where Hmisc's knots differ
from the percentiles that its help page gives, computed here without CytePool; a
refusal for ties where Hmisc kept the percentiles counts as a mismatch. The
restricted cubic form is compared with ``rcspline.eval(x, knots, inclx = TRUE)`` for
3 to 8 knots at points from below the first to far beyond the last knot: its first
column plus the lower boundary knot against Hmisc's x, relative to the largest
absolute value, and Hmisc's columns, with an intercept, as combinations of
CytePool's, by the largest residual relative to each column's largest absolute value
(tier exact). CytePool's columns after the first differ from Hmisc's truncated cubes
by a change of basis that keeps them well conditioned, and regressing Hmisc's columns
on them keeps rounding small where Hmisc's are nearly collinear. The overall and
non-linear F-statistics of ``anova(ols(y ~ rcs(x, nk)))`` (rms) are compared with
those of least squares fits on an intercept and the restricted cubic columns, and the
overall F with the B-spline form on the same knots (tier exact), and rms's knots with
Harrell's rule (identical).

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer (the stats and splines packages ship with R), with the CRAN
    packages Hmisc and rms. Set ``RSCRIPT`` to the Rscript executable when it is not
    on PATH.

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

from cytepool.stats.smoothing import NaturalSplineBasis, lowess, natural_spline_basis

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "smoothing"
CLOSE = 1e-6
EXACT = 1e-10
LARGEST_FIXTURE = 300

R_SCRIPT = r"""
suppressPackageStartupMessages({library(splines); library(Hmisc); library(rms)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
read_doubles <- function(path, n) readBin(path, "double", n = n, size = 8, endian = "little")
read_all <- function(path) read_doubles(path, file.size(path) / 8)
write_doubles <- function(x, path) writeBin(as.double(x), path, size = 8, endian = "little")
numbers <- function(line) if (line == "none") numeric(0) else as.numeric(strsplit(line, ",")[[1]])
# Hmisc prints a table of x when it fails; the comparison reads only the files.
quietly <- function(expr) {
  out <- NULL
  invisible(capture.output(
    out <- tryCatch(suppressWarnings(expr), error = function(e) NULL)
  ))
  out
}
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
    write_doubles(attr(basis, "knots"), file.path(dir, "r_knots.bin"))
    write_doubles(attr(basis, "Boundary.knots"), file.path(dir, "r_boundary.bin"))
    # The new points, with the knots and boundary knots added.
    newx <- read_all(file.path(dir, "newx.bin"))
    newx[is.nan(newx)] <- NA
    newx <- c(newx, attr(basis, "knots"), attr(basis, "Boundary.knots"))
    write_doubles(newx, file.path(dir, "r_newx.bin"))
    write_doubles(predict(basis, newx), file.path(dir, "r_predict.bin"))
    # R's rule on the distinct values, where the rule places the knots.
    if (p[2] != "default" && p[4] == "default") {
      args$x <- unique(x)
      levels <- tryCatch(suppressWarnings(do.call(splines::ns, args)), error = function(e) NULL)
      if (!is.null(levels)) write_doubles(attr(levels, "knots"), file.path(dir, "r_levels.bin"))
    }
  }
}
# R's knots on seeded datasets: ns(x, df) and ns(unique(x), df).
for (case in readLines(file.path(root, "knots.txt"))) {
  dir <- file.path(root, "knots", case)
  x <- read_all(file.path(dir, "x.bin"))
  for (df in 2:8) for (intercept in c(FALSE, TRUE)) {
    label <- paste0("df", df, "_", as.integer(intercept))
    basis <- tryCatch(suppressWarnings(ns(x, df = df, intercept = intercept)), error = function(e) NULL)
    if (!is.null(basis)) {
      write_doubles(attr(basis, "knots"), file.path(dir, paste0("knots_", label, ".bin")))
      write_doubles(attr(basis, "Boundary.knots"), file.path(dir, paste0("boundary_", label, ".bin")))
    }
    levels <- tryCatch(suppressWarnings(ns(unique(x), df = df, intercept = intercept)), error = function(e) NULL)
    if (!is.null(levels)) write_doubles(attr(levels, "knots"), file.path(dir, paste0("levels_", label, ".bin")))
  }
}
# Hmisc's knots by Harrell's rule.
for (case in readLines(file.path(root, "harrell.txt"))) {
  dir <- file.path(root, "harrell", case)
  x <- read_all(file.path(dir, "x.bin"))
  x[is.nan(x)] <- NA
  for (nk in 2:9) {
    knots <- quietly(rcspline.eval(x, nk = nk, knots.only = TRUE))
    if (!is.null(knots)) write_doubles(knots, file.path(dir, paste0("knots_nk", nk, ".bin")))
  }
}
# Hmisc's restricted cubic columns, x first, at given knots.
for (case in readLines(file.path(root, "rcs.txt"))) {
  dir <- file.path(root, "rcs", case)
  knots <- read_all(file.path(dir, "knots.bin"))
  newx <- read_all(file.path(dir, "newx.bin"))
  newx[is.nan(newx)] <- NA
  basis <- quietly(rcspline.eval(newx, knots = knots, inclx = TRUE))
  if (!is.null(basis)) {
    write_doubles(dim(basis), file.path(dir, "r_dim.bin"))
    write_doubles(basis, file.path(dir, "r_basis.bin"))
  }
}
# rms: ols(y ~ rcs(x, nk)), its anova's overall and non-linear F, and its knots.
for (case in readLines(file.path(root, "rms.txt"))) {
  dir <- file.path(root, "rms", case)
  x <- read_all(file.path(dir, "x.bin"))
  x[is.nan(x)] <- NA
  y <- read_all(file.path(dir, "y.bin"))
  nk <- as.integer(readLines(file.path(dir, "nk.txt")))
  data <- data.frame(x = x, y = y)
  fit <- ols(y ~ rcs(x, nk), data = data)
  table <- anova(fit)
  write_doubles(c(table[1, "F"], table[2, "F"], table[1, "d.f."], table[2, "d.f."],
                  table[nrow(table), "d.f."]), file.path(dir, "r_anova.bin"))
  write_doubles(fit$Design$parms$x, file.path(dir, "r_knots.bin"))
}
version <- function(name) paste(name, packageDescription(name, fields = "Version"))
writeLines(paste0("R ", as.character(getRversion()), " (stats::lowess, splines::ns), ",
                  version("Hmisc"), ", ", version("rms")), file.path(root, "versions.txt"))
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
    cases = [
        dict(name=name, x=np.asarray(a, float)) | defaults | options
        for name, a, options in specs
    ]
    for case in cases:
        case["newx"] = new_points(case["x"], case["boundary_knots"])
    return cases


def new_points(x: np.ndarray, boundary_knots) -> np.ndarray:
    """Where a basis is evaluated anew: 23 points from half the range below the lower
    boundary knot to half the range above the upper one, and NaN; R adds the knots."""
    observed = x[~np.isnan(x)]
    if boundary_knots is None:
        lower, upper = float(observed.min()), float(observed.max())
    else:
        lower, upper = sorted(float(b) for b in boundary_knots)
    width = upper - lower
    return np.r_[np.linspace(lower - 0.5 * width, upper + 0.5 * width, 23), np.nan]


DOSES = np.log10([0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0])


def knot_datasets() -> dict[str, np.ndarray]:
    """Inputs for R's knots, ns(x, df) and ns(unique(x), df) at df 2 to 8."""
    rng = np.random.default_rng(20261120)
    ladder = np.repeat(DOSES, 3)
    return {
        "normal150": rng.normal(0.0, 1.0, 150),
        "exponential99": rng.exponential(1.0, 99),
        "uniform30": rng.uniform(0.0, 1.0, 30),
        "normal2000": rng.normal(6.0, 2.5, 2000),
        "ladder": ladder,
        "ladder_unequal": np.repeat(DOSES, [3, 3, 2, 3, 3, 2, 3, 3]),
        "ladder_offset": np.r_[np.full(4, np.log10(0.001)), ladder],
        "ladder_nine": np.repeat(np.r_[DOSES, 2.0], [2, 3, 3, 3, 2, 3, 3, 3, 2]),
        "half_log": np.repeat([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], 4),
        "integers": rng.integers(0, 10, 80).astype(float),
        "rounded": np.round(rng.normal(0.0, 1.0, 500), 2),
    }


def harrell_datasets() -> dict[str, np.ndarray]:
    """Inputs for Hmisc's knots by Harrell's rule, rcspline.eval(x, nk) at nk 2 to 9:
    samples above and below 100 values, NaN, dose ladders, and values tied at an
    extreme (at 5% and just below it) or between the extremes."""
    rng = np.random.default_rng(20261121)
    uniform = rng.uniform
    return {
        "normal150": rng.normal(0.0, 1.0, 150),
        "exponential100": rng.exponential(1.0, 100),
        "exponential99": rng.exponential(1.0, 99),
        "normal60": rng.normal(0.0, 1.0, 60),
        "uniform30": uniform(0.0, 1.0, 30),
        "uniform12": uniform(0.0, 1.0, 12),
        "uniform10": uniform(0.0, 1.0, 10),
        "uniform9": uniform(0.0, 1.0, 9),
        "uniform6": uniform(0.0, 1.0, 6),
        "nan_in_x": np.r_[np.nan, np.nan, rng.exponential(1.0, 98)],
        "ladder": np.repeat(DOSES, 3),
        "ladder_unequal": np.repeat(DOSES, [3, 3, 2, 3, 3, 2, 3, 3]),
        "ladder_offset": np.r_[np.full(4, np.log10(0.001)), np.repeat(DOSES, 3)],
        "ladder_15_each": np.repeat(DOSES, 15),
        "ladder_five": np.repeat([1.0, 2.0, 3.0, 4.0, 5.0], 4),
        "four_values": np.repeat([1.0, 2.0, 3.0, 4.0], 10),
        "ties_at_min_10pct": np.r_[np.zeros(10), uniform(0.0, 1.0, 90)],
        "ties_at_min_5pct": np.r_[np.zeros(10), uniform(0.0, 1.0, 190)],
        "ties_at_min_4.5pct": np.r_[np.zeros(9), uniform(0.0, 1.0, 191)],
        "ties_at_max_10pct": np.r_[uniform(0.0, 1.0, 90), np.ones(10)],
        "ties_at_both_ends": np.r_[np.zeros(10), uniform(0.0, 1.0, 80), np.ones(10)],
        "ties_at_min_and_between": np.r_[
            np.zeros(10), np.full(10, 0.5), uniform(0.0, 1.0, 80)
        ],
        "ties_at_min_n46": np.r_[np.zeros(6), uniform(0.0, 1.0, 40)],
        "ties_at_min_few_values": np.r_[np.zeros(20), 1.0, 2.0, 3.0, 4.0, 5.0],
        "ties_at_max_few_values": np.r_[1.0, 2.0, 3.0, 4.0, 5.0, np.full(20, 9.0)],
        "ties_at_min_seven_values": np.r_[np.zeros(30), np.arange(1.0, 7.0)],
    }


def restricted_cubic_cases() -> list[dict]:
    """Knots and points for Hmisc's restricted cubic columns, rcspline.eval(x, knots,
    inclx = TRUE): 3 to 8 knots at random, and a dose ladder; points from below the
    first knot to far beyond the last, the knots themselves, and NaN."""
    rng = np.random.default_rng(20261122)
    cases = []
    for count in range(3, 9):
        knots = np.sort(rng.uniform(-1.2, 1.5, count))
        cases.append({"name": f"knots_{count}", "knots": knots})
    cases.append({"name": "dose_ladder_knots_5", "knots": DOSES[[0, 2, 4, 6, 7]]})
    # Hmisc drops a repeated knot; CytePool's restricted cubic form refuses it.
    cases.append({"name": "repeated_knot", "knots": np.array([-1.2, 0.4, 0.4, 1.5])})
    for case in cases:
        knots = case["knots"]
        case["newx"] = np.r_[np.linspace(-3.0, 3.0, 25), knots, 10.0, 50.0, np.nan]
    return cases


def rms_cases(harrell_inputs: dict[str, np.ndarray]) -> list[dict]:
    """Responses for rms's ols(y ~ rcs(x, nk)) on datasets of harrell_datasets."""
    rng = np.random.default_rng(20261123)
    cases = []
    for name, nk in (
        ("normal150", 4),
        ("normal150", 5),
        ("exponential99", 5),
        ("uniform30", 3),
        ("ladder", 4),
        ("nan_in_x", 4),
    ):
        x = harrell_inputs[name]
        y = np.sin(2.0 * np.nan_to_num(x)) + rng.normal(0.0, 0.3, x.size)
        cases.append({"name": f"{name}_nk{nk}", "x": x, "y": y, "nk": nk})
    return cases


def run_r(
    lowess_inputs: list[dict],
    spline_inputs: list[dict],
    knot_inputs: dict[str, np.ndarray],
    harrell_inputs: dict[str, np.ndarray],
    rcs_inputs: list[dict],
    rms_inputs: list[dict],
) -> tuple[dict, str]:
    def text(values) -> str:
        values = np.atleast_1d(np.asarray(values, float))
        return ",".join(repr(float(v)) for v in values) if values.size else "none"

    def write(path: Path, values) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(np.asarray(values, float).astype("<f8").tobytes())

    def read(path: Path) -> np.ndarray | None:
        if not path.exists():
            return None
        values = np.fromfile(path, "<f8")
        values[np.isnan(values)] = np.nan  # R's NA is a signalling NaN
        return values

    def matrix(directory: Path, name: str, rows: int | None = None) -> np.ndarray:
        values = read(directory / name)
        if rows is None:
            rows, columns = read(directory / "r_dim.bin").astype(int)
        else:
            columns = values.size // rows if rows else 0
        return values.reshape(columns, rows).T.copy()

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for kind, inputs in (("lowess", lowess_inputs), ("ns", spline_inputs)):
            (root / f"{kind}.txt").write_text("".join(c["name"] + "\n" for c in inputs))
            for case in inputs:
                write(root / kind / case["name"] / "x.bin", case["x"])
        for case in lowess_inputs:
            directory = root / "lowess" / case["name"]
            write(directory / "y.bin", case["y"])
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
            directory = root / "ns" / case["name"]
            (directory / "parameters.txt").write_text("\n".join(lines) + "\n")
            write(directory / "newx.bin", case["newx"])
        for kind, inputs in (("knots", knot_inputs), ("harrell", harrell_inputs)):
            (root / f"{kind}.txt").write_text("".join(f"{name}\n" for name in inputs))
            for name, x in inputs.items():
                write(root / kind / name / "x.bin", x)
        (root / "rcs.txt").write_text("".join(c["name"] + "\n" for c in rcs_inputs))
        for case in rcs_inputs:
            write(root / "rcs" / case["name"] / "knots.bin", case["knots"])
            write(root / "rcs" / case["name"] / "newx.bin", case["newx"])
        (root / "rms.txt").write_text("".join(c["name"] + "\n" for c in rms_inputs))
        for case in rms_inputs:
            directory = root / "rms" / case["name"]
            write(directory / "x.bin", case["x"])
            write(directory / "y.bin", case["y"])
            (directory / "nk.txt").write_text(f"{case['nk']}\n")
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
            basis = matrix(directory, "r_basis.bin")
            results["ns", case["name"]] = basis
            newx = read(directory / "r_newx.bin")
            results["ns at new points", case["name"]] = {
                "knots": read(directory / "r_knots.bin"),
                "boundary_knots": read(directory / "r_boundary.bin"),
                "newx": newx,
                "basis": matrix(directory, "r_predict.bin", newx.size),
                "levels": read(directory / "r_levels.bin"),
            }
        for name in knot_inputs:
            directory = root / "knots" / name
            results["knots", name] = {
                (df, intercept): {
                    key: read(directory / f"{key}_df{df}_{int(intercept)}.bin")
                    for key in ("knots", "boundary", "levels")
                }
                for df in range(2, 9)
                for intercept in (False, True)
            }
        for name in harrell_inputs:
            directory = root / "harrell" / name
            results["harrell", name] = {
                nk: read(directory / f"knots_nk{nk}.bin") for nk in range(2, 10)
            }
        for case in rcs_inputs:
            directory = root / "rcs" / case["name"]
            failed = not (directory / "r_basis.bin").exists()
            results["rcs", case["name"]] = (
                None if failed else matrix(directory, "r_basis.bin")
            )
        for case in rms_inputs:
            directory = root / "rms" / case["name"]
            results["rms", case["name"]] = (
                read(directory / "r_anova.bin"),
                read(directory / "r_knots.bin"),
            )
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


def placed_knots(basis: NaturalSplineBasis) -> np.ndarray:
    """All the knots of a basis, the boundary knots included, in order."""
    lower, upper = basis.boundary_knots
    return np.array([lower, *basis.knots, upper])


def compare_spline_knots(cases: list[dict], results: dict, table: Table) -> None:
    """Knots of R's rule and of the rule on the distinct values, and bases at new
    points, of the spline cases for which R returns a basis."""
    rng = np.random.default_rng(11)
    differ, levels_differ, placed, levels = [], [], 0, 0
    for case in cases:
        reference = results.get(("ns at new points", case["name"]))
        if reference is None or case["name"].startswith("deviation"):
            continue
        keywords = {k: case[k] for k in ("intercept", "boundary_knots")}
        given = case["knots"]
        basis = NaturalSplineBasis(
            case["x"],
            case["df"],
            knots="quantiles" if given is None else given,
            **keywords,
        )
        if given is None and case["df"] is not None:
            placed += 1
            same = np.array_equal(
                np.array(basis.knots), reference["knots"]
            ) and np.array_equal(
                np.array(basis.boundary_knots), reference["boundary_knots"]
            )
            if not same:
                differ.append(case["name"])
            if reference["levels"] is not None:
                levels += 1
                ours = NaturalSplineBasis(
                    case["x"], case["df"], knots="levels", **keywords
                )
                if not np.array_equal(np.array(ours.knots), reference["levels"]):
                    levels_differ.append(case["name"])
        # Predictions at the new points of least squares fits at x on either basis.
        train = results["ns", case["name"]]
        ours_train = basis.matrix(case["x"])
        complete = ~np.isnan(train).any(axis=1)
        if np.linalg.matrix_rank(train[complete]) < train.shape[1]:
            continue
        responses = rng.normal(0.0, 1.0, (int(complete.sum()), 3))
        r_coefficients = np.linalg.lstsq(train[complete], responses, rcond=None)[0]
        coefficients = np.linalg.lstsq(ours_train[complete], responses, rcond=None)[0]
        r_predicted = reference["basis"] @ r_coefficients
        predicted = basis.matrix(reference["newx"]) @ coefficients
        missing = np.isnan(r_predicted[:, 0])
        difference = np.inf
        if np.array_equal(missing, np.isnan(predicted[:, 0])):
            scale = float(np.max(np.abs(r_predicted[~missing])))
            difference = (
                float(np.max(np.abs(predicted - r_predicted)[~missing])) / scale
            )
        check = f"ns {case['name']} at {reference['newx'].size} new points"
        metric = "LS predictions: max diff / max abs"
        table.add(check, "predict(ns)", metric, difference, EXACT)
    check = f"knots of the {placed} spline cases whose knots R's rule places"
    if differ:
        check += f" ({', '.join(differ)})"
    table.add(
        check, "attr(ns(x, df))", "cases that differ", len(differ), 0, str(len(differ))
    )
    check = f"knots on the distinct values of the {levels} cases above"
    if levels_differ:
        check += f" ({', '.join(levels_differ)})"
    shown = str(len(levels_differ))
    table.add(
        check, "ns(unique(x), df)", "cases that differ", len(levels_differ), 0, shown
    )


def compare_knot_datasets(datasets: dict, results: dict, table: Table) -> None:
    """R's knots, with ns(x, df) and ns(unique(x), df), at df 2 to 8 with and without
    an intercept: identical, and failing where R fails."""
    for name, x in datasets.items():
        for rule, key in (("quantiles", "knots"), ("levels", "levels")):
            differ = 0
            for (df, intercept), reference in results["knots", name].items():
                try:
                    basis = NaturalSplineBasis(x, df, knots=rule, intercept=intercept)
                except ValueError:
                    differ += int(reference[key] is not None)
                    continue
                if reference[key] is None:
                    differ += 1
                    continue
                same = np.array_equal(np.array(basis.knots), reference[key])
                if rule == "quantiles":
                    same = same and np.array_equal(
                        np.array(basis.boundary_knots), reference["boundary"]
                    )
                differ += not same
            call = "ns(x, df)" if rule == "quantiles" else "ns(unique(x), df)"
            check = f"knots by {rule!r}, {name} (n={x.size}), df 2 to 8, 14 cases"
            table.add(check, call, "cases that differ", differ, 0, str(differ))


HARRELL_DEPARTURES = {
    "tie": "Hmisc switches to its method for values tied at an extreme",
    "few": "Hmisc places fewer knots than asked",
    "equal": "Hmisc drops a repeated knot and places fewer knots than asked",
}


def percentile_knots(x: np.ndarray, nk: int) -> np.ndarray:
    """Harrell's percentiles as the help page of rcspline.eval gives them, without its
    method for ties: quantiles of type 7 at nk equally spaced probabilities from lo to
    1 - lo, lo 0.1 for 3 knots, 0.05 for 4 to 6 and 0.025 for more, the outer two
    replaced by the 5th smallest and largest values below 100 values, sorted."""
    values = np.sort(x[~np.isnan(x)])
    outer = 0.1 if nk == 3 else 0.05 if nk <= 6 else 0.025
    knots = np.quantile(values, np.linspace(outer, 1.0 - outer, nk))
    if values.size < 100:
        knots[0], knots[-1] = values[4], values[-5]
    return np.sort(knots)


def harrell_outcome(x: np.ndarray, nk: int, hmisc: np.ndarray | None) -> str:
    """How CytePool's knots by Harrell's rule compare with Hmisc's: "identical",
    "both fail", one of HARRELL_DEPARTURES (a deliberate difference), or "mismatch"."""
    try:
        basis = NaturalSplineBasis(x, nk - 1, knots="harrell")
    except ValueError as error:
        if hmisc is None:
            return "both fail"
        message = str(error)
        if "tie at" in message:
            # Hmisc's method for ties only where Hmisc left the percentiles.
            kept = hmisc.size == nk and np.allclose(
                percentile_knots(x, nk), hmisc, rtol=1e-12, atol=0.0
            )
            return "mismatch" if kept else "tie"
        if hmisc.size < nk and "distinct values" in message:
            return "few"
        if hmisc.size < nk and "two knots at" in message:
            return "equal"
        return "mismatch"
    if hmisc is not None and np.array_equal(placed_knots(basis), hmisc):
        return "identical"
    return "mismatch"


def compare_harrell(datasets: dict, results: dict, table: Table) -> None:
    departures: dict[str, list[str]] = {kind: [] for kind in HARRELL_DEPARTURES}
    for name, x in datasets.items():
        outcomes = {
            nk: harrell_outcome(x, nk, knots)
            for nk, knots in results["harrell", name].items()
        }
        counts = {
            kind: list(outcomes.values()).count(kind) for kind in set(outcomes.values())
        }
        for kind in HARRELL_DEPARTURES:
            departures[kind] += [
                f"{name} nk={nk}" for nk, o in outcomes.items() if o == kind
            ]
        mismatches = counts.get("mismatch", 0)
        observed = x[~np.isnan(x)]
        check = (
            f"Harrell's knots, {name} (n={observed.size}, {np.unique(observed).size} "
            f"distinct), nk 2 to 9: {counts.get('identical', 0)} identical, "
            f"{counts.get('both fail', 0)} fail in both"
        )
        if mismatches:
            check += f" ({', '.join(f'nk={nk}' for nk, o in outcomes.items() if o == 'mismatch')})"
        metric = "cases with other knots or outcome"
        table.add(check, "rcspline.eval", metric, mismatches, 0, str(mismatches))
    for kind, description in HARRELL_DEPARTURES.items():
        if departures[kind]:
            check = f"{description}: {len(departures[kind])} cases"
            table.info(
                check, "rcspline.eval", "behaviour", "CytePool raises", "deviation"
            )


def compare_restricted_cubic(cases: list[dict], results: dict, table: Table) -> None:
    for case in cases:
        reference = results["rcs", case["name"]]
        knots = case["knots"]
        if case["name"] == "repeated_knot":
            try:
                NaturalSplineBasis(
                    knots=knots[1:-1],
                    boundary_knots=knots[[0, -1]],
                    form="restricted_cubic",
                )
                outcome = "no deviation"
            except ValueError:
                outcome = (
                    f"Hmisc drops the repeated knot and returns {reference.shape[1] - 1} "
                    f"non-linear columns instead of {knots.size - 2}; CytePool raises"
                )
            table.info(
                "restricted cubic columns with a repeated knot",
                "rcspline.eval",
                "behaviour",
                outcome,
                "deviation",
            )
            continue
        basis = NaturalSplineBasis(
            knots=knots[1:-1], boundary_knots=knots[[0, -1]], form="restricted_cubic"
        )
        ours = basis.matrix(case["newx"])
        label = (
            f"restricted cubic columns, {case['name']} ({knots.size} knots), "
            f"{case['newx'].size} points to {np.nanmax(case['newx']):g} and NaN"
        )
        first = combined = np.inf
        if ours.shape == reference.shape and np.array_equal(
            np.isnan(ours), np.isnan(reference)
        ):
            present = ~np.isnan(reference[:, 0])
            first = float(
                np.max(np.abs(ours[present, 0] + knots[0] - reference[present, 0]))
                / np.max(np.abs(reference[present, 0]))
            )
            ones = np.ones((int(present.sum()), 1))
            combined = combinations(
                np.hstack([ones, ours[present]]), np.hstack([ones, reference[present]])
            )
        table.add(
            f"{label}: first column plus the lower knot",
            "rcspline.eval",
            "max diff / max abs",
            first,
            EXACT,
        )
        table.add(
            f"{label}: Hmisc's columns as combinations of CytePool's",
            "rcspline.eval",
            "max residual / column max abs",
            combined,
            EXACT,
        )


def combinations(ours: np.ndarray, reference: np.ndarray) -> float:
    """The reference's columns regressed on ours: the largest residual relative to each
    column's largest absolute value; infinite when the combinations lose rank, so that
    the column spaces differ."""
    weights = np.linalg.lstsq(ours, reference, rcond=None)[0]
    if np.linalg.matrix_rank(weights) < reference.shape[1]:
        return np.inf
    residuals = np.max(np.abs(reference - ours @ weights), axis=0)
    return float(np.max(residuals / np.max(np.abs(reference), axis=0)))


def f_statistics(
    design: np.ndarray, y: np.ndarray, tested: list[list[int]]
) -> list[float]:
    """F-statistics of least squares fits that drop each list of columns."""

    def rss(columns: np.ndarray) -> float:
        residuals = y - columns @ np.linalg.lstsq(columns, y, rcond=None)[0]
        return float(residuals @ residuals)

    full = rss(design)
    df_residual = y.size - design.shape[1]
    values = []
    for dropped in tested:
        kept = [j for j in range(design.shape[1]) if j not in dropped]
        values.append(
            ((rss(design[:, kept]) - full) / len(dropped)) / (full / df_residual)
        )
    return values


def compare_rms(cases: list[dict], results: dict, table: Table) -> None:
    for case in cases:
        anova, rms_knots = results["rms", case["name"]]
        complete = ~np.isnan(case["x"])
        x, y, nk = case["x"][complete], case["y"][complete], case["nk"]
        basis = NaturalSplineBasis(
            case["x"], nk - 1, knots="harrell", form="restricted_cubic"
        )
        knots_differ = int(not np.array_equal(placed_knots(basis), rms_knots))
        check = f"rms {case['name']}: knots of rcs(x, {nk})"
        table.add(
            check, "rms ols", "knots that differ", knots_differ, 0, str(knots_differ)
        )
        design = np.column_stack([np.ones(x.size), basis.matrix(x)])
        overall, nonlinear = f_statistics(
            design, y, [list(range(1, nk)), list(range(2, nk))]
        )
        b_spline = NaturalSplineBasis(
            knots=basis.knots, boundary_knots=basis.boundary_knots
        ).matrix(x)
        (b_overall,) = f_statistics(
            np.column_stack([np.ones(x.size), b_spline]), y, [list(range(1, nk))]
        )
        for label, ours, reference in (
            ("overall F", overall, anova[0]),
            ("non-linear F", nonlinear, anova[1]),
            ("overall F of the B-spline form", b_overall, anova[0]),
        ):
            difference = abs(ours - reference) / abs(reference)
            check = f"rms {case['name']} (n={x.size}): {label}"
            table.add(
                check, "anova(ols(y ~ rcs(x, nk)))", "relative diff", difference, EXACT
            )


def write_fixtures(
    lowess_inputs,
    spline_inputs,
    knot_inputs,
    harrell_inputs,
    rcs_inputs,
    rms_inputs,
    results,
) -> None:
    """Small cases: lowess.npz holds one (3, n) array of x, y and R's fitted
    values per case, with spans, iterations and delta (NaN: default) in
    ``parameters``; natural_splines.npz holds x, the arguments, R's basis, knots,
    boundary knots, knots on the distinct values, new points and basis there
    (absent where R fails), and R's knots of the seeded datasets of up to 300 values;
    restricted_cubic_splines.npz holds Hmisc's knots by Harrell's rule (absent where
    Hmisc fails), its restricted cubic columns, and rms's F-statistics and knots."""
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
            reference = results["ns at new points", c["name"]]
            for key in ("knots", "boundary_knots", "levels", "newx"):
                if reference[key] is not None:
                    arrays[f"{c['name']}__r_{key}"] = reference[key]
            arrays[c["name"] + "__r_newx_basis"] = reference["basis"]
    datasets = [name for name, x in knot_inputs.items() if x.size <= LARGEST_FIXTURE]
    arrays["knot_datasets"] = np.array(datasets)
    for name in datasets:
        arrays[f"knots__{name}__x"] = knot_inputs[name]
        for (df, intercept), reference in results["knots", name].items():
            for key, values in reference.items():
                if values is not None:
                    arrays[f"knots__{name}__df{df}_{int(intercept)}__{key}"] = values
    np.savez_compressed(FIXTURE_DIRECTORY / "natural_splines.npz", **arrays)
    arrays = {"harrell_datasets": np.array(list(harrell_inputs))}
    for name, x in harrell_inputs.items():
        arrays[f"harrell__{name}__x"] = x
        for nk, knots in results["harrell", name].items():
            if knots is not None:
                arrays[f"harrell__{name}__nk{nk}"] = knots
    arrays["restricted_cubic_cases"] = np.array([c["name"] for c in rcs_inputs])
    for case in rcs_inputs:
        arrays[f"rcs__{case['name']}__knots"] = case["knots"]
        arrays[f"rcs__{case['name']}__newx"] = case["newx"]
        arrays[f"rcs__{case['name']}__basis"] = results["rcs", case["name"]]
    arrays["rms_cases"] = np.array([c["name"] for c in rms_inputs])
    for case in rms_inputs:
        anova, knots = results["rms", case["name"]]
        for key, value in (
            ("x", case["x"]),
            ("y", case["y"]),
            ("nk", case["nk"]),
            ("anova", anova),
            ("knots", knots),
        ):
            arrays[f"rms__{case['name']}__{key}"] = np.asarray(value)
    np.savez_compressed(FIXTURE_DIRECTORY / "restricted_cubic_splines.npz", **arrays)
    print(
        "wrote lowess.npz, natural_splines.npz and restricted_cubic_splines.npz to "
        f"{FIXTURE_DIRECTORY}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    lowess_inputs, spline_inputs = lowess_cases(), spline_cases()
    knot_inputs, harrell_inputs = knot_datasets(), harrell_datasets()
    rcs_inputs, rms_inputs = restricted_cubic_cases(), rms_cases(harrell_inputs)
    inputs = (
        lowess_inputs,
        spline_inputs,
        knot_inputs,
        harrell_inputs,
        rcs_inputs,
        rms_inputs,
    )
    results, version = run_r(*inputs)
    print(f"Reference: {version}")
    table = Table()
    compare_lowess(lowess_inputs, results, table)
    compare_splines(spline_inputs, results, table)
    compare_spline_knots(spline_inputs, results, table)
    compare_knot_datasets(knot_inputs, results, table)
    compare_harrell(harrell_inputs, results, table)
    compare_restricted_cubic(rcs_inputs, results, table)
    compare_rms(rms_inputs, results, table)
    table.print()
    if arguments.write_fixtures:
        if table.failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(*inputs, results)
    print(f"{len(table.rows)} checks, {table.failures} failed")
    return 1 if table.failures else 0


if __name__ == "__main__":
    sys.exit(main())
