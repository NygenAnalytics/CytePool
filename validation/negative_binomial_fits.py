"""Check negative binomial fits and deviances against edgeR.

Compares ``cytepool.stats.negative_binomial_models`` with edgeR 4.4 on negative
binomial counts simulated from fixed seeds (2,000 genes, 6 to 16 samples), with genes
without counts and groups whose counts are all zero planted in most cases:
``fit_negative_binomial_model`` with ``glmFit``, ``negative_binomial_deviance`` with
``nbinomDeviance``, and ``negative_binomial_unit_deviance`` with ``nbinomUnitDeviance``
and with mpmath. The cases cover one-way layouts (two groups, group means with TMM
factors, an intercept alone, one coefficient per sample, sample weights, and
dispersions, offsets and weights per observation) and other designs (groups and a
covariate, a paired design, a covariate such as a date next to an intercept), common,
gene-wise and trended dispersions, dispersions, offsets and weights per observation,
sample weights, the Poisson model, sparse counts (about one read per gene and sample),
means near 1e8, a dispersion below 1e-4, and inputs both programs refuse.

edgeR fits a one-way layout group by group, to its last bits, so those fits are
compared with ``glmFit``. Its Levenberg fits of other designs stop short of the
maximum: by up to about 3e-4 in ``glmFit`` (1e-2 with dispersions and weights per
observation, 1.6 with a date), and by about 2e-7 in ``mglmLevenberg(tol = 1e-14)``,
but by 2e-3 with dispersions and weights per observation, where Fisher scoring, its
method, crawls, 2e-6 with sparse counts, 1e-5 with a date and for large counts, and
2e-3 for a dispersion of 2e-5, where the errors of its deviance hide the last steps.
Those fits are therefore compared with the maximum that Newton's method with the
observed information, written in R for this script, reaches from edgeR's fit (with a
date, whose design has a condition number near 1e13, the coefficients of the two agree
to about 2e-7 and the fitted values to 2e-10), and with ``mglmLevenberg(tol = 1e-14)``
(run on edgeR's ``addPriorCount`` output for the shrunk coefficients), whose distance
is reported for information where it does not converge. Their deviances are compared
with ``nbinomDeviance`` at the maximum: with dispersions and weights per observation,
``glmFit`` reports for 5 of 2,000 genes a deviance that is not that of its own fitted
values (found by probing), which the script counts.

Coefficients are compared relative to max(|reference|, 1), fitted values relative to
the reference, and deviances relative to the larger of the reference and the number
of samples: edgeR adds 1e-8 to every count and mean in its deviances (found by
probing), which lowers the unit deviance of a zero count by up to about 5e-7. A gene
whose maximum lies at infinity (fitted values that tend to zero) has coefficients that
depend on where each program stops; for those genes, the fitted values that tend to
zero must be below 1e-6 in both, and the other fitted values and the deviance agree
in the close tier.

The unit deviance is compared with mpmath (150 digits) over counts and means from
1e-30 to 1e30 and dispersions from 0 to 1e12, at a relative tolerance of 1e-13, and
with ``nbinomUnitDeviance`` where edgeR is accurate (dispersions of at least 1e-4,
dispersion times mean below 1e6, means and positive counts of at least 1e-3), relative
to max(d, 1). Outside those bounds edgeR approximates the deviance, which the script
reports, for counts and means from 1e-3 to 1e10, as deliberate differences.

Requirements:
    Python 3.13 with numpy, numba and mpmath, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor package edgeR. Set ``RSCRIPT`` to the Rscript
    executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/negative_binomial_fits.py
    PYTHONPATH=. python validation/negative_binomial_fits.py --write-fixtures
"""

import argparse
import itertools
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mpmath
import numpy as np

from cytepool.design import Design
from cytepool.normalization import tmm_factors
from cytepool.stats.negative_binomial_models import (
    fit_negative_binomial_model,
    negative_binomial_deviance,
    negative_binomial_unit_deviance,
)

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "negative_binomial_fits"
)
CLOSE = 1e-6
UNIT_DEVIANCE = 1e-13  # computed to about 1e-14
VANISHED = 1e-6  # fitted values that tend to zero are below this in both programs
NEWTON = "Newton's method in R from edgeR's fit"
TIGHT = "mglmLevenberg(tol = 1e-14)"
SCALED = "max scaled difference"
PER_SAMPLE = "max difference / max(D, n)"
RELATIVE = "max relative difference"
PRIOR = "mglmLevenberg(tol = 1e-14) after addPriorCount"
DEVIANCES = "glmFit, nbinomDeviance"
OPTIONAL = ("weights", "offsets", "library_sizes")

R_SCRIPT = r"""
suppressPackageStartupMessages(library(edgeR))
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  path <- file.path(dir, paste0(name, ".bin"))
  if (!file.exists(path)) return(NULL)
  readBin(path, "double", n = file.size(path) / 8, size = 8)
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")), size = 8)
as_matrix <- function(x, G, n) if (length(x) == G * n) matrix(x, G, n) else x
row_of <- function(x, g, G, n) {
  if (is.matrix(x)) x[g, ] else if (length(x) == G && G != n) rep(x[g], n) else rep_len(x, n)
}
# Newton's method with the observed information, which is positive definite for this
# model, from edgeR's fit near the maximum: it converges quadratically from there.
newton <- function(y, X, phi, off, w, b) {
  for (iteration in 1:60) {
    mu <- exp(drop(X %*% b) + off)
    score <- w * (y - mu) / (1 + phi * mu)
    information <- w * mu * (1 + phi * y) / (1 + phi * mu)^2
    step <- qr.solve(X * sqrt(information), score / sqrt(information))
    b <- b + step
    if (max(abs(step)) < 1e-13 * max(1, abs(b))) break
  }
  b
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  shape <- rd(dir, "shape"); n <- shape[1]; G <- shape[2]
  y <- matrix(rd(dir, "counts"), G, n)
  X <- matrix(rd(dir, "design"), n)
  disp <- as_matrix(rd(dir, "dispersion"), G, n)
  w <- as_matrix(rd(dir, "weights"), G, n)
  off <- as_matrix(rd(dir, "offsets"), G, n)
  if (is.null(off)) {
    lib <- rd(dir, "library_sizes")
    if (is.null(lib)) lib <- colSums(y)
    if (file.exists(file.path(dir, "tmm.txt"))) {
      nf <- calcNormFactors(DGEList(y, lib.size = lib))$samples$norm.factors
      wr(dir, "norm_factors", nf)
      lib <- lib * nf
    }
    off <- log(lib)
  }
  fit <- tryCatch(glmFit(y, X, dispersion = disp, offset = off, weights = w),
                  error = function(e) conditionMessage(e))
  if (is.character(fit)) { writeLines(fit, file.path(dir, "out_error.txt")); next }
  wr(dir, "seconds", system.time(glmFit(y, X, dispersion = disp, offset = off, weights = w))[["elapsed"]])
  writeLines(fit$method, file.path(dir, "out_method.txt"))
  wr(dir, "coefficients", fit$unshrunk.coefficients); wr(dir, "shrunk", fit$coefficients)
  wr(dir, "fitted", fit$fitted.values); wr(dir, "deviance", fit$deviance)
  wr(dir, "df", fit$df.residual)
  wr(dir, "nbinom_deviance", nbinomDeviance(y, fit$fitted.values, disp, weights = w))
  if (fit$method == "levenberg") {
    tight <- mglmLevenberg(y, X, dispersion = disp, offset = off, weights = w, tol = 1e-14, maxit = 1000)
    wr(dir, "tight_coefficients", tight$coefficients); wr(dir, "tight_fitted", tight$fitted.values)
    wr(dir, "tight_deviance", tight$deviance)
    prior <- addPriorCount(y, offset = off, prior.count = 0.125)
    shrunk <- mglmLevenberg(prior$y, X, dispersion = disp, offset = prior$offset, weights = w,
                            tol = 1e-14, maxit = 1000)
    wr(dir, "tight_shrunk", shrunk$coefficients)
    maximum <- matrix(NA_real_, G, ncol(X))
    interior <- which(rowSums(y) > 0 & apply(y > 0 | tight$fitted.values >= 1e-6, 1, all))
    weights <- if (is.null(w)) 1 else w
    for (g in interior) {
      maximum[g, ] <- newton(y[g, ], X, row_of(disp, g, G, n), row_of(off, g, G, n),
                             row_of(weights, g, G, n), tight$coefficients[g, ])
    }
    wr(dir, "newton", maximum)
    # The deviance at the maximum, and glmFit's deviances that are not those of its
    # fitted values.
    fitted <- exp(maximum %*% t(X) + if (is.matrix(off)) off else matrix(off, G, n, byrow = TRUE))
    rows <- function(x) if (is.matrix(x)) x[interior, , drop = FALSE] else if (length(x) == G) x[interior] else x
    at_maximum <- rep(NA_real_, G)
    at_maximum[interior] <- nbinomDeviance(y[interior, , drop = FALSE], fitted[interior, , drop = FALSE],
                                          rows(disp), weights = rows(w))
    wr(dir, "maximum_deviance", at_maximum)
    own <- nbinomDeviance(y, fit$fitted.values, disp, weights = w)
    wr(dir, "inconsistent", sum(abs(fit$deviance - own) > 1e-6 * pmax(own, n)))
  }
}
grid <- matrix(rd(root, "grid"), ncol = 3)
writeBin(as.double(nbinomUnitDeviance(grid[, 1], grid[, 2], grid[, 3])),
         file.path(root, "out_grid.bin"), size = 8)
writeLines(c(paste("R", getRversion()), paste("edgeR", packageVersion("edgeR"))),
           file.path(root, "versions.txt"))
"""


def simulate(rng, labels, n_genes, sizes, dispersion, covariate=None, empty=0.03):
    """Samples x genes counts: 10% of genes differ between labels and some follow the
    covariate; a fraction ``empty`` has no counts in one label, and 2% none at all."""
    levels, index = np.unique(labels, return_inverse=True)
    proportion = np.exp(rng.normal(0.0, 1.5, n_genes))
    proportion /= proportion.sum()
    changed = rng.random((levels.size, n_genes)) < 0.1
    effect = np.where(changed, rng.normal(0.0, 1.5, changed.shape), 0.0)
    log_mean = np.log(sizes[:, None] * proportion) + effect[index]
    if covariate is not None:
        log_mean += np.outer(covariate, rng.normal(0.0, 0.3, n_genes))
    mean = np.exp(log_mean)
    if np.all(dispersion == 0):
        counts = rng.poisson(mean).astype(float)
    else:
        size = 1.0 / dispersion
        counts = rng.negative_binomial(size, size / (size + mean)).astype(float)
    for g in np.flatnonzero(rng.random(n_genes) < empty):
        counts[index == rng.integers(levels.size), g] = 0.0
    counts[:, rng.random(n_genes) < 0.02] = 0.0
    return counts


def make_cases(rng, n_genes):
    """Cases by name: a sample table, a formula, counts and the other inputs."""
    cases = {}

    def add(name, samples, formula, y, dispersion, **inputs):
        cases[name] = {"samples": samples, "formula": formula, "counts": y}
        cases[name] |= {"dispersion": dispersion} | inputs

    def draw(labels, dispersion, covariate=None, scale=1e6, **options):
        sizes = rng.uniform(0.5 * scale, 2.0 * scale, len(labels))
        return simulate(rng, labels, n_genes, sizes, dispersion, covariate, **options)

    def trended(y):  # falls with the average count, as trended estimates do
        return 0.05 + 1.0 / np.sqrt(1.0 + y.mean(axis=0))

    two = {"group": np.repeat(["A", "B"], 3)}
    three = {"group": np.repeat(["A", "B", "C"], 4)}
    aged = three | {"age": rng.uniform(20.0, 70.0, 12).round(1)}
    age, formula = (aged["age"] - 45.0) / 15.0, "~ group + age"
    add("two_groups", two, "~ group", draw(two["group"], 0.1), 0.1)
    genewise = np.exp(rng.normal(np.log(0.08), 0.6, n_genes))
    y = draw(three["group"], genewise)
    add("group_means_tmm", three, "~ 0 + group", y, genewise, tmm=True)
    one = {"group": np.repeat(["A"], 6)}
    y = draw(one["group"], 0.1)
    add("intercept_only", one, "~ 1", y, trended(y))
    each = {"sample": np.array([f"S{i}" for i in range(6)])}
    add(
        "one_coefficient_per_sample",
        each,
        "~ 0 + sample",
        draw(each["sample"], 0.1),
        0.1,
    )
    weights = rng.uniform(0.5, 2.0, 6)
    add("sample_weights", two, "~ group", draw(two["group"], 0.1), 0.1, weights=weights)
    y = draw(three["group"], 0.1, age)
    lib = y.sum(axis=1) * rng.uniform(0.8, 1.25, 12)
    add("groups_covariate", aged, formula, y, trended(y), library_sizes=lib)
    paired = {"donor": np.repeat([f"D{i}" for i in range(1, 9)], 2)}
    paired["condition"] = np.tile(["ctrl", "stim"], 8)
    y = draw(paired["condition"], 0.05)
    add("paired", paired, "~ donor + condition", y, 0.05, tmm=True)
    y = draw(three["group"], 0.1, age)
    inputs = {"weights": rng.uniform(0.5, 2.0, y.shape)}
    inputs["offsets"] = np.log(rng.uniform(5e5, 2e6, 12))[:, None] + rng.normal(
        0, 0.2, y.shape
    )
    # Fisher scoring, edgeR's method, crawls on some of these genes.
    inputs["edger_converges"] = False
    add("per_observation", aged, formula, y, rng.uniform(0.03, 0.3, y.shape), **inputs)
    weights = rng.uniform(0.5, 2.0, 12)
    y = draw(three["group"], 0.1, age)
    add("covariate_sample_weights", aged, formula, y, 0.1, weights=weights)
    add("poisson", aged, formula, draw(three["group"], np.zeros(1), age), 0.0)
    # About one read per gene and sample: many genes have one to three counts, and
    # the fitted values of the others tend to zero at different rates. edgeR stops
    # up to about 2e-6 from the maximum where fitted values are small.
    y = draw(three["group"], 0.1, age, scale=float(n_genes))
    add("sparse_counts", aged, formula, y, 0.1, edger_converges=False)
    # A covariate far from zero next to the intercept, such as a date: nearly
    # collinear columns.
    dated = three | {"day": 20240101.0 + rng.integers(0, 300, 12)}
    y = draw(three["group"], 0.1, (dated["day"] - dated["day"].mean()) / 90)
    add("date_covariate", dated, "~ group + day", y, 0.1, edger_converges=False)
    # Means up to about 1e8, with dispersion times mean below 1e6, where edgeR's
    # deviance is accurate.
    y = draw(three["group"], 0.005, age, scale=1e11, empty=0.0)
    add("large_counts", aged, formula, y, 0.005, edger_converges=False)
    y = draw(three["group"], 2e-5, age)
    inputs = {"edger_converges": False, "edger_deviance": False}
    add("dispersion_below_1e-4", aged, formula, y, 2e-5, **inputs)
    # A one-way layout with dispersions, offsets and weights per observation.
    y = draw(three["group"], 0.1)
    inputs = {"weights": rng.uniform(0.5, 2.0, y.shape)}
    inputs["offsets"] = np.log(rng.uniform(5e5, 2e6, 12))[:, None] + rng.normal(
        0, 0.2, y.shape
    )
    phi = rng.uniform(0.03, 0.3, y.shape)
    add("groups_per_observation", three, "~ group", y, phi, **inputs)
    return cases


def refused_cases(rng):
    """Inputs that edgeR and CytePool both refuse, with part of CytePool's message."""
    two = {"group": np.repeat(["A", "B"], 3), "batch": np.repeat(["x", "y"], 3)}
    y = simulate(rng, two["group"], 50, rng.uniform(5e5, 2e6, 6), 0.1, empty=0.0)
    zero = {"weights": np.r_[0.0, np.ones(5)]}
    cases = {
        "aliased column": ("~ group + batch", 0.1, {}, "not of full column rank"),
        "zero weight": ("~ group", 0.1, zero, "positive"),
        "negative dispersion": ("~ group", -0.1, {}, "non-negative"),
    }
    return {
        f"refused: {name}": {"samples": two, "formula": formula, "counts": y}
        | {"dispersion": dispersion, "refused": message}
        | inputs
        for name, (formula, dispersion, inputs, message) in cases.items()
    }


def deviance_grid(rng):
    """(count, mean, dispersion) rows over every regime of the unit deviance."""
    means = np.r_[0.0, np.geomspace(1e-30, 1e30, 13), rng.uniform(0.1, 1e3, 4)]
    factors = [0.0, 1e-12, 1e-6, 0.01, 0.1249, 0.1251, 0.5, 0.9, 2.0, 1e3]
    dispersions = np.r_[0.0, np.geomspace(1e-14, 1e12, 14), 9.9e-5, 1e-4]
    rows = set()
    for mean, factor, phi in itertools.product(means, factors, dispersions):
        low = mean * (1 - factor) if factor < 1 else mean / factor
        for count in {mean * (1 + factor), low, 0.0, 3.0}:
            # 150 digits hold y + 1 / phi and the cancellations that follow when
            # they differ by at most 40 decades.
            if phi == 0 or abs(np.log10(max(count, mean, 1e-300) * phi)) <= 40:
                rows.add((float(count), float(mean), float(phi)))
    return np.array(sorted(rows))


def exact_unit_deviance(y, mu, phi):
    y, mu, phi = mpmath.mpf(y), mpmath.mpf(mu), mpmath.mpf(phi)
    if mu == 0:
        return mpmath.inf if y > 0 else mpmath.mpf(0)
    first = y * mpmath.log(y / mu) if y > 0 else mpmath.mpf(0)
    if phi == 0:
        return 2 * (first - (y - mu))
    return 2 * (first - (y + 1 / phi) * mpmath.log((1 + phi * y) / (1 + phi * mu)))


def run_edger(cases, grid):
    """edgeR's outputs for each case and for the grid, and the versions."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, case in cases.items():
            directory = root / names[name]
            directory.mkdir()
            arrays = {"shape": case["counts"].shape, "counts": case["counts"]}
            arrays["design"] = case["design"].ravel(order="F")
            arrays["dispersion"] = np.atleast_1d(case["dispersion"])
            arrays |= {k: case[k] for k in OPTIONAL if k in case}
            for key, value in arrays.items():
                data = np.asarray(value, float).tobytes()
                (directory / f"{key}.bin").write_bytes(data)
            if case.get("tmm"):
                (directory / "tmm.txt").write_text("1\n")
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "grid.bin").write_bytes(grid.tobytes(order="F"))
        (root / "compare.R").write_text(R_SCRIPT)
        command = [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "compare.R", "."]
        subprocess.run(command, check=True, cwd=root)
        for name, case in cases.items():
            directory = root / names[name]
            outputs = directory.glob("out_*.bin")
            case["edgeR"] = {path.stem[4:]: np.fromfile(path) for path in outputs}
            for kind in ("method", "error"):
                path = directory / f"out_{kind}.txt"
                text = path.read_text().strip() if path.exists() else None
                case[f"edgeR {kind}"] = text
        values = np.fromfile(root / "out_grid.bin")
        return values, (root / "versions.txt").read_text().split("\n")[:2]


def inputs(case):
    """The keyword arguments of CytePool's fit of a case, as edgeR got them."""
    keywords = {k: case[k] for k in OPTIONAL if k in case}
    if case.get("tmm"):
        lib = case.get("library_sizes")
        keywords["norm_factors"] = tmm_factors(case["counts"], library_sizes=lib)
    return keywords | {"dispersion": case["dispersion"]}


def fit_case(case):
    design = Design(case["samples"], case["formula"])
    return fit_negative_binomial_model(case["counts"], design, **inputs(case))


def scaled(ours, reference, floor):
    """Largest |ours - reference| / max(|reference|, floor) over finite references."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    keep = np.isfinite(reference)
    if not keep.any():
        return 0.0
    scale = np.maximum(np.abs(reference[keep]), floor)
    return float(np.max(np.abs(ours[keep] - reference[keep]) / scale))


def row(table, check, reference, metric, value, tolerance, result=None):
    """A row; without a tolerance, the result is "information" unless given."""
    if result is None and tolerance is None:
        result = "information"
    elif result is None:
        result = "pass" if value <= tolerance else "FAIL"
    shown = "n/a" if tolerance is None else f"{tolerance:.0e}"
    table.append((check, reference, metric, f"{value:.1e}", shown, result))


def compare_refusal(name, case, table):
    try:
        fit_case(case)
        ours = "values"
    except ValueError as error:
        ours = "error" if case["refused"] in str(error) else f"other error: {error}"
    edger = "error" if case["edgeR error"] else "values"
    result = "pass" if (ours, edger) == ("error", "error") else "FAIL"
    value = f"edgeR: {edger}; CytePool: {ours}"
    table.append((name, "glmFit", "behaviour", value, "n/a", result))


def compare_case(name, case, table):
    fit, edger, counts = fit_case(case), case["edgeR"], case["counts"]
    n, n_genes = counts.shape
    out = {k: v.reshape(-1, n_genes).T for k, v in edger.items() if v.size >= n_genes}
    levenberg = case["edgeR method"] == "levenberg"
    empty = ~counts.any(axis=0)
    reference = out["tight_fitted" if levenberg else "fitted"].T
    # Genes whose maximum is at infinity: their fitted values tend to zero.
    vanishing = (counts == 0) & (reference < VANISHED) & levenberg
    boundary = vanishing.any(axis=0) & ~empty
    interior = ~boundary & ~(empty & levenberg)

    def both(coefficients, fitted):  # coefficients and fitted values, scaled
        mine = fit.coefficients[interior], fit.fitted_values[:, interior]
        first = scaled(mine[0], coefficients[interior], 1.0)
        return max(first, scaled(mine[1], fitted.T[:, interior], 1e-300))

    converges = CLOSE if case.get("edger_converges", True) else None
    accurate = case.get("edger_deviance", True)
    weights = {"weights": case["weights"]} if "weights" in case else {}
    phi = case["dispersion"]
    at_edger = negative_binomial_deviance(counts, out["fitted"].T, phi, **weights)
    at_edger = scaled(at_edger, edger["nbinom_deviance"], n)
    mismatches = np.count_nonzero(fit.df_residual != edger["df"])
    mismatches += np.count_nonzero(~fit.converged)
    if levenberg:
        with np.errstate(invalid="ignore"):  # NaN for genes without a maximum
            maximum = np.exp(fit.design @ out["newton"].T + fit.offsets).T
        newton = both(out["newton"], maximum)
        tight = both(out["tight_coefficients"], out["tight_fitted"])
        shrunk = scaled(fit.shrunk_coefficients, out["tight_shrunk"], 1.0)
        loose = scaled(out["coefficients"][interior], fit.coefficients[interior], 1.0)
        rows = [
            ("coefficients and fitted values", NEWTON, SCALED, newton, CLOSE),
            ("coefficients and fitted values", TIGHT, SCALED, tight, converges),
            ("shrunk coefficients", PRIOR, SCALED, shrunk, converges),
            ("glmFit's own fits", "glmFit", SCALED, loose, None),
        ]
    else:
        coefficients = both(out["coefficients"], out["fitted"])
        shrunk = scaled(fit.shrunk_coefficients, out["shrunk"], 1.0)
        rows = [
            ("coefficients and fitted values", "glmFit", SCALED, coefficients, CLOSE),
            ("shrunk coefficients", "glmFit", SCALED, shrunk, CLOSE),
        ]
    deviation = None if accurate else "deviation"
    tolerance = CLOSE if accurate else None
    if levenberg:
        # glmFit's deviance can differ from that of its own fitted values, so the
        # deviance is compared with nbinomDeviance at the maximum.
        deviance = max(scaled(fit.deviance, edger["maximum_deviance"], n), at_edger)
        check = "deviance at the maximum, and at edgeR's fitted values"
        rows.append(
            (check, "nbinomDeviance", PER_SAMPLE, deviance, tolerance, deviation)
        )
        inconsistent = float(edger["inconsistent"][0])
        check = "glmFit deviances that are not those of its fitted values"
        rows.append((check, "glmFit, nbinomDeviance", "genes", inconsistent, None))
    else:
        deviance = max(scaled(fit.deviance, edger["deviance"], n), at_edger)
        check = "deviance, and at edgeR's fitted values"
        rows.append((check, DEVIANCES, PER_SAMPLE, deviance, tolerance, deviation))
    check = "residual df, and fits that did not converge"
    rows.append((check, "glmFit", "count", mismatches, 0))
    if boundary.any():
        k, kept = int(boundary.sum()), ~vanishing[:, boundary]
        largest = float(np.max(fit.fitted_values[vanishing]))
        ours, theirs = fit.fitted_values[:, boundary], reference[:, boundary]
        tight = scaled(fit.deviance[boundary], edger["tight_deviance"][boundary], n)
        others = max(scaled(ours[kept], theirs[kept], 1e-300), tight)
        check = f"{k} genes at infinity, fitted values that tend to zero"
        rows.append((check, TIGHT, "largest", largest, VANISHED))
        check = f"{k} genes at infinity, the others and deviance"
        rows.append((check, TIGHT, SCALED, others, tolerance, deviation))
    method = "Levenberg" if levenberg else "one-way"
    for check, reference, metric, value, *rest in rows:
        row(table, f"{name} ({method}): {check}", reference, metric, value, *rest)
    if levenberg and empty.any():
        # The Levenberg cases have an intercept in the first column.
        edger_na = np.isnan(out["coefficients"][empty]).all()
        expected = np.r_[-1e8, np.zeros(fit.coefficients.shape[1] - 1)]
        constant = np.all(fit.coefficients[empty] == expected)
        zero = not fit.fitted_values[:, empty].any() and not fit.deviance[empty].any()
        result = "deviation" if edger_na and constant and zero else "FAIL"
        value = f"{int(empty.sum())} genes; edgeR: NA; CytePool: -1e8, then 0"
        check = f"{name}: coefficients of genes without counts"
        table.append((check, "glmFit", "values", value, "n/a", result))


def compare_grid(grid, edger_values, table):
    ours = negative_binomial_unit_deviance(grid[:, 0], grid[:, 1], grid[:, 2])
    mpmath.mp.dps = 150
    exact = np.array([float(exact_unit_deviance(*point)) for point in grid])
    finite = np.isfinite(exact) & (exact > 0)
    identical = np.all(ours[~finite] == exact[~finite])  # zero and infinite values
    value = scaled(ours[finite], exact[finite], 0.0) if identical else np.inf
    check = f"unit deviance at {len(grid)} points, {(~finite).sum()} zero or infinite"
    row(table, check, "mpmath, 150 digits", RELATIVE, value, UNIT_DEVIANCE)
    y, mu, phi = grid.T
    usual = (mu >= 1e-3) & ((y == 0) | (y >= 1e-3))
    accurate = usual & (phi >= 1e-4) & (phi * mu < 1e6)
    value = scaled(ours[accurate], edger_values[accurate], 1.0)
    check = f"unit deviance where edgeR is accurate ({accurate.sum()} points)"
    row(table, check, "nbinomUnitDeviance", "max difference / max(d, 1)", value, CLOSE)
    usual &= (mu <= 1e10) & (y <= 1e10)
    small = ("dispersion below 1e-4", (phi > 0) & (phi < 1e-4))
    for label, where in (small, ("dispersion times mean from 1e6", phi * mu >= 1e6)):
        where = where & usual & finite
        value = scaled(edger_values[where], exact[where], 1.0)
        check = f"unit deviance, {label} ({where.sum()} points)"
        metric = "max difference / max(d, 1)"
        row(table, check, "edgeR against mpmath", metric, value, None, "deviation")
    return exact


def write_fixtures(cases, grid, exact):
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        arrays = {"counts": case["counts"], "formula": np.array(case["formula"])}
        arrays |= {f"samples {k}": np.asarray(v) for k, v in case["samples"].items()}
        arrays |= {k: np.asarray(v, float) for k, v in inputs(case).items()}
        arrays["method"] = np.array(case["edgeR method"])
        arrays |= {f"edgeR {k}": v for k, v in case["edgeR"].items() if k != "seconds"}
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    sample = np.random.default_rng(5).choice(len(grid), 400, replace=False)
    path = FIXTURE_DIRECTORY / "unit_deviance.npz"
    np.savez_compressed(path, points=grid[sample], mpmath=exact[sample])
    print(f"wrote {len(cases) + 1} fixtures to {FIXTURE_DIRECTORY}")


def timing_rows(cases, table):
    """Seconds of a fit, CytePool's best of three after a first one, and glmFit's."""
    for name in ("group_means_tmm", "paired"):
        case = cases[name]
        keywords = inputs(case)
        seconds = []
        for _ in range(4):
            start = time.perf_counter()
            fit_negative_binomial_model(case["counts"], case["design"], **keywords)
            seconds.append(time.perf_counter() - start)
        value = f"{min(seconds[1:]):.3f} and {float(case['edgeR']['seconds'][0]):.3f}"
        check = f"{name}: seconds for 2,000 genes"
        table.append((check, "glmFit", "CytePool, edgeR", value, "n/a", "information"))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261025), 2000)
    cases |= refused_cases(np.random.default_rng(3))
    if arguments.write_fixtures:
        small = make_cases(np.random.default_rng(7), 40)
        cases |= {f"{name}_fixture": case for name, case in small.items()}
    for case in cases.values():
        case["design"] = Design(case["samples"], case["formula"]).matrix
    grid = deviance_grid(np.random.default_rng(11))
    edger_values, versions = run_edger(cases, grid)
    print("Reference: " + ", ".join(versions) + f"; mpmath {mpmath.__version__}")
    table = []
    for name, case in cases.items():
        (compare_refusal if "refused" in case else compare_case)(name, case, table)
    exact = compare_grid(grid, edger_values, table)
    timing_rows(cases, table)
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for entry in table:
        print("| " + " | ".join(entry) + " |")
    failures = sum(entry[-1] == "FAIL" for entry in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        small = {n[:-8]: c for n, c in cases.items() if n.endswith("_fixture")}
        write_fixtures(small, grid, exact)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
