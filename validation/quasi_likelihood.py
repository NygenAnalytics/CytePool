"""Check edgeR's quasi-likelihood pipeline against edgeR 4.4 and limma 3.62.

Compares ``cytepool.stats.quasi_likelihood`` with ``glmQLFit`` and ``glmQLFTest`` of
edgeR 4.4 (``legacy = FALSE``, the default), and with limma's ``squeezeVar``, on
negative binomial counts simulated from fixed seeds, with a dispersion per gene, in 14
cases of 1,000 genes and 8 or 9 samples: two groups with TMM factors, three groups
with a covariate and several contrasts, a paired design, robust priors with and
without genes that have no counts, small counts with groups whose counts are all zero,
a dispersion given below the true one (so that the average quasi-dispersion is well
above 1), prior weights, trended dispersions, no trend, a trend in another covariate,
offsets per observation, Poisson counts, and a test that both programs refuse.

Every intermediate that edgeR exposes is compared: the dispersion and the share of
genes it comes from, the average quasi-dispersion, the adjusted deviances and residual
degrees of freedom, the prior and posterior quasi-dispersions and prior degrees of
freedom, and the test columns (logFC, logCPM, F, PValue, FDR and df.total). Three
differences set the tolerances, and the script isolates each:

- edgeR approximates the mean and variance of the unit deviance (the rows "edgeR's
  chi-squared match": its scale s is off by up to 1.2e-3 and its df k by up to 2e-4
  at dispersions up to 0.2, both more at 1), where CytePool's moments are exact (the
  rows against mpmath at 30 digits and the gamma limit). The adjusted deviances, the
  quasi-dispersions and the F-statistics therefore agree to a few 1e-3, the average
  quasi-dispersion to about 2e-3, and log10 p-values to about 4e-3 of their size.
  The prior and posterior that ``squeeze_variances`` computes from edgeR's own
  adjusted deviances agree with edgeR's in the close tier, which leaves the moments
  as the only cause.
- edgeR's search for the Cox-Reid dispersion stops within its tolerance of 1e-5;
  ``dispCoxReid(tol = 1e-12)`` on the same genes agrees with CytePool to about 1e-6.
- edgeR's Levenberg fits stop short of the maximum (see
  ``validation/negative_binomial_fits.py``): log fold changes of designs that are not
  one-way layouts agree to about 2e-4, those of one-way layouts in the close tier.

Two further differences are reported as deliberate: the robust prior with genes that
have no residual degrees of freedom (limma counts them as zero variances), and the log
CPM of the first gene with offsets per observation (edgeR's ``aveLogCPM`` ignores that
gene's offsets; the other genes are compared).

R is run with its address space capped at 8 GB.

Requirements:
    Python 3.13 with numpy, numba and mpmath, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages edgeR and limma. Set ``RSCRIPT`` to
    the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/quasi_likelihood.py
    PYTHONPATH=. python validation/quasi_likelihood.py --write-fixtures
"""

import argparse
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mpmath
import numpy as np

from cytepool.normalization import tmm_factors
from cytepool.stats.empirical_bayes import squeeze_variances
from cytepool.stats.quasi_likelihood import (
    average_log_cpm,
    common_dispersion,
    fit_quasi_likelihood,
    negative_binomial_unit_deviance_moments,
    quasi_likelihood_f_test,
)

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "quasi_likelihood"
)
MEMORY = 8_000_000_000  # bytes of address space for R
CLOSE = 1e-6
# edgeR's moments are off by up to about 1.2e-3 in the scale and 2e-4 in the df at
# dispersions up to 0.2, and by a few 1e-3 at 1; a quasi-dispersion combines both.
MOMENTS = 5e-3
# The average quasi-dispersion compounds them over two passes.
AVERAGE = 1e-2
# The prior df maximizes a likelihood that is flat at large values: 1e-3 changes of
# the quasi-dispersions move it by up to about 7e-3.
PRIOR_DF = 2e-2
PROBABILITY = 5e-3  # of max(1, |log10 p|)
SEARCH = 1e-4  # edgeR's search for the dispersion stops within 1e-5
SEARCH_LOG_CPM = 1e-5  # the log CPM at dispersions 1e-5 apart
TIGHT = 1e-6
# edgeR's Levenberg fits stop up to about 4e-4 short of the maximum, which moves its
# likelihood's maximum, and its log fold changes, by about that.
LEVENBERG = 1e-3
LEVENBERG_SEARCH = 1e-5
FLOOR = 1e-2  # adjusted residual df below limma's 0.01 never enter the prior
EXACT_MOMENTS = 1e-11
OPTIONAL = ("library_sizes", "offsets", "weights", "dispersion", "covariate")
FIT, TEST = "glmQLFit", "glmQLFTest"
RELATIVE, MAX_RELATIVE = "relative difference", "max relative difference"
SCALED = "max scaled difference"
DIRECT_WEIGHTS = np.random.default_rng(5).uniform(0.5, 2.0, (8, 1000))

R_SCRIPT = r"""
suppressPackageStartupMessages(library(edgeR))
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  path <- file.path(dir, paste0(name, ".bin"))
  if (!file.exists(path)) return(NULL)
  readBin(path, "double", n = file.size(path) / 8, size = 8)
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")), size = 8)
by_gene <- function(x, G, n) if (!is.null(x) && length(x) == G * n) matrix(x, G, n) else x
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  shape <- rd(dir, "shape"); n <- shape[1]; G <- shape[2]
  y <- matrix(rd(dir, "counts"), G, n)
  X <- matrix(rd(dir, "design"), n)
  lib <- rd(dir, "library_sizes"); w <- by_gene(rd(dir, "weights"), G, n)
  off <- by_gene(rd(dir, "offsets"), G, n); disp <- rd(dir, "dispersion")
  args <- list(y, X, robust = file.exists(file.path(dir, "robust.txt")), keep.unit.mat = TRUE)
  if (!is.null(lib)) args$lib.size <- lib
  if (!is.null(off)) args$offset <- off
  if (!is.null(w)) args$weights <- w
  if (!is.null(disp)) args$dispersion <- disp
  if (!is.null(rd(dir, "covariate"))) args$covariate.trend <- rd(dir, "covariate")
  if (file.exists(file.path(dir, "no_trend.txt"))) args$abundance.trend <- FALSE
  fit <- tryCatch(do.call(glmQLFit, args), error = function(e) conditionMessage(e))
  if (is.character(fit)) { writeLines(fit, file.path(dir, "out_error.txt")); next }
  for (key in c("dispersion", "average.ql.dispersion", "deviance.adj", "df.residual.adj",
                "s2.prior", "df.prior", "s2.post", "fitted.values", "leverage"))
    wr(dir, key, fit[[key]])
  if (!is.null(fit$top.proportion)) wr(dir, "top.proportion", fit$top.proportion)
  # The abundance of the fit's trends: aveLogCPM at the given dispersion, or at 0.05.
  ave <- list(y, dispersion = if (is.null(disp)) 0.05 else disp)
  if (!is.null(off)) ave$offset <- off else if (!is.null(lib)) ave$lib.size <- lib
  if (!is.null(w)) ave$weights <- w
  abundance <- do.call(aveLogCPM, ave)
  wr(dir, "abundance", abundance)
  if (is.null(disp)) {
    top <- order(abundance, decreasing = TRUE)[seq_len(ceiling(fit$top.proportion * G))]
    o <- if (!is.null(off)) off[top, , drop = FALSE] else log(if (is.null(lib)) colSums(y) else lib)
    wt <- if (is.null(w)) NULL else w[top, , drop = FALSE]
    wr(dir, "tight_dispersion", dispCoxReid(y[top, , drop = FALSE], X, offset = o, weights = wt, tol = 1e-12))
  }
  tests <- readLines(file.path(dir, "tests.txt"))
  for (i in seq_along(tests)) {
    words <- strsplit(tests[i], " ")[[1]]
    res <- tryCatch(
      if (words[1] == "coef") glmQLFTest(fit, coef = as.integer(words[-1]))
      else glmQLFTest(fit, contrast = matrix(rd(dir, words[2]), ncol(X))),
      error = function(e) conditionMessage(e))
    if (is.character(res)) { writeLines(res, file.path(dir, paste0("out_test", i, "_error.txt"))); next }
    tab <- res$table
    wr(dir, paste0("test", i, "_logFC"), as.matrix(tab[, grep("^logFC", names(tab)), drop = FALSE]))
    for (key in c("logCPM", "F", "PValue")) wr(dir, paste0("test", i, "_", key), tab[[key]])
    wr(dir, paste0("test", i, "_FDR"), topTags(res, n = Inf, sort.by = "none")$table$FDR)
    wr(dir, paste0("test", i, "_df.total"), res$df.total)
  }
}
# The common dispersion and the average log CPM on their own, on the first case.
dir <- file.path(root, "direct")
y <- matrix(rd(dir, "counts"), rd(dir, "shape")[2]); X <- matrix(rd(dir, "design"), ncol(y))
lib <- rd(dir, "library_sizes"); w <- matrix(rd(dir, "weights"), nrow(y))
wr(dir, "common", c(estimateGLMCommonDisp(y, X, offset = log(lib)),
                    dispCoxReid(y, X, offset = log(lib), tol = 1e-12),
                    estimateGLMCommonDisp(y, X, offset = log(lib), weights = w),
                    dispCoxReid(y, X, offset = log(lib), weights = w, tol = 1e-12),
                    estimateGLMCommonDisp(y, X, offset = log(lib), subset = 100),
                    estimateGLMCommonDisp(y, X, offset = log(lib), weights = w, subset = 100)))
wr(dir, "abundance", cbind(aveLogCPM(y, lib.size = lib),
                           aveLogCPM(y, lib.size = lib, prior.count = 0.5, dispersion = 0.3),
                           aveLogCPM(y, lib.size = lib, weights = w, dispersion = 0),
                           aveLogCPM(y + 0.5, offset = log(lib), prior.count = 0)))
# edgeR's chi-squared match, implied by its unit matrices: unit.df.adj = k (1 - h) and
# unit.deviance.adj = d / s, with d edgeR's unit deviance at the working dispersion.
moments <- NULL
for (phi in c(0.01, 0.2, 1)) {
  means <- 10^seq(-3, 5, by = 0.25)
  y <- t(sapply(means, function(m) m * c(0.5, 1.5, 0.8, 1.2)))
  fit <- glmQLFit(y, matrix(1, 4, 1), dispersion = phi, lib.size = rep(1e9, 4),
                  prior.count = 0, keep.unit.mat = TRUE)
  a <- fit$average.ql.dispersion
  d <- nbinomUnitDeviance(y, fit$fitted.values, phi / a)
  moments <- rbind(moments, cbind(phi, a, fit$fitted.values[, 1],
                                  fit$unit.df.adj[, 1] / (1 - fit$leverage[, 1]),
                                  d[, 2] / fit$unit.deviance.adj[, 2]))
}
writeBin(as.double(moments), file.path(root, "out_moments.bin"))
writeLines(c(paste("R", getRversion()), paste("edgeR", packageVersion("edgeR")),
             paste("limma", packageVersion("limma"))), file.path(root, "versions.txt"))
"""


def simulate(rng, labels, n_genes, sizes, dispersion, covariate=None, empty=0.03):
    """Samples x genes counts: 10% of genes differ between labels and some follow the
    covariate; a fraction ``empty`` has no counts in one label, and 2% none at all.
    Each gene's dispersion is ``dispersion`` times a log-normal factor (sd 0.4 on the
    log scale), so that the quasi-dispersions vary between genes and the prior df is
    finite, as in real data."""
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
        size = 1.0 / (dispersion * np.exp(rng.normal(0.0, 0.4, n_genes)))
        counts = rng.negative_binomial(size, size / (size + mean)).astype(float)
    for g in np.flatnonzero(rng.random(n_genes) < empty):
        counts[index == rng.integers(levels.size), g] = 0.0
    counts[:, rng.random(n_genes) < 0.02 * (empty > 0)] = 0.0
    return counts


def make_cases(rng, n_genes):
    """Cases by name: counts, design, the inputs both programs get, and the tests."""
    cases = {}

    def add(name, counts, design, tests, **inputs):
        cases[name] = {"counts": counts, "design": design, "tests": tests} | inputs

    def sizes(n, scale=2e6):
        return rng.uniform(0.5 * scale, 2.0 * scale, n)

    two = np.repeat([0.0, 1.0], 4)
    x2 = np.column_stack([np.ones(8), two])
    y = simulate(rng, two, n_genes, sizes(8), 0.15)
    lib = y.sum(axis=1) * tmm_factors(y)
    add("two_groups", y, x2, ["coef 2"], library_sizes=lib)
    group = np.repeat([0, 1, 2], 3)
    age = rng.uniform(-1.0, 1.0, 9)
    x3 = np.column_stack([np.ones(9), group == 1, group == 2, age]).astype(float)
    y = simulate(rng, group, n_genes, sizes(9), 0.1, covariate=age)
    contrasts = {"b_vs_a": [0, 1, 0, 0], "c_vs_b": [0, -1, 1, 0]}
    add(
        "groups_covariate",
        y,
        x3,
        ["coef 4", "b_vs_a", "c_vs_b", "joint"],
        contrasts=contrasts,
    )
    donor = np.repeat(np.arange(4), 2)
    condition = np.tile([0.0, 1.0], 4)
    xp = np.column_stack([np.ones(8)] + [donor == d for d in range(1, 4)] + [condition])
    y = simulate(rng, condition, n_genes, sizes(8), 0.05)
    add("paired", y, xp.astype(float), ["coef 5"])
    y = simulate(rng, two, n_genes, sizes(8), 0.15)
    y[:, :20] *= np.array([1, 1, 1, 6, 1, 1, 1, 1.0])[:, None]
    add("robust_with_empty_genes", y, x2, ["coef 2"], robust=True)
    add("robust", y[:, y.any(axis=0)], x2, ["coef 2"], robust=True)
    y = simulate(rng, two, n_genes, sizes(8, 1e3), 0.3, empty=0.1)
    add("small_counts", y, x2, ["coef 2"])
    y = simulate(rng, two, n_genes, sizes(8), 0.15)
    add("dispersion_given_low", y, x2, ["coef 2"], dispersion=0.03)
    add("prior_weights", y, x2, ["coef 2"], weights=rng.uniform(0.5, 2.0, y.shape))
    trended = 0.05 + 1.0 / np.sqrt(1.0 + y.mean(axis=0))
    add("trended_dispersion", y, x2, ["coef 2"], dispersion=trended)
    add("no_trend", y, x2, ["coef 2"], trend=False)
    add("covariate_trend", y, x2, ["coef 2"], covariate=rng.normal(size=n_genes))
    offsets = np.log(sizes(8))[:, None] + rng.normal(0.0, 0.1, y.shape)
    first_empty = y.copy()
    first_empty[:, 0] = 0.0
    # edgeR's aveLogCPM ignores the offsets of the first gene (minus infinity for a
    # gene without counts): with a dispersion given, no counts and no trend, that
    # value ranks nothing and enters no trend.
    add(
        "offsets_per_observation",
        first_empty,
        x2,
        ["coef 2"],
        offsets=offsets,
        dispersion=0.15,
        trend=False,
    )
    y = simulate(rng, two, n_genes, sizes(8), np.zeros(1), empty=0.0)
    add("poisson", y, x2, ["coef 2"], dispersion=0.0)
    # Both refuse a test that leaves the null model without a coefficient.
    add("intercept_only", y[:4], np.ones((4, 1)), ["coef 1"], dispersion=0.1)
    return cases


def run_edger(cases):
    """edgeR's outputs for every case and its chi-squared match, and the versions."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, case in cases.items():
            directory = root / names[name]
            directory.mkdir()
            arrays = {"shape": case["counts"].shape, "counts": case["counts"]}
            arrays["design"] = case["design"].ravel(order="F")
            arrays |= {k: np.atleast_1d(case[k]) for k in OPTIONAL if k in case}
            lines = []
            for test in case["tests"]:
                if test.startswith("coef"):
                    lines.append(test)
                    continue
                matrix = contrast_matrix(case, test)
                arrays[test] = matrix.ravel(order="F")
                lines.append(f"contrast {test}")
            for key, value in arrays.items():
                (directory / f"{key}.bin").write_bytes(
                    np.asarray(value, float).tobytes()
                )
            (directory / "tests.txt").write_text("\n".join(lines) + "\n")
            for flag, present in (
                ("robust", case.get("robust")),
                ("no_trend", case.get("trend") is False),
            ):
                if present:
                    (directory / f"{flag}.txt").write_text("1\n")
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        direct = root / "direct"
        direct.mkdir()
        first = cases[next(iter(cases))]
        arrays = {"shape": first["counts"].shape, "counts": first["counts"]}
        arrays |= {"design": first["design"].ravel(order="F")}
        arrays |= {"library_sizes": first["library_sizes"], "weights": DIRECT_WEIGHTS}
        for key, value in arrays.items():
            (direct / f"{key}.bin").write_bytes(np.asarray(value, float).tobytes())
        (root / "compare.R").write_text(R_SCRIPT)
        command = [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "compare.R", "."]
        limit = (MEMORY, MEMORY)
        subprocess.run(
            command,
            check=True,
            cwd=root,
            preexec_fn=lambda: resource.setrlimit(resource.RLIMIT_AS, limit),
        )
        for name, case in cases.items():
            directory = root / names[name]
            case["edgeR"] = {
                p.stem[4:]: np.fromfile(p) for p in directory.glob("out_*.bin")
            }
            case["edgeR errors"] = {
                p.stem[4:]: p.read_text().strip() for p in directory.glob("out_*.txt")
            }
        moments = np.fromfile(root / "out_moments.bin").reshape(5, -1).T
        direct = {p.stem[4:]: np.fromfile(p) for p in direct.glob("out_*.bin")}
        versions = (root / "versions.txt").read_text().split("\n")[:3]
        return moments, direct, versions


def contrast_matrix(case, test):
    if test == "joint":
        return np.column_stack(list(case["contrasts"].values())).astype(float)
    return np.asarray(case["contrasts"][test], float)[:, None]


def ours(case):
    """CytePool's fit and tests of a case, as edgeR got it; a refused test is its
    error message."""
    keywords = {
        k: case[k]
        for k in ("library_sizes", "offsets", "weights", "dispersion")
        if k in case
    }
    trend = case.get("covariate", case.get("trend", True))
    fit = fit_quasi_likelihood(
        case["counts"],
        case["design"],
        trend=trend,
        robust=bool(case.get("robust")),
        **keywords,
    )
    tests = []
    for test in case["tests"]:
        try:
            if test.startswith("coef"):
                positions = [int(word) - 1 for word in test.split()[1:]]
                tests.append(quasi_likelihood_f_test(fit, coefficients=positions))
            else:
                matrix = contrast_matrix(case, test)
                tests.append(quasi_likelihood_f_test(fit, contrasts=matrix))
        except ValueError as error:
            tests.append(str(error))
    return fit, tests


def relative(a, b):
    """Largest |a / b - 1| over the entries where b is finite and non-zero."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    keep = np.isfinite(b) & (b != 0)
    return float(np.max(np.abs(a[keep] / b[keep] - 1))) if keep.any() else 0.0


def scaled(a, b, floor=1.0):
    """Largest |a - b| / max(|b|, floor) over the entries where b is finite."""
    a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
    keep = np.isfinite(b)
    if not keep.any():
        return 0.0
    return float(np.max(np.abs(a[keep] - b[keep]) / np.maximum(np.abs(b[keep]), floor)))


def log_p(a, b):
    """Largest |log10 a - log10 b| / max(1, |log10 b|); p-values that are zero in
    both agree, and a zero in one of them only is an infinite difference."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    both = (a == 0) & (b == 0)
    with np.errstate(divide="ignore"):
        la, lb = np.log10(a[~both]), np.log10(b[~both])
    if la.size == 0:
        return 0.0
    difference = np.where(la == lb, 0.0, np.abs(la - lb))
    return float(np.max(difference / np.maximum(1.0, np.abs(lb))))


def row(table, check, reference, metric, value, tolerance, result=None):
    """A row; without a tolerance it is information unless a result is given."""
    if result is None:
        result = (
            "information"
            if tolerance is None
            else ("pass" if value <= tolerance else "FAIL")
        )
    shown = "n/a" if tolerance is None else f"{tolerance:.0e}"
    table.append((check, reference, metric, f"{value:.1e}", shown, result))


def compare_case(name, case, table):
    """Rows comparing a case with edgeR.

    edgeR's moments are off by up to about 1.2e-3 at these dispersions, so that the
    adjusted deviances, quasi-dispersions and F-statistics agree to MOMENTS, and the
    average quasi-dispersion, which two passes compound, to AVERAGE. Where the two
    averages differ, so do the working dispersions, and with them the fits: the
    quasi-dispersions, log fold changes, leverages and likelihood ratios then agree
    to that tier too.
    """
    edger, (n, n_genes) = case["edgeR"], case["counts"].shape
    fit, tests = ours(case)
    one_way = np.unique(case["design"], axis=0).shape[0] == case["design"].shape[1]
    estimated = "top.proportion" in edger
    shared = edger["average.ql.dispersion"][0] == 1.0 == fit.average_quasi_dispersion
    exact = one_way and shared
    # With offsets per observation, edgeR's aveLogCPM ignores those of the first gene,
    # which has no counts here, so that its value enters nothing else.
    genes = np.arange(n_genes) >= (1 if "offsets" in case else 0)

    def check(label, reference, metric, value, tolerance, robust_prior=False):
        """A row; one that limma's robust prior changes where some genes have no
        counts is a deliberate difference in that case."""
        if robust_prior and name == "robust_with_empty_genes":
            tolerance, result = None, "deviation"
        else:
            result = None if tolerance is not None else "deviation"
        row(table, f"{name}: {label}", reference, metric, value, tolerance, result)

    if estimated:
        share = abs(fit.top_proportion - edger["top.proportion"][0])
        check("share of genes for the dispersion", FIT, "difference", share, 0.0)
        value = relative(fit.dispersion, edger["dispersion"])
        check("NB dispersion", FIT, RELATIVE, value, SEARCH)
        value = relative(fit.dispersion, edger["tight_dispersion"])
        tight = TIGHT if one_way else LEVENBERG_SEARCH
        check(
            "NB dispersion",
            "dispCoxReid(tol = 1e-12), same genes",
            RELATIVE,
            value,
            tight,
        )
    value = relative(fit.average_quasi_dispersion, edger["average.ql.dispersion"])
    check("average quasi-dispersion", FIT, RELATIVE, value, AVERAGE)
    value = max(
        scaled(fit.deviance_adjusted, edger["deviance.adj"], FLOOR),
        scaled(fit.df_residual_adjusted, edger["df.residual.adj"], FLOOR),
    )
    metric = f"max difference / max(value, {FLOOR:g})"
    check("adjusted deviances and residual df", FIT, metric, value, MOMENTS)
    prior = np.broadcast_to(fit.prior_quasi_dispersion, n_genes)
    value = max(
        relative(prior[genes], np.broadcast_to(edger["s2.prior"], n_genes)[genes]),
        relative(fit.posterior_quasi_dispersion[genes], edger["s2.post"][genes]),
    )
    tolerance = MOMENTS if shared else AVERAGE
    check(
        "prior and posterior quasi-dispersions",
        FIT,
        MAX_RELATIVE,
        value,
        tolerance,
        True,
    )
    value = relative(np.broadcast_to(fit.prior_df, n_genes), edger["df.prior"])
    check("prior df", FIT, MAX_RELATIVE, value, PRIOR_DF, True)
    # The moderation of edgeR's own adjusted deviances: only the moments differ above.
    with np.errstate(divide="ignore", invalid="ignore"):
        quasi = edger["deviance.adj"] / edger["df.residual.adj"]
    covariate = edger["abundance"] if case.get("trend", True) is not False else None
    squeezed = squeeze_variances(
        quasi,
        edger["df.residual.adj"],
        covariate=case.get("covariate", covariate),
        robust=bool(case.get("robust")),
        prior_estimator="likelihood",
    )
    value = max(
        relative(np.broadcast_to(squeezed.prior_variance, n_genes), edger["s2.prior"]),
        relative(np.broadcast_to(squeezed.prior_df, n_genes), edger["df.prior"]),
        relative(squeezed.posterior_variance, edger["s2.post"]),
    )
    label = "the same from edgeR's adjusted deviances"
    check(label, "glmQLFit, squeezeVar", MAX_RELATIVE, value, CLOSE, True)
    # Observations whose fitted values tend to zero (near 1e-8 in edgeR's fits) get
    # the leverages of the limit in both.
    value = float(np.max(np.abs(fit.leverage - edger["leverage"].reshape(n, n_genes))))
    check("leverages", FIT, "max difference", value, CLOSE if exact else LEVENBERG)
    for i, (test, result) in enumerate(zip(case["tests"], tests), start=1):
        label = f"test {test}"
        error = case["edgeR errors"].get(f"test{i}_error")
        if error is not None or isinstance(result, str):
            outcome = (
                f"edgeR: {'error' if error else 'values'}; CytePool: "
                f"{'error' if isinstance(result, str) else 'values'}"
            )
            both = error is not None and isinstance(result, str)
            table.append(
                (
                    f"{name}: {label}",
                    TEST,
                    "behaviour",
                    outcome,
                    "n/a",
                    "pass" if both else "FAIL",
                )
            )
            continue
        e = {
            k[len(f"test{i}_") :]: v
            for k, v in edger.items()
            if k.startswith(f"test{i}_")
        }
        logcpm = np.asarray(result.average_log_cpm)
        if not genes[0]:
            value = abs(logcpm[0] - e["logCPM"][0])
            check(f"{label}: logCPM of the first gene", TEST, "difference", value, None)
        # At a dispersion estimated by both, the log CPM is at slightly different
        # dispersions.
        value = float(np.max(np.abs(logcpm - e["logCPM"])[genes]))
        tolerance = SEARCH_LOG_CPM if estimated else CLOSE
        check(f"{label}: logCPM", TEST, "max difference", value, tolerance)
        lfc = e["logFC"].reshape(-1, n_genes).T
        value = float(np.max(np.abs(result.log_fold_change - lfc)))
        check(
            f"{label}: logFC",
            TEST,
            "max difference",
            value,
            CLOSE if exact else LEVENBERG,
        )
        # The likelihood ratio: F times the posterior quasi-dispersion and r.
        reference = e["F"] * edger["s2.post"] * result.df_test
        value = scaled(result.likelihood_ratio, reference)
        if not shared:
            tolerance = AVERAGE
        elif estimated or not one_way:
            tolerance = SEARCH
        else:
            tolerance = CLOSE
        check(f"{label}: likelihood ratio", TEST, SCALED, value, tolerance)
        value = scaled(result.f_statistic[genes], e["F"][genes])
        check(f"{label}: F", TEST, SCALED, value, MOMENTS, True)
        value = max(
            log_p(result.p_value[genes], e["PValue"][genes]),
            log_p(result.adjusted_p_value[genes], e["FDR"][genes]),
        )
        metric = "max |log10 ratio| / max(1, |log10 p|)"
        check(
            f"{label}: PValue and FDR",
            f"{TEST}, topTags",
            metric,
            value,
            PROBABILITY,
            True,
        )
        value = relative(result.df_total, e["df.total"])
        check(f"{label}: df.total", TEST, MAX_RELATIVE, value, PRIOR_DF, True)


def compare_direct(case, direct, table):
    """common_dispersion and average_log_cpm against edgeR's functions."""
    y, x, lib = case["counts"], case["design"], case["library_sizes"]
    w = DIRECT_WEIGHTS
    reference = direct["common"]
    for label, weights, default, tight in (
        ("common dispersion", None, reference[0], reference[1]),
        ("common dispersion, prior weights", w, reference[2], reference[3]),
    ):
        value = common_dispersion(y, x, library_sizes=lib, weights=weights)
        row(
            table,
            label,
            "estimateGLMCommonDisp",
            RELATIVE,
            relative(value, default),
            SEARCH,
        )
        row(
            table,
            label,
            "dispCoxReid(tol = 1e-12)",
            RELATIVE,
            relative(value, tight),
            TIGHT,
        )
    value = max(
        relative(common_dispersion(y, x, library_sizes=lib, subset=100), reference[4]),
        relative(
            common_dispersion(y, x, library_sizes=lib, weights=w, subset=100),
            reference[5],
        ),
    )
    label = "common dispersion, a systematic subset of 100 genes"
    row(
        table, label, "estimateGLMCommonDisp(subset = 100)", MAX_RELATIVE, value, SEARCH
    )
    mine = np.column_stack(
        [
            average_log_cpm(y, library_sizes=lib),
            average_log_cpm(y, library_sizes=lib, prior_count=0.5, dispersion=0.3),
            average_log_cpm(y, library_sizes=lib, weights=w, dispersion=0.0),
            average_log_cpm(y + 0.5, offsets=np.log(lib), prior_count=0.0),
        ]
    )
    value = float(np.max(np.abs(mine - direct["abundance"].reshape(4, -1).T)))
    row(
        table,
        "average log CPM: four settings",
        "aveLogCPM",
        "max difference",
        value,
        CLOSE,
    )


def exact_moments(mu, phi, terms):
    """Mean and variance of the unit deviance by summation at 30 digits."""
    mpmath.mp.dps = 30
    mu, phi = mpmath.mpf(mu), mpmath.mpf(phi)
    log_p = -mu if phi == 0 else -mpmath.log1p(phi * mu) / phi
    r = None if phi == 0 else 1 / phi
    sums = [mpmath.mpf(0)] * 3
    for y in range(terms):
        if y > 0:
            log_p += (
                mpmath.log(mu / y)
                if r is None
                else mpmath.log((y - 1 + r) / y * mu / (mu + r))
            )
        p, yy = mpmath.exp(log_p), mpmath.mpf(y)
        first = yy * mpmath.log(yy / mu) if y > 0 else 0
        second = (
            yy - mu
            if r is None
            else (yy + r) * mpmath.log((1 + phi * yy) / (1 + phi * mu))
        )
        d = 2 * (first - second)
        sums = [sums[0] + p, sums[1] + p * d, sums[2] + p * d * d]
    mean = sums[1] / sums[0]
    return float(mean), float(sums[2] / sums[0] - mean * mean)


def moment_grid():
    """(mean, dispersion, terms) where 30-digit sums need at most 25,000 terms."""
    grid = []
    for phi in (0.0, 1e-3, 0.05, 0.2, 1.0, 5.0):
        for mu in (1e-10, 1e-3, 0.03, 0.4, 3.0, 30.0, 300.0):
            sd = np.sqrt(mu + phi * mu * mu)
            tail = 0.0 if phi == 0 else 60.0 / np.log1p(1.0 / (phi * mu))
            terms = int(mu + 14.0 * sd + tail + 60)
            if terms <= 25_000:
                grid.append((mu, phi, terms))
    return grid


def compare_moments(moments, table):
    """The moments against mpmath and the gamma limit, and edgeR's approximation."""
    worst = 0.0
    for mu, phi, terms in moment_grid():
        mean, variance = negative_binomial_unit_deviance_moments(mu, phi)
        exact = exact_moments(mu, phi, terms)
        worst = max(worst, abs(mean / exact[0] - 1), abs(variance / exact[1] - 1))
    row(
        table,
        f"moments of the unit deviance at {len(moment_grid())} points",
        "mpmath, 30 digits",
        "max relative difference",
        worst,
        EXACT_MOMENTS,
    )
    worst = 0.0
    for phi in (0.01, 0.2, 1.0):
        r = mpmath.mpf(1) / phi
        limit = (
            2 * r * (mpmath.log(r) - mpmath.digamma(r)),
            4 * r * r * (mpmath.polygamma(1, r) - 1 / r),
        )
        mean, variance = negative_binomial_unit_deviance_moments(1e14, phi)
        worst = max(worst, abs(mean / limit[0] - 1), abs(variance / limit[1] - 1))
    row(
        table,
        "moments at a mean of 1e14",
        "the gamma limit, mpmath",
        "max relative difference",
        float(worst),
        EXACT_MOMENTS,
    )
    # edgeR's scale s and df k, implied by its unit matrices at means 1e-3 to 1e5.
    for phi in (0.01, 0.2, 1.0):
        part = moments[moments[:, 0] == phi]
        mean, variance = negative_binomial_unit_deviance_moments(
            part[:, 2] / part[:, 1], phi
        )
        exact = {"scale s": variance / (2 * mean), "df k": 2 * mean * mean / variance}
        for (label, value), column in zip(exact.items(), (4, 3)):
            row(
                table,
                f"edgeR's chi-squared match: {label}, dispersion {phi:g}",
                "glmQLFit unit matrices against the exact moments",
                "max relative difference",
                relative(part[:, column], value),
                None,
                "deviation",
            )


def write_fixtures(rng):
    """Small versions of some cases (40 genes) with edgeR's outputs, for the tests."""
    cases = make_cases(rng, 40)
    global DIRECT_WEIGHTS
    DIRECT_WEIGHTS = DIRECT_WEIGHTS[:, :40]
    run_edger(cases)
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name in (
        "two_groups",
        "groups_covariate",
        "paired",
        "prior_weights",
        "dispersion_given_low",
        "robust",
        "small_counts",
        "trended_dispersion",
        "no_trend",
        "covariate_trend",
        "offsets_per_observation",
    ):
        case = cases[name]
        arrays = {"counts": case["counts"], "design": case["design"]}
        arrays |= {k: np.asarray(case[k], float) for k in OPTIONAL if k in case}
        arrays |= {f"edger_{k}": v for k, v in case["edgeR"].items()}
        # The tests, as "coef j" (1-based) or a contrast matrix.
        arrays["tests"] = np.array(case["tests"])
        for test in case["tests"]:
            if not test.startswith("coef"):
                arrays[f"contrast_{test}"] = contrast_matrix(case, test)
        arrays["robust"] = np.array(bool(case.get("robust")))
        arrays["no_trend"] = np.array(case.get("trend") is False)
        np.savez_compressed(FIXTURE_DIRECTORY / f"{name}.npz", **arrays)
    grid = moment_grid()
    exact = np.array([exact_moments(mu, phi, terms) for mu, phi, terms in grid])
    np.savez_compressed(
        FIXTURE_DIRECTORY / "moments.npz",
        means=np.array([g[0] for g in grid]),
        dispersions=np.array([g[1] for g in grid]),
        mean=exact[:, 0],
        variance=exact[:, 1],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true")
    arguments = parser.parse_args()
    start = time.perf_counter()
    rng = np.random.default_rng(20261025)
    cases = make_cases(rng, 1000)
    moments, direct, versions = run_edger(cases)
    table = []
    for name, case in cases.items():
        compare_case(name, case, table)
    compare_direct(cases["two_groups"], direct, table)
    compare_moments(moments, table)
    if arguments.write_fixtures:
        write_fixtures(np.random.default_rng(20261026))
    print("Reference: " + ", ".join(versions) + f", mpmath {mpmath.__version__}")
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for line in table:
        print("| " + " | ".join(line) + " |")
    results = [line[-1] for line in table]
    print(
        f"{len(table)} checks: {results.count('pass')} pass, "
        f"{results.count('information')} information, "
        f"{results.count('deviation')} deviation, {results.count('FAIL')} failed; "
        f"{time.perf_counter() - start:.1f} s"
    )
    return 1 if "FAIL" in results else 0


if __name__ == "__main__":
    sys.exit(main())
