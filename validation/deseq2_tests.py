"""Check DESeq2's dispersion estimates and its Wald and likelihood-ratio tests.

Compares ``cytepool.stats.deseq2`` with DESeq2 1.46 and PyDESeq2 0.5 on negative
binomial counts simulated from fixed seeds (600 genes, 4 to 12 samples, 2% of the genes
without counts): two groups; four groups, with the likelihood ratio test of the factor
and a Wald test of two levels that some genes have no counts in; groups and a covariate;
a paired design; small counts; two samples per group (two residual degrees of freedom);
an outlier sample; dispersions that rise with the mean, where the parametric trend
fails; and nearly Poisson counts, most of whose gene-wise estimates are at the lower
bound. Counts with outliers, at given dispersions, test the rules by which Cook's
distance removes p-values, under four designs.

DESeq2 is run on every case with its defaults (``DESeq(dds, minReplicatesForReplace =
Inf)``, with ``fitType = "mean"`` where its parametric trend fails, so that the stages
after it compare), and each stage of CytePool is compared given DESeq2's own inputs to
it:

- the means of the likelihood: DESeq2's for one-way layouts; for the other designs its
  fit at the rough dispersions with ``betaTol = 1e-14``, polished by Newton's method in
  R (this script);
- the gene-wise and maximum a posteriori dispersions, from those means, against their
  maxima found in R: the best point of a grid of the Cox-Reid adjusted log-likelihood
  (written here from the paper, with ``dnbinom``), refined by ``uniroot()`` on its
  derivative, whose digamma differences are summed exactly;
- the trend, the variance of the log residuals and the prior's variance, from DESeq2's
  gene-wise estimates, against DESeq2's; for two residual degrees of freedom the prior's
  variance against its maximum likelihood estimate computed here by integration on a
  grid, and DESeq2's simulation as a deliberate difference;
- the fits, from DESeq2's final dispersions: the coefficients and standard errors
  against DESeq2's fits with ``betaTol = 1e-14`` polished by Newton's method in R; the
  fitted values, hat values, Cook's distances, deviances and the columns of
  ``results()`` against those unpolished fits, with its defaults and other options.

The analysis from the counts is compared with DESeq2's defaults and with PyDESeq2, for
information: it differs where DESeq2's line searches and iterations stop short. Each
row pools the cases a check applies to and shows the worst, with its case; rows that
count genes say how many genes they cover, and fail when there are none.

Requirements:
    Python 3.13 with numpy, numba, scipy and mpmath, run from the repository root so
    that ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor package DESeq2 and jsonlite. Set ``RSCRIPT``
    to the Rscript executable when it is not on PATH.
    PyDESeq2 0.5 with pandas, in ``PYDESEQ2_PYTHON`` (default: this interpreter).
    R and PyDESeq2 run with their address space limited to 8 GB.

Usage:
    PYTHONPATH=. python validation/deseq2_tests.py
    PYTHONPATH=. python validation/deseq2_tests.py --write-fixtures
"""

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

from cytepool import Design
from cytepool.stats.deseq2 import (
    _count_score,
    _independent_filtering,
    deseq2_dispersions,
    deseq2_likelihood_ratio_test,
    deseq2_wald_test,
    fit_deseq2,
)

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "data" / "deseq2_tests"
FIXTURE_CASES = (
    "two_groups several_groups covariate paired small_counts two_df outlier_sample "
    "rising_dispersion near_poisson"
).split()
EXACT, CLOSE = 1e-10, 1e-6
# DESeq2 stops its fits when the deviance changes by less than betaTol of it. For genes
# of large dispersion, where its Fisher scoring converges slowly, its coefficients are
# then up to about 5e-6 from the fixed point even at betaTol = 1e-14 (the information
# row against the polished fits shows it), and what is computed from them as far. The
# deviance is stationary there, so deviances and the likelihood ratio statistics made
# of them agree far closer. P-values are compared on the log scale, relative to
# max(|log p|, 1): an error e in a statistic z moves log p by about z^2 e.
SLOW, STATIONARY = 1e-5, 1e-9
MEMORY = 8 * 1024**3
DEFAULT, TIGHT = "DESeq2 defaults", "DESeq2, betaTol = 1e-14"
POLISHED, MAXIMUM = "DESeq2, betaTol = 1e-14, Newton in R", "maximum in R (this script)"
CHECK, COUNT, DELIBERATE, INFORMATION = "check", "count", "deliberate", "information"
TESTS = {"lfc", "wald", "lrt", "flags", "counts p", "counts stat", "counts calls"}
RULES = {"rules", "rules cooks", "rules spared"}
# Families of checks, in the order printed: key, what is compared, reference, metric,
# tolerance (a bound for deliberate differences) and kind. A check passes when its worst
# case is within the tolerance; a count adds up the genes that differ.
FAMILIES = [
    ("design", "design matrices", "model.matrix", "max abs difference", 0, CHECK),
    ("moments", "size factors, means and variances of the normalized counts", DEFAULT,
     "max relative difference", EXACT, CHECK),
    ("means", "means of the likelihood", "estimateDispersionsGeneEst, or DESeq2 at the "
     "rough dispersions, Newton in R", "max relative difference", CLOSE, CHECK),
    ("genewise", "gene-wise dispersions of at least 1e-6", MAXIMUM,
     "max relative difference", CLOSE, CHECK),
    ("flat", "gene-wise dispersions below 1e-6", MAXIMUM,
     "genes on the other side of 1e-6", 0, COUNT),
    ("held", "genes whose rough dispersion is at the bound keep 1e-8", DEFAULT,
     "genes that do not", 0, COUNT),
    ("trend", "parametric trend from DESeq2's gene-wise estimates", DEFAULT,
     "max relative difference", EXACT, CHECK),
    ("trend fails", "parametric trend fails where DESeq2's does", DEFAULT,
     "cases that differ", 0, CHECK),
    ("mean trend", "mean trend from DESeq2's gene-wise estimates", "fitType = 'mean'",
     "max relative difference", EXACT, CHECK),
    ("var log", "variance of the log residuals", DEFAULT, "relative difference",
     EXACT, CHECK),
    ("prior", "variance of the prior, more than 3 residual df", DEFAULT,
     "relative difference", EXACT, CHECK),
    ("prior mle", "variance of the prior, 2 residual df", "maximum likelihood, "
     "integration on a grid (this script)", "relative difference", CLOSE, CHECK),
    ("map", "maximum a posteriori dispersions", MAXIMUM, "max relative difference",
     CLOSE, CHECK),
    ("outliers", "dispersion outliers, and the final dispersions they choose",
     DEFAULT, "genes that differ", 0, COUNT),
    ("coefficients", "coefficients and standard errors", POLISHED,
     "max relative difference, floor 1", CLOSE, CHECK),
    ("deviance", "deviances", TIGHT, "max relative difference", STATIONARY, CHECK),
    ("fitted", "fitted values, hat values, Cook's distances", TIGHT,
     "max difference, relative or to 1", SLOW, CHECK),
    ("lfc", "every test: log2 fold changes and standard errors", TIGHT,
     "max relative difference, floor 1", SLOW, CHECK),
    ("wald", "Wald tests: statistics, p-values, adjusted p-values", TIGHT,
     "max relative difference, p on the log scale", SLOW, CHECK),
    ("lrt", "likelihood ratio tests: statistics, p-values, adjusted p-values", TIGHT,
     "max relative difference, p on the log scale", STATIONARY, CHECK),
    ("flags", "genes flagged by Cook's distance; filtered genes, filter quantile",
     TIGHT, "differences", 0, COUNT),
    ("options", "genes flagged at cooksCutoff = 0.5; NaN without filtering", TIGHT,
     "differences", 0, COUNT),
    ("options padj", "adjusted p-values at cooksCutoff = 0.5 and without filtering",
     TIGHT, "max relative difference on the log scale", SLOW, CHECK),
    ("filters", "independent filtering on 20 filter statistics",
     "results(filter = ...)", "quantiles and filtered genes that differ", 0, CHECK),
    ("filter padj", "independent filtering on 20 filter statistics",
     "results(filter = ...)", "max relative difference of adjusted p-values", EXACT,
     CHECK),
    ("rules", "flags of Cook's distance at given dispersions", TIGHT,
     "genes flagged differently", 0, COUNT),
    ("rules cooks", "largest Cook's distances at given dispersions", TIGHT,
     "max difference, relative or to 1", SLOW, CHECK),
    ("score", "derivative in log(alpha) of a count's log-likelihood, 192 points",
     "mpmath, 50 digits", "max relative difference", 1e-12, CHECK),
    ("own means", "means of the likelihood, designs that are not one-way layouts",
     "estimateDispersionsGeneEst", "max relative difference", 1e-2, DELIBERATE),
    ("line search", "gene-wise dispersions, DESeq2's line search", DEFAULT,
     "median |log ratio|", 1e-3, DELIBERATE),
    ("map line search", "maximum a posteriori dispersions, DESeq2's line search",
     DEFAULT, "max relative difference", 0.1, DELIBERATE),
    ("default coefficients", "coefficients of DESeq2's fits at its defaults",
     POLISHED, "max relative difference, floor 1", 1e-2, DELIBERATE),
    ("mean trend used", "trend where the parametric fit fails: the mean, with a "
     "warning, where DESeq2 uses a local fit", DEFAULT, "cases that differ", 0,
     DELIBERATE),
    ("prior simulated", "variance of the prior, 2 residual df, DESeq2's simulation",
     DEFAULT, "relative difference", 0.2, DELIBERATE),
    ("rules spared", "genes above the cutoff spared in a comparison of two groups",
     TIGHT, "genes", None, INFORMATION),
    ("tight coefficients", "coefficients of DESeq2's fits at betaTol = 1e-14",
     POLISHED, "max relative difference, floor 1", None, INFORMATION),
    ("line search max", "gene-wise dispersions, DESeq2's line search", DEFAULT,
     "max |log ratio|", None, INFORMATION),
    ("final", "final dispersions from the counts", DEFAULT, "median |log ratio|",
     None, INFORMATION),
    ("counts p", "p-values from the counts", DEFAULT, "median relative difference",
     None, INFORMATION),
    ("counts stat", "statistics from the counts", DEFAULT,
     "max relative difference, floor 1", None, INFORMATION),
    ("counts calls", "genes called at a false discovery rate of 0.1 from the counts",
     DEFAULT, "called differently", None, INFORMATION),
    ("py genewise", "gene-wise dispersions from the counts", "PyDESeq2",
     "median |log ratio|", None, INFORMATION),
    ("py final", "final dispersions from the counts", "PyDESeq2", "median |log ratio|",
     None, INFORMATION),
    ("py p", "p-values from the counts", "PyDESeq2", "median relative difference",
     None, INFORMATION),
    ("py calls", "genes called at a false discovery rate of 0.1 from the counts",
     "PyDESeq2", "called differently", None, INFORMATION),
]  # fmt: skip

R_PROGRAM = r"""
suppressPackageStartupMessages({library(DESeq2); library(jsonlite)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(dir, name) {
  p <- file.path(dir, paste0(name, ".bin"))
  readBin(p, "double", n = file.size(p) / 8)
}
wr <- function(dir, name, x) writeBin(as.double(x), file.path(dir, paste0("out_", name, ".bin")))
lambda <- 1e-6 / log(2)^2
# The Cox-Reid adjusted log-likelihood of the dispersion exp(la), plus a normal log
# prior, and its derivative in la, with psi(y + r) - psi(r) summed exactly.
objective <- function(la, y, mu, design, centre, precision) {
  a <- exp(la); w <- mu / (1 + a * mu)
  sum(dnbinom(y, mu = mu, size = 1 / a, log = TRUE)) -
    0.5 * determinant(crossprod(design * w, design))$modulus[1] - 0.5 * precision * (la - centre)^2
}
score <- function(la, y, mu, design, centre, precision) {
  a <- exp(la); r <- 1 / a; w <- mu / (1 + a * mu)
  shift <- vapply(y, function(k) if (k == 0) 0 else sum(1 / (r + seq_len(k) - 1)), 0)
  h <- w * rowSums((design %*% solve(crossprod(design * w, design))) * design)
  sum(r * (log1p(a * mu) - shift) + (y - mu) / (1 + a * mu)) + 0.5 * sum(h * a * w) -
    precision * (la - centre)
}
# The maximum over [1e-8, high]: the best point of a grid, refined by the root of the
# derivative next to it, or the bound the derivative points to.
maximum <- function(y, mu, design, high, centre = 0, precision = 0) {
  grid <- seq(log(1e-8), log(high), length.out = 60)
  v <- vapply(grid, objective, 0, y = y, mu = mu, design = design, centre = centre, precision = precision)
  k <- which.max(v)
  s <- function(la) score(la, y, mu, design, centre, precision)
  if (k == 1 && s(grid[1]) <= 0) return(1e-8)
  if (k == 60 && s(grid[60]) >= 0) return(high)
  lo <- grid[max(k - 1, 1)]; up <- grid[min(k + 1, 60)]
  at <- c(s(lo), s(up))
  if (at[1] > 0 && at[2] < 0) return(exp(uniroot(s, c(lo, up), f.lower = at[1], f.upper = at[2], tol = 1e-14, maxiter = 1000)$root))
  # Where the derivative does not change sign there, the objective is flat to rounding.
  best <- optimize(objective, c(lo, up), y = y, mu = mu, design = design, centre = centre,
                   precision = precision, maximum = TRUE, tol = 1e-12)
  exp(if (best$objective >= v[k]) best$maximum else grid[k])
}
# The rough dispersions, from the moments of the normalized counts (genes x samples).
rough <- function(norm, X, s, high) {
  fitted <- pmax(norm %*% (X %*% solve(crossprod(X), t(X))), 1)
  rde <- pmax(rowSums(((norm - fitted)^2 - fitted) / fitted^2) / (ncol(norm) - ncol(X)), 0)
  m <- rowMeans(norm)
  mde <- (apply(norm, 1, var) - mean(1 / s) * m) / m^2
  pmin(pmax(pmin(rde, mde), 1e-8), high)
}
# Newton's method on the function DESeq2's fits maximize: the log-likelihood continued
# below a mean of 0.5 by its quadratic expansion there, less the ridge penalty.
polish <- function(y, X, offset, a, beta) {
  c0 <- 0.5 / (1 + 0.5 * a)
  for (k in 1:50) {
    eta <- drop(X %*% beta) + offset; mu <- exp(eta); low <- mu < 0.5
    u <- ifelse(low, (y - 0.5) / (1 + 0.5 * a) - c0 * (eta - log(0.5)), (y - mu) / (1 + a * mu))
    w <- ifelse(low, c0, mu * (1 + a * y) / (1 + a * mu)^2)
    step <- drop(solve(crossprod(X * w, X) + diag(lambda, ncol(X)), crossprod(X, u) - lambda * beta))
    beta <- beta + step
    if (max(abs(X %*% step)) < 1e-13) break
  }
  beta
}
standard_errors <- function(X, mu, a) {
  w <- pmax(mu, 0.5) / (1 + a * pmax(mu, 0.5)); info <- crossprod(X * w, X)
  inverse <- solve(info + diag(lambda, ncol(X)))
  sqrt(diag(inverse %*% info %*% inverse))
}
columns <- c("baseMean", "log2FoldChange", "lfcSE", "stat", "pvalue", "padj")
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  spec <- read_json(file.path(dir, "spec.json"))
  shape <- rd(dir, "shape"); n <- shape[1]; G <- shape[2]
  cts <- matrix(as.integer(rd(dir, "counts")), G, n)
  samples <- read.csv(file.path(dir, "samples.csv"), colClasses = "character")
  for (name in names(samples)) {
    samples[[name]] <- if (name %in% names(spec$levels)) factor(samples[[name]], levels = unlist(spec$levels[[name]])) else as.numeric(samples[[name]])
  }
  if (length(spec$formulas)) {
    # Cook's distances at given size factors and dispersions, under each design.
    for (i in seq_along(spec$formulas)) {
      dds <- suppressWarnings(suppressMessages(DESeqDataSetFromMatrix(cts, samples, as.formula(spec$formulas[[i]]))))
      sizeFactors(dds) <- rd(dir, "size_factors"); dispersions(dds) <- rd(dir, "dispersion")
      dds <- nbinomWaldTest(dds, betaTol = 1e-14, maxit = 1000, quiet = TRUE)
      wr(dir, paste0("maxCooks", i), mcols(dds)$maxCooks)
      wr(dir, paste0("pvalue", i), results(dds, contrast = rd(dir, paste0("contrast", i)))$pvalue)
    }
    next
  }
  dds <- suppressMessages(DESeqDataSetFromMatrix(cts, samples, as.formula(spec$formula)))
  X <- model.matrix(design(dds), samples); wr(dir, "design", X)
  e <- suppressWarnings(suppressMessages(estimateDispersions(estimateSizeFactors(dds), quiet = TRUE)))
  fit_type <- if (attr(dispersionFunction(e), "fitType") == "parametric") "parametric" else "mean"
  writeLines(fit_type, file.path(dir, "out_fit_type.txt"))
  d <- DESeq(dds, fitType = fit_type, minReplicatesForReplace = Inf, quiet = TRUE)
  mc <- mcols(d); f <- dispersionFunction(d)
  for (k in c("baseMean", "baseVar", "allZero", "dispGeneEst", "dispFit", "dispMAP", "dispersion", "dispOutlier")) wr(dir, k, mc[[k]])
  wr(dir, "size_factors", sizeFactors(d)); wr(dir, "coef", coef(d))
  wr(dir, "trend", if (is.null(attr(f, "coefficients"))) c(NA, NA) else attr(f, "coefficients"))
  wr(dir, "var_log", attr(f, "varLogDispEsts")); wr(dir, "prior_var", attr(f, "dispPriorVar"))
  t <- nbinomWaldTest(d, betaTol = 1e-14, maxit = 1000, quiet = TRUE)
  for (k in c("deviance", "maxCooks")) wr(dir, paste0("tight_", k), mcols(t)[[k]])
  for (k in c("mu", "H", "cooks")) wr(dir, paste0("tight_", k), assays(t)[[k]])
  wr(dir, "tight_coef", coef(t))
  keep <- which(!mc$allZero); s <- sizeFactors(d); high <- max(10, n)
  beta <- se <- matrix(NA_real_, G, ncol(X))
  for (g in keep) {
    b <- polish(cts[g, ], X, log(s), mc$dispersion[g], coef(t)[g, ] * log(2))
    beta[g, ] <- b / log(2)
    se[g, ] <- standard_errors(X, s * exp(drop(X %*% b)), mc$dispersion[g]) / log(2)
  }
  wr(dir, "polished_coef", beta); wr(dir, "polished_se", se)
  for (i in seq_along(spec$kinds)) {
    for (stage in c("default", "tight")) {
      fit <- if (stage == "default") d else t
      if (spec$kinds[[i]] == "wald") {
        r <- results(fit, contrast = rd(dir, paste0("contrast", i)))
      } else {
        tol <- if (stage == "default") 1e-8 else 1e-14
        r <- results(nbinomLRT(fit, reduced = as.formula(spec$reduced[[i]]), betaTol = tol,
                               maxit = 1000, quiet = TRUE))
      }
      for (k in columns) wr(dir, paste0(stage, i, "_", k), r[[k]])
      wr(dir, paste0(stage, i, "_theta"), metadata(r)$filterTheta)
    }
  }
  w <- rd(dir, paste0("contrast", which(unlist(spec$kinds) == "wald")[1]))
  r <- results(t, contrast = w, cooksCutoff = 0.5)
  wr(dir, "cutoff_pvalue", r$pvalue); wr(dir, "cutoff_padj", r$padj)
  wr(dir, "unfiltered_padj", results(t, contrast = w, independentFiltering = FALSE)$padj)
  if (spec$filters > 0) {
    filters <- matrix(rd(dir, "filters"), G)
    for (k in seq_len(spec$filters)) {
      r <- results(t, contrast = rd(dir, "contrast1"), filter = filters[, k])
      wr(dir, paste0("filter", k, "_padj"), r$padj); wr(dir, paste0("filter", k, "_theta"), metadata(r)$filterTheta)
    }
  }
  # The means of the likelihood, and the maxima of the dispersions at them.
  mu <- assays(estimateDispersionsGeneEst(d, quiet = TRUE))[["mu"]]
  wr(dir, "mu_gene", mu)
  r0 <- rough(counts(d, normalized = TRUE), X, s, high); wr(dir, "rough", r0)
  if (nrow(unique(X)) != ncol(X)) {
    a <- d; dispersions(a) <- r0
    start <- coef(nbinomWaldTest(a, betaTol = 1e-14, maxit = 1000, quiet = TRUE)) * log(2)
    for (g in keep) mu[g, ] <- pmax(s * exp(drop(X %*% polish(cts[g, ], X, log(s), r0[g], start[g, ]))), 0.5)
  }
  wr(dir, "mu_ref", mu)
  gw <- mp <- rep(NA_real_, G)
  for (g in keep) {
    gw[g] <- maximum(cts[g, ], mu[g, ], X, high)
    mp[g] <- maximum(cts[g, ], mu[g, ], X, high, log(mc$dispFit[g]), 1 / attr(f, "dispPriorVar"))
  }
  wr(dir, "genewise_ref", gw); wr(dir, "map_ref", mp)
}
writeLines(c(paste("R", getRversion()), paste("DESeq2", packageVersion("DESeq2"))), file.path(root, "versions.txt"))
"""

PYDESEQ2_PROGRAM = r"""
import json, sys, warnings
from importlib.metadata import version
import numpy as np, pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats
warnings.filterwarnings("ignore")
directory = sys.argv[1]
for case in json.load(open(f"{directory}/pydeseq2.json")):
    counts = pd.DataFrame(np.load(f"{directory}/{case}_counts.npy"))
    counts.columns = [f"g{j}" for j in counts.columns]
    spec = json.load(open(f"{directory}/{case}_spec.json"))
    metadata = pd.DataFrame(spec["samples"], index=counts.index)
    inference = DefaultInference(n_cpus=1)
    dds = DeseqDataSet(counts=counts, metadata=metadata, design=spec["formula"], refit_cooks=False,
                       inference=inference, quiet=True)
    dds.deseq2()
    contrast = np.array([spec["weights"].get(c, 0.0) for c in dds.obsm["design_matrix"].columns])
    stats = DeseqStats(dds, contrast=contrast, alpha=0.1, inference=inference, quiet=True)
    stats.summary()
    out = {k: dds.var[k].to_numpy() for k in ("genewise_dispersions", "dispersions")}
    out |= {k: stats.results_df[k].to_numpy() for k in ("log2FoldChange", "pvalue", "padj")}
    np.savez(f"{directory}/{case}_pydeseq2.npz", **out)
json.dump({"pydeseq2": version("pydeseq2")}, open(f"{directory}/pydeseq2_versions.json", "w"))
"""


def simulate(
    rng, labels, n_genes, sizes, *, mean=(3.0, 2.0), dispersion=None, covariate=None
):
    """Samples x genes counts: 10% of genes differ between labels, and 2% have none."""
    levels, index = np.unique(labels, return_inverse=True)
    mu = np.exp(rng.normal(*mean, n_genes))
    alpha = (0.03 + 1.0 / mu) * np.exp(rng.normal(0.0, 0.5, n_genes))
    if dispersion is not None:
        alpha = dispersion(mu)
    changed = rng.random((levels.size, n_genes)) < 0.1
    effect = np.where(changed, rng.normal(0.0, 1.5, changed.shape), 0.0)
    log_mean = np.log(sizes[:, None] * mu) + effect[index]
    if covariate is not None:
        log_mean += np.outer(covariate, rng.normal(0.0, 0.4, n_genes))
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + np.exp(log_mean))).astype(float)
    y[:, rng.random(n_genes) < 0.02] = 0.0
    return y


def filter_statistics(rng, y, samples, levels, formula, n_filters=20):
    """Statistics for independent filtering, from -log p of CytePool's Wald test, its
    mean counts and noise, in proportions that give rejection curves of every shape,
    among them flat ones that no quantile exceeds by the root mean square residual."""
    fit = fit_deseq2(y, Design(samples, formula, levels=levels))
    p = np.nan_to_num(deseq2_wald_test(fit, cooks_cutoff=False).p_value, nan=1.0)
    signal = -np.log(np.maximum(p, 1e-300))
    depth = np.argsort(np.argsort(fit.base_mean)) / p.size
    columns = [
        signal * rng.uniform(0, 3)
        + rng.normal(0, rng.uniform(0, 3), p.size)
        + rng.uniform(-1, 1) * 5 * depth
        for _ in range(n_filters)
    ]
    return np.column_stack(columns)


def make_cases(rng, n_genes):
    """Cases by name: samples, factor levels, formula, counts and tests."""
    cases = {}
    group = {"group": ["A", "B"]}
    wald = [("wald", [0.0, 1.0])]

    def add(name, samples, levels, formula, y, tests, **options):
        cases[name] = dict(samples=samples, levels=levels, formula=formula, counts=y)
        cases[name] |= dict(tests=tests) | options

    def two(k):
        return {"group": np.repeat(group["group"], k)}

    def sizes(n):
        return rng.uniform(0.6, 1.6, n)

    y = simulate(rng, two(3)["group"], n_genes, sizes(6))
    filters = filter_statistics(np.random.default_rng(31), y, two(3), group, "~ group")
    add("two_groups", two(3), group, "~ group", y, wald, filters=filters,
        pydeseq2={"group[T.B]": 1.0})  # fmt: skip
    four = {"group": ["A", "B", "C", "D"]}
    several = {"group": np.repeat(four["group"], 3)}
    y = simulate(rng, several["group"], n_genes, sizes(12))
    y[6:, : n_genes // 40] = 0.0  # groups C and D without counts: their contrast is 0
    tests = [("lrt", "~ 1"), ("wald", [0.0, 0.0, -1.0, 1.0])]
    add("several_groups", several, four, "~ group", y, tests,
        pydeseq2={"group[T.C]": -1.0, "group[T.D]": 1.0})  # fmt: skip
    aged = two(5) | {"age": rng.uniform(-1.5, 1.5, 10).round(2)}
    y = simulate(rng, aged["group"], n_genes, sizes(10), covariate=aged["age"])
    tests = [("wald", [0.0, 0.0, 1.0]), ("lrt", "~ group")]
    add("covariate", aged, group, "~ group + age", y, tests)
    donors = [f"D{i}" for i in range(1, 7)]
    paired = {"donor": np.repeat(donors, 2), "condition": np.tile(["ctrl", "stim"], 6)}
    levels = {"donor": donors, "condition": ["ctrl", "stim"]}
    shift = rng.normal(0, 1, 12)
    y = simulate(rng, paired["condition"], n_genes, sizes(12), covariate=shift)
    tests = [("wald", [0.0] * 6 + [1.0]), ("lrt", "~ donor")]
    add("paired", paired, levels, "~ donor + condition", y, tests,
        pydeseq2={"condition[T.stim]": 1.0})  # fmt: skip
    y = simulate(rng, two(4)["group"], n_genes, sizes(8), mean=(0.5, 1.0))
    add("small_counts", two(4), group, "~ group", y, wald, pydeseq2={"group[T.B]": 1.0})

    def wide(mu):
        return 0.08 * np.exp(rng.normal(0.0, 1.5, mu.size))

    y = simulate(rng, two(2)["group"], n_genes, sizes(4), dispersion=wide)
    add("two_df", two(2), group, "~ group", y, wald)
    y = simulate(rng, two(4)["group"], n_genes, sizes(8))
    y[0, rng.random(n_genes) < 0.04] *= 30.0  # an outlier sample
    add("outlier_sample", two(4), group, "~ group", y, wald)

    def rising(mu):
        return 0.01 * mu**0.5 * np.exp(rng.normal(0.0, 0.3, mu.size))

    y = simulate(rng, two(3)["group"], n_genes, sizes(6), mean=(4.0, 1.5),
                 dispersion=rising)  # fmt: skip
    add("rising_dispersion", two(3), group, "~ group", y, wald)

    def poisson(mu):
        return (3e-5 + 4e-3 / mu) * np.exp(rng.normal(0.0, 0.3, mu.size))

    y = simulate(rng, two(6)["group"], n_genes, sizes(12), mean=(6.0, 2.0),
                 dispersion=poisson)  # fmt: skip
    add("near_poisson", two(6), group, "~ group", y, wald)
    return cases


def cooks_rules(rng, n_genes=120):
    """Counts with an outlier in most genes, some with other samples at or next to its
    count, at a dispersion of 0.05: two groups of four, and groups of two and five as a
    factor, as means without an intercept and as a numeric covariate."""

    def counts(n):
        mean = rng.uniform(100.0, 1000.0, n_genes)
        y = rng.negative_binomial(20, 20 / (20 + mean), (n, n_genes)).astype(float)
        for g in range(n_genes):
            i, factor = rng.integers(n), np.exp(rng.uniform(np.log(2), np.log(40)))
            y[i, g] = np.round(
                y[i, g] * factor if rng.random() < 0.6 else y[i, g] / factor
            )
            others = rng.choice(
                np.delete(np.arange(n), i), rng.integers(0, 4), replace=False
            )
            y[others, g] = y[i, g] + rng.integers(-1, 2, others.size) * (
                rng.random() < 0.7
            )
        return y

    balanced = counts(8)
    balanced[:, :3] = [
        [1e3, 1e3, 1e3],
        [1e3, 1e3, 1e3],
        [1e3, 1e3, 1e3],
        [1e4] * 3,
        [9e3, 1e4, 1.05e4],
        [1e4] * 3,
        [1.1e4] * 3,
        [1e4, 1.1e4, 1.1e4],
    ]
    unbalanced = counts(7)
    unbalanced[:, :4] = [
        [5, 5, 20, 60],
        [600, 1500, 3000, 600],
        [150] * 4,
        [140] * 4,
        [160] * 4,
        [150] * 4,
        [1000, 1500, 1000, 1000],
    ]
    labels = ["A", "A", "B", "B", "B", "B", "B"]
    return {
        "cooks_two_groups": dict(
            samples=dict(group=np.repeat(["A", "B"], 4)), counts=balanced,
            size_factors=np.ones(8), formulas=[("~ group", [0.0, 1.0])]),
        "cooks_unbalanced": dict(
            samples=dict(group=np.array(labels), treated=np.array([0, 0, 1, 1, 1, 1, 1])),
            counts=unbalanced, size_factors=np.array([1.4, 1, 1, 1, 1, 1, 1.0]),
            formulas=[("~ group", [0.0, 1.0]), ("~ 0 + group", [-1.0, 1.0]),
                      ("~ treated", [0.0, 1.0])]),
    }  # fmt: skip


def limited():
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY, MEMORY))


def write_case(folder, case):
    """The inputs of one case for the R program."""
    folder.mkdir()
    y = case["counts"]
    np.asarray(y.shape, float).tofile(folder / "shape.bin")
    y.astype(float).tofile(folder / "counts.bin")
    columns = list(case["samples"])
    rows = zip(*(np.asarray(case["samples"][c]).astype(str) for c in columns))
    lines = [",".join(columns)] + [",".join(r) for r in rows]
    (folder / "samples.csv").write_text("\n".join(lines) + "\n")
    spec = {"levels": case.get("levels", {"group": ["A", "B"]})}
    if "formulas" in case:
        spec["formulas"] = [formula for formula, _ in case["formulas"]]
        for i, (_, weights) in enumerate(case["formulas"], start=1):
            np.asarray(weights, float).tofile(folder / f"contrast{i}.bin")
        case["size_factors"].astype(float).tofile(folder / "size_factors.bin")
        np.full(y.shape[1], 0.05).tofile(folder / "dispersion.bin")
    else:
        spec["formula"] = case["formula"]
        spec["kinds"] = [kind for kind, _ in case["tests"]]
        spec["reduced"] = [v if kind == "lrt" else "" for kind, v in case["tests"]]
        for i, (kind, value) in enumerate(case["tests"], start=1):
            if kind == "wald":
                np.asarray(value, float).tofile(folder / f"contrast{i}.bin")
        spec["filters"] = case["filters"].shape[1] if "filters" in case else 0
        if spec["filters"]:
            case["filters"].ravel(order="F").tofile(folder / "filters.bin")
    (folder / "spec.json").write_text(json.dumps(spec))


def run_references(cases, directory):
    """DESeq2's outputs for every case, and PyDESeq2's for those that name a contrast."""
    names = {name: f"case{i}" for i, name in enumerate(cases)}
    for name, case in cases.items():
        write_case(directory / names[name], case)
    (directory / "cases.txt").write_text("\n".join(names.values()) + "\n")
    (directory / "compare.R").write_text(R_PROGRAM)
    command = [os.environ.get("RSCRIPT", "Rscript"), "--vanilla", "compare.R", "."]
    subprocess.run(command, check=True, cwd=directory, preexec_fn=limited)
    for name, case in cases.items():
        folder = directory / names[name]
        case["R"] = {p.stem[4:]: np.fromfile(p) for p in folder.glob("out_*.bin")}
        for path in folder.glob("out_*.txt"):
            case["R"][path.stem[4:]] = path.read_text().strip()
    py = [name for name, case in cases.items() if "pydeseq2" in case]
    for name in py:
        np.save(
            directory / f"{name}_counts.npy", cases[name]["counts"].astype(np.int64)
        )
        samples = {k: np.asarray(v).tolist() for k, v in cases[name]["samples"].items()}
        spec = dict(samples=samples, formula=cases[name]["formula"])
        spec["weights"] = cases[name]["pydeseq2"]
        (directory / f"{name}_spec.json").write_text(json.dumps(spec))
    (directory / "pydeseq2.json").write_text(json.dumps(py))
    (directory / "pydeseq2.py").write_text(PYDESEQ2_PROGRAM)
    environment = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    python = os.environ.get("PYDESEQ2_PYTHON", sys.executable)
    command = [python, "-I", str(directory / "pydeseq2.py"), str(directory)]
    subprocess.run(command, check=True, env=environment, preexec_fn=limited)
    for name in py:
        with np.load(directory / f"{name}_pydeseq2.npz") as data:
            cases[name]["PyDESeq2"] = {k: data[k] for k in data.files}
    versions = (directory / "versions.txt").read_text().split("\n")[:2]
    found = json.loads((directory / "pydeseq2_versions.json").read_text())
    return versions + [f"PyDESeq2 {found['pydeseq2']}"]


def relative(ours, reference, floor=0.0, where=None):
    """Largest |ours - reference| / max(|reference|, floor) where the reference is finite;
    NaN where none is, or where ours is not finite where the reference is."""
    ours = np.atleast_1d(np.asarray(ours, float))
    reference = np.atleast_1d(np.asarray(reference, float))
    keep = np.isfinite(reference) if where is None else np.isfinite(reference) & where
    if not keep.any() or not np.isfinite(ours[keep]).all():
        return float("nan")
    scale = np.maximum(np.abs(reference[keep]), max(floor, 1e-300))
    return float(np.max(np.abs(ours[keep] - reference[keep]) / scale))


def log_relative(ours, reference):
    """Largest |log ours - log reference| / max(|log reference|, 1) where the reference
    is finite, for p-values, which below 1e-300 count as 1e-300."""
    keep = np.isfinite(reference)
    if not keep.any() or not np.isfinite(ours[keep]).all():
        return float("nan")
    ours, reference = (np.log(np.maximum(v[keep], 1e-300)) for v in (ours, reference))
    return float(np.max(np.abs(ours - reference) / np.maximum(np.abs(reference), 1.0)))


def nan_mismatches(ours, reference):
    return int(np.count_nonzero(np.isnan(ours) != np.isnan(reference)))


def median_relative(ours, reference):
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.nanmedian(np.abs(ours - reference) / reference))


def calls(ours, reference):
    """Genes significant at 0.1 in one and not the other, and in the reference."""
    return int(np.count_nonzero((ours < 0.1) != (reference < 0.1))), int(
        np.nansum(reference < 0.1)
    )


def few_df_maximum_likelihood(residuals, df):
    """The maximum likelihood estimate of s2 in r = log(chi2_df / df) + N(0, s2), the
    density integrated by the trapezoidal rule on a fixed grid of u, at least 0.25."""
    k = df / 2.0
    u = np.linspace(-45.0, 6.0, 40001)
    log_f = k * (u + np.log(k)) - k * np.exp(u) - gammaln(k)

    def minus_log_likelihood(s2):
        total = 0.0
        for chunk in np.array_split(residuals, max(1, residuals.size // 200)):
            e = log_f[None, :] - (chunk[:, None] - u[None, :]) ** 2 / (2 * s2)
            top = e.max(axis=1, keepdims=True)
            density = np.trapezoid(np.exp(e - top), u, axis=1)
            total += np.sum(top[:, 0] + np.log(density) - 0.5 * np.log(2 * np.pi * s2))
        return -total

    options = {"xatol": 1e-11}
    found = minimize_scalar(
        minus_log_likelihood, bounds=(0.25, 8.0), method="bounded", options=options
    )
    return max(found.x, 0.25)


def compare_case(name, case, found):
    """Add every check of one simulated case to ``found``."""
    R, y = case["R"], case["counts"]
    n, genes = y.shape
    samples = {k: np.asarray(v) for k, v in case["samples"].items()}
    design = Design(samples, case["formula"], levels=case["levels"])
    x = design.matrix
    p = x.shape[1]
    counted = ~R["allZero"].astype(bool)
    bound = counted & (R["rough"] <= 1e-8)

    def add(key, value, size=None):
        found.setdefault(key, []).append((name, value, size))

    def matrix(key, columns):
        """An R matrix of genes x columns, written column by column."""
        return R[key].reshape(genes, columns, order="F")

    add("design", float(np.max(np.abs(x - R["design"].reshape(n, p, order="F")))))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_deseq2(y, design)
    d = fit.dispersion_estimates
    if R["fit_type"] == "mean":
        warned = any("parametric dispersion trend" in str(w.message) for w in caught)
        add("mean trend used", int(d.trend != "mean" or not warned))
    moments = [(d.size_factors, "size_factors"), (d.base_mean, "baseMean")]
    moments.append((d.base_variance, "baseVar"))
    add("moments", max(relative(ours, R[key]) for ours, key in moments))
    add("means", relative(d.means, matrix("mu_ref", n).T))
    if np.unique(x, axis=0).shape[0] != p:
        add("own means", relative(d.means, matrix("mu_gene", n).T))
    # Below 1e-6 the adjusted likelihood is flat in the log dispersion (its derivative
    # is proportional to the dispersion), and a gene shows no overdispersion to speak of.
    curved = counted & ~bound & (R["genewise_ref"] >= 1e-6)
    add("genewise", relative(d.genewise, R["genewise_ref"], where=curved))
    free = counted & ~bound
    sides = (d.genewise[free] < 1e-6) != (R["genewise_ref"][free] < 1e-6)
    add("flat", int(np.count_nonzero(sides)), int(np.count_nonzero(free & ~curved)))
    held = (d.genewise[bound] != 1e-8) | (R["dispGeneEst"][bound] != 1e-8)
    add("held", int(np.count_nonzero(held)), int(np.count_nonzero(bound)))
    near = np.abs(np.log(d.genewise[counted] / R["dispGeneEst"][counted]))
    add("line search", float(np.median(near)))
    add("line search max", float(near.max()))
    final = np.abs(np.log(d.dispersion[counted] / R["dispersion"][counted]))
    add("final", float(np.median(final)))
    given = dict(size_factors=R["size_factors"], genewise_dispersions=R["dispGeneEst"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        staged = deseq2_dispersions(y, design, **given)
    if R["fit_type"] == "parametric":
        ours = [staged.asymptotic_dispersion, staged.extra_poisson]
        add("trend", relative(ours, R["trend"]))
    else:
        add("trend fails", int(staged.trend != "mean" or not caught))
        add("mean trend", relative(staged.trended, R["dispFit"]))
    given["trended_dispersions"] = R["dispFit"]
    trended = deseq2_dispersions(y, design, **given)
    add("var log", relative(trended.log_residual_variance, R["var_log"]))
    if n - p > 3:
        add("prior", relative(trended.prior_variance, R["prior_var"]))
    else:
        usable = counted & (R["dispGeneEst"] >= 1e-6)
        residuals = np.log(R["dispGeneEst"][usable] / R["dispFit"][usable])
        mle = case["mle_prior_variance"] = few_df_maximum_likelihood(residuals, n - p)
        add("prior mle", relative(trended.prior_variance, mle))
        add("prior simulated", relative(trended.prior_variance, R["prior_var"]))
    posterior = deseq2_dispersions(y, design, **given, prior_variance=R["prior_var"][0])
    add("map", relative(posterior.map_estimate, R["map_ref"]))
    add("map line search", relative(posterior.map_estimate, R["dispMAP"]))
    outlier = R["dispOutlier"][counted] == 1.0
    chosen = np.where(posterior.outlier, R["dispGeneEst"], R["dispMAP"])[counted]
    differ = np.count_nonzero(posterior.outlier[counted] != outlier)
    differ += np.count_nonzero(chosen != R["dispersion"][counted])
    add("outliers", int(differ), int(np.count_nonzero(outlier)))
    given_fit = fit_deseq2(
        y, design, size_factors=R["size_factors"], dispersion=R["dispersion"]
    )
    polished = matrix("polished_coef", p)
    add("coefficients", max(relative(given_fit.coefficients, polished, 1.0),
                            relative(given_fit.standard_errors, matrix("polished_se", p))))  # fmt: skip
    add("tight coefficients", relative(matrix("tight_coef", p), polished, 1.0))
    add("default coefficients", relative(matrix("coef", p), polished, 1.0))
    add("deviance", relative(given_fit.minus_twice_log_likelihood, R["tight_deviance"]))
    tight = {k: matrix(f"tight_{k}", n).T for k in ("mu", "H", "cooks")}
    value = max(
        relative(given_fit.fitted_values, tight["mu"]),
        relative(given_fit.hat_values, tight["H"], 1.0),
        relative(given_fit.cooks_distances, tight["cooks"], 1.0),
    )
    if np.isfinite(R["tight_maxCooks"]).any():
        value = max(value, relative(given_fit.max_cooks, R["tight_maxCooks"], 1.0))
    add("fitted", value + nan_mismatches(given_fit.max_cooks, R["tight_maxCooks"]))
    for i, (kind, value) in enumerate(case["tests"], start=1):
        if kind == "wald":
            ours, end = (deseq2_wald_test(f, np.array(value)) for f in (given_fit, fit))
        else:
            reduced = Design(samples, value, levels=case["levels"])
            ours, end = (deseq2_likelihood_ratio_test(f, reduced) for f in (given_fit, fit))  # fmt: skip
        t = {
            k: R[f"tight{i}_{k}"] for k in ("log2FoldChange", "lfcSE", "stat", "pvalue")
        }
        t |= {k: R[f"tight{i}_{k}"] for k in ("padj", "theta")}
        add("lfc", max(relative(ours.log2_fold_change, t["log2FoldChange"], 1.0),
                       relative(ours.standard_error, t["lfcSE"])))  # fmt: skip
        statistics = [relative(ours.statistic, t["stat"], 1.0)]
        statistics += [log_relative(ours.p_value, t["pvalue"])]
        add(kind, max(statistics + [log_relative(ours.adjusted_p_value, t["padj"])]))
        differ = nan_mismatches(ours.p_value, t["pvalue"])
        differ += nan_mismatches(ours.adjusted_p_value, t["padj"])
        add("flags", differ + int(ours.filter_quantile != t["theta"][0]),
            int(np.count_nonzero(ours.cooks_outlier)))  # fmt: skip
        default = {k: R[f"default{i}_{k}"] for k in ("stat", "pvalue", "padj")}
        add("counts p", median_relative(end.p_value, default["pvalue"]))
        add("counts stat", relative(end.statistic, default["stat"], 1.0))
        add("counts calls", calls(end.adjusted_p_value, default["padj"]))
    w = next(np.array(v) for kind, v in case["tests"] if kind == "wald")
    cut = deseq2_wald_test(given_fit, w, cooks_cutoff=0.5)
    unfiltered = deseq2_wald_test(given_fit, w, independent_filtering=False)
    differ = nan_mismatches(cut.p_value, R["cutoff_pvalue"])
    differ += nan_mismatches(cut.adjusted_p_value, R["cutoff_padj"])
    differ += nan_mismatches(unfiltered.adjusted_p_value, R["unfiltered_padj"])
    add("options", differ, int(np.count_nonzero(cut.cooks_outlier)))
    add("options padj", max(log_relative(cut.adjusted_p_value, R["cutoff_padj"]),
                            log_relative(unfiltered.adjusted_p_value, R["unfiltered_padj"])))  # fmt: skip
    if "filters" in case:
        differ, largest = 0, 0.0
        for k in range(case["filters"].shape[1]):
            adjusted, _, theta, _, _, _ = _independent_filtering(
                R["tight1_pvalue"], case["filters"][:, k], 0.1
            )
            reference = R[f"filter{k + 1}_padj"]
            differ += nan_mismatches(adjusted, reference)
            differ += int(theta != R[f"filter{k + 1}_theta"][0])
            largest = max(largest, relative(adjusted, reference))
        add("filters", differ)
        add("filter padj", largest)
    if "PyDESeq2" in case:
        py = case["PyDESeq2"]
        end = deseq2_wald_test(fit, w)
        ratio = np.log(d.genewise[counted] / py["genewise_dispersions"][counted])
        add("py genewise", float(np.median(np.abs(ratio))))
        ratio = np.log(d.dispersion[counted] / py["dispersions"][counted])
        add("py final", float(np.median(np.abs(ratio))))
        add("py p", median_relative(end.p_value, py["pvalue"]))
        add("py calls", calls(end.adjusted_p_value, py["padj"]))


def compare_rules(name, case, found):
    """Cook's flags at given dispersions, under each design of one case."""
    for i, (formula, weights) in enumerate(case["formulas"], start=1):
        design = Design(case["samples"], formula, levels={"group": ["A", "B"]})
        dispersion = np.full(case["counts"].shape[1], 0.05)
        fit = fit_deseq2(case["counts"], design, size_factors=case["size_factors"],
                         dispersion=dispersion)  # fmt: skip
        result = deseq2_wald_test(fit, np.array(weights))
        reference = case["R"][f"pvalue{i}"]
        flagged = int(np.count_nonzero(np.isnan(reference)))
        label = f"{name}, {formula}"
        found.setdefault("rules", []).append(
            (label, nan_mismatches(result.p_value, reference), flagged)
        )
        cooks = relative(fit.max_cooks, case["R"][f"maxCooks{i}"], 1.0)
        found.setdefault("rules cooks", []).append((label, cooks, None))
        with np.errstate(invalid="ignore"):
            spared = (fit.max_cooks > result.cooks_cutoff) & ~result.cooks_outlier
        found.setdefault("rules spared", []).append(
            (label, int(np.count_nonzero(spared)), None)
        )


def report(found):
    """The table: one row per family of checks, over the cases it applies to."""
    table = []
    for key, what, reference, metric, tolerance, kind in FAMILIES:
        entries = found.pop(key, [])
        if not entries:
            table.append((what, reference, metric, "no cases", "n/a", "FAIL"))
            continue
        names, values, sizes = zip(*entries)
        noun = "tests" if key in TESTS else "designs" if key in RULES else "cases"
        cases = f"{len(entries)} {noun}" if len(entries) > 1 else names[0]
        if kind == COUNT:
            size = sum(sizes)
            genes = f"{size:,} gene{'s' if size != 1 else ''}"
            value, label = sum(values), f"{what} ({genes}, {cases})"
            ok = value <= tolerance and size > 0
            table.append((label, reference, metric, str(value), "0", "pass" if ok else "FAIL"))  # fmt: skip
            continue
        label = f"{what} ({cases})"
        if kind == INFORMATION:
            if isinstance(values[0], tuple):
                k = int(np.argmax([v[0] for v in values]))
                text = f"{sum(v[0] for v in values)} of {sum(v[1] for v in values)}"
                text += f", most in {names[k]}" if values[k][0] else ""
            elif isinstance(values[0], int):
                text = str(sum(values))
            elif len(values) > 1:
                k = int(np.argmax(values))
                text = f"{min(values):.1e} to {max(values):.1e} ({names[k]})"
            else:
                text = f"{values[0]:.1e}"
            table.append((label, reference, metric, text, "n/a", "information"))
            continue
        k = int(np.argmax([np.inf if np.isnan(v) else v for v in values]))
        text = f"{values[k]:.1e}" if isinstance(values[k], float) else str(values[k])
        if len(entries) > 1 and values[k] != 0:
            text += f" ({names[k]})"
        ok = not np.isnan(values[k]) and values[k] <= tolerance
        result = ("deliberate difference" if kind == DELIBERATE else "pass") if ok else "FAIL"  # fmt: skip
        shown = f"{tolerance:g}" if tolerance else "0"
        table.append((label, reference, metric, text, shown, result))
    if found:
        raise KeyError(f"checks without a family: {sorted(found)}")
    return table


def score_grid():
    """Counts, means and dispersions across the regimes of the derivative of the
    log-likelihood in log(alpha), with its value from mpmath (50 digits)."""
    import mpmath

    mpmath.mp.dps = 50
    points = [
        (y, mu, a)
        for y in (0, 1, 3, 40, 2000, 123456)
        for mu in (0.5, 2.0, 37.0, 3000.0)
        for a in (1e-8, 1e-6, 1e-4, 0.01, 0.0999, 0.1001, 0.7, 20.0)
    ]

    def loglik(y, mu, t):
        a = mpmath.exp(t)
        r = 1 / a
        return mpmath.loggamma(y + r) - mpmath.loggamma(r) - (y + r) * mpmath.log1p(a * mu) + y * t  # fmt: skip

    exact = [
        float(mpmath.diff(lambda t: loglik(y, mu, t), mpmath.log(a)))
        for y, mu, a in points
    ]
    return np.array(points, float), np.array(exact)


def score_check(found):
    points, exact = score_grid()
    ours = np.array([_count_score(*point) for point in points])
    # Relative to the larger of the score and the terms it is a difference of: for
    # dispersions up to 0.1, whose terms of order y and mu are cancelled analytically,
    # alpha ((y - mu)^2 + y), and above that terms of order 1.
    y, mu, alpha = points.T
    terms = np.where(alpha <= 0.1, alpha * ((y - mu) ** 2 + y), 1.0)
    value = float(np.max(np.abs(ours - exact) / np.maximum(np.abs(exact), terms)))
    found["score"] = [("mpmath", value, None)]
    return points, exact


FIXTURE_KEYS = (
    "size_factors allZero dispGeneEst dispFit dispMAP dispersion dispOutlier trend var_log "
    "prior_var rough mu_ref genewise_ref map_ref polished_coef polished_se tight_cooks "
    "tight_maxCooks tight_deviance cutoff_pvalue cutoff_padj unfiltered_padj"
).split()


def write_fixtures(cases, points, exact):
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for name, case in cases.items():
        arrays = {"counts": case["counts"].astype(np.int32)}
        arrays |= {f"samples {k}": np.asarray(v) for k, v in case["samples"].items()}
        if "formulas" in case:
            arrays |= dict(formulas=np.array(json.dumps(case["formulas"])),
                           size_factors=case["size_factors"])  # fmt: skip
        else:
            arrays |= {f"levels {k}": np.asarray(v) for k, v in case["levels"].items()}
            arrays |= dict(formula=np.array(case["formula"]),
                           tests=np.array(json.dumps(case["tests"])),
                           fit_type=np.array(case["R"]["fit_type"]))  # fmt: skip
        pattern = r"(tight|filter)\d+_\w+|maxCooks\d+|pvalue\d+"
        for k, v in case["R"].items():
            if not isinstance(v, str) and (
                k in FIXTURE_KEYS or re.fullmatch(pattern, k)
            ):
                arrays[f"R {k}"] = v
        if "mle_prior_variance" in case:
            arrays["mle_prior_variance"] = np.array(case["mle_prior_variance"])
        if "filters" in case:
            arrays["filters"] = case["filters"]
        np.savez_compressed(FIXTURES / f"{name}.npz", **arrays)
    np.savez_compressed(FIXTURES / "count_score.npz", points=points, mpmath=exact)
    print(f"wrote {len(cases) + 1} fixtures to {FIXTURES}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261025), 600)
    rules = cooks_rules(np.random.default_rng(17))
    if arguments.write_fixtures:
        small = make_cases(np.random.default_rng(7), 150)
        cases |= {f"{name}_fixture": small[name] for name in FIXTURE_CASES}
    with tempfile.TemporaryDirectory() as temporary:
        versions = run_references(cases | rules, Path(temporary))
    import mpmath

    print("References: " + ", ".join(versions) + f", mpmath {mpmath.__version__}")
    found = {}
    for name, case in cases.items():
        compare_case(name.replace("_fixture", " (fixture)"), case, found)
    for name, case in rules.items():
        compare_rules(name, case, found)
    points, exact = score_check(found)
    table = report(found)
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
        write_fixtures(small | rules, points, exact)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
