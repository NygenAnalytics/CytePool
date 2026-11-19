"""Check limma_voom and limma_trend against limma, group by group, with their F-tests.

Simulates cells (seeds in ``main``), pools them with ``from_cells``, and runs
``limma_voom`` and ``limma_trend`` on every cell type. R runs the workflow of the limma
and edgeR User's Guides on the same pseudobulk counts: per group, it leaves out the
samples with fewer than 10 cells or no counts, keeps their rows of the design
``model.matrix`` builds on the full sample table, and runs ``filterByExpr``,
``calcNormFactors``, ``voom`` (or ``cpm(log = TRUE, prior.count = 3)``), ``lmFit``,
``contrasts.fit``, ``eBayes`` and ``topTable(sort.by = "none")``, one contrast at a
time, so that a contrast limma refuses does not stop the others. voom runs with
``adaptive.span = TRUE`` where ``limma_voom`` chooses the span from the number of
genes, as limma's voom does by default since limma 3.66, and with ``span = 0.5,
adaptive.span = FALSE`` in the case that fixes it (``span=0.5``).

F-tests: for every F-test, R runs ``contrasts.fit`` of its contrasts alone, ``eBayes``,
``topTable(coef = ...)`` of all of them and ``classifyTestsF(fstat.only = TRUE)``,
whose ``df1`` and ``df2`` are the numerator and denominator degrees of freedom; an
F-test of coefficients runs ``eBayes`` on ``lmFit``'s fit and ``topTable`` of those
coefficients, limma's own path. For limma-voom, R also computes the exact F-statistic,
which limma does not: per gene, ``lm.wfit`` of the group's design and of the design
restricted by C' b = 0, with voom's weights, gives F = (RSS0 - RSS1) / (r s2.post),
with r the rank of the contrasts C and ``s2.post`` from ``eBayes``, on r and
``df.residual + df.prior`` degrees of freedom. ``limma_voom`` with its default, exact,
standard errors is compared with these.

Cases: a paired design ``~ donor + condition`` (6 donors x 2 conditions, 4 cell types,
1,000 genes) by limma-voom and by limma-trend, with the F-tests
``condition_and_donor`` (``stim_vs_ctrl`` and ``D2_vs_D6``) and ``donors``
(``D2_vs_D1`` and ``D3_vs_D2``); an unpaired design with a covariate, ``~ condition +
age`` (10 donors, 3 cell types), by limma-voom with robust moderation and a span fixed
at 0.5, and by limma-trend without filtering or normalization factors, with the F-test
``condition_and_age``; and a dose series (seed 20261119), the vehicle and four doses
(0.1, 0.3, 1 and 3) x 3 samples in 2 cell types and 1,000 genes, of which a fifth
respond along a saturating curve. The dose series is tested as dose groups, ``~
dose_group`` with each dose against the vehicle and ``d0.3_vs_d0.1``, with the F-tests
``any_dose`` (the four doses against the vehicle) and ``any_dose_redundant`` (all five
contrasts, of rank 4), by limma-voom with its default and with limma's standard errors
and robust moderation, ``any_dose`` against ``topTable(coef = 2:5)`` of ``lmFit``; and
as a natural spline, ``~ dose_s1 + dose_s2 + dose_s3`` with the columns of
``natural_spline_basis(dose, df=3)``, against R's ``~ ns(dose, df = 3)``: limma-trend
against ``topTable(coef = 2:4)``, and limma-voom against the nested fits. Some sample x
cell type entries have fewer than 10 cells. In the NK cells both samples of donor D1,
the reference level, are left out, so the column ``donorD6`` is a combination of the
others; in the DC cells four samples remain, too few for residual degrees of freedom.

Per group, one row counts mismatches in the samples used, the features tested, the
coefficients limma cannot estimate, the contrasts tested by one program only (other
than the deliberate differences), voom's span and the residual and prior degrees of
freedom; one gives the largest difference in logFC, AveExpr, t, P.Value, adj.P.Val and
B over the contrasts both compute, and in the moderated standard error of logFC,
``sqrt(s2.post) * stdev.unscaled``; F-tests get rows of their own for the logFC of
each contrast, AveExpr, F, P.Value, adj.P.Val and ``df2``, for the numerator degrees
of freedom, and for the nested fits. Differences are scaled by max(|reference|, 1),
and for p-values by max(p, 1e-300) max(1, |log p|). Tiers: exact (1e-10); close
(1e-6) with robust moderation, which limma solves only to a tolerance. limma-voom runs
with ``contrast_standard_errors="limma"`` against ``topTable``, except where the
contrasts are single coefficients, for which the default exact standard errors are
limma's; with the default, its F-tests use each gene's own correlation of the
contrasts, which limma does not, and are compared with the nested fits.

Deliberate differences, in the NK cells: limma's ``contrasts.fit`` refuses
``D2_vs_D6`` (``donorD2 - donorD6``), which is estimable but weighs the aliased
column; CytePool computes it, and its values are checked against limma's values for
``D2_vs_D1`` (``donorD2``), which limma computes from the kept columns, where it
means D2 against D6. CytePool reports ``D2_vs_D1`` as not estimable. So limma refuses
the F-test ``condition_and_donor``, which CytePool runs and which is checked against
limma's F-test of ``stim_vs_ctrl`` and ``D2_vs_D1``, and computes ``donors``, which
CytePool does not run.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree.
    R 4.4 or newer with the Bioconductor packages limma and edgeR, and jsonlite. Set
    ``RSCRIPT`` to the Rscript executable when it is not on PATH.

Usage:
    PYTHONPATH=. python validation/differential_expression.py
    PYTHONPATH=. python validation/differential_expression.py --write-fixtures
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cytepool import Design, from_cells, limma_trend, limma_voom
from cytepool.stats.smoothing import natural_spline_basis

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "differential_expression"
)
EXACT = 1e-10
CLOSE = 1e-6
# topTable's columns and CytePool's, in the same order, then the moderated standard
# error of logFC, sqrt(s2.post) * stdev.unscaled from eBayes.
COLUMNS = {
    "logFC": "log2_fold_change",
    "AveExpr": "average_log2_cpm",
    "t": "t",
    "P.Value": "p_value",
    "adj.P.Val": "adjusted_p_value",
    "B": "log_odds",
    "SE": "standard_error",
}
# An F-test's columns after the logFC of each contrast: topTable's AveExpr, F, P.Value
# and adj.P.Val, and classifyTestsF's df2, with CytePool's names.
F_COLUMNS = {
    "AveExpr": "average_log2_cpm",
    "F": "f_statistic",
    "P.Value": "p_value",
    "adj.P.Val": "adjusted_p_value",
    "df2": "df",
}
DOSES = (0.0, 0.1, 0.3, 1.0, 3.0)
SPLINE = ["dose_s1", "dose_s2", "dose_s3"]
LIMMA = "contrast_standard_errors"

R_SCRIPT = r"""
suppressPackageStartupMessages({library(limma); library(edgeR); library(splines)})
root <- commandArgs(trailingOnly = TRUE)[[1]]
rd <- function(path) readBin(path, "double", n = file.size(path) / 8, size = 8)
wr <- function(x, dir, name) writeBin(as.double(x), file.path(dir, name), size = 8)
# Runs expr without warnings; returns NULL or the error as "<step>: <message>". lmFit
# prints "Coefficients not estimable" to standard output (found by probing).
run <- function(expr) {
  capture.output(res <- tryCatch(suppressWarnings({ expr; NULL }),
                                 error = function(e) paste0(step, ": ", conditionMessage(e))))
  res
}
# The contrasts of an F-test: coefficients by position, or contrasts by name.
f_contrasts <- function(kind, members, X, cs) {
  if (kind == "coefficients") return(diag(ncol(X))[, as.integer(members), drop = FALSE])
  C <- makeContrasts(contrasts = cs$expression[match(members, cs$name)], levels = X)
  rownames(C) <- colnames(X)  # makeContrasts renames (Intercept)
  C
}
for (case in readLines(file.path(root, "cases.txt"))) {
  dir <- file.path(root, case)
  nf <- length(readLines(file.path(dir, "features.txt")))
  ng <- length(readLines(file.path(dir, "groups.txt")))
  levs <- jsonlite::fromJSON(file.path(dir, "levels.json"))
  # Factor columns are read as text, so that levels such as "01" keep their spelling.
  samples <- read.delim(file.path(dir, "samples.tsv"), check.names = FALSE,
                        colClasses = setNames(rep("character", length(levs)), names(levs)))
  for (v in names(levs)) samples[[v]] <- factor(samples[[v]], levels = levs[[v]])
  # One design for all samples, so that every group has the same columns.
  X_full <- model.matrix(as.formula(readLines(file.path(dir, "formula.txt"))), samples)
  cs <- read.delim(file.path(dir, "contrasts.tsv"), colClasses = "character")
  ft <- read.delim(file.path(dir, "f_tests.tsv"), colClasses = "character")
  set <- jsonlite::fromJSON(file.path(dir, "settings.json"))
  counts <- array(rd(file.path(dir, "counts.bin")), dim = c(nf, nrow(samples), ng))
  n_cells <- matrix(rd(file.path(dir, "n_cells.bin")), nrow = nrow(samples), ncol = ng)
  for (g in seq_len(ng)) {
    out <- file.path(dir, paste0("group", g - 1))
    dir.create(out)
    y <- matrix(counts[, , g], nrow = nf)
    used <- n_cells[, g] >= set$min_cells & colSums(y) > 0
    X <- X_full[used, , drop = FALSE]
    wr(used, out, "used.bin")
    not_estimable <- character(0)
    # Found by probing: without residual df, voom and lmFit return values, eBayes stops.
    step <- "DGEList"
    status <- run({
      dge <- DGEList(y[, used, drop = FALSE])
      step <- "filterByExpr"
      keep <- if (set$filter) filterByExpr(dge, design = X) else rep(TRUE, nf)
      wr(keep, out, "keep.bin")
      dge <- dge[keep, , keep.lib.sizes = FALSE]
      step <- "calcNormFactors"
      if (set$normalization == "TMM") dge <- calcNormFactors(dge)
      if (set$method == "voom") {
        # adaptive.span is pinned either way, as its default changed in limma 3.66.
        # The result holds the span voom chose, and none when given one (found by
        # probing).
        step <- "voom"
        E <- if (is.null(set$span)) voom(dge, X, adaptive.span = TRUE) else
          voom(dge, X, span = set$span, adaptive.span = FALSE)
        wr(if (is.null(E$span)) set$span else E$span, out, "span.bin")
      } else {
        step <- "cpm"; E <- cpm(dge, log = TRUE, prior.count = set$prior_count)
      }
      step <- "lmFit"; fit <- lmFit(E, X)
      # Coefficients that lmFit cannot estimate are NA for every gene.
      not_estimable <- colnames(fit$coefficients)[colSums(!is.na(fit$coefficients)) == 0]
      wr(fit$df.residual, out, "df_residual.bin")
    })
    writeLines(if (is.null(status)) "fitted" else status, file.path(out, "status.txt"))
    writeLines(not_estimable, file.path(out, "not_estimable.txt"))
    # The prior and posterior variances, which no contrast changes; eBayes stops for a
    # group without residual df, as each contrast's does below.
    posterior <- NULL
    if (is.null(status)) {
      step <- "eBayes"
      run({
        posterior <- eBayes(fit, trend = set$method == "trend", robust = set$robust)
        wr(posterior$df.prior, out, "df_prior.bin")
      })
    }
    # Found by probing: contrasts.fit stops for a contrast of an NA coefficient.
    if (is.null(status)) for (j in seq_len(nrow(cs))) {
      step <- "makeContrasts"
      error <- run({
        C <- makeContrasts(contrasts = cs$expression[j], levels = X)
        rownames(C) <- colnames(X)  # makeContrasts renames (Intercept)
        step <- "contrasts.fit"; cf <- contrasts.fit(fit, C)
        step <- "eBayes"
        eb <- eBayes(cf, trend = set$method == "trend", robust = set$robust)
        tt <- topTable(eb, coef = 1, number = Inf, sort.by = "none")
        se <- sqrt(eb$s2.post) * eb$stdev.unscaled[, 1]
        wr(cbind(as.matrix(tt[, c("logFC", "AveExpr", "t", "P.Value", "adj.P.Val", "B")]),
                 se), out, sprintf("contrast%d.bin", j - 1))
      })
      if (!is.null(error)) {
        writeLines(error, file.path(out, sprintf("contrast%d_error.txt", j - 1)))
      }
    }
    if (is.null(status)) for (k in seq_len(nrow(ft))) {
      members <- strsplit(ft$members[k], ";", fixed = TRUE)[[1]]
      step <- "makeContrasts"
      error <- run({
        C <- f_contrasts(ft$kind[k], members, X, cs)
        step <- "contrasts.fit"
        cf <- if (ft$kind[k] == "coefficients") fit else contrasts.fit(fit, C)
        cols <- if (ft$kind[k] == "coefficients") as.integer(members) else seq_along(members)
        step <- "eBayes"
        eb <- eBayes(cf, trend = set$method == "trend", robust = set$robust)
        step <- "classifyTestsF"
        cl <- classifyTestsF(eb[, cols], fstat.only = TRUE)
        df2 <- rep_len(as.numeric(attr(cl, "df2")), nrow(eb))
        if (is.null(eb$F)) {
          # Found by probing: for a design that lost a column, eBayes computes no
          # F-statistic and topTable stops; classifyTestsF computes it, and its
          # p-value is that of eBayes's F.p.value elsewhere, gene by gene.
          p <- pf(as.numeric(cl), attr(cl, "df1"), df2, lower.tail = FALSE)
          values <- cbind(eb$coefficients[, cols, drop = FALSE], eb$Amean,
                          as.numeric(cl), p, p.adjust(p, "BH"))
          route <- "classifyTestsF"
        } else {
          step <- "topTable"
          tt <- topTable(eb, coef = cols, number = Inf, sort.by = "none")
          values <- cbind(as.matrix(tt[, seq_along(cols)]),
                          as.matrix(tt[, c("AveExpr", "F", "P.Value", "adj.P.Val")]))
          route <- "topTable"
        }
        wr(cbind(values, df2), out, sprintf("ftest%d.bin", k - 1))
        wr(attr(cl, "df1"), out, sprintf("ftest%d_df1.bin", k - 1))
        writeLines(route, file.path(out, sprintf("ftest%d_route.txt", k - 1)))
      })
      if (!is.null(error)) {
        writeLines(error, file.path(out, sprintf("ftest%d_error.txt", k - 1)))
      }
      if (set$method != "voom") next
      # The exact F-statistic: nested weighted least-squares fits of each gene, with
      # voom's weights and limma's posterior variance.
      step <- "nested fits"
      error <- run({
        stopifnot(!is.null(posterior))
        C <- f_contrasts(ft$kind[k], members, X, cs)
        r <- qr(C)$rank
        free <- qr.Q(qr(C), complete = TRUE)[, -seq_len(r), drop = FALSE]
        X0 <- X %*% free
        rss <- function(Z, i) {
          w <- E$weights[i, ]
          sum(w * lm.wfit(Z, E$E[i, ], w)$residuals^2)
        }
        f_nested <- vapply(seq_len(nrow(E)), function(i) {
          (rss(X0, i) - rss(X, i)) / (r * posterior$s2.post[i])
        }, 0)
        df2 <- posterior$df.residual + posterior$df.prior
        wr(cbind(f_nested, pf(f_nested, r, df2, lower.tail = FALSE)), out,
           sprintf("nested%d.bin", k - 1))
      })
      if (!is.null(error)) {
        writeLines(error, file.path(out, sprintf("nested%d_error.txt", k - 1)))
      }
    }
  }
}
writeLines(c(paste("R", getRversion()), paste("limma", packageVersion("limma")),
             paste("edgeR", packageVersion("edgeR"))), file.path(root, "versions.txt"))
"""


@dataclass
class Case:
    """An analysis of a cube, and what R runs and compares on the same counts."""

    cube: Any
    design: Design
    function: Any
    options: dict[str, Any]
    # (group, contrast): the contrast limma computes in place of one it refuses.
    pairs: dict[tuple[str, str], str] = field(default_factory=dict)
    f_tests: dict[str, list[str]] = field(default_factory=dict)
    # (group, F-test): the contrasts limma tests in place of an F-test it refuses.
    f_pairs: dict[tuple[str, str], list[str]] = field(default_factory=dict)
    # R's formula, contrasts and F-tests where they differ from CytePool's; an F-test
    # is (kind, members), with kind "contrasts" or "coefficients" (positions from 1).
    r_formula: str | None = None
    r_contrasts: dict[str, str] | None = None
    r_f_tests: dict[str, tuple[str, list[Any]]] | None = None
    # Whether CytePool's F-tests are limma's, compared at the case's tier, or only
    # reported beside limma's (information).
    f_against_limma: bool = True

    @property
    def tolerance(self) -> float:
        return CLOSE if self.options.get("robust") else EXACT

    def r_spec(self) -> tuple[str, dict[str, str], dict[str, tuple[str, list[Any]]]]:
        """R's formula, contrasts by name and F-tests by name."""
        contrasts = self.r_contrasts
        if contrasts is None:
            contrasts = {n: _expression(self.design, n) for n in self.design.contrasts}
        tests = self.r_f_tests
        if tests is None:
            tests = {name: ("contrasts", m) for name, m in self.f_tests.items()}
            tests |= {
                f"{name} as limma reads it": ("contrasts", members)
                for (_, name), members in self.f_pairs.items()
            }
        return self.r_formula or self.design.formula, contrasts, tests


def simulate_cells(rng, samples, cells, n_genes):
    """Cells x genes counts of every sample and cell type, with the cells' labels.

    Each cell type has its own log-normal expression profile. A donor shifts every gene
    of every cell type, stimulation changes a tenth of the genes of each cell type, and
    age a twentieth of them. Cells differ in depth (1,000 to 5,000 reads) and by gamma
    noise around their profile.
    """
    n = len(samples["sample_id"])
    donor = {d: rng.normal(0.0, 0.3, n_genes) for d in sorted(set(samples["donor"]))}
    age = np.asarray(samples.get("age", np.full(n, 50.0)), dtype=float)
    blocks, sample_labels, type_labels = [], [], []
    for cell_type, numbers in cells.items():
        profile = rng.normal(0.0, 1.8, n_genes)
        stim = np.where(rng.random(n_genes) < 0.1, rng.normal(0.0, 1.2, n_genes), 0.0)
        slope = np.where(rng.random(n_genes) < 0.05, rng.normal(0.0, 0.4, n_genes), 0.0)
        for s in range(n):
            log_mean = profile + donor[samples["donor"][s]] + slope * (age[s] - 50) / 25
            log_mean = log_mean + stim * (samples["condition"][s] == "stim")
            blocks.append(_cell_counts(rng, log_mean, numbers[s]))
            sample_labels += [samples["sample_id"][s]] * numbers[s]
            type_labels += [cell_type] * numbers[s]
    return np.vstack(blocks).astype(np.uint16), sample_labels, type_labels


def simulate_dose_cells(rng, samples, cells, n_genes):
    """Cells x genes counts of a dose series, with the cells' labels.

    Each cell type has its own log-normal expression profile; a fifth of its genes
    respond along a saturating curve, log fold change E d / (d + EC50) at dose d, with
    E normal (SD 1.5) and EC50 between 0.05 and 1. Each sample shifts every gene by
    normal noise (SD 0.2), its biological variation; cells are drawn as in
    ``simulate_cells``.
    """
    n = len(samples["sample_id"])
    blocks, sample_labels, type_labels = [], [], []
    for cell_type, numbers in cells.items():
        profile = rng.normal(0.0, 1.8, n_genes)
        top = np.where(rng.random(n_genes) < 0.2, rng.normal(0.0, 1.5, n_genes), 0.0)
        ec50 = rng.uniform(0.05, 1.0, n_genes)
        for s in range(n):
            dose = samples["dose"][s]
            log_mean = profile + top * dose / (dose + ec50)
            log_mean = log_mean + rng.normal(0.0, 0.2, n_genes)
            blocks.append(_cell_counts(rng, log_mean, numbers[s]))
            sample_labels += [samples["sample_id"][s]] * numbers[s]
            type_labels += [cell_type] * numbers[s]
    return np.vstack(blocks).astype(np.uint16), sample_labels, type_labels


def _cell_counts(rng, log_mean, n_cells):
    """Counts of cells around one profile: depth 1,000 to 5,000, gamma noise."""
    proportion = np.exp(log_mean) / np.exp(log_mean).sum()
    depth = rng.uniform(1000.0, 5000.0, n_cells)[:, np.newaxis]
    rate = rng.gamma(4.0, 0.25, (n_cells, log_mean.size)) * depth * proportion
    return rng.poisson(rate)


def pooled(rng, samples, cells, n_genes, simulate=simulate_cells):
    """The cube that from_cells pools from simulated cells, and the cells."""
    counts, labels, types = simulate(rng, samples, cells, n_genes)
    cube = from_cells(
        counts,
        sample_labels=labels,
        group_labels=types,
        samples=samples,
        layers="counts",
    )
    return cube, {"counts": counts, "samples": labels, "groups": types}


def paired_samples(n_donors):
    donors = [f"D{d}" for d in range(1, n_donors + 1)]
    return {
        "sample_id": [f"{d}_{c}" for d in donors for c in ("ctrl", "stim")],
        "donor": [d for d in donors for _ in range(2)],
        "condition": ["ctrl", "stim"] * n_donors,
    }


def paired_design(cube, last, more=()):
    contrasts = {
        "stim_vs_ctrl": "conditionstim",
        "D2_vs_D1": "donorD2",
        f"D2_vs_{last}": f"donorD2 - donor{last}",
    } | dict(more)
    levels = {"condition": ["ctrl", "stim"]}
    return Design(
        cube.samples, "~ donor + condition", contrasts=contrasts, levels=levels
    )


def dose_samples(replicates):
    """The vehicle and four doses, each with its replicates."""
    doses = [d for d in DOSES for _ in range(replicates)]
    return {
        "sample_id": [f"d{d:g}_{k + 1}" for d in DOSES for k in range(replicates)],
        "dose": doses,
        "dose_group": [f"{d:g}" for d in doses],
    }


def spline_design(cube):
    """~ dose_s1 + dose_s2 + dose_s3, the columns of natural_spline_basis(dose, df=3)
    on every sample, tested together as the F-test "dose"."""
    basis = natural_spline_basis(np.asarray(cube.samples["dose"], float), df=3)
    samples = cube.samples.with_columns(dict(zip(SPLINE, basis.T, strict=True)))
    return Design(samples, "~ " + " + ".join(SPLINE), contrasts=SPLINE)


def make_cases(rng, dose_rng):
    """Analyses by name."""
    bounds = {"T": (40, 120), "B": (15, 60), "NK": (12, 40), "DC": (0, 9)}
    cells = {name: rng.integers(low, high, 12) for name, (low, high) in bounds.items()}
    cells["B"][[7, 10]] = [6, 0]  # D4_stim and D6_ctrl
    cells["NK"][[0, 1]] = [4, 7]  # both samples of D1, the reference donor
    cells["DC"][[2, 3, 4, 9]] = [12, 15, 11, 14]  # D2_ctrl, D2_stim, D3_ctrl, D5_stim
    cube, _ = pooled(rng, paired_samples(6), cells, 1000)
    # D3_vs_D2 weighs two columns that the NK cells keep.
    paired = paired_design(cube, "D6", [("D3_vs_D2", "donorD3 - donorD2")])
    # Without D1's samples, donorD6 = (Intercept) - donorD2 - ... - donorD5 in the NK
    # cells, and limma's estimate for donorD2 is that of donorD2 - donorD6.
    pairs = {("NK", "D2_vs_D6"): "D2_vs_D1"}
    f_tests = {
        "condition_and_donor": ["stim_vs_ctrl", "D2_vs_D6"],
        "donors": ["D2_vs_D1", "D3_vs_D2"],
    }
    f_pairs = {("NK", "condition_and_donor"): ["stim_vs_ctrl", "D2_vs_D1"]}
    donors = [f"U{d:02d}" for d in range(1, 11)]
    unpaired = {"sample_id": donors, "donor": donors, "condition": ["ctrl", "stim"] * 5}
    unpaired["age"] = rng.uniform(25.0, 70.0, 10).round(1).tolist()
    bounds = {"T": (30, 90), "B": (10, 50), "Mono": (10, 40)}
    cells = {name: rng.integers(low, high, 10) for name, (low, high) in bounds.items()}
    cells["B"][2] = 5
    cells["Mono"][[5, 8]] = [3, 8]
    other, _ = pooled(rng, unpaired, cells, 1000)
    contrasts = {"stim_vs_ctrl": "conditionstim", "age": "age"}
    with_age = Design(other.samples, "~ condition + age", contrasts=contrasts)
    age_test = {"condition_and_age": ["stim_vs_ctrl", "age"]}
    unfiltered = {"expression_filter": False, "normalization": None}
    robust, limma = {"robust": True, "span": 0.5}, {LIMMA: "limma"}
    paired_tests = {"pairs": pairs, "f_tests": f_tests, "f_pairs": f_pairs}
    cases = {
        "paired, limma-voom": Case(cube, paired, limma_voom, limma, **paired_tests),
        "paired, limma-trend": Case(cube, paired, limma_trend, {}, **paired_tests),
        # The contrasts are single coefficients, but the F-test combines two whose
        # correlation changes with voom's weights, so its default is not limma's.
        "unpaired, limma-voom, robust, span 0.5": Case(
            other, with_age, limma_voom, robust, f_tests=age_test, f_against_limma=False
        ),
        "unpaired, limma-trend, unfiltered, unnormalized": Case(
            other, with_age, limma_trend, unfiltered, f_tests=age_test
        ),
    }
    return cases | dose_cases(dose_rng)


def dose_cases(rng):
    """The dose series as dose groups and as a natural spline."""
    cells = {
        "hepatocyte": rng.integers(30, 90, 15),
        "kupffer": rng.integers(12, 40, 15),
    }
    cells["kupffer"][7] = 6  # one sample at dose 0.3
    cube, _ = pooled(rng, dose_samples(3), cells, 1000, simulate_dose_cells)
    groups = {f"d{d:g}_vs_0": f"dose_group{d:g}" for d in DOSES[1:]}
    groups["d0.3_vs_d0.1"] = "dose_group0.3 - dose_group0.1"
    by_group = Design(
        cube.samples,
        "~ dose_group",
        levels={"dose_group": [f"{d:g}" for d in DOSES]},
        contrasts=groups,
    )
    any_dose = list(groups)[:4]
    tests = {"any_dose": any_dose, "any_dose_redundant": list(groups)}
    # limma's own path: the F-test of coefficients 2 to 5 of lmFit's fit.
    own = {
        "any_dose": ("coefficients", [2, 3, 4, 5]),
        "any_dose_redundant": ("contrasts", list(groups)),
    }
    spline = spline_design(cube)
    # R's own natural spline: no contrasts, and the F-test of its three columns.
    on_ns = {
        "r_formula": "~ ns(dose, df = 3)",
        "r_contrasts": {},
        "r_f_tests": {"dose": ("coefficients", [2, 3, 4])},
    }
    return {
        "dose groups, limma-voom": Case(
            cube, by_group, limma_voom, {}, f_tests=tests, f_against_limma=False
        ),
        "dose groups, limma-voom, limma's standard errors, robust": Case(
            cube,
            by_group,
            limma_voom,
            {LIMMA: "limma", "robust": True},
            f_tests=tests,
            r_f_tests=own,
        ),
        "dose spline, limma-trend": Case(
            cube, spline, limma_trend, {}, f_tests={"dose": SPLINE}, **on_ns
        ),
        "dose spline, limma-voom": Case(
            cube,
            spline,
            limma_voom,
            {},
            f_tests={"dose": SPLINE},
            f_against_limma=False,
            **on_ns,
        ),
    }


def fixture_cases(rng, spline_rng):
    """The small study whose cells the tests pool, its three analyses, and the dose
    series of the spline fixture with its two."""
    cells = {"T": rng.integers(15, 30, 8), "NK": rng.integers(12, 25, 8)}
    cells["T"][5] = 4  # D3_stim
    cells["NK"][[0, 1]] = [3, 5]  # both samples of D1
    samples = paired_samples(4)
    cube, inputs = pooled(rng, samples, cells, 120)
    inputs |= {f"sample_table {k}": np.array(v) for k, v in samples.items()}
    design, pairs = paired_design(cube, "D4"), {("NK", "D2_vs_D4"): "D2_vs_D1"}
    tests = {
        "pairs": pairs,
        "f_tests": {
            "condition_and_donor": ["stim_vs_ctrl", "D2_vs_D4"],
            "donors": ["D2_vs_D1", "D2_vs_D4"],
        },
        "f_pairs": {("NK", "condition_and_donor"): ["stim_vs_ctrl", "D2_vs_D1"]},
    }
    unfiltered = {"expression_filter": False, "normalization": None, "robust": True}
    limma = {LIMMA: "limma"}
    robust = limma | {"robust": True, "span": 0.5}
    spline_cube, _ = pooled(
        spline_rng,
        dose_samples(2),
        {"hepatocyte": spline_rng.integers(20, 60, 10)},
        150,
        simulate_dose_cells,
    )
    spline = spline_design(spline_cube)
    on_ns = {
        "f_tests": {"dose": SPLINE},
        "r_formula": "~ ns(dose, df = 3)",
        "r_contrasts": {},
        "r_f_tests": {"dose": ("coefficients", [2, 3, 4])},
    }
    return inputs, {
        "fixture voom": Case(cube, design, limma_voom, limma, **tests),
        "fixture robust": Case(cube, design, limma_voom, robust, **tests),
        "fixture trend": Case(cube, design, limma_trend, unfiltered, **tests),
        "fixture dose spline, limma-trend": Case(
            spline_cube, spline, limma_trend, {}, **on_ns
        ),
        "fixture dose spline, limma-voom": Case(
            spline_cube, spline, limma_voom, {}, f_against_limma=False, **on_ns
        ),
    }


def run_limma(cases):
    """Write every case for R, run limma, and read its outputs back."""
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        names = {name: f"case{i}" for i, name in enumerate(cases)}
        for name, case in cases.items():
            cube, design, function, options = (
                case.cube,
                case.design,
                case.function,
                case.options,
            )
            directory = root / names[name]
            directory.mkdir()
            formula, contrasts, f_tests = case.r_spec()
            columns = list(design.samples.columns)
            rows = zip(*(design.samples[c].tolist() for c in columns), strict=True)
            normalized = options.get("normalization", "tmm") is not None
            settings = {
                "method": "voom" if function is limma_voom else "trend",
                "min_cells": 10,
                "filter": options.get("expression_filter", True),
                "normalization": "TMM" if normalized else "none",
                "robust": options.get("robust", False),
                "prior_count": 3,
                "span": options.get("span"),
            }
            levels = {k: list(v) for k, v in design.levels.items()}
            texts = {
                "groups.txt": cube.groups["group_id"],
                "features.txt": cube.features["feature_id"],
                "samples.tsv": ["\t".join(map(str, row)) for row in [columns, *rows]],
                "contrasts.tsv": ["name\texpression"]
                + [f"{k}\t{v}" for k, v in contrasts.items()],
                "f_tests.tsv": ["name\tkind\tmembers"]
                + [
                    f"{k}\t{kind}\t{';'.join(map(str, members))}"
                    for k, (kind, members) in f_tests.items()
                ],
                "formula.txt": [formula],
                "levels.json": [json.dumps(levels)],
                "settings.json": [json.dumps(settings)],
            }
            for file, lines in texts.items():
                (directory / file).write_text("\n".join(lines) + "\n")
            arrays = {"counts.bin": cube.layer("counts"), "n_cells.bin": cube.n_cells.T}
            for file, values in arrays.items():
                values = np.ascontiguousarray(values, dtype="<f8")
                (directory / file).write_bytes(values.tobytes())
        (root / "cases.txt").write_text("\n".join(names.values()) + "\n")
        (root / "compare.R").write_text(R_SCRIPT)
        rscript = os.environ.get("RSCRIPT", "Rscript")
        subprocess.run(
            [rscript, "--vanilla", str(root / "compare.R"), str(root)], check=True
        )
        outputs = {
            name: [
                _read_group(root / names[name] / f"group{k}", case)
                for k in range(case.cube.n_groups)
            ]
            for name, case in cases.items()
        }
        return outputs, (root / "versions.txt").read_text().split("\n")[:3]


def _expression(design, name):
    """A contrast of the design as an expression over its columns."""
    weights = zip(design.contrasts[name], design.columns, strict=True)
    return " ".join(f"{w:+.17g}*{column}" for w, column in weights if w != 0.0)


def _read_group(directory, case):
    out = {
        "status": (directory / "status.txt").read_text().strip(),
        "not_estimable": (directory / "not_estimable.txt").read_text().split(),
        "contrasts": {},
        "f_tests": {},
        "df1": {},
        "routes": {},
        "nested": {},
    }
    for name in ("used", "keep", "df_residual", "df_prior", "span"):
        if (directory / f"{name}.bin").exists():
            out[name] = np.fromfile(directory / f"{name}.bin", dtype="<f8")
    _, contrasts, f_tests = case.r_spec()
    for j, name in enumerate(contrasts):
        if (directory / f"contrast{j}_error.txt").exists():
            out["contrasts"][name] = (directory / f"contrast{j}_error.txt").read_text()
        elif (directory / f"contrast{j}.bin").exists():
            values = np.fromfile(directory / f"contrast{j}.bin", dtype="<f8")
            out["contrasts"][name] = values.reshape(len(COLUMNS), -1)
    for k, (name, (_, members)) in enumerate(f_tests.items()):
        if (directory / f"ftest{k}_error.txt").exists():
            out["f_tests"][name] = (directory / f"ftest{k}_error.txt").read_text()
        elif (directory / f"ftest{k}.bin").exists():
            values = np.fromfile(directory / f"ftest{k}.bin", dtype="<f8")
            out["f_tests"][name] = values.reshape(len(members) + len(F_COLUMNS), -1)
            df1 = np.fromfile(directory / f"ftest{k}_df1.bin", dtype="<f8")
            out["df1"][name] = int(df1[0])
            out["routes"][name] = (
                (directory / f"ftest{k}_route.txt").read_text().strip()
            )
        if (directory / f"nested{k}_error.txt").exists():
            out["nested"][name] = (directory / f"nested{k}_error.txt").read_text()
        elif (directory / f"nested{k}.bin").exists():
            values = np.fromfile(directory / f"nested{k}.bin", dtype="<f8")
            out["nested"][name] = values.reshape(2, -1)
    return out


def difference(ours, reference, p_value=False):
    """Largest |ours - reference| / scale, as the module docstring describes; equal
    values, infinities included, differ by 0."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    if ours.shape != reference.shape:
        return np.inf
    if p_value:
        scale = np.maximum(reference, 1e-300)
        scale = scale * np.maximum(1.0, np.abs(np.log(scale)))
    else:
        scale = np.maximum(np.abs(reference), 1.0)
    with np.errstate(invalid="ignore"):
        gap = np.where(ours == reference, 0.0, np.abs(ours - reference) / scale)
    return float(np.max(gap, initial=0.0))


def largest(table, reference):
    """The largest difference of a CytePool table from a topTable, over its columns."""
    return max(
        difference(table[column], reference[i], p_value="P" in name)
        for i, (name, column) in enumerate(COLUMNS.items())
    )


def f_largest(table, reference, members, estimates=True):
    """The largest difference of an F-test's table from limma's: the logFC of each
    contrast (when the programs test the same contrasts), AveExpr, F, P.Value,
    adj.P.Val and df2."""
    m = len(members)
    found = [
        difference(table[f"log2_fold_change_{c}"], reference[i])
        for i, c in enumerate(members)
        if estimates
    ]
    found += [
        difference(table[column], reference[m + i], p_value="P" in name)
        for i, (name, column) in enumerate(F_COLUMNS.items())
    ]
    return max(found)


def compare(name, case, result, limma):
    """Rows of (check, reference, metric, value, tolerance, result) for one case."""
    cube, pairs = case.cube, case.pairs
    tolerance = case.tolerance
    passed = {True: "pass", False: "FAIL"}
    rows = []
    for k, group in enumerate(cube.groups["group_id"]):
        theirs, record = limma[k], result.provenance["groups"][k]
        label = f"{name}, {group}"
        used = result.samples["used"][result.samples["group_id"] == group]
        if record["status"] != "fitted":
            # Both must refuse every contrast, for the same samples.
            errors = list(theirs["contrasts"].values()) or [theirs["status"]]
            refused = all(isinstance(e, str) for e in errors)
            same = refused and np.array_equal(used, theirs["used"].astype(bool))
            first = errors[0].strip() if refused else "values"
            value = f"limma: {first}; CytePool: {record['status']}"
            rows.append((label, "limma", "behaviour", value, "n/a", passed[same]))
            continue
        tables = {
            c: t
            for (g, c), t in result.tables.items()
            if g == group and c in result.contrasts.names
        }
        ids = next(iter(tables.values()))["feature_id"]
        tested = np.isin(cube.features["feature_id"], ids)
        row = list(result.groups).index(group)
        df_prior = result.summary["df_prior"][row]
        prior = float(np.median(theirs["df_prior"]))
        # A contrast that one program tests and the other does not, other than the
        # deliberate differences, among those R tests.
        listed = {
            name
            for (where, contrast), meant in pairs.items()
            if where == group
            for name in (contrast, meant)
        }
        computed = {c for c, v in theirs["contrasts"].items() if not isinstance(v, str)}
        compared = set(tables) if case.r_contrasts is None else set(case.r_contrasts)
        mismatches = (
            np.count_nonzero(used != theirs["used"].astype(bool))
            + np.count_nonzero(tested != theirs["keep"].astype(bool))
            + len(set(record["aliased_columns"]) ^ set(theirs["not_estimable"]))
            + len(((set(tables) & compared) ^ computed) - listed)
            + (result.summary["df_residual"][row] != theirs["df_residual"][0])
            + (df_prior != prior and difference(df_prior, prior) > tolerance)
            # voom's span: limma's, or none for limma-trend.
            + (
                record.get("voom_span")
                != (theirs["span"][0] if "span" in theirs else None)
            )
        )
        rows.append(
            (
                f"{label}: samples, features, aliased columns, contrasts, span, df",
                "limma",
            )
            + ("mismatches", str(mismatches), "0", passed[mismatches == 0])
        )
        # limma-voom's exact standard errors are limma's for a contrast of one
        # coefficient; contrasts of several are compared where it runs with limma's.
        exact = case.function is limma_voom and case.options.get(LIMMA) != "limma"
        both = [
            c
            for c in tables
            if not isinstance(theirs["contrasts"].get(c, ""), str)
            and not (exact and np.count_nonzero(case.design.contrasts[c]) > 1)
        ]
        if both:
            value = max(largest(tables[c], theirs["contrasts"][c]) for c in both)
            rows.append(
                (f"{label}: {len(both)} contrast{'s' * (len(both) > 1)}", "topTable")
                + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
                + (passed[value <= tolerance],)
            )
        for (where, contrast), meant in pairs.items():
            if where == group:
                rows += deliberate(
                    label, contrast, meant, tables, theirs, record, tolerance
                )
    return rows


def compare_f_tests(name, case, result, default, limma):
    """Rows for the F-tests of one case: against limma's, against the nested fits for
    limma-voom's default (``default``, the case run with exact standard errors), and
    the deliberate differences."""
    passed = {True: "pass", False: "FAIL"}
    tolerance = case.tolerance
    # The programs test the same contrasts unless R fits a design of its own.
    estimates = case.r_formula is None
    rows = []
    for k, group in enumerate(case.cube.groups["group_id"]):
        theirs, record = limma[k], result.provenance["groups"][k]
        if record["status"] != "fitted":
            continue
        label = f"{name}, {group}"
        r_tests = theirs["f_tests"]
        ours = [f for f in case.f_tests if (group, f) in result.tables]
        computed = {f for f in case.f_tests if not isinstance(r_tests.get(f, ""), str)}
        both = [f for f in ours if f in computed]
        # The deliberate differences: F-tests that limma refuses and CytePool runs,
        # and, in the same group, those that limma runs from contrasts it misreads.
        paired = {f for (where, f) in case.f_pairs if where == group}
        deliberate = paired | (
            {f for f in case.f_tests if f in record["not_estimable"]}
            if paired
            else set()
        )
        mismatches = len((set(ours) ^ computed) - deliberate)
        mismatches += sum(record["f_test_df1"][f] != theirs["df1"][f] for f in both)
        rows.append(
            (f"{label}: F-tests run, numerator df", "classifyTestsF", "mismatches")
            + (str(mismatches), "0", passed[mismatches == 0])
        )
        if both and case.f_against_limma:
            value = max(
                f_largest(
                    result.table(group, f), r_tests[f], case.f_tests[f], estimates
                )
                for f in both
            )
            rows.append(
                (f"{label}: F-tests {', '.join(both)}", "topTable, classifyTestsF")
                + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
                + (passed[value <= tolerance],)
            )
        elif both and estimates:
            # The default against limma's F-statistics on the same columns, which take
            # the correlation of the design without weights for every gene.
            gap, changed, calls = 0.0, 0, 0
            for f in both:
                table, m = result.table(group, f), len(case.f_tests[f])
                gap = max(gap, _relative(table["f_statistic"], r_tests[f][m + 1]))
                ours_called = table["adjusted_p_value"] < 0.05
                changed += int(np.sum(ours_called != (r_tests[f][m + 3] < 0.05)))
                calls += len(table)
            rows.append(
                (f"{label}: F-tests {', '.join(both)} by default", "topTable")
                + ("max rel. difference of F; calls at FDR 5% that differ",)
                + (f"{gap:.2g}; {changed} of {calls}", "n/a", "information")
            )
        if case.function is limma_voom:
            rows.append(nested_row(label, case, default, group, theirs, tolerance))
        for where, f_name in case.f_pairs:
            if where == group:
                rows += f_deliberate(label, group, f_name, case, result, theirs)
    return rows


def nested_row(label, case, default, group, theirs, tolerance):
    """The row of limma-voom's default F-tests against the nested weighted fits."""
    passed = {True: "pass", False: "FAIL"}
    tested = [f for f in case.f_tests if (group, f) in default.tables]
    value = np.inf
    if tested and all(not isinstance(theirs["nested"].get(f, ""), str) for f in tested):
        value = max(
            max(
                difference(
                    default.table(group, f)["f_statistic"], theirs["nested"][f][0]
                ),
                difference(
                    default.table(group, f)["p_value"],
                    theirs["nested"][f][1],
                    p_value=True,
                ),
            )
            for f in tested
        )
    return (
        (f"{label}: F-tests {', '.join(tested)} by default: F, p", "lm.wfit, eBayes")
        + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
        + (passed[value <= tolerance],)
    )


def _relative(ours, reference):
    """The largest relative difference, for information."""
    ours, reference = np.asarray(ours, float), np.asarray(reference, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        gap = np.where(ours == reference, 0.0, np.abs(ours - reference) / reference)
    return float(np.nanmax(gap))


def deliberate(label, contrast, meant, tables, theirs, record, tolerance):
    """The rows of a group where limma refuses one contrast and misreads another.

    The first compares the contrast that limma refuses with limma's values for the one
    it misreads, which estimate the same quantity.
    """
    refused, misread = theirs["contrasts"][contrast], theirs["contrasts"][meant]
    expected = (
        isinstance(refused, str)
        and "non-estimable" in refused
        and not isinstance(misread, str)
        and meant in record["not_estimable"]
        and contrast in tables
    )
    value = largest(tables[contrast], misread) if expected else np.inf
    outcome = "deliberate difference" if expected else "FAIL"
    refusal = refused.strip() if isinstance(refused, str) else "values"
    return [
        (f"{label}: {contrast}, against limma's {meant}", "topTable")
        + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
        + ("pass" if value <= tolerance else "FAIL",),
        (f"{label}: {contrast} (estimable, weighs an aliased column)", "limma")
        + ("behaviour", f"limma: {refusal}; CytePool: values", "n/a", outcome),
        (f"{label}: {meant} (not estimable without donor D1)", "limma", "behaviour")
        + (f"limma: values, those of {contrast}; CytePool: not estimable", "n/a")
        + (outcome,),
    ]


def f_deliberate(label, group, f_name, case, result, theirs):
    """The rows of a group where limma refuses an F-test that CytePool runs, compared
    with limma's F-test of the contrasts it reads in their place, and of each F-test
    that limma runs from contrasts CytePool finds not estimable."""
    tolerance = case.tolerance
    reading = f"{f_name} as limma reads it"
    refused, read = theirs["f_tests"].get(f_name), theirs["f_tests"].get(reading)
    record = result.provenance["groups"][list(result.groups).index(group)]
    expected = (
        isinstance(refused, str)
        and "non-estimable" in refused
        and not isinstance(read, str)
        and (group, f_name) in result.tables
        and record["f_test_df1"][f_name] == theirs["df1"].get(reading)
    )
    members = case.f_tests[f_name]
    value = (
        f_largest(result.table(group, f_name), read, members) if expected else np.inf
    )
    instead = ", ".join(case.f_pairs[(group, f_name)])
    refusal = refused.strip() if isinstance(refused, str) else "values"
    rows = [
        (f"{label}: F-test {f_name}, against limma's of {instead}", "classifyTestsF")
        + ("max scaled difference", f"{value:.1e}", f"{tolerance:.0e}")
        + ("pass" if value <= tolerance else "FAIL",),
        (f"{label}: F-test {f_name} (its contrasts are estimable)", "limma")
        + ("behaviour", f"limma: {refusal}; CytePool: values", "n/a")
        + ("deliberate difference" if expected else "FAIL",),
    ]
    for other in case.f_tests:
        if other in record["not_estimable"] and not isinstance(
            theirs["f_tests"].get(other, ""), str
        ):
            rows.append(
                (f"{label}: F-test {other} (a contrast is not estimable)", "limma")
                + (
                    "behaviour",
                    "limma: values, of contrasts it misreads; CytePool: none",
                )
                + ("n/a", "deliberate difference")
            )
    return rows


def write_fixtures(inputs, cases, limma, results):
    """Write the cells, and limma's outputs for the fixture analyses: the F-tests that
    CytePool runs, limma's F-tests in place of those it refuses, its refusals, and the
    nested fits of limma-voom's F-tests."""
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FIXTURE_DIRECTORY / "cells.npz", **inputs)
    arrays, spline = {}, {}
    for name, case in cases.items():
        cube = case.cube
        key_of = name.removeprefix("fixture ")
        target = spline if "spline" in name else arrays
        for k, group in enumerate(cube.groups["group_id"]):
            out = limma[name][k]
            key = key_of if "spline" in name else f"{key_of} {group}"
            target[f"{key} used"] = out["used"].astype(bool)
            target[f"{key} keep"] = out["keep"].astype(bool)
            target[f"{key} aliased"] = np.array(out["not_estimable"], dtype=str)
            target[f"{key} df_prior"] = out["df_prior"]
            if "span" in out:
                target[f"{key} span"] = out["span"]
            for contrast, values in out["contrasts"].items():
                if isinstance(values, str):
                    target[f"{key} {contrast} error"] = np.array(values.strip())
                else:
                    target[f"{key} {contrast}"] = values
            run = {f for g, f in results[name].tables if g == group}
            readings = {
                f"{f} as limma reads it" for where, f in case.f_pairs if where == group
            }
            for f_name, values in out["f_tests"].items():
                if isinstance(values, str):
                    if f_name in case.f_tests:
                        target[f"{key} F {f_name} error"] = np.array(values.strip())
                elif (f_name in run and case.f_against_limma) or f_name in readings:
                    target[f"{key} F {f_name}"] = values
                    target[f"{key} F {f_name} df1"] = np.array(out["df1"][f_name])
            for f_name, values in out["nested"].items():
                if f_name in run and not isinstance(values, str):
                    target[f"{key} F {f_name} nested"] = values
        if "spline" in name:
            # The counts and doses of the one cell type, as the tests rebuild it.
            spline["counts"] = np.asarray(cube.layer("counts", "hepatocyte"))
            spline["sample_id"] = np.asarray(cube.samples["sample_id"], dtype=str)
            spline["dose"] = np.asarray(cube.samples["dose"], dtype=float)
    np.savez_compressed(FIXTURE_DIRECTORY / "limma.npz", **arrays)
    np.savez_compressed(FIXTURE_DIRECTORY / "dose_spline.npz", **spline)
    print(f"wrote cells.npz, limma.npz and dose_spline.npz to {FIXTURE_DIRECTORY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--write-fixtures", action="store_true", help="write test fixtures"
    )
    arguments = parser.parse_args()
    cases = make_cases(np.random.default_rng(20261025), np.random.default_rng(20261119))
    inputs, fixtures = fixture_cases(
        np.random.default_rng(20261026), np.random.default_rng(20261120)
    )
    if arguments.write_fixtures:
        cases |= fixtures
    limma, versions = run_limma(cases)
    print("Reference: " + ", ".join(versions))
    print(
        "voom in R: adaptive.span = TRUE, or span = 0.5, adaptive.span = FALSE where "
        "a case fixes the span"
    )
    table, results = [], {}
    for name, case in cases.items():
        result = case.function(
            case.cube, case.design, f_tests=case.f_tests, **case.options
        )
        results[name] = result
        table += compare(name, case, result, limma[name])
        if case.f_tests:
            # limma-voom's default, exact, standard errors, for the nested fits.
            exact = {k: v for k, v in case.options.items() if k != LIMMA}
            default = (
                case.function(case.cube, case.design, f_tests=case.f_tests, **exact)
                if case.function is limma_voom
                else result
            )
            table += compare_f_tests(name, case, result, default, limma[name])
    print("| check | reference | metric | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- | --- |")
    for row in table:
        print("| " + " | ".join(row) + " |")
    failures = sum(row[-1] == "FAIL" for row in table)
    print(f"{len(table)} checks, {failures} failed")
    if arguments.write_fixtures and failures:
        print("not writing fixtures: some checks failed")
    elif arguments.write_fixtures:
        write_fixtures(inputs, fixtures, limma, results)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
