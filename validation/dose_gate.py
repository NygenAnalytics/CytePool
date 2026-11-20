"""Check the gate of dose_response and permutation_check against limma, stageR,
variancePartition and R's own count and enumeration of the assignments of dose
labels.

Dose series simulated by ``simulate_dose_series`` from fixed seeds (20260901 to
20260905): ``fixture``, the vehicle and 5 doses from 0.1 to 10 ug/kg x 3, 300 genes of
which 20% respond, in a cell type of every sample and one whose profiles at the top
dose hold too few cells; ``duplicate``, the vehicle, 0.1 and 1 ug/kg x 2, 600 genes of
which 30% respond, 300 cells per profile, whose 6 samples allow 90 assignments of
their doses; ``sex``, the vehicle and 1 ug/kg x 4, two of each sex, under
``"~ sex"``, 36 assignments within the sexes; ``donors``, 4 donors whose cells
receive the vehicle and 4 doses, under ``(1 | donor)``; and ``animals``, the vehicle
and 3 doses x 6 samples, 2 from each of 3 animals, two female and one male at each
dose, under ``"~ sex"`` and ``(1 | animal)``, so that the sexes split the animals into
blocks.

R, which sees the counts, cells, doses, the design's variables and the settings and
nothing that CytePool computed, keeps each cell type's samples of 10 cells or more
and some counts and fits the design's terms with an indicator of each treated dose
they hold: edgeR's ``filterByExpr`` and ``calcNormFactors``, then ``voom`` and
``lmFit`` (limma-voom) or log-CPM and ``lmFit`` (limma-trend). The gates' p-values:
the dose groups' F-test, exact under voom's weights (each gene's ``lm.wfit``,
CytePool's default) and limma's ``classifyTestsF``
(``contrast_standard_errors="limma"``), referred to each gene's degrees of freedom;
with a fold change, ``treat`` and ``topTreat`` of each dose, the smallest p-value
times the doses, capped at 1, and stageR's ``stageWiseAdjustment`` for the lowest
effective doses. Calls are ``p.adjust(, "BH") <= fdr``, of the spline gate on
CytePool's spline p-values, which ``dose_response_tests.py`` checks against limma.
From the sample table R builds the blocks of exchangeable units itself, counts the
assignments as products of binomial coefficients, enumerates them where they are few,
refits limma on each assignment of ``duplicate`` (90) and ``sex`` (36), counting as
ties the assignments whose refit gives the true doses' genes and p-values within
1e-10, and refits one recorded assignment of every case; variancePartition 1.36.2
fits the dream designs.

Tiers: exact (1e-10) for limma's p-values and TREAT, close (1e-6) for robust priors
and dream's F-statistics; identical for calls, genes, counts of assignments, the
assignments enumerated, the genes each run passes and the smallest possible
p-values.
p-values are compared as their difference over p max(1, |log p|), statistics over
max(|reference|, 1).

Requirements:
    Python 3.13 with numpy and numba, run from the repository root.
    R 4.4 with limma, edgeR, stageR and jsonlite (``RSCRIPT``), and
    variancePartition, lme4, minqa, edgeR and jsonlite (``DREAM_RSCRIPT``); each
    variable names an Rscript executable and defaults to ``Rscript``.

Usage:
    PYTHONPATH=. python validation/dose_gate.py
    PYTHONPATH=. python validation/dose_gate.py --write-fixtures
    PYTHONPATH=. python validation/dose_gate.py --benchmark
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import dream_voom, limma_trend, limma_voom
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import DoseResponseResult, dose_response
from cytepool.permutation_checks import PermutationCheck, permutation_check
from cytepool.simulated_dose_series import simulate_dose_series
from cytepool.stats.empirical_bayes import treat
from cytepool.stats.linear_models import fit_linear_model

FIXTURE_DIRECTORY = Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_gate"
EXACT, CLOSE = 1e-10, 1e-6
MIN_CELLS = 10
FDR = 0.05
LEVELS = (0.05, 0.1)
FOLD_CHANGES = (1.2, 1.5)
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
# The fold change of the gates whose runs R repeats.
CHECKED_FOLD_CHANGE = 1.5

R_LIMMA = r"""
suppressPackageStartupMessages({
  library(limma); library(edgeR); library(stageR); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("limma", "edgeR", "stageR"), function(p) as.character(packageVersion(p))))))
HELPERS

# The gate's p-value of each gene: the dose groups' F-test, exact under voom's
# weights (each gene's own weighted least squares) or limma's (classifyTestsF),
# referred to each gene's degrees of freedom; or, with a fold change, each dose's
# TREAT p-value and the smallest of them times the doses, capped at 1.
gate_p_values <- function(c, gate, analysis, robust, exact) {
  k <- length(c$doses) - 1
  trend <- analysis == "trend"
  if (gate$test == "each_dose") {
    tr <- treat(c$fit[, c$indicators], lfc = log2(gate$fold_change), trend = trend,
                robust = robust)
    P <- matrix(sapply(seq_len(k), function(j)
      topTreat(tr, coef = j, number = Inf, sort.by = "none")$P.Value), ncol = k)
    return(list(p = pmin(1, k * apply(P, 1, min)), per_dose = P))
  }
  L <- rbind(matrix(0, c$n_others, k), diag(k))
  eb <- eBayes(contrasts.fit(c$fit, L), trend = trend, robust = robust)
  df2 <- eb$df.residual + eb$df.prior
  if (exact && !is.null(c$W)) {
    s2 <- eb$s2.post
    f <- sapply(seq_len(nrow(c$E)), function(g) {
      w <- lm.wfit(c$X, c$E[g, ], c$W[g, ])
      V <- chol2inv(qr.R(w$qr)); est <- crossprod(L, w$coefficients)
      drop(crossprod(est, solve(crossprod(L, V %*% L), est))) / ncol(L) / s2[g]
    })
    return(list(p = pf(f, ncol(L), df2, lower.tail = FALSE)))
  }
  fs <- classifyTestsF(eb, fstat.only = TRUE)
  list(p = pf(as.numeric(fs), attr(fs, "df1"), df2, lower.tail = FALSE))
}
calls <- function(p, fdr) p.adjust(p, "BH") <= fdr

# stageR's stage-wise adjustment of each dose in the genes the gate passes.
stage_wise <- function(screen, p, alpha) {
  names(screen) <- paste0("f", seq_along(screen))
  dimnames(p) <- list(names(screen), paste0("h", seq_len(ncol(p))))
  o <- quiet(stageWiseAdjustment(
    stageR(pScreen = screen, pConfirmation = p, pScreenAdjusted = FALSE),
    method = "holm", alpha = alpha))
  a <- quiet(getAdjustedPValues(o, onlySignificantGenes = FALSE, order = FALSE))
  full <- matrix(NA_real_, nrow(p), ncol(p))
  full[match(rownames(a), names(screen)), ] <- a[, -1, drop = FALSE]
  full
}

# The limma jobs: the dose groups' gate, the spline gate's calls on CytePool's
# p-values, each dose's TREAT p-values and the gate with a fold change, and the
# lowest effective doses under it.
for (name in names(input$limma)) {
  job <- input$limma[[name]]
  case <- input$cases[[job$case]]
  result <- list()
  for (ct in names(case$cell_types)) {
    c <- fit_cell_type(case, ct, case$samples$dose, job$analysis)
    res <- list(samples = c$samples, doses = c$doses, genes = c$genes, gates = list(),
                treat = list())
    for (test in job$tests) {
      gate <- list(test = test)
      exact <- gate_p_values(c, gate, job$analysis, job$robust, TRUE)$p
      res$gates[[test]] <- list(exact = exact, exact_adjusted = p.adjust(exact, "BH"))
      if (job$analysis == "voom") {
        approximate <- gate_p_values(c, gate, job$analysis, job$robust, FALSE)$p
        res$gates[[test]]$limma <- approximate
        res$gates[[test]]$limma_adjusted <- p.adjust(approximate, "BH")
      }
    }
    if (!is.null(job$spline[[ct]])) {
      p <- as.numeric(job$spline[[ct]])
      res$gates$spline <- list(exact_adjusted = p.adjust(p, "BH"))
    }
    for (fc in input$fold_changes) {
      g <- gate_p_values(c, list(test = "each_dose", fold_change = fc),
                         job$analysis, job$robust, TRUE)
      # stageR's "holm" fails on one hypothesis per gene; one treated dose has no
      # stage-wise step.
      res$treat[[format(fc)]] <- list(
        per_dose = g$per_dose, p = g$p, adjusted = p.adjust(g$p, "BH"),
        stage_wise = if (ncol(g$per_dose) > 1) stage_wise(g$p, g$per_dose, input$fdr))
    }
    result[[ct]] <- res
  }
  out$limma[[name]] <- result
}

# One assignment of a cell type's dose labels, given as each design sample's dose:
# the gate's p-values of R's fit of those doses.
for (i in names(input$reruns)) {
  job <- input$reruns[[i]]
  case <- input$cases[[job$case]]
  c <- fit_cell_type(case, job$cell_type, job$doses, job$analysis)
  p <- gate_p_values(c, job$gate, job$analysis, FALSE, TRUE)$p
  out$reruns[[i]] <- list(samples = c$samples, genes = c$genes, p = p,
                          adjusted = p.adjust(p, "BH"))
}

# The exchangeable blocks of each case's cell types, their assignments, and, where
# asked, every assignment refitted: the genes each gate passes, and the ties, the
# assignments, the true one included, whose refit gives the true one's genes and
# p-values within 1e-10.
same_fit <- function(f, g) identical(f$genes, g$genes) &&
  identical(is.na(f$p), is.na(g$p)) && all(abs(f$p - g$p) <= 1e-10 * abs(g$p), na.rm = TRUE)
for (name in names(input$cases)) {
  case <- input$cases[[name]]
  for (ct in names(case$cell_types)) {
    b <- blocks_of(case, ct)
    entry <- list(kind = b$kind, samples = b$samples, doses = b$doses,
                  labels = b$label, n_assignments = count_assignments(b))
    if (ct %in% names(case$every)) {
      every <- every_assignment(b)
      entry$assignments <- every
      true <- which(apply(every, 1, function(label) all(label == b$label)))
      refits <- lapply(case$every[[ct]], function(gate) lapply(seq_len(nrow(every)),
        function(i) {
          doses <- case$samples$dose
          doses[match(b$samples, case$samples$sample_id)] <- b$doses[every[i, ] + 1]
          c <- fit_cell_type(case, ct, doses, "voom")
          list(genes = c$genes, p = gate_p_values(c, gate, "voom", FALSE, TRUE)$p)
        }))
      entry$counts <- lapply(refits, function(r) sapply(r, function(f)
        sum(calls(f$p, input$fdr))))
      entry$ties <- lapply(refits, function(r) sum(sapply(r, same_fit, r[[true]])))
    }
    out$blocks[[name]][[ct]] <- entry
  }
}
write_json(out, args[[2]], digits = I(17), matrix = "rowmajor", auto_unbox = TRUE,
           na = "null")
"""

R_DREAM = r"""
suppressPackageStartupMessages({
  library(variancePartition); library(lme4); library(edgeR); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
HELPERS
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE,
                     check.conv.singular = "ignore")
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("variancePartition", "lme4", "edgeR"), function(p) as.character(packageVersion(p))))))
# The dose groups' F-test of a cell type under dream, with the doses given for the
# design's samples: the design's terms, the indicators of the treated doses and the
# case's random intercept, voomWithDreamWeights, dream with the doses as contrasts,
# eBayes and topTable.
for (i in names(input$dream)) {
  job <- input$dream[[i]]
  case <- input$cases[[job$case]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  data$dose <- job$doses
  used <- used_samples(case, job$cell_type)
  d <- data[used, , drop = FALSE]
  y <- case$cell_types[[job$cell_type]]$counts[, used, drop = FALSE]
  treated <- sort(unique(d$dose[d$dose > 0]))
  ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
  colnames(ind) <- paste0("dose", seq_along(treated))
  others <- model.matrix(as.formula(case$formula), d)
  X <- cbind(others, ind)
  dge <- DGEList(y); keep <- filterByExpr(dge, design = X)
  dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
  d <- cbind(d, ind); d[[case$unit]] <- factor(d[[case$unit]])
  terms <- sub("^\\s*~\\s*", "", case$formula)
  form <- as.formula(paste("~", terms, "+", paste(colnames(ind), collapse = " + "),
                           "+ (1 |", case$unit, ")"))
  v <- quiet(voomWithDreamWeights(dge, form, d, span = 0.5, control = tight))
  L <- rbind(matrix(0, ncol(others), length(treated)), diag(length(treated)))
  rownames(L) <- c(colnames(others), colnames(ind))
  colnames(L) <- paste0("L", seq_along(treated))
  e <- eBayes(quiet(dream(v, form, d, L = L, control = tight)))
  top <- topTable(e, coef = colnames(L), number = Inf, sort.by = "none")
  out$dream[[i]] <- list(samples = d$sample_id, genes = which(keep), F = top$F)
}
write_json(out, args[[2]], digits = I(17), matrix = "rowmajor", auto_unbox = TRUE,
           na = "null")
"""

R_HELPERS = r"""
# The samples of a cell type with enough cells and some counts.
used_samples <- function(case, ct) {
  y <- case$cell_types[[ct]]$counts
  which(case$cell_types[[ct]]$n_cells >= input$min_cells & colSums(y) > 0)
}

# One cell type of a case with the doses given for the design's samples: its
# samples, the design's terms and an indicator of each treated dose they hold,
# edgeR's filter and normalisation, then voom and lmFit (limma-voom) or log-CPM and
# lmFit (limma-trend).
fit_cell_type <- function(case, ct, doses, analysis) {
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  data$dose <- doses
  used <- used_samples(case, ct)
  d <- data[used, , drop = FALSE]; y <- case$cell_types[[ct]]$counts[, used, drop = FALSE]
  treated <- sort(unique(d$dose[d$dose > 0]))
  others <- model.matrix(as.formula(case$formula), d)
  ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
  colnames(ind) <- paste0("dose", seq_along(treated))
  X <- cbind(others, ind)
  dge <- DGEList(y); keep <- filterByExpr(dge, design = X)
  dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
  if (analysis == "voom") {
    v <- voom(dge, X, adaptive.span = TRUE); E <- v$E; W <- v$weights; fit <- lmFit(v, X)
  } else {
    E <- cpm(dge, log = TRUE, prior.count = 3); W <- NULL; fit <- lmFit(E, X)
  }
  compared <- c(0, treated); n <- sapply(compared, function(k) sum(d$dose == k))
  list(fit = fit, E = E, W = W, X = X, samples = d$sample_id, doses = compared, n = n,
       genes = which(keep), indicators = colnames(ind), n_others = ncol(others), data = d)
}

# The exchangeable blocks of a cell type's samples under the null, read from the
# sample table: without a unit, the samples with the same row of the model of the
# design's terms; with a unit that holds one dose, the units with the same rows,
# moved whole; otherwise the samples of one unit with the same row.
blocks_of <- function(case, ct) {
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  d <- data[used_samples(case, ct), , drop = FALSE]
  rows <- apply(model.matrix(as.formula(case$formula), d), 1, paste, collapse = ",")
  doses <- sort(unique(d$dose)); label <- match(d$dose, doses) - 1
  kind <- "samples"; items <- as.list(seq_len(nrow(d))); key <- rows
  if (!is.null(case$unit)) {
    unit <- as.character(d[[case$unit]])
    if (all(tapply(label, unit, function(l) length(unique(l)) == 1))) {
      kind <- "whole units"
      items <- unname(split(seq_len(nrow(d)), factor(unit, levels = unique(unit))))
      key <- sapply(items, function(i) paste(sort(rows[i]), collapse = ";"))
    } else {
      kind <- "doses within units"; key <- paste(unit, rows)
    }
  }
  item_label <- sapply(items, function(i) label[i[1]])
  blocks <- unname(split(seq_along(items), factor(key, levels = unique(key))))
  list(kind = kind, samples = d$sample_id, doses = doses, label = label, items = items,
       item_label = item_label, blocks = blocks)
}
# The count of assignments: the product over blocks of the multinomial coefficient of
# the labels' counts, as a product of binomial coefficients, exact below 2^53.
count_assignments <- function(b) {
  prod(sapply(b$blocks, function(items) {
    counts <- as.numeric(table(b$item_label[items])); left <- sum(counts); total <- 1
    for (k in counts) { total <- total * choose(left, k); left <- left - k }
    total
  }))
}
# The distinct arrangements of a multiset, one per row.
arrangements <- function(v) {
  if (length(v) <= 1) return(matrix(v, nrow = 1))
  do.call(rbind, lapply(unique(v), function(x) {
    i <- match(x, v); cbind(x, arrangements(v[-i]))
  }))
}
# Every assignment of the labels to the units within blocks, as each sample's label.
every_assignment <- function(b) {
  per_block <- lapply(b$blocks, function(items) arrangements(b$item_label[items]))
  grid <- as.matrix(expand.grid(lapply(per_block, function(m) seq_len(nrow(m)))))
  t(apply(grid, 1, function(choice) {
    item_label <- b$item_label
    for (j in seq_along(b$blocks)) item_label[b$blocks[[j]]] <- per_block[[j]][choice[j], ]
    label <- integer(length(b$label))
    for (i in seq_along(b$items)) label[b$items[[i]]] <- item_label[i]
    label
  }))
}
"""


# The series.


@dataclass
class Case:
    """A simulated dose series, its design, and the analysis R and CytePool run."""

    name: str
    cube: Cube
    design: Design
    formula: str
    unit: str | None  # the column of the design's random intercept
    analysis: str  # "limma" or "dream"
    tests: tuple[str, ...] = ()  # the gates checked against limma
    every: tuple[str, ...] = ()  # the cell types R reruns on every assignment


def cases() -> list[Case]:
    """The series, from fixed seeds."""

    def series(doses: list[float], seed: int, columns: Any = None, **options: Any):
        cube, _ = simulate_dose_series(doses, rng=seed, **options)
        samples = cube.samples
        if columns is not None:
            samples = samples.with_columns(columns(samples))
        return cube, samples

    found = []
    cube, samples = series(
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        20260901,
        replicates=3,
        cell_types=["full", "lost_top_dose"],
        cells=[300.0, 200.0],
        n_genes=300,
        composition={"lost_top_dose": [1.0, 1.0, 1.0, 1.0, 1.0, 0.005]},
    )
    found.append(
        Case(
            "fixture",
            cube,
            Design(samples, "~ 1"),
            "~ 1",
            None,
            "limma",
            ("spline", "dose_groups"),
        )
    )
    cube, samples = series(
        [0.0, 0.1, 1.0],
        20260902,
        replicates=2,
        cell_types=1,
        cells=300.0,
        n_genes=600,
        responsive=0.3,
    )
    found.append(
        Case(
            "duplicate",
            cube,
            Design(samples, "~ 1"),
            "~ 1",
            None,
            "limma",
            ("spline", "dose_groups"),
            ("cell_type_1",),
        )
    )

    def sexes(samples: Any) -> dict[str, list[str]]:
        ids = samples["sample_id"].tolist()
        return {"sex": ["F" if i.endswith(("_r1", "_r2")) else "M" for i in ids]}

    # The simulator places curves on two doses or more; the design keeps the
    # vehicle and 1 ug/kg.
    cube, samples = series(
        [0.0, 1.0, 10.0],
        20260903,
        sexes,
        replicates=4,
        cell_types=1,
        cells=300.0,
        n_genes=400,
        responsive=0.3,
    )
    kept = samples.filter(np.asarray(samples["dose"]) < 10.0)
    found.append(
        Case(
            "sex",
            cube,
            Design(kept, "~ sex"),
            "~ sex",
            None,
            "limma",
            ("dose_groups",),
            ("cell_type_1",),
        )
    )
    cube, samples = series(
        [0.0, 0.1, 1.0, 10.0, 100.0],
        20260904,
        replicates=4,
        cell_types=1,
        cells=300.0,
        n_genes=300,
        donor_sd=0.3,
    )
    found.append(
        Case(
            "donors",
            cube,
            Design(samples, "~ 1", random="(1 | donor)"),
            "~ 1",
            "donor",
            "dream",
        )
    )

    def animals(samples: Any) -> dict[str, list[str]]:
        ids = samples["sample_id"].tolist()
        # Replicates 1 and 2 of a dose come from its first animal, 3 and 4 from its
        # second, a male, 5 and 6 from its third.
        animal = [(int(i.split("_r")[1]) + 1) // 2 for i in ids]
        return {
            "animal": [
                f"{i.split('_')[0]}_a{a}" for i, a in zip(ids, animal, strict=True)
            ],
            "sex": ["M" if a == 2 else "F" for a in animal],
        }

    cube, samples = series(
        [0.0, 0.1, 1.0, 10.0],
        20260905,
        animals,
        replicates=6,
        cell_types=1,
        cells=300.0,
        n_genes=300,
    )
    found.append(
        Case(
            "animals",
            cube,
            Design(samples, "~ sex", random="(1 | animal)"),
            "~ sex",
            "animal",
            "dream",
        )
    )
    return found


# CytePool's runs.


ANALYSES = {"voom": limma_voom, "trend": limma_trend, "dream": dream_voom}
# The limma settings each gate runs under: limma-voom, robustly, and limma-trend.
SETTINGS = (("voom", False), ("voom", True), ("trend", False))


def analyse(case: Case, analysis: str, gate: dict[str, Any], **options: Any) -> Any:
    if analysis == "dream":
        # voom's trend in variancePartition's voomWithDreamWeights, with its span.
        options = {"span": 0.5} | options
    return dose_response(
        case.cube,
        case.design,
        dose=AXIS,
        gate=gate,
        analysis=ANALYSES[analysis],
        **options,
    )


def label(case: Case, analysis: str, robust: bool) -> str:
    return f"{case.name} {analysis}{' robust' if robust else ''}"


@dataclass
class Ours:
    """CytePool's results of the limma cases, by job and gate."""

    gates: dict[tuple[str, str, float, str], DoseResponseResult] = field(
        default_factory=dict
    )
    fold_changes: dict[tuple[str, float], DoseResponseResult] = field(
        default_factory=dict
    )
    checks: dict[str, PermutationCheck] = field(default_factory=dict)
    every: dict[tuple[str, str], PermutationCheck] = field(default_factory=dict)


def limma_runs(found: list[Case]) -> Ours:
    """Each limma case's gates at each level and setting, its fold-change gates, and
    a check of one run; the duplicate design's every assignment."""
    ours = Ours()
    for case in found:
        if case.analysis != "limma":
            continue
        for analysis, robust in SETTINGS if case.name != "sex" else (("voom", False),):
            job = label(case, analysis, robust)
            for test in case.tests:
                for level in LEVELS:
                    gate = {"test": test, "fdr": level}
                    ours.gates[(job, test, level, "exact")] = analyse(
                        case, analysis, gate, robust=robust
                    )
                    if analysis == "voom" and test == "dose_groups":
                        ours.gates[(job, test, level, "limma")] = analyse(
                            case,
                            analysis,
                            gate,
                            robust=robust,
                            contrast_standard_errors="limma",
                        )
            for change in FOLD_CHANGES:
                ours.fold_changes[(job, change)] = analyse(
                    case, analysis, {"fold_change": change}, robust=robust
                )
    return ours


def check_runs(found: list[Case], ours: Ours) -> None:
    """One random run of every case, with the dose-group gate, and every other
    assignment of ``duplicate`` and ``sex``, with the dose-group gate and the
    fold-change gate."""
    for case in found:
        result = analyse(
            case,
            "dream" if case.analysis == "dream" else "voom",
            {"test": "dose_groups"},
        )
        ours.checks[case.name] = permutation_check(result, n_permutations=1, rng=1)
    for name in ("duplicate", "sex"):
        case = next(c for c in found if c.name == name)
        for gate in ({"test": "dose_groups"}, {"fold_change": CHECKED_FOLD_CHANGE}):
            result = analyse(case, "voom", gate)
            key = "dose_groups" if "test" in gate else "each_dose"
            ours.every[(name, key)] = permutation_check(
                result, n_permutations=10_000, rng=1
            )


def recorded_doses(case: Case, entry: dict[str, Any], run: int) -> list[float]:
    """The doses of the design's samples in a run of a check: the cell type's used
    samples take the doses the provenance records for the run."""
    doses = np.asarray(case.design.samples["dose"], dtype=float).copy()
    ids = case.design.samples["sample_id"].tolist()
    for sample, label_ in zip(entry["samples"], entry["assignments"][run], strict=True):
        doses[ids.index(sample)] = entry["doses"][label_]
    return doses.tolist()


def permuted(case: Case, doses: list[float]) -> Design:
    """The case's design with other doses."""
    random = [str(effect) for effect in case.design.random] or None
    return Design(
        case.design.samples.with_columns({"dose": np.asarray(doses)}),
        case.formula,
        random=random,
    )


# R.


def r_input(found: list[Case], ours: Ours) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "min_cells": MIN_CELLS,
        "fdr": FDR,
        "fold_changes": list(FOLD_CHANGES),
        "cases": {},
        "limma": {},
        "reruns": {},
        "dream": {},
    }
    for case in found:
        cube = case.cube
        ids = cube.samples["sample_id"].tolist()
        rows = [ids.index(sample) for sample in case.design.samples["sample_id"]]
        samples = {
            name: (
                np.asarray(case.design.samples[name], dtype=float).tolist()
                if name == "dose"
                else case.design.samples[name].tolist()
            )
            for name in case.design.samples.columns
        }
        # The cell types R enumerates, and the gates it reruns on each assignment.
        gates = {
            "dose_groups": {"test": "dose_groups"},
            "each_dose": {"test": "each_dose", "fold_change": CHECKED_FOLD_CHANGE},
        }
        payload["cases"][case.name] = {
            "samples": samples,
            "formula": case.formula,
            "unit": case.unit,
            "every": {group: gates for group in case.every},
            # The profiles of the design's samples, in its order.
            "cell_types": {
                group: {
                    "counts": cube.layer("counts", g)[rows].T.astype(int).tolist(),
                    "n_cells": np.asarray(cube.n_cells)[rows, g].astype(int).tolist(),
                }
                for g, group in enumerate(cube.groups["group_id"].tolist())
            },
        }
        if case.analysis == "limma":
            settings = SETTINGS if case.name != "sex" else (("voom", False),)
            for analysis, robust in settings:
                job = label(case, analysis, robust)
                spline = ours.gates.get((job, "spline", FDR, "exact"))
                payload["limma"][job] = {
                    "case": case.name,
                    "analysis": analysis,
                    "robust": robust,
                    "tests": [test for test in case.tests if test != "spline"],
                    # CytePool's spline p-values, whose calls R makes.
                    "spline": {}
                    if spline is None
                    else {
                        group: [
                            None if np.isnan(p) else p
                            for p in spline.table(group, "spline")["p_value"].tolist()
                        ]
                        for group in spline.groups
                    },
                }
        check = ours.checks[case.name]
        for entry in check.provenance["groups"]:
            if not entry["checked"]:
                continue
            job = {
                "case": case.name,
                "cell_type": entry["group_id"],
                "doses": recorded_doses(case, entry, 0),
                "gate": {"test": "dose_groups"},
                "analysis": "voom",
            }
            if case.analysis == "dream":
                for run in (False, True):
                    found = job if run else job | {"doses": samples["dose"]}
                    payload["dream"][str(len(payload["dream"]) + 1)] = found
            else:
                payload["reruns"][str(len(payload["reruns"]) + 1)] = job
    return payload


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "input.json").write_text(json.dumps(payload))
        (root / "program.R").write_text(program.replace("HELPERS", R_HELPERS))
        command = [
            os.environ.get(variable, "Rscript"),
            "--vanilla",
            str(root / "program.R"),
            str(root / "input.json"),
            str(root / "output.json"),
        ]
        subprocess.run(command, check=True)
        found: dict[str, Any] = json.loads((root / "output.json").read_text())
        return found


# The comparisons.


def scaled(ours: Any, reference: Any) -> float:
    """The largest difference over max(|reference|, 1); infinite when the shapes or
    the NaNs differ."""
    a, b = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    present = ~np.isnan(b)
    if not present.any():
        return 0.0
    return float(np.max(np.abs(a - b)[present] / np.maximum(np.abs(b[present]), 1.0)))


def p_scaled(ours: Any, reference: Any) -> float:
    """The largest difference of p-values over p max(1, |log p|); infinite when the
    shapes or the NaNs differ."""
    a, b = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    present = ~np.isnan(b)
    a, b = a[present], b[present]
    if not a.size:
        return 0.0
    tiny = np.maximum(b, 1e-300)
    scale = tiny * np.maximum(1.0, np.abs(np.log(tiny)))
    return float(np.max(np.where((a == 0.0) & (b == 0.0), 0.0, np.abs(a - b) / scale)))


@dataclass
class Rows:
    """The rows of the printed table, and how many failed."""

    rows: list[tuple[str, ...]] = field(default_factory=list)
    failed: int = 0

    def bounded(self, check: str, reference: str, value: float, bound: float) -> None:
        passed = bool(value <= bound)
        self.failed += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{value:.1e}",
                f"{bound:g}",
                "pass" if passed else "FAIL",
            )
        )

    def identical(self, check: str, reference: str, same: int, total: int) -> None:
        passed = same == total
        self.failed += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{same:,} of {total:,} identical",
                "identical",
                "pass" if passed else "FAIL",
            )
        )

    def information(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "info"))


def numbers(values: Any) -> np.ndarray:
    """R's numbers as a float64 array, with NA (None) as NaN."""

    def with_nan(value: Any) -> Any:
        if isinstance(value, list):
            return [with_nan(item) for item in value]
        return np.nan if value is None else value

    return np.array(with_nan(values), dtype=np.float64)


def vector(values: Any) -> np.ndarray:
    return np.atleast_1d(numbers(values))


def genes_of(case: Case, theirs: dict[str, Any]) -> list[str]:
    features = case.cube.features["feature_id"].tolist()
    return [features[int(i) - 1] for i in np.atleast_1d(theirs["genes"])]


def compare_gates(found: list[Case], ours: Ours, limma: dict, rows: Rows) -> None:
    """The dose groups' gate's p-values and calls against R's fit, and the spline
    gate's calls against R's adjustment of CytePool's spline p-values, at each level
    and setting."""
    largest: dict[str, float] = {}
    same: dict[str, list[int]] = {}

    def tally(family: str, equal: int, total: int) -> None:
        counts = same.setdefault(family, [0, 0])
        counts[0] += equal
        counts[1] += total

    for (job, test, level, errors), result in ours.gates.items():
        case = next(c for c in found if job.startswith(c.name + " "))
        robust = job.endswith("robust")
        for group, theirs in limma[job].items():
            table = result.table(group, "gate")
            if table["feature_id"].tolist() != genes_of(case, theirs):
                tally("genes tested", 0, 1)
                continue
            tally("genes tested", 1, 1)
            values = theirs["gates"][test]
            if errors in values:  # R's own p-values: not the spline's
                family = f"'{test}' gate, {'limma' if errors == 'limma' else 'exact'}"
                difference = p_scaled(table["p_value"], vector(values[errors]))
                key = f"{family}{', robust' if robust else ''}"
                largest[key] = max(largest.get(key, 0.0), difference)
            called = vector(values[errors + "_adjusted"]) <= level
            mine = np.asarray(table["passes_gate"], dtype=bool)
            tally(
                f"'{test}' gate: genes passing", int(np.sum(mine == called)), mine.size
            )
    for key, value in sorted(largest.items()):
        rows.bounded(
            f"gate p-values: {key} standard errors, fixture, duplicate and sex, "
            "limma-voom and limma-trend",
            "limma 3.62.1 from the counts: lm.wfit under voom's weights or "
            "classifyTestsF, pf on each gene's df",
            value,
            CLOSE if "robust" in key else EXACT,
        )
    references = {
        "'spline' gate: genes passing": "R: p.adjust(p, 'BH') <= fdr on CytePool's "
        "spline p-values, which dose_response_tests.py checks against limma",
    }
    for family, (equal, total) in sorted(same.items()):
        rows.identical(
            f"{family}, at 5% and 10% FDR",
            references.get(family, "R: p.adjust(p, 'BH') <= fdr on R's p-values"),
            equal,
            total,
        )


def compare_fold_changes(
    found: list[Case], ours: Ours, limma: dict, rows: Rows
) -> None:
    """Each dose's TREAT p-values, the gate's p-values and calls, and the lowest
    effective doses under it, against limma's treat and stageR."""
    per_dose: dict[str, float] = {}
    gate: dict[str, float] = {}
    calls = [0, 0]
    lowest = [0, 0]
    stage_wise = 0.0
    for (job, change), result in ours.fold_changes.items():
        analysis, robust = job.split(" ")[1], job.endswith("robust")
        for group, theirs in limma[job].items():
            reference = theirs["treat"][format(change, "g")]
            k = len(theirs["doses"]) - 1
            fit = result.fit(group)
            names = [f"dose_{dose:.12g}" for dose in theirs["doses"][1:]]
            tested = treat(
                fit.moderated.fit,
                fc=change,
                trend=analysis == "trend",
                robust=robust,
            )
            columns = [fit.moderated.fit.coefficient_names.index(n) for n in names]
            key = f"{analysis}{', robust' if robust else ''}"
            per_dose[key] = max(
                per_dose.get(key, 0.0),
                p_scaled(
                    tested.p_value[:, columns],
                    np.asarray(reference["per_dose"], dtype=float).reshape(-1, k),
                ),
            )
            table = result.table(group, "gate")
            gate[key] = max(
                gate.get(key, 0.0), p_scaled(table["p_value"], reference["p"])
            )
            if change == CHECKED_FOLD_CHANGE and k > 1:
                called = vector(reference["adjusted"]) <= FDR
                mine = np.asarray(table["passes_gate"], dtype=bool)
                calls[0] += int(np.sum(mine == called))
                calls[1] += mine.size
                doses = np.asarray(theirs["doses"][1:], dtype=float)
                adjusted = numbers(reference["stage_wise"]).reshape(-1, k)
                hit = np.nan_to_num(adjusted, nan=2.0) <= FDR
                theirs_lowest = np.where(
                    hit.any(axis=1), doses[np.argmax(hit, axis=1)], np.nan
                )
                led = result.lowest_effective_doses(group)
                ours_lowest = np.asarray(led["lowest_effective_dose"], dtype=float)
                lowest[0] += int(
                    np.sum(
                        (ours_lowest == theirs_lowest)
                        | (np.isnan(ours_lowest) & np.isnan(theirs_lowest))
                    )
                )
                lowest[1] += ours_lowest.size
                at = np.where(hit.any(axis=1), np.argmax(hit, axis=1), 0)
                theirs_p = np.where(
                    hit.any(axis=1), adjusted[np.arange(at.size), at], np.nan
                )
                stage_wise = max(
                    stage_wise, p_scaled(led["stage_wise_adjusted_p_value"], theirs_p)
                )
    for key, value in sorted(per_dose.items()):
        rows.bounded(
            f"TREAT p-values of each dose at 1.2- and 1.5-fold, limma-{key}, the "
            "cell type's fit",
            "limma 3.62.1 treat(fit, lfc = log2(fc)), topTreat(sort.by = 'none')",
            value,
            CLOSE if "robust" in key else EXACT,
        )
    for key, value in sorted(gate.items()):
        rows.bounded(
            f"gate p-values with a fold change of 1.2 and 1.5, limma-{key}",
            "R: pmin(1, m * apply(P, 1, min)) on treat's p-values",
            value,
            CLOSE if "robust" in key else EXACT,
        )
    rows.identical(
        "genes passing the 1.5-fold gate at 5% FDR",
        "R: p.adjust(p, 'BH') <= 0.05",
        *calls,
    )
    rows.identical(
        "lowest effective doses under the 1.5-fold gate",
        "stageR 1.28.0 stageWiseAdjustment(method = 'holm') on R's treat p-values",
        *lowest,
    )
    rows.bounded(
        "stage-wise adjusted p-values at the lowest effective doses, 1.5-fold gate",
        "stageR 1.28.0",
        stage_wise,
        CLOSE,
    )


def compare_assignments(
    found: list[Case], ours: Ours, blocks: dict, rows: Rows
) -> None:
    """The units, the counts of assignments and, where they are few, the assignments
    themselves, against R's reading of the sample table."""
    kinds = [0, 0]
    counts = [0, 0]
    for case in found:
        for entry in ours.checks[case.name].provenance["groups"]:
            theirs = blocks[case.name][entry["group_id"]]
            kinds[0] += entry.get("units") == theirs["kind"]
            kinds[1] += 1
            # Every count here lies below 2^53, where R's doubles hold it exactly.
            reference = float(theirs["n_assignments"])
            if reference >= 2.0**53:
                raise SystemExit(
                    f"{case.name}: {reference:.3g} assignments, beyond the integers "
                    "that R's doubles hold exactly; compare such counts relatively"
                )
            counts[0] += entry.get("n_assignments") == int(reference)
            counts[1] += 1
    rows.identical(
        "the units permuted: samples, whole units or doses within units, every case "
        "and cell type",
        "R's reading of the sample table, the design's terms and its unit",
        *kinds,
    )
    rows.identical(
        "assignments of the dose labels, every case and cell type, up to 1.4e11",
        "R: products of choose() over the blocks R builds",
        *counts,
    )
    same = [0, 0]
    for (name, key), check in ours.every.items():
        entry = check.provenance["groups"][0]
        theirs = blocks[name][entry["group_id"]]
        reference = {
            tuple(row) for row in np.atleast_2d(theirs["assignments"]).tolist()
        }
        mine = [tuple(entry["labels"])] + [tuple(a) for a in entry["assignments"]]
        same[0] += int(set(mine) == reference and len(mine) == len(set(mine)))
        same[1] += 1
    rows.identical(
        "every assignment, the true one and every other run once: duplicate (90) and "
        "sex (36), two gates each",
        "R's enumeration of the distinct label vectors of each block",
        *same,
    )


def compare_every_run(ours: Ours, blocks: dict, rows: Rows) -> None:
    """The genes each run of duplicate and sex passes, the share, the p-value and the
    smallest possible p-value, against R's limma on each assignment."""
    runs = [0, 0]
    statistics = 0.0
    smallest = [0, 0]
    for (name, key), check in ours.every.items():
        entry = check.provenance["groups"][0]
        theirs = blocks[name][entry["group_id"]]
        assignments = [tuple(row) for row in theirs["assignments"]]
        counts = dict(zip(assignments, vector(theirs["counts"][key]).astype(int)))
        table = check.table(entry["group_id"])
        mine = np.asarray(table["n_genes_passing_gate"], dtype=int)
        reference = np.array([counts[tuple(a)] for a in entry["assignments"]])
        runs[0] += int(np.sum(mine == reference))
        runs[1] += mine.size
        observed = counts[tuple(entry["labels"])]
        share = np.mean(reference >= 1)
        p = (1 + np.sum(reference >= observed)) / (1 + reference.size)
        row = check.summary
        statistics = max(
            statistics,
            abs(float(row["share_of_runs_with_passing_genes"][0]) - share),
            abs(float(row["p_value"][0]) - p) / p,
        )
        ties = int(theirs["ties"][key])
        smallest[0] += int(
            float(row["smallest_possible_p_value"][0]) == ties / len(assignments)
        )
        smallest[1] += 1
    rows.identical(
        "duplicate and sex, 89 and 35 runs of the dose-group gate and of the 1.5-fold "
        "gate: the genes each run passes",
        "limma 3.62.1 refitted by R on each assignment it enumerates",
        *runs,
    )
    rows.bounded(
        "duplicate and sex: the share of runs with genes and the p-value, from R's "
        "counts",
        "R's counts of every assignment",
        statistics,
        EXACT,
    )
    rows.identical(
        "duplicate and sex: the smallest possible p-value, 6/90 and 2/36 for the "
        "dose groups, 2/90 and 2/36 for the fold-change gate",
        "R: the ties, assignments whose refit gives the true doses' genes and "
        "p-values within 1e-10, over the assignments",
        *smallest,
    )


def compare_reruns(
    found: list[Case], ours: Ours, reruns: list, dream: list, rows: Rows
) -> None:
    """One recorded run of each case, made again by dose_response on the permuted
    design, against R's fit of the same doses; and the check's count of that run."""
    largest = 0.0
    calls = [0, 0]
    counts = [0, 0]
    dream_f = 0.0
    limma_jobs = iter(reruns)
    dream_jobs = iter(dream)
    for case in found:
        if case.analysis == "dream":
            # The true doses, then the recorded run, of each cell type.
            for entry in ours.checks[case.name].provenance["groups"]:
                if not entry["checked"]:
                    continue
                true = analyse(case, "dream", {"test": "dose_groups"})
                f = true.table(entry["group_id"], "dose_groups")["f_statistic"]
                dream_f = max(dream_f, scaled(f, next(dream_jobs)["F"]))
        check = ours.checks[case.name]
        for entry in check.provenance["groups"]:
            if not entry["checked"]:
                continue
            doses = recorded_doses(case, entry, 0)
            again = dose_response(
                case.cube,
                permuted(case, doses),
                dose=AXIS,
                gate={"test": "dose_groups"},
                analysis=ANALYSES["dream" if case.analysis == "dream" else "voom"],
                groups=[entry["group_id"]],
                **({"span": 0.5} if case.analysis == "dream" else {}),
            )
            table = again.table(entry["group_id"], "gate")
            counts[0] += int(
                np.count_nonzero(table["passes_gate"])
                == check.table(entry["group_id"])["n_genes_passing_gate"][0]
            )
            counts[1] += 1
            if case.analysis == "dream":
                theirs = next(dream_jobs)
                f = again.table(entry["group_id"], "dose_groups")["f_statistic"]
                dream_f = max(dream_f, scaled(f, theirs["F"]))
                continue
            theirs = next(limma_jobs)
            largest = max(largest, p_scaled(table["p_value"], theirs["p"]))
            called = vector(theirs["adjusted"]) <= FDR
            mine = np.asarray(table["passes_gate"], dtype=bool)
            calls[0] += int(np.sum(mine == called))
            calls[1] += mine.size
    rows.bounded(
        "one recorded run of each limma case: the dose-group gate's p-values on the "
        "permuted doses",
        "limma 3.62.1 refitted by R on the recorded doses",
        largest,
        EXACT,
    )
    rows.identical(
        "the same runs: genes passing at 5% FDR", "R: p.adjust(p, 'BH') <= 0.05", *calls
    )
    rows.identical(
        "the check's count of each recorded run, every case",
        "dose_response called on the permuted design",
        *counts,
    )
    rows.bounded(
        "dream, donors (doses within units) and animals (whole units in blocks by "
        "sex): the dose-group F-statistics of the true doses and of a recorded run",
        "variancePartition 1.36.2 dream, topTable",
        dream_f,
        CLOSE,
    )


# Fixtures: the limma cases and R's results for them.


def write_fixtures(found: list[Case], limma: dict) -> None:
    """The series of the limma cases, R's gates and TREAT at 1.5-fold for fixture
    and duplicate, and R's blocks, counts and every assignment of each."""
    record: dict[str, Any] = {"versions": limma["versions"], "cases": {}, "limma": {}}
    for case in found:
        if case.analysis != "limma":
            continue
        cube = case.cube
        design = case.design.samples
        record["cases"][case.name] = {
            "samples": {
                "sample_id": cube.samples["sample_id"].tolist(),
                "dose": np.asarray(cube.samples["dose"], dtype=float).tolist(),
            },
            "design": {
                name: design[name].tolist()
                for name in design.columns
                if name in ("sample_id", "dose", "sex")
            },
            "formula": case.formula,
            "groups": cube.groups["group_id"].tolist(),
            "features": cube.features["feature_id"].tolist(),
            "counts": [
                cube.layer("counts", g).astype(int).tolist()
                for g in range(cube.n_groups)
            ],
            "n_cells": np.asarray(cube.n_cells).astype(int).tolist(),
            "blocks": limma["blocks"][case.name],
        }
    for job, groups in limma["limma"].items():
        # The robust jobs, whose priors agree within 1e-6, stay in the script.
        if job.startswith("sex ") or job.endswith(" robust"):
            continue
        record["limma"][job] = {
            group: {
                **{key: values[key] for key in ("samples", "doses", "genes", "gates")},
                "treat": values["treat"][format(CHECKED_FOLD_CHANGE, "g")],
            }
            for group, values in groups.items()
        }
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIRECTORY / "limma.json"
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {path.name} to {FIXTURE_DIRECTORY}")


# The benchmark.


def benchmark() -> None:
    """permutation_check of 100 runs of one cell type of 20,000 genes x 24 profiles,
    the vehicle and 7 doses x 3, with the spline gate and with a fold change, and of
    5 runs with the Williams-type gate, against the baseline of the epic's targets:
    the weighted fit of 20,000 genes x 200 samples, as multiple_contrast_tests.py
    times it. The ordinary least squares fit that dose_contrasts.py times is printed
    beside it."""
    rng = np.random.default_rng(1)
    n_samples = 200
    groups = np.repeat(np.arange(4), n_samples // 4)
    x = np.column_stack(
        [np.ones(n_samples)]
        + [(groups == g).astype(float) for g in (1, 2, 3)]
        + [rng.normal(50.0, 10.0, n_samples), rng.normal(0.0, 1.0, n_samples)]
    )
    y = rng.normal(6.0, 1.0, (n_samples, 20_000))
    weights = np.exp(rng.normal(0.0, 0.5, y.shape))

    def timed(run: Any, repeats: int = 3) -> float:
        run()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        return float(np.median(times))

    baseline = timed(lambda: fit_linear_model(y, x, weights=weights))
    ordinary = timed(lambda: fit_linear_model(y, x))
    print(
        "Baseline: weighted fit_linear_model, 20,000 genes x 200 samples, "
        f"{baseline:.3f} s (ordinary least squares {ordinary:.3f} s)"
    )
    cube, _ = simulate_dose_series(
        [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=20_000,
        rng=20260906,
    )
    design = Design(cube.samples, "~ 1")
    print("| gate | runs | seconds | seconds per run | baselines | target |")
    print("| --- | --- | --- | --- | --- | --- |")
    for name, gate, runs in (
        ("spline", None, 100),
        ("1.5-fold", {"fold_change": 1.5}, 100),
        ("williams", {"test": "williams"}, 5),
    ):
        result = dose_response(cube, design, dose=AXIS, gate=gate, rng=1)
        seconds = timed(
            lambda result=result, runs=runs: permutation_check(
                result, n_permutations=runs, rng=1
            )
        )
        per_run = seconds / (runs + 1)
        print(
            f"| {name} | {runs} | {seconds:.1f} | {per_run:.2f} | "
            f"{seconds / baseline:.0f} | 350 per 100 runs, 525 at most |"
        )


# Main.


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument(
        "--benchmark", action="store_true", help="time permutation_check"
    )
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
        return 0
    found = cases()
    ours = limma_runs(found)
    check_runs(found, ours)
    payload = r_input(found, ours)
    limma = run_r(R_LIMMA, payload, "RSCRIPT")
    dream = run_r(R_DREAM, payload, "DREAM_RSCRIPT")
    versions = limma["versions"] | dream["versions"]
    print(
        "Reference: "
        + ", ".join(f"{name} {version}" for name, version in versions.items())
    )
    rows = Rows()
    compare_gates(found, ours, limma["limma"], rows)
    compare_fold_changes(found, ours, limma["limma"], rows)
    compare_assignments(found, ours, limma["blocks"], rows)
    compare_every_run(ours, limma["blocks"], rows)
    compare_reruns(
        found,
        ours,
        [limma["reruns"][key] for key in sorted(limma["reruns"], key=int)],
        [dream["dream"][key] for key in sorted(dream["dream"], key=int)],
        rows,
    )
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows.rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if rows.failed:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(found, limma)
    print(f"{len(rows.rows)} checks, {rows.failed} failed")
    return 1 if rows.failed else 0


if __name__ == "__main__":
    sys.exit(main())
