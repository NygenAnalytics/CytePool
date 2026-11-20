"""Check the trend contrasts, the MCP step and the lowest effective dose against
DoseFinding, multcomp, mvtnorm, stageR, limma and variancePartition.

Inputs come from fixed seeds (``SEED`` 20261216 and the seeds after it): the dose
ladders of the candidate shapes; seeded estimates, covariances and p-values; and six
dose series simulated by ``simulate_dose_series``: ``fixture``, the vehicle and 5 doses
from 0.1 to 10 ug/kg x 3, 200 genes, a cell type of every sample and one that loses the
top dose and one sample of 3 ug/kg (fewer than 10 cells); ``ladder``, the vehicle and
8 half-log doses from 0.01 to 30 ug/kg x 3, 1,000 genes, the same two kinds of cell
type; ``four_groups``, the vehicle and 3 decades x 2; ``three_groups``, the vehicle
and 1 and 10 ug/kg x 3, the minimum EMA names; ``linear``, the vehicle and 25 to 100
mg/kg in steps of 25 x 3 on a linear axis; and ``donors``, 4 donors crossed with 5
dose groups under ``(1 | donor)``.

References, each run as a separate program on inputs it sees as numbers only:

- DoseFinding 1.4-2: ``guesst`` for the exponential root and the beta model's closed
  form that ``CandidateShapes.spread`` places, the ED50s of its Emax and sigmoid
  Emax shapes computed in R from the ladder and the rule; ``getResp(Mods(...))`` for
  ``CandidateShapes.means``; ``optContr`` with ``w = n``, a general ``S`` and
  ``placAdj = TRUE`` for ``optimal_contrasts``; ``MCTtest(type = "general")`` for the
  contrasts and t-statistics of the MCP step on seeded inputs with integer df.
- multcomp 1.4-32 and mvtnorm 1.4-2: ``summary(glht(parm(theta, V, df)))`` for the
  p-values of seeded inputs, singular correlations included, each with its error
  attribute; ``contrMat`` and ``glht(parm())`` for the contrasts and statistics of
  ``dose_response``'s three tests on the dose series.
- limma 3.62.1 and edgeR 4.4.0 (``RSCRIPT``): each cell type fitted from the counts,
  as ``dose_response`` fits it (``filterByExpr``, ``calcNormFactors``, ``voom`` with
  ``adaptive.span = TRUE`` or ``cpm(log = TRUE, prior.count = 3)`` with ``eBayes(trend
  = TRUE)``, ``lmFit``, ``contrasts.fit`` to the treated doses, ``eBayes``), with each
  gene's covariance of the doses' estimates from its own ``lm.wfit`` (voom's exact
  default), from limma's ``contrasts.fit`` (``contrast_standard_errors="limma"``), or
  shared (limma-trend).
- variancePartition 1.36.2 (``DREAM_RSCRIPT``): ``voomWithDreamWeights`` and ``dream``
  for the donors series; each gene's covariance from ``cov.coefficients.list`` over
  sigma^2 times ``s2.post``.
- stageR 1.28.0: ``stageWiseAdjustment`` with ``"holm"`` and ``"none"``, ``allowNA``
  where values are missing, for ``stage_wise_adjusted_p_values`` and the lowest
  effective doses; and, coded for this epic, the closure of Bonferroni tests, every
  intersection of a called gene's hypotheses enumerated, the intersection of all of
  them taken as rejected by the screen.
- mpmath 1.3.0, at 40 digits, for the exponential shapes' roots, and scipy 1.18.1's
  ``multivariate_t.cdf`` (8 calls with different random states) for the p-values of
  the dose series at their non-integer df, which MCTtest and multcomp refuse, on full
  rank correlations only: scipy's ``allow_singular`` complements reached three times
  Bonferroni's bound on singular correlations in ``multiple_contrast_tests.py``. Both
  run in a process of their own, this script run as ``--reference DIRECTORY``, which
  imports them and not CytePool.

Tiers: exact (1e-10 relative, statistics relative to max(|reference|, 1)); close
(1e-6) for dream's statistics, and 1e-5 for its degrees of freedom and against
guesst's exponential root, which guesst finds by a search to about 2e-6; identical for
doses, counts and names; calibrated: |CytePool -
reference| at most 5.5 combined standard errors, the 99.994% point of the t
distribution on 15 degrees of freedom, on which CytePool's errors are estimated, with
mvtnorm's standard error its error attribute over 3 and scipy's from the spread of its
calls. CytePool runs at ``relative_error=1e-3`` for the seeded p-values.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root so that
    ``cytepool`` imports from the working tree; mpmath and scipy for the references.
    R 4.4 with limma, edgeR and jsonlite (``RSCRIPT``); DoseFinding, multcomp, mvtnorm,
    stageR and jsonlite (``DOSE_RSCRIPT``); variancePartition, lme4, edgeR and
    jsonlite (``DREAM_RSCRIPT``). Each variable names an Rscript executable and
    defaults to ``Rscript``.

Usage:
    PYTHONPATH=. python validation/dose_contrasts.py
    PYTHONPATH=. python validation/dose_contrasts.py --write-fixtures
    PYTHONPATH=. python validation/dose_contrasts.py --benchmark

``--benchmark`` times ``dose_response`` with each of the three tests on 20,000 genes x
24 profiles, the vehicle and 7 doses x 3, and on 20,000 genes with the vehicle and 3
doses x 3, against the target of 1 ms per gene on 2 threads, beside the default tests,
with the ordinary least squares fit of 20,000 genes x 200 samples of
``linear_models.py --benchmark`` as the baseline.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.candidate_shapes import CandidateShapes
from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.differential_expression import dream_voom, limma_trend, limma_voom
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import dose_response
from cytepool.simulated_dose_series import simulate_dose_series
from cytepool.stats.linear_models import fit_linear_model
from cytepool.stats.multiple_contrasts import (
    multiple_contrast_test,
    optimal_contrasts,
    stage_wise_adjusted_p_values,
)

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_contrasts"
)
SEED = 20261216
EXACT = 1e-10
CLOSE = 1e-6
GUESST = 1e-5  # guesst's exponential root, which it finds by a search
CALIBRATED = 5.5
TIGHT = 1e-3  # relative_error of CytePool in the seeded p-value checks
MIN_CELLS = 10
ALPHA = 0.05
SCIPY_CALLS = 8
SCIPY_GENES = 100  # genes per cell type and test whose p-values scipy computes
BRACKET_GENES = 30  # genes per singular cell type whose p-values mvtnorm brackets
SMALLEST_P = 1e-3  # p-values below this are beyond mvtnorm's and scipy's resolution
TESTS = ("dunnett", "williams", "mcp")
HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
LOG_AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
LINEAR_AXIS = DoseAxis("dose", unit="mg/kg", metric="administered", scale="linear")

LIMMA = r"""
suppressPackageStartupMessages({library(limma); library(edgeR); library(jsonlite)})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("limma", "edgeR"), function(p) as.character(packageVersion(p))))))
# Each job: a series, an analysis ("voom", "voom_limma" with limma's standard errors,
# or "trend") and its cell types. R keeps each cell type's samples of min_cells cells
# or more and some counts, the treated doses that keep a sample and that limma's
# nonEstimable does not name with their indicator placed last, and fits the design's
# terms with an indicator of each such dose.
for (name in names(input$jobs)) {
  job <- input$jobs[[name]]
  case <- input$series[[job$series]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  levels <- sort(unique(data$dose))
  result <- list()
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    used <- cells >= input$min_cells & colSums(y) > 0
    d <- data[used, , drop = FALSE]; yy <- y[, used, drop = FALSE]
    treated <- levels[levels > 0 & levels %in% d$dose]
    others <- model.matrix(as.formula(case$formula), d)
    ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
    colnames(ind) <- paste0("dose", seq_along(treated))
    estimable <- sapply(seq_along(treated), function(j) {
      X <- cbind(others, ind[, -j, drop = FALSE], ind[, j, drop = FALSE])
      !(colnames(ind)[j] %in% nonEstimable(X))
    })
    compared <- c(0, treated[estimable])
    n <- sapply(compared, function(k) sum(d$dose == k))
    X <- cbind(others, ind[, estimable, drop = FALSE])
    k <- length(compared) - 1
    columns <- ncol(others) + seq_len(k)
    L <- rbind(matrix(0, ncol(others), k), diag(k))
    dge <- DGEList(yy); keep <- filterByExpr(dge, design = X)
    dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
    voomed <- job$analysis %in% c("voom", "voom_limma")
    if (voomed) {
      v <- voom(dge, X, adaptive.span = TRUE); E <- v$E; W <- v$weights; fit <- lmFit(v, X)
    } else {
      E <- cpm(dge, log = TRUE, prior.count = 3); W <- NULL; fit <- lmFit(E, X)
    }
    eb <- eBayes(contrasts.fit(fit, L), trend = !voomed)
    s2 <- eb$s2.post; df_total <- eb$df.total
    theta <- unname(eb$coefficients)
    # The unscaled covariance of each gene's theta.
    if (job$analysis == "voom") {
      V <- lapply(seq_len(nrow(E)), function(g) {
        f <- lm.wfit(X, E[g, ], W[g, ])
        stopifnot(identical(f$qr$pivot, seq_len(ncol(X))))
        chol2inv(qr.R(f$qr))[columns, columns, drop = FALSE]
      })
    } else if (job$analysis == "voom_limma") {
      # limma's contrasts.fit: each gene's unscaled standard errors with the
      # correlation of the design without weights.
      R <- cov2cor(eb$cov.coefficients)
      V <- lapply(seq_len(nrow(E)), function(g) {
        su <- eb$stdev.unscaled[g, ]; R * outer(su, su)
      })
    } else {
      V <- lapply(seq_len(nrow(E)), function(g) eb$cov.coefficients)
    }
    each_t <- unname(eb$t); each_p <- unname(eb$p.value)
    # The dose-group test: theta' (s2 V)^-1 theta / k on k and df.total.
    f <- sapply(seq_len(nrow(E)), function(g)
      drop(crossprod(theta[g, ], solve(V[[g]], theta[g, ]))) / k / s2[g])
    result$cell_types[[ct]] <- list(
      samples = d$sample_id, doses = compared, n = n, genes = which(keep) - 1,
      theta = theta, V = matrix(unlist(lapply(V, as.numeric)), ncol = k * k, byrow = TRUE),
      s2_post = s2, df_total = df_total,
      each_t = each_t, each_p = each_p, f = f,
      f_p = pf(f, k, df_total, lower.tail = FALSE))
  }
  out$jobs[[name]] <- result
}
write_json(out, args[[2]], digits = I(17), matrix = "rowmajor", auto_unbox = TRUE, na = "null")
"""

DREAM = r"""
suppressPackageStartupMessages({
  library(variancePartition); library(lme4); library(edgeR); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE,
                     check.conv.singular = "ignore")
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("variancePartition", "limma", "edgeR", "lme4"),
  function(p) as.character(packageVersion(p))))))
# Each cell type's dose-group estimates from dream: theta, the unscaled covariance of
# each gene's theta (cov.coefficients.list over sigma^2), s2.post and df.total.
for (name in names(input$jobs)) {
  job <- input$jobs[[name]]
  case <- input$series[[job$series]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  levels <- sort(unique(data$dose))
  result <- list()
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    used <- cells >= input$min_cells & colSums(y) > 0
    d <- data[used, , drop = FALSE]
    treated <- levels[levels > 0 & levels %in% d$dose]
    ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
    colnames(ind) <- paste0("dose", seq_along(treated))
    d <- cbind(d, ind); d$donor <- factor(d$donor)
    X <- cbind("(Intercept)" = 1, ind)
    dge <- DGEList(y[, used, drop = FALSE]); keep <- filterByExpr(dge, design = X)
    dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
    form <- as.formula(paste("~", paste(colnames(ind), collapse = " + "), "+ (1 | donor)"))
    v <- quiet(voomWithDreamWeights(dge, form, d, span = 0.5, control = tight))
    L <- rbind(0, diag(ncol(ind)))
    rownames(L) <- colnames(X); colnames(L) <- paste0("L", seq_len(ncol(ind)))
    fit <- quiet(dream(v, form, d, L = L, control = tight))
    e <- eBayes(fit)
    cols <- colnames(L)
    k <- length(cols)
    V <- matrix(unlist(lapply(seq_len(nrow(e$coefficients)), function(g)
      as.numeric(fit$cov.coefficients.list[[g]][cols, cols] / fit$sigma[g]^2))),
      ncol = k * k, byrow = TRUE)
    result$cell_types[[ct]] <- list(
      samples = d$sample_id, doses = c(0, treated),
      n = sapply(c(0, treated), function(z) sum(d$dose == z)), genes = which(keep) - 1,
      theta = unname(e$coefficients[, cols, drop = FALSE]), V = V,
      s2_post = unname(e$s2.post), df_total = unname(e$df.total[, cols, drop = FALSE]),
      each_t = unname(e$t[, cols, drop = FALSE]), each_p = unname(e$p.value[, cols, drop = FALSE]))
  }
  out$jobs[[name]] <- result
}
write_json(out, args[[2]], digits = I(17), matrix = "rowmajor", auto_unbox = TRUE, na = "null")
"""

DOSE = r"""
suppressPackageStartupMessages({
  library(DoseFinding); library(multcomp); library(mvtnorm); library(stageR)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE, simplifyDataFrame = FALSE)
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("DoseFinding", "multcomp", "mvtnorm", "stageR"),
  function(p) as.character(packageVersion(p))))))
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
pairs_of <- function(x) {
  # JSON [[a, b], [c, d]] reads as a 2-column matrix, one shape per row.
  if (!length(x)) return(NULL)
  if (is.matrix(x)) return(x)
  matrix(unlist(x), ncol = 2, byrow = TRUE)
}
# The candidate models in the order emax, sigEmax, exponential, betaMod.
models_of <- function(shapes, doses) {
  a <- list(doses = doses)
  if (length(shapes$emax)) a$emax <- shapes$emax
  if (length(shapes$sigmoid_emax)) a$sigEmax <- pairs_of(shapes$sigmoid_emax)
  if (length(shapes$exponential)) a$exponential <- shapes$exponential
  if (length(shapes$beta)) {
    a$betaMod <- pairs_of(shapes$beta); a$addArgs <- list(scal = shapes$beta_scale)
  }
  do.call(Mods, a)
}
# The axis's positions and their inverse.
position_of <- function(axis) {
  if (axis$scale == "linear") return(function(d) d)
  if (axis$vehicle == "on_axis") return(function(d) log10(d + axis$offset))
  function(d) log10(d)
}
dose_of <- function(axis) {
  if (axis$scale == "linear") return(function(x) x)
  if (axis$vehicle == "on_axis") return(function(x) 10^x - axis$offset)
  function(x) 10^x
}

## The rule of CandidateShapes.spread, computed from the ladder, and guesst's
## exponential root and beta closed form for the parameters it places.
out$spread <- lapply(input$spread, function(case) {
  position <- position_of(case$axis); dose <- dose_of(case$axis)
  treated <- sort(unique(case$doses[case$doses > 0])); top <- max(treated)
  low <- position(min(treated)); high <- position(top)
  placed <- function(m) if (m > 0) low + seq_len(m) / (m + 1) * (high - low) else numeric(0)
  counts <- case$counts
  peaks <- placed(counts$beta)
  scal <- 1.2 * top
  halves <- if (counts$exponential > 0)
    (0.5 + seq_len(counts$exponential) / (2 * (counts$exponential + 1))) * top else numeric(0)
  list(
    emax = dose(placed(counts$emax)),
    sigmoid_ed50 = dose(placed(counts$sigmoid_emax)),
    exponential_half = halves,
    exponential = vapply(halves, function(h)
      unname(guesst(d = h, p = 0.5, model = "exponential", Maxd = top)), 0.0),
    beta_peak = dose(peaks), beta_half = dose((low + peaks) / 2),
    beta = lapply(peaks, function(pk) unname(guesst(
      d = dose((low + pk) / 2), p = 0.5, model = "betaMod", dMax = dose(pk), scal = scal,
      Maxd = top))),
    beta_scale = scal)
})

## Means of the shapes at the doses: getResp(Mods(...)).
out$means <- lapply(input$means, function(case) {
  m <- models_of(case$shapes, case$doses)
  r <- getResp(m, doses = case$doses)
  list(names = colnames(r), means = t(unclass(r)))
})

## An exponential shape of delta D / 1000, whose exp(D / delta) overflows.
out$overflow <- tryCatch({
  m <- models_of(input$overflow$shapes, input$overflow$doses)
  as.numeric(getResp(m, doses = input$overflow$doses))
}, error = function(e) conditionMessage(e))

## Optimal contrasts: w = n, a general S, and placebo-adjusted with V = D S D'.
out$contrasts <- lapply(input$contrasts, function(case) {
  m <- models_of(case$shapes, case$doses)
  S <- case$S; k <- length(case$doses)
  D <- cbind(-1, diag(k - 1)); V <- D %*% S %*% t(D)
  list(w = unclass(optContr(m, w = case$n)$contMat),
       S = unclass(optContr(m, S = S)$contMat),
       placebo_adjusted = unclass(optContr(m, S = V, placAdj = TRUE, doses = case$doses[-1])$contMat))
})

## MCTtest on seeded inputs with integer df: contrasts and t; the p-values of
## summary(glht(parm(theta, V, df))), which carry their error.
out$mct <- lapply(input$mct, function(case) {
  m <- models_of(case$shapes, case$doses)
  r <- MCTtest(dose = case$doses[-1], resp = case$theta, S = case$V, type = "general",
               placAdj = TRUE, models = m, df = case$df, alternative = "two.sided")
  C <- unclass(r$contMat)
  set.seed(case$seed)
  g <- glht(parm(case$theta, case$V, df = case$df), linfct = t(C))
  s <- tryCatch(summary(g), error = function(e) NULL)
  list(contrasts = C, t = as.numeric(r$tStat),
       glht_t = if (is.null(s)) NULL else as.numeric(s$test$tstat),
       p = if (is.null(s)) NULL else as.numeric(s$test$pvalues),
       error = if (is.null(s)) NULL else attr(s$test$pvalues, "error"))
})
# MCTtest and glht refuse a non-integer df.
out$refusals <- list(
  mcttest = tryCatch({
    m <- models_of(input$mct[[1]]$shapes, input$mct[[1]]$doses)
    MCTtest(dose = input$mct[[1]]$doses[-1], resp = input$mct[[1]]$theta,
            S = input$mct[[1]]$V, type = "general", placAdj = TRUE, models = m,
            df = 12.7, alternative = "two.sided"); "accepted"
  }, error = function(e) conditionMessage(e)),
  glht = tryCatch({
    summary(glht(parm(c(1, 2), diag(2), df = 12.7), linfct = diag(2))); "accepted"
  }, error = function(e) conditionMessage(e)))

## stageR's stage-wise adjustment, and the closure of Bonferroni tests coded for this
## epic: a hypothesis is rejected when every intersection of the hypotheses that holds
## it is rejected, the intersection of all m by the screen, the others by Bonferroni at
## R alpha / G.
stage_wise <- function(screen, p, alpha, method) {
  names(screen) <- paste0("f", seq_along(screen))
  dimnames(p) <- list(names(screen), paste0("h", seq_len(ncol(p))))
  o <- tryCatch(quiet(stageWiseAdjustment(
    stageR(pScreen = screen, pConfirmation = p, pScreenAdjusted = FALSE),
    method = method, alpha = alpha, allowNA = anyNA(screen) || anyNA(p))),
    error = function(e) conditionMessage(e))
  if (is.character(o)) return(list(error = o))
  a <- quiet(getAdjustedPValues(o, onlySignificantGenes = FALSE, order = FALSE))
  full <- matrix(NA_real_, nrow(p), ncol(p))
  full[match(rownames(a), names(screen)), ] <- a[, -1, drop = FALSE]
  list(adjusted = full)
}
closure <- function(screen, p, alpha) {
  q <- p.adjust(screen, "BH"); G <- sum(!is.na(screen)); called <- which(q <= alpha)
  rejected <- matrix(FALSE, nrow(p), ncol(p))
  if (!length(called)) return(rejected)
  level <- length(called) * alpha / G
  for (i in called) {
    present <- which(!is.na(p[i, ])); m <- length(present)
    if (!m) next
    if (m == 1) {
      # One hypothesis left: Bonferroni at the level, not the screen alone, which is
      # the hypothesis's own test only when nothing else was tested.
      rejected[i, present] <- p[i, present] <= level
      next
    }
    subsets <- unlist(lapply(seq_len(m - 1), function(s)
      combn(present, s, simplify = FALSE)), recursive = FALSE)
    for (j in present) {
      tests <- Filter(function(I) j %in% I, subsets)
      # The intersection of all m hypotheses is the screen's, which called the feature.
      ok <- all(vapply(tests, function(I) min(p[i, I]) <= level / length(I), TRUE))
      rejected[i, j] <- ok
    }
  }
  rejected
}
out$stager <- lapply(input$stager, function(case) {
  p <- if (is.matrix(case$p)) case$p else matrix(case$p, ncol = 1)
  list(adjusted = stage_wise(case$screen, p, case$alpha, case$method),
       closure = closure(case$screen, p, case$alpha))
})

## The dose series: each cell type's contrasts, as multcomp's contrMat and
## DoseFinding's optContr build them from its doses and samples, and their statistics
## from glht(parm(theta, s2 V)) for every gene; stageR and the closure on the screen and
## per-dose p-values given.
out$cubes <- lapply(input$cubes, function(job) {
  n <- job$n; doses <- job$doses; k <- length(doses) - 1
  names(n) <- c("vehicle", paste0("d", seq_len(k)))
  sets <- list()
  if (k >= 2) {
    sets$dunnett <- unclass(contrMat(n, type = "Dunnett"))[, -1, drop = FALSE]
    sets$williams <- unclass(contrMat(n, type = "Williams"))[, -1, drop = FALSE]
  } else {
    sets$dunnett <- matrix(1, 1, 1); sets$williams <- matrix(1, 1, 1)
  }
  if (!is.null(job$shapes) && k >= 2) {
    m <- models_of(job$shapes, doses)
    C <- unclass(optContr(m, w = n)$contMat)[-1, , drop = FALSE]
    C <- sweep(C, 2, sqrt(colSums(C^2)), "/")
    sets$mcp <- t(C)
  }
  theta <- if (is.matrix(job$theta)) job$theta else matrix(job$theta, ncol = 1)
  s2 <- job$s2_post
  V <- if (is.matrix(job$V)) job$V else matrix(job$V, ncol = 1)
  stats <- lapply(sets, function(K) {
    matrix(unlist(lapply(seq_len(nrow(theta)), function(g) {
      cov <- matrix(V[g, ], k, k) * s2[g]
      gl <- glht(parm(theta[g, ], cov), linfct = K)
      as.numeric(coef(gl) / sqrt(diag(vcov(gl))))
    })), ncol = nrow(K), byrow = TRUE)
  })
  res <- list(contrasts = sets, t = stats)
  # mvtnorm handles singular correlations but not a non-integer df: the gene's
  # p-value of the MCP step at the integer df below and above its own.
  if (length(job$bracket) && !is.null(sets$mcp)) {
    res$bracket <- lapply(job$bracket, function(g) {
      cov <- matrix(V[g, ], k, k) * s2[g]
      sapply(c(floor(job$df[g]), ceiling(job$df[g])), function(df) {
        set.seed(g)
        s <- summary(glht(parm(theta[g, ], cov, df = df), linfct = sets$mcp))
        c(min(s$test$pvalues), attr(s$test$pvalues, "error"))
      })
    })
  }
  for (screen in names(job$screens)) {
    sp <- job$screens[[screen]]
    p <- if (is.matrix(sp$p)) sp$p else matrix(sp$p, ncol = 1)
    res$stager[[screen]] <- stage_wise(sp$screen, p, job$alpha, "holm")
    res$closure[[screen]] <- closure(sp$screen, p, job$alpha)
  }
  res
})
write_json(out, args[[2]], digits = I(17), matrix = "rowmajor", auto_unbox = TRUE, na = "null")
"""


# ---------------------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------------------

LADDERS = {
    "half-log": HALF_LOG,
    "decades": [0.0, 0.1, 1.0, 10.0, 100.0],
    "uneven": [0.0, 0.05, 0.2, 2.0, 3.0, 50.0],
    "two doses": [0.0, 1.0, 10.0],
    "even steps": [0.0, 25.0, 50.0, 75.0, 100.0],
}
COUNTS = [(2, 1, 1, 1), (3, 2, 3, 2), (1, 3, 0, 3), (0, 0, 1, 0)]
FAMILIES = ("emax", "sigmoid_emax", "exponential", "beta")


def placements(doses: list[float]) -> dict[str, DoseAxis]:
    """The three placements of the vehicle and scales of the axis."""
    lowest = min(d for d in doses if d > 0)
    return {
        "log10, vehicle apart": LOG_AXIS,
        "log10, vehicle on the axis": DoseAxis(
            "dose",
            unit="ug/kg",
            metric="administered",
            vehicle="on_axis",
            offset=lowest / 10,
        ),
        "linear": LINEAR_AXIS,
    }


def spread_cases() -> list[dict[str, Any]]:
    """Five ladders, three placements and four sets of counts."""
    cases = []
    for ladder, doses in LADDERS.items():
        for placement, axis in placements(doses).items():
            for counts in COUNTS:
                cases.append(
                    {
                        "ladder": ladder,
                        "placement": placement,
                        "doses": doses,
                        "axis": {
                            "scale": axis.scale,
                            "vehicle": axis.vehicle,
                            "offset": axis.offset,
                        },
                        "counts": dict(zip(FAMILIES, counts)),
                    }
                )
    return cases


def axis_of(record: dict[str, Any]) -> DoseAxis:
    return DoseAxis(
        "dose",
        unit="ug/kg",
        metric="administered",
        scale=record["scale"],
        vehicle=record["vehicle"],
        offset=record["offset"],
    )


def means_cases() -> list[dict[str, Any]]:
    """Six designs from 3 to 9 dose groups, with shapes spread on three placements
    and shapes of extreme parameters: a beta shape peaking between doses and one
    peaking far above the top dose, an Emax shape of ED50 1e-4, sigmoid slopes of 0.5
    and 8, and exponential shapes of delta D / 500 and 1000 D (DoseFinding's exp(d /
    delta) - 1 loses digits for larger delta)."""
    on_axis = DoseAxis(
        "dose", unit="ug/kg", metric="administered", vehicle="on_axis", offset=0.001
    )
    designs = [
        (
            "half-log, spread",
            HALF_LOG,
            CandidateShapes.spread(LOG_AXIS, {"dose": HALF_LOG}),
        ),
        (
            "two doses, five shapes",
            [0.0, 1.0, 10.0],
            CandidateShapes(
                emax=[0.5, 5.0],
                sigmoid_emax=[(2.0, 3.0)],
                exponential=[4.0],
                beta=[(1.0, 2.0)],
                beta_scale=12.0,
            ),
        ),
        (
            "decades, beta peaks between doses and far above the top",
            [0.0, 0.1, 1.0, 10.0],
            CandidateShapes(
                emax=[0.03], beta=[(0.3, 10.0), (1.0, 0.1)], beta_scale=100.0
            ),
        ),
        (
            "uneven, extreme parameters",
            [0.0, 0.05, 0.2, 2.0, 3.0, 50.0],
            CandidateShapes(
                emax=[1e-4],
                sigmoid_emax=[(100.0, 0.5), (0.1, 8.0)],
                exponential=[0.1, 5e4],
            ),
        ),
        (
            "even steps, spread on a linear axis",
            [0.0, 25.0, 50.0, 75.0, 100.0],
            CandidateShapes.spread(
                LINEAR_AXIS, {"dose": [0.0, 25.0, 50.0, 75.0, 100.0]}
            ),
        ),
        (
            "vehicle on the axis, spread",
            [0.0, 0.01, 0.1, 1.0, 10.0],
            CandidateShapes.spread(on_axis, {"dose": [0.0, 0.01, 0.1, 1.0, 10.0]}),
        ),
    ]
    return [
        {"label": label, "doses": doses, "shapes": shapes.to_dict()}
        for label, doses, shapes in designs
    ]


def random_covariance(rng: np.random.Generator, k: int) -> np.ndarray:
    a = rng.standard_normal((k, k))
    covariance: np.ndarray = a @ a.T / k + np.diag(rng.uniform(0.05, 0.3, k))
    return covariance


def contrast_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """The designs of ``means_cases`` with unequal group sizes and a random S."""
    return [
        {
            **case,
            "n": rng.integers(2, 6, size=len(case["doses"])).tolist(),
            "S": random_covariance(rng, len(case["doses"])).tolist(),
        }
        for case in means_cases()
    ]


def mct_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """Eight features per design and df 5, 12 and 40: dose-group differences theta
    with covariance V = D S D' for independent groups S, some flat and some following
    a shape, so that p-values range from 1e-3 to 1. With five shapes, the designs of
    two and three treated doses give singular correlations."""
    cases = []
    seed = SEED + 1000
    for design in means_cases():
        doses = design["doses"]
        k = len(doses)
        means = CandidateShapes(**design["shapes"]).means(doses)
        difference = np.hstack([-np.ones((k - 1, 1)), np.eye(k - 1)])
        for df in (5, 12, 40):
            for _ in range(8):
                groups = np.diag(1.0 / rng.integers(2, 5, size=k)) * rng.uniform(0.2, 1)
                v = difference @ groups @ difference.T
                effect = rng.choice([0.0, 0.5, 1.0, 2.0]) * rng.choice([-1.0, 1.0])
                shape = means[rng.integers(means.shape[0])]
                theta = rng.multivariate_normal(effect * (shape[1:] - shape[0]), v)
                seed += 1
                cases.append(
                    {
                        "label": design["label"],
                        "doses": doses,
                        "shapes": design["shapes"],
                        "S": groups.tolist(),
                        "theta": theta.tolist(),
                        "V": v.tolist(),
                        "df": df,
                        "seed": seed,
                    }
                )
    return cases


def stager_cases(rng: np.random.Generator) -> list[dict[str, Any]]:
    """60 features with 1, 2, 3, 5 and 8 hypotheses, screens from 1e-9 to 1, ties within
    features, missing hypotheses and screen p-values, at alpha 0.05 and 0.1 (which
    call different numbers of features), with "holm" and "none"; and a screen that
    calls no feature."""
    cases = []
    features = 60
    for m in (1, 2, 3, 5, 8):
        screen = np.concatenate(
            [
                10.0 ** rng.uniform(-9, -4, 8),
                10.0 ** rng.uniform(-3.2, -1.6, 12),
                rng.uniform(0, 1, features - 20),
            ]
        )
        p = rng.uniform(0, 1, (features, m)) ** 3
        p[:5] = np.round(p[:5], 2)
        p[5, :] = p[5, 0]
        # stageR's "holm" gives a called feature with one hypothesis left 0, and fails
        # when such a feature, or one with none left, had 3 hypotheses or fewer (the
        # deliberate differences below), so values go missing only where it does not.
        if m >= 3:
            p[6, 1] = np.nan
        if m >= 4:
            p[7, :] = np.nan
        if m >= 2:
            screen[9] = np.nan
        order = rng.permutation(features)
        for alpha in (0.05, 0.1):
            for method in ("holm", "none"):
                cases.append(
                    {
                        "label": f"{m} hypotheses",
                        "m": m,
                        "screen": screen[order].tolist(),
                        "p": p[order].tolist(),
                        "alpha": alpha,
                        "method": method,
                    }
                )
    # One hypothesis left of four, in a called feature, beside complete features.
    screen = np.concatenate([[1e-8, 1e-7, 1e-6], rng.uniform(0, 1, 17)])
    p = rng.uniform(0, 1, (20, 4)) ** 3
    p[1, :3] = np.nan
    cases.append(
        {
            "label": "one hypothesis left of four",
            "m": 4,
            "screen": screen.tolist(),
            "p": p.tolist(),
            "alpha": ALPHA,
            "method": "holm",
            "one_left": 1,
        }
    )
    cases.append(
        {
            "label": "a screen that calls no feature",
            "m": 3,
            "screen": rng.uniform(0.5, 1, 20).tolist(),
            "p": (rng.uniform(0, 1, (20, 3)) ** 3).tolist(),
            "alpha": ALPHA,
            "method": "holm",
        }
    )
    return cases


@dataclass
class Series:
    """A simulated dose series, its axis and the analyses run on it."""

    name: str
    cube: Cube
    axis: DoseAxis
    analyses: tuple[str, ...]  # "voom", "voom_limma", "trend", "dream"

    @property
    def design(self) -> Design:
        random = ["(1 | donor)"] if "dream" in self.analyses else None
        return Design(self.cube.samples, "~ 1", random=random)


def series() -> list[Series]:
    """The dose series, from fixed seeds."""
    found = []

    def add(name, doses, axis, analyses, *, seed, **options):
        cube, _ = simulate_dose_series(doses, rng=seed, **options)
        found.append(Series(name, cube, axis, analyses))

    add(
        "fixture",
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        LOG_AXIS,
        ("voom", "voom_limma", "trend"),
        seed=SEED,
        replicates=3,
        cell_types=["full", "depleted"],
        cells=[300.0, 200.0],
        n_genes=200,
        composition={"depleted": [1.0, 1.0, 1.0, 1.0, 0.05, 0.015]},
    )
    add(
        "ladder",
        HALF_LOG,
        LOG_AXIS,
        ("voom", "voom_limma", "trend"),
        seed=SEED + 1,
        replicates=3,
        cell_types=["abundant", "depleted"],
        cells=[300.0, 200.0],
        n_genes=1000,
        composition={"depleted": [1.0] * 7 + [0.05, 0.015]},
    )
    add(
        "four_groups",
        [0.0, 0.1, 1.0, 10.0],
        LOG_AXIS,
        ("voom", "trend"),
        seed=SEED + 2,
        replicates=2,
        cell_types=1,
        cells=300.0,
        n_genes=500,
    )
    add(
        "three_groups",
        [0.0, 1.0, 10.0],
        LOG_AXIS,
        ("voom", "trend"),
        seed=SEED + 3,
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=500,
    )
    add(
        "linear",
        [0.0, 25.0, 50.0, 75.0, 100.0],
        LINEAR_AXIS,
        ("voom", "trend"),
        seed=SEED + 4,
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=500,
        unit="mg/kg",
    )
    add(
        "donors",
        [0.0, 0.1, 1.0, 10.0, 100.0],
        LOG_AXIS,
        ("dream",),
        seed=SEED + 5,
        replicates=4,
        cell_types=1,
        cells=300.0,
        n_genes=300,
        donor_sd=0.3,
    )
    return found


def series_record(item: Series) -> dict[str, Any]:
    """What R sees of a series: the samples' ids, doses and donors, and each cell
    type's counts (genes x samples) and cells."""
    cube = item.cube
    samples = cube.samples
    cell_types = {}
    for g, group in enumerate(cube.groups["group_id"].tolist()):
        cell_types[group] = {
            "counts": np.asarray(cube.layer("counts", g)).T.astype(np.int64).tolist(),
            "n_cells": np.asarray(cube.n_cells)[:, g].astype(np.int64).tolist(),
        }
    return {
        "samples": {
            "sample_id": samples["sample_id"].tolist(),
            "dose": np.asarray(samples["dose"], dtype=np.float64).tolist(),
            "donor": samples["donor"].tolist(),
        },
        "formula": "~ 1",
        "cell_types": cell_types,
    }


# ---------------------------------------------------------------------------------------
# Running the references
# ---------------------------------------------------------------------------------------


def plain(value: Any) -> Any:
    """Arrays as lists and NaN as None, which jsonlite reads as NA."""
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return plain(value.item())
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def numbers(value: Any) -> np.ndarray:
    """JSON numbers as a float64 array, with None (R's NA) as NaN."""
    return np.array(_with_nan(value), dtype=np.float64)


def _with_nan(value: Any) -> Any:
    if isinstance(value, list):
        return [_with_nan(item) for item in value]
    return np.nan if value is None else value


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    """Run an R program on a JSON payload; returns what it writes."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "program.R").write_text(program)
        (root / "input.json").write_text(json.dumps(plain(payload)))
        command = [
            os.environ.get(variable, "Rscript"),
            "--vanilla",
            str(root / "program.R"),
            str(root / "input.json"),
            str(root / "output.json"),
        ]
        subprocess.run(command, check=True)
        result: dict[str, Any] = json.loads((root / "output.json").read_text())
    return result


def run_reference(payload: dict[str, Any]) -> dict[str, Any]:
    """Run this script with ``--reference`` in a process of its own."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "cases.json").write_text(json.dumps(plain(payload)))
        subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--reference", str(root)],
            check=True,
        )
        result: dict[str, Any] = json.loads((root / "reference.json").read_text())
    return result


def reference_main(directory: Path) -> None:
    """mpmath's roots of the exponential shapes and scipy's p-values of the dose
    series: imports mpmath and scipy, not CytePool."""
    import mpmath
    import scipy
    from scipy import stats

    mpmath.mp.dps = 40
    cases = json.loads((directory / "cases.json").read_text())
    roots = []
    for half, top in cases["exponential"]:
        h, d = mpmath.mpf(half), mpmath.mpf(top)

        def excess(delta: Any, h: Any = h, d: Any = d) -> Any:
            return mpmath.expm1(h / delta) / mpmath.expm1(d / delta) - mpmath.mpf(1) / 2

        # Bisection on log delta: the ratio rises from 0 to h / D as delta grows.
        low, high = mpmath.log(d) - 40, mpmath.log(d) + 40
        for _ in range(400):
            middle = (low + high) / 2
            if excess(mpmath.exp(middle)) > 0:
                high = middle
            else:
                low = middle
        roots.append(float(mpmath.exp((low + high) / 2)))
    p_values = []
    for case in cases["scipy"]:
        correlation = np.asarray(case["correlation"], dtype=np.float64)
        upper = np.full(correlation.shape[0], case["threshold"])
        values = [
            1.0
            - stats.multivariate_t.cdf(
                upper,
                shape=correlation,
                df=case["df"],
                lower_limit=-upper,
                random_state=state,
            )
            for state in range(SCIPY_CALLS)
        ]
        p_values.append(
            [
                float(np.mean(values)),
                float(np.std(values, ddof=1) / math.sqrt(SCIPY_CALLS)),
            ]
        )
    out = {
        "exponential": roots,
        "scipy": p_values,
        "versions": {"mpmath": mpmath.__version__, "scipy": scipy.__version__},
    }
    (directory / "reference.json").write_text(json.dumps(out))


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------


def ours_spread(case: dict[str, Any]) -> CandidateShapes:
    counts = case["counts"]
    return CandidateShapes.spread(
        axis_of(case["axis"]),
        {"dose": case["doses"]},
        n_emax=counts["emax"],
        n_sigmoid_emax=counts["sigmoid_emax"],
        n_exponential=counts["exponential"],
        n_beta=counts["beta"],
    )


def theta_contrasts(
    shapes: CandidateShapes, doses: list[float], covariance: Any
) -> np.ndarray:
    """The optimal contrasts of the dose groups with the vehicle's row dropped, scaled to
    unit length: DoseFinding's placebo-adjusted contrasts when the covariance of the
    differences from the vehicle is D S D'."""
    means = shapes.means(doses)
    matrix = optimal_contrasts(
        dict(zip(shapes.names, means, strict=True)), np.asarray(covariance)
    ).matrix[1:]
    scaled: np.ndarray = matrix / np.linalg.norm(matrix, axis=0)
    return scaled


@dataclass
class Job:
    """One analysis of one series by CytePool and R."""

    series: Series
    analysis: str

    @property
    def name(self) -> str:
        return f"{self.series.name} {self.analysis}"

    def run(self, seed: int) -> Any:
        options: dict[str, Any] = {}
        if self.analysis == "voom_limma":
            options["contrast_standard_errors"] = "limma"
        if self.analysis == "dream":
            # voomWithDreamWeights' span, which dream_voom takes from voom by default.
            options["span"] = 0.5
        analysis = {
            "voom": limma_voom,
            "voom_limma": limma_voom,
            "trend": limma_trend,
            "dream": dream_voom,
        }[self.analysis]
        return dose_response(
            self.series.cube,
            self.series.design,
            dose=self.series.axis,
            tests=("spline", "dose_groups", *TESTS),
            analysis=analysis,
            rng=seed,
            **options,
        )


def job_seed(name: str) -> int:
    return SEED + zlib.crc32(name.encode())


def cell_type_inputs(
    job: Job, result: Any, fit: dict[str, Any], group: str
) -> dict[str, Any]:
    """The DoseFinding program's input for one cell type: R's own doses, samples and
    estimates, CytePool's shapes, and two screens for the stage-wise calls: the spline
    test and the per-dose tests of CytePool, and R's own dose-group test and per-dose
    tests."""
    genes = len(fit["genes"])
    theta = numbers(fit["theta"]).reshape(genes, -1)
    each = result.table(group, "each_dose")
    k = theta.shape[1]
    spline = result.table(group, "spline")
    entry = cell_type_entry(result, group)
    df = numbers(fit["df_total"]).reshape(genes, -1).min(axis=1)
    singular = entry["contrast_tests"]["mcp"]["dimensions"] < len(
        entry["contrast_tests"]["mcp"]["contrasts"]
    )
    bracket = (
        np.linspace(0, genes - 1, min(BRACKET_GENES, genes)).astype(int) + 1
        if singular and job.analysis == "voom"
        else []
    )
    return {
        "label": f"{job.name}, {group}",
        "doses": fit["doses"],
        "n": fit["n"],
        "shapes": result.candidates.to_dict(),
        "theta": theta,
        "V": fit["V"],
        "s2_post": fit["s2_post"],
        "df": df,
        "alpha": ALPHA,
        "bracket": bracket,
        "screens": {
            "spline": {
                "screen": np.asarray(spline["p_value"]),
                "p": np.asarray(each["p_value"]).reshape(k, genes).T,
            },
            **(
                {
                    "dose_groups": {
                        "screen": fit["f_p"],
                        "p": numbers(fit["each_p"]).reshape(genes, k),
                    }
                }
                if "f_p" in fit
                else {}
            ),
        },
    }


def cell_type_entry(result: Any, group: str) -> dict[str, Any]:
    for entry in result.provenance["groups"]:
        if entry["group_id"] == group:
            found: dict[str, Any] = entry
            return found
    raise KeyError(group)


# ---------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------


class Rows:
    """The printed checks: check, reference, value, tolerance and outcome."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, str, str]] = []
        self.failures = 0

    def bound(self, check: str, reference: str, value: float, tolerance: float) -> None:
        passed = bool(value <= tolerance)
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{value:.1e}",
                f"{tolerance:g}",
                "pass" if passed else "FAIL",
            )
        )

    def identical(self, check: str, reference: str, same: int, total: int) -> None:
        passed = same == total
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{same:,} of {total:,} identical",
                "identical",
                "pass" if passed else "FAIL",
            )
        )

    def calibrated(self, check: str, reference: str, z: np.ndarray) -> None:
        largest = float(np.max(np.abs(z))) if z.size else 0.0
        passed = bool(largest <= CALIBRATED)
        self.failures += not passed
        self.rows.append(
            (
                check,
                reference,
                f"{largest:.2f} standard errors",
                f"{CALIBRATED:g} standard errors",
                "pass" if passed else "FAIL",
            )
        )

    def information(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "information"))

    def deliberate(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "deliberate difference"))

    def print(self) -> None:
        print("| check | reference | value | tolerance | result |")
        print("| --- | --- | --- | --- | --- |")
        for row in self.rows:
            print("| " + " | ".join(row) + " |")


def relative(ours: Any, reference: Any) -> float:
    """The largest difference relative to the reference, absolute where it is 0;
    infinite when NaN patterns differ."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return math.inf
    keep = ~np.isnan(b)
    if not keep.any():
        return 0.0
    a, b = a[keep], b[keep]
    scale = np.where(b == 0.0, 1.0, np.abs(b))
    return float(np.max(np.abs(a - b) / scale))


def statistic(ours: Any, reference: Any) -> float:
    """The largest difference relative to max(|reference|, 1)."""
    a = np.asarray(ours, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or np.isnan(a).any() or np.isnan(b).any():
        return math.inf
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1.0))) if a.size else 0.0


def compare_spread(rows: Rows, cases: list, ours: list, r: list, roots: list) -> None:
    emax, sigmoid, slope, guess, exact, beta, scale = [], [], [], [], [], [], []
    position = 0
    for case, shapes, reference in zip(cases, ours, r, strict=True):
        emax.append(relative(shapes.emax, numbers(reference["emax"]).ravel()))
        sigmoid.append(
            relative(
                [e for e, _ in shapes.sigmoid_emax],
                numbers(reference["sigmoid_ed50"]).ravel(),
            )
        )
        slope.append(all(h == 3.0 for _, h in shapes.sigmoid_emax))
        count = case["counts"]["exponential"]
        guess.append(
            relative(shapes.exponential, numbers(reference["exponential"]).ravel())
        )
        exact.append(relative(shapes.exponential, roots[position : position + count]))
        position += count
        if shapes.beta:
            beta.append(
                relative(
                    np.array(shapes.beta).ravel(),
                    np.array([numbers(pair) for pair in reference["beta"]]).ravel(),
                )
            )
            scale.append(relative(shapes.beta_scale, reference["beta_scale"]))
    where = f"{len(LADDERS)} ladders, 3 placements, counts 0 to 3 ({len(cases)} cases)"
    rule = "the rule's positions computed in R from the ladder"
    rows.bound(f"CandidateShapes.spread: Emax ED50s, {where}", rule, max(emax), EXACT)
    rows.bound(
        f"CandidateShapes.spread: sigmoid Emax ED50s, {where}",
        rule,
        max(sigmoid),
        EXACT,
    )
    rows.identical(
        "CandidateShapes.spread: sigmoid Emax Hill slopes of 3",
        "the rule",
        sum(slope),
        len(slope),
    )
    rows.bound(
        f"CandidateShapes.spread: exponential delta, {where}",
        "mpmath 1.3.0, 40 digits, the root of expm1(h / delta) / expm1(D / delta) = 1/2",
        max(exact),
        EXACT,
    )
    rows.bound(
        f"CandidateShapes.spread: exponential delta, {where}",
        "DoseFinding guesst (a root search)",
        max(guess),
        GUESST,
    )
    rows.bound(
        f"CandidateShapes.spread: beta delta1 and delta2, {where}",
        "DoseFinding guesst, betaMod (closed form)",
        max(beta),
        EXACT,
    )
    rows.bound(
        "CandidateShapes.spread: the beta scale, 1.2 times the top dose",
        "the rule in R",
        max(scale),
        EXACT,
    )


def compare_means(rows: Rows, cases: list, r: list, overflow: Any) -> None:
    for case, reference in zip(cases, r, strict=True):
        shapes = CandidateShapes(**case["shapes"])
        ours = shapes.means(case["doses"])
        theirs = numbers(reference["means"]).reshape(ours.shape)
        rows.bound(
            f"CandidateShapes.means: {case['label']}, {len(shapes.names)} shapes at "
            f"{len(case['doses'])} doses",
            "DoseFinding getResp(Mods(...))",
            relative(ours, theirs),
            EXACT,
        )
    doses = [0.0, 0.05, 0.2, 2.0, 3.0, 50.0]
    ours = CandidateShapes(exponential=[0.05]).means(doses)[0]
    theirs = numbers(overflow).ravel()
    rows.deliberate(
        "CandidateShapes.means: an exponential shape of delta D / 1000",
        "DoseFinding getResp(Mods(...))",
        f"DoseFinding {theirs.tolist()}, where exp(D / delta) overflows; CytePool "
        f"{[float(f'{v:.3g}') for v in ours]}",
    )


def compare_contrasts(rows: Rows, cases: list, r: list) -> None:
    worst = {"w": 0.0, "S": 0.0, "placebo_adjusted": 0.0}
    for case, reference in zip(cases, r, strict=True):
        shapes = CandidateShapes(**case["shapes"])
        means = dict(zip(shapes.names, shapes.means(case["doses"]), strict=True))
        k = len(case["doses"])
        ours = {
            "w": optimal_contrasts(
                means, np.diag(1.0 / np.asarray(case["n"], float))
            ).matrix,
            "S": optimal_contrasts(means, np.asarray(case["S"])).matrix,
            "placebo_adjusted": theta_contrasts(shapes, case["doses"], case["S"]),
        }
        for key, matrix in ours.items():
            theirs = numbers(reference[key]).reshape(matrix.shape[0], -1)
            worst[key] = max(worst[key], statistic(matrix, theirs))
        del k
    where = f"{len(cases)} designs of 3 to 9 groups"
    rows.bound(
        f"optimal_contrasts, independent groups of sizes n, {where}",
        "DoseFinding optContr(w = n)",
        worst["w"],
        EXACT,
    )
    rows.bound(
        f"optimal_contrasts, a general covariance S, {where}",
        "DoseFinding optContr(S = S)",
        worst["S"],
        EXACT,
    )
    rows.bound(
        f"optimal_contrasts with the vehicle's row dropped, {where}",
        "DoseFinding optContr(S = D S D', placAdj = TRUE)",
        worst["placebo_adjusted"],
        EXACT,
    )


def ours_mct(case: dict[str, Any]) -> Any:
    shapes = CandidateShapes(**case["shapes"])
    contrasts = theta_contrasts(shapes, case["doses"], case["S"])
    test = multiple_contrast_test(
        np.asarray(case["theta"]),
        np.asarray(case["V"]),
        contrasts,
        df=case["df"],
        relative_error=TIGHT,
        rng=case["seed"],
    )
    return contrasts, test


def compare_mct(rows: Rows, cases: list, r: list, refusals: dict) -> None:
    contrast_gap, t_gap, absolute_gap = 0.0, 0.0, 0.0
    z: dict[bool, list[float]] = {True: [], False: []}
    below = failed = 0
    for case, reference in zip(cases, r, strict=True):
        contrasts, test = ours_mct(case)
        theirs = numbers(reference["contrasts"]).reshape(contrasts.shape)
        contrast_gap = max(contrast_gap, statistic(contrasts, theirs))
        t = test.t[0]
        absolute_gap = max(
            absolute_gap, statistic(np.abs(t), numbers(reference["t"]).ravel())
        )
        t_gap = max(t_gap, statistic(t, numbers(reference["glht_t"]).ravel()))
        singular = np.linalg.matrix_rank(contrasts) < contrasts.shape[1]
        if reference.get("p") is None:
            failed += 1
            continue
        p = numbers(reference["p"]).ravel()
        error = float(reference["error"]) / 3.0
        for ours, ours_error, theirs_p in zip(
            test.p_value[0], test.p_value_error[0], p, strict=True
        ):
            if theirs_p < SMALLEST_P:
                below += 1
                continue
            z[bool(singular)].append((ours - theirs_p) / math.hypot(ours_error, error))
    n = len(cases)
    rows.bound(
        f"optimal contrasts of the MCP step on the doses' differences, {n} features",
        "DoseFinding MCTtest(type = 'general', placAdj = TRUE) contMat",
        contrast_gap,
        EXACT,
    )
    rows.bound(
        f"t of the MCP step, {n} features, df 5, 12 and 40",
        "multcomp summary(glht(parm(theta, V, df))) on MCTtest's contrasts",
        t_gap,
        EXACT,
    )
    rows.deliberate(
        "MCTtest's two-sided t-statistics",
        "DoseFinding MCTtest",
        f"|t|, equal to CytePool's |t| within {absolute_gap:.1e}; CytePool and glht keep "
        "the sign",
    )
    for singular, label in (
        (False, "full rank"),
        (True, "singular, more shapes than treated doses"),
    ):
        rows.calibrated(
            f"p-values of the MCP step, {label}: {len(z[singular])} from {SMALLEST_P:g} to 1",
            "multcomp summary(glht()), mvtnorm 1.4-2, error over 3",
            np.asarray(z[singular]),
        )
    if failed:
        rows.bound(
            f"multcomp's summary(glht()) stopped on {failed} features",
            "multcomp",
            math.inf,
            0.0,
        )
    rows.information(
        "p-values of the MCP step below 1e-3",
        "multcomp summary(glht())",
        f"{below} not compared: mvtnorm's default absolute error of 1e-3 cannot resolve them",
    )
    for name, message in refusals.items():
        rows.deliberate(
            f"a moderated df of 12.7 in {'MCTtest' if name == 'mcttest' else 'summary(glht())'}",
            "DoseFinding" if name == "mcttest" else "multcomp",
            f"refused: {message}; CytePool takes any positive df",
        )


def compare_stager(rows: Rows, cases: list, r: list) -> None:
    groups: dict[tuple[str, float], list[float]] = {}
    calls: dict[tuple[str, float], list[str]] = {}
    closure_same = closure_total = 0
    for case, reference in zip(cases, r, strict=True):
        screen = numbers(case["screen"]).ravel()
        p = numbers(case["p"]).reshape(screen.size, -1)
        ours = stage_wise_adjusted_p_values(
            screen, p, alpha=case["alpha"], method=case["method"]
        )
        if case["method"] == "holm":
            rejected = np.asarray(reference["closure"], dtype=bool).reshape(p.shape)
            mine = np.nan_to_num(ours, nan=2.0) <= case["alpha"]
            closure_same += int(np.count_nonzero(rejected == mine))
            closure_total += rejected.size
        adjusted = reference["adjusted"]
        if "error" in adjusted:
            if case["m"] == 1 and case["method"] == "holm":
                # stageR's "holm" also fails on a called feature of 3 hypotheses or
                # fewer with none or one left.
                one = stage_wise_adjusted_p_values(
                    screen, p, alpha=case["alpha"], method="none"
                )
                rows.deliberate(
                    f"stage_wise_adjusted_p_values, one hypothesis per feature, alpha {case['alpha']:g}",
                    "stageR stageWiseAdjustment(method = 'holm')",
                    f"stageR fails ({adjusted['error']}); CytePool multiplies by 1, as "
                    f"stageR's 'none' does (difference {relative(ours, one):.1e})",
                )
                continue
            rows.bound(
                f"stage_wise_adjusted_p_values, {case['label']}",
                "stageR",
                math.inf,
                EXACT,
            )
            continue
        theirs = numbers(adjusted["adjusted"]).reshape(p.shape)
        if "one_left" in case:
            row = case["one_left"]
            left = int(np.flatnonzero(~np.isnan(p[row]))[0])
            others = np.arange(p.shape[0]) != row
            rows.deliberate(
                "stage_wise_adjusted_p_values, a called feature with one hypothesis left "
                "of four",
                "stageR stageWiseAdjustment(method = 'holm', allowNA = TRUE)",
                f"stageR gives it {theirs[row, left]:.3g}, Shaffer's multiplier of 0, "
                f"calling it on the screen alone; CytePool {ours[row, left]:.3g}, "
                "Bonferroni at the screen's level, which holds whatever the screen "
                f"tested; the other features differ by {relative(ours[others], theirs[others]):.1e}",
            )
            continue
        key = (case["method"], case["alpha"])
        groups.setdefault(key, []).append(relative(ours, theirs))
        q = np.asarray([v for v in screen if not math.isnan(v)], dtype=np.float64)
        called = int(np.count_nonzero(_bh(q) <= case["alpha"]))
        calls.setdefault(key, []).append(f"{case['m']}: R = {called}")
    for (method, alpha), gaps in groups.items():
        rows.bound(
            f"stage_wise_adjusted_p_values, '{method}', alpha {alpha:g}: hypotheses per "
            f"feature and features called ({'; '.join(calls[(method, alpha)])})",
            "stageR 1.28.0 stageWiseAdjustment, allowNA where values are missing",
            max(gaps),
            EXACT,
        )
    rows.identical(
        "per-hypothesis calls of 'holm' at alpha, every case",
        "the closure of Bonferroni tests, every intersection enumerated (R)",
        closure_same,
        closure_total,
    )


def _bh(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg, coded here for the counts printed."""
    order = np.argsort(p)
    ranked = p[order] * p.size / np.arange(1, p.size + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(p)
    out[order] = np.minimum(adjusted, 1.0)
    return out


@dataclass
class CellType:
    """One cell type of one job: CytePool's result and R's fit and references."""

    job: Job
    group: str
    result: Any
    fit: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)  # the DoseFinding program's
    bracket: list[int] = field(default_factory=list)  # genes whose p-values R brackets
    reference: dict[str, Any] | None = None

    @property
    def genes(self) -> int:
        return len(self.fit["genes"])

    @property
    def k(self) -> int:
        return len(self.fit["doses"]) - 1

    def ours_weights(self, test: str) -> tuple[list[str], np.ndarray]:
        contrasts = cell_type_entry(self.result, self.group)["contrast_tests"][test][
            "contrasts"
        ]
        names = list(contrasts)
        return names, np.array([contrasts[name] for name in names], dtype=np.float64)

    def r_weights(self, test: str) -> np.ndarray:
        assert self.reference is not None
        return numbers(self.reference["contrasts"][test]).reshape(-1, self.k)

    def r_t(self, test: str) -> np.ndarray:
        assert self.reference is not None
        return numbers(self.reference["t"][test]).reshape(self.genes, -1)

    def correlation(self, test: str, gene: int) -> np.ndarray:
        """The correlation of R's contrasts of a gene, from R's own covariance."""
        weights = self.r_weights(test)
        v = numbers(self.fit["V"]).reshape(self.genes, self.k, self.k)[gene]
        covariance = weights @ v @ weights.T
        scale = np.sqrt(np.diag(covariance))
        correlation: np.ndarray = covariance / np.outer(scale, scale)
        return correlation

    def df(self) -> np.ndarray:
        df: np.ndarray = (
            numbers(self.fit["df_total"]).reshape(self.genes, -1).min(axis=1)
        )
        return df


def compare_series(
    rows: Rows, cells: list[CellType], scipy_cases: list[dict], scipy: list
) -> None:
    """dose_response's three tests and lowest effective doses against R."""
    same = total = 0
    for cell in cells:
        ids = cell.job.series.cube.features["feature_id"].tolist()
        expected = [ids[g] for g in cell.fit["genes"]]
        tested = cell.result.table(cell.group, "spline")["feature_id"].tolist()
        entry = cell_type_entry(cell.result, cell.group)
        checks = [
            tested == expected,
            np.allclose(entry["doses"], cell.fit["doses"], rtol=0, atol=0),
            [int(n) for n in entry["samples"]] == [int(n) for n in cell.fit["n"]],
        ]
        same += sum(checks)
        total += len(checks)
    rows.identical(
        f"dose_response: genes tested, doses compared and samples per dose, "
        f"{len(cells)} cell types",
        "R from the counts and cells",
        same,
        total,
    )
    for analysis, label, tolerance in (
        ("voom", "limma-voom, exact standard errors", EXACT),
        ("voom_limma", "limma-voom, contrast_standard_errors='limma'", EXACT),
        ("trend", "limma-trend", EXACT),
        ("dream", "dream_voom", CLOSE),
    ):
        group = [cell for cell in cells if cell.job.analysis == analysis]
        if not group:
            continue
        reference = (
            "variancePartition 1.36.2 dream, then multcomp glht(parm()) on contrMat and optContr"
            if analysis == "dream"
            else "limma 3.62.1 from the counts, then multcomp glht(parm()) on contrMat and optContr"
        )
        for test in TESTS:
            weights = t_gap = df_gap = 0.0
            chosen = genes = 0
            for cell in group:
                names, ours = cell.ours_weights(test)
                weights = max(weights, statistic(ours, cell.r_weights(test)))
                table = cell.result.table(cell.group, test)
                theirs = cell.r_t(test)
                pick = np.argmax(np.abs(theirs), axis=1)
                t_gap = max(
                    t_gap,
                    statistic(
                        np.asarray(table["t"]), theirs[np.arange(cell.genes), pick]
                    ),
                )
                chosen += sum(
                    names.index(name) == j
                    for name, j in zip(table["contrast"].tolist(), pick, strict=True)
                )
                genes += cell.genes
                df_gap = max(df_gap, relative(np.asarray(table["df"]), cell.df()))
            cases = f"{len(group)} cell types, {genes:,} genes"
            rows.bound(
                f"'{test}', {label}: contrasts, {cases}", reference, weights, tolerance
            )
            rows.bound(
                f"'{test}', {label}: each gene's largest |t|",
                reference,
                t_gap,
                tolerance,
            )
            rows.identical(
                f"'{test}', {label}: the contrast of the largest |t|",
                reference,
                chosen,
                genes,
            )
            rows.bound(
                f"'{test}', {label}: df of the multivariate t",
                "df.total"
                if analysis != "dream"
                else "the smallest df.total of the doses",
                df_gap,
                tolerance if analysis != "dream" else 1e-5,
            )
    by_test: dict[str, list[float]] = {test: [] for test in TESTS}
    for case, (value, error) in zip(scipy_cases, scipy, strict=True):
        by_test[case["test"]].append(
            (case["ours"] - value) / math.hypot(case["ours_error"], error)
        )
    for test, z in by_test.items():
        rows.calibrated(
            f"'{test}': p-values at non-integer df, full rank, {len(z)} genes from "
            f"{SMALLEST_P:g} to 1",
            f"scipy multivariate_t.cdf, {SCIPY_CALLS} calls, on R's t, correlation and df",
            np.asarray(z),
        )
    inside = considered = 0
    for cell in cells:
        if cell.reference is None or "bracket" not in cell.reference:
            continue
        table = cell.result.table(cell.group, "mcp")
        rank = cell_type_entry(cell.result, cell.group)["contrast_tests"]["mcp"][
            "dimensions"
        ]
        for g, pair in zip(cell.bracket, cell.reference["bracket"], strict=True):
            (low_p, low_error), (high_p, high_error) = numbers(pair).T.tolist()
            ours = float(table["p_value"][g - 1])
            margin = CALIBRATED * math.hypot(
                float(table["p_value_error"][g - 1]), max(low_error, high_error) / 3.0
            )
            inside += min(high_p, low_p) - margin <= ours <= max(high_p, low_p) + margin
            considered += 1
        del rank
    rows.information(
        "'mcp' p-values of singular cell types (5 shapes on 2 to 4 treated doses)",
        "multcomp summary(glht()) at the integer df below and above the gene's",
        f"{inside} of {considered} between the two, within 5.5 standard errors",
    )
    compare_lowest_doses(rows, cells)


def lowest_from(
    adjusted: np.ndarray, doses: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The lowest dose called (at most ALPHA) per gene, and the number called."""
    called = np.nan_to_num(adjusted, nan=2.0) <= ALPHA
    lowest = np.where(called.any(axis=1), doses[np.argmax(called, axis=1)], np.nan)
    return lowest, called.sum(axis=1)


def compare_lowest_doses(rows: Rows, cells: list[CellType]) -> None:
    for screen, label in (
        ("spline", "CytePool's spline test and per-dose tests"),
        ("dose_groups", "R's own dose-group test and per-dose tests, from the counts"),
    ):
        same = total = 0
        gap = 0.0
        closure_same = closure_total = 0
        for cell in cells:
            if cell.reference is None or screen not in cell.reference.get("stager", {}):
                continue
            doses = np.asarray(cell.fit["doses"][1:], dtype=np.float64)
            ours = cell.result.lowest_effective_doses(
                cell.group, screen=screen, alpha=ALPHA
            )
            stager = cell.reference["stager"][screen]
            adjusted = numbers(stager["adjusted"]).reshape(cell.genes, cell.k)
            lowest, n_called = lowest_from(adjusted, doses)
            screen_p = numbers(cell.inputs["screens"][screen]["screen"]).ravel()
            responds = _bh(screen_p) <= ALPHA
            checks = [
                np.array_equal(
                    np.asarray(ours["lowest_effective_dose"]), lowest, equal_nan=True
                ),
                np.array_equal(np.asarray(ours["n_doses_called"]), n_called),
                np.array_equal(np.asarray(ours["responds"]), responds),
            ]
            same += sum(checks)
            total += len(checks)
            at = np.where(
                np.isnan(lowest),
                -1,
                np.argmax(np.nan_to_num(adjusted, nan=2.0) <= ALPHA, axis=1),
            )
            theirs = np.where(
                at >= 0, adjusted[np.arange(cell.genes), np.maximum(at, 0)], np.nan
            )
            gap = max(
                gap, relative(np.asarray(ours["stage_wise_adjusted_p_value"]), theirs)
            )
            rejected = np.asarray(
                cell.reference["closure"][screen], dtype=bool
            ).reshape(cell.genes, cell.k)
            closure_lowest = np.where(
                rejected.any(axis=1), doses[np.argmax(rejected, axis=1)], np.nan
            )
            closure_same += int(
                np.count_nonzero(
                    (closure_lowest == np.asarray(ours["lowest_effective_dose"]))
                    | (
                        np.isnan(closure_lowest)
                        & np.isnan(np.asarray(ours["lowest_effective_dose"]))
                    )
                )
            )
            closure_total += cell.genes
        if not total:
            continue
        rows.identical(
            f"lowest_effective_doses, screen '{screen}': lowest effective doses, doses called "
            "and calls of the screen, every cell type",
            f"stageR stageWiseAdjustment('holm') on {label}",
            same,
            total,
        )
        rows.bound(
            f"lowest_effective_doses, screen '{screen}': stage-wise adjusted p-value at the "
            "lowest effective dose",
            f"stageR on {label}",
            gap,
            EXACT,
        )
        rows.identical(
            f"lowest_effective_doses, screen '{screen}': each gene's lowest effective dose",
            "the closure of Bonferroni tests, every intersection enumerated (R)",
            closure_same,
            closure_total,
        )


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


def write_fixtures(
    inputs: dict[str, Any],
    r: dict[str, Any],
    roots: list,
    cells: list[CellType],
    scipy_cases: list[dict[str, Any]],
    scipy: list,
) -> None:
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    dosefinding = {
        "versions": r["versions"],
        "spread": [
            {**case, "reference": reference}
            for case, reference in zip(inputs["spread"], r["spread"], strict=True)
        ],
        "exponential_roots": roots,
        "means": [
            {**case, "reference": reference["means"]}
            for case, reference in zip(inputs["means"], r["means"], strict=True)
        ],
        "contrasts": [
            {**case, "reference": reference}
            for case, reference in zip(inputs["contrasts"], r["contrasts"], strict=True)
        ],
        "mct": [
            {**case, "reference": reference}
            for case, reference in zip(inputs["mct"], r["mct"], strict=True)
        ],
    }
    # The tests enumerate the closure themselves, so the fixture keeps stageR's values.
    stager = {
        "versions": r["versions"],
        "cases": [
            {**case, "reference": {"adjusted": reference["adjusted"]}}
            for case, reference in zip(inputs["stager"], r["stager"], strict=True)
        ],
    }
    fixture = next(item for item in inputs["series"] if item.name == "fixture")
    tests = {
        "series": series_record(fixture),
        "cell_types": [
            {
                "analysis": cell.job.analysis,
                "group": cell.group,
                "genes": cell.fit["genes"],
                "doses": cell.fit["doses"],
                "n": cell.fit["n"],
                "df": cell.df(),
                "contrasts": {test: cell.r_weights(test) for test in TESTS},
                "t": {
                    test: cell.r_t(test)[
                        np.arange(cell.genes), np.argmax(np.abs(cell.r_t(test)), axis=1)
                    ]
                    for test in TESTS
                },
                "chosen": {
                    test: np.argmax(np.abs(cell.r_t(test)), axis=1) for test in TESTS
                },
                "f_p": cell.fit["f_p"],
                "stage_wise": cell.reference["stager"]["dose_groups"]["adjusted"],
            }
            for cell in cells
            if cell.job.series.name == "fixture"
        ],
        "scipy": [
            {
                "group": case["group"],
                "test": case["test"],
                "feature_id": case["feature_id"],
                "p_value": value,
                "error": error,
            }
            for case, (value, error) in zip(scipy_cases, scipy, strict=True)
            if case["job"] == "fixture voom"
        ],
    }
    for name, payload in (
        ("dosefinding.json", dosefinding),
        ("stager.json", stager),
        ("r_tests.json", tests),
    ):
        path = FIXTURE_DIRECTORY / name
        path.write_text(json.dumps(plain(payload), separators=(",", ":")) + "\n")
        print(f"wrote {path} ({path.stat().st_size / 1000:.0f} KB)")


# ---------------------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------------------


def benchmark() -> None:
    """dose_response with each multiple contrast test against the default tests, with
    the ordinary least squares fit of 20,000 genes x 200 samples as the baseline."""
    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 20_000))
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])

    def timed(run: Any) -> float:
        run()
        times = []
        for _ in range(3):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        return float(np.median(times))

    baseline = timed(lambda: fit_linear_model(y, x))
    print(
        f"Baseline: ordinary least squares, 20,000 genes x 200 samples, {baseline:.3f} s"
    )
    print(
        "| design | tests | seconds | ms per gene, beyond the default tests | baselines |"
    )
    print("| --- | --- | --- | --- | --- |")
    for doses in (
        [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
        [0.0, 0.1, 1.0, 10.0],
    ):
        cube, _ = simulate_dose_series(
            doses, replicates=3, cell_types=1, cells=300.0, n_genes=20_000, rng=20261216
        )
        design = Design(cube.samples, "~ 1")
        genes = cube.n_features
        defaults = dose_response(cube, design, dose=LOG_AXIS).tests
        default = timed(lambda: dose_response(cube, design, dose=LOG_AXIS))
        label = f"the vehicle and {len(doses) - 1} doses x 3"
        print(f"| {label} | default | {default:.2f} | | {default / baseline:.0f} |")
        for test in TESTS:
            seconds = timed(
                lambda test=test: dose_response(
                    cube, design, dose=LOG_AXIS, tests=(*defaults, test), rng=1
                )
            )
            extra = (seconds - default) / genes * 1000.0
            print(
                f"| {label} | default and '{test}' | {seconds:.2f} | {extra:.3f} | "
                f"{(seconds - default) / baseline:.0f} |"
            )


# ---------------------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time dose_response")
    parser.add_argument("--reference", type=Path, help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.reference is not None:
        reference_main(arguments.reference)
        return 0
    if arguments.benchmark:
        benchmark()
        return 0
    started = time.perf_counter()
    rng = np.random.default_rng(SEED)
    found = series()
    inputs: dict[str, Any] = {
        "spread": spread_cases(),
        "means": means_cases(),
        "contrasts": contrast_cases(rng),
        "mct": mct_cases(rng),
        "stager": stager_cases(rng),
        "series": found,
    }
    records = {item.name: series_record(item) for item in found}
    jobs = [Job(item, analysis) for item in found for analysis in item.analyses]
    limma = run_r(
        LIMMA,
        {
            "series": records,
            "jobs": {
                job.name: {"series": job.series.name, "analysis": job.analysis}
                for job in jobs
                if job.analysis != "dream"
            },
            "min_cells": MIN_CELLS,
        },
        "RSCRIPT",
    )
    dream = run_r(
        DREAM,
        {
            "series": records,
            "jobs": {
                job.name: {"series": job.series.name}
                for job in jobs
                if job.analysis == "dream"
            },
            "min_cells": MIN_CELLS,
        },
        "DREAM_RSCRIPT",
    )
    print(f"R fits finished after {time.perf_counter() - started:.0f} s", flush=True)
    fits = {**limma["jobs"], **dream["jobs"]}
    cells: list[CellType] = []
    for job in jobs:
        result = job.run(job_seed(job.name))
        for group, fit in fits[job.name]["cell_types"].items():
            cell_inputs = cell_type_inputs(job, result, fit, group)
            bracket = [int(g) for g in cell_inputs["bracket"]]
            cells.append(CellType(job, group, result, fit, cell_inputs, bracket))
    print(f"CytePool finished after {time.perf_counter() - started:.0f} s", flush=True)
    dose = run_r(
        DOSE,
        {
            "spread": inputs["spread"],
            "means": inputs["means"],
            "overflow": {
                "doses": [0.0, 0.05, 0.2, 2.0, 3.0, 50.0],
                "shapes": CandidateShapes(exponential=[0.05]).to_dict(),
            },
            "contrasts": inputs["contrasts"],
            "mct": inputs["mct"],
            "stager": inputs["stager"],
            "cubes": [cell.inputs for cell in cells],
        },
        "DOSE_RSCRIPT",
    )
    for cell, reference in zip(cells, dose["cubes"], strict=True):
        cell.reference = reference
    print(
        f"DoseFinding, multcomp and stageR finished after {time.perf_counter() - started:.0f} s",
        flush=True,
    )
    halves = [
        (half, max(case["doses"]))
        for case, reference in zip(inputs["spread"], dose["spread"], strict=True)
        for half in numbers(reference["exponential_half"]).ravel().tolist()
    ]
    scipy_cases = scipy_inputs(cells)
    references = run_reference({"exponential": halves, "scipy": scipy_cases})
    print(
        f"References finished after {time.perf_counter() - started:.0f} s", flush=True
    )
    # One R version: the three R environments share it.
    versions = {
        **limma["versions"],
        **dream["versions"],
        **dose["versions"],
        **references["versions"],
    }
    print("Reference: " + ", ".join(f"{k} {v}" for k, v in versions.items()))
    rows = Rows()
    spread = [ours_spread(case) for case in inputs["spread"]]
    compare_spread(
        rows, inputs["spread"], spread, dose["spread"], references["exponential"]
    )
    compare_means(rows, inputs["means"], dose["means"], dose["overflow"])
    compare_contrasts(rows, inputs["contrasts"], dose["contrasts"])
    compare_mct(rows, inputs["mct"], dose["mct"], dose["refusals"])
    compare_stager(rows, inputs["stager"], dose["stager"])
    compare_series(rows, cells, scipy_cases, references["scipy"])
    rows.print()
    counts = {}
    for row in rows.rows:
        counts[row[4]] = counts.get(row[4], 0) + 1
    print(
        f"{len(rows.rows)} checks: "
        + ", ".join(f"{n} {kind}" for kind, n in counts.items())
        + f"; {rows.failures} failed; {time.perf_counter() - started:.0f} s"
    )
    if arguments.write_fixtures:
        write_fixtures(
            inputs,
            dose,
            references["exponential"],
            cells,
            scipy_cases,
            references["scipy"],
        )
    return 1 if rows.failures else 0


def scipy_inputs(cells: list[CellType]) -> list[dict[str, Any]]:
    """The genes whose p-values scipy computes: in the cell types of limma-voom's exact
    default and dream, full rank, CytePool's p-value from 1e-3 to 1, up to
    ``SCIPY_GENES`` per cell type and test spread over the genes."""
    cases = []
    for cell in cells:
        if cell.job.analysis not in ("voom", "dream"):
            continue
        df = cell.df()
        for test in TESTS:
            weights = cell.r_weights(test)
            if np.linalg.matrix_rank(weights) < weights.shape[0]:
                continue
            table = cell.result.table(cell.group, test)
            p = np.asarray(table["p_value"])
            usable = np.flatnonzero((p >= SMALLEST_P) & (p < 1.0))
            chosen = (
                usable[
                    np.linspace(
                        0, usable.size - 1, min(SCIPY_GENES, usable.size)
                    ).astype(int)
                ]
                if usable.size
                else []
            )
            theirs = cell.r_t(test)
            ids = table["feature_id"].tolist()
            for g in chosen:
                cases.append(
                    {
                        "job": cell.job.name,
                        "group": cell.group,
                        "feature_id": ids[g],
                        "test": test,
                        "correlation": cell.correlation(test, int(g)),
                        "threshold": float(np.max(np.abs(theirs[g]))),
                        "df": float(df[g]),
                        "ours": float(p[g]),
                        "ours_error": float(table["p_value_error"][g]),
                    }
                )
    return cases


if __name__ == "__main__":
    sys.exit(main())
