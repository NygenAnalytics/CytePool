"""Check dose_response against limma, edgeR, rms and variancePartition, and compare its
calls with DRomics, dreamlet and muscat.

Dose series simulated by ``simulate_dose_series`` from fixed seeds (20261208 to
20261215): ``fixture``, the vehicle and 5 doses from 0.1 to 10 ug/kg x 3, 300 genes,
a cell type of every sample and one that loses the top dose and one sample of 3 ug/kg
(fewer than 10 cells); ``ladder``, the vehicle and 8 half-log doses from 0.01 to 30
ug/kg x 3, 1,000 genes, the same two kinds of cell type; ``four_groups``, the vehicle
and 3 doses a decade apart x 2; ``on_axis``, the vehicle on the axis at an offset of
0.001; ``linear``, a linear axis; ``sex``, ``"~ sex"`` with 2 females and 2 males at
each dose; and ``donors``, 4 donors crossed with 5 dose groups under ``(1 | donor)``.

R, which sees the counts, cells, doses and settings and nothing that CytePool
computed, keeps each cell type's samples of 10 cells or more and some counts, takes
the dose groups that keep a sample and that limma's ``nonEstimable`` does not name
with their indicator placed last, and fits the design's terms with an indicator of
each such treated dose: edgeR's ``filterByExpr`` and ``calcNormFactors``, then
``voom(adaptive.span = TRUE)`` and ``lmFit`` (limma-voom) or ``cpm(log = TRUE,
prior.count = 3)``, ``lmFit`` and ``eBayes(trend = TRUE)`` (limma-trend). It places
the knots with R's own rule (``attr(ns(unique(x), df), "knots")`` on the positions of
the design's samples), fits the spline to the means of the dose groups with its own
columns, ``ns()`` for the spline, Hmisc's ``rcspline.eval`` for the non-linearity and
the line ``x - min(x)`` for the linear trend, each group weighted by its samples, and
tests the rows of A = (Z'NZ)^-1 Z'N as contrasts, with the trapezoidal rule's weights
at the treated doses' positions, over the axis length between the lowest and the top
one, for the average change: ``contrasts.fit``, ``eBayes``,
``topTable(sort.by = "none")`` for t-tests and ``classifyTestsF`` for F-tests, with
p-values from ``pf`` per gene, as ``topTable`` of several columns refers every gene
to the first gene's degrees of freedom when those are infinite, and ``p.adjust`` within
and across cell types. The exact default under voom's weights is checked against each
gene's ``lm.wfit`` with voom's weights and ``eBayes``'s posterior variance. Curves and
95% bands are checked against ``topTable(confint = TRUE)`` of the curve's contrast at
each dose and against the same formula from ``lm.wfit``. rms's ``ols`` and ``anova``
give the partial sums of squares of the spline (TOTAL) and of its non-linear part on
limma-trend's values, which without weights equal F x df1 x s2.post. variancePartition
checks the dream route: ``voomWithDreamWeights``, ``dream`` with the same contrasts as
``L``, ``eBayes`` and ``topTable``. DRomics' ``itemselect`` on log2(CPM + 1) values (its
limma path) and on counts (its DESeq2 path), dreamlet with the spline columns, and
muscat's ``pbDS`` with a dose factor run on the same pseudobulk profiles, each with its
own filters, and print their calls beside CytePool's and the truth's, as information.

Tiers: exact (1e-10) for limma, edgeR, the per-gene fits and rms; close (1e-6) for
dream, with 1e-5 for its degrees of freedom and p-values, as validation/dream.py
bounds them; identical for samples, genes and degrees of freedom. Statistics are
compared as their difference divided by max(|reference|, 1), p-values as their
difference divided by p max(1, |log p|).

Requirements:
    Python 3.13 with numpy and numba, run from the repository root.
    R 4.4 with limma, edgeR, Hmisc, rms, DRomics, DESeq2, muscat, SingleCellExperiment
    and jsonlite (``RSCRIPT``), and variancePartition, dreamlet and jsonlite
    (``DREAM_RSCRIPT``); each variable names an Rscript executable and defaults to
    ``Rscript``.

Usage:
    PYTHONPATH=. python validation/dose_response_tests.py
    PYTHONPATH=. python validation/dose_response_tests.py --write-fixtures
    PYTHONPATH=. python validation/dose_response_tests.py --benchmark
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
from cytepool.design.formula import quote_name
from cytepool.differential_expression import dream_voom, limma_trend, limma_voom
from cytepool._doses import dose_group_column
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import dose_response
from cytepool.simulated_dose_series import simulate_dose_series
from cytepool.stats.distributions import f_sf

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_response_tests"
)
EXACT, CLOSE, DREAM_DF = 1e-10, 1e-6, 1e-5
MIN_CELLS = 10
CALLED = 0.05  # FDR of the calls the information rows count
CURVE_POINTS = 50
FIXTURE_CURVE_POINTS = 30
FIXTURE_CURVE_GENES = 20  # the genes whose curves the fixture keeps
RMS_GENES = 50
TESTS = ("spline", "nonlinearity", "linear_trend", "average_change", "dose_groups")

LIMMA = r"""
suppressPackageStartupMessages({
  library(limma); library(edgeR); library(splines); library(Hmisc); library(rms)
  library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
HELPERS
pvalue <- function(f, df1, df2) pf(f, df1, df2, lower.tail = FALSE)
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("limma", "edgeR", "Hmisc", "rms"), function(p) as.character(packageVersion(p))))))
for (name in names(input$jobs)) {
  job <- input$jobs[[name]]
  case <- input$cases[[job$case]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  axis <- case$axis; own <- axis$vehicle == "own_group"
  position <- position_of(axis)
  x <- position(data$dose); xs <- x[!is.na(x)]
  knots <- c(min(xs), attr(ns(unique(xs), df = job$df), "knots"), max(xs))
  ns_at <- columns_of(position, knots, "ns"); rcs_at <- columns_of(position, knots, "rcs")
  line_at <- columns_of(position, knots[c(1, length(knots))], "line")
  levels <- sort(unique(data$dose))
  result <- list(knots = knots)
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    used <- cells >= input$min_cells & colSums(y) > 0
    d <- data[used, , drop = FALSE]; yy <- y[, used, drop = FALSE]
    treated <- levels[levels > 0 & levels %in% d$dose]
    others <- model.matrix(as.formula(case$formula), d)
    ind <- sapply(treated, function(k) as.numeric(d$dose == k))
    ind <- matrix(ind, nrow = nrow(d)); colnames(ind) <- paste0("dose", seq_along(treated))
    estimable <- sapply(seq_along(treated), function(j) {
      X <- cbind(others, ind[, -j, drop = FALSE], ind[, j, drop = FALSE])
      !(colnames(ind)[j] %in% nonEstimable(X))
    })
    compared <- c(0, treated[estimable]); n <- sapply(compared, function(k) sum(d$dose == k))
    X <- cbind(others, ind[, estimable, drop = FALSE])
    k <- length(compared) - 1
    pad <- function(rows) rbind(matrix(0, ncol(others), nrow(rows)), t(rows))
    dge <- DGEList(yy); keep <- filterByExpr(dge, design = X)
    dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
    voomed <- job$analysis == "voom"
    if (voomed) {
      v <- voom(dge, X, adaptive.span = TRUE); E <- v$E; W <- v$weights; fit <- lmFit(v, X)
    } else {
      E <- cpm(dge, log = TRUE, prior.count = 3); W <- NULL; fit <- lmFit(E, X)
    }
    treated_column <- function(dd) if (own) as.numeric(dd > 0) else NULL
    rows <- list()
    A <- least_squares(cbind(1, treated_column(compared), ns_at(compared)), n)
    if ("spline" %in% job$tests && !is.null(A)) rows$spline <- A[-1, -1, drop = FALSE]
    if ("nonlinearity" %in% job$tests) {
      Ar <- least_squares(cbind(1, treated_column(compared), rcs_at(compared)), n)
      if (!is.null(Ar)) rows$nonlinearity <- Ar[-seq_len(2 + own), -1, drop = FALSE]
    }
    if ("linear_trend" %in% job$tests) {
      A1 <- least_squares(cbind(1, treated_column(compared), line_at(compared)), n)
      if (!is.null(A1)) rows$linear_trend <- A1[nrow(A1), -1, drop = FALSE]
    }
    if ("average_change" %in% job$tests) {
      w <- average_change_row(position, compared)
      if (!is.null(w)) rows$average_change <- w
    }
    if ("dose_groups" %in% job$tests) rows$dose_groups <- diag(k)
    rows$each_dose <- diag(k)
    L <- do.call(cbind, lapply(rows, pad))
    owner <- rep(names(rows), sapply(rows, nrow))
    eb <- eBayes(contrasts.fit(fit, L), trend = !voomed)
    df2 <- eb$df.residual + eb$df.prior
    s2 <- eb$s2.post
    exact_fits <- if (voomed) lapply(seq_len(nrow(E)), function(g) {
      f <- lm.wfit(X, E[g, ], W[g, ])
      stopifnot(identical(f$qr$pivot, seq_len(ncol(X))))
      list(beta = f$coefficients, V = chol2inv(qr.R(f$qr)))
    }) else NULL
    exact_f <- function(Lt) sapply(seq_along(exact_fits), function(g) {
      e <- exact_fits[[g]]; est <- crossprod(Lt, e$beta)
      drop(crossprod(est, solve(crossprod(Lt, e$V %*% Lt), est))) / ncol(Lt) / s2[g]
    })
    exact_se <- function(Lt) do.call(rbind, lapply(seq_along(exact_fits), function(g) {
      e <- exact_fits[[g]]; sqrt(diag(crossprod(Lt, e$V %*% Lt)) * s2[g])
    }))
    res <- list(samples = d$sample_id, doses = compared, n = n, genes = which(keep),
                df_prior = eb$df.prior, s2_post = s2, df_total = eb$df.total, tests = list())
    for (test in setdiff(names(rows), "each_dose")) {
      cols <- which(owner == test)
      if (test %in% c("linear_trend", "average_change")) {
        top <- topTable(eb, coef = cols, sort.by = "none", number = Inf)
        r <- list(logFC = top$logFC, t = top$t, p = top$P.Value, adj = top$adj.P.Val,
                  se = eb$stdev.unscaled[, cols] * sqrt(s2), B = top$B)
        if (voomed) {
          se <- drop(exact_se(L[, cols, drop = FALSE]))
          r$exact_se <- se; r$exact_t <- top$logFC / se
          r$exact_p <- 2 * pt(abs(r$exact_t), eb$df.total, lower.tail = FALSE)
          r$exact_adj <- p.adjust(r$exact_p, "BH")
        }
      } else {
        fs <- classifyTestsF(eb[, cols], fstat.only = TRUE)
        df1 <- attr(fs, "df1"); fs <- as.numeric(fs); p <- pvalue(fs, df1, df2)
        r <- list(F = fs, df1 = df1, df2 = df2 + 0 * fs, p = p, adj = p.adjust(p, "BH"))
        if (voomed) {
          fe <- exact_f(L[, cols, drop = FALSE]); pe <- pvalue(fe, df1, df2)
          r$exact_F <- fe; r$exact_p <- pe; r$exact_adj <- p.adjust(pe, "BH")
        }
      }
      res$tests[[test]] <- r
    }
    # limma's approximate standard errors under voom's weights make its F depend on
    # the basis of the hypothesis: its F of the rows CytePool tests.
    given <- job$given[[ct]]
    if (voomed && length(given)) {
      res$given <- lapply(given, function(rows) {
        rows <- matrix(rows, ncol = k)
        e <- eBayes(contrasts.fit(fit, pad(rows)))
        fs <- classifyTestsF(e, fstat.only = TRUE)
        df1 <- attr(fs, "df1"); fs <- as.numeric(fs)
        p <- pvalue(fs, df1, e$df.residual + e$df.prior)
        list(F = fs, df1 = df1, p = p, adj = p.adjust(p, "BH"))
      })
    }
    cols <- which(owner == "each_dose")
    tops <- lapply(cols, function(j) topTable(eb, coef = j, sort.by = "none", number = Inf))
    column <- function(tables, name) do.call(cbind, lapply(tables, `[[`, name))
    res$each_dose <- list(
      logFC = column(tops, "logFC"), t = column(tops, "t"), p = column(tops, "P.Value"),
      adj = column(tops, "adj.P.Val"), se = eb$stdev.unscaled[, cols, drop = FALSE] * sqrt(s2))
    if (voomed) {
      se <- exact_se(L[, cols, drop = FALSE]); tt <- res$each_dose$logFC / se
      pe <- 2 * pt(abs(tt), eb$df.total, lower.tail = FALSE)
      res$each_dose$exact_se <- se; res$each_dose$exact_t <- tt; res$each_dose$exact_p <- pe
      res$each_dose$exact_adj <- matrix(apply(pe, 2, p.adjust, method = "BH"), nrow = nrow(pe))
    }
    at <- job$curves[[ct]]
    if (length(at) && !is.null(A)) {
      C <- (cbind(1, treated_column(at), ns_at(at)) %*% A)[, -1, drop = FALSE]
      Lc <- pad(C)
      ebc <- eBayes(contrasts.fit(fit, Lc), trend = !voomed)
      tops <- lapply(seq_len(ncol(Lc)), function(j)
        topTable(ebc, coef = j, sort.by = "none", number = Inf, confint = TRUE))
      res$curve <- list(at = at, estimate = column(tops, "logFC"),
                        lower = column(tops, "CI.L"), upper = column(tops, "CI.R"))
      if (voomed) {
        est <- res$curve$estimate; se <- exact_se(Lc); q <- qt(0.975, eb$df.total)
        res$curve$exact_lower <- est - q * se; res$curve$exact_upper <- est + q * se
      }
    }
    if (job$rms > 0 && !voomed && job$df >= 2) {
      xr <- position(d$dose); if (own) xr[is.na(xr)] <- knots[1]
      spec <- sprintf("rcs(x, c(%s))", paste(sprintf("%.17g", knots), collapse = ", "))
      form <- as.formula(paste("y ~", if (own) "tr +", spec))
      ss <- t(sapply(seq_len(min(job$rms, nrow(E))), function(g) {
        a <- anova(ols(form, data = data.frame(y = E[g, ], x = xr, tr = as.numeric(d$dose > 0))))
        c(a[rownames(a) == "TOTAL", "Partial SS"], a[trimws(rownames(a)) == "Nonlinear", "Partial SS"])
      }))
      res$rms <- list(total = ss[, 1], nonlinear = ss[, 2])
    }
    result$cell_types[[ct]] <- res
  }
  for (test in c("spline", "nonlinearity")) {
    have <- Filter(function(ct) !is.null(result$cell_types[[ct]]$given[[test]]), names(result$cell_types))
    if (!length(have)) next
    stacked <- unlist(lapply(have, function(ct) result$cell_types[[ct]]$given[[test]]$p))
    adjusted <- p.adjust(stacked, "BH"); start <- 0
    for (ct in have) {
      m <- length(result$cell_types[[ct]]$given[[test]]$p)
      result$cell_types[[ct]]$given[[test]]$p_across <- adjusted[start + seq_len(m)]
      start <- start + m
    }
  }
  # Benjamini-Hochberg across the cell types that ran each test.
  for (test in job$tests) {
    have <- Filter(function(ct) !is.null(result$cell_types[[ct]]$tests[[test]]), names(result$cell_types))
    if (!length(have)) next
    for (kind in c("p", "exact_p")) {
      if (is.null(result$cell_types[[have[1]]]$tests[[test]][[kind]])) next
      stacked <- unlist(lapply(have, function(ct) result$cell_types[[ct]]$tests[[test]][[kind]]))
      adjusted <- p.adjust(stacked, "BH"); start <- 0
      for (ct in have) {
        m <- length(result$cell_types[[ct]]$tests[[test]][[kind]])
        result$cell_types[[ct]]$tests[[test]][[paste0(kind, "_across")]] <- adjusted[start + seq_len(m)]
        start <- start + m
      }
    }
  }
  out$jobs[[name]] <- result
}
write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE)
"""

# The helpers that the R programs share: the axis, the spline's columns, least squares
# through the dose groups' means, and each cell type's samples and dose groups.
R_HELPERS = r"""
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
position_of <- function(axis) function(d) {
  if (axis$vehicle == "own_group") {
    out <- rep(NA_real_, length(d)); t <- d > 0
    out[t] <- if (axis$scale == "log10") log10(d[t]) else d[t]
    out
  } else if (axis$scale == "log10") log10(d + axis$offset) else d
}
columns_of <- function(position, knots, form) function(d) {
  z <- position(d); on <- !is.na(z); k <- length(knots)
  width <- if (form == "line") 1 else k - 1
  m <- matrix(0, length(d), width)
  if (any(on)) {
    zz <- z[on]
    m[on, ] <- switch(form,
      line = zz - knots[1],
      ns = ns(zz, knots = knots[-c(1, k)], Boundary.knots = knots[c(1, k)]),
      rcs = if (k == 2) matrix(zz - knots[1]) else {
        r <- rcspline.eval(zz, knots = knots, inclx = TRUE); r[, 1] <- r[, 1] - knots[1]; r
      })
  }
  m
}
least_squares <- function(Z, n) {
  if (qr(Z)$rank < ncol(Z)) return(NULL)
  solve(crossprod(Z, Z * n), t(Z * n))
}
# The average change along the axis: the trapezoidal rule's weights at the treated
# doses' positions, divided by the axis length from the lowest to the top one; NULL
# with fewer than two treated doses.
average_change_row <- function(position, compared) {
  u <- position(compared[-1])
  if (length(u) < 2) return(NULL)
  g <- diff(u)
  matrix((c(g, 0) + c(0, g)) / 2 / (u[length(u)] - u[1]), nrow = 1)
}
# The samples a cell type keeps, its dose groups, and the contrasts of each test on
# the treated doses compared (rows), from R's own knots and columns.
cell_type_of <- function(case, ct, df, tests, min_cells) {
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  axis <- case$axis; own <- axis$vehicle == "own_group"
  position <- position_of(axis)
  x <- position(data$dose); xs <- x[!is.na(x)]
  knots <- c(min(xs), attr(ns(unique(xs), df = df), "knots"), max(xs))
  y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
  used <- cells >= min_cells & colSums(y) > 0
  d <- data[used, , drop = FALSE]
  levels <- sort(unique(data$dose)); treated <- levels[levels > 0 & levels %in% d$dose]
  compared <- c(0, treated); n <- sapply(compared, function(k) sum(d$dose == k))
  ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
  colnames(ind) <- paste0("dose", seq_along(treated))
  tr <- function(dd) if (own) as.numeric(dd > 0) else NULL
  rows <- list()
  A <- least_squares(cbind(1, tr(compared), columns_of(position, knots, "ns")(compared)), n)
  if ("spline" %in% tests && !is.null(A)) rows$spline <- A[-1, -1, drop = FALSE]
  # Departure from a line through ns() alone: the line x minus the lowest knot is a
  # combination a of the spline's columns, and the curve is a line when the spline's
  # coefficients are a multiple of a, so the test is of their part orthogonal to a.
  if ("nonlinearity" %in% tests && !is.null(A) && length(knots) > 2) {
    grid <- seq(knots[1], knots[length(knots)], length.out = 50)
    N <- ns(grid, knots = knots[-c(1, length(knots))], Boundary.knots = knots[c(1, length(knots))])
    a <- qr.solve(N, grid - knots[1])
    M <- qr.Q(qr(cbind(a)), complete = TRUE)[, -1, drop = FALSE]
    rows$nonlinearity <- crossprod(M, A[-seq_len(1 + own), -1, drop = FALSE])
  }
  A1 <- least_squares(cbind(1, tr(compared), columns_of(position, knots[c(1, length(knots))], "line")(compared)), n)
  if ("linear_trend" %in% tests && !is.null(A1)) {
    rows$linear_trend <- A1[nrow(A1), -1, drop = FALSE]
  }
  if ("average_change" %in% tests) {
    w <- average_change_row(position, compared)
    if (!is.null(w)) rows$average_change <- w
  }
  if ("dose_groups" %in% tests) rows$dose_groups <- diag(length(treated))
  list(data = d, counts = y[, used, drop = FALSE], indicators = ind, compared = compared,
       n = n, rows = rows, knots = knots, position = position, own = own)
}
"""

LIMMA = LIMMA.replace("HELPERS", R_HELPERS)

DREAM = r"""
suppressPackageStartupMessages({
  library(variancePartition); library(lme4); library(edgeR); library(splines); library(dreamlet)
  library(SingleCellExperiment); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
HELPERS
tight_bobyqa <- function(par, fn, lower, upper, control) {
  res <- minqa::bobyqa(par, fn, lower = lower, upper = upper,
    control = list(rhobeg = 0.2, rhoend = 1e-10, maxfun = 1e5))
  list(par = res$par, fval = res$fval, conv = 0, message = res$msg)
}
tight <- lmerControl(optimizer = "tight_bobyqa", calc.derivs = FALSE, check.conv.singular = "ignore")
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("variancePartition", "limma", "edgeR", "lme4", "dreamlet"),
  function(p) as.character(packageVersion(p))))))
for (name in names(input$dream)) {
  job <- input$dream[[name]]
  case <- input$cases[[job$case]]
  result <- list()
  for (ct in names(case$cell_types)) {
    c <- cell_type_of(case, ct, job$df, job$tests, input$min_cells)
    d <- cbind(c$data, c$indicators); d$donor <- factor(d$donor)
    X <- cbind("(Intercept)" = 1, c$indicators)
    dge <- DGEList(c$counts); keep <- filterByExpr(dge, design = X)
    dge <- dge[keep, , keep.lib.sizes = FALSE]; dge <- calcNormFactors(dge)
    form <- as.formula(paste("~", paste(colnames(c$indicators), collapse = " + "), "+ (1 | donor)"))
    v <- quiet(voomWithDreamWeights(dge, form, d, span = 0.5, control = tight))
    rows <- c(c$rows, list(each_dose = diag(ncol(c$indicators))))
    owner <- rep(names(rows), sapply(rows, nrow))
    L <- rbind(0, do.call(cbind, lapply(rows, t)))
    rownames(L) <- c("(Intercept)", colnames(c$indicators))
    colnames(L) <- paste0("L", seq_len(ncol(L)))
    fit <- quiet(dream(v, form, d, L = L, control = tight))
    e <- eBayes(fit)
    res <- list(samples = d$sample_id, genes = which(keep), tests = list())
    for (test in setdiff(names(rows), "each_dose")) {
      cols <- colnames(L)[owner == test]
      top <- topTable(e, coef = cols, number = Inf, sort.by = "none")
      res$tests[[test]] <- if (length(cols) == 1)
        list(logFC = top$logFC, t = top$t, p = top$P.Value, df = e$df.total[, cols])
      else list(F = top$F, p = top$P.Value, df1 = length(cols), df_first = e$df.total[1, cols[1]])
    }
    cols <- colnames(L)[owner == "each_dose"]
    tops <- lapply(cols, function(k) topTable(e, coef = k, number = Inf, sort.by = "none"))
    res$each_dose <- list(logFC = do.call(cbind, lapply(tops, `[[`, "logFC")),
                          t = do.call(cbind, lapply(tops, `[[`, "t")),
                          p = do.call(cbind, lapply(tops, `[[`, "P.Value")))
    result$cell_types[[ct]] <- res
  }
  out$dream[[name]] <- result
}
# dreamlet with the spline's columns, on cells that sum to each profile: the first
# cell of a profile holds its counts and the others none, so that dreamlet's
# pseudobulk is the cube's, with its cells.
for (name in names(input$dreamlet)) {
  job <- input$dreamlet[[name]]
  case <- input$cases[[job$case]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  position <- position_of(case$axis); x <- position(data$dose); xs <- x[!is.na(x)]
  knots <- c(min(xs), attr(ns(unique(xs), df = job$df), "knots"), max(xs))
  spline <- columns_of(position, knots, "ns")(data$dose)
  colnames(spline) <- paste0("s", seq_len(ncol(spline)))
  data$treated <- as.numeric(data$dose > 0)
  blocks <- list(); meta <- list()
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    for (s in seq_len(nrow(data))) {
      if (cells[s] == 0) next
      m <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0),
                                dims = c(nrow(y), cells[s]))
      m[, 1] <- y[, s]
      blocks[[length(blocks) + 1]] <- m
      meta[[length(meta) + 1]] <- data.frame(sample = data$sample_id[s], cell_type = ct,
                                             n = cells[s])
    }
  }
  meta <- do.call(rbind, meta)
  counts <- do.call(cbind, blocks)
  rownames(counts) <- paste0("g", seq_len(nrow(counts)))
  colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  cd <- meta[rep(seq_len(nrow(meta)), meta$n), c("sample", "cell_type")]
  rownames(cd) <- colnames(counts)
  sce <- SingleCellExperiment(assays = list(counts = counts), colData = DataFrame(cd))
  pb <- quiet(aggregateToPseudoBulk(sce, assay = "counts", cluster_id = "cell_type",
                                    sample_id = "sample", verbose = FALSE))
  extra <- data[match(colnames(pb), data$sample_id), ]
  for (term in c("treated", colnames(spline))) {
    colData(pb)[[term]] <- if (term == "treated") extra$treated else
      spline[match(colnames(pb), data$sample_id), term]
  }
  terms <- c(if (case$axis$vehicle == "own_group") "treated", colnames(spline))
  form <- as.formula(paste("~", paste(terms, collapse = " + ")))
  processed <- quiet(processAssays(pb, form, min.cells = input$min_cells, verbose = FALSE))
  fit <- quiet(dreamlet(processed, form, verbose = FALSE))
  top <- as.data.frame(topTable(fit, coef = terms, number = Inf))
  out$dreamlet[[name]] <- lapply(split(top, top$assay), function(t)
    list(genes = as.integer(sub("^g", "", t$ID)), adj = p.adjust(t$P.Value, "BH")))
}
write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE)
""".replace("HELPERS", R_HELPERS)

TOOLS = r"""
suppressPackageStartupMessages({
  library(edgeR); library(splines); library(DRomics); library(muscat)
  library(SingleCellExperiment); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
quiet <- function(expr) suppressWarnings(suppressMessages(expr))
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("DRomics", "DESeq2", "limma", "muscat", "edgeR"), function(p) as.character(packageVersion(p))))))
for (name in names(input$tools)) {
  job <- input$tools[[name]]
  case <- input$cases[[job$case]]
  data <- as.data.frame(case$samples, stringsAsFactors = FALSE)
  result <- list()
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    used <- cells >= input$min_cells & colSums(y) > 0
    yy <- y[, used, drop = FALSE]; dose <- data$dose[used]
    ids <- paste0("g", seq_len(nrow(yy)))
    # The layout of read.table(file, header = FALSE): the doses on the first line.
    as_input <- function(values) {
      table <- data.frame(c("item", ids), rbind(dose, values), stringsAsFactors = FALSE,
                          row.names = NULL)
      colnames(table) <- paste0("V", seq_len(ncol(table)))
      table
    }
    dge <- calcNormFactors(DGEList(yy))
    logged <- log2(edgeR::cpm(dge) + 1)
    r <- list()
    continuous <- quiet(continuousomicdata(as_input(logged)))
    for (method in c("quadratic", "linear")) {
      s <- quiet(itemselect(continuous, select.method = method, FDR = input$fdr))
      r[[paste0("limma_", method)]] <- as.integer(s$selectindex)
    }
    counted <- quiet(RNAseqdata(as_input(yy)))
    s <- quiet(itemselect(counted, select.method = "quadratic", FDR = input$fdr))
    r$deseq2_quadratic <- as.integer(s$selectindex)
    result[[ct]] <- r
  }
  # muscat's pbDS with a factor of dose groups, every dose coefficient at once, on
  # cells that sum to each profile, as for dreamlet.
  blocks <- list(); meta <- list()
  for (ct in names(case$cell_types)) {
    y <- case$cell_types[[ct]]$counts; cells <- case$cell_types[[ct]]$n_cells
    for (s in seq_len(nrow(data))) {
      if (cells[s] == 0) next
      m <- Matrix::sparseMatrix(i = integer(0), j = integer(0), x = numeric(0), dims = c(nrow(y), cells[s]))
      m[, 1] <- y[, s]
      blocks[[length(blocks) + 1]] <- m
      meta[[length(meta) + 1]] <- data.frame(sample_id = data$sample_id[s], cluster_id = ct,
                                             group_id = sprintf("d%02d", match(data$dose[s], sort(unique(data$dose)))),
                                             n = cells[s])
    }
  }
  meta <- do.call(rbind, meta); counts <- do.call(cbind, blocks)
  rownames(counts) <- paste0("g", seq_len(nrow(counts))); colnames(counts) <- paste0("c", seq_len(ncol(counts)))
  cd <- meta[rep(seq_len(nrow(meta)), meta$n), c("sample_id", "cluster_id", "group_id")]
  rownames(cd) <- colnames(counts)
  sce <- SingleCellExperiment(assays = list(counts = counts), colData = DataFrame(cd))
  sce <- quiet(prepSCE(sce, kid = "cluster_id", gid = "group_id", sid = "sample_id", drop = TRUE))
  pb <- quiet(aggregateData(sce, assay = "counts", fun = "sum", by = c("cluster_id", "sample_id")))
  ei <- metadata(sce)$experiment_info
  design <- model.matrix(~ group_id, ei); rownames(design) <- ei$sample_id
  res <- quiet(pbDS(pb, method = "edgeR", design = design, coef = list(seq(2, ncol(design))),
                    min_cells = input$min_cells, verbose = FALSE))
  tables <- res$table[[1]]
  out$muscat[[name]] <- lapply(tables, function(t)
    list(genes = as.integer(sub("^g", "", t$gene)), adj = t$p_adj.loc))
  out$dromics[[name]] <- result
}
write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE)
"""


# The dose series.


@dataclass(frozen=True)
class Job:
    """One analysis of a series by both programs."""

    name: str
    analysis: str  # "voom" or "trend"
    df: int
    tests: tuple[str, ...]
    curves: bool = False
    rms: int = 0  # genes whose sums of squares rms gives


@dataclass
class Series:
    """A simulated dose series, its design and axis, and the analyses run on it."""

    name: str
    cube: Cube
    truth: Any
    design: Design
    axis: DoseAxis
    jobs: list[Job]
    random: bool = False  # donors crossed with dose, under (1 | donor)


HALF_LOG = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")


def series() -> list[Series]:
    """The dose series, from fixed seeds."""
    found = []

    def add(name, doses, axis, jobs, *, seed, formula="~ 1", columns=None, **options):
        cube, truth = simulate_dose_series(doses, rng=seed, **options)
        samples = cube.samples.with_columns(columns(cube.samples)) if columns else None
        table = cube.samples if samples is None else samples
        random = "(1 | donor)" if options.get("donor_sd") is not None else None
        design = Design(table, formula, random=random)
        found.append(Series(name, cube, truth, design, axis, jobs, random is not None))

    every = TESTS
    add(
        "fixture",
        [0.0, 0.1, 0.3, 1.0, 3.0, 10.0],
        AXIS,
        [
            Job("fixture voom", "voom", 2, every, curves=True),
            Job("fixture trend", "trend", 2, every, curves=True, rms=RMS_GENES),
        ],
        seed=20261208,
        replicates=3,
        cell_types=["full", "depleted"],
        cells=[300.0, 200.0],
        n_genes=300,
        composition={"depleted": [1.0, 1.0, 1.0, 1.0, 0.05, 0.015]},
    )
    add(
        "ladder",
        HALF_LOG,
        AXIS,
        [
            Job("ladder voom", "voom", 2, every, curves=True),
            Job("ladder voom, 3 columns", "voom", 3, every),
            Job("ladder trend", "trend", 2, every, curves=True, rms=RMS_GENES),
            Job("ladder trend, 3 columns", "trend", 3, every, rms=RMS_GENES),
        ],
        seed=20261210,
        replicates=3,
        cell_types=["abundant", "depleted"],
        cells=[300.0, 200.0],
        n_genes=1000,
        composition={"depleted": [1.0] * 7 + [0.05, 0.015]},
    )
    add(
        "four_groups",
        [0.0, 0.1, 1.0, 10.0],
        AXIS,
        [
            Job(
                "four_groups voom",
                "voom",
                1,
                ("spline", "linear_trend", "dose_groups"),
                True,
            ),
            Job("four_groups trend, 2 columns", "trend", 2, every, curves=True),
        ],
        seed=20261211,
        replicates=2,
        cell_types=1,
        cells=300.0,
        n_genes=500,
    )
    add(
        "on_axis",
        [0.0, 0.01, 0.1, 1.0, 10.0],
        DoseAxis(
            "dose", unit="ug/kg", metric="administered", vehicle="on_axis", offset=0.001
        ),
        [
            Job("on_axis voom", "voom", 2, every, curves=True),
            Job("on_axis trend", "trend", 2, every, curves=True, rms=RMS_GENES),
        ],
        seed=20261212,
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=500,
    )
    add(
        "linear",
        [0.0, 1.0, 2.0, 4.0, 8.0],
        DoseAxis("dose", unit="mg/kg", metric="administered", scale="linear"),
        [
            Job("linear voom", "voom", 2, every, curves=True),
            Job("linear trend", "trend", 2, every, rms=RMS_GENES),
        ],
        seed=20261213,
        replicates=3,
        cell_types=1,
        cells=300.0,
        n_genes=500,
        unit="mg/kg",
    )
    add(
        "sex",
        [0.0, 0.1, 1.0, 10.0, 30.0],
        AXIS,
        [
            Job("sex voom", "voom", 2, every, curves=True),
            Job("sex trend", "trend", 2, every),
        ],
        seed=20261214,
        formula="~ sex",
        columns=lambda s: {
            "sex": [
                "F" if sample.endswith(("_r1", "_r2")) else "M"
                for sample in s["sample_id"].tolist()
            ]
        },
        replicates=4,
        cell_types=1,
        cells=300.0,
        n_genes=500,
    )
    add(
        "donors",
        [0.0, 0.1, 1.0, 10.0, 100.0],
        AXIS,
        [Job("donors dream", "dream", 2, every)],
        seed=20261215,
        replicates=4,
        cell_types=1,
        cells=300.0,
        n_genes=300,
        donor_sd=0.3,
    )
    return found


def kept_doses(cube: Cube, g: int, doses: np.ndarray) -> np.ndarray:
    """The doses of the samples a cell type keeps, by limma_voom's rule."""
    counts = cube.layer("counts", g)
    used = (np.asarray(cube.n_cells)[:, g] >= MIN_CELLS) & (counts.sum(axis=1) > 0)
    return np.unique(doses[used])


def curve_doses(axis: DoseAxis, kept: np.ndarray, points: int) -> np.ndarray:
    """Doses at which to compare curves: the vehicle, and doses spread along the axis
    over the doses a cell type keeps, with one between the vehicle and the lowest
    dose when the vehicle sits on the axis."""
    treated = kept[kept > 0.0]
    low, high = treated[0], treated[-1]
    if axis.scale == "linear":
        spread = np.linspace(low, high, points - 2)
    else:
        spread = np.geomspace(low, high, points - 2)
    between = [low / 3.0] if axis.vehicle == "on_axis" else [float(np.sqrt(low * high))]
    return np.sort(np.r_[0.0, between, spread])


# The inputs of R, which see the counts, cells, doses and settings.


def given_rows(result: Any) -> dict[str, dict[str, list[list[float]]]]:
    """The rows of the spline and non-linearity tests of each cell type, on the
    treated doses it compares, as CytePool tests them."""
    found: dict[str, dict[str, list[list[float]]]] = {}
    columns = result.provenance["design"]["columns"]
    for entry in result.provenance["groups"]:
        if not entry["tested"]:
            continue
        ct = entry["group_id"]
        # The design names a column as its formula does, in backticks where needed.
        positions = [
            columns.index(quote_name(dose_group_column("dose", dose)))
            for dose in entry["doses"][1:]
        ]
        tests = entry["tests"]
        found[ct] = {
            test: [
                [entry["contrasts"][member][p] for p in positions]
                for member in tests[test]
            ]
            for test in ("spline", "nonlinearity")
            if test in tests
        }
    return found


def r_input(found: list[Series], runs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cases = {}
    for item in found:
        samples = item.design.samples
        columns = ["sample_id", "dose"] + [
            name for name in ("sex", "donor") if name in samples
        ]
        doses = np.asarray(samples["dose"], dtype=float)
        cell_types = {}
        for g, ct in enumerate(item.cube.groups["group_id"].tolist()):
            cell_types[ct] = {
                "counts": item.cube.layer("counts", g).T.astype(int).tolist(),
                "n_cells": np.asarray(item.cube.n_cells)[:, g].astype(int).tolist(),
            }
        axis = item.axis
        cases[item.name] = {
            "samples": {name: samples[name].tolist() for name in columns},
            "cell_types": cell_types,
            "formula": item.design.formula,
            "axis": {
                "scale": axis.scale,
                "vehicle": axis.vehicle,
                "offset": axis.offset,
            },
        }
        del doses
    jobs: dict[str, Any] = {}
    dream: dict[str, Any] = {}
    for item in found:
        groups = item.cube.groups["group_id"].tolist()
        for job in item.jobs:
            doses = np.asarray(item.design.samples["dose"], dtype=float)
            points = FIXTURE_CURVE_POINTS if item.name == "fixture" else CURVE_POINTS
            curves = {
                ct: curve_doses(
                    item.axis, kept_doses(item.cube, g, doses), points
                ).tolist()
                if job.curves
                else []
                for g, ct in enumerate(groups)
            }
            given = (
                given_rows(runs[job.name]["limma"]) if job.analysis == "voom" else {}
            )
            entry = {
                "name": job.name,
                "given": given,
                "case": item.name,
                "analysis": job.analysis,
                "df": job.df,
                "tests": list(job.tests),
                "curves": curves,
                "rms": job.rms,
            }
            (dream if job.analysis == "dream" else jobs)[job.name] = entry
    return {
        "cases": cases,
        "jobs": jobs,
        "dream": dream,
        "dreamlet": {"ladder": {"case": "ladder", "df": 2}},
        "tools": {"ladder": {"case": "ladder"}},
        "min_cells": MIN_CELLS,
        "fdr": CALLED,
    }


def run_r(program: str, payload: dict[str, Any], variable: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "input.json").write_text(json.dumps(payload))
        (root / "program.R").write_text(program)
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


# CytePool's analyses.


def ours(item: Series, job: Job, standard_errors: str = "exact") -> Any:
    options: dict[str, Any] = {}
    if job.analysis == "voom":
        analysis = limma_voom
        options["contrast_standard_errors"] = standard_errors
    elif job.analysis == "trend":
        analysis = limma_trend
    else:
        analysis, options = dream_voom, {"span": 0.5}
    return dose_response(
        item.cube,
        item.design,
        dose=item.axis,
        tests=list(job.tests),
        spline_df=job.df,
        analysis=analysis,
        **options,
    )


# Comparisons.


def scaled(ours: Any, reference: Any) -> float:
    """The largest difference over max(|reference|, 1); infinite when the shapes or
    the NaNs differ."""
    a, b = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return np.inf
    if not a.size:
        return 0.0
    present = ~np.isnan(b)
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
    both_zero = (a == 0.0) & (b == 0.0)
    with np.errstate(divide="ignore"):
        scale = np.maximum(b, 1e-300) * np.maximum(
            1.0, np.abs(np.log(np.maximum(b, 1e-300)))
        )
    difference = np.where(both_zero, 0.0, np.abs(a - b) / scale)
    return float(np.max(difference))


@dataclass
class Checks:
    """The largest difference of each family of checks, and the cases it covers."""

    largest: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    # Values reported for context: the change of limma's approximate F with the
    # spline's basis, and CytePool's F-test p-values over dream's.
    basis: list[float] = field(default_factory=list)
    ratios: list[float] = field(default_factory=list)

    def record(self, family: str, value: float) -> None:
        # A NaN difference fails: max() would keep whichever value comes first.
        value = np.inf if np.isnan(value) else value
        self.largest[family] = max(self.largest.get(family, 0.0), value)
        self.counts[family] = self.counts.get(family, 0) + 1

    def same(self, family: str, label: str, equal: bool) -> None:
        self.record(family, 0.0 if equal else np.inf)
        if not equal:
            self.mismatches.append(f"{family}: {label}")


def vector(values: Any) -> np.ndarray:
    return np.atleast_1d(np.asarray(values, dtype=float))


def matrix(values: Any, columns: int) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1, columns)


def analyses(item: Series, job: Job) -> dict[str, Any]:
    """CytePool's results of a job: limma-voom with exact standard errors and with
    limma's, or limma-trend."""
    runs = {"exact": ours(item, job)}
    if job.analysis == "voom":
        runs["limma"] = ours(item, job, "limma")
    return runs


def compare_limma(
    item: Series, job: Job, runs: dict[str, Any], reference: dict, checks: Checks
) -> None:
    """Compare CytePool's tables of a job with R's."""
    label = "limma-voom" if job.analysis == "voom" else "limma-trend"
    features = item.cube.features["feature_id"]
    for ct, theirs in reference["cell_types"].items():
        for kind, result in runs.items():
            # limma-voom's default against each gene's lm.wfit; the rest against
            # limma's own tables.
            exact = job.analysis == "voom" and kind == "exact"
            name = (
                f"{label}, exact default"
                if exact
                else f"{label}, limma's standard errors"
                if job.analysis == "voom"
                else label
            )
            entry = next(e for e in result.provenance["groups"] if e["group_id"] == ct)
            checks.same(
                "samples, dose groups and genes",
                f"{job.name}, {ct}",
                entry["tested"]
                and entry["fit"]["samples"] == list(np.atleast_1d(theirs["samples"]))
                and entry["doses"] == vector(theirs["doses"]).tolist()
                and entry["samples"] == vector(theirs["n"]).tolist()
                and result.table(ct, "each_dose")["feature_id"].tolist()
                == np.tile(
                    features[vector(theirs["genes"]).astype(int) - 1],
                    len(entry["doses"]) - 1,
                ).tolist(),
            )
            for test, values in theirs["tests"].items():
                table = result.table(ct, test)
                prefix = "exact_" if exact else ""
                across = table["adjusted_p_value_across_groups"]
                if test in ("linear_trend", "average_change"):
                    se = values[prefix + "se"] if exact else values["se"]
                    t = values["exact_t"] if exact else values["t"]
                    p = values["exact_p"] if exact else values["p"]
                    what = "slopes" if test == "linear_trend" else "average changes"
                    checks.record(
                        f"{name}: {what}, standard errors and t",
                        max(
                            scaled(table["log2_fold_change"], values["logFC"]),
                            scaled(table["standard_error"], se),
                            scaled(table["t"], t),
                        ),
                    )
                    checks.record(
                        f"{name}: p-values within and across cell types",
                        max(
                            p_scaled(table["p_value"], p),
                            p_scaled(table["adjusted_p_value"], values[prefix + "adj"]),
                            p_scaled(across, values[prefix + "p_across"]),
                        ),
                    )
                    continue
                if kind == "limma" and test in theirs.get("given", {}):
                    # limma's F of the rows CytePool tests, which its approximate
                    # standard errors depend on.
                    own = vector(values["F"])
                    values = theirs["given"][test]
                    prefix = ""
                    checks.basis.append(
                        float(np.max(np.abs(vector(values["F"]) / own - 1.0)))
                    )
                f = values["exact_F"] if exact else values["F"]
                p = values["exact_p"] if exact else values["p"]
                adjusted = values["exact_adj"] if exact else values["adj"]
                checks.record(f"{name}: F-statistics", scaled(table["f_statistic"], f))
                if "df2" in values:
                    checks.record(
                        f"{name}: F-statistics", scaled(table["df"], values["df2"])
                    )
                checks.record(
                    f"{name}: p-values within and across cell types",
                    max(
                        p_scaled(table["p_value"], p),
                        p_scaled(table["adjusted_p_value"], adjusted),
                        p_scaled(across, values[prefix + "p_across"]),
                    ),
                )
                checks.same(
                    "numerator degrees of freedom",
                    f"{job.name}, {ct}, {test}",
                    entry["f_test_df1"][test] == int(values["df1"]),
                )
            each = result.table(ct, "each_dose")
            k = len(entry["doses"]) - 1
            values = theirs["each_dose"]
            columns = {
                "log2_fold_change": "logFC",
                "standard_error": "exact_se" if exact else "se",
                "t": "exact_t" if exact else "t",
            }
            difference = 0.0
            for ours_name, their_name in columns.items():
                mine = np.asarray(each[ours_name], dtype=float).reshape(k, -1).T
                difference = max(
                    difference, scaled(mine, matrix(values[their_name], k))
                )
            checks.record(f"{name}: each dose against the vehicle", difference)
            p = np.asarray(each["p_value"], dtype=float).reshape(k, -1).T
            adjusted = (
                np.asarray(each["adjusted_p_value"], dtype=float).reshape(k, -1).T
            )
            checks.record(
                f"{name}: p-values within and across cell types",
                max(
                    p_scaled(p, matrix(values["exact_p" if exact else "p"], k)),
                    p_scaled(
                        adjusted, matrix(values["exact_adj" if exact else "adj"], k)
                    ),
                ),
            )
            if "curve" in theirs and job.analysis in ("voom", "trend"):
                at = vector(theirs["curve"]["at"])
                curve = result.curve(ct, at=at)
                n = at.size
                lower = theirs["curve"]["exact_lower" if exact else "lower"]
                upper = theirs["curve"]["exact_upper" if exact else "upper"]
                checks.record(
                    f"{name}: curves and 95% bands",
                    max(
                        scaled(
                            curve["log2_fold_change"].reshape(-1, n),
                            matrix(theirs["curve"]["estimate"], n),
                        ),
                        scaled(curve["ci_lower"].reshape(-1, n), matrix(lower, n)),
                        scaled(curve["ci_upper"].reshape(-1, n), matrix(upper, n)),
                    ),
                )
            if "rms" in theirs and kind == "exact":
                fit = result.fit(ct)
                s2 = fit.moderated.posterior_variance[
                    : vector(theirs["rms"]["total"]).size
                ]
                for test, key in (("spline", "total"), ("nonlinearity", "nonlinear")):
                    table = result.table(ct, test)
                    m = vector(theirs["rms"][key]).size
                    sums = table["f_statistic"][:m] * entry["f_test_df1"][test] * s2[:m]
                    checks.record(
                        "rms: partial sums of squares",
                        scaled_relative(sums, theirs["rms"][key]),
                    )


def scaled_relative(ours: Any, reference: Any) -> float:
    a, b = np.asarray(ours, dtype=float), vector(reference)
    return float(np.max(np.abs(a - b) / np.abs(b)))


def compare_dream(item: Series, job: Job, reference: dict, checks: Checks) -> None:
    """dream's route against variancePartition: the same contrasts as L."""
    result = ours(item, job)
    features = item.cube.features["feature_id"]
    for ct, theirs in reference["cell_types"].items():
        entry = next(e for e in result.provenance["groups"] if e["group_id"] == ct)
        genes = features[vector(theirs["genes"]).astype(int) - 1].tolist()
        checks.same(
            "samples, dose groups and genes",
            f"{job.name}, {ct}",
            entry["tested"]
            and entry["fit"]["samples"] == list(np.atleast_1d(theirs["samples"]))
            and result.table(ct, "dose_groups")["feature_id"].tolist() == genes,
        )
        for test, values in theirs["tests"].items():
            table = result.table(ct, test)
            if test in ("linear_trend", "average_change"):
                checks.record(
                    "dream: F-statistics, slopes and t",
                    max(
                        scaled(table["log2_fold_change"], values["logFC"]),
                        scaled(table["t"], values["t"]),
                    ),
                )
                checks.record(
                    "dream: degrees of freedom and p-values",
                    max(
                        scaled(table["df"], values["df"]),
                        p_scaled(table["p_value"], values["p"]),
                    ),
                )
            else:
                # topTable gives a test of one contrast as a t-test: F is t^2.
                f = values["F"] if "F" in values else vector(values["t"]) ** 2
                checks.record(
                    "dream: F-statistics, slopes and t",
                    scaled(table["f_statistic"], f),
                )
                if "F" not in values:
                    checks.record(
                        "dream: degrees of freedom and p-values",
                        p_scaled(table["p_value"], values["p"]),
                    )
                    continue
                # topTable refers every gene's F to the first gene's degrees of
                # freedom; CytePool gives each gene its own.
                rule = f_sf(
                    vector(values["F"]), float(values["df1"]), float(values["df_first"])
                )
                checks.record(TOPTABLE_RULE, p_scaled(rule, values["p"]))
                checks.ratios.extend((table["p_value"] / vector(values["p"])).tolist())
        each = result.table(ct, "each_dose")
        k = len(entry["doses"]) - 1
        for ours_name, their_name in (("log2_fold_change", "logFC"), ("t", "t")):
            mine = np.asarray(each[ours_name], dtype=float).reshape(k, -1).T
            checks.record(
                "dream: F-statistics, slopes and t",
                scaled(mine, matrix(theirs["each_dose"][their_name], k)),
            )
        mine = np.asarray(each["p_value"], dtype=float).reshape(k, -1).T
        checks.record(
            "dream: degrees of freedom and p-values",
            p_scaled(mine, matrix(theirs["each_dose"]["p"], k)),
        )


# Information: other routes and tools against CytePool's calls and the truth.


def truth_counts(item: Series, ct: str, genes: list[str], called: set[str]) -> str:
    """The responsive and flat genes called among those tested, against the truth."""
    table = item.truth.table(ct)
    size = dict(
        zip(table["feature_id"].tolist(), np.abs(table["max_log2_change"]).tolist())
    )
    responsive = [g for g in genes if size[g] > 0]
    flat = [g for g in genes if size[g] == 0]
    hits = sum(g in called for g in responsive)
    false = sum(g in called for g in flat)
    return f"{hits} of {len(responsive)} responsive, {false} of {len(flat)} flat"


def information(
    found: list[Series], results: dict, tools: dict
) -> list[tuple[str, ...]]:
    rows = []
    ladder = next(item for item in found if item.name == "ladder")
    result = results["ladder voom"]["exact"]
    features = ladder.cube.features["feature_id"]
    # The spline as contrasts of the dose-group fit, against a fit of the spline's
    # columns, on the same genes.
    basis = ladder.axis.spline_basis(ladder.design.samples, df=2)
    columns = ladder.axis.design_columns(ladder.design.samples, basis)
    spline_design = Design(
        ladder.design.samples.with_columns(columns), "~ " + " + ".join(columns)
    )
    direct = limma_voom(
        ladder.cube,
        spline_design,
        contrasts=list(columns),
        f_tests={"spline": list(columns)},
        expression_filter={"sample_groups": "dose_group"},
    )
    for ct in ladder.cube.groups["group_id"].tolist():
        one, other = result.table(ct, "spline"), direct.table(ct, "spline")
        shared = sorted(
            set(one["feature_id"].tolist()) & set(other["feature_id"].tolist())
        )
        a = dict(zip(one["feature_id"].tolist(), one["p_value"].tolist()))
        b = dict(zip(other["feature_id"].tolist(), other["p_value"].tolist()))
        calls_a = {
            g
            for g, q in zip(one["feature_id"].tolist(), one["adjusted_p_value"])
            if q < CALLED
        }
        calls_b = {
            g
            for g, q in zip(other["feature_id"].tolist(), other["adjusted_p_value"])
            if q < CALLED
        }
        correlation = np.corrcoef(
            [-np.log10(a[g]) for g in shared], [-np.log10(b[g]) for g in shared]
        )[0, 1]
        rows.append(
            (
                f"ladder, {ct}: the spline as contrasts of the dose-group fit against a fit "
                f"of the spline's columns (limma_voom), {len(shared)} genes",
                "CytePool",
                f"calls {len(calls_a)} and {len(calls_b)}, {len(calls_a & calls_b)} both; "
                f"correlation of -log10 p {correlation:.3f}; truth: "
                f"{truth_counts(ladder, ct, shared, calls_a)} against "
                f"{truth_counts(ladder, ct, shared, calls_b)}",
                "n/a",
                "info",
            )
        )
    left_out = [
        ct
        for ct in ladder.cube.groups["group_id"].tolist()
        if ct not in tools["muscat"]["ladder"]
    ]
    if left_out:
        rows.append(
            (
                f"ladder: cell types muscat's pbDS did not test, {', '.join(left_out)}",
                "muscat",
                "a dose group of the design keeps no sample there",
                "n/a",
                "info",
            )
        )
    for tool, by_type in (
        ("DRomics itemselect", tools["dromics"]["ladder"]),
        ("dreamlet", tools["dreamlet"]["ladder"]),
        ("muscat pbDS", tools["muscat"]["ladder"]),
    ):
        for ct, values in by_type.items():
            spline = result.table(ct, "spline")
            ours_called = {
                g
                for g, q in zip(
                    spline["feature_id"].tolist(), spline["adjusted_p_value"]
                )
                if q < CALLED
            }
            tested = spline["feature_id"].tolist()
            if tool == "DRomics itemselect":
                for method, indices in values.items():
                    picked = (
                        {features[i - 1] for i in vector(indices).astype(int)}
                        if np.size(indices)
                        else set()
                    )
                    rows.append(
                        (
                            f"ladder, {ct}: DRomics {method.replace('_', ' ')} trend test at 5% FDR",
                            "DRomics 2.6.3",
                            f"{len(picked)} called, {len(picked & ours_called)} of them by the "
                            f"spline F ({len(ours_called)}); truth: "
                            f"{truth_counts(ladder, ct, tested, picked)} of CytePool's tested genes",
                            "n/a",
                            "info",
                        )
                    )
                continue
            genes = features[vector(values["genes"]).astype(int) - 1].tolist()
            picked = {g for g, q in zip(genes, vector(values["adj"])) if q < CALLED}
            against = "spline F" if tool == "dreamlet" else "dose-group F"
            if tool == "muscat pbDS":
                groups = result.table(ct, "dose_groups")
                ours_called = {
                    g
                    for g, q in zip(
                        groups["feature_id"].tolist(), groups["adjusted_p_value"]
                    )
                    if q < CALLED
                }
            rows.append(
                (
                    f"ladder, {ct}: {tool}, {len(genes)} genes tested by its own filters",
                    tool.split()[0],
                    f"{len(picked)} called, {len(picked & ours_called)} of them by the "
                    f"{against} ({len(ours_called)}); truth: {truth_counts(ladder, ct, genes, picked)}",
                    "n/a",
                    "info",
                )
            )
    return rows


# The table.

TOPTABLE_RULE = (
    "topTable's p-values of dream's F-tests, every gene on the first gene's total df"
)
REFERENCES = {
    TOPTABLE_RULE: "pf(F, df1, df.total[1])",
    "samples, dose groups and genes": "R: n_cells, colSums, nonEstimable, filterByExpr",
    "numerator degrees of freedom": "classifyTestsF",
    "rms: partial sums of squares": "rms ols, anova",
}


def table_rows(checks: Checks) -> tuple[list[tuple[str, ...]], int]:
    rows, failures = [], 0
    for family, value in checks.largest.items():
        identical = family in (
            "samples, dose groups and genes",
            "numerator degrees of freedom",
        )
        if identical:
            tolerance = 0.0
        elif family.startswith("dream"):
            tolerance = DREAM_DF if "p-values" in family else CLOSE
        else:
            tolerance = EXACT
        passed = value <= tolerance
        failures += not passed
        reference = REFERENCES.get(family) or (
            "variancePartition dream, topTable"
            if family.startswith("dream")
            else "lm.wfit, eBayes"
            if "exact default" in family
            else "topTable(confint = TRUE)"
            if "curves" in family
            else "limma contrasts.fit, eBayes, topTable, classifyTestsF"
        )
        shown = "0 mismatches" if identical and passed else f"{value:.1e}"
        rows.append(
            (
                f"{family} ({checks.counts[family]} comparisons)",
                reference,
                shown,
                f"{tolerance:g}",
                "pass" if passed else "FAIL",
            )
        )
    if checks.basis:
        rows.append(
            (
                "limma-voom with limma's standard errors: limma's F of the spline and "
                "non-linearity tests on R's own ns() rows against the restricted cubic rows "
                f"CytePool tests, {len(checks.basis)} tests: limma's approximation depends "
                "on the basis, the exact default and limma-trend do not",
                "limma classifyTestsF",
                f"up to {max(checks.basis):.2f} relative",
                "n/a",
                "info",
            )
        )
    if checks.ratios:
        ratios = np.asarray(checks.ratios)
        rows.append(
            (
                f"dream's F-tests, {ratios.size} genes and tests: CytePool's p-values on "
                "each gene's own df over topTable's on the first gene's",
                "variancePartition topTable",
                f"{np.min(ratios):.2f} to {np.max(ratios):.2f}",
                "n/a",
                "deliberate difference",
            )
        )
    for line in checks.mismatches:
        print(f"differs: {line}")
    return rows, failures


# Fixtures: the fixture series and R's results for its two jobs.


def trimmed(job: dict[str, Any]) -> dict[str, Any]:
    """A job's results with the curves of the first genes only, which keep the
    fixture small."""
    found = json.loads(json.dumps(job))
    for values in found["cell_types"].values():
        curve = values.get("curve")
        if curve is not None:
            for key, rows in curve.items():
                if key != "at":
                    curve[key] = rows[:FIXTURE_CURVE_GENES]
    return found


def write_fixtures(found: list[Series], reference: dict) -> None:
    item = next(s for s in found if s.name == "fixture")
    cube = item.cube
    record = {
        "series": {
            "sample_id": cube.samples["sample_id"].tolist(),
            "dose": np.asarray(cube.samples["dose"], dtype=float).tolist(),
            "groups": cube.groups["group_id"].tolist(),
            "features": cube.features["feature_id"].tolist(),
            "counts": [
                cube.layer("counts", g).astype(int).tolist()
                for g in range(cube.n_groups)
            ],
            "n_cells": np.asarray(cube.n_cells).astype(int).tolist(),
        },
        "jobs": {job.name: trimmed(reference["jobs"][job.name]) for job in item.jobs},
        "settings": {
            job.name: {"analysis": job.analysis, "df": job.df, "tests": list(job.tests)}
            for job in item.jobs
        },
        "versions": reference["versions"],
    }
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = FIXTURE_DIRECTORY / "limma.json"
    path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
    print(f"wrote {path.name} to {FIXTURE_DIRECTORY}")


# The benchmark.


def benchmark() -> None:
    """dose_response against limma_voom with the dose-group F-test, 20,000 genes x 24
    profiles x 11 cell types, with the ordinary least squares fit of 20,000 genes x
    200 samples of linear_models.py's benchmark as the machine's baseline."""
    from cytepool.stats.linear_models import fit_linear_model

    doses = [0.0, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    cube, _ = simulate_dose_series(
        doses, replicates=3, cell_types=11, cells=300.0, n_genes=20_000, rng=20261216
    )
    groups_design = Design(cube.samples, "~ dose_group")
    group_columns = list(groups_design.columns[1:])
    other_terms = Design(cube.samples, "~ 1")

    def baseline() -> Any:
        return limma_voom(
            cube,
            groups_design,
            contrasts=group_columns,
            f_tests={"dose_groups": group_columns},
        )

    def tests() -> Any:
        return dose_response(cube, other_terms, dose=AXIS)

    rng = np.random.default_rng(1)
    y = rng.normal(size=(200, 20_000))
    x = np.column_stack([np.ones(200), rng.normal(size=(200, 3))])

    def ordinary() -> Any:
        return fit_linear_model(y, x)

    timings = {}
    for name, run in (
        ("baseline", baseline),
        ("dose_response", tests),
        ("ols", ordinary),
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
    print(
        f"| limma_voom with the dose-group F-test, 11 cell types | {timings['baseline']:.2f} |"
    )
    print(
        f"| dose_response, its four tests, 11 cell types | {timings['dose_response']:.2f} |"
    )
    print(
        f"| fit_linear_model, 20,000 genes x 200 samples (baseline) | {timings['ols']:.3f} |"
    )
    print(f"ratio {timings['dose_response'] / timings['baseline']:.2f} (target 1.5)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument("--benchmark", action="store_true", help="time dose_response")
    arguments = parser.parse_args()
    if arguments.benchmark:
        benchmark()
        return 0
    found = series()
    runs = {
        job.name: analyses(item, job)
        for item in found
        for job in item.jobs
        if job.analysis != "dream"
    }
    payload = r_input(found, runs)
    limma = run_r(LIMMA, payload, "RSCRIPT")
    dream = run_r(DREAM, payload, "DREAM_RSCRIPT")
    tools = run_r(TOOLS, payload, "RSCRIPT")
    versions = limma["versions"] | dream["versions"] | tools["versions"]
    print(
        "Reference: "
        + ", ".join(f"{name} {version}" for name, version in versions.items())
    )
    checks = Checks()
    for item in found:
        for job in item.jobs:
            if job.analysis == "dream":
                compare_dream(item, job, dream["dream"][job.name], checks)
            else:
                compare_limma(
                    item, job, runs[job.name], limma["jobs"][job.name], checks
                )
    rows, failures = table_rows(checks)
    rows += information(found, runs, {**tools, "dreamlet": dream["dreamlet"]})
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(
                found, {"jobs": limma["jobs"], "versions": limma["versions"]}
            )
    print(f"{len(rows)} checks, {failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
