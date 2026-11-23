"""Check dose_response on pathway scores, and its tests in one direction, against
limma, Hmisc, DoseFinding, multcomp and mvtnorm. Activities of samples take the same
path, which ``tests/test_dose_response_signatures.py`` checks by their identical
statistics.

Inputs come from fixed seeds (20261250 to 20261254). ``ladder``: the vehicle and 0.1,
0.3, 1, 3 and 10 ug/kg x 3, 1,000 genes of which 40% respond, a cell type of every
sample (``full``) and one that loses the second sample of 3 ug/kg to 5 cells
(``depleted``), scored by CytePool's ``gsva`` on 24 sets drawn from the truth: 30
Hill or exponential genes that fall, 30 that rise, 30 flat ones, 15 rising with 15
falling, and 20 random sets of 10 to 150 genes. ``sex``: the same doses x 4, two
females and two males per dose, under ``"~ sex"``. ``missing``: the ladder's scores
with a set missing in two samples, a set missing in the vehicle group and a set
missing everywhere. ``trial``: the vehicle and 0.3, 3 and 30 ug/kg x 8, a set of 8
falling and 22 flat genes and one of 8 rising and 22 flat genes, each scored alone,
and the ladder's falling set alone. CytePool's GSVA scores, which
``validation/pathway_scores.py`` matches to GSVA 2.0.0 within 1e-10, are the input of
both programs; R sees the scores, the samples, their doses and the settings.

References, run by R 4.4.3 (``RSCRIPT``) as a separate program:

- limma 3.62.1 and Hmisc 5.3-1: the design samples with a score, the dose groups that
  keep a sample and that ``nonEstimable`` does not name with their indicator placed
  last, R's own knots and spline columns (``ns()``, ``rcspline.eval``) and the rows of
  A = (Z'NZ)^-1 Z'N as contrasts, as ``dose_response_tests.py`` builds them; ``lmFit``,
  ``contrasts.fit``, ``eBayes`` (with the set sizes as ``trend``, and ``robust``),
  ``topTable(sort.by = "none")``, ``classifyTestsF`` with ``pf``, and
  ``topTable(confint = TRUE)`` for curves. For a score with missing values limma's
  ``contrasts.fit`` approximates the standard errors of contrasts that combine
  coefficients, so R computes that score's unscaled covariance from the samples it was
  observed in, gives ``eBayes`` its exact unscaled standard errors and computes its F
  by hand; limma's own values print as information. The one-sided linear trend is
  ``pt`` of limma's t on its total degrees of freedom.
- DoseFinding 1.4-2, multcomp 1.4-32 and mvtnorm 1.4-2, on each score tested alone:
  ``Mods`` of the recorded candidate shapes in both directions, ``optContr(w = n)``,
  ``MCTtest`` one- and two-sided; ``glht`` with Dunnett's and Williams' contrasts and
  ``alternative = "greater"``, ``"less"`` and ``"two.sided"`` on ``lm(score ~
  factor(dose))``; and the max-t p-values of the chosen contrasts from 10 million
  draws of the multivariate t with the contrasts' correlation, with their binomial
  standard error. ``MCTtest`` and ``glht`` compute theirs to 1e-3, and two runs of
  ``pmvt`` at an absolute error of 1e-10 on the same input differed by more than the
  error it reports, so ``pmvt``'s values print as information.

Tiers: exact (1e-10; statistics relative to max(|reference|, 1), p-values relative to
p max(1, |log p|)); close (1e-6) under ``robust = TRUE``, as
``empirical_bayes.py`` bounds limma's robust estimator; identical for samples, dose
groups, degrees of freedom and chosen contrasts; calibrated: CytePool's p-values within
5.5 combined standard errors of the draws', CytePool's reported error and the draws'
binomial standard error, and a one-sided p-value of 1 that the bound of its box
settles, whose error is that bound, no further below 1 minus the bound than 5.5 of
the draws' standard errors. The information rows give error rates of scores on null
dose series, which no reference has.

Requirements:
    Python 3.13 with numpy and numba, run from the repository root.
    R 4.4 with limma, Hmisc, DoseFinding, multcomp, mvtnorm and jsonlite (``RSCRIPT``,
    an Rscript executable; ``Rscript`` by default).

Usage:
    PYTHONPATH=. python validation/dose_response_signatures.py
    PYTHONPATH=. python validation/dose_response_signatures.py --write-fixtures
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from cytepool.candidate_shapes import CandidateShapes
from cytepool.cube import Cube
from cytepool.design import Design
from cytepool.dose_axes import DoseAxis
from cytepool.dose_responses import dose_response
from cytepool.gene_sets import GeneSets
from cytepool.pathway_scores import PathwayScores, gsva
from cytepool.simulated_dose_series import simulate_dose_series
from cytepool.table import Table

FIXTURE_DIRECTORY = (
    Path(__file__).resolve().parents[1] / "tests" / "data" / "dose_response_signatures"
)
SEED = 20261250
EXACT = 1e-10
CLOSE = 1e-6
CALIBRATED = 5.5
LADDER = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0]
TRIAL = [0.0, 0.3, 3.0, 30.0]
AXIS = DoseAxis("dose", unit="ug/kg", metric="administered")
TESTS = ("spline", "nonlinearity", "linear_trend", "dose_groups")
CONTRAST_TESTS = ("dunnett", "williams", "mcp")
# DoseFinding's names of CytePool's directions, which key the reference's models.
DOSEFINDING = {"up": "increasing", "down": "decreasing"}
CURVE_POINTS = 50
# The scores whose curves are compared with DoseFinding's fitMod.
CURVE_CASES = ("ladder falling",)
FIXTURE_CURVE_POINTS = 20
# The null series of the information rows: the designs of the calibration grid of
# dose_response_calibration.py, a cell type of 1,000 genes and 100 cells per profile.
NULL_DOSES = {
    4: [0.0, 0.1, 1.0, 10.0],
    9: [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0],
}
NULL_REPLICATES = (2, 3)
NULL_DATASETS = 1000
NULL_SETS = (3, 5, 10, 30, 50)

R_PROGRAM = r"""suppressPackageStartupMessages({
  library(limma); library(splines); library(Hmisc); library(DoseFinding)
  library(multcomp); library(mvtnorm); library(jsonlite)
})
args <- commandArgs(trailingOnly = TRUE)
input <- read_json(args[[1]], simplifyVector = TRUE)
out <- list(versions = as.list(c(R = as.character(getRversion()), sapply(
  c("limma", "Hmisc", "DoseFinding", "multcomp", "mvtnorm"),
  function(p) as.character(packageVersion(p))))))
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
pvalue <- function(f, df1, df2) pf(f, df1, df2, lower.tail = FALSE)
# limma on the scores of each job: the design samples with a score, the dose groups
# they keep that nonEstimable does not name with their indicator placed last, R's
# own knots and spline columns, and the rows of A = (Z'NZ)^-1 Z'N as contrasts.
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
    sc <- case$cell_types[[ct]]
    S <- matrix(sc$scores, nrow = length(sc$sample_id))
    at <- match(data$sample_id, sc$sample_id)
    has <- !is.na(at)
    has[has] <- rowSums(!is.na(S[at[has], , drop = FALSE])) > 0
    d <- data[has, , drop = FALSE]
    Y <- t(S[at[has], , drop = FALSE])
    treated <- levels[levels > 0 & levels %in% d$dose]
    others <- model.matrix(as.formula(case$formula), d)
    ind <- matrix(sapply(treated, function(k) as.numeric(d$dose == k)), nrow = nrow(d))
    colnames(ind) <- paste0("dose", seq_along(treated))
    estimable <- sapply(seq_along(treated), function(j) {
      X <- cbind(others, ind[, -j, drop = FALSE], ind[, j, drop = FALSE])
      !(colnames(ind)[j] %in% nonEstimable(X))
    })
    compared <- c(0, treated[estimable]); n <- sapply(compared, function(k) sum(d$dose == k))
    X <- cbind(others, ind[, estimable, drop = FALSE])
    k <- length(compared) - 1
    pad <- function(rows) rbind(matrix(0, ncol(others), nrow(rows)), t(rows))
    tr <- function(dd) if (own) as.numeric(dd > 0) else NULL
    rows <- list()
    A <- least_squares(cbind(1, tr(compared), ns_at(compared)), n)
    if (!is.null(A)) rows$spline <- A[-1, -1, drop = FALSE]
    Ar <- least_squares(cbind(1, tr(compared), rcs_at(compared)), n)
    if (!is.null(Ar) && length(knots) > 2) rows$nonlinearity <- Ar[-seq_len(2 + own), -1, drop = FALSE]
    A1 <- least_squares(cbind(1, tr(compared), line_at(compared)), n)
    if (!is.null(A1)) rows$linear_trend <- A1[nrow(A1), -1, drop = FALSE]
    rows$dose_groups <- diag(k)
    rows$each_dose <- diag(k)
    L <- do.call(cbind, lapply(rows, pad))
    owner <- rep(names(rows), sapply(rows, nrow))
    trend <- if (is.null(job$trend)) FALSE else if (identical(job$trend, "sizes")) sc$sizes else job$trend
    fit <- suppressWarnings(lmFit(Y, X))
    cf <- contrasts.fit(fit, L)
    eb <- eBayes(cf, trend = trend, robust = job$robust)
    # Each score with a missing value: its own unscaled covariance from the samples
    # it was observed in, which limma's contrasts.fit approximates; NA where those
    # samples cannot estimate every column.
    exact <- cf
    V <- lapply(seq_len(nrow(Y)), function(g) {
      seen <- !is.na(Y[g, ])
      if (!any(seen)) return(NULL)
      q <- qr(X[seen, , drop = FALSE])
      if (q$rank < ncol(X)) return(NULL)
      chol2inv(qr.R(q))
    })
    incomplete <- which(apply(is.na(Y), 1, any))
    unestimable <- which(sapply(V, is.null))
    for (g in incomplete) {
      if (is.null(V[[g]])) exact$stdev.unscaled[g, ] <- NA
      else exact$stdev.unscaled[g, ] <- sqrt(diag(crossprod(L, V[[g]] %*% L)))
    }
    exact$coefficients[unestimable, ] <- NA
    eb_exact <- eBayes(exact, trend = trend, robust = job$robust)
    s2 <- eb$s2.post; df2 <- eb$df.residual + eb$df.prior
    res <- list(samples = d$sample_id, doses = compared, n = n,
                df_prior = eb$df.prior, s2_post = s2, df_total = eb$df.total,
                incomplete = incomplete - 1, unestimable = unestimable - 1, tests = list())
    for (test in setdiff(names(rows), "each_dose")) {
      cols <- which(owner == test)
      if (test == "linear_trend") {
        top <- topTable(eb_exact, coef = cols, sort.by = "none", number = Inf)
        limma_top <- topTable(eb, coef = cols, sort.by = "none", number = Inf)
        r <- list(estimate = top$logFC, t = top$t, p = top$P.Value, adj = top$adj.P.Val,
                  B = top$B, average = top$AveExpr,
                  se = eb_exact$stdev.unscaled[, cols] * sqrt(s2),
                  limma_t = limma_top$t,
                  up = pt(top$t, eb_exact$df.total, lower.tail = FALSE),
                  down = pt(top$t, eb_exact$df.total))
        r$up_adj <- p.adjust(r$up, "BH")
        r$down_adj <- p.adjust(r$down, "BH")
      } else {
        fs <- classifyTestsF(eb[, cols], fstat.only = TRUE)
        df1 <- attr(fs, "df1"); limma_f <- as.numeric(fs)
        # The exact F of each score, from its own covariance of the contrasts.
        f <- limma_f
        for (g in incomplete) {
          if (is.null(V[[g]])) { f[g] <- NA; next }
          Lt <- L[, cols, drop = FALSE]; b <- fit$coefficients[g, ]
          est <- crossprod(Lt, b)
          f[g] <- drop(crossprod(est, solve(crossprod(Lt, V[[g]] %*% Lt), est))) / ncol(Lt) / s2[g]
        }
        p <- pvalue(f, df1, df2)
        r <- list(F = f, df1 = df1, df2 = df2, p = p, adj = p.adjust(p, "BH"), limma_F = limma_f)
      }
      res$tests[[test]] <- r
    }
    cols <- which(owner == "each_dose")
    tops <- lapply(cols, function(j) topTable(eb_exact, coef = j, sort.by = "none", number = Inf))
    column <- function(tables, name) do.call(cbind, lapply(tables, `[[`, name))
    res$each_dose <- list(
      estimate = column(tops, "logFC"), t = column(tops, "t"), p = column(tops, "P.Value"),
      adj = column(tops, "adj.P.Val"), B = column(tops, "B"), average = tops[[1]]$AveExpr,
      se = eb_exact$stdev.unscaled[, cols, drop = FALSE] * sqrt(s2))
    curve_at <- job$curves[[ct]]
    if (length(curve_at) && !is.null(A)) {
      C <- (cbind(1, tr(curve_at), ns_at(curve_at)) %*% A)[, -1, drop = FALSE]
      ebc <- eBayes(contrasts.fit(fit, pad(C)), trend = trend, robust = job$robust)
      ctops <- lapply(seq_len(nrow(C)), function(j)
        topTable(ebc, coef = j, sort.by = "none", number = Inf, confint = TRUE))
      res$curve <- list(at = curve_at, estimate = column(ctops, "logFC"),
                        lower = column(ctops, "CI.L"), upper = column(ctops, "CI.R"))
    }
    result$cell_types[[ct]] <- res
  }
  out$jobs[[name]] <- result
}
# One score tested alone, by DoseFinding and multcomp on the samples' scores: the
# contrasts, t-statistics and chosen contrast of MCTtest and glht. The max-t p-value of
# the chosen contrast comes from 10 million draws of the multivariate t with the
# contrasts' correlation, the distribution's definition, with its binomial standard
# error; pmvt's value at a tight tolerance prints beside it, as information, since its
# Genz-Bretz integration moved by more than its error attribute between runs.
set.seed(20261250)
tight <- GenzBretz(maxpts = 2e6, abseps = 1e-10, releps = 0)
DRAWS <- 1e7
pmvt_p <- function(t, R, df, side) {
  m <- nrow(R)
  q <- switch(side,
    greater = pmvt(lower = rep(-Inf, m), upper = rep(t, m), corr = R, df = df, algorithm = tight),
    less = pmvt(lower = rep(t, m), upper = rep(Inf, m), corr = R, df = df, algorithm = tight),
    two.sided = pmvt(lower = rep(-abs(t), m), upper = rep(abs(t), m), corr = R, df = df, algorithm = tight))
  1 - as.numeric(q)
}
# The largest, smallest and largest absolute value of each of DRAWS multivariate t
# vectors with correlation R, in chunks.
extremes <- function(R, df) {
  e <- eigen(R, symmetric = TRUE)
  root <- e$vectors %*% diag(sqrt(pmax(e$values, 0)), nrow(R))
  chunk <- 1e6; out <- list(high = numeric(0), low = numeric(0), wide = numeric(0))
  for (i in seq_len(DRAWS / chunk)) {
    z <- matrix(rnorm(chunk * nrow(R)), chunk) %*% t(root)
    tt <- z / sqrt(rchisq(chunk, df) / df)
    columns <- lapply(seq_len(ncol(tt)), function(j) tt[, j])
    out$high <- c(out$high, do.call(pmax, columns))
    out$low <- c(out$low, do.call(pmin, columns))
    out$wide <- c(out$wide, do.call(pmax, lapply(columns, abs)))
  }
  out
}
max_t_p <- function(t, draws, R, df, side) {
  hits <- switch(side, greater = draws$high >= t, less = draws$low <= t,
                 two.sided = draws$wide >= abs(t))
  p <- mean(hits)
  list(p = p, se = sqrt(p * (1 - p) / DRAWS), pmvt = pmvt_p(t, R, df, side))
}
for (name in names(input$alone)) {
  case <- input$alone[[name]]
  dat <- data.frame(dose = case$dose, resp = case$score)
  doses <- sort(unique(dat$dose)); n <- as.numeric(table(factor(dat$dose, levels = doses)))
  shapes <- case$shapes
  args_of <- function(direction) {
    a <- list(doses = doses, direction = direction)
    if (length(shapes$emax)) a$emax <- shapes$emax
    if (length(shapes$sigmoid_emax)) a$sigEmax <- as.numeric(t(as.matrix(shapes$sigmoid_emax)))
    if (length(shapes$exponential)) a$exponential <- shapes$exponential
    if (length(shapes$beta)) {
      a$betaMod <- as.numeric(t(as.matrix(shapes$beta)))
      a$addArgs <- list(scal = shapes$beta_scale)
    }
    a
  }
  res <- list(n = n, df = nrow(dat) - length(doses))
  for (direction in c("increasing", "decreasing")) {
    mods <- do.call(Mods, args_of(direction))
    C <- optContr(mods, w = n)$contMat
    mct <- MCTtest(dose, resp, dat, models = mods, alternative = "one.sided")
    two <- MCTtest(dose, resp, dat, models = mods, alternative = "two.sided")
    res$mcp[[direction]] <- list(contrasts = C, t = as.numeric(mct$tStat), names = colnames(C),
                                 t_two_sided = as.numeric(two$tStat),
                                 p_mcttest = as.numeric(attr(mct$tStat, "pVal")))
  }
  C <- res$mcp$increasing$contrasts; R <- cov2cor(crossprod(C, C / n))
  draws <- extremes(R, res$df)
  p_of <- function(t, side) max_t_p(t, draws, R, res$df, side)
  inc <- res$mcp$increasing$t
  for (direction in c("increasing", "decreasing")) {
    t <- res$mcp[[direction]]$t; chosen <- which.max(t)
    # DoseFinding's decreasing models give t of the opposite sign, so their largest
    # t is the smallest of the increasing models'.
    side <- if (direction == "increasing") "greater" else "less"
    res$mcp[[direction]] <- c(res$mcp[[direction]], list(chosen = chosen - 1),
                              p_of(inc[chosen], side))
  }
  res$mcp$two_sided <- c(list(chosen = which.max(abs(inc)) - 1),
                         p_of(inc[which.max(abs(inc))], "two.sided"))
  res$mcp$rank <- qr(R)$rank
  dat$f <- factor(dat$dose)
  model <- lm(resp ~ f, data = dat)
  for (kind in c("Dunnett", "Williams")) {
    for (side in c("greater", "less", "two.sided")) {
      g <- glht(model, linfct = mcp(f = kind), alternative = side)
      s <- summary(g, test = adjusted("none"))
      t <- as.numeric(s$test$tstat)
      R <- cov2cor(g$linfct %*% vcov(model) %*% t(g$linfct))
      if (side == "greater") draws <- extremes(R, res$df)
      chosen <- switch(side, greater = which.max(t), less = which.min(t), two.sided = which.max(abs(t)))
      res[[tolower(kind)]][[side]] <- c(list(t = t, names = rownames(g$linfct), chosen = chosen - 1),
                                        max_t_p(t[chosen], draws, R, res$df, side))
    }
  }
  out$alone[[name]] <- res
}
# Curves of one score: DoseFinding's fitMod of the samples (type "normal") and of the
# dose-group means with their covariance, the residual variance times diag(1 / n)
# (type "general"), with the bounds of the curves it is compared with: linear, and
# sigEmax with PR 10's bounds of the Hill model (ED50 to 5 D, h from 1 to 18) and of
# the log-logistic model (ED50 to 5 D, h from 0.1 to 18).
for (name in names(input$curves)) {
  case <- input$curves[[name]]
  dat <- data.frame(dose = case$dose, resp = case$score)
  doses <- sort(unique(dat$dose)); n <- as.numeric(table(factor(dat$dose, levels = doses)))
  top <- max(doses)
  mu <- as.numeric(tapply(dat$resp, factor(dat$dose, levels = doses), mean))
  s2 <- sum((dat$resp - mu[match(dat$dose, doses)])^2) / (nrow(dat) - length(doses))
  S <- s2 * diag(1 / n)
  res <- list(doses = doses, means = mu, s2 = s2)
  bounds <- list(hill = rbind(c(1e-6 * top, 5 * top), c(1, 18)),
                 ll4 = rbind(c(1e-6 * top, 5 * top), c(0.1, 18)))
  fit_of <- function(f, general) list(coef = as.numeric(coef(f)),
                                      objective = if (general) f$gRSS else f$RSS)
  res$constant$linear <- fit_of(fitMod(dose, resp, dat, model = "linear"), FALSE)
  res$weighted$linear <- fit_of(fitMod(doses, mu, S = S, model = "linear", type = "general"), TRUE)
  for (model in names(bounds)) {
    res$constant[[model]] <- fit_of(fitMod(dose, resp, dat, model = "sigEmax", bnds = bounds[[model]]), FALSE)
    res$weighted[[model]] <- fit_of(fitMod(doses, mu, S = S, model = "sigEmax", type = "general",
                                           bnds = bounds[[model]]), TRUE)
  }
  out$curves[[name]] <- res
}
write_json(out, args[[2]], digits = NA, matrix = "rowmajor", auto_unbox = TRUE, na = "string")
"""


# ---------------------------------------------------------------------------------------
# The inputs
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
    """R's numbers as float64, with its NA, NaN and Inf, which it writes as text."""
    return np.array(_floats(value), dtype=np.float64)


def _floats(value: Any) -> Any:
    if isinstance(value, list):
        return [_floats(item) for item in value]
    if value is None or value in ("NA", "NaN"):
        return math.nan
    if isinstance(value, str):
        return float(value.replace("Inf", "inf"))
    return value


def depleted(cube: Cube, group: str, sample: str, cells: int) -> Cube:
    """The cube with one sample of one group left with ``cells`` cells."""
    n_cells = np.array(cube.n_cells, copy=True)
    row = cube.samples["sample_id"].tolist().index(sample)
    n_cells[row, cube.groups["group_id"].tolist().index(group)] = cells
    layers = np.stack([cube.layer("counts", g) for g in range(cube.n_groups)])
    return Cube(
        {"counts": layers},
        samples=cube.samples,
        groups=cube.groups,
        features=cube.features,
        n_cells=n_cells,
    )


def signature_sets(truth: Any, group: str, rng: np.random.Generator) -> dict:
    """Sets drawn from the truth: 30 Hill or exponential genes that fall, 30 that
    rise, 30 flat ones, 15 rising with 15 falling, and 20 random sets of 10 to 150
    genes."""
    table = truth.table(group)
    ids = np.array(table["feature_id"].tolist())
    shape = np.array(table["shape"].tolist())
    change = np.asarray(table["max_log2_change"], dtype=float)
    smooth = np.isin(shape, ["hill", "exponential"])
    falling = ids[smooth & (change < 0)]
    rising = ids[smooth & (change > 0)]
    flat = ids[change == 0]
    sets = {
        "falling": falling[:30].tolist(),
        "rising": rising[:30].tolist(),
        "flat": flat[:30].tolist(),
        "mixed": [*rising[30:45].tolist(), *falling[30:45].tolist()],
    }
    for i in range(20):
        size = int(rng.integers(10, 151))
        sets[f"random_{i + 1:02d}"] = rng.choice(ids, size, replace=False).tolist()
    return sets


def scores_record(scores: PathwayScores) -> dict:
    return {
        "scores": scores.scores.tolist(),
        "sample_id": scores.profiles["sample_id"].tolist(),
        "sizes": scores.n_genes.tolist(),
        "names": list(scores.gene_sets),
    }


@dataclass
class Inputs:
    """The series, their scores and the designs, from fixed seeds."""

    cube: Cube
    scores: dict[str, PathwayScores]  # the ladder's, by cell type
    missing: PathwayScores
    constant: PathwayScores
    sex_scores: PathwayScores
    sex_samples: dict[str, list]
    alone: dict[str, PathwayScores] = field(default_factory=dict)


def inputs() -> Inputs:
    rng = np.random.default_rng(SEED + 3)
    cube, truth = simulate_dose_series(
        LADDER,
        replicates=3,
        cell_types=["full", "depleted"],
        cells=300.0,
        n_genes=1000,
        responsive=0.4,
        rng=SEED,
    )
    # The second sample of 3 ug/kg keeps 5 cells in the depleted cell type, fewer
    # than gsva's min_cells.
    cube = depleted(cube, "depleted", "d4_r2", 5)
    sets = signature_sets(truth, "full", rng)
    scores = {group: gsva(cube, sets, group=group) for group in ("full", "depleted")}
    full = scores["full"]
    values = np.array(full.scores, copy=True)
    names = list(full.gene_sets)
    doses = np.asarray(full.profiles["dose"], dtype=float)
    # A set missing in two samples, a set missing in the vehicle group, and a set that
    # no sample has.
    values[[1, 7], names.index("rising")] = np.nan
    values[doses == 0.0, names.index("flat")] = np.nan
    values[:, names.index("random_05")] = np.nan
    missing = replace(full, scores=values)
    constant = replace(
        full,
        scores=np.full((values.shape[0], 1), 0.3),
        gene_sets=GeneSets({"constant": list(full.gene_sets["flat"])}),
    )
    sex_cube, sex_truth = simulate_dose_series(
        LADDER,
        replicates=4,
        cell_types=["pDC"],
        cells=300.0,
        n_genes=1000,
        responsive=0.4,
        rng=SEED + 1,
    )
    sex = [
        "F" if s.endswith(("_r1", "_r2")) else "M"
        for s in sex_cube.samples["sample_id"].tolist()
    ]
    sex_scores = gsva(sex_cube, signature_sets(sex_truth, "pDC", rng), group="pDC")
    sex_samples = {
        "sample_id": sex_cube.samples["sample_id"].tolist(),
        "dose": np.asarray(sex_cube.samples["dose"], dtype=float).tolist(),
        "sex": sex,
    }
    trial_cube, trial_truth = simulate_dose_series(
        TRIAL,
        replicates=8,
        cell_types=1,
        cells=300.0,
        n_genes=1000,
        responsive=0.4,
        rng=SEED + 2,
    )
    trial_sets = signature_sets(trial_truth, "cell_type_1", rng)
    found = Inputs(cube, scores, missing, constant, sex_scores, sex_samples)
    # Signatures of 8 responsive and 22 flat genes, each scored alone, and the
    # ladder's falling set alone.
    for name in ("falling", "rising"):
        signature = trial_sets[name][:8] + trial_sets["flat"][:22]
        found.alone[f"trial {name}"] = gsva(trial_cube, {name: signature})
    found.alone["ladder falling"] = gsva(
        cube, {"falling": sets["falling"]}, group="full"
    )
    return found


@dataclass(frozen=True)
class Job:
    """A dose_response run on scores, and R's reference for it."""

    name: str
    case: str
    groups: tuple[str, ...]
    trend: bool = False  # the set sizes as the trend's covariate
    robust: bool = False
    curves: bool = False


JOBS = (
    Job("ladder", "ladder", ("full", "depleted"), curves=True),
    Job("ladder, trend", "ladder", ("full", "depleted"), trend=True),
    Job("ladder, robust", "ladder", ("full", "depleted"), robust=True),
    Job("sex", "sex", ("pDC",)),
    Job("missing", "missing", ("full",)),
    Job("missing, trend", "missing", ("full",), trend=True),
    Job("constant", "constant", ("full",)),
)


def curve_at() -> list[float]:
    return np.geomspace(0.1, 10.0, CURVE_POINTS).tolist()


def case_scores(found: Inputs, case: str, group: str) -> PathwayScores:
    if case == "ladder":
        return found.scores[group]
    if case == "sex":
        return found.sex_scores
    return found.missing if case == "missing" else found.constant


def r_input(found: Inputs) -> dict[str, Any]:
    samples = {
        "sample_id": found.cube.samples["sample_id"].tolist(),
        "dose": np.asarray(found.cube.samples["dose"], dtype=float).tolist(),
    }
    axis = {"vehicle": AXIS.vehicle, "scale": AXIS.scale, "offset": AXIS.offset}
    cases = {
        case: {
            "samples": found.sex_samples if case == "sex" else samples,
            "axis": axis,
            "formula": "~ sex" if case == "sex" else "~ 1",
            "cell_types": {
                group: scores_record(case_scores(found, case, group))
                for group in next(job.groups for job in JOBS if job.case == case)
            },
        }
        for case in ("ladder", "sex", "missing", "constant")
    }
    jobs = {
        job.name: {
            "case": job.case,
            "df": 2,
            "robust": job.robust,
            **({"trend": "sizes"} if job.trend else {}),
            "curves": {"full": curve_at()} if job.curves else {},
        }
        for job in JOBS
    }
    alone = {
        name: {
            "dose": np.asarray(one.profiles["dose"], dtype=float).tolist(),
            "score": one.scores[:, 0].tolist(),
            "shapes": shapes_of(one).to_dict(),
        }
        for name, one in found.alone.items()
    }
    curves = {
        name: {key: alone[name][key] for key in ("dose", "score")}
        for name in CURVE_CASES
    }
    return {"cases": cases, "jobs": jobs, "alone": alone, "curves": curves}


def shapes_of(one: PathwayScores) -> CandidateShapes:
    """The candidate shapes dose_response spreads over the doses of the scores."""
    return CandidateShapes.spread(AXIS, one.profiles)


def run_r(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the R program on a JSON payload; returns what it writes."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "program.R").write_text(R_PROGRAM)
        (root / "input.json").write_text(json.dumps(plain(payload)))
        command = [
            os.environ.get("RSCRIPT", "Rscript"),
            "--vanilla",
            str(root / "program.R"),
            str(root / "input.json"),
            str(root / "output.json"),
        ]
        subprocess.run(command, check=True)
        result: dict[str, Any] = json.loads((root / "output.json").read_text())
    return result


# ---------------------------------------------------------------------------------------
# CytePool
# ---------------------------------------------------------------------------------------


def design_of(found: Inputs, case: str) -> Design:
    if case == "sex":
        return Design(found.sex_samples, "~ sex")
    return Design(found.cube.samples, "~ 1")


def ours(found: Inputs, job: Job, group: str, **options: Any) -> Any:
    scores = case_scores(found, job.case, group)
    moderation: dict[str, Any] = {"robust": True} if job.robust else {}
    if job.trend:
        moderation["trend"] = scores.n_genes
    return dose_response(
        scores,
        design_of(found, job.case),
        dose=AXIS,
        spline_df=2,
        **({"tests": TESTS} | options | moderation),
    )


# ---------------------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------------------


def scaled(ours: Any, reference: Any) -> float:
    """The largest difference over max(|reference|, 1); infinite when the shapes or
    the NaNs differ."""
    a, b = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return math.inf
    present = ~np.isnan(b)
    if not present.any():
        return 0.0
    with np.errstate(invalid="ignore"):
        same = a[present] == b[present]  # infinities alike
        difference = np.where(same, 0.0, np.abs(a[present] - b[present]))
    return float(np.max(difference / np.maximum(np.abs(b[present]), 1.0)))


def p_scaled(ours: Any, reference: Any) -> float:
    """The largest difference of p-values over p max(1, |log p|); infinite when the
    shapes or the NaNs differ."""
    a, b = np.asarray(ours, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or not np.array_equal(np.isnan(a), np.isnan(b)):
        return math.inf
    present = ~np.isnan(b)
    a, b = a[present], b[present]
    if not a.size:
        return 0.0
    both_zero = (a == 0.0) & (b == 0.0)
    scale = np.maximum(b, 1e-300) * np.maximum(
        1.0, np.abs(np.log(np.maximum(b, 1e-300)))
    )
    return float(np.max(np.where(both_zero, 0.0, np.abs(a - b) / scale)))


@dataclass
class Checks:
    """The largest difference of each family of checks, the cases it covers, its
    tolerance and its reference."""

    largest: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    tolerance: dict[str, float] = field(default_factory=dict)
    reference: dict[str, str] = field(default_factory=dict)
    mismatches: list[str] = field(default_factory=list)
    rows: list[tuple[str, ...]] = field(default_factory=list)

    def record(
        self, family: str, value: float, tolerance: float, reference: str
    ) -> None:
        # A NaN difference fails: max() would keep whichever value comes first.
        value = math.inf if math.isnan(value) else value
        self.largest[family] = max(self.largest.get(family, 0.0), value)
        self.counts[family] = self.counts.get(family, 0) + 1
        self.tolerance[family] = tolerance
        self.reference[family] = reference
        if value > tolerance:
            self.mismatches.append(f"{family}: {value:.3g}")

    def same(self, family: str, label: str, equal: bool, reference: str) -> None:
        self.record(family, 0.0 if equal else math.inf, 0.0, reference)
        if not equal:
            self.mismatches.append(f"{family}: {label}")

    def information(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "information"))

    def deliberate(self, check: str, reference: str, value: str) -> None:
        self.rows.append((check, reference, value, "n/a", "deliberate difference"))

    def table(self) -> tuple[list[tuple[str, ...]], int]:
        rows, failures = [], 0
        for family, value in self.largest.items():
            tolerance = self.tolerance[family]
            passed = value <= tolerance
            failures += not passed
            if family.endswith("(calibrated)"):
                shown, bound = (
                    f"{value:.2f} standard errors",
                    f"{CALIBRATED:g} standard errors",
                )
            elif tolerance == 0.0:
                shown, bound = ("identical" if passed else "differ"), "identical"
            else:
                shown, bound = f"{value:.1e}", f"{tolerance:g}"
            rows.append(
                (
                    f"{family} ({self.counts[family]} comparisons)",
                    self.reference[family],
                    shown,
                    bound,
                    "pass" if passed else "FAIL",
                )
            )
        return rows + self.rows, failures


LIMMA = "limma lmFit, contrasts.fit, eBayes, topTable, classifyTestsF"
DRAWN = "10 million draws of the multivariate t in R"
EXACT_R = "limma eBayes on each score's exact covariance, F by hand in R"


def compare_job(
    checks: Checks, job: Job, results: dict[str, Any], reference: dict[str, Any]
) -> None:
    """Every table of a job's results of scores against R's."""
    robust = job.robust
    tolerance = CLOSE if robust else EXACT
    tag = "robust" if robust else "exact"
    for group, result in results.items():
        theirs = reference["cell_types"][group]
        entry = result.provenance["groups"][0]
        checks.same(
            "the check of scores: samples fitted, dose groups compared and their samples",
            f"{job.name}, {group}",
            entry["fit"]["samples"] == list(theirs["samples"])
            and entry["doses"] == numbers(theirs["doses"]).tolist()
            and entry["samples"] == numbers(theirs["n"]).tolist(),
            "R: the design samples with a score, nonEstimable",
        )
        prior = numbers(theirs["df_prior"])
        checks.record(
            f"prior degrees of freedom ({tag})",
            scaled(
                result.summary["df_prior"][0],
                np.median(prior) if prior.ndim else prior,
            ),
            tolerance,
            "limma eBayes",
        )
        unestimable = numbers(theirs.get("unestimable", [])).astype(int)
        for test in TESTS:
            values = theirs["tests"][test]
            table = result.table(group, test)
            if test == "linear_trend":
                family = (
                    f"linear trend: estimate, standard error, average, t, B ({tag})"
                )
                for ours_column, column, scale in (
                    ("estimate", "estimate", scaled),
                    ("standard_error", "se", scaled),
                    ("average", "average", scaled),
                    ("t", "t", scaled),
                    ("log_odds", "B", scaled),
                ):
                    checks.record(
                        family,
                        scale(table[ours_column], numbers(values[column])),
                        tolerance,
                        EXACT_R if job.case == "missing" else LIMMA,
                    )
                family = f"linear trend: p-values and adjusted p-values ({tag})"
            else:
                family = f"F-tests: F, df, p-values and adjusted p-values ({tag})"
                checks.record(
                    family,
                    scaled(table["f_statistic"], numbers(values["F"])),
                    tolerance,
                    EXACT_R if job.case == "missing" else LIMMA,
                )
                checks.record(
                    family,
                    scaled(
                        table["df"],
                        np.broadcast_to(numbers(values["df2"]), len(table)),
                    ),
                    tolerance,
                    LIMMA,
                )
                checks.same(
                    "numerator degrees of freedom",
                    f"{job.name}, {group}, {test}",
                    entry["f_test_df1"][test] == int(values["df1"]),
                    "classifyTestsF",
                )
            checks.record(
                family,
                p_scaled(table["p_value"], numbers(values["p"])),
                tolerance,
                LIMMA,
            )
            for column in ("adjusted_p_value", "adjusted_p_value_across_groups"):
                checks.record(
                    family,
                    p_scaled(table[column], numbers(values["adj"])),
                    tolerance,
                    "p.adjust",
                )
        each = result.table(group, "each_dose")
        k = len(theirs["doses"]) - 1

        def by_dose(column: str) -> np.ndarray:
            return np.asarray(each[column], dtype=float).reshape(k, -1).T

        family = (
            f"each dose: estimate, standard error, average, t, p, adjusted p, B ({tag})"
        )
        for ours_column, column in (
            ("estimate", "estimate"),
            ("standard_error", "se"),
            ("t", "t"),
            ("log_odds", "B"),
        ):
            checks.record(
                family,
                scaled(by_dose(ours_column), numbers(theirs["each_dose"][column])),
                tolerance,
                LIMMA,
            )
        checks.record(
            family,
            scaled(by_dose("average")[:, 0], numbers(theirs["each_dose"]["average"])),
            tolerance,
            LIMMA,
        )
        for ours_column, column in (("p_value", "p"), ("adjusted_p_value", "adj")):
            checks.record(
                family,
                p_scaled(by_dose(ours_column), numbers(theirs["each_dose"][column])),
                tolerance,
                LIMMA,
            )
        if unestimable.size:
            # Scores whose samples cannot estimate every dose group: NaN in every
            # test, as limma's coefficients are NA where it drops a column.
            ids = result.table(group, "spline")["gene_set"].tolist()
            nan = all(
                np.isnan(
                    np.asarray(result.table(group, test)["p_value"], dtype=float)[row]
                )
                for test in TESTS
                for row in unestimable
            )
            checks.same(
                "scores that cannot estimate every dose group: no test, and NA in R",
                f"{job.name}, {[ids[row] for row in unestimable]}",
                nan,
                "R: lm.fit of the samples each score has",
            )
        if job.curves and group == "full":
            curve = result.curve(group, at=np.asarray(theirs["curve"]["at"]))
            n = len(theirs["curve"]["at"])
            for column, key in (
                ("estimate", "estimate"),
                ("ci_lower", "lower"),
                ("ci_upper", "upper"),
            ):
                checks.record(
                    "curves and their 95% bands, 50 doses",
                    scaled(
                        np.asarray(curve[column], dtype=float).reshape(-1, n),
                        numbers(theirs["curve"][key]),
                    ),
                    EXACT,
                    "topTable(confint = TRUE)",
                )


def compare_limma_approximation(checks: Checks, reference: dict[str, Any]) -> None:
    """limma's own t and F of the scores with missing values, which its contrasts.fit
    approximates, against the exact ones that CytePool and the reference give."""
    largest_t, largest_f = 0.0, 0.0
    for name in ("missing", "missing, trend"):
        values = reference["jobs"][name]["cell_types"]["full"]
        trend = values["tests"]["linear_trend"]
        exact_t, limma_t = numbers(trend["t"]), numbers(trend["limma_t"])
        keep = ~np.isnan(exact_t)
        largest_t = max(
            largest_t,
            float(
                np.max(np.abs(limma_t[keep] - exact_t[keep]) / np.abs(exact_t[keep]))
            ),
        )
        for test in ("spline", "nonlinearity", "dose_groups"):
            f, limma_f = (
                numbers(values["tests"][test]["F"]),
                numbers(values["tests"][test]["limma_F"]),
            )
            keep = ~np.isnan(f)
            largest_f = max(
                largest_f, float(np.max(np.abs(limma_f[keep] - f[keep]) / f[keep]))
            )
    checks.information(
        "Scores with missing values: limma's own t of the linear trend and F of the "
        "spline, non-linearity and dose groups, from contrasts.fit's approximate "
        "standard errors, against the exact ones",
        "limma contrasts.fit, classifyTestsF",
        f"up to {largest_t:.2f} (t) and {largest_f:.2f} (F) relative",
    )


def compare_one_sided_trend(
    checks: Checks, found: Inputs, reference: dict[str, Any]
) -> None:
    """The one-sided linear trend in both directions against pt of limma's t."""
    for job in (JOBS[0], JOBS[1], JOBS[4]):
        for group in job.groups:
            values = reference["jobs"][job.name]["cell_types"][group]["tests"][
                "linear_trend"
            ]
            for direction in ("up", "down"):
                result = ours(
                    found, job, group, tests="linear_trend", direction=direction
                )
                table = result.table(group, "linear_trend")
                checks.record(
                    "one-sided linear trend: p-values and adjusted p-values, both directions",
                    max(
                        p_scaled(table["p_value"], numbers(values[direction])),
                        p_scaled(
                            table["adjusted_p_value"],
                            numbers(values[f"{direction}_adj"]),
                        ),
                    ),
                    EXACT,
                    "pt(t, df.total), p.adjust",
                )


def compare_alone(
    checks: Checks, found: Inputs, reference: dict[str, Any]
) -> list[dict]:
    """Each score tested alone: the MCP step's contrasts, t and chosen shape against
    DoseFinding, Dunnett's and the Williams-type tests against multcomp, and their
    p-values against draws of the multivariate t, two-sided and in both
    directions."""
    rows = []
    pmvt_rows: list[tuple[float, float, float]] = []
    for name, one in found.alone.items():
        theirs = reference["alone"][name]
        design = Design(one.profiles, "~ 1")
        df = float(theirs["df"])
        n_shapes = len(numbers(theirs["mcp"]["increasing"]["t"]))
        for direction in (None, "up", "down"):
            result = dose_response(
                one,
                design,
                dose=AXIS,
                tests=CONTRAST_TESTS,
                direction=direction,
                rng=SEED,
            )
            group = result.groups[0]
            entry = result.provenance["groups"][0]
            record = {"name": name, "direction": direction}
            # The MCP step.
            mcp = result.table(group, "mcp")
            weights = entry["contrast_tests"]["mcp"]["contrasts"]
            ours_contrasts = np.array([weights[shape] for shape in weights])
            increasing = theirs["mcp"]["increasing"]
            contrasts = numbers(increasing["contrasts"]).reshape(-1, n_shapes)[1:].T
            contrasts /= np.linalg.norm(contrasts, axis=1, keepdims=True)
            checks.record(
                "MCP step on one score: contrasts on the treated doses",
                scaled(ours_contrasts, contrasts),
                EXACT,
                "DoseFinding optContr(w = n), both directions",
            )
            if direction == "down":
                decreasing = theirs["mcp"]["decreasing"]
                checks.record(
                    "MCP step on one score: contrasts on the treated doses",
                    scaled(
                        -ours_contrasts,
                        _unit(
                            numbers(decreasing["contrasts"]).reshape(-1, n_shapes)[1:].T
                        ),
                    ),
                    EXACT,
                    "DoseFinding optContr(w = n), both directions",
                )
            names = list(weights)
            chosen = names.index(str(mcp["contrast"][0]))
            if direction is None:
                t_all = numbers(increasing["t"])
                found_p = theirs["mcp"]["two_sided"]
                expected, sign = int(found_p["chosen"]), 1.0
            else:
                found_p = theirs["mcp"][DOSEFINDING[direction]]
                t_all = numbers(found_p["t"])
                expected = int(found_p["chosen"])
                sign = 1.0 if direction == "up" else -1.0
            checks.same(
                "MCP step on one score: the shape of the most extreme t",
                f"{name}, {direction}",
                chosen == expected,
                "DoseFinding MCTtest",
            )
            checks.record(
                "MCP step on one score: t (MCTtest's sign flipped for decreasing)",
                scaled(sign * float(mcp["t"][0]), t_all[expected]),
                EXACT,
                "DoseFinding MCTtest",
            )
            checks.same(
                "One score: degrees of freedom of the multivariate t",
                f"{name}",
                float(mcp["df"][0]) == df,
                "n - K",
            )
            calibrated(
                checks,
                "MCP step on one score: p-values, two-sided and in both directions, "
                "5 shapes on 3 and on 5 treated doses (calibrated)",
                mcp,
                found_p,
            )
            pmvt_rows.append(
                (float(found_p["pmvt"]), float(found_p["p"]), float(found_p["se"]))
            )
            record["mcp"] = {
                "t": float(mcp["t"][0]),
                "contrast": str(mcp["contrast"][0]),
                "p_value": float(mcp["p_value"][0]),
                "p_value_error": float(mcp["p_value_error"][0]),
            }
            if direction is None:
                two = numbers(theirs["mcp"]["increasing"]["t_two_sided"])
                record["mct_two_sided"] = bool(
                    np.allclose(two, np.abs(t_all), rtol=EXACT, atol=0.0)
                )
            for test in ("dunnett", "williams"):
                table = result.table(group, test)
                side = {None: "two.sided", "up": "greater", "down": "less"}[direction]
                values = theirs[test][side]
                t_all = numbers(values["t"])
                expected = int(values["chosen"])
                ours_names = list(entry["contrast_tests"][test]["contrasts"])
                checks.same(
                    "Dunnett's and Williams-type tests on one score: the contrast of the most extreme t",
                    f"{name}, {test}, {direction}",
                    ours_names.index(str(table["contrast"][0])) == expected,
                    "multcomp glht",
                )
                checks.record(
                    "Dunnett's and Williams-type tests on one score: t",
                    scaled(float(table["t"][0]), t_all[expected]),
                    EXACT,
                    "multcomp glht on lm(score ~ factor(dose))",
                )
                calibrated(
                    checks,
                    "Dunnett's and Williams-type tests on one score: p-values, "
                    "two-sided and in both directions (calibrated)",
                    table,
                    values,
                )
                pmvt_rows.append(
                    (float(values["pmvt"]), float(values["p"]), float(values["se"]))
                )
                record[test] = {
                    "t": float(table["t"][0]),
                    "contrast": str(table["contrast"][0]),
                    "p_value": float(table["p_value"][0]),
                    "p_value_error": float(table["p_value_error"][0]),
                }
            rows.append(record)
    apart = max(abs(pmvt - drawn) / se for pmvt, drawn, se in pmvt_rows)
    checks.information(
        f"The {len(pmvt_rows)} p-values of the tests of one score from pmvt's Genz-Bretz "
        "integration at an absolute error of 1e-10, against the draws",
        "mvtnorm pmvt",
        f"up to {apart:.1f} standard errors of the draws apart",
    )
    return rows


def _unit(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


SETTLED = (
    "Tests of one score: one-sided p-values of 1 that the bound of their box settles, "
    "the draws' p-value against 1 minus the bound (calibrated)"
)


def calibrated(
    checks: Checks, family: str, table: Table, reference: dict[str, Any]
) -> None:
    """A max-t p-value of one score against the draws': within its combined standard
    errors, or, for a p-value of 1 that a bound settled, whose error is that bound,
    how far the draws' p-value lies below 1 minus the bound, in its standard errors
    (0 within it)."""
    ours, error = float(table["p_value"][0]), float(table["p_value_error"][0])
    drawn, se = float(reference["p"]), float(reference["se"])
    if ours == 1.0:
        checks.record(SETTLED, max(0.0, 1.0 - error - drawn) / se, CALIBRATED, DRAWN)
        return
    checks.record(family, abs(ours - drawn) / math.hypot(error, se), CALIBRATED, DRAWN)


# The curves compared with fitMod: (set, model), and how each maps to DoseFinding's
# parameters, e0 and delta of its linear model and e0, eMax, ED50 and h of sigEmax.
CURVE_MODELS = (("epa", "linear"), ("epa", "hill"), ("ll4", "ll4"))


def dosefinding_parameters(model: str, values: dict[str, float]) -> list[float]:
    if model == "linear":
        return [values["g"], values["b1"]]
    if model == "hill":
        return [values["g"], values["v"], values["k"], values["n"]]
    return [values["d"], values["c"] - values["d"], values["e"], values["b"]]


def compare_curves(checks: Checks, found: Inputs, reference: dict[str, Any]) -> None:
    """Curves of one score under constant variance and in the weighted mode against
    DoseFinding's fitMod: the objective each minimises, the samples' residual sum of
    squares or the means' generalised one, and the parameters where both reach it."""
    for name in CURVE_CASES:
        one = found.alone[name]
        theirs = reference["curves"][name]
        design = Design(one.profiles, "~ 1")
        n_samples = len(one.profiles)
        for variance in ("constant", "weighted"):
            for model_set, model in CURVE_MODELS:
                result = dose_response(
                    one, design, dose=AXIS, models=model_set, variance=variance
                )
                fits = result.curve_fits()
                m = fits.models.index(model)
                log_likelihood = float(fits.log_likelihood[0, m])
                if variance == "constant":
                    # The variance at its maximum, the samples' residual sum of
                    # squares over n.
                    residual = math.exp(-2.0 * log_likelihood / n_samples - 1.0)
                    objective = n_samples * residual / (2.0 * math.pi)
                else:
                    covariance = fits.covariance_scale[0] * np.asarray(
                        fits.covariance[0]
                    )
                    objective = (
                        -2.0 * log_likelihood
                        - fits.doses.size * math.log(2.0 * math.pi)
                        - float(np.linalg.slogdet(covariance)[1])
                    )
                fit = theirs[variance][model]
                gap = (objective - float(fit["objective"])) / float(fit["objective"])
                parameters = dict(
                    zip(
                        fits.parameter_names,
                        fits.parameters[0, m].tolist(),
                        strict=True,
                    )
                )
                what = (
                    "residual sum of squares"
                    if variance == "constant"
                    else "generalised residual sum of squares"
                )
                kind = "linear" if model == "linear" else "sigmoid Emax (hill, ll4)"
                tolerance = EXACT if model == "linear" else CLOSE
                reference_name = (
                    'DoseFinding fitMod(type = "normal")'
                    if variance == "constant"
                    else 'DoseFinding fitMod(type = "general", S)'
                )
                if gap > tolerance:
                    checks.record(
                        f"curves of one score, {kind}, {variance} variance: the {what}",
                        gap,
                        tolerance,
                        reference_name,
                    )
                    continue
                if gap < -tolerance:
                    checks.deliberate(
                        f"curves of one score, {variance} variance: {model}",
                        reference_name,
                        f"DoseFinding stops {-gap:.1e} above CytePool's optimum",
                    )
                    continue
                checks.record(
                    f"curves of one score, {kind}, {variance} variance: the {what}",
                    abs(gap),
                    tolerance,
                    reference_name,
                )
                checks.record(
                    f"curves of one score, {kind}, {variance} variance: parameters",
                    scaled(
                        dosefinding_parameters(model, parameters),
                        numbers(fit["coef"]),
                    ),
                    EXACT if model == "linear" else 1e-4,
                    reference_name,
                )


def compare_constant(checks: Checks, found: Inputs, reference: dict[str, Any]) -> None:
    """A score that does not vary, tested alone: limma tests its residuals of
    rounding, and CytePool does not test it."""
    result = ours(found, JOBS[-1], "full")
    theirs = reference["jobs"]["constant"]["cell_types"]["full"]["tests"]
    tested = bool(result.summary["tested"][0])
    checks.deliberate(
        "A score that does not vary (0.3 in every sample), tested alone: limma's "
        "dose-group F and linear trend's t come from residuals of rounding",
        "limma lmFit, eBayes",
        f"limma F {numbers(theirs['dose_groups']['F']).ravel()[0]:.3g} and t "
        f"{numbers(theirs['linear_trend']['t']).ravel()[0]:.3g}; CytePool "
        + ("tests it" if tested else "does not test it"),
    )


# ---------------------------------------------------------------------------------------
# Error rates of scores on null series (information)
# ---------------------------------------------------------------------------------------


def null_rows(checks: Checks, datasets: int) -> None:
    """The share of p-values below 0.05 of scores of null series: one set of 30 genes
    alone, and 3, 5, 10, 30 and 50 random sets of 10 to 200 genes moderated together,
    without and with the set sizes as the trend's covariate."""
    for groups in sorted(NULL_DOSES):
        for replicates in NULL_REPLICATES:
            alone: dict[str, list[float]] = {}
            moderated: dict[tuple[int, bool, str], list[float]] = {}
            for dataset in range(datasets):
                rng = np.random.default_rng([SEED + 4, groups, replicates, dataset])
                cube, _ = simulate_dose_series(
                    NULL_DOSES[groups],
                    replicates=replicates,
                    cell_types=1,
                    cells=100.0,
                    n_genes=1000,
                    responsive=0.0,
                    rng=rng,
                )
                genes = np.array(cube.features["feature_id"].tolist())
                design = Design(cube.samples, "~ 1")
                sets = {"alone": rng.choice(genes, 30, replace=False).tolist()}
                # A few sets more than the most compared, of which the expression
                # filter can leave one without a gene.
                sizes = rng.integers(10, 201, max(NULL_SETS) + 5)
                for i, size in enumerate(sizes):
                    sets[f"set_{i:02d}"] = rng.choice(
                        genes, int(size), replace=False
                    ).tolist()
                scores = gsva(cube, sets)
                # The random sets that keep a gene after the expression filter.
                names = list(scores.gene_sets)
                random = [j for j, name in enumerate(names) if name != "alone"]
                if len(random) < max(NULL_SETS):
                    raise RuntimeError(f"only {len(random)} random sets kept a gene")
                one = _subset(scores, [names.index("alone")])
                result = dose_response(one, design, dose=AXIS)
                for test in result.tests:
                    p = float(result.table(name=test)["p_value"][0])
                    alone.setdefault(test, []).append(float(p < 0.05))
                mcp = dose_response(
                    one, design, dose=AXIS, tests="mcp", direction="up", rng=rng
                )
                alone.setdefault("mcp, one-sided", []).append(
                    float(float(mcp.table(name="mcp")["p_value"][0]) < 0.05)
                )
                for m in NULL_SETS:
                    chosen = _subset(scores, random[:m])
                    n_genes = np.asarray(chosen.n_genes)
                    order = np.argsort(n_genes, kind="stable")
                    third = max(m // 3, 1)
                    for trend in (False, True):
                        options = {"trend": n_genes} if trend else {}
                        tested = dose_response(
                            chosen, design, dose=AXIS, tests="dose_groups", **options
                        )
                        p = np.asarray(
                            tested.table(name="dose_groups")["p_value"], dtype=float
                        )
                        below = p < 0.05
                        moderated.setdefault((m, trend, "all"), []).append(
                            float(below.mean())
                        )
                        moderated.setdefault((m, trend, "small"), []).append(
                            float(below[order[:third]].mean())
                        )
                        moderated.setdefault((m, trend, "large"), []).append(
                            float(below[order[-third:]].mean())
                        )
            label = f"K = {groups}, {replicates} replicates, {datasets} null datasets"
            checks.information(
                f"{label}: one set of 30 genes scored by GSVA and tested alone, share of "
                "p below 0.05 (Monte Carlo standard error)",
                "the null",
                "; ".join(f"{test} {_share(values)}" for test, values in alone.items()),
            )
            for trend in (False, True):
                parts = []
                for m in NULL_SETS:
                    parts.append(
                        f"{m} sets: {_share(moderated[(m, trend, 'all')])}, smallest third "
                        f"{_share(moderated[(m, trend, 'small')])}, largest "
                        f"{_share(moderated[(m, trend, 'large')])}"
                    )
                checks.information(
                    f"{label}: random sets of 10 to 200 genes moderated together, "
                    f"{'with' if trend else 'without'} the set sizes as trend, share of "
                    "dose-group F p-values below 0.05",
                    "the null",
                    "; ".join(parts),
                )


def _subset(scores: PathwayScores, columns: list[int]) -> PathwayScores:
    names = list(scores.gene_sets)
    return replace(
        scores,
        scores=np.ascontiguousarray(scores.scores[:, columns]),
        gene_sets=GeneSets(
            {names[j]: list(scores.gene_sets[names[j]]) for j in columns}
        ),
    )


def _share(values: list[float]) -> str:
    array = np.asarray(values)
    return f"{array.mean():.3f} ({array.std(ddof=1) / math.sqrt(array.size):.3f})"


# ---------------------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------------------


def write_fixtures(found: Inputs, reference: dict[str, Any]) -> None:
    def scores_of(scores: PathwayScores) -> dict[str, Any]:
        return {
            "scores": plain(scores.scores),
            "sample_id": scores.profiles["sample_id"].tolist(),
            "dose": np.asarray(scores.profiles["dose"], dtype=float).tolist(),
            "names": list(scores.gene_sets),
            "sizes": scores.n_genes.tolist(),
        }

    def trimmed(job: str, group: str = "full") -> dict[str, Any]:
        """A job's results without what the tests do not read: limma's own
        approximate values, the posterior variances and degrees of freedom, and the
        curve at 20 of its doses."""
        values = json.loads(json.dumps(reference["jobs"][job]["cell_types"][group]))
        for key in ("s2_post", "df_total", "incomplete"):
            values.pop(key)
        for test in values["tests"].values():
            test.pop("limma_t", None)
            test.pop("limma_F", None)
        curve = values.get("curve")
        if curve is not None:
            keep = (
                np.linspace(0, CURVE_POINTS - 1, FIXTURE_CURVE_POINTS)
                .round()
                .astype(int)
            )
            values["curve"] = {
                "at": [curve["at"][i] for i in keep],
                **{
                    key: [[row[i] for i in keep] for row in curve[key]]
                    for key in ("estimate", "lower", "upper")
                },
            }
        return values

    limma = {
        "scores": {
            "ladder": scores_of(found.scores["full"]),
            "missing": scores_of(found.missing),
            "sex": scores_of(found.sex_scores) | {"sex": found.sex_samples["sex"]},
        },
        "jobs": {
            job: trimmed(job)
            for job in (
                "ladder",
                "ladder, trend",
                "ladder, robust",
                "missing",
                "missing, trend",
            )
        }
        | {"sex": trimmed("sex", "pDC")},
        "versions": reference["versions"],
    }
    dosefinding = {
        "scores": {name: scores_of(one) for name, one in found.alone.items()},
        "reference": reference["alone"],
        "curves": reference["curves"],
        "seed": SEED,
        "versions": reference["versions"],
    }
    FIXTURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for name, record in (("limma.json", limma), ("dosefinding.json", dosefinding)):
        path = FIXTURE_DIRECTORY / name
        path.write_text(json.dumps(record, separators=(",", ":")) + "\n")
        print(f"wrote {name} to {FIXTURE_DIRECTORY}")


# ---------------------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write-fixtures", action="store_true", help="write fixtures")
    parser.add_argument(
        "--null-datasets",
        type=int,
        default=NULL_DATASETS,
        help="null datasets per design of the information rows",
    )
    arguments = parser.parse_args()
    started = time.perf_counter()
    found = inputs()
    reference = run_r(r_input(found))
    print(f"R finished after {time.perf_counter() - started:.0f} s", flush=True)
    print(
        "Reference: " + ", ".join(f"{k} {v}" for k, v in reference["versions"].items())
    )
    checks = Checks()
    for job in JOBS[:-1]:
        results = {group: ours(found, job, group) for group in job.groups}
        compare_job(checks, job, results, reference["jobs"][job.name])
    compare_one_sided_trend(checks, found, reference)
    alone = compare_alone(checks, found, reference)
    compare_limma_approximation(checks, reference)
    compare_curves(checks, found, reference)
    compare_constant(checks, found, reference)
    two_sided = all(record.get("mct_two_sided", True) for record in alone)
    checks.deliberate(
        "MCTtest's two-sided t-statistics of the one-score cases",
        'DoseFinding MCTtest(alternative = "two.sided")',
        "MCTtest reports |t|, CytePool keeps the sign"
        + ("" if two_sided else "; |t| differs"),
    )
    print(f"CytePool finished after {time.perf_counter() - started:.0f} s", flush=True)
    null_rows(checks, arguments.null_datasets)
    rows, failures = checks.table()
    print("| check | reference | value | tolerance | result |")
    print("| --- | --- | --- | --- | --- |")
    for row in rows:
        print("| " + " | ".join(row) + " |")
    for line in checks.mismatches:
        print(f"differs: {line}")
    if arguments.write_fixtures:
        if failures:
            print("not writing fixtures: some checks failed")
        else:
            write_fixtures(found, reference)
    print(
        f"{len(rows)} checks, {failures} failed; {time.perf_counter() - started:.0f} s"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
